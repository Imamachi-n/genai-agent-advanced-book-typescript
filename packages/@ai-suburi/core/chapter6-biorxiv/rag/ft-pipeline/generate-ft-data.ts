import * as fs from 'node:fs';
import * as path from 'node:path';
import { ChatOpenAI } from '@langchain/openai';
import { loadSettings } from '../../configs.js';
import { setupLogger } from '../../custom-logger.js';
import { QdrantStore } from '../qdrant-store.js';
import { extractAllPapers } from './paper-extractor.js';
import { createIdealQueryGenerator } from './ideal-query-generator.js';
import { synthesizeUserQueries } from './query-synthesizer.js';
import { TrainingDataWriter } from './training-data-formatter.js';
import { validateTrainingData } from './validation.js';

const logger = setupLogger('generate-ft-data');

const DEFAULT_OUTPUT_DIR = 'storage/ft-training-data';
const DEFAULT_MODEL = 'gpt-5-nano';

/** temperature=0 をサポートしないモデル一覧。これらは temperature を指定しない */
const TEMPERATURE_FIXED_MODELS = new Set(['gpt-5-nano', 'gpt-5-mini']);

// --- プログレス管理 ---

interface Progress {
  processedDois: string[];
  model: string;
  outputDir: string;
}

function getProgressPath(outputDir: string): string {
  return path.join(outputDir, '.ft-progress.json');
}

function loadProgress(outputDir: string): Progress | null {
  const progressPath = getProgressPath(outputDir);
  if (!fs.existsSync(progressPath)) return null;
  return JSON.parse(fs.readFileSync(progressPath, 'utf-8')) as Progress;
}

function saveProgress(progress: Progress): void {
  const progressPath = getProgressPath(progress.outputDir);
  fs.writeFileSync(progressPath, JSON.stringify(progress, null, 2), 'utf-8');
}

function deleteProgress(outputDir: string): void {
  const progressPath = getProgressPath(outputDir);
  if (fs.existsSync(progressPath)) {
    fs.unlinkSync(progressPath);
  }
}

// --- メイン ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let outputDir = DEFAULT_OUTPUT_DIR;
  let queriesPerPaper = 5;
  let limit: number | undefined;
  let validate = false;
  let validateOnly: string | undefined;
  let sampleSize = 50;
  let model = DEFAULT_MODEL;
  let resume = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--output' && args[i + 1]) {
      outputDir = args[i + 1]!;
      i++;
    } else if (args[i] === '--queries-per-paper' && args[i + 1]) {
      queriesPerPaper = Number.parseInt(args[i + 1]!, 10);
      i++;
    } else if (args[i] === '--limit' && args[i + 1]) {
      limit = Number.parseInt(args[i + 1]!, 10);
      i++;
    } else if (args[i] === '--model' && args[i + 1]) {
      model = args[i + 1]!;
      i++;
    } else if (args[i] === '--validate') {
      validate = true;
    } else if (args[i] === '--validate-only' && args[i + 1]) {
      validateOnly = args[i + 1]!;
      i++;
    } else if (args[i] === '--sample-size' && args[i + 1]) {
      sampleSize = Number.parseInt(args[i + 1]!, 10);
      i++;
    } else if (args[i] === '--resume') {
      resume = true;
    }
  }

  const settings = loadSettings();

  // --validate-only モード: 既存 JSONL の品質検証のみ
  if (validateOnly) {
    const store = new QdrantStore({
      collectionName: settings.qdrantCollectionName,
      openaiApiKey: settings.openaiApiKey,
      embeddingModel: settings.embeddingModel,
    });
    const metadataPath = validateOnly.replace('.jsonl', '_metadata.jsonl');
    await validateTrainingData({
      trainingPath: validateOnly,
      metadataPath,
      store,
      sampleSize,
    });
    return;
  }

  // --- メインパイプライン ---

  fs.mkdirSync(outputDir, { recursive: true });

  // レジューム: 前回の進捗を読み込み
  let processedDois = new Set<string>();
  if (resume) {
    const progress = loadProgress(outputDir);
    if (progress) {
      processedDois = new Set(progress.processedDois);
      logger.info(`Resuming: ${processedDois.size} papers already processed`);
    } else {
      logger.info('No progress file found. Starting fresh.');
    }
  }

  // Step 1: Qdrant から全論文を抽出
  logger.info('Step 1: Extracting papers from Qdrant...');
  let papers = await extractAllPapers({
    collectionName: settings.qdrantCollectionName,
    qdrantUrl: settings.qdrantUrl,
  });

  if (limit) {
    papers = papers.slice(0, limit);
    logger.info(`Limited to ${papers.length} papers`);
  }

  // レジューム時は処理済みをスキップ
  const remainingPapers = papers.filter((p) => !processedDois.has(p.doi));
  logger.info(
    `Total: ${papers.length} papers, remaining: ${remainingPapers.length}`,
  );

  // Step 2〜4: 合成クエリ + 理想クエリ生成 → JSONL にストリーミング書き出し
  const useFixedTemp = TEMPERATURE_FIXED_MODELS.has(model);
  logger.info(`Using model: ${model}${useFixedTemp ? ' (temperature fixed by API)' : ' (temperature=0)'}`);
  const llm = new ChatOpenAI({
    model,
    ...(useFixedTemp ? {} : { temperature: 0 }),
  });
  const generateIdealQuery = createIdealQueryGenerator(
    llm,
    settings.openaiApiKey,
    settings.embeddingModel,
  );

  // ストリーミング Writer（レジューム時は append モード）
  const writer = new TrainingDataWriter(outputDir, { append: resume && processedDois.size > 0 });
  let failedCount = 0;

  for (let i = 0; i < remainingPapers.length; i++) {
    const paper = remainingPapers[i]!;
    logger.info(
      `Processing paper ${i + 1}/${remainingPapers.length}: ${paper.title.slice(0, 60)}...`,
    );

    try {
      // Step 2: 合成クエリ生成
      const syntheticQueries = await synthesizeUserQueries(
        llm,
        paper,
        queriesPerPaper,
      );

      // Step 3: 理想クエリ生成
      const idealQuery = await generateIdealQuery(paper);

      // Step 4: JSONL に即座に書き出し（メモリに溜めない）
      writer.writeEntry({ paper, syntheticQueries, idealQuery });
      processedDois.add(paper.doi);

      // 1 件ごとにプログレス保存（LLM API コールに比べて writeFileSync のコストは無視できる）
      saveProgress({
        processedDois: [...processedDois],
        model,
        outputDir,
      });
    } catch (error) {
      failedCount++;
      logger.warn(
        `Failed to process paper "${paper.title.slice(0, 60)}...": ${(error as Error).message}`,
      );

      // レート制限エラーの場合は少し待つ
      const errorMessage = (error as Error).message ?? '';
      if (errorMessage.includes('429') || errorMessage.includes('rate')) {
        logger.info('Rate limit detected. Waiting 30 seconds...');
        await new Promise((resolve) => setTimeout(resolve, 30000));
      }
    }
  }

  // ストリームを閉じる
  const { totalExamples } = await writer.close();

  // 正常完了: プログレスファイルを削除
  deleteProgress(outputDir);

  const { trainingPath, metadataPath } = writer;
  console.log(`\nDone! Generated ${totalExamples} training examples.`);
  console.log(`  Training data: ${trainingPath}`);
  console.log(`  Metadata: ${metadataPath}`);
  console.log(`  Processed: ${processedDois.size}, Failed: ${failedCount}`);

  // Step 5: 品質検証（オプション）
  if (validate) {
    logger.info('Step 5: Running validation...');
    const store = new QdrantStore({
      collectionName: settings.qdrantCollectionName,
      openaiApiKey: settings.openaiApiKey,
      embeddingModel: settings.embeddingModel,
    });
    await validateTrainingData({
      trainingPath,
      metadataPath,
      store,
      sampleSize,
    });
  }
}

main().catch((error) => {
  console.error(`\nError: ${(error as Error).message}`);
  console.error('\nUsage:');
  console.error(
    '  npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts [options]',
  );
  console.error('\nOptions:');
  console.error(
    '  --output <dir>           出力ディレクトリ（default: storage/ft-training-data）',
  );
  console.error(
    '  --queries-per-paper <n>  論文あたりの合成クエリ数（default: 5）',
  );
  console.error(
    '  --limit <n>              処理する論文数の上限（テスト用）',
  );
  console.error(
    `  --model <name>           使用する LLM モデル（default: ${DEFAULT_MODEL}）`,
  );
  console.error(
    '  --resume                 前回の中断地点から再開',
  );
  console.error(
    '  --validate               生成後に品質検証を実行',
  );
  console.error(
    '  --validate-only <path>   既存 JSONL の品質検証のみ実行',
  );
  console.error(
    '  --sample-size <n>        検証時のサンプル数（default: 50）',
  );
  process.exit(1);
});
