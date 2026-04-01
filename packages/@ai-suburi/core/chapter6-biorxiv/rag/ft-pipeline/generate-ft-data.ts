import * as fs from 'node:fs';
import * as path from 'node:path';
import { ChatOpenAI } from '@langchain/openai';
import { loadSettings } from '../../configs.js';
import { setupLogger } from '../../custom-logger.js';
import type { BiorxivPaper } from '../../models.js';
import { QdrantStore } from '../qdrant-store.js';
import { extractAllPapers } from './paper-extractor.js';
import { createIdealQueryGenerator } from './ideal-query-generator.js';
import { synthesizeUserQueries } from './query-synthesizer.js';
import {
  type TrainingEntry,
  TrainingDataWriter,
} from './training-data-formatter.js';
import { validateTrainingData } from './validation.js';

const logger = setupLogger('generate-ft-data');

const DEFAULT_OUTPUT_DIR = 'storage/ft-training-data';
const DEFAULT_MODEL = 'gpt-5-nano';
const MAX_RETRY_COUNT = 3;
const RATE_LIMIT_WAIT_MS = 30000;

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

// --- リトライ付き処理 ---

function isRateLimitError(error: unknown): boolean {
  const msg = (error as Error)?.message ?? String(error);
  return msg.includes('429') || msg.includes('rate') || msg.includes('Rate');
}

/**
 * 1 論文の処理をリトライ付きで実行する。
 * レート制限エラー時はエクスポネンシャルバックオフでリトライする。
 */
async function processPaperWithRetry(
  paper: BiorxivPaper,
  llm: ChatOpenAI,
  generateIdealQuery: (paper: BiorxivPaper) => Promise<string>,
  queriesPerPaper: number,
): Promise<TrainingEntry> {
  for (let attempt = 0; attempt < MAX_RETRY_COUNT; attempt++) {
    try {
      const [syntheticQueries, idealQuery] = await Promise.all([
        synthesizeUserQueries(llm, paper, queriesPerPaper),
        generateIdealQuery(paper),
      ]);
      return { paper, syntheticQueries, idealQuery };
    } catch (error) {
      if (isRateLimitError(error) && attempt < MAX_RETRY_COUNT - 1) {
        const waitMs = RATE_LIMIT_WAIT_MS * (attempt + 1);
        logger.info(
          `Rate limit for "${paper.title.slice(0, 40)}..." - retry ${attempt + 1}/${MAX_RETRY_COUNT} in ${waitMs / 1000}s`,
        );
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        continue;
      }
      throw error;
    }
  }
  throw new Error(`Max retries exceeded for "${paper.title.slice(0, 60)}..."`);
}

// --- メイン ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let outputDir = DEFAULT_OUTPUT_DIR;
  let queriesPerPaper = 3;
  let limit: number | undefined;
  let validate = false;
  let validateOnly: string | undefined;
  let sampleSize = 50;
  let model = DEFAULT_MODEL;
  let resume = false;
  let skipEmbeddingCheck = true;
  let concurrency = 5;

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
    } else if (args[i] === '--embedding-check') {
      skipEmbeddingCheck = false;
    } else if (args[i] === '--concurrency' && args[i + 1]) {
      concurrency = Number.parseInt(args[i + 1]!, 10);
      i++;
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
  logger.info(`Concurrency: ${concurrency}, Embedding check: ${skipEmbeddingCheck ? 'skip' : 'enabled'}`);
  const llm = new ChatOpenAI({
    model,
    ...(useFixedTemp ? {} : { temperature: 0 }),
  });
  const generateIdealQuery = createIdealQueryGenerator(
    llm,
    settings.openaiApiKey,
    { embeddingModel: settings.embeddingModel, skipEmbeddingCheck },
  );

  // ストリーミング Writer（レジューム時は append モード）
  const writer = new TrainingDataWriter(outputDir, { append: resume && processedDois.size > 0 });
  let failedCount = 0;

  // バッチ並列処理（レート制限時はリトライ）
  for (let batchStart = 0; batchStart < remainingPapers.length; batchStart += concurrency) {
    const batch = remainingPapers.slice(batchStart, batchStart + concurrency);
    logger.info(
      `Processing batch ${Math.floor(batchStart / concurrency) + 1} (papers ${batchStart + 1}-${batchStart + batch.length}/${remainingPapers.length})`,
    );

    const results = await Promise.allSettled(
      batch.map((paper) => processPaperWithRetry(paper, llm, generateIdealQuery, queriesPerPaper)),
    );

    // 結果をストリーミング書き出し（書き込みは逐次で安全に）
    for (const result of results) {
      if (result.status === 'fulfilled') {
        writer.writeEntry(result.value);
        processedDois.add(result.value.paper.doi);
      } else {
        failedCount++;
        logger.warn(`Failed after ${MAX_RETRY_COUNT} retries: ${result.reason?.message ?? String(result.reason)}`);
      }
    }

    // バッチごとにプログレス保存
    saveProgress({
      processedDois: [...processedDois],
      model,
      outputDir,
    });
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
    '  --embedding-check        Embedding品質検証を有効化（デフォルト: スキップ）',
  );
  console.error(
    '  --concurrency <n>        バッチ並列数（default: 5）',
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
