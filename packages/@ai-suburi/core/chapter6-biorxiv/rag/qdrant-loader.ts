import * as fs from 'node:fs';
import * as readline from 'node:readline';
import { loadSettings } from '../configs.js';
import { setupLogger } from '../custom-logger.js';
import type { BiorxivPaper } from '../models.js';
import { QdrantStore } from './qdrant-store.js';

const logger = setupLogger('qdrant-loader');

/**
 * JSONL ファイルから論文データを行単位で読み込み、Qdrant に投入する。
 * biorxiv-fetcher.ts で出力した JSONL を入力として使う。
 *
 * 行単位のストリーム処理のため、大量データでもメモリを圧迫しない。
 */
export async function loadPapersToQdrant(options: {
  jsonPath: string;
  batchSize?: number;
  force?: boolean;
}): Promise<{ totalLoaded: number; skipped: number }> {
  const settings = loadSettings();
  const batchSize = options.batchSize ?? 50;

  const store = new QdrantStore({
    collectionName: settings.qdrantCollectionName,
    openaiApiKey: settings.openaiApiKey,
    embeddingModel: settings.embeddingModel,
  });

  let totalLoaded = 0;
  let skipped = 0;
  let batchNumber = 0;
  let batch: BiorxivPaper[] = [];

  // JSONL を行単位でストリーム読み込み
  const fileStream = fs.createReadStream(options.jsonPath, {
    encoding: 'utf-8',
  });
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const paper: BiorxivPaper = JSON.parse(trimmed);
    batch.push(paper);

    if (batch.length >= batchSize) {
      const result = await processBatch(store, batch, batchNumber, options.force);
      totalLoaded += result.loaded;
      skipped += result.skipped;
      batchNumber++;
      batch = [];
    }
  }

  // 残りのバッチを処理
  if (batch.length > 0) {
    const result = await processBatch(store, batch, batchNumber, options.force);
    totalLoaded += result.loaded;
    skipped += result.skipped;
  }

  const totalCount = await store.getDocumentCount();
  logger.info(
    `Loading complete. ${totalLoaded} new papers added, ${skipped} skipped. Total in collection: ${totalCount}`,
  );

  return { totalLoaded, skipped };
}

async function processBatch(
  store: QdrantStore,
  batch: BiorxivPaper[],
  batchNumber: number,
  force?: boolean,
): Promise<{ loaded: number; skipped: number }> {
  let loaded = 0;
  let skipped = 0;

  if (force) {
    // 強制モード: 重複チェックせず全件 upsert（既存データを上書き）
    await store.addDocuments(batch);
    loaded = batch.length;
  } else {
    // 通常モード: 既に Qdrant にある論文はスキップ
    const newPapers: BiorxivPaper[] = [];
    for (const paper of batch) {
      const alreadyExists = await store.exists(paper.doi);
      if (alreadyExists) {
        skipped++;
      } else {
        newPapers.push(paper);
      }
    }

    if (newPapers.length > 0) {
      await store.addDocuments(newPapers);
      loaded = newPapers.length;
    }
  }

  logger.info(`Batch ${batchNumber + 1}: ${loaded} added, ${skipped} skipped`);

  return { loaded, skipped };
}

// --- CLI エントリーポイント ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let jsonPath = '';
  let batchSize: number | undefined;
  let force = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && args[i + 1]) {
      jsonPath = args[i + 1]!;
      i++;
    } else if (args[i] === '--batch-size' && args[i + 1]) {
      batchSize = Number.parseInt(args[i + 1]!, 10);
      i++;
    } else if (args[i] === '--force') {
      force = true;
    } else if (!args[i]!.startsWith('--')) {
      jsonPath = args[i]!;
    }
  }

  if (!jsonPath) {
    console.error(
      'Usage: npx tsx rag/qdrant-loader.ts --input <path-to-jsonl> [--batch-size 50] [--force]',
    );
    console.error('   or: npx tsx rag/qdrant-loader.ts <path-to-jsonl>');
    console.error('');
    console.error('Options:');
    console.error('  --force    既存データを上書き（abstract 等の payload 更新時に使用）');
    process.exit(1);
  }

  if (!fs.existsSync(jsonPath)) {
    console.error(`File not found: ${jsonPath}`);
    process.exit(1);
  }

  const result = await loadPapersToQdrant({
    jsonPath,
    ...(batchSize != null ? { batchSize } : {}),
    force,
  });
  console.log(
    `Done! ${result.totalLoaded} papers loaded, ${result.skipped} skipped.`,
  );
}

main();
