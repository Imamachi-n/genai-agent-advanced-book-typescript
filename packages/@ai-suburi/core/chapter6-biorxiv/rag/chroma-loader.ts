import * as fs from 'node:fs';
import * as readline from 'node:readline';

import type { BiorxivPaper } from '../models.js';
import { loadSettings } from '../configs.js';
import { setupLogger } from '../custom-logger.js';
import { ChromaStore } from './chroma-store.js';

const logger = setupLogger('chroma-loader');

/**
 * JSONL ファイルから論文データを行単位で読み込み、Chroma に投入する。
 * biorxiv-fetcher.ts で出力した JSONL を入力として使う。
 *
 * 行単位のストリーム処理のため、大量データでもメモリを圧迫しない。
 */
export async function loadPapersToChroma(options: {
  jsonPath: string;
  batchSize?: number;
}): Promise<{ totalLoaded: number; skipped: number }> {
  const settings = loadSettings();
  const batchSize = options.batchSize ?? 50;

  const store = new ChromaStore({
    collectionName: settings.chromaCollectionName,
    openaiApiKey: settings.openaiApiKey,
    embeddingModel: settings.embeddingModel,
  });

  let totalLoaded = 0;
  let skipped = 0;
  let batchNumber = 0;
  let batch: BiorxivPaper[] = [];

  // JSONL を行単位でストリーム読み込み
  const fileStream = fs.createReadStream(options.jsonPath, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const paper: BiorxivPaper = JSON.parse(trimmed);
    batch.push(paper);

    if (batch.length >= batchSize) {
      const result = await processBatch(store, batch, batchNumber);
      totalLoaded += result.loaded;
      skipped += result.skipped;
      batchNumber++;
      batch = [];
    }
  }

  // 残りのバッチを処理
  if (batch.length > 0) {
    const result = await processBatch(store, batch, batchNumber);
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
  store: ChromaStore,
  batch: BiorxivPaper[],
  batchNumber: number,
): Promise<{ loaded: number; skipped: number }> {
  let loaded = 0;
  let skipped = 0;

  // 重複チェック: 既に Chroma にある論文はスキップ
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

  logger.info(
    `Batch ${batchNumber + 1}: ${loaded} added, ${skipped} skipped`,
  );

  return { loaded, skipped };
}

// --- CLI エントリーポイント ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let jsonPath = '';
  let batchSize: number | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && args[i + 1]) {
      jsonPath = args[i + 1]!;
      i++;
    } else if (args[i] === '--batch-size' && args[i + 1]) {
      batchSize = Number.parseInt(args[i + 1]!, 10);
      i++;
    } else if (!args[i]!.startsWith('--')) {
      // 引数なしで JSON パスを指定できるようにする
      jsonPath = args[i]!;
    }
  }

  if (!jsonPath) {
    console.error(
      'Usage: npx tsx rag/chroma-loader.ts --input <path-to-json> [--batch-size 50]',
    );
    console.error('   or: npx tsx rag/chroma-loader.ts <path-to-json>');
    process.exit(1);
  }

  if (!fs.existsSync(jsonPath)) {
    console.error(`File not found: ${jsonPath}`);
    process.exit(1);
  }

  const result = await loadPapersToChroma({
    jsonPath,
    ...(batchSize != null ? { batchSize } : {}),
  });
  console.log(
    `Done! ${result.totalLoaded} papers loaded, ${result.skipped} skipped.`,
  );
}

main();
