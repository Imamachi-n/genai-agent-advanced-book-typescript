import * as fs from 'node:fs';

import type { BiorxivPaper } from '../models.js';
import { loadSettings } from '../configs.js';
import { setupLogger } from '../custom-logger.js';
import { ChromaStore } from './chroma-store.js';

const logger = setupLogger('chroma-loader');

/**
 * JSON ファイルから論文データを読み込み、Chroma に投入する。
 * biorxiv-fetcher.ts で出力した JSON を入力として使う。
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

  // JSON ファイル読み込み
  const raw = fs.readFileSync(options.jsonPath, 'utf-8');
  const papers: BiorxivPaper[] = JSON.parse(raw);
  logger.info(`Loaded ${papers.length} papers from ${options.jsonPath}`);

  let totalLoaded = 0;
  let skipped = 0;

  // バッチに分割して処理（Embedding API のレート制限対策）
  for (let i = 0; i < papers.length; i += batchSize) {
    const batch = papers.slice(i, i + batchSize);

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
      totalLoaded += newPapers.length;
    }

    logger.info(
      `Batch ${Math.floor(i / batchSize) + 1}: ${newPapers.length} added, ${batch.length - newPapers.length} skipped (total: ${totalLoaded} loaded, ${skipped} skipped)`,
    );
  }

  const totalCount = await store.getDocumentCount();
  logger.info(
    `Loading complete. ${totalLoaded} new papers added, ${skipped} skipped. Total in collection: ${totalCount}`,
  );

  return { totalLoaded, skipped };
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
