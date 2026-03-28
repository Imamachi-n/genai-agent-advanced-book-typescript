import * as fs from 'node:fs';
import * as path from 'node:path';

import type { BiorxivPaper } from '../models.js';
import { loadSettings } from '../configs.js';
import { setupLogger } from '../custom-logger.js';

const logger = setupLogger('biorxiv-fetcher');

const DEFAULT_OUTPUT_DIR = 'storage/biorxiv-tmp';

interface BiorxivApiResponse {
  messages: { status: string; count: number; total: number }[];
  collection: BiorxivApiEntry[];
}

interface BiorxivApiEntry {
  biorxiv_doi: string;
  title: string;
  authors: string;
  author_corresponding: string;
  author_corresponding_institution: string;
  date: string;
  version: string;
  type: string;
  license: string;
  category: string;
  jatsxml: string;
  abstract: string;
  published: string;
  server: string;
}

function apiEntryToPaper(entry: BiorxivApiEntry): BiorxivPaper {
  const doi = entry.biorxiv_doi;
  const version = Number.parseInt(entry.version, 10) || 1;
  return {
    doi,
    title: entry.title,
    link: `https://doi.org/${doi}`,
    pdfLink: `https://www.biorxiv.org/content/${doi}v${version}.full.pdf`,
    abstract: entry.abstract,
    published: entry.date,
    authors: entry.authors.split('; ').map((a) => a.trim()),
    category: entry.category,
    version,
    relevanceScore: null,
  };
}

async function fetchBiorxivPage(
  startDate: string,
  endDate: string,
  cursor: number,
  category?: string,
): Promise<BiorxivApiResponse> {
  let url = `https://api.biorxiv.org/details/biorxiv/${startDate}/${endDate}/${cursor}`;
  if (category) {
    url += `?category=${encodeURIComponent(category)}`;
  }
  logger.info(`Fetching: ${url}`);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`bioRxiv API error: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<BiorxivApiResponse>;
}

/**
 * bioRxiv API から論文メタデータを取得し、JSON ファイルとして保存する。
 * Chroma への投入は行わない。
 */
export async function fetchBiorxivPapers(options: {
  startDate: string;
  endDate: string;
  outputDir?: string;
  category?: string;
  batchSize?: number;
  delayMs?: number;
}): Promise<{ outputPath: string; totalFetched: number }> {
  const settings = loadSettings();
  const category = options.category ?? settings.biorxivCategory;
  const batchSize = options.batchSize ?? settings.ingestionBatchSize;
  const delayMs = options.delayMs ?? 1000;
  const outputDir = options.outputDir ?? DEFAULT_OUTPUT_DIR;

  fs.mkdirSync(outputDir, { recursive: true });

  const allPapers: BiorxivPaper[] = [];
  let cursor = 0;

  while (true) {
    const response = await fetchBiorxivPage(
      options.startDate,
      options.endDate,
      cursor,
      category,
    );

    const entries = response.collection ?? [];
    if (entries.length === 0) {
      logger.info('No more entries from API.');
      break;
    }

    const papers = entries.map(apiEntryToPaper);
    allPapers.push(...papers);

    logger.info(
      `Fetched ${entries.length} entries (total so far: ${allPapers.length})`,
    );

    // ページネーション: 100件ずつ
    if (entries.length < batchSize) {
      logger.info('Reached end of results.');
      break;
    }

    cursor += batchSize;

    // レート制限対策
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  // JSON ファイルに保存
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const fileName = `biorxiv_${options.startDate}_${options.endDate}_${timestamp}.json`;
  const outputPath = path.join(outputDir, fileName);

  fs.writeFileSync(outputPath, JSON.stringify(allPapers, null, 2), 'utf-8');
  logger.info(`Saved ${allPapers.length} papers to ${outputPath}`);

  return { outputPath, totalFetched: allPapers.length };
}

// --- CLI エントリーポイント ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let startDate = '';
  let endDate = '';
  let category: string | undefined;
  let outputDir: string | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--start' && args[i + 1]) {
      startDate = args[i + 1]!;
      i++;
    } else if (args[i] === '--end' && args[i + 1]) {
      endDate = args[i + 1]!;
      i++;
    } else if (args[i] === '--category' && args[i + 1]) {
      category = args[i + 1]!;
      i++;
    } else if (args[i] === '--output' && args[i + 1]) {
      outputDir = args[i + 1]!;
      i++;
    }
  }

  if (!startDate || !endDate) {
    console.error(
      'Usage: npx tsx rag/biorxiv-fetcher.ts --start YYYY-MM-DD --end YYYY-MM-DD [--category bioinformatics] [--output storage/biorxiv-tmp]',
    );
    process.exit(1);
  }

  const result = await fetchBiorxivPapers({
    startDate,
    endDate,
    ...(category != null ? { category } : {}),
    ...(outputDir != null ? { outputDir } : {}),
  });
  console.log(`Done! Fetched ${result.totalFetched} papers → ${result.outputPath}`);
}

main();
