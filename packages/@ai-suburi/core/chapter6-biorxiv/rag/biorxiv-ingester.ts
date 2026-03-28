import type { BiorxivPaper } from '../models.js';
import { loadSettings } from '../configs.js';
import { setupLogger } from '../custom-logger.js';
import { ChromaStore } from './chroma-store.js';

const logger = setupLogger('biorxiv-ingester');

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

export async function ingestBiorxivPapers(options: {
  startDate: string;
  endDate: string;
  category?: string;
  batchSize?: number;
  delayMs?: number;
}): Promise<number> {
  const settings = loadSettings();
  const category = options.category ?? settings.biorxivCategory;
  const batchSize = options.batchSize ?? settings.ingestionBatchSize;
  const delayMs = options.delayMs ?? 1000;

  const store = new ChromaStore({
    collectionName: settings.chromaCollectionName,
    openaiApiKey: settings.openaiApiKey,
    embeddingModel: settings.embeddingModel,
  });

  let cursor = 0;
  let totalIngested = 0;

  while (true) {
    const response = await fetchBiorxivPage(
      options.startDate,
      options.endDate,
      cursor,
      category,
    );

    const entries = response.collection ?? [];
    if (entries.length === 0) {
      logger.info('No more entries. Ingestion complete.');
      break;
    }

    const papers = entries.map(apiEntryToPaper);

    // 重複チェック: 既にDBにある論文はスキップ
    const newPapers: BiorxivPaper[] = [];
    for (const paper of papers) {
      const alreadyExists = await store.exists(paper.doi);
      if (!alreadyExists) {
        newPapers.push(paper);
      }
    }

    if (newPapers.length > 0) {
      await store.addDocuments(newPapers);
      totalIngested += newPapers.length;
    }

    logger.info(
      `Processed ${entries.length} entries (${newPapers.length} new). Total ingested: ${totalIngested}`,
    );

    // ページネーション: 100件ずつ
    const status = response.messages?.[0];
    if (!status || entries.length < batchSize) {
      logger.info('Reached end of results.');
      break;
    }

    cursor += batchSize;

    // レート制限対策
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  const totalCount = await store.getDocumentCount();
  logger.info(
    `Ingestion complete. ${totalIngested} new papers added. Total in collection: ${totalCount}`,
  );

  return totalIngested;
}

// --- CLI エントリーポイント ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let startDate = '';
  let endDate = '';
  let category: string | undefined;

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
    }
  }

  if (!startDate || !endDate) {
    console.error('Usage: npx tsx rag/biorxiv-ingester.ts --start YYYY-MM-DD --end YYYY-MM-DD [--category bioinformatics]');
    process.exit(1);
  }

  const count = await ingestBiorxivPapers({
    startDate,
    endDate,
    ...(category != null ? { category } : {}),
  });
  console.log(`Done! Ingested ${count} papers.`);
}

main();
