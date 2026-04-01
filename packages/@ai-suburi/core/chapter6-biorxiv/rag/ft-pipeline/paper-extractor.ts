import { QdrantClient } from '@qdrant/js-client-rest';
import { setupLogger } from '../../custom-logger.js';
import type { BiorxivPaper } from '../../models.js';
import { QdrantStore } from '../qdrant-store.js';

const logger = setupLogger('paper-extractor');

/**
 * Qdrant の scroll API で全論文を取得する。
 * QdrantStore にはバルク取得メソッドがないため、QdrantClient を直接使用する。
 */
export async function extractAllPapers(options: {
  collectionName: string;
  qdrantUrl?: string;
  batchSize?: number;
}): Promise<BiorxivPaper[]> {
  const client = new QdrantClient({
    url: options.qdrantUrl ?? 'http://localhost:6333',
  });
  const batchSize = options.batchSize ?? 100;
  const papers: BiorxivPaper[] = [];

  let nextOffset: string | number | undefined = undefined;

  while (true) {
    const result = await client.scroll(options.collectionName, {
      limit: batchSize,
      with_payload: true,
      ...(nextOffset != null ? { offset: nextOffset } : {}),
    });

    for (const point of result.points) {
      const p = point.payload ?? {};
      papers.push({
        doi: (p.doi as string) ?? '',
        title: (p.title as string) ?? '',
        link: (p.link as string) ?? '',
        pdfLink: (p.pdfLink as string) ?? '',
        abstract: (p.abstract as string) ?? '',
        published: (p.published as string) ?? '',
        authors: ((p.authors as string) ?? '').split('; '),
        category: (p.category as string) ?? '',
        version: (p.version as number) ?? 1,
        relevanceScore: null,
      });
    }

    logger.info(`Extracted ${papers.length} papers so far...`);

    if (!result.next_page_offset) {
      break;
    }
    nextOffset = result.next_page_offset as string | number;
  }

  logger.info(`Total papers extracted: ${papers.length}`);
  return papers;
}

/**
 * キーワードリストで Qdrant をベクトル検索し、関連論文だけを抽出する。
 * 複数キーワードの場合は各キーワードで検索して結果を DOI で重複排除して統合する。
 */
export async function extractPapersByQueries(options: {
  queries: string[];
  collectionName: string;
  openaiApiKey: string;
  embeddingModel?: string;
  qdrantUrl?: string;
  topK?: number;
}): Promise<BiorxivPaper[]> {
  const topK = options.topK ?? 100;
  const store = new QdrantStore({
    collectionName: options.collectionName,
    openaiApiKey: options.openaiApiKey,
    ...(options.embeddingModel ? { embeddingModel: options.embeddingModel } : {}),
    ...(options.qdrantUrl ? { qdrantUrl: options.qdrantUrl } : {}),
  });

  const uniquePapers = new Map<string, BiorxivPaper>();

  for (const query of options.queries) {
    logger.info(`Searching for: "${query}" (top ${topK})`);
    const papers = await store.search(query, topK);
    for (const paper of papers) {
      if (!uniquePapers.has(paper.doi)) {
        uniquePapers.set(paper.doi, paper);
      }
    }
    logger.info(`Found ${papers.length} papers, unique so far: ${uniquePapers.size}`);
  }

  const result = Array.from(uniquePapers.values());
  logger.info(`Total unique papers for queries: ${result.length}`);
  return result;
}
