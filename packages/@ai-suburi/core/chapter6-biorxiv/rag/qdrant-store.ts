import { createHash } from 'node:crypto';
import { QdrantClient } from '@qdrant/js-client-rest';
import OpenAI from 'openai';
import { setupLogger } from '../custom-logger.js';
import type { BiorxivPaper } from '../models.js';

const logger = setupLogger('qdrant-store');

export class QdrantStore {
  private client: QdrantClient;
  private openai: OpenAI;
  private embeddingModel: string;
  private collectionName: string;
  private vectorSize: number;

  constructor(options: {
    collectionName: string;
    openaiApiKey: string;
    embeddingModel?: string;
    qdrantUrl?: string;
    vectorSize?: number;
  }) {
    this.client = new QdrantClient({
      url: options.qdrantUrl ?? 'http://localhost:6333',
    });
    this.openai = new OpenAI({ apiKey: options.openaiApiKey });
    this.embeddingModel = options.embeddingModel ?? 'text-embedding-3-small';
    this.collectionName = options.collectionName;
    // text-embedding-3-small = 1536次元
    this.vectorSize = options.vectorSize ?? 1536;
  }

  /** コレクションが存在しなければ作成する */
  async ensureCollection(): Promise<void> {
    const collections = await this.client.getCollections();
    const exists = collections.collections.some(
      (c) => c.name === this.collectionName,
    );
    if (!exists) {
      await this.client.createCollection(this.collectionName, {
        vectors: {
          size: this.vectorSize,
          distance: 'Cosine',
        },
      });
      // DOI でフィルタリングするためのペイロードインデックス
      await this.client.createPayloadIndex(this.collectionName, {
        field_name: 'doi',
        field_schema: 'keyword',
      });
      logger.info(`Created collection "${this.collectionName}"`);
    }
  }

  private async getEmbeddings(texts: string[]): Promise<number[][]> {
    const response = await this.openai.embeddings.create({
      model: this.embeddingModel,
      input: texts,
    });
    return response.data.map((d) => d.embedding);
  }

  async addDocuments(papers: BiorxivPaper[]): Promise<void> {
    await this.ensureCollection();

    const documents = papers.map((p) => `${p.title}\n\n${p.abstract}`);
    const embeddings = await this.getEmbeddings(documents);

    const points = papers.map((paper, i) => ({
      id: this.doiToPointId(paper.doi),
      vector: embeddings[i]!,
      payload: {
        doi: paper.doi,
        title: paper.title,
        authors: paper.authors.join('; '),
        published: paper.published,
        category: paper.category,
        pdfLink: paper.pdfLink,
        link: paper.link,
        version: paper.version,
        document: documents[i]!,
      },
    }));

    await this.client.upsert(this.collectionName, { points });
    logger.info(
      `Added ${papers.length} documents to collection "${this.collectionName}"`,
    );
  }

  async search(query: string, topK: number = 20): Promise<BiorxivPaper[]> {
    await this.ensureCollection();

    const queryEmbeddings = await this.getEmbeddings([query]);

    const results = await this.client.search(this.collectionName, {
      vector: queryEmbeddings[0]!,
      limit: topK,
      with_payload: true,
    });

    return results.map((result) => {
      const p = result.payload ?? {};
      return {
        doi: (p.doi as string) ?? '',
        title: (p.title as string) ?? '',
        link: (p.link as string) ?? '',
        pdfLink: (p.pdfLink as string) ?? '',
        abstract: '',
        published: (p.published as string) ?? '',
        authors: ((p.authors as string) ?? '').split('; '),
        category: (p.category as string) ?? '',
        version: (p.version as number) ?? 1,
        relevanceScore: result.score,
      };
    });
  }

  async getDocumentCount(): Promise<number> {
    await this.ensureCollection();
    const info = await this.client.getCollection(this.collectionName);
    return info.points_count ?? 0;
  }

  async exists(doi: string): Promise<boolean> {
    await this.ensureCollection();
    const results = await this.client.scroll(this.collectionName, {
      filter: {
        must: [{ key: 'doi', match: { value: doi } }],
      },
      limit: 1,
    });
    return results.points.length > 0;
  }

  /** DOI を SHA-1 ハッシュして UUID v5 形式の文字列に変換（Qdrant は数値 or UUID を ID に使う） */
  private doiToPointId(doi: string): string {
    const hash = createHash('sha1').update(doi).digest('hex');
    // SHA-1 の先頭 32 文字を UUID 形式 (8-4-4-4-12) に整形
    return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-${hash.slice(12, 16)}-${hash.slice(16, 20)}-${hash.slice(20, 32)}`;
  }
}
