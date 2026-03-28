import { ChromaClient } from 'chromadb';
import OpenAI from 'openai';

import type { BiorxivPaper } from '../models.js';
import { setupLogger } from '../custom-logger.js';

const logger = setupLogger('chroma-store');

export class ChromaStore {
  private client: ChromaClient;
  private openai: OpenAI;
  private embeddingModel: string;
  private collectionName: string;

  constructor(options: {
    collectionName: string;
    openaiApiKey: string;
    embeddingModel?: string;
    chromaHost?: string;
  }) {
    this.client = new ChromaClient({
      host: options.chromaHost ?? 'http://localhost:8000',
    });
    this.openai = new OpenAI({ apiKey: options.openaiApiKey });
    this.embeddingModel = options.embeddingModel ?? 'text-embedding-3-small';
    this.collectionName = options.collectionName;
  }

  private async getEmbeddings(texts: string[]): Promise<number[][]> {
    const response = await this.openai.embeddings.create({
      model: this.embeddingModel,
      input: texts,
    });
    return response.data.map((d) => d.embedding);
  }

  async addDocuments(papers: BiorxivPaper[]): Promise<void> {
    const collection = await this.client.getOrCreateCollection({
      name: this.collectionName,
    });

    const ids = papers.map((p) => p.doi);
    const documents = papers.map(
      (p) => `${p.title}\n\n${p.abstract}`,
    );
    const metadatas = papers.map((p) => ({
      doi: p.doi,
      title: p.title,
      authors: p.authors.join('; '),
      published: p.published,
      category: p.category,
      pdfLink: p.pdfLink,
      link: p.link,
      version: p.version,
    }));

    // OpenAI でエンベディングを計算
    const embeddings = await this.getEmbeddings(documents);

    await collection.add({ ids, documents, metadatas, embeddings });
    logger.info(`Added ${papers.length} documents to collection "${this.collectionName}"`);
  }

  async search(
    query: string,
    topK: number = 20,
  ): Promise<BiorxivPaper[]> {
    const collection = await this.client.getOrCreateCollection({
      name: this.collectionName,
    });

    // クエリのエンベディングを計算
    const queryEmbeddings = await this.getEmbeddings([query]);

    const results = await collection.query({
      queryEmbeddings,
      nResults: topK,
    });

    const papers: BiorxivPaper[] = [];
    const metadatas = results.metadatas?.[0] ?? [];
    const distances = results.distances?.[0] ?? [];

    for (let i = 0; i < metadatas.length; i++) {
      const meta = metadatas[i];
      if (!meta) continue;

      // Chroma の distance を類似度スコアに変換（L2距離: 小さいほど類似）
      const distance = distances[i] ?? 0;
      const similarityScore = 1 / (1 + distance);

      papers.push({
        doi: (meta.doi as string) ?? '',
        title: (meta.title as string) ?? '',
        link: (meta.link as string) ?? '',
        pdfLink: (meta.pdfLink as string) ?? '',
        abstract: '', // クエリ結果にはドキュメントテキストのみ
        published: (meta.published as string) ?? '',
        authors: ((meta.authors as string) ?? '').split('; '),
        category: (meta.category as string) ?? '',
        version: (meta.version as number) ?? 1,
        relevanceScore: similarityScore,
      });
    }

    return papers;
  }

  async getDocumentCount(): Promise<number> {
    const collection = await this.client.getOrCreateCollection({
      name: this.collectionName,
    });
    return collection.count();
  }

  async exists(doi: string): Promise<boolean> {
    const collection = await this.client.getOrCreateCollection({
      name: this.collectionName,
    });
    const result = await collection.get({ ids: [doi] });
    return result.ids.length > 0;
  }
}
