import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import OpenAI from 'openai';

import { setupLogger } from '../custom-logger.js';
import type { BiorxivPaper } from '../models.js';
import type { Searcher } from '../searcher/searcher.js';
import { QdrantStore } from './qdrant-store.js';

const logger = setupLogger('rag-searcher');

const EXPAND_QUERY_PROMPT = `\
<system>
あなたは、与えられたサブクエリからbioRxiv論文のRAG検索に最適な検索クエリを生成する専門家です。
バイオインフォマティクス分野の学術的な文脈を理解し、ベクトル検索で効果的にヒットするクエリを作成してください。

{feedback}
</system>

## 主要タスク

1. 提供されたサブクエリを分析する
2. サブクエリから重要なキーワードを抽出する
3. ベクトル検索に最適化された自然言語の検索クエリを構築する

## 重要なルール

<rules>
1. 検索クエリは英語で生成すること（bioRxivの論文は英語のため）
2. クエリには1〜3つの主要なキーワードまたはフレーズを含めること
3. バイオインフォマティクスの専門用語を適切に使用すること
4. 自然言語の文として生成すること（ベクトル検索に最適）
5. 説明や理由付けは含めず、純粋な検索クエリのみを出力すること
</rules>

## 入力フォーマット

<input_format>
目標: {goal_setting}
クエリ: {query}
</input_format>

REMEMBER: rulesタグの内容に必ず従うこと`;

function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class RagSearcher implements Searcher {
  static readonly RELEVANCE_SCORE_THRESHOLD = 0.3;

  private llm: ChatOpenAI;
  private store: QdrantStore;
  private openai: OpenAI;
  private embeddingModel: string;
  private maxSearchResults: number;
  private maxPapers: number;
  private maxRetries: number;
  private debug: boolean;

  constructor(
    llm: ChatOpenAI,
    store: QdrantStore,
    options: {
      openaiApiKey: string;
      embeddingModel?: string;
      maxSearchResults?: number;
      maxPapers?: number;
      maxRetries?: number;
      debug?: boolean;
    },
  ) {
    this.llm = llm;
    this.store = store;
    this.openai = new OpenAI({ apiKey: options.openaiApiKey });
    this.embeddingModel = options.embeddingModel ?? 'text-embedding-3-small';
    this.maxSearchResults = options.maxSearchResults ?? 20;
    this.maxPapers = options.maxPapers ?? 3;
    this.maxRetries = options.maxRetries ?? 3;
    this.debug = options.debug ?? true;
  }

  private async expandQuery(
    goalSetting: string,
    query: string,
    feedback: string = '',
  ): Promise<string> {
    const prompt = ChatPromptTemplate.fromTemplate(EXPAND_QUERY_PROMPT);
    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
    return chain.invoke({
      goal_setting: goalSetting,
      query,
      feedback,
    });
  }

  private async rerank(
    query: string,
    papers: BiorxivPaper[],
  ): Promise<BiorxivPaper[]> {
    // クエリのエンベディングを取得
    const queryEmbedding = await this.openai.embeddings.create({
      model: this.embeddingModel,
      input: query,
    });
    const queryVector = queryEmbedding.data[0]!.embedding;

    // 各論文のエンベディングを取得してコサイン類似度を計算
    const paperTexts = papers.map((p) => `${p.title}\n${p.abstract}`);
    const paperEmbeddings = await this.openai.embeddings.create({
      model: this.embeddingModel,
      input: paperTexts,
    });

    const scored = papers.map((paper, i) => {
      const paperVector = paperEmbeddings.data[i]!.embedding;
      const score = cosineSimilarity(queryVector, paperVector);
      return { paper: { ...paper, relevanceScore: score }, score };
    });

    // スコア降順でソート
    scored.sort((a, b) => b.score - a.score);

    return scored
      .slice(0, this.maxPapers)
      .filter((s) => s.score >= RagSearcher.RELEVANCE_SCORE_THRESHOLD)
      .map((s) => s.paper);
  }

  async run(goalSetting: string, query: string, excludeDois: string[] = []): Promise<BiorxivPaper[]> {
    let retryCount = 0;
    let feedback = '';
    let papers: BiorxivPaper[] = [];

    while (retryCount < this.maxRetries) {
      const expandedQuery = await this.expandQuery(
        goalSetting,
        query,
        feedback,
      );
      logger.info(`Searching with query: "${expandedQuery}"`);

      papers = await this.store.search(expandedQuery, this.maxSearchResults, excludeDois);

      if (this.debug) {
        logger.info(`Found ${papers.length} papers from Chroma.`);
      }

      if (papers.length > 0) {
        // リランキング: OpenAI Embeddings + コサイン類似度
        const searchQuery = `${goalSetting}\n${query}`;
        papers = await this.rerank(searchQuery, papers);
        logger.info(`After reranking: ${papers.length} papers above threshold.`);

        if (papers.length > 0) {
          break;
        }
      }

      retryCount++;
      if (retryCount < this.maxRetries) {
        feedback =
          '検索結果が0件または関連度が低い結果のみでした。より一般的なキーワードや別の関連用語で検索してください。';
        logger.info(
          `Retrying with adjusted query. Attempt ${retryCount}/${this.maxRetries}`,
        );
      } else {
        logger.info('Max retries reached. No results found.');
      }
    }

    return papers;
  }
}
