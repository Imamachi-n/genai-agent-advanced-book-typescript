import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import OpenAI from 'openai';

import { setupLogger } from '../../custom-logger.js';
import type { BiorxivPaper } from '../../models.js';

const logger = setupLogger('ideal-query-generator');

const IDEAL_QUERY_PROMPT = `\
あなたは、bioRxiv論文のベクトル検索に最適な英語検索クエリを生成する専門家です。

以下の論文のタイトルとアブストラクトを読み、この論文をベクトル検索（コサイン類似度）で確実にヒットさせるための最適な英語検索クエリを1つ生成してください。

<paper>
<title>{title}</title>
<abstract>{abstract}</abstract>
</paper>

## ルール

1. 論文の核心的なテーマを2〜3つのキーフレーズで表現すること
2. タイトルをそのままコピーしないこと（ベクトル検索では意味的な類似性が重要）
3. 論文で使われている主要な技術名・手法名を含めること
4. 自然言語の文として生成すること（キーワードの羅列ではなく）
5. 略語を使う場合は正式名称も併記すること（例: APA (Alternative Polyadenylation)）
6. 検索クエリのみを出力し、説明は含めないこと`;

const MIN_COSINE_SIMILARITY = 0.6;
const MAX_RETRIES = 3;

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

/**
 * IdealQueryGenerator のファクトリ。OpenAI クライアントを使い回すために
 * createIdealQueryGenerator() でインスタンスを生成する。
 */
export function createIdealQueryGenerator(
  llm: ChatOpenAI,
  openaiApiKey: string,
  embeddingModel: string = 'text-embedding-3-small',
): (paper: BiorxivPaper) => Promise<string> {
  const openai = new OpenAI({ apiKey: openaiApiKey });
  const prompt = ChatPromptTemplate.fromTemplate(IDEAL_QUERY_PROMPT);
  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  return async (paper: BiorxivPaper): Promise<string> => {
    // 論文の Embedding を事前計算
    const paperText = `${paper.title}\n\n${paper.abstract}`;
    const paperEmbedding = await openai.embeddings.create({
      model: embeddingModel,
      input: paperText,
    });
    const paperVector = paperEmbedding.data[0]!.embedding;

    let lastQuery = '';

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      lastQuery = await chain.invoke({
        title: paper.title,
        abstract: paper.abstract,
      });

      // クエリの Embedding を計算してコサイン類似度を検証
      const queryEmbedding = await openai.embeddings.create({
        model: embeddingModel,
        input: lastQuery,
      });
      const queryVector = queryEmbedding.data[0]!.embedding;
      const similarity = cosineSimilarity(paperVector, queryVector);

      logger.info(
        `Attempt ${attempt + 1}: similarity=${similarity.toFixed(3)} for "${lastQuery.slice(0, 80)}..."`,
      );

      if (similarity >= MIN_COSINE_SIMILARITY) {
        return lastQuery;
      }
    }

    // 最大リトライ到達：最後に生成したクエリを返す
    logger.warn(
      `Max retries reached for "${paper.title.slice(0, 60)}...". Using last generated query.`,
    );
    return lastQuery;
  };
}
