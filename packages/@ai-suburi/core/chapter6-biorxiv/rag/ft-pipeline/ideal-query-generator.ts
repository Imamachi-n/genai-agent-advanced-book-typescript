import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import OpenAI from 'openai';

import { setupLogger } from '../../custom-logger.js';
import type { BiorxivPaper } from '../../models.js';

const logger = setupLogger('ideal-query-generator');

/** LangChain テンプレートの変数展開を防ぐために波括弧をエスケープする */
function escapeBraces(text: string): string {
  return text.replace(/\{/g, '{{').replace(/\}/g, '}}');
}

const IDEAL_QUERY_PROMPT = `\
あなたは、bioRxiv論文のベクトル検索に最適な英語検索クエリを生成する専門家です。

以下の論文のタイトルとアブストラクトを読み、この論文をベクトル検索（コサイン類似度）で確実にヒットさせるための最適な英語検索クエリを1つ生成してください。

<paper>
<title>{title}</title>
<abstract>{abstract}</abstract>
</paper>

## ルール

1. 出力は必ず15語以内の簡潔な英語クエリにすること（論文の要約文ではなく検索クエリ）
2. 論文の核心的なテーマを2〜3つのキーフレーズで表現すること
3. タイトルをそのままコピーしないこと（ベクトル検索では意味的な類似性が重要）
4. 論文で使われている主要な技術名・手法名・ツール名を含めること
5. 略語を使う場合は正式名称も併記すること（例: APA (Alternative Polyadenylation)）
6. 検索クエリのみを出力し、説明や結果の記述は含めないこと
7. NG例: "Using shotgun metagenomic sequencing, this study introduces PStrain-tracer, a framework to track..."（長すぎ・論文要約）
8. OK例: "strain-level engraftment tracing FMT shotgun metagenomics PStrain-tracer"（簡潔・検索向き）`;

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
 * IdealQueryGenerator のファクトリ。
 * skipEmbeddingCheck=true で Embedding 品質検証をスキップし高速化できる。
 */
export function createIdealQueryGenerator(
  llm: ChatOpenAI,
  openaiApiKey: string,
  options: {
    embeddingModel?: string;
    skipEmbeddingCheck?: boolean;
  } = {},
): (paper: BiorxivPaper) => Promise<string> {
  const embeddingModel = options.embeddingModel ?? 'text-embedding-3-small';
  const skipCheck = options.skipEmbeddingCheck ?? false;
  const openai = skipCheck ? null : new OpenAI({ apiKey: openaiApiKey });
  const prompt = ChatPromptTemplate.fromTemplate(IDEAL_QUERY_PROMPT);
  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  return async (paper: BiorxivPaper): Promise<string> => {
    // スキップモード: LLM 1 回コールのみ（Embedding 検証なし）
    if (skipCheck || !openai) {
      const query = await chain.invoke({
        title: escapeBraces(paper.title),
        abstract: escapeBraces(paper.abstract),
      });
      logger.info(`Generated (no validation): "${query.slice(0, 80)}..."`);
      return query;
    }

    // 検証モード: Embedding コサイン類似度チェック付き
    const paperText = `${paper.title}\n\n${paper.abstract}`;
    const paperEmbedding = await openai.embeddings.create({
      model: embeddingModel,
      input: paperText,
    });
    const paperVector = paperEmbedding.data[0]!.embedding;

    let lastQuery = '';

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      lastQuery = await chain.invoke({
        title: escapeBraces(paper.title),
        abstract: escapeBraces(paper.abstract),
      });

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

    logger.warn(
      `Max retries reached for "${paper.title.slice(0, 60)}...". Using last generated query.`,
    );
    return lastQuery;
  };
}
