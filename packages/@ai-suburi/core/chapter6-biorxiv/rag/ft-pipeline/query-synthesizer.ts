import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod/v4';

import { setupLogger } from '../../custom-logger.js';
import type { BiorxivPaper } from '../../models.js';

const logger = setupLogger('query-synthesizer');

export interface SyntheticQuery {
  query: string;
  language: 'ja';
  queryType: 'keyword' | 'question' | 'task-description';
}

const syntheticQueriesSchema = z.object({
  queries: z.array(
    z.object({
      query: z.string().describe('生成されたクエリ'),
      language: z.literal('ja').describe('クエリの言語（日本語固定）'),
      queryType: z
        .enum(['keyword', 'question', 'task-description'])
        .describe('クエリの種類'),
    }),
  ),
});

const SYNTHESIZE_PROMPT = `\
あなたは、バイオインフォマティクス分野の日本語ユーザーがどのような検索クエリを使うかをシミュレートする専門家です。

以下の論文情報をもとに、日本語ユーザーがこの論文を検索で見つけるために入力しそうな日本語クエリを{queries_per_paper}種類生成してください。

<paper>
<title>{title}</title>
<abstract>{abstract}</abstract>
<category>{category}</category>
</paper>

## 生成ルール

1. 以下の3種類をすべて日本語で生成してください:
   - キーワード型: スペース区切りの日本語キーワード。説明部分は必ず日本語にすること。
   - 質問型: 自然な日本語の質問文
   - タスク記述型: 「〜を調べる」「〜を調査する」「〜を分析する」で終わる日本語の調査タスク文

2. 重要: 説明・接続・動詞は必ず日本語にしてください。専門用語（ツール名・技術名の固有名詞）のみ英語を許可します。
3. クエリは論文タイトルのコピーではなく、研究者が実際に入力しそうな自然な表現にしてください
4. 各クエリは異なるキーワードやアプローチを使ってください

## 出力例

以下は「single-cell RNA-seq の細胞アノテーション自動化ツール」に関する論文の場合の例です:

<example>
- キーワード型: 「scRNA-seq 細胞タイプ自動分類 ツール」
- 質問型: 「scRNA-seqデータの細胞アノテーションを自動化するツールにはどんなものがありますか？」
- タスク記述型: 「scRNA-seqにおける自動細胞分類手法の最新動向を調査する」
</example>

NG例:
- NG: 「long-read mapping parameter optimization CycSim context-aware simulation」（英語のキーワード羅列）
- OK: 「ロングリードのマッピングパラメータ最適化 CycSim」（日本語の説明 + 固有名詞のみ英語）`;

/**
 * 論文ごとに多様な合成ユーザークエリを LLM で生成する。
 */
export async function synthesizeUserQueries(
  llm: ChatOpenAI,
  paper: BiorxivPaper,
  queriesPerPaper: number = 3,
): Promise<SyntheticQuery[]> {
  const prompt = ChatPromptTemplate.fromTemplate(SYNTHESIZE_PROMPT);
  const chain = prompt.pipe(
    llm.withStructuredOutput(syntheticQueriesSchema),
  );

  const result = await chain.invoke({
    title: paper.title,
    abstract: paper.abstract,
    category: paper.category,
    queries_per_paper: queriesPerPaper,
  });

  logger.info(
    `Generated ${result.queries.length} synthetic queries for: ${paper.title.slice(0, 60)}...`,
  );

  return result.queries.map((q) => ({
    query: q.query,
    language: q.language,
    queryType: q.queryType,
  }));
}
