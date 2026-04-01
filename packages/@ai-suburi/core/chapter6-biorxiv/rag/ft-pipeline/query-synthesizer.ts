import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod/v4';

import { setupLogger } from '../../custom-logger.js';
import type { BiorxivPaper } from '../../models.js';

const logger = setupLogger('query-synthesizer');

export interface SyntheticQuery {
  query: string;
  language: 'ja' | 'en';
  queryType: 'keyword' | 'question' | 'task-description' | 'goal';
}

const syntheticQueriesSchema = z.object({
  queries: z.array(
    z.object({
      query: z.string().describe('生成されたクエリ'),
      language: z.enum(['ja', 'en']).describe('クエリの言語'),
      queryType: z
        .enum(['keyword', 'question', 'task-description', 'goal'])
        .describe('クエリの種類'),
    }),
  ),
});

const SYNTHESIZE_PROMPT = `\
あなたは、バイオインフォマティクス分野の研究者がどのような検索クエリを使うかをシミュレートする専門家です。

以下の論文情報をもとに、この論文を検索で見つけるために研究者が入力しそうなクエリを{queries_per_paper}種類生成してください。

<paper>
<title>{title}</title>
<abstract>{abstract}</abstract>
<category>{category}</category>
</paper>

## 生成ルール

1. 以下の5種類を1つずつ生成してください:
   - 日本語キーワード型（例: 「一細胞RNA解析 最新手法」）
   - 日本語質問型（例: 「scRNA-seqの細胞アノテーション自動化ツールは？」）
   - 英語キーワード型（例: 「single-cell RNA-seq cell type annotation」）
   - 英語タスク記述型（例: 「Investigate automated methods for cell classification in scRNA-seq datasets」）
   - ゴール記述型（日英どちらでも可。例: 「生成AIを用いたゲノム解析の最新動向を調べる」）

2. クエリは論文タイトルのコピーではなく、研究者が実際に入力しそうな自然な表現にしてください
3. 略語と正式名称の両方を使い分けてください
4. 各クエリは異なるキーワードやアプローチを使ってください`;

/**
 * 論文ごとに多様な合成ユーザークエリを LLM で生成する。
 */
export async function synthesizeUserQueries(
  llm: ChatOpenAI,
  paper: BiorxivPaper,
  queriesPerPaper: number = 5,
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
