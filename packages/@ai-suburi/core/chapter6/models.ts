import { z } from 'zod/v4';

// --- ArxivPaper ---
export const arxivPaperSchema = z.object({
  id: z.string().describe('arXiv ID'),
  title: z.string().describe('論文タイトル'),
  link: z.string().describe('論文リンク'),
  pdfLink: z.string().describe('PDFリンク'),
  abstract: z.string().describe('論文アブストラクト'),
  published: z.string().describe('公開日 (ISO string)'),
  updated: z.string().describe('更新日 (ISO string)'),
  version: z.number().describe('バージョン'),
  authors: z.array(z.string()).describe('著者'),
  categories: z.array(z.string()).describe('カテゴリ'),
  relevanceScore: z.number().nullable().optional().describe('関連度スコア'),
});
export type ArxivPaper = z.infer<typeof arxivPaperSchema>;

export function arxivPaperToXml(paper: ArxivPaper): string {
  return `<paper>
  <id>${paper.id}</id>
  <title>${paper.title}</title>
  <link>${paper.link}</link>
  <pdf_link>${paper.pdfLink}</pdf_link>
  <abstract>${paper.abstract}</abstract>
  <published>${paper.published}</published>
  <updated>${paper.updated}</updated>
  <version>${paper.version}</version>
  <authors>${paper.authors.join(', ')}</authors>
  <categories>${paper.categories.join(', ')}</categories>
  ${paper.relevanceScore != null ? `<relevance_score>${paper.relevanceScore}</relevance_score>` : ''}
</paper>`;
}

// --- ReadingResult ---
export const readingResultSchema = z.object({
  id: z.number().describe('ID'),
  task: z.string().describe('調査タスク'),
  paper: arxivPaperSchema.describe('論文データ'),
  markdownPath: z.string().describe('論文のmarkdownファイルへの相対パス'),
  answer: z.string().default('').describe('タスクに対する回答'),
  isRelated: z.boolean().nullable().optional().describe('タスクとの関係性'),
});
export type ReadingResult = z.infer<typeof readingResultSchema>;

// --- Section ---
export const sectionSchema = z.object({
  header: z.string().describe('セクションのヘッダー'),
  content: z.string().describe('セクションの内容'),
  charCount: z.number().describe('セクションの文字数'),
});
export type Section = z.infer<typeof sectionSchema>;

// --- LLM 構造化出力用スキーマ ---

export const hearingSchema = z.object({
  is_need_human_feedback: z
    .boolean()
    .describe('追加の質問が必要かどうか'),
  additional_question: z.string().describe('追加の質問'),
});
export type Hearing = z.infer<typeof hearingSchema>;

export const decomposedTasksSchema = z.object({
  tasks: z.array(z.string()).describe('分解されたタスクのリスト'),
});
export type DecomposedTasks = z.infer<typeof decomposedTasksSchema>;

export const taskEvaluationSchema = z.object({
  need_more_information: z
    .boolean()
    .describe('必要な情報が足りている場合はfalse'),
  reason: z.string().describe('評価の理由を日本語で端的に表す'),
  content: z
    .string()
    .describe('追加の調査として必要な内容を詳細に日本語で記述'),
});
export type TaskEvaluation = z.infer<typeof taskEvaluationSchema>;

export const sufficiencySchema = z.object({
  is_sufficient: z.boolean().describe('十分かどうか'),
  reason: z.string().describe('十分性の判断理由'),
});
export type Sufficiency = z.infer<typeof sufficiencySchema>;

export const arxivFieldsSchema = z.object({
  values: z
    .array(z.string())
    .describe(
      'The arXiv categories that need to be searched based on the user\'s query.',
    ),
});
export type ArxivFields = z.infer<typeof arxivFieldsSchema>;

export const arxivTimeRangeSchema = z.object({
  start: z
    .string()
    .nullable()
    .describe('The start date of the time range (YYYY-MM-DD). null if not specified.'),
  end: z
    .string()
    .nullable()
    .describe('The end date of the time range (YYYY-MM-DD). null if not specified.'),
});
export type ArxivTimeRange = z.infer<typeof arxivTimeRangeSchema>;

export function formatTimeRange(range: ArxivTimeRange): string | null {
  const formatDate = (dateStr: string): string => {
    return dateStr.replace(/-/g, '');
  };
  if (range.start && range.end) {
    return `${formatDate(range.start)}+TO+${formatDate(range.end)}`;
  }
  if (range.start) {
    return `${formatDate(range.start)}+TO+LATEST`;
  }
  if (range.end) {
    return `EARLIEST+TO+${formatDate(range.end)}`;
  }
  return null;
}
