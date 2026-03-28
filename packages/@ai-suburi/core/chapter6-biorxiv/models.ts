import { z } from 'zod/v4';

// --- BiorxivPaper ---
export const biorxivPaperSchema = z.object({
  doi: z.string().describe('bioRxiv DOI'),
  title: z.string().describe('論文タイトル'),
  link: z.string().describe('論文リンク'),
  pdfLink: z.string().describe('PDFリンク'),
  abstract: z.string().describe('論文アブストラクト'),
  published: z.string().describe('公開日 (YYYY-MM-DD)'),
  authors: z.array(z.string()).describe('著者'),
  category: z.string().describe('カテゴリ'),
  version: z.number().describe('バージョン'),
  relevanceScore: z.number().nullable().optional().describe('関連度スコア（コサイン類似度）'),
});
export type BiorxivPaper = z.infer<typeof biorxivPaperSchema>;

export function biorxivPaperToXml(paper: BiorxivPaper): string {
  return `<paper>
  <doi>${paper.doi}</doi>
  <title>${paper.title}</title>
  <link>${paper.link}</link>
  <pdf_link>${paper.pdfLink}</pdf_link>
  <abstract>${paper.abstract}</abstract>
  <published>${paper.published}</published>
  <version>${paper.version}</version>
  <authors>${paper.authors.join(', ')}</authors>
  <category>${paper.category}</category>
  ${paper.relevanceScore != null ? `<relevance_score>${paper.relevanceScore}</relevance_score>` : ''}
</paper>`;
}

// --- ReadingResult ---
export const readingResultSchema = z.object({
  id: z.number().describe('ID'),
  task: z.string().describe('調査タスク'),
  paper: biorxivPaperSchema.describe('論文データ'),
  markdownPath: z.string().describe('論文のテキストファイルへの相対パス'),
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
