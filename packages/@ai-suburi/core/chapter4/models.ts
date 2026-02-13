import { z } from 'zod/v4';

/**
 * 検索結果の型定義
 */
export interface SearchOutput {
  fileName: string;
  content: string;
}

/**
 * 計画のZodスキーマ（Structured Output用）
 */
export const planSchema = z.object({
  subtasks: z
    .array(z.string())
    .describe('問題を解決するためのサブタスクリスト'),
});

export type Plan = z.infer<typeof planSchema>;

/**
 * ツール実行結果
 */
export interface ToolResult {
  toolName: string;
  args: string;
  results: SearchOutput[];
}

/**
 * リフレクション結果のZodスキーマ（Structured Output用）
 */
export const reflectionResultSchema = z.object({
  advice: z.string().describe(
    '評価がNGの場合は、別のツールを試す、別の文言でツールを試すなど、なぜNGなのかとどうしたら改善できるかを考えアドバイスを作成してください。' +
      'アドバイスの内容は過去のアドバイスと計画内の他のサブタスクと重複しないようにしてください。' +
      'アドバイスの内容をもとにツール選択・実行からやり直します。',
  ),
  isCompleted: z
    .boolean()
    .describe(
      'ツールの実行結果と回答から、サブタスクに対して正しく回答できているかの評価結果',
    ),
});

export type ReflectionResult = z.infer<typeof reflectionResultSchema>;

/**
 * サブタスク結果
 */
export interface Subtask {
  taskName: string;
  toolResults: ToolResult[][];
  reflectionResults: ReflectionResult[];
  isCompleted: boolean;
  subtaskAnswer: string;
  challengeCount: number;
}

/**
 * エージェント実行結果
 */
export interface AgentResult {
  question: string;
  plan: Plan;
  subtasks: Subtask[];
  answer: string;
}
