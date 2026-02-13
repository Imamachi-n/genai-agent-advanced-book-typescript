import type { AgentState, AgentSubGraphState } from '../agent.js';
import { HelpDeskAgent } from '../agent.js';
import { loadSettings } from '../configs.js';
import { searchXyzManual } from '../tools/search-xyz-manual/search-xyz-manual.js';
import { searchXyzQa } from '../tools/search-xyz-qa/search-xyz-qa.js';

const settings = loadSettings();

const agent = new HelpDeskAgent(settings, [searchXyzManual, searchXyzQa]);

const question = `
お世話になっております。

現在、XYZシステムを利用しており、以下の点についてご教示いただければと存じます。

1. 二段階認証の設定について
SMS認証が使えない環境のため、認証アプリを利用した二段階認証の設定手順を教えていただけますでしょうか。

2. バックアップ失敗時の通知について
バックアップ監視機能で通知を設定しているにもかかわらず、バックアップ失敗時に通知が届きません。確認すべき箇所を教えていただけますでしょうか。

お忙しいところ恐縮ですが、ご対応のほどよろしくお願い申し上げます。
`;

// const question = `
// お世話になっております。
//
// 現在、XYZシステムを利用を検討しており、以下の点についてご教示いただければと存じます。
//
// 1. 特定のプロジェクトに対してのみ通知を制限する方法について
//
// 2. パスワードに利用可能な文字の制限について
// 当該システムにてパスワードを設定する際、使用可能な文字の範囲（例：英数字、記号、文字数制限など）について詳しい情報をいただけますでしょうか。安全かつシステムでの認証エラーを防ぐため、具体的な仕様を確認したいと考えております。
//
// お忙しいところ恐縮ですが、ご対応のほどよろしくお願い申し上げます。
// `;

// 計画ステップ
const inputDataPlan: AgentState = {
  question,
  plan: [],
  currentStep: 0,
  subtaskResults: [],
  lastAnswer: '',
};

const planResult = await agent.createPlan(inputDataPlan);

console.log('=== Plan ===');
console.log(planResult.plan);

// ツール選択ステップ
const inputDataSelectTool: AgentSubGraphState = {
  question,
  plan: planResult.plan,
  subtask: planResult.plan[0] ?? '',
  challengeCount: 0,
  isCompleted: false,
  messages: [],
  toolResults: [],
  reflectionResults: [],
  subtaskAnswer: '',
};

const selectToolResult = await agent.selectTools(inputDataSelectTool);
console.log('\n=== Select Tool Result ===');
console.log(selectToolResult);
console.log(
  '\n=== Last Message ===',
  selectToolResult.messages[selectToolResult.messages.length - 1],
);
console.log('\n=== All Messages ===', selectToolResult.messages);

// ツール実行ステップ
const inputDataExecuteTool: AgentSubGraphState = {
  question,
  plan: planResult.plan,
  subtask: planResult.plan[0] ?? '',
  challengeCount: 0,
  messages: selectToolResult.messages,
  isCompleted: false,
  toolResults: [],
  reflectionResults: [],
  subtaskAnswer: '',
};

const toolResults = await agent.executeTools(inputDataExecuteTool);
console.log('\n=== Tool Results ===');
console.log(toolResults.toolResults[0]?.[0]?.results);
console.log(toolResults);

// サブタスク回答
const inputDataSubtaskAnswer: AgentSubGraphState = {
  question,
  plan: planResult.plan,
  subtask: planResult.plan[0] ?? '',
  challengeCount: 0,
  messages: toolResults.messages,
  toolResults: toolResults.toolResults,
  isCompleted: false,
  reflectionResults: [],
  subtaskAnswer: '',
};

const subtaskAnswer = await agent.createSubtaskAnswer(inputDataSubtaskAnswer);
console.log('\n=== Subtask Answer ===');
console.log(subtaskAnswer);
console.log('\n=== Subtask Answer Text ===');
console.log(subtaskAnswer.subtaskAnswer);

// リフレクション
const inputDataReflection: AgentSubGraphState = {
  question,
  plan: planResult.plan,
  subtask: planResult.plan[0] ?? '',
  challengeCount: 0,
  messages: subtaskAnswer.messages,
  toolResults: toolResults.toolResults,
  isCompleted: false,
  reflectionResults: [],
  subtaskAnswer: subtaskAnswer.subtaskAnswer,
};

const reflectionResult = await agent.reflectSubtask(inputDataReflection);
console.log('\n=== Reflection Result ===');
console.log(reflectionResult);

// 最初に選択されたツールを確認
const thirdMessage = reflectionResult.messages[2];
if (thirdMessage?.role === 'assistant' && 'tool_calls' in thirdMessage) {
  const firstToolCall = thirdMessage.tool_calls?.[0];
  if (firstToolCall?.type === 'function') {
    console.log(
      '\n=== Selected Tool Name ===',
      firstToolCall.function.name,
    );
  }
}

// リフレクション結果の確認
console.log(
  'is_completed =',
  reflectionResult.reflectionResults[0]?.isCompleted,
);
console.log('advice =', reflectionResult.reflectionResults[0]?.advice);
