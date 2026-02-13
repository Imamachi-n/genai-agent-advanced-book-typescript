import type { StructuredToolInterface } from '@langchain/core/tools';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import { Annotation, END, Send, START, StateGraph } from '@langchain/langgraph';
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

import type { Settings } from './configs.js';
import { setupLogger } from './custom-logger.js';
import {
  type AgentResult,
  planSchema,
  type ReflectionResult,
  reflectionResultSchema,
  type SearchOutput,
  type Subtask,
  type ToolResult,
} from './models.js';
import { HelpDeskAgentPrompts } from './prompts.js';

const MAX_CHALLENGE_COUNT = 3;

const logger = setupLogger('agent');

// メイングラフの状態定義
const AgentStateAnnotation = Annotation.Root({
  question: Annotation<string>,
  plan: Annotation<string[]>,
  currentStep: Annotation<number>,
  subtaskResults: Annotation<Subtask[]>({
    reducer: (a: Subtask[], b: Subtask[]) => [...a, ...b],
    default: () => [],
  }),
  lastAnswer: Annotation<string>,
});

export type AgentState = typeof AgentStateAnnotation.State;

// サブグラフの状態定義
const AgentSubGraphStateAnnotation = Annotation.Root({
  question: Annotation<string>,
  plan: Annotation<string[]>,
  subtask: Annotation<string>,
  isCompleted: Annotation<boolean>,
  messages: Annotation<ChatCompletionMessageParam[]>({
    // 新しい値で上書き（accumulate不要のためlast write wins）
    reducer: (
      _old: ChatCompletionMessageParam[],
      newVal: ChatCompletionMessageParam[],
    ) => newVal,
    default: () => [],
  }),
  challengeCount: Annotation<number>,
  toolResults: Annotation<ToolResult[][]>({
    reducer: (a: ToolResult[][], b: ToolResult[][]) => [...a, ...b],
    default: () => [],
  }),
  reflectionResults: Annotation<ReflectionResult[]>({
    reducer: (a: ReflectionResult[], b: ReflectionResult[]) => [...a, ...b],
    default: () => [],
  }),
  subtaskAnswer: Annotation<string>,
});

export type AgentSubGraphState = typeof AgentSubGraphStateAnnotation.State;

export class HelpDeskAgent {
  private settings: Settings;
  private tools: StructuredToolInterface[];
  private toolMap: Map<string, StructuredToolInterface>;
  private prompts: HelpDeskAgentPrompts;
  private client: OpenAI;

  constructor(
    settings: Settings,
    tools: StructuredToolInterface[] = [],
    prompts: HelpDeskAgentPrompts = new HelpDeskAgentPrompts(),
  ) {
    this.settings = settings;
    this.tools = tools;
    this.toolMap = new Map(tools.map((tool) => [tool.name, tool]));
    this.prompts = prompts;
    this.client = new OpenAI({ apiKey: this.settings.openaiApiKey });
  }

  /**
   * 計画を作成する
   */
  async createPlan(state: AgentState): Promise<{ plan: string[] }> {
    logger.info('🚀 Starting plan generation process...');

    const systemPrompt = this.prompts.plannerSystemPrompt;
    const userPrompt = this.prompts.plannerUserPrompt.replace(
      '{question}',
      state.question,
    );

    const messages: ChatCompletionMessageParam[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ];
    logger.debug(`Final prompt messages: ${JSON.stringify(messages)}`);

    logger.info('Sending request to OpenAI...');
    const response = await this.client.chat.completions.parse({
      model: this.settings.openaiModel,
      messages,
      response_format: zodResponseFormat(planSchema, 'plan'),
      temperature: 0,
      seed: 0,
    });
    logger.info('✅ Successfully received response from OpenAI.');

    const plan = response.choices[0]?.message.parsed ?? { subtasks: [] };

    logger.info('Plan generation complete!');

    return { plan: plan.subtasks };
  }

  /**
   * ツールを選択する
   */
  async selectTools(
    state: AgentSubGraphState,
  ): Promise<{ messages: ChatCompletionMessageParam[] }> {
    logger.info('🚀 Starting tool selection process...');

    // OpenAI対応のtool定義に変換
    logger.debug('Converting tools for OpenAI format...');
    const openaiTools = this.tools.map((tool) => convertToOpenAITool(tool));

    let messages: ChatCompletionMessageParam[];

    if (state.challengeCount === 0) {
      logger.debug('Creating user prompt for tool selection...');
      const userPrompt = this.prompts.subtaskToolSelectionUserPrompt
        .replace('{question}', state.question)
        .replace('{plan}', JSON.stringify(state.plan))
        .replace('{subtask}', state.subtask);

      messages = [
        { role: 'system', content: this.prompts.subtaskSystemPrompt },
        { role: 'user', content: userPrompt },
      ];
    } else {
      logger.debug('Creating user prompt for tool retry...');

      // リトライされた場合は過去の対話情報にプロンプトを追加する
      // NOTE: トークン数節約のため過去の検索結果は除く
      // roleがtoolまたはtool_callsを持つものは除く
      messages = state.messages.filter(
        (message) => message.role !== 'tool' || !('tool_calls' in message),
      );

      const userRetryPrompt = this.prompts.subtaskRetryAnswerUserPrompt;
      messages.push({ role: 'user', content: userRetryPrompt });
    }

    logger.info('Sending request to OpenAI...');
    const response = await this.client.chat.completions.create({
      model: this.settings.openaiModel,
      messages,
      tools: openaiTools,
      temperature: 0,
      seed: 0,
    });
    logger.info('✅ Successfully received response from OpenAI.');

    const selectChoice = response.choices[0];
    const toolCalls = selectChoice?.message.tool_calls;
    if (!toolCalls || toolCalls.length === 0) {
      // モデルがツールを呼ばずにテキストで応答した場合
      logger.info('No tool calls returned, using text response as fallback.');
      const textContent = selectChoice?.message.content ?? '';
      messages.push({ role: 'assistant', content: textContent });
      return { messages };
    }

    const aiMessage: ChatCompletionMessageParam = {
      role: 'assistant',
      tool_calls: toolCalls,
    };

    logger.info('Tool selection complete!');
    messages.push(aiMessage);

    return { messages };
  }

  /**
   * ツールを実行する
   */
  async executeTools(state: AgentSubGraphState): Promise<{
    messages: ChatCompletionMessageParam[];
    toolResults: ToolResult[][];
  }> {
    logger.info('🚀 Starting tool execution process...');
    const messages = [...state.messages];

    const lastMessage = messages[messages.length - 1];
    const toolCalls =
      lastMessage?.role === 'assistant' && 'tool_calls' in lastMessage
        ? lastMessage.tool_calls
        : null;

    if (!toolCalls || toolCalls.length === 0) {
      // selectToolsでツールが選択されなかった場合はスキップ
      logger.info('No tool calls found, skipping tool execution.');
      return { messages, toolResults: [[]] };
    }

    const toolResults: ToolResult[] = [];

    for (const toolCall of toolCalls) {
      if (toolCall.type !== 'function') {
        continue;
      }
      const toolName: string = toolCall.function.name;
      const toolArgs: string = toolCall.function.arguments;

      const tool = this.toolMap.get(toolName);
      if (!tool) {
        throw new Error(`Tool not found: ${toolName}`);
      }
      const toolResult: SearchOutput[] = await tool.invoke(
        JSON.parse(toolArgs),
      );

      toolResults.push({
        toolName,
        args: toolArgs,
        results: toolResult,
      });

      messages.push({
        role: 'tool',
        content: JSON.stringify(toolResult),
        tool_call_id: toolCall.id,
      });
    }
    logger.info('Tool execution complete!');
    return { messages, toolResults: [toolResults] };
  }

  /**
   * サブタスク回答を作成する
   */
  async createSubtaskAnswer(state: AgentSubGraphState): Promise<{
    messages: ChatCompletionMessageParam[];
    subtaskAnswer: string;
  }> {
    logger.info('🚀 Starting subtask answer creation process...');
    const messages = [...state.messages];

    logger.info('Sending request to OpenAI...');
    const response = await this.client.chat.completions.create({
      model: this.settings.openaiModel,
      messages,
      temperature: 0,
      seed: 0,
    });
    logger.info('✅ Successfully received response from OpenAI.');

    const subtaskAnswer = response.choices[0]?.message.content ?? '';

    const aiMessage: ChatCompletionMessageParam = {
      role: 'assistant',
      content: subtaskAnswer,
    };
    messages.push(aiMessage);

    logger.info('Subtask answer creation complete!');

    return { messages, subtaskAnswer };
  }

  /**
   * サブタスク回答を内省する
   */
  async reflectSubtask(state: AgentSubGraphState): Promise<{
    messages: ChatCompletionMessageParam[];
    reflectionResults: ReflectionResult[];
    challengeCount: number;
    isCompleted: boolean;
    subtaskAnswer?: string;
  }> {
    logger.info('🚀 Starting reflection process...');
    const messages = [...state.messages];

    const userPrompt = this.prompts.subtaskReflectionUserPrompt;
    messages.push({ role: 'user', content: userPrompt });

    logger.info('Sending request to OpenAI...');
    const response = await this.client.chat.completions.parse({
      model: this.settings.openaiModel,
      messages,
      response_format: zodResponseFormat(
        reflectionResultSchema,
        'reflection_result',
      ),
      temperature: 0,
      seed: 0,
    });
    logger.info('✅ Successfully received response from OpenAI.');

    const reflectionResult = response.choices[0]?.message.parsed ?? {
      advice: '',
      isCompleted: false,
    };

    messages.push({
      role: 'assistant',
      content: JSON.stringify(reflectionResult),
    });

    const updateState: {
      messages: ChatCompletionMessageParam[];
      reflectionResults: ReflectionResult[];
      challengeCount: number;
      isCompleted: boolean;
      subtaskAnswer?: string;
    } = {
      messages,
      reflectionResults: [reflectionResult],
      challengeCount: state.challengeCount + 1,
      isCompleted: reflectionResult.isCompleted,
    };

    if (
      updateState.challengeCount >= MAX_CHALLENGE_COUNT &&
      !reflectionResult.isCompleted
    ) {
      updateState.subtaskAnswer = `${state.subtask}の回答が見つかりませんでした。`;
    }

    logger.info('Reflection complete!');
    return updateState;
  }

  /**
   * 最終回答を作成する
   */
  async createAnswer(state: AgentState): Promise<{ lastAnswer: string }> {
    logger.info('🚀 Starting final answer creation process...');
    const systemPrompt = this.prompts.createLastAnswerSystemPrompt;

    // サブタスク結果のうちタスク内容と回答のみを取得
    const subtaskResults = state.subtaskResults.map((result) => [
      result.taskName,
      result.subtaskAnswer,
    ]);
    const userPrompt = this.prompts.createLastAnswerUserPrompt
      .replace('{question}', state.question)
      .replace('{subtask_results}', JSON.stringify(subtaskResults));

    const messages: ChatCompletionMessageParam[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ];

    logger.info('Sending request to OpenAI...');
    const response = await this.client.chat.completions.create({
      model: this.settings.openaiModel,
      messages,
      temperature: 0,
      seed: 0,
    });
    logger.info('✅ Successfully received response from OpenAI.');

    logger.info('Final answer creation complete!');

    return { lastAnswer: response.choices[0]?.message.content ?? '' };
  }

  /**
   * サブタスクをサブグラフで実行し、結果を返す
   * @param state - メイングラフの現在の状態
   * @returns サブタスク実行結果の配列
   */
  private async executeSubgraph(state: AgentState) {
    const subgraph = this.createSubgraph();

    const result = await subgraph.invoke({
      question: state.question,
      plan: state.plan,
      subtask: state.plan[state.currentStep] ?? '',
      isCompleted: false,
      challengeCount: 0,
    });

    const subtaskResult: Subtask = {
      taskName: result.subtask,
      toolResults: result.toolResults,
      reflectionResults: result.reflectionResults,
      isCompleted: result.isCompleted,
      subtaskAnswer: result.subtaskAnswer,
      challengeCount: result.challengeCount,
    };

    return { subtaskResults: [subtaskResult] };
  }

  /**
   * 計画内の各サブタスクを並列実行するためのSendリストを生成する
   * @param state - メイングラフの現在の状態
   * @returns 各サブタスクに対応するSendオブジェクトの配列
   */
  private shouldContinueExecSubtasks(state: AgentState): Send[] {
    return state.plan.map(
      (_, idx) =>
        new Send('execute_subtasks', {
          question: state.question,
          plan: state.plan,
          currentStep: idx,
        }),
    );
  }

  /**
   * サブタスクの実行フローを継続するか終了するか判定する
   * @param state - サブグラフの現在の状態
   * @returns 継続する場合は'continue'、終了する場合は'end'
   */
  private shouldContinueExecSubtaskFlow(
    state: AgentSubGraphState,
  ): 'end' | 'continue' {
    if (state.isCompleted || state.challengeCount >= MAX_CHALLENGE_COUNT) {
      return 'end';
    }
    return 'continue';
  }

  /**
   * サブグラフを作成する
   */
  private createSubgraph() {
    // メソッドチェーンでノードとエッジを追加（型推論のため）
    return new StateGraph(AgentSubGraphStateAnnotation)
      .addNode('select_tools', (state) => this.selectTools(state))
      .addNode('execute_tools', (state) => this.executeTools(state))
      .addNode('create_subtask_answer', (state) =>
        this.createSubtaskAnswer(state),
      )
      .addNode('reflect_subtask', (state) => this.reflectSubtask(state))
      .addEdge(START, 'select_tools')
      .addEdge('select_tools', 'execute_tools')
      .addEdge('execute_tools', 'create_subtask_answer')
      .addEdge('create_subtask_answer', 'reflect_subtask')
      .addConditionalEdges(
        'reflect_subtask',
        (state) => this.shouldContinueExecSubtaskFlow(state),
        { continue: 'select_tools', end: END },
      )
      .compile();
  }

  /**
   * エージェントのメイングラフを作成する
   */
  createGraph() {
    // メソッドチェーンでノードとエッジを追加（型推論のため）
    return new StateGraph(AgentStateAnnotation)
      .addNode('create_plan', (state) => this.createPlan(state))
      .addNode('execute_subtasks', (state) => this.executeSubgraph(state))
      .addNode('create_answer', (state) => this.createAnswer(state))
      .addEdge(START, 'create_plan')
      .addConditionalEdges('create_plan', (state) =>
        this.shouldContinueExecSubtasks(state),
      )
      .addEdge('execute_subtasks', 'create_answer')
      .addEdge('create_answer', END)
      .compile();
  }

  /**
   * エージェントを実行する
   */
  async runAgent(question: string): Promise<AgentResult> {
    const app = this.createGraph();
    const result = await app.invoke({
      question,
      currentStep: 0,
    });

    return {
      question,
      plan: { subtasks: result.plan },
      subtasks: result.subtaskResults,
      answer: result.lastAnswer,
    };
  }
}
