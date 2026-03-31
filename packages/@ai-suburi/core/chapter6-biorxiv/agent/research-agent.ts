import type { BaseMessage } from '@langchain/core/messages';
import { messagesStateReducer } from '@langchain/langgraph';
import {
  Annotation,
  Command,
  MemorySaver,
  StateGraph,
  interrupt,
} from '@langchain/langgraph';
import type {
  CompiledStateGraph,
  LangGraphRunnableConfig,
} from '@langchain/langgraph';
import type { ChatOpenAI } from '@langchain/openai';
import * as readline from 'node:readline';

import { HearingChain } from '../chains/hearing-chain.js';
import { GoalOptimizer } from '../chains/goal-optimizer-chain.js';
import { QueryDecomposer } from '../chains/query-decomposer-chain.js';
import { TaskEvaluator } from '../chains/task-evaluator-chain.js';
import { Reporter } from '../chains/reporter-chain.js';
import {
  type Settings,
  createFastLlm,
  createLlm,
  createReporterLlm,
  loadSettings,
} from '../configs.js';
import { setupLogger } from '../custom-logger.js';
import type { Hearing, ReadingResult, TaskEvaluation } from '../models.js';
import { QdrantStore } from '../rag/qdrant-store.js';
import { RagSearcher } from '../rag/rag-searcher.js';
import { PaperSearchAgent } from './paper-search-agent.js';

const logger = setupLogger('research-agent');

// --- State 定義 ---

const ResearchAgentAnnotation = Annotation.Root({
  // Input
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  // Private
  hearing: Annotation<Hearing | undefined>({
    reducer: (_prev, next) => next,
    default: () => undefined,
  }),
  goal: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => '',
  }),
  tasks: Annotation<string[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  readingResults: Annotation<ReadingResult[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  evaluation: Annotation<TaskEvaluation | undefined>({
    reducer: (_prev, next) => next,
    default: () => undefined,
  }),
  retryCount: Annotation<number>({
    reducer: (_prev, next) => next,
    default: () => 0,
  }),
  searchedPaperList: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => '',
  }),
  analysisMode: Annotation<'simple' | 'detailed'>({
    reducer: (_prev, next) => next,
    default: () => 'detailed',
  }),
  // Output
  finalOutput: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => '',
  }),
});

// --- Agent ---

/**
 * リサーチエージェント。ユーザーの質問からゴール設定・クエリ分解・論文検索・評価・レポート生成までを統括する。
 */
export class ResearchAgent {
  readonly graph: CompiledStateGraph<any, any, any, any>;
  private recursionLimit: number;
  private paperSearchAgent: PaperSearchAgent;

  constructor(
    llm?: ChatOpenAI,
    fastLlm?: ChatOpenAI,
    reporterLlm?: ChatOpenAI,
    settings?: Settings,
  ) {
    const s = settings ?? loadSettings();
    const _llm = llm ?? createLlm(s);
    const _fastLlm = fastLlm ?? createFastLlm(s);
    const _reporterLlm = reporterLlm ?? createReporterLlm(s);

    this.recursionLimit = s.maxRecursionLimit;

    const userHearing = new HearingChain(_llm);
    const goalSetting = new GoalOptimizer(_llm);
    const decomposeQuery = new QueryDecomposer(_llm, {
      minDecomposedTasks: s.minDecomposedTasks,
      maxDecomposedTasks: s.maxDecomposedTasks,
    });

    // RAG Searcher のセットアップ
    const store = new QdrantStore({
      collectionName: s.qdrantCollectionName,
      openaiApiKey: s.openaiApiKey,
      embeddingModel: s.embeddingModel,
    });
    const searcher = new RagSearcher(_fastLlm, store, {
      openaiApiKey: s.openaiApiKey,
      embeddingModel: s.embeddingModel,
      maxSearchResults: s.maxSearchResults,
      maxPapers: s.maxPapers,
      debug: s.debug,
    });

    this.paperSearchAgent = new PaperSearchAgent(_fastLlm, searcher, {
      recursionLimit: s.maxRecursionLimit,
      maxWorkers: s.maxWorkers,
    });
    const evaluateTask = new TaskEvaluator(_llm, s.maxEvaluationRetryCount);
    const generateReport = new Reporter(_reporterLlm);

    this.graph = this.createGraph(
      userHearing,
      goalSetting,
      decomposeQuery,
      evaluateTask,
      generateReport,
    );
  }

  /**
   * リサーチエージェントの LangGraph ワークフローを構築する。
   * @param userHearing - ユーザー意図のヒアリングチェーン
   * @param goalSetting - ゴール最適化チェーン
   * @param decomposeQuery - クエリ分解チェーン
   * @param evaluateTask - タスク評価チェーン
   * @param generateReport - レポート生成チェーン
   * @returns コンパイル済みの StateGraph
   */
  private createGraph(
    userHearing: HearingChain,
    goalSetting: GoalOptimizer,
    decomposeQuery: QueryDecomposer,
    evaluateTask: TaskEvaluator,
    generateReport: Reporter,
  ): CompiledStateGraph<any, any, any, any> {
    const checkpointer = new MemorySaver();

    const workflow = new StateGraph(ResearchAgentAnnotation)
      .addNode('user_hearing', (state) => {
        logger.info('|--> user_hearing');
        return userHearing.invoke(state);
      }, { ends: ['human_feedback', 'select_mode'] })
      .addNode('human_feedback', (state: Record<string, unknown>) => {
        logger.info('|--> human_feedback');
        return this.humanFeedback(state);
      }, { ends: ['user_hearing'] })
      .addNode('select_mode', () => {
        logger.info('|--> select_mode');
        return this.selectMode();
      }, { ends: ['goal_setting'] })
      .addNode('goal_setting', (state) => {
        logger.info('|--> goal_setting');
        return goalSetting.invoke(state);
      }, { ends: ['decompose_query'] })
      .addNode('decompose_query', (state) => {
        logger.info('|--> decompose_query');
        return decomposeQuery.invoke(state);
      }, { ends: ['paper_search_agent'] })
      .addNode('paper_search_agent', (state, config) => {
        logger.info('|--> paper_search_agent');
        return this.invokePaperSearchAgent(state, config);
      }, { ends: ['evaluate_task'] })
      .addNode('evaluate_task', (state) => {
        logger.info('|--> evaluate_task');
        return evaluateTask.invoke(state);
      }, { ends: ['decompose_query', 'generate_report'] })
      .addNode('generate_report', (state) => {
        logger.info('|--> generate_report');
        return generateReport.invoke(state);
      }, { ends: [] })
      .addEdge('__start__', 'user_hearing');

    return workflow.compile({ checkpointer });
  }

  /**
   * select_mode ノードの処理。interrupt で分析モードの選択をユーザーに求める。
   * @returns goal_setting への遷移コマンド（analysisMode を含む）
   */
  private selectMode(): Command {
    const selection = interrupt(
      '分析モードを選択してください:\n1. 簡易版（タイトル+アブストラクトのみ・高速）\n2. 詳細版（PDF全文分析・高精度）',
    );

    const mode: 'simple' | 'detailed' =
      selection === '1' || selection === 'simple' ? 'simple' : 'detailed';

    logger.info(`分析モード: ${mode}`);

    return new Command({
      goto: 'goal_setting',
      update: { analysisMode: mode },
    });
  }

  /**
   * human_feedback ノードの処理。interrupt でユーザー入力を受け取り、user_hearing へ遷移する。
   * @param state - 現在のワークフロー状態
   * @returns user_hearing への遷移コマンド
   */
  private humanFeedback(
    state: Record<string, unknown>,
  ): Command {
    const messages = (state.messages as BaseMessage[]) ?? [];
    const lastMessage = messages[messages.length - 1];
    const content =
      lastMessage && typeof lastMessage.content === 'string'
        ? lastMessage.content
        : '';
    const humanFeedback = interrupt(content);
    const feedbackText =
      typeof humanFeedback === 'string' && humanFeedback
        ? humanFeedback
        : 'そのままの条件で検索し、調査してください。';

    return new Command({
      goto: 'user_hearing',
      update: {
        messages: [{ role: 'human', content: feedbackText }],
      },
    });
  }

  /**
   * PaperSearchAgent サブグラフを実行し、結果を evaluate_task へ渡す。
   * @param state - 現在のワークフロー状態
   * @param config - LangGraph 実行設定
   * @returns evaluate_task への遷移コマンド（readingResults と searchedPaperList を含む）
   */
  private async invokePaperSearchAgent(
    state: Record<string, unknown>,
    config: LangGraphRunnableConfig,
  ): Promise<Command> {
    const output = await this.paperSearchAgent.graph.invoke(state, {
      ...config,
      recursionLimit: this.recursionLimit,
    });
    return new Command({
      goto: 'evaluate_task',
      update: {
        readingResults: (output.readingResults as ReadingResult[]) ?? [],
        searchedPaperList: (output.searchedPaperList as string) ?? '',
      },
    });
  }
}

// --- ワークフロー実行 ---

/**
 * LangGraph ワークフローを実行し、human_feedback による中断時はユーザー入力を受け付けて再開する。
 * @param workflow - コンパイル済みの StateGraph
 * @param inputData - ワークフローへの入力データまたは再開用 Command
 * @param config - 実行設定（thread_id や recursionLimit など）
 * @param options - human_feedback スキップなどのオプション
 * @returns ワークフローの最終出力
 */
export async function invokeWorkflow(
  workflow: CompiledStateGraph<any, any, any, any>,
  inputData: Record<string, unknown> | Command,
  config: Record<string, unknown>,
  options: { skipFeedback?: boolean } = {},
): Promise<Record<string, unknown>> {
  const result = await workflow.invoke(inputData, config);

  // human_feedback ノードで中断された場合の処理
  const hearing = result.hearing as { is_need_human_feedback?: boolean } | undefined;
  if (hearing?.is_need_human_feedback) {
    // スキップモード: 自動的にデフォルト応答で続行
    if (options.skipFeedback) {
      logger.info('human_feedback をスキップ（自動続行）');
      return invokeWorkflow(
        workflow,
        new Command({ resume: '' }),
        config,
        options,
      );
    }

    // 対話モード: ユーザー入力を受け付ける
    const userInput = await promptUser('User Feedback: ');
    return invokeWorkflow(
      workflow,
      new Command({ resume: userInput }),
      config,
      options,
    );
  }

  // select_mode ノードで中断された場合の処理（finalOutput がまだない = ワークフロー未完了）
  if (result.finalOutput === '' || result.finalOutput == null) {
    if (options.skipFeedback) {
      logger.info('select_mode をスキップ（デフォルト: detailed）');
      return invokeWorkflow(
        workflow,
        new Command({ resume: '2' }),
        config,
        options,
      );
    }

    // 対話モード: モード選択をユーザーに求める
    const modeInput = await promptUser('モード選択 (1: 簡易版 / 2: 詳細版): ');
    return invokeWorkflow(
      workflow,
      new Command({ resume: modeInput }),
      config,
      options,
    );
  }

  return result;
}

/**
 * CLI でユーザー入力を受け付けるヘルパー関数。
 * @param question - 表示するプロンプト文字列
 * @returns ユーザーの入力文字列
 */
async function promptUser(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise<string>((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

/** --- LangGraph Studio 用のグラフエクスポート --- */
export const graph = new ResearchAgent().graph;

// --- エントリーポイント ---

/**
 * CLI エントリーポイント。ユーザークエリを受け取りリサーチエージェントを実行する。
 */
async function main(): Promise<void> {
  const skipFeedback = process.argv.includes('--skip-feedback');
  const args = process.argv.filter((a) => !a.startsWith('--'));
  const userQuery = args[2] ?? 'single-cell RNA-seq解析の最新手法について調べる';
  const recursionLimit = Number(args[3] ?? 5);

  const agent = new ResearchAgent();
  const result = await invokeWorkflow(
    agent.graph,
    {
      messages: [{ role: 'user', content: userQuery }],
    },
    {
      configurable: { thread_id: 'biorxiv-research-001' },
      recursionLimit,
    },
    { skipFeedback },
  );

  console.log(result.finalOutput);
}

main();
