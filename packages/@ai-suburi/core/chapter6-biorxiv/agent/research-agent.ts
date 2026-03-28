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
import { ChromaStore } from '../rag/chroma-store.js';
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
  // Output
  finalOutput: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => '',
  }),
});

// --- Agent ---

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
    const store = new ChromaStore({
      collectionName: s.chromaCollectionName,
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
      }, { ends: ['human_feedback', 'goal_setting'] })
      .addNode('human_feedback', (state: Record<string, unknown>) => {
        logger.info('|--> human_feedback');
        return this.humanFeedback(state);
      }, { ends: ['user_hearing'] })
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
      },
    });
  }
}

// --- ワークフロー実行 ---

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
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    const userInput = await new Promise<string>((resolve) => {
      rl.question('User Feedback: ', (answer) => {
        rl.close();
        resolve(answer);
      });
    });
    return invokeWorkflow(
      workflow,
      new Command({ resume: userInput }),
      config,
      options,
    );
  }
  return result;
}

// --- LangGraph Studio 用のグラフエクスポート ---

export const graph = new ResearchAgent().graph;

// --- エントリーポイント ---

async function main(): Promise<void> {
  const skipFeedback = process.argv.includes('--skip-feedback');
  const args = process.argv.filter((a) => !a.startsWith('--'));
  const userQuery = args[2] ?? 'single-cell RNA-seq解析の最新手法について調べる';
  const recursionLimit = Number(args[3] ?? '1000');

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
