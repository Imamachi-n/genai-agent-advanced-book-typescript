import { Annotation, StateGraph } from '@langchain/langgraph';
import type {
  CompiledStateGraph,
  LangGraphRunnableConfig,
} from '@langchain/langgraph';
import type { ChatOpenAI } from '@langchain/openai';

import { PaperProcessor } from '../chains/paper-processor-chain.js';
import { SimpleAnalyzer } from '../chains/simple-analyzer-chain.js';
import { setupLogger } from '../custom-logger.js';
import { type ReadingResult, formatPaperList } from '../models.js';
import type { Searcher } from '../searcher/searcher.js';
import { PaperAnalyzerAgent } from './paper-analyzer-agent.js';

const logger = setupLogger('paper-search-agent');

// --- State 定義 ---

const PaperSearchAgentAnnotation = Annotation.Root({
  // Input
  goal: Annotation<string>,
  tasks: Annotation<string[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  analysisMode: Annotation<'simple' | 'detailed'>({
    reducer: (_prev, next) => next,
    default: () => 'detailed',
  }),
  // Processing (operator.add 相当: 追加蓄積)
  processingReadingResults: Annotation<ReadingResult[]>({
    reducer: (prev, next) => [...prev, ...next],
    default: () => [],
  }),
  // Output
  readingResults: Annotation<ReadingResult[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  searchedPaperList: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => '',
  }),
});

type PaperSearchAgentState = typeof PaperSearchAgentAnnotation.State;

// --- Agent ---

/**
 * 論文検索エージェント。RAG 検索で論文を取得し、各論文を並列に分析して関連論文をフィルタリングする。
 */
export class PaperSearchAgent {
  readonly graph: CompiledStateGraph<any, any, any, any>;
  private recursionLimit: number;
  private paperAnalyzer: PaperAnalyzerAgent;
  private simpleAnalyzer: SimpleAnalyzer;

  constructor(
    llm: ChatOpenAI,
    searcher: Searcher,
    options: {
      recursionLimit?: number;
      maxWorkers?: number;
    } = {},
  ) {
    this.recursionLimit = options.recursionLimit ?? 1000;
    const maxWorkers = options.maxWorkers ?? 3;
    const paperProcessor = new PaperProcessor(searcher, maxWorkers);
    this.paperAnalyzer = new PaperAnalyzerAgent(llm);
    this.simpleAnalyzer = new SimpleAnalyzer(llm);

    this.graph = this.createGraph(paperProcessor);
  }

  /**
   * 論文検索エージェントの LangGraph ワークフローを構築する。
   * @param paperProcessor - RAG 検索と PDF 変換を行うプロセッサ
   * @returns コンパイル済みの StateGraph
   */
  private createGraph(
    paperProcessor: PaperProcessor,
  ): CompiledStateGraph<any, any, any, any> {
    const workflow = new StateGraph(PaperSearchAgentAnnotation)
      .addNode('search_papers', (state) => {
        logger.info('|--> search_papers');
        return paperProcessor.invoke(state);
      }, { ends: ['analyze_paper', 'simple_analyze_paper'] })
      .addNode('analyze_paper', (state, config) => {
        logger.info('|--> analyze_paper');
        return this.analyzePaper(state, config);
      })
      .addNode('simple_analyze_paper', async (state) => {
        logger.info('|--> simple_analyze_paper');
        return this.simpleAnalyzePaper(state);
      })
      .addNode('organize_results', (state: PaperSearchAgentState) => {
        logger.info('|--> organize_results');
        return this.organizeResults(state);
      })
      .addEdge('__start__', 'search_papers')
      .addEdge('analyze_paper', 'organize_results')
      .addEdge('simple_analyze_paper', 'organize_results');

    return workflow.compile();
  }

  /**
   * PaperAnalyzerAgent サブグラフを実行し、個別論文の分析結果を返す。
   * @param state - analyze_paper ノードの入力状態
   * @param config - LangGraph 実行設定
   * @returns processingReadingResults に追加する分析結果
   */
  private async analyzePaper(
    state: Record<string, unknown>,
    config: LangGraphRunnableConfig,
  ): Promise<Record<string, unknown>> {
    const output = await this.paperAnalyzer.graph.invoke(state, {
      ...config,
      recursionLimit: this.recursionLimit,
    });
    const readingResult = output.readingResult as ReadingResult | undefined;
    return {
      processingReadingResults: readingResult ? [readingResult] : [],
    };
  }

  /**
   * SimpleAnalyzer で論文を簡易分析し、結果を processingReadingResults に追加する。
   * @param state - simple_analyze_paper ノードの入力状態
   * @returns processingReadingResults に追加する分析結果
   */
  private async simpleAnalyzePaper(
    state: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const output = await this.simpleAnalyzer.invoke(state);
    const readingResult = output.readingResult as ReadingResult | undefined;
    return {
      processingReadingResults: readingResult ? [readingResult] : [],
    };
  }

  /**
   * 分析済みの論文結果を整理し、関連論文のみをフィルタリングしてURLリンク付きリストを生成する。
   * @param state - organize_results ノードの入力状態
   * @returns readingResults（関連論文のみ）と searchedPaperList（全論文のリスト）
   */
  private organizeResults(
    state: PaperSearchAgentState,
  ): Record<string, unknown> {
    const processingReadingResults =
      state.processingReadingResults ?? [];

    // 検索でヒットした全論文をURLリンク付きでリストアップ
    const searchedPaperList = formatPaperList(processingReadingResults);
    logger.info(`\n${searchedPaperList}`);

    const readingResults = processingReadingResults.filter(
      (result) => result && result.isRelated === true,
    );

    const totalCount = processingReadingResults.length;
    const relatedCount = readingResults.length;
    logger.info(
      `論文フィルタリング: 全${totalCount}件 → 関連あり${relatedCount}件（除外${totalCount - relatedCount}件）`,
    );

    return { readingResults, searchedPaperList };
  }
}
