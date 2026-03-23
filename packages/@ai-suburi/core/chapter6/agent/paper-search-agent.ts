import { Annotation, StateGraph } from '@langchain/langgraph';
import type { CompiledStateGraph } from '@langchain/langgraph';
import type { ChatOpenAI } from '@langchain/openai';

import { PaperProcessor } from '../chains/paper-processor-chain.js';
import { setupLogger } from '../custom-logger.js';
import type { ReadingResult } from '../models.js';
import type { ArxivSearcher } from '../searcher/arxiv-searcher.js';
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
});

type PaperSearchAgentState = typeof PaperSearchAgentAnnotation.State;

// --- Agent ---

export class PaperSearchAgent {
  readonly graph: CompiledStateGraph<any, any, any, any>;
  private recursionLimit: number;
  private paperAnalyzer: PaperAnalyzerAgent;

  constructor(
    llm: ChatOpenAI,
    searcher: ArxivSearcher,
    options: {
      recursionLimit?: number;
      maxWorkers?: number;
    } = {},
  ) {
    this.recursionLimit = options.recursionLimit ?? 1000;
    const maxWorkers = options.maxWorkers ?? 3;
    const paperProcessor = new PaperProcessor(searcher, maxWorkers);
    this.paperAnalyzer = new PaperAnalyzerAgent(llm);

    this.graph = this.createGraph(paperProcessor);
  }

  private createGraph(
    paperProcessor: PaperProcessor,
  ): CompiledStateGraph<any, any, any, any> {
    const workflow = new StateGraph(PaperSearchAgentAnnotation)
      .addNode('search_papers', (state) => {
        logger.info('|--> search_papers');
        return paperProcessor.invoke(state);
      }, { ends: ['analyze_paper'] })
      .addNode('analyze_paper', (state) => {
        logger.info('|--> analyze_paper');
        return this.analyzePaper(state);
      })
      .addNode('organize_results', (state: PaperSearchAgentState) => {
        logger.info('|--> organize_results');
        return this.organizeResults(state);
      })
      .addEdge('__start__', 'search_papers')
      .addEdge('analyze_paper', 'organize_results');

    return workflow.compile();
  }

  private async analyzePaper(
    state: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const output = await this.paperAnalyzer.graph.invoke(state, {
      recursionLimit: this.recursionLimit,
    });
    const readingResult = output.readingResult as ReadingResult | undefined;
    return {
      processingReadingResults: readingResult ? [readingResult] : [],
    };
  }

  private organizeResults(
    state: PaperSearchAgentState,
  ): Record<string, unknown> {
    const processingReadingResults =
      state.processingReadingResults ?? [];
    const readingResults = processingReadingResults.filter(
      (result) => result && result.isRelated === true,
    );
    return { readingResults };
  }
}
