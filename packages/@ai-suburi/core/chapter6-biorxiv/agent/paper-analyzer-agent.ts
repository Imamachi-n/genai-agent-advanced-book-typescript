import { Annotation, Command, StateGraph } from '@langchain/langgraph';
import type { CompiledStateGraph } from '@langchain/langgraph';
import type { ChatOpenAI } from '@langchain/openai';

import {
  CheckSufficiency,
  SetSection,
  Summarizer,
} from '../chains/reading-chains.js';
import { setupLogger } from '../custom-logger.js';
import type { ReadingResult, Sufficiency } from '../models.js';

const logger = setupLogger('paper-analyzer-agent');

// --- State 定義 ---

const PaperAnalyzerAgentAnnotation = Annotation.Root({
  // Input
  goal: Annotation<string>,
  readingResult: Annotation<ReadingResult>,
  // Processing
  selectedSectionIndices: Annotation<number[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),
  sufficiency: Annotation<Sufficiency | undefined>({
    reducer: (_prev, next) => next,
    default: () => undefined,
  }),
  checkCount: Annotation<number>({
    reducer: (_prev, next) => next,
    default: () => 0,
  }),
});

type PaperAnalyzerAgentState = typeof PaperAnalyzerAgentAnnotation.State;

// --- Agent ---

export class PaperAnalyzerAgent {
  static readonly MAX_SECTIONS = 5;
  static readonly CHECK_COUNT = 3;

  readonly graph: CompiledStateGraph<any, any, any, any>;

  constructor(llm: ChatOpenAI) {
    const setSection = new SetSection(llm, PaperAnalyzerAgent.MAX_SECTIONS);
    const checkSufficiency = new CheckSufficiency(
      llm,
      PaperAnalyzerAgent.CHECK_COUNT,
    );
    const summarizer = new Summarizer(llm);

    this.graph = this.createGraph(setSection, checkSufficiency, summarizer);
  }

  private createGraph(
    setSection: SetSection,
    checkSufficiency: CheckSufficiency,
    summarizer: Summarizer,
  ): CompiledStateGraph<any, any, any, any> {
    const workflow = new StateGraph(PaperAnalyzerAgentAnnotation)
      .addNode('set_section', (state) => {
        logger.info('|--> set_section');
        return setSection.invoke(state);
      }, { ends: ['check_sufficiency'] })
      .addNode('check_sufficiency', (state) => {
        logger.info('|--> check_sufficiency');
        return checkSufficiency.invoke(state);
      }, { ends: ['set_section', 'summarize', 'mark_as_not_related'] })
      .addNode('mark_as_not_related', (state: PaperAnalyzerAgentState) => {
        logger.info('|--> mark_as_not_related');
        return this.markAsNotRelated(state);
      }, { ends: [] })
      .addNode('summarize', (state) => {
        logger.info('|--> summarize');
        return summarizer.invoke(state);
      }, { ends: [] })
      .addEdge('__start__', 'set_section');

    return workflow.compile();
  }

  private markAsNotRelated(state: PaperAnalyzerAgentState): Command {
    const readingResult = state.readingResult;
    if (!readingResult) {
      throw new Error('readingResult is not set');
    }
    const updatedResult: ReadingResult = {
      ...readingResult,
      isRelated: false,
    };
    return new Command({
      update: { readingResult: updatedResult },
    });
  }
}
