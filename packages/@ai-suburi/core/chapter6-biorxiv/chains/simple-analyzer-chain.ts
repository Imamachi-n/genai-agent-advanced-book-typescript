import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';

import {
  type ReadingResult,
  type SimpleAnalysis,
  simpleAnalysisSchema,
} from '../models.js';
import { loadPrompt } from './utils.js';

/**
 * 簡易分析チェーン。タイトルとアブストラクトのみで関連度判定と回答生成を行う。
 * PDF のダウンロードやセクション分析を行わない軽量版。
 */
export class SimpleAnalyzer {
  private static readonly PROMPT = loadPrompt('simple_analyze');
  private llm: ChatOpenAI;

  constructor(llm: ChatOpenAI) {
    this.llm = llm;
  }

  async invoke(state: Record<string, unknown>): Promise<Record<string, unknown>> {
    const goal = (state.goal as string) ?? '';
    const readingResult = state.readingResult as ReadingResult;
    const paper = readingResult.paper;

    const prompt = ChatPromptTemplate.fromTemplate(SimpleAnalyzer.PROMPT);
    const chain = prompt.pipe(
      this.llm.withStructuredOutput(simpleAnalysisSchema),
    );

    const result: SimpleAnalysis = await chain.invoke({
      title: paper.title,
      authors: paper.authors.join(', '),
      abstract: paper.abstract,
      published: paper.published,
      goal,
      task: readingResult.task,
    });

    const updatedResult: ReadingResult = {
      ...readingResult,
      isRelated: result.is_related,
      answer: result.is_related ? result.answer : '',
    };

    return { readingResult: updatedResult };
  }
}
