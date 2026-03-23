import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { Command } from '@langchain/langgraph';

import {
  type ReadingResult,
  type TaskEvaluation,
  taskEvaluationSchema,
} from '../models.js';
import { dictToXmlStr } from './utils.js';
import { loadPrompt } from './utils.js';

export class TaskEvaluator {
  private llm: ChatOpenAI;
  private currentDate: string;

  constructor(llm: ChatOpenAI) {
    this.llm = llm;
    this.currentDate = new Date().toISOString().split('T')[0]!;
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    let currentRetryCount = (state.retryCount as number) ?? 0;
    const readingResults = (state.readingResults as ReadingResult[]) ?? [];
    const goal = (state.goal as string) ?? '';

    const context = readingResults
      .map((item) => {
        const { markdownPath: _mp, ...rest } = item;
        return dictToXmlStr(rest as unknown as Record<string, unknown>);
      })
      .join('\n');

    const evaluation = await this.run(context, goal);

    if (evaluation.need_more_information) {
      currentRetryCount++;
    }

    const nextNode = evaluation.need_more_information
      ? 'decompose_query'
      : 'generate_report';

    return new Command({
      goto: nextNode,
      update: {
        retryCount: currentRetryCount,
        evaluation,
      },
    });
  }

  async run(context: string, goalSetting: string): Promise<TaskEvaluation> {
    const prompt = ChatPromptTemplate.fromTemplate(
      loadPrompt('task_evaluator'),
    );
    const chain = prompt.pipe(
      this.llm.withStructuredOutput(taskEvaluationSchema),
    );
    return chain.invoke({
      current_date: this.currentDate,
      context,
      goal_setting: goalSetting,
    });
  }
}
