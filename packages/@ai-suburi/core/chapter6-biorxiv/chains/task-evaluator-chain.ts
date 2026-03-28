import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { Command } from '@langchain/langgraph';

import {
  type ReadingResult,
  type TaskEvaluation,
  taskEvaluationSchema,
} from '../models.js';
import { setupLogger } from '../custom-logger.js';
import { dictToXmlStr } from './utils.js';
import { loadPrompt } from './utils.js';

const logger = setupLogger('task-evaluator');

export class TaskEvaluator {
  private llm: ChatOpenAI;
  private currentDate: string;
  private maxRetryCount: number;

  constructor(llm: ChatOpenAI, maxRetryCount: number = 3) {
    this.llm = llm;
    this.currentDate = new Date().toISOString().split('T')[0]!;
    this.maxRetryCount = maxRetryCount;
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

    logger.info(
      `retryCount=${currentRetryCount}/${this.maxRetryCount}, readingResults=${readingResults.length}件`,
    );

    // 読み取り結果が0件の場合はリトライせず即レポート生成へ
    if (readingResults.length === 0) {
      logger.info(
        '読み取り結果が0件のため、現状の情報でレポート生成に進みます',
      );
      return new Command({
        goto: 'generate_report',
        update: {
          retryCount: currentRetryCount,
          evaluation: {
            need_more_information: false,
            reason: '読み取り結果が0件のため、取得済みの情報でレポートを生成します',
            content: '',
          },
        },
      });
    }

    const evaluation = await this.run(context, goal);
    logger.info(
      `evaluation: need_more_information=${evaluation.need_more_information}, reason=${evaluation.reason}`,
    );

    if (evaluation.need_more_information) {
      currentRetryCount++;
    }

    // リトライ上限に達した場合は強制的にレポート生成に進む
    let nextNode: string;
    if (currentRetryCount >= this.maxRetryCount) {
      logger.info(
        `リトライ上限（${this.maxRetryCount}回）に達したため、レポート生成に進みます`,
      );
      nextNode = 'generate_report';
    } else {
      nextNode = evaluation.need_more_information
        ? 'decompose_query'
        : 'generate_report';
    }

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
