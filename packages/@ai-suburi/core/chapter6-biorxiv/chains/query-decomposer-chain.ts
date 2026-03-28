import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { Command } from '@langchain/langgraph';

import {
  type DecomposedTasks,
  type TaskEvaluation,
  decomposedTasksSchema,
} from '../models.js';
import { loadPrompt } from './utils.js';

export class QueryDecomposer {
  private llm: ChatOpenAI;
  private currentDate: string;
  private minDecomposedTasks: number;
  private maxDecomposedTasks: number;

  constructor(
    llm: ChatOpenAI,
    options: {
      minDecomposedTasks?: number;
      maxDecomposedTasks?: number;
    } = {},
  ) {
    this.llm = llm;
    this.currentDate = new Date().toISOString().split('T')[0]!;
    this.minDecomposedTasks = options.minDecomposedTasks ?? 3;
    this.maxDecomposedTasks = options.maxDecomposedTasks ?? 5;
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const evaluation = state.evaluation as TaskEvaluation | undefined;
    const content = evaluation?.content ?? (state.goal as string) ?? '';
    const decomposedTasks = await this.run(content);

    return new Command({
      goto: 'paper_search_agent',
      update: { tasks: decomposedTasks.tasks },
    });
  }

  async run(query: string): Promise<DecomposedTasks> {
    const prompt = ChatPromptTemplate.fromTemplate(
      loadPrompt('query_decomposer'),
    );
    const chain = prompt.pipe(
      this.llm.withStructuredOutput(decomposedTasksSchema),
    );
    return chain.invoke({
      min_decomposed_tasks: this.minDecomposedTasks,
      max_decomposed_tasks: this.maxDecomposedTasks,
      current_date: this.currentDate,
      query,
    });
  }
}
