import { StringOutputParser } from '@langchain/core/output_parsers';
import type { BaseMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { Command } from '@langchain/langgraph';

import { loadPrompt } from './utils.js';

export class GoalOptimizer {
  private llm: ChatOpenAI;
  private currentDate: string;

  constructor(llm: ChatOpenAI) {
    this.llm = llm;
    this.currentDate = new Date().toISOString().split('T')[0]!;
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const messages = (state.messages as BaseMessage[]) ?? [];
    const goal = await this.run({ messages });
    return new Command({
      goto: 'decompose_query',
      update: { goal },
    });
  }

  async run(params: {
    messages: BaseMessage[];
    mode?: 'conversation' | 'search';
    searchResults?: Record<string, string>[] | null;
    improvementHint?: string | null;
  }): Promise<string> {
    const {
      messages,
      mode = 'conversation',
      searchResults,
      improvementHint,
    } = params;

    const template =
      mode === 'search'
        ? loadPrompt('goal_optimizer_search')
        : loadPrompt('goal_optimizer_conversation');

    const prompt = ChatPromptTemplate.fromTemplate(template);
    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());

    const inputs: Record<string, string> = {
      current_date: this.currentDate,
      conversation_history: this.formatHistory(messages),
    };

    if (mode === 'search' && searchResults) {
      inputs.search_results = this.formatSearchResults(searchResults);
    }
    if (improvementHint) {
      inputs.improvement_hint = improvementHint;
    }

    return chain.invoke(inputs);
  }

  private formatHistory(messages: BaseMessage[]): string {
    return messages
      .map(
        (message) =>
          `${message.getType()}: ${typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}`,
      )
      .join('\n');
  }

  private formatSearchResults(
    results: Record<string, string>[],
  ): string {
    return results
      .map(
        (result) =>
          `Title: ${result.title ?? ''}\nAbstract: ${result.abstract ?? ''}`,
      )
      .join('\n\n');
  }
}
