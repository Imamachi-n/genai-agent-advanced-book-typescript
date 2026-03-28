import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { Command } from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';

import { type Hearing, hearingSchema } from '../models.js';
import { loadPrompt } from './utils.js';

export class HearingChain {
  private llm: ChatOpenAI;
  private currentDate: string;

  constructor(llm: ChatOpenAI) {
    this.llm = llm;
    this.currentDate = new Date().toISOString().split('T')[0]!;
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const messages = (state.messages as BaseMessage[]) ?? [];
    const hearing = await this.run(messages);
    const message: Record<string, string>[] = [];

    if (hearing.is_need_human_feedback) {
      message.push({
        role: 'assistant',
        content: hearing.additional_question,
      });
    }

    const nextNode = hearing.is_need_human_feedback
      ? 'human_feedback'
      : 'goal_setting';

    return new Command({
      goto: nextNode,
      update: { hearing, messages: message },
    });
  }

  async run(messages: BaseMessage[]): Promise<Hearing> {
    const prompt = ChatPromptTemplate.fromTemplate(loadPrompt('hearing'));
    const chain = prompt.pipe(
      this.llm.withStructuredOutput(hearingSchema),
    );
    const hearing = await chain.invoke({
      current_date: this.currentDate,
      conversation_history: this.formatHistory(messages),
    });
    return hearing;
  }

  private formatHistory(messages: BaseMessage[]): string {
    return messages
      .map(
        (message) =>
          `${message.getType()}: ${typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}`,
      )
      .join('\n');
  }
}
