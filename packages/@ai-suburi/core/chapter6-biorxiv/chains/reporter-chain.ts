import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { Command } from '@langchain/langgraph';

import { type ReadingResult, biorxivPaperToXml } from '../models.js';
import { loadPrompt } from './utils.js';

export class Reporter {
  private llm: ChatOpenAI;
  private currentDate: string;

  constructor(llm: ChatOpenAI) {
    this.llm = llm;
    this.currentDate = new Date().toISOString().split('T')[0]!;
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const results = (state.readingResults as ReadingResult[]) ?? [];
    const query = (state.goal as string) ?? '';

    const context = results
      .map((item) => {
        return `<item>
<id>${item.id}</id>
<task>${item.task}</task>
${biorxivPaperToXml(item.paper)}
<answer>${item.answer}</answer>
<is_related>${item.isRelated}</is_related>
</item>`;
      })
      .join('\n');

    const finalOutput = await this.run(context, query);
    return new Command({
      update: { finalOutput },
    });
  }

  async run(context: string, query: string): Promise<string> {
    const prompt = ChatPromptTemplate.fromMessages([
      ['system', loadPrompt('reporter_system')],
      ['user', loadPrompt('reporter_user')],
    ]);
    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
    return chain.invoke({
      current_date: this.currentDate,
      context,
      query,
    });
  }
}
