import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { Command } from '@langchain/langgraph';

import type { ReadingResult, Sufficiency } from '../models.js';
import { sufficiencySchema } from '../models.js';
import { MarkdownParser } from '../service/markdown-parser.js';
import { MarkdownStorage } from '../service/markdown-storage.js';
import { loadPrompt } from './utils.js';

export class SetSection {
  private static readonly PROMPT = loadPrompt('set_section');
  private llm: ChatOpenAI;
  private maxSections: number;
  private storage: MarkdownStorage;
  private parser: MarkdownParser;

  constructor(llm: ChatOpenAI, maxSections: number) {
    this.llm = llm;
    this.maxSections = maxSections;
    this.storage = new MarkdownStorage();
    this.parser = new MarkdownParser();
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const goal = (state.goal as string) ?? '';
    const readingResult = state.readingResult as ReadingResult;
    const paper = readingResult.paper;
    const selectedSectionIndices =
      (state.selectedSectionIndices as number[]) ?? [];
    const sufficiency = state.sufficiency as Sufficiency | undefined;

    const prompt = ChatPromptTemplate.fromTemplate(SetSection.PROMPT);
    const chain = prompt
      .pipe(this.llm)
      .pipe(new StringOutputParser());

    const sufficiencyCheckStr = sufficiency
      ? `十分性の判断結果: ${sufficiency.is_sufficient}\n十分性の判断理由: ${sufficiency.reason}\n`
      : '';

    const markdownText = this.storage.read(readingResult.markdownPath);
    const result = await chain.invoke({
      title: paper.title,
      authors: paper.authors.join(', '),
      abstract: paper.abstract,
      context: this.parser.getSectionsOverview(markdownText),
      goal,
      selected_section_indices: selectedSectionIndices.join(','),
      sufficiency_check: sufficiencyCheckStr,
      task: readingResult.task,
      max_sections: this.maxSections,
    });

    const sectionIndices = result
      .split(',')
      .map((s) => Number.parseInt(s.trim(), 10))
      .filter((n) => !Number.isNaN(n));

    return new Command({
      goto: 'check_sufficiency',
      update: { selectedSectionIndices: sectionIndices },
    });
  }
}

export class CheckSufficiency {
  private static readonly PROMPT = loadPrompt('check_sufficiency');
  private llm: ChatOpenAI;
  private checkCountLimit: number;
  private storage: MarkdownStorage;
  private parser: MarkdownParser;

  constructor(llm: ChatOpenAI, checkCount: number) {
    this.llm = llm;
    this.checkCountLimit = checkCount;
    this.storage = new MarkdownStorage();
    this.parser = new MarkdownParser();
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const goal = (state.goal as string) ?? '';
    const readingResult = state.readingResult as ReadingResult;
    const paper = readingResult.paper;
    const selectedSectionIndices =
      (state.selectedSectionIndices as number[]) ?? [];
    const checkCount = ((state.checkCount as number) ?? 0) + 1;

    const markdownText = this.storage.read(readingResult.markdownPath);

    const prompt = ChatPromptTemplate.fromTemplate(
      CheckSufficiency.PROMPT,
    );
    const chain = prompt.pipe(
      this.llm.withStructuredOutput(sufficiencySchema),
    );

    const sufficiency: Sufficiency = await chain.invoke({
      title: paper.title,
      authors: paper.authors.join(', '),
      abstract: paper.abstract,
      sections: this.parser.getSelectedSections(
        markdownText,
        selectedSectionIndices,
      ),
      goal,
      task: readingResult.task,
    });

    let nextNode: string;
    if (sufficiency.is_sufficient) {
      nextNode = 'summarize';
    } else if (checkCount >= this.checkCountLimit) {
      nextNode = 'mark_as_not_related';
    } else {
      nextNode = 'set_section';
    }

    return new Command({
      goto: nextNode,
      update: { sufficiency, checkCount },
    });
  }
}

export class Summarizer {
  private static readonly PROMPT = loadPrompt('summarize');
  private llm: ChatOpenAI;
  private storage: MarkdownStorage;
  private parser: MarkdownParser;

  constructor(llm: ChatOpenAI) {
    this.llm = llm;
    this.storage = new MarkdownStorage();
    this.parser = new MarkdownParser();
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const goal = (state.goal as string) ?? '';
    const selectedSectionIndices =
      (state.selectedSectionIndices as number[]) ?? [];
    const readingResult = state.readingResult as ReadingResult;
    const paper = readingResult.paper;
    const task = readingResult.task;

    const prompt = ChatPromptTemplate.fromTemplate(Summarizer.PROMPT);
    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
    const markdownText = this.storage.read(readingResult.markdownPath);

    const answer = await chain.invoke({
      title: paper.title,
      authors: paper.authors.join(', '),
      abstract: paper.abstract,
      context: this.parser.getSelectedSections(
        markdownText,
        selectedSectionIndices,
      ),
      goal,
      task,
    });

    const updatedResult: ReadingResult = {
      ...readingResult,
      answer,
      isRelated: true,
    };

    return new Command({
      update: { readingResult: updatedResult },
    });
  }
}
