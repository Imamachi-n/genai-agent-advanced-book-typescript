import { Command, Send } from '@langchain/langgraph';

import type { ArxivPaper, ReadingResult } from '../models.js';
import type { ArxivSearcher } from '../searcher/arxiv-searcher.js';
import { MarkdownStorage } from '../service/markdown-storage.js';
import { PdfToMarkdown } from '../service/pdf-to-markdown.js';

interface PaperProcessorInput {
  goal: string;
  tasks: string[];
}

export class PaperProcessor {
  private searcher: ArxivSearcher;
  private maxWorkers: number;
  private markdownStorage: MarkdownStorage;

  constructor(searcher: ArxivSearcher, maxWorkers: number = 3) {
    this.searcher = searcher;
    this.maxWorkers = maxWorkers;
    this.markdownStorage = new MarkdownStorage();
  }

  async invoke(state: Record<string, unknown>): Promise<Command> {
    const input: PaperProcessorInput = {
      goal: (state.goal as string) ?? '',
      tasks: (state.tasks as string[]) ?? [],
    };

    const gotos: Send[] = [];
    const readingResults = await this.run(input);

    for (const readingResult of readingResults) {
      gotos.push(
        new Send('analyze_paper', {
          goal: input.goal,
          readingResult,
        }),
      );
    }

    return new Command({
      goto: gotos,
      update: { readingResults },
    });
  }

  async convertPdfs(papers: ArxivPaper[]): Promise<string[]> {
    // Promise.all で並列実行（Python の ThreadPoolExecutor 相当）
    const promises = papers.map(async (paper) => {
      const converter = new PdfToMarkdown(paper.pdfLink);
      const markdownText = await converter.convert();
      const filename = `${paper.id}.md`;
      return this.markdownStorage.write(filename, markdownText);
    });

    return Promise.all(promises);
  }

  async run(state: PaperProcessorInput): Promise<ReadingResult[]> {
    let resultIndex = 0;
    const readingResults: ReadingResult[] = [];
    const uniquePapers = new Map<string, ArxivPaper>();
    const taskPapers = new Map<string, string[]>();

    // タスクの処理
    for (const task of state.tasks) {
      const searchedPapers = await this.searcher.run(state.goal, task);
      taskPapers.set(
        task,
        searchedPapers.map((paper) => paper.pdfLink),
      );
      for (const paper of searchedPapers) {
        uniquePapers.set(paper.pdfLink, paper);
      }
    }

    // 重複排除後の論文リストに対してPDF変換を実行
    const uniquePapersList = Array.from(uniquePapers.values());
    const markdownPaths = await this.convertPdfs(uniquePapersList);

    // 各タスクに対して関連する論文を割り当て
    const uniqueKeys = Array.from(uniquePapers.keys());
    for (const task of state.tasks) {
      const pdfLinks = taskPapers.get(task) ?? [];
      for (const pdfLink of pdfLinks) {
        const paper = uniquePapers.get(pdfLink)!;
        const paperIndex = uniqueKeys.indexOf(pdfLink);
        readingResults.push({
          id: resultIndex,
          task,
          paper,
          markdownPath: markdownPaths[paperIndex]!,
          answer: '',
          isRelated: null,
        });
        resultIndex++;
      }
    }

    return readingResults;
  }
}
