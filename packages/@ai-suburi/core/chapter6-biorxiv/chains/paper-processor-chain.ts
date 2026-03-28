import { Command, Send } from '@langchain/langgraph';

import type { BiorxivPaper, ReadingResult } from '../models.js';
import type { Searcher } from '../searcher/searcher.js';
import { MarkdownStorage } from '../service/markdown-storage.js';
import { PdfToText } from '../service/pdf-to-text.js';

interface PaperProcessorInput {
  goal: string;
  tasks: string[];
}

export class PaperProcessor {
  private searcher: Searcher;
  private maxWorkers: number;
  private markdownStorage: MarkdownStorage;

  constructor(searcher: Searcher, maxWorkers: number = 3) {
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

  async convertPdfs(papers: BiorxivPaper[]): Promise<string[]> {
    // Promise.all で並列実行
    const promises = papers.map(async (paper) => {
      const converter = new PdfToText(paper.pdfLink);
      const text = await converter.convert();
      // DOIからファイル名を生成（スラッシュをアンダースコアに変換）
      const filename = `${paper.doi.replace(/\//g, '_')}.md`;
      return this.markdownStorage.write(filename, text);
    });

    return Promise.all(promises);
  }

  async run(state: PaperProcessorInput): Promise<ReadingResult[]> {
    let resultIndex = 0;
    const readingResults: ReadingResult[] = [];
    const uniquePapers = new Map<string, BiorxivPaper>();
    const taskPapers = new Map<string, string[]>();

    // タスクの処理
    for (const task of state.tasks) {
      const searchedPapers = await this.searcher.run(state.goal, task);
      taskPapers.set(
        task,
        searchedPapers.map((paper) => paper.doi),
      );
      for (const paper of searchedPapers) {
        uniquePapers.set(paper.doi, paper);
      }
    }

    // 重複排除後の論文リストに対してPDF変換を実行
    const uniquePapersList = Array.from(uniquePapers.values());
    const markdownPaths = await this.convertPdfs(uniquePapersList);

    // 各タスクに対して関連する論文を割り当て
    const uniqueKeys = Array.from(uniquePapers.keys());
    for (const task of state.tasks) {
      const dois = taskPapers.get(task) ?? [];
      for (const doi of dois) {
        const paper = uniquePapers.get(doi)!;
        const paperIndex = uniqueKeys.indexOf(doi);
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
