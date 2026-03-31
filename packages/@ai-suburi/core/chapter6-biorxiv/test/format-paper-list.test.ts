import { describe, expect, it } from 'vitest';
import type { BiorxivPaper, ReadingResult } from '../models.js';
import { formatPaperList } from '../models.js';

// --- テストヘルパー ---

function createPaper(overrides: Partial<BiorxivPaper> = {}): BiorxivPaper {
  return {
    doi: '10.1101/2025.01.01.000001',
    title: 'Sample Paper Title',
    link: 'https://www.biorxiv.org/content/10.1101/2025.01.01.000001',
    pdfLink: 'https://www.biorxiv.org/content/10.1101/2025.01.01.000001v1.full.pdf',
    abstract: 'This is a sample abstract.',
    published: '2025-01-01',
    authors: ['Alice', 'Bob'],
    category: 'bioinformatics',
    version: 1,
    relevanceScore: null,
    ...overrides,
  };
}

function createResult(overrides: Partial<ReadingResult> = {}): ReadingResult {
  return {
    id: 0,
    task: 'デフォルトタスク',
    paper: createPaper(),
    markdownPath: 'storage/markdown/sample.md',
    answer: '',
    isRelated: null,
    ...overrides,
  };
}

// --- テスト ---

describe('formatPaperList', () => {
  it('空配列の場合、ヒットなしメッセージを返す', () => {
    const result = formatPaperList([]);
    expect(result).toBe('検索にヒットした論文はありませんでした。');
  });

  it('1件の論文をURLリンク付きでフォーマットする', () => {
    const results: ReadingResult[] = [
      createResult({
        task: 'scRNA-seq clustering methods',
        paper: createPaper({
          title: 'Novel Clustering for scRNA-seq',
          link: 'https://www.biorxiv.org/content/10.1101/2025.03.01.999999',
          doi: '10.1101/2025.03.01.999999',
          authors: ['Taro Yamada', 'Hanako Suzuki'],
          published: '2025-03-01',
          relevanceScore: 0.85,
        }),
      }),
    ];

    const output = formatPaperList(results);

    expect(output).toContain('## 検索ヒット論文一覧');
    expect(output).toContain('### タスク: scRNA-seq clustering methods');
    expect(output).toContain('[Novel Clustering for scRNA-seq](https://www.biorxiv.org/content/10.1101/2025.03.01.999999)');
    expect(output).toContain('(関連度: 0.85)');
    expect(output).toContain('DOI: 10.1101/2025.03.01.999999');
    expect(output).toContain('著者: Taro Yamada, Hanako Suzuki');
    expect(output).toContain('公開日: 2025-03-01');
  });

  it('relevanceScore が null の場合、関連度を表示しない', () => {
    const results: ReadingResult[] = [
      createResult({
        paper: createPaper({ relevanceScore: null }),
      }),
    ];

    const output = formatPaperList(results);
    expect(output).not.toContain('関連度');
  });

  it('複数タスクの論文をタスクごとにグルーピングする', () => {
    const results: ReadingResult[] = [
      createResult({
        id: 0,
        task: 'タスクA',
        paper: createPaper({ doi: 'doi-a1', title: 'Paper A1' }),
      }),
      createResult({
        id: 1,
        task: 'タスクB',
        paper: createPaper({ doi: 'doi-b1', title: 'Paper B1' }),
      }),
      createResult({
        id: 2,
        task: 'タスクA',
        paper: createPaper({ doi: 'doi-a2', title: 'Paper A2' }),
      }),
    ];

    const output = formatPaperList(results);

    // タスクAが先に出現し、その中にPaper A1とA2がある
    const taskAIndex = output.indexOf('### タスク: タスクA');
    const taskBIndex = output.indexOf('### タスク: タスクB');
    expect(taskAIndex).toBeLessThan(taskBIndex);

    expect(output).toContain('Paper A1');
    expect(output).toContain('Paper A2');
    expect(output).toContain('Paper B1');
  });

  it('同一タスク内で DOI が重複する論文は1件だけ表示する', () => {
    const sameDoi = '10.1101/duplicate';
    const results: ReadingResult[] = [
      createResult({
        id: 0,
        task: 'タスクX',
        paper: createPaper({ doi: sameDoi, title: 'Duplicate Paper' }),
      }),
      createResult({
        id: 1,
        task: 'タスクX',
        paper: createPaper({ doi: sameDoi, title: 'Duplicate Paper' }),
      }),
    ];

    const output = formatPaperList(results);

    // "Duplicate Paper" のマッチが1回だけ
    const matches = output.match(/Duplicate Paper/g);
    expect(matches).toHaveLength(1);
  });

  it('isRelated の値に関係なく全論文をリストアップする', () => {
    const results: ReadingResult[] = [
      createResult({
        id: 0,
        paper: createPaper({ doi: 'doi-related', title: 'Related Paper' }),
        isRelated: true,
      }),
      createResult({
        id: 1,
        paper: createPaper({ doi: 'doi-not-related', title: 'Not Related Paper' }),
        isRelated: false,
      }),
      createResult({
        id: 2,
        paper: createPaper({ doi: 'doi-null', title: 'Null Paper' }),
        isRelated: null,
      }),
    ];

    const output = formatPaperList(results);

    expect(output).toContain('Related Paper');
    expect(output).toContain('Not Related Paper');
    expect(output).toContain('Null Paper');
  });
});
