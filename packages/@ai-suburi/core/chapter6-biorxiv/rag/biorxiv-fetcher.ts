import * as fs from 'node:fs';
import * as path from 'node:path';
import { loadSettings } from '../configs.js';
import { setupLogger } from '../custom-logger.js';
import type { BiorxivPaper } from '../models.js';

const logger = setupLogger('biorxiv-fetcher');

const DEFAULT_OUTPUT_DIR = 'storage/biorxiv-tmp';
const MAX_RETRIES = 5;
const INITIAL_BACKOFF_MS = 2000;

interface BiorxivApiResponse {
  messages: { status: string; count: number; total: number }[];
  collection: BiorxivApiEntry[];
}

interface BiorxivApiEntry {
  doi: string;
  title: string;
  authors: string;
  author_corresponding: string;
  author_corresponding_institution: string;
  date: string;
  version: string;
  type: string;
  license: string;
  category: string;
  jatsxml: string;
  abstract: string;
  published: string;
  server: string;
}

/** レジューム用のプログレス情報 */
interface FetchProgress {
  startDate: string;
  endDate: string;
  category: string;
  cursor: number;
  totalFetched: number;
  outputPath: string;
}

function apiEntryToPaper(entry: BiorxivApiEntry): BiorxivPaper {
  const doi = entry.doi;
  const version = Number.parseInt(entry.version, 10) || 1;
  return {
    doi,
    title: entry.title,
    link: `https://doi.org/${doi}`,
    pdfLink: `https://www.biorxiv.org/content/${doi}v${version}.full.pdf`,
    abstract: entry.abstract,
    published: entry.date,
    authors: entry.authors.split('; ').map((a) => a.trim()),
    category: entry.category,
    version,
    relevanceScore: null,
  };
}

/**
 * エクスポネンシャルバックオフ付きリトライで bioRxiv API を叩く。
 * 429（レート制限）や 5xx（サーバーエラー）の場合に自動リトライする。
 */
async function fetchBiorxivPage(
  startDate: string,
  endDate: string,
  cursor: number,
  category?: string,
): Promise<BiorxivApiResponse> {
  let url = `https://api.biorxiv.org/details/biorxiv/${startDate}/${endDate}/${cursor}`;
  if (category) {
    url += `?category=${encodeURIComponent(category)}`;
  }

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    logger.info(
      `Fetching: ${url}${attempt > 0 ? ` (retry ${attempt}/${MAX_RETRIES})` : ''}`,
    );

    try {
      const response = await fetch(url);

      if (response.ok) {
        return response.json() as Promise<BiorxivApiResponse>;
      }

      // リトライ対象: 429 (Rate Limit) or 5xx (Server Error)
      if (response.status === 429 || response.status >= 500) {
        const backoff = INITIAL_BACKOFF_MS * 2 ** attempt;
        logger.warn(
          `API returned ${response.status}. Retrying in ${backoff / 1000}s...`,
        );
        await new Promise((resolve) => setTimeout(resolve, backoff));
        continue;
      }

      // それ以外のエラー（4xx等）はリトライしない
      throw new Error(
        `bioRxiv API error: ${response.status} ${response.statusText}`,
      );
    } catch (error) {
      // ネットワークエラー（DNS, timeout 等）もリトライ
      if (error instanceof TypeError && attempt < MAX_RETRIES - 1) {
        const backoff = INITIAL_BACKOFF_MS * 2 ** attempt;
        logger.warn(
          `Network error: ${(error as Error).message}. Retrying in ${backoff / 1000}s...`,
        );
        await new Promise((resolve) => setTimeout(resolve, backoff));
        continue;
      }
      throw error;
    }
  }

  throw new Error(`Max retries (${MAX_RETRIES}) exceeded for ${url}`);
}

function getProgressPath(
  outputDir: string,
  startDate: string,
  endDate: string,
): string {
  return path.join(outputDir, `.progress_${startDate}_${endDate}.json`);
}

function saveProgress(progress: FetchProgress, outputDir: string): void {
  const progressPath = getProgressPath(
    outputDir,
    progress.startDate,
    progress.endDate,
  );
  fs.writeFileSync(progressPath, JSON.stringify(progress, null, 2), 'utf-8');
}

function loadProgress(
  outputDir: string,
  startDate: string,
  endDate: string,
): FetchProgress | null {
  const progressPath = getProgressPath(outputDir, startDate, endDate);
  if (!fs.existsSync(progressPath)) return null;
  const raw = fs.readFileSync(progressPath, 'utf-8');
  return JSON.parse(raw) as FetchProgress;
}

function deleteProgress(
  outputDir: string,
  startDate: string,
  endDate: string,
): void {
  const progressPath = getProgressPath(outputDir, startDate, endDate);
  if (fs.existsSync(progressPath)) {
    fs.unlinkSync(progressPath);
  }
}

/**
 * bioRxiv API から論文メタデータを取得し、JSONL ファイルとして逐次保存する。
 * Chroma への投入は行わない。
 *
 * JSONL 形式: 1行に1つの BiorxivPaper JSON オブジェクト。
 * ページ取得ごとにファイルに追記するため、大量データでもメモリを圧迫しない。
 *
 * - リトライ: 429/5xx/ネットワークエラー時にエクスポネンシャルバックオフで自動リトライ
 * - レジューム: プログレスファイルでカーソル位置を記録。中断後 --resume で再開可能
 * - 追記: --append で既存 JSONL ファイルに別の日付範囲のデータを追加取得可能
 */
export async function fetchBiorxivPapers(options: {
  startDate: string;
  endDate: string;
  outputDir?: string;
  appendTo?: string;
  category?: string;
  batchSize?: number;
  delayMs?: number;
  resume?: boolean;
}): Promise<{ outputPath: string; totalFetched: number }> {
  const settings = loadSettings();
  const category = options.category ?? settings.biorxivCategory;
  const batchSize = options.batchSize ?? settings.ingestionBatchSize;
  const delayMs = options.delayMs ?? 1000;
  const outputDir = options.outputDir ?? DEFAULT_OUTPUT_DIR;

  fs.mkdirSync(outputDir, { recursive: true });

  let cursor = 0;
  let totalFetched = 0;
  let outputPath: string;

  // レジューム: 前回の中断地点から再開
  if (options.resume) {
    const progress = loadProgress(
      outputDir,
      options.startDate,
      options.endDate,
    );
    if (progress) {
      cursor = progress.cursor;
      totalFetched = progress.totalFetched;
      outputPath = progress.outputPath;
      logger.info(
        `Resuming from cursor=${cursor}, totalFetched=${totalFetched}, file=${outputPath}`,
      );
    } else {
      logger.info('No progress file found. Starting fresh.');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const fileName = `biorxiv_${options.startDate}_${options.endDate}_${timestamp}.jsonl`;
      outputPath = path.join(outputDir, fileName);
    }
  } else if (options.appendTo) {
    // 追記モード: 既存ファイルに追加
    if (!fs.existsSync(options.appendTo)) {
      throw new Error(`Append target not found: ${options.appendTo}`);
    }
    outputPath = options.appendTo;
    logger.info(`Appending to existing file: ${outputPath}`);
  } else {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const fileName = `biorxiv_${options.startDate}_${options.endDate}_${timestamp}.jsonl`;
    outputPath = path.join(outputDir, fileName);
  }

  // JSONL ファイルに追記モードで開く（レジューム時は既存ファイルに追記）
  const writeStream = fs.createWriteStream(outputPath, {
    flags: 'a',
    encoding: 'utf-8',
  });

  try {
    while (true) {
      const response = await fetchBiorxivPage(
        options.startDate,
        options.endDate,
        cursor,
        category,
      );

      const entries = response.collection ?? [];
      if (entries.length === 0) {
        logger.info('No more entries from API.');
        break;
      }

      const papers = entries.map(apiEntryToPaper);

      // JSONL: 1行1オブジェクトでファイルに追記
      for (const paper of papers) {
        writeStream.write(`${JSON.stringify(paper)}\n`);
      }
      totalFetched += papers.length;

      // 次のカーソル位置を計算してプログレス保存
      cursor += batchSize;
      saveProgress(
        {
          startDate: options.startDate,
          endDate: options.endDate,
          category,
          cursor,
          totalFetched,
          outputPath,
        },
        outputDir,
      );

      logger.info(
        `Fetched ${entries.length} entries (total so far: ${totalFetched})`,
      );

      // ページネーション: 最後のページ判定
      if (entries.length < batchSize) {
        logger.info('Reached end of results.');
        break;
      }

      // レート制限対策
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  } finally {
    // ストリームを閉じる（エラー時も確実に閉じる）
    await new Promise<void>((resolve, reject) => {
      writeStream.end(() => resolve());
      writeStream.on('error', reject);
    });
  }

  // 正常完了: プログレスファイルを削除
  deleteProgress(outputDir, options.startDate, options.endDate);

  logger.info(`Saved ${totalFetched} papers to ${outputPath}`);
  return { outputPath, totalFetched };
}

// --- CLI エントリーポイント ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let startDate = '';
  let endDate = '';
  let category: string | undefined;
  let outputDir: string | undefined;
  let appendTo: string | undefined;
  let resume = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--start' && args[i + 1]) {
      startDate = args[i + 1]!;
      i++;
    } else if (args[i] === '--end' && args[i + 1]) {
      endDate = args[i + 1]!;
      i++;
    } else if (args[i] === '--category' && args[i + 1]) {
      category = args[i + 1]!;
      i++;
    } else if (args[i] === '--output' && args[i + 1]) {
      outputDir = args[i + 1]!;
      i++;
    } else if (args[i] === '--append' && args[i + 1]) {
      appendTo = args[i + 1]!;
      i++;
    } else if (args[i] === '--resume') {
      resume = true;
    }
  }

  if (!startDate || !endDate) {
    console.error(
      'Usage: npx tsx rag/biorxiv-fetcher.ts --start YYYY-MM-DD --end YYYY-MM-DD [options]',
    );
    console.error('');
    console.error('Options:');
    console.error('  --category <name>    bioRxiv カテゴリ（デフォルト: bioinformatics）');
    console.error('  --output <dir>       出力ディレクトリ（デフォルト: storage/biorxiv-tmp）');
    console.error('  --append <file>      既存 JSONL ファイルに追記');
    console.error('  --resume             前回の中断地点から再開');
    process.exit(1);
  }

  try {
    const result = await fetchBiorxivPapers({
      startDate,
      endDate,
      ...(category != null ? { category } : {}),
      ...(outputDir != null ? { outputDir } : {}),
      ...(appendTo != null ? { appendTo } : {}),
      resume,
    });
    console.log(
      `Done! Fetched ${result.totalFetched} papers → ${result.outputPath}`,
    );
  } catch (error) {
    console.error(`\nError: ${(error as Error).message}`);
    // レジュームコマンドを表示
    const resumeArgs = [
      'npx tsx chapter6-biorxiv/rag/biorxiv-fetcher.ts',
      `--start ${startDate}`,
      `--end ${endDate}`,
      ...(category != null ? [`--category ${category}`] : []),
      ...(outputDir != null ? [`--output ${outputDir}`] : []),
      '--resume',
    ];
    console.error(`\nTo resume, run:\n  ${resumeArgs.join(' ')}\n`);
    process.exit(1);
  }
}

main();
