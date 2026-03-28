import * as fs from 'node:fs';
import * as readline from 'node:readline';

import type { BiorxivPaper } from '../models.js';
import { setupLogger } from '../custom-logger.js';

const logger = setupLogger('dedup-jsonl');

/**
 * JSONL ファイル内の重複 DOI を除去し、各 DOI の最新バージョンのみを残す。
 *
 * ストリーム処理で DOI → 最新レコードの Map を構築し、
 * 最後にまとめて出力するため、メモリ使用量はユニーク DOI 数に比例する。
 */
export async function deduplicateJsonl(options: {
  inputPath: string;
  outputPath?: string;
}): Promise<{ totalInput: number; totalOutput: number; duplicatesRemoved: number }> {
  const outputPath = options.outputPath ?? options.inputPath.replace('.jsonl', '_dedup.jsonl');

  // DOI → 最新バージョンの論文を保持する Map
  const latestByDoi = new Map<string, BiorxivPaper>();

  // 入力ファイルをストリーム読み込み
  const fileStream = fs.createReadStream(options.inputPath, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let totalInput = 0;

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const paper: BiorxivPaper = JSON.parse(trimmed);
    totalInput++;

    const existing = latestByDoi.get(paper.doi);
    if (!existing || paper.version > existing.version) {
      latestByDoi.set(paper.doi, paper);
    }
  }

  // 重複除去後のデータを出力
  const writeStream = fs.createWriteStream(outputPath, { encoding: 'utf-8' });
  for (const paper of latestByDoi.values()) {
    writeStream.write(`${JSON.stringify(paper)}\n`);
  }
  await new Promise<void>((resolve, reject) => {
    writeStream.end(() => resolve());
    writeStream.on('error', reject);
  });

  const totalOutput = latestByDoi.size;
  const duplicatesRemoved = totalInput - totalOutput;

  logger.info(
    `Dedup complete: ${totalInput} input → ${totalOutput} output (${duplicatesRemoved} duplicates removed)`,
  );
  logger.info(`Output: ${outputPath}`);

  return { totalInput, totalOutput, duplicatesRemoved };
}

// --- CLI エントリーポイント ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let inputPath = '';
  let outputPath: string | undefined;
  let inPlace = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && args[i + 1]) {
      inputPath = args[i + 1]!;
      i++;
    } else if (args[i] === '--output' && args[i + 1]) {
      outputPath = args[i + 1]!;
      i++;
    } else if (args[i] === '--in-place') {
      inPlace = true;
    } else if (!args[i]!.startsWith('--')) {
      inputPath = args[i]!;
    }
  }

  if (!inputPath) {
    console.error('Usage: npx tsx rag/dedup-jsonl.ts <input.jsonl> [--output <output.jsonl>] [--in-place]');
    process.exit(1);
  }

  if (!fs.existsSync(inputPath)) {
    console.error(`File not found: ${inputPath}`);
    process.exit(1);
  }

  // --in-place: 一時ファイルに書き出してから元ファイルを置き換え
  if (inPlace) {
    const tmpPath = `${inputPath}.tmp`;
    const result = await deduplicateJsonl({ inputPath, outputPath: tmpPath });
    fs.renameSync(tmpPath, inputPath);
    console.log(
      `Done! ${result.totalInput} → ${result.totalOutput} (${result.duplicatesRemoved} duplicates removed) [in-place: ${inputPath}]`,
    );
  } else {
    const result = await deduplicateJsonl({
      inputPath,
      ...(outputPath != null ? { outputPath } : {}),
    });
    console.log(
      `Done! ${result.totalInput} → ${result.totalOutput} (${result.duplicatesRemoved} duplicates removed) → ${outputPath ?? inputPath.replace('.jsonl', '_dedup.jsonl')}`,
    );
  }
}

main();
