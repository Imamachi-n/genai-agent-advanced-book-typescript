import * as fs from 'node:fs';
import * as readline from 'node:readline';

import { setupLogger } from '../../custom-logger.js';
import { QdrantStore } from '../qdrant-store.js';

const logger = setupLogger('ft-validation');

export interface ValidationReport {
  totalSampled: number;
  targetPaperFoundInTop3: number;
  targetPaperFoundInTop10: number;
  averageRelevanceScore: number;
  hitRateTop3: number;
  hitRateTop10: number;
}

interface TrainingLine {
  messages: { role: string; content: string }[];
}

interface MetadataLine {
  lineIndex: number;
  doi: string;
  queryType: string;
  language: string;
}

/**
 * 学習データの品質を検証する。
 * 理想クエリで Qdrant を検索し、対象論文が上位にヒットするかを確認する。
 */
export async function validateTrainingData(options: {
  trainingPath: string;
  metadataPath: string;
  store: QdrantStore;
  sampleSize?: number;
}): Promise<ValidationReport> {
  const sampleSize = options.sampleSize ?? 50;

  // メタデータを読み込み（DOI 紐付け）
  const metadata = await readJsonlFile<MetadataLine>(options.metadataPath);

  // 学習データを読み込み
  const trainingData = await readJsonlFile<TrainingLine>(options.trainingPath);

  // ランダムサンプリング（重複 DOI を避けるため DOI 単位でサンプル）
  const uniqueDois = [...new Set(metadata.map((m) => m.doi))];
  const sampledDois = shuffle(uniqueDois).slice(0, sampleSize);

  // サンプルされた DOI に対応する学習データの中から各 DOI 1 件ずつ
  const samples: { idealQuery: string; doi: string }[] = [];
  for (const doi of sampledDois) {
    const meta = metadata.find((m) => m.doi === doi);
    if (!meta) continue;
    const training = trainingData[meta.lineIndex];
    if (!training) continue;

    // assistant の content = 理想クエリ
    const assistantMsg = training.messages.find((m) => m.role === 'assistant');
    if (!assistantMsg) continue;

    samples.push({ idealQuery: assistantMsg.content, doi });
  }

  logger.info(`Validating ${samples.length} samples...`);

  let foundInTop3 = 0;
  let foundInTop10 = 0;
  let totalScore = 0;
  let scoredCount = 0;

  for (const sample of samples) {
    const results = await options.store.search(sample.idealQuery, 10);
    const dois = results.map((r) => r.doi);

    const rankInTop10 = dois.indexOf(sample.doi);
    if (rankInTop10 !== -1) {
      foundInTop10++;
      if (rankInTop10 < 3) {
        foundInTop3++;
      }
      const paper = results[rankInTop10];
      if (paper?.relevanceScore != null) {
        totalScore += paper.relevanceScore;
        scoredCount++;
      }
    }
  }

  const report: ValidationReport = {
    totalSampled: samples.length,
    targetPaperFoundInTop3: foundInTop3,
    targetPaperFoundInTop10: foundInTop10,
    averageRelevanceScore: scoredCount > 0 ? totalScore / scoredCount : 0,
    hitRateTop3: samples.length > 0 ? foundInTop3 / samples.length : 0,
    hitRateTop10: samples.length > 0 ? foundInTop10 / samples.length : 0,
  };

  logger.info('=== Validation Report ===');
  logger.info(`  Total sampled: ${report.totalSampled}`);
  logger.info(`  Found in top 3: ${report.targetPaperFoundInTop3} (${(report.hitRateTop3 * 100).toFixed(1)}%)`);
  logger.info(`  Found in top 10: ${report.targetPaperFoundInTop10} (${(report.hitRateTop10 * 100).toFixed(1)}%)`);
  logger.info(`  Avg relevance score: ${report.averageRelevanceScore.toFixed(3)}`);

  return report;
}

async function readJsonlFile<T>(filePath: string): Promise<T[]> {
  const items: T[] = [];
  const stream = fs.createReadStream(filePath, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    items.push(JSON.parse(trimmed) as T);
  }

  return items;
}

function shuffle<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}
