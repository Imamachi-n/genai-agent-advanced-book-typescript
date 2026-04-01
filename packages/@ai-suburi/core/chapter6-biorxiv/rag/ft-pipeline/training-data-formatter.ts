import * as fs from 'node:fs';
import * as path from 'node:path';

import { setupLogger } from '../../custom-logger.js';
import type { BiorxivPaper } from '../../models.js';
import type { SyntheticQuery } from './query-synthesizer.js';

const logger = setupLogger('training-data-formatter');

export interface TrainingEntry {
  paper: BiorxivPaper;
  syntheticQueries: SyntheticQuery[];
  idealQuery: string;
}

interface TrainingMessage {
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[];
}

interface MetadataEntry {
  lineIndex: number;
  doi: string;
  queryType: string;
  language: string;
}

/**
 * EXPAND_QUERY_PROMPT の system 部分。rag-searcher.ts と完全一致させる。
 * feedback は空文字（通常パス）で固定。
 */
const SYSTEM_PROMPT = `<system>
あなたは、与えられたサブクエリからbioRxiv論文のRAG検索に最適な検索クエリを生成する専門家です。
バイオインフォマティクス分野の学術的な文脈を理解し、ベクトル検索で効果的にヒットするクエリを作成してください。


</system>

## 主要タスク

1. 提供されたサブクエリを分析する
2. サブクエリから重要なキーワードを抽出する
3. ベクトル検索に最適化された自然言語の検索クエリを構築する

## 重要なルール

<rules>
1. 検索クエリは英語で生成すること（bioRxivの論文は英語のため）
2. クエリには1〜3つの主要なキーワードまたはフレーズを含めること
3. バイオインフォマティクスの専門用語を適切に使用すること
4. 自然言語の文として生成すること（ベクトル検索に最適）
5. 説明や理由付けは含めず、純粋な検索クエリのみを出力すること
6. バイオインフォマティクス特有の表記揺れを考慮すること:
   - 略語と正式名称の両方を含める（例: APA / Alternative Polyadenylation）
   - ハイフン有無の揺れを考慮する（例: single-cell / single cell）
   - 技術名とその応用分野を組み合わせる（例: "machine learning" + "genomic variant"）
</rules>

## 入力フォーマット

<input_format>
目標: {goal_setting}
クエリ: {query}
</input_format>

REMEMBER: rulesタグの内容に必ず従うこと`;

/**
 * TrainingEntry の配列を OpenAI FT 用 JSONL + メタデータ JSONL に書き出す。
 */
export async function formatTrainingData(options: {
  entries: TrainingEntry[];
  outputDir: string;
}): Promise<{ trainingPath: string; metadataPath: string; totalExamples: number }> {
  const { entries, outputDir } = options;
  fs.mkdirSync(outputDir, { recursive: true });

  const timestamp = new Date().toISOString().split('T')[0]!;
  const trainingPath = path.join(outputDir, `training_${timestamp}.jsonl`);
  const metadataPath = path.join(outputDir, `training_${timestamp}_metadata.jsonl`);

  const trainingStream = fs.createWriteStream(trainingPath, {
    flags: 'w',
    encoding: 'utf-8',
  });
  const metadataStream = fs.createWriteStream(metadataPath, {
    flags: 'w',
    encoding: 'utf-8',
  });

  let lineIndex = 0;

  for (const entry of entries) {
    for (const sq of entry.syntheticQueries) {
      // ゴール記述型クエリの場合はゴール部分にも使う
      const goal =
        sq.queryType === 'goal'
          ? sq.query
          : `${entry.paper.category}分野の研究動向を調査する`;

      const trainingLine: TrainingMessage = {
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          {
            role: 'user',
            content: `目標: ${goal}\nクエリ: ${sq.query}`,
          },
          { role: 'assistant', content: entry.idealQuery },
        ],
      };

      const metadataLine: MetadataEntry = {
        lineIndex,
        doi: entry.paper.doi,
        queryType: sq.queryType,
        language: sq.language,
      };

      trainingStream.write(`${JSON.stringify(trainingLine)}\n`);
      metadataStream.write(`${JSON.stringify(metadataLine)}\n`);
      lineIndex++;
    }
  }

  await Promise.all([
    new Promise<void>((resolve, reject) => {
      trainingStream.end(() => resolve());
      trainingStream.on('error', reject);
    }),
    new Promise<void>((resolve, reject) => {
      metadataStream.end(() => resolve());
      metadataStream.on('error', reject);
    }),
  ]);

  logger.info(
    `Written ${lineIndex} training examples to ${trainingPath}`,
  );
  logger.info(`Written metadata to ${metadataPath}`);

  return { trainingPath, metadataPath, totalExamples: lineIndex };
}
