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
 * JSONL ストリーミング Writer。1 エントリずつ追記できる。
 */
export class TrainingDataWriter {
  readonly trainingPath: string;
  readonly metadataPath: string;
  private trainingStream: fs.WriteStream;
  private metadataStream: fs.WriteStream;
  private lineIndex = 0;

  constructor(outputDir: string, options?: { append?: boolean }) {
    fs.mkdirSync(outputDir, { recursive: true });

    const timestamp = new Date().toISOString().split('T')[0]!;
    this.trainingPath = path.join(outputDir, `training_${timestamp}.jsonl`);
    this.metadataPath = path.join(
      outputDir,
      `training_${timestamp}_metadata.jsonl`,
    );

    const flags = options?.append ? 'a' : 'w';
    this.trainingStream = fs.createWriteStream(this.trainingPath, {
      flags,
      encoding: 'utf-8',
    });
    this.metadataStream = fs.createWriteStream(this.metadataPath, {
      flags,
      encoding: 'utf-8',
    });

    // append モードの場合、既存行数を数える
    if (options?.append && fs.existsSync(this.trainingPath)) {
      const content = fs.readFileSync(this.trainingPath, 'utf-8');
      this.lineIndex = content.split('\n').filter((l) => l.trim()).length;
      logger.info(`Appending to existing file (${this.lineIndex} lines)`);
    }
  }

  /** 1 エントリ分（論文 × N クエリ）を JSONL に追記する */
  writeEntry(entry: TrainingEntry): number {
    let written = 0;

    for (const sq of entry.syntheticQueries) {
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
        lineIndex: this.lineIndex,
        doi: entry.paper.doi,
        queryType: sq.queryType,
        language: sq.language,
      };

      this.trainingStream.write(`${JSON.stringify(trainingLine)}\n`);
      this.metadataStream.write(`${JSON.stringify(metadataLine)}\n`);
      this.lineIndex++;
      written++;
    }

    return written;
  }

  /** ストリームを閉じて書き込みを完了する */
  async close(): Promise<{ totalExamples: number }> {
    await Promise.all([
      new Promise<void>((resolve, reject) => {
        this.trainingStream.end(() => resolve());
        this.trainingStream.on('error', reject);
      }),
      new Promise<void>((resolve, reject) => {
        this.metadataStream.end(() => resolve());
        this.metadataStream.on('error', reject);
      }),
    ]);

    logger.info(
      `Written ${this.lineIndex} training examples to ${this.trainingPath}`,
    );
    logger.info(`Written metadata to ${this.metadataPath}`);

    return { totalExamples: this.lineIndex };
  }
}
