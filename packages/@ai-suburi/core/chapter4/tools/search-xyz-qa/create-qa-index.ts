import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import OpenAI from 'openai';
import { openDatabase } from '../db.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../data');

/**
 * CSVファイルを読み込み、各行を Q&A オブジェクトとして返す
 * @param filePath - 読み込むCSVファイルのパス
 * @returns Q&A オブジェクトの配列
 */
function loadCsv(filePath: string): Array<{ q: string; a: string }> {
  const text = fs.readFileSync(filePath, 'utf-8');
  const lines = text.split('\n').filter((line) => line.trim().length > 0);

  // ヘッダー行をスキップ
  const rows: Array<{ q: string; a: string }> = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i] ?? '';
    // 最初のカンマで分割（A列にカンマが含まれる可能性があるため）
    const commaIndex = line.indexOf(',');
    if (commaIndex === -1) continue;
    const q = line.slice(0, commaIndex);
    const a = line.slice(commaIndex + 1);
    rows.push({ q, a });
  }
  return rows;
}

/**
 * OpenAI API で embedding ベクトルを生成する
 * @param client - OpenAI クライアント
 * @param text - embedding を生成するテキスト
 * @returns embedding ベクトル
 */
async function generateEmbedding(
  client: OpenAI,
  text: string,
): Promise<number[]> {
  const response = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  return response.data[0]?.embedding ?? [];
}

/**
 * CSVファイルを読み込み、embeddingを生成してDBに投入する
 */
export async function createQaIndex(): Promise<void> {
  const db = openDatabase();

  // 既存データをクリア
  db.exec('DELETE FROM qa_documents');

  const csvFiles = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith('.csv'));

  if (csvFiles.length === 0) {
    console.log(`CSVファイルが見つかりません: ${DATA_DIR}`);
    db.close();
    return;
  }

  const openaiClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  const insert = db.prepare(
    'INSERT INTO qa_documents (file_name, content, embedding) VALUES (?, ?, ?)',
  );

  let totalDocs = 0;

  for (const csvFile of csvFiles) {
    const filePath = path.join(DATA_DIR, csvFile);
    const rows = loadCsv(filePath);

    for (const row of rows) {
      // Python版と同様に content を結合（CSVLoaderの出力形式を再現）
      const content = `Q: ${row.q}\nA: ${row.a}`;
      // Python版と同様にスペースを除去
      const cleanContent = content.replace(/ /g, '');

      console.log(
        `[${totalDocs + 1}] embedding生成中: ${cleanContent.slice(0, 50)}...`,
      );
      const embedding = await generateEmbedding(openaiClient, cleanContent);

      // Float64Array → Buffer に変換して BLOB として保存
      const buffer = Buffer.from(new Float64Array(embedding).buffer);
      insert.run(csvFile, cleanContent, buffer);
      totalDocs++;
    }

    console.log(`${csvFile}: ${rows.length} ドキュメントをインデックスに追加`);
  }

  console.log(`\n合計: ${totalDocs} ドキュメントをインデックスに追加完了`);
  db.close();
}

// 直接実行された場合のみインデックスを作成
if (fileURLToPath(import.meta.url) === process.argv[1]) {
  createQaIndex();
}
