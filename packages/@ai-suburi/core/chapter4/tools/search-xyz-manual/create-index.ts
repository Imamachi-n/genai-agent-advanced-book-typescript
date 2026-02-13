import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PDFParse } from 'pdf-parse';
import { openDatabase } from './db.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../data');

// テキストを分割する関数（Python の RecursiveCharacterTextSplitter を再現）
function splitText(
  text: string,
  chunkSize = 300,
  chunkOverlap = 20,
  separators = ['\n\n', '\n', ' ', ''],
): string[] {
  if (text.length === 0) return [];
  if (text.length <= chunkSize) return [text];

  // 使用可能な最初のセパレータを見つける
  let separator = '';
  for (const sep of separators) {
    if (sep === '' || text.includes(sep)) {
      separator = sep;
      break;
    }
  }

  // セパレータでテキストを分割
  const parts = separator === '' ? [...text] : text.split(separator);

  const chunks: string[] = [];
  let currentChunk = '';

  for (const part of parts) {
    const candidate = currentChunk ? currentChunk + separator + part : part;

    if (candidate.length > chunkSize && currentChunk) {
      chunks.push(currentChunk.trim());
      // オーバーラップ: 前のチャンクの末尾を次のチャンクの先頭に含める
      const overlapText = currentChunk.slice(-chunkOverlap);
      currentChunk = overlapText + separator + part;
    } else {
      currentChunk = candidate;
    }
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  return chunks.filter((c) => c.length > 0);
}

// PDFファイルを読み込み、テキストをチャンク分割してDBに投入する
export async function createIndex(): Promise<void> {
  const db = openDatabase();

  // 既存データをクリア
  db.exec('DELETE FROM documents');
  // FTSインデックスもリビルド
  db.exec("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')");

  const pdfFiles = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith('.pdf'));

  if (pdfFiles.length === 0) {
    console.log(`PDFファイルが見つかりません: ${DATA_DIR}`);
    db.close();
    return;
  }

  const insert = db.prepare(
    'INSERT INTO documents (file_name, content) VALUES (?, ?)',
  );
  const insertMany = db.transaction((rows: Array<[string, string]>) => {
    for (const row of rows) {
      insert.run(...row);
    }
  });

  let totalChunks = 0;

  for (const pdfFile of pdfFiles) {
    const filePath = path.join(DATA_DIR, pdfFile);
    const buffer = fs.readFileSync(filePath);

    // pdf-parse v2: PDFParse クラスで読み込み → getText() でテキスト抽出
    const parser = new PDFParse({ data: new Uint8Array(buffer) });
    const textResult = await parser.getText();
    await parser.destroy();

    // テキストをチャンク分割
    const chunks = splitText(textResult.text, 300, 20);
    const rows = chunks.map((chunk): [string, string] => [pdfFile, chunk]);

    insertMany(rows);
    totalChunks += chunks.length;
    console.log(`${pdfFile}: ${chunks.length} チャンクをインデックスに追加`);
  }

  console.log(`\n合計: ${totalChunks} チャンクをインデックスに追加完了`);
  db.close();
}

// 直接実行された場合のみインデックスを作成
if (fileURLToPath(import.meta.url) === process.argv[1]) {
  createIndex();
}
