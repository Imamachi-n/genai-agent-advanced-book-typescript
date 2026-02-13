import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// デフォルトのデータベースパス
const DEFAULT_DB_PATH = path.resolve(
  __dirname,
  '../data/xyz-manual-search.db',
);

// データベースを開き、FTS5テーブルを初期化する
export function openDatabase(
  dbPath: string = DEFAULT_DB_PATH,
): Database.Database {
  const db = new Database(dbPath);

  // WALモードを有効にしてパフォーマンスを向上
  db.pragma('journal_mode = WAL');

  // ドキュメント格納テーブル
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_name TEXT NOT NULL,
      content TEXT NOT NULL
    )
  `);

  // FTS5仮想テーブル（trigramトークナイザーで日本語対応）
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      content,
      content='documents',
      content_rowid='id',
      tokenize='trigram'
    )
  `);

  // 挿入トリガー: documentsに追加されたらFTSインデックスも更新
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
      INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
    END
  `);

  // 削除トリガー
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
      INSERT INTO documents_fts(documents_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    END
  `);

  // 更新トリガー
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
      INSERT INTO documents_fts(documents_fts, rowid, content)
        VALUES('delete', old.id, old.content);
      INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
    END
  `);

  return db;
}
