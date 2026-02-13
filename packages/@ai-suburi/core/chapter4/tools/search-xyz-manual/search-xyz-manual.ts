import { tool } from '@langchain/core/tools';
import { z } from 'zod/v4';
import { createIndex } from './create-index.js';
import { openDatabase } from './db.js';

// 検索結果の最大取得数
const MAX_SEARCH_RESULTS = 3;

// 検索結果の型定義
interface SearchOutput {
  fileName: string;
  content: string;
}

// 入力スキーマを定義
const searchKeywordSchema = z.object({
  keywords: z.string().describe('全文検索用のキーワード'),
});

// FTS5 trigramの最小トークン長
const MIN_TRIGRAM_LENGTH = 3;

// LangChainのtool関数を使って、検索機能をツール化
const searchXyzManual = tool(
  async ({ keywords }): Promise<SearchOutput[]> => {
    console.log(`Searching XYZ manual by keyword: ${keywords}`);

    const db = openDatabase();

    // インデックスが未作成の場合は自動で作成
    const count = db.prepare('SELECT COUNT(*) as cnt FROM documents').get() as {
      cnt: number;
    };
    if (count.cnt === 0) {
      db.close();
      console.log('インデックスが空のため、自動作成を開始します...');
      await createIndex();
      // インデックス作成後に再接続
      return searchXyzManual.invoke({ keywords });
    }

    try {
      const terms = keywords.split(/\s+/).filter((k) => k.length > 0);

      if (terms.length === 0) {
        console.log('Search results: 0 hits');
        return [];
      }

      let rows: Array<{ file_name: string; content: string }>;

      // trigram トークナイザーは3文字未満のキーワードに対応できないため、
      // 短いキーワードの場合は LIKE にフォールバック
      const hasShortTerm = terms.some((t) => t.length < MIN_TRIGRAM_LENGTH);

      if (hasShortTerm) {
        // LIKE による部分一致検索にフォールバック
        const conditions = terms.map(() => 'content LIKE ?').join(' AND ');
        const params = terms.map((t) => `%${t}%`);
        rows = db
          .prepare(
            `SELECT file_name, content FROM documents WHERE ${conditions} LIMIT ?`,
          )
          .all(...params, MAX_SEARCH_RESULTS) as Array<{
          file_name: string;
          content: string;
        }>;
      } else {
        // FTS5 trigram 検索: 各キーワードをダブルクォートで囲み AND で結合
        const ftsQuery = terms.map((t) => `"${t}"`).join(' AND ');
        rows = db
          .prepare(
            `
						SELECT d.file_name, d.content
						FROM documents d
						JOIN documents_fts fts ON d.id = fts.rowid
						WHERE documents_fts MATCH ?
						ORDER BY rank
						LIMIT ?
					`,
          )
          .all(ftsQuery, MAX_SEARCH_RESULTS) as Array<{
          file_name: string;
          content: string;
        }>;
      }

      console.log(`Search results: ${rows.length} hits`);

      const outputs: SearchOutput[] = rows.map((row) => ({
        fileName: row.file_name,
        content: row.content,
      }));

      console.log('Finished searching XYZ manual by keyword');
      return outputs;
    } finally {
      db.close();
    }
  },
  {
    name: 'search_xyz_manual',
    description:
      'XYZシステムのドキュメントを調査する関数。エラーコードや固有名詞が質問に含まれる場合は、この関数を使ってキーワード検索を行う。',
    schema: searchKeywordSchema,
  },
);

export { searchXyzManual, type SearchOutput };
