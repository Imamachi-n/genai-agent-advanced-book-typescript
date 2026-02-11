import Database from 'better-sqlite3';
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod/v4';

// --- Zodスキーマ: LLMの構造化出力用 ---
const SQLQuery = z.object({
  sql: z.string().describe('実行するSQLクエリ'),
  explanation: z.string().describe('クエリの簡単な説明'),
});

// --- データベース初期化 ---
function initializeDatabase(): Database.Database {
  const db = new Database(':memory:');

  db.exec(`
    CREATE TABLE IF NOT EXISTS employees (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      department TEXT,
      salary INTEGER,
      hire_date TEXT
    )
  `);

  const insert = db.prepare(
    'INSERT INTO employees (name, department, salary, hire_date) VALUES (?, ?, ?, ?)',
  );

  const insertMany = db.transaction(
    (rows: Array<[string, string, number, string]>) => {
      for (const row of rows) {
        insert.run(...row);
      }
    },
  );

  insertMany([
    ['Tanaka Taro', 'IT', 600000, '2020-04-01'],
    ['Yamada Hanako', 'HR', 550000, '2019-03-15'],
    ['Suzuki Ichiro', 'Finance', 700000, '2021-01-20'],
    ['Watanabe Yuki', 'IT', 650000, '2020-07-10'],
    ['Kato Akira', 'Marketing', 580000, '2022-02-01'],
    ['Nakamura Yui', 'IT', 620000, '2021-05-15'],
    ['Yoshida Saki', 'Finance', 680000, '2020-12-01'],
    ['Matsumoto Ryu', 'HR', 540000, '2022-08-20'],
    ['Inoue Kana', 'Marketing', 590000, '2021-11-10'],
    ['Takahashi Ken', 'IT', 710000, '2019-09-05'],
  ]);

  return db;
}

// --- スキーマ情報の取得 ---
function getSchemaInfo(db: Database.Database): string {
  const tables = db
    .prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    )
    .all() as Array<{ name: string }>;

  const schemaLines: string[] = [];

  for (const table of tables) {
    const columns = db
      .prepare(`PRAGMA table_info('${table.name}')`)
      .all() as Array<{
      cid: number;
      name: string;
      type: string;
      notnull: number;
      dflt_value: string | null;
      pk: number;
    }>;

    const columnDefs = columns
      .map((col) => {
        const parts = [col.name, col.type];
        if (col.pk) parts.push('PRIMARY KEY');
        if (col.notnull) parts.push('NOT NULL');
        return parts.join(' ');
      })
      .join(', ');

    schemaLines.push(`CREATE TABLE ${table.name} (${columnDefs})`);

    // サンプルデータ（3行）をLLMのコンテキストとして追加
    const sampleRows = db
      .prepare(`SELECT * FROM "${table.name}" LIMIT 3`)
      .all();
    if (sampleRows.length > 0) {
      schemaLines.push(`/* Sample rows from ${table.name}: */`);
      schemaLines.push(`/* ${JSON.stringify(sampleRows)} */`);
    }
  }

  return schemaLines.join('\n');
}

// --- OpenAIでSQL生成 ---
async function generateSQL(
  client: OpenAI,
  schema: string,
  keywords: string,
): Promise<z.infer<typeof SQLQuery>> {
  const response = await client.chat.completions.parse({
    model: 'gpt-4o-mini',
    temperature: 0,
    messages: [
      {
        role: 'system',
        content: `あなたはSQLの専門家です。以下のSQLiteデータベーススキーマに基づいて、ユーザーの自然言語リクエストからSQLクエリを生成してください。

データベーススキーマ:
${schema}

ルール:
- 有効なSQLite SQL構文を生成すること
- SELECTクエリのみ生成すること（INSERT, UPDATE, DELETE, DROPなどは不可）
- PostgreSQL固有の構文は使用しないこと
- 上記のスキーマに対してそのまま実行可能なクエリを生成すること`,
      },
      {
        role: 'user',
        content: keywords,
      },
    ],
    response_format: zodResponseFormat(SQLQuery, 'sql_query'),
  });

  const parsed = response.choices[0]?.message.parsed;
  if (!parsed) {
    const refusal = response.choices[0]?.message.refusal;
    if (refusal) {
      throw new Error(`LLMがリクエストを拒否しました: ${refusal}`);
    }
    throw new Error('LLMからの構造化出力のパースに失敗しました');
  }
  return parsed;
}

// --- SQL実行 ---
function executeQuery(
  db: Database.Database,
  sql: string,
): { columns: string[]; rows: unknown[][] } {
  // 安全性チェック: SELECTクエリのみ許可
  const normalized = sql.trim().toUpperCase();
  if (!normalized.startsWith('SELECT')) {
    throw new Error(
      `安全性チェック: SELECTクエリのみ許可されています。受信: ${sql.substring(0, 50)}...`,
    );
  }

  const rows = db.prepare(sql).all() as Array<Record<string, unknown>>;

  if (rows.length === 0) {
    return { columns: [], rows: [] };
  }

  const columns = Object.keys(rows[0]!);
  const rowArrays = rows.map((row) => columns.map((col) => row[col]));

  return { columns, rows: rowArrays };
}

// --- 結果フォーマット ---
function formatResults(columns: string[], rows: unknown[][]): string {
  if (rows.length === 0) {
    return '結果が見つかりませんでした。';
  }

  const header = columns.join(' | ');
  const separator = columns.map(() => '---').join(' | ');
  const dataRows = rows.map((row) => row.map(String).join(' | '));

  return [header, separator, ...dataRows].join('\n');
}

// --- text_to_sql_search ツール関数 ---
/**
 * 自然言語でのクエリをSQLクエリに変換し、SQLデータベースで検索を実行します。
 *
 * 機能:
 * - このToolは、与えられた自然言語形式のキーワードをもとに、SQLクエリを生成します。
 * - LLMを使用してSQL文を生成し、インメモリSQLiteデータベースで検索を実行します。
 * - 取得した検索結果を返します。
 *
 * @param keywords - 実行したいクエリの自然言語キーワード
 *   例: "employeeテーブルの情報は何件ありますか？"
 * @returns データベース検索結果の文字列
 */
async function textToSqlSearch(keywords: string): Promise<string> {
  try {
    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const db = initializeDatabase();

    try {
      const schema = getSchemaInfo(db);
      console.log('Database Schema:\n', schema, '\n');

      console.log('Query:', keywords);
      const { sql, explanation } = await generateSQL(client, schema, keywords);
      console.log('Generated SQL:', sql);
      console.log('Explanation:', explanation);

      const { columns, rows } = executeQuery(db, sql);

      const result = formatResults(columns, rows);
      console.log('\nResults:');
      console.log(result);

      return result;
    } finally {
      db.close();
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`エラー: ${message}`);
    return `エラー: ${message}`;
  }
}

// --- 実行例 ---
const args = { keywords: 'employeeテーブルの情報は何件ありますか？' };
await textToSqlSearch(args.keywords);
