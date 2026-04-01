# クエリ拡張モデルの FT 学習データ生成パイプラインのプラン

## Context

bioRxiv 論文リサーチエージェントのクエリ拡張（サブタスク → 英語検索クエリ変換）を gpt-4o-mini で行っているが、関連度スコアが 0.44〜0.56 と中程度に留まっている。バイオインフォマティクス特有の略語・同義語・表記揺れへの対応が弱いことが原因。

Qdrant に既に格納されている数千件の論文データ（タイトル+アブストラクト）を活用して、**クエリ拡張モデルの FT 用学習データを自動生成するパイプライン**を構築する。

---

## アーキテクチャ

```
Qdrant → [論文抽出] → [合成クエリ生成(LLM)] → [理想クエリ生成(LLM)] → [JSONL整形] → [品質検証]
```

### 出力ファイル

- `storage/ft-training-data/training_YYYY-MM-DD.jsonl` — OpenAI FT 用学習データ
- `storage/ft-training-data/training_YYYY-MM-DD_metadata.jsonl` — DOI 紐付けメタデータ（検証用）

---

## ファイル構成

すべて `packages/@ai-suburi/core/chapter6-biorxiv/rag/ft-pipeline/` に配置:

| ファイル | 役割 |
|----------|------|
| `paper-extractor.ts` | Qdrant から全論文を scroll API で抽出 |
| `query-synthesizer.ts` | 論文ごとに合成ユーザークエリを LLM で生成（日英混在・5種） |
| `ideal-query-generator.ts` | 論文ごとに理想の英語検索クエリを LLM で生成 |
| `training-data-formatter.ts` | EXPAND_QUERY_PROMPT + 合成クエリ + 理想クエリ → JSONL 整形 |
| `validation.ts` | 学習データの品質検証（理想クエリで Qdrant 検索し対象論文がヒットするか） |
| `generate-ft-data.ts` | CLI エントリーポイント |

---

## Step 1: 論文抽出 (`paper-extractor.ts`)

Qdrant の `scroll` API で全論文を取得。既存の `QdrantStore` はバルク取得メソッドがないため、`QdrantClient` を直接使用。

```typescript
export async function extractAllPapers(options: {
  collectionName: string;
  qdrantUrl?: string;
  batchSize?: number; // default: 100
}): Promise<BiorxivPaper[]>
```

**再利用**: `qdrant-store.ts` L138 の `scroll` パターン、payload → BiorxivPaper のマッピング（L113-128）

---

## Step 2: 合成クエリ生成 (`query-synthesizer.ts`)

論文ごとに gpt-4o（smart モデル）で 5 種の多様なクエリを生成:

- 日本語キーワード型（例: 「一細胞RNA解析 最新手法」）
- 日本語質問型（例: 「scRNA-seqの細胞アノテーション自動化ツールは？」）
- 英語キーワード型（例: 「single-cell RNA-seq cell type annotation」）
- 英語タスク記述型（例: 「Investigate automated methods for cell classification in scRNA-seq datasets」）
- ゴール記述型（例: 「生成AIを用いたゲノム解析の最新動向を調べる」）

```typescript
export async function synthesizeUserQueries(
  llm: ChatOpenAI,
  paper: BiorxivPaper,
  queriesPerPaper?: number, // default: 5
): Promise<SyntheticQuery[]>
```

Zod スキーマで構造化出力。`withStructuredOutput` パターンを使用。

---

## Step 3: 理想クエリ生成 (`ideal-query-generator.ts`)

論文のタイトル+アブストラクトから、ベクトル検索で最もヒットしやすい英語検索クエリを生成。

```typescript
export async function generateIdealQuery(
  llm: ChatOpenAI,
  paper: BiorxivPaper,
): Promise<string>
```

**ポイント**: 生成したクエリの Embedding を計算し、論文の Embedding とのコサイン類似度が 0.6 以上であることを確認。未達なら再生成（最大 3 回）。

---

## Step 4: JSONL 整形 (`training-data-formatter.ts`)

OpenAI FT 形式で出力。**system プロンプトは `rag-searcher.ts` の `EXPAND_QUERY_PROMPT` と完全一致させる**（推論時と同じコンテキストで学習させるため）。

1 論文 × 5 クエリ = 5 行の学習データ:

```jsonl
{"messages": [
  {"role": "system", "content": "<EXPAND_QUERY_PROMPT（feedback 空）>"},
  {"role": "user", "content": "目標: <合成ゴール>\nクエリ: <合成ユーザークエリ>"},
  {"role": "assistant", "content": "<理想の英語検索クエリ>"}
]}
```

メタデータ JSONL（検証用）:

```jsonl
{"lineIndex": 0, "doi": "10.1101/...", "queryType": "keyword", "language": "ja"}
```

---

## Step 5: 品質検証 (`validation.ts`)

学習データからランダムサンプルを抽出し、理想クエリで Qdrant を検索して対象論文がヒットするか確認。

```typescript
interface ValidationReport {
  totalSampled: number;
  targetPaperFoundInTop3: number;
  targetPaperFoundInTop10: number;
  averageRelevanceScore: number;
  hitRate: number; // top10 ヒット率
}
```

---

## Step 6: CLI エントリーポイント (`generate-ft-data.ts`)

```bash
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts [options]

Options:
  --output <dir>           出力ディレクトリ（default: storage/ft-training-data）
  --queries-per-paper <n>  論文あたりの合成クエリ数（default: 5）
  --limit <n>              処理する論文数の上限（テスト用）
  --validate               生成後に品質検証を実行
  --validate-only <path>   既存 JSONL の品質検証のみ実行
```

CLI パーシングは `biorxiv-fetcher.ts` の手動 `process.argv` パターンを踏襲。

---

## 再利用するモジュール

| モジュール | ファイル | 用途 |
|-----------|----------|------|
| `QdrantClient` | `@qdrant/js-client-rest` | scroll API で全論文取得 |
| `BiorxivPaper` | `models.ts` | 論文型定義 |
| `loadSettings` | `configs.ts` | API キー・Qdrant 接続情報 |
| `createLlm` | `configs.ts` | gpt-4o インスタンス（学習データ生成用） |
| `setupLogger` | `custom-logger.ts` | ログ出力 |
| `EXPAND_QUERY_PROMPT` | `rag-searcher.ts` | system プロンプト（学習データに埋め込み） |
| `QdrantStore` | `qdrant-store.ts` | 検証時のベクトル検索 |

---

## 検証方法

1. `--limit 10` で 10 論文分のテストデータを生成
2. 出力 JSONL が OpenAI FT フォーマットに準拠していることを確認
3. `--validate` で品質検証を実行し、hitRate > 70% を確認
4. 全論文で本番データを生成（`--limit` なし）
