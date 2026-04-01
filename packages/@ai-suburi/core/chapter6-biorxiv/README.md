# Chapter 6-bioRxiv: bioRxiv 論文 RAG 知識ベース + リサーチ AI エージェント

bioRxiv の bioinformatics 分野の論文を RAG（Retrieval-Augmented Generation）で知識ベース化し、質問に応じて論文を検索・分析・レポート生成する AI エージェント。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│              データ取り込みパイプライン（2ステップ）         │
│                                                     │
│  Step A: bioRxiv API ──→ JSONL ファイル保存（tmp）      │
│          (日付+カテゴリ)    (タイトル+Abstract+メタデータ)  │
│                                                     │
│  Step B: JSONL 読み込み ──→ OpenAI Embeddings          │
│                           ──→ Qdrant 格納             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                リサーチ AI エージェント（3層構造）          │
│                                                     │
│  ResearchAgent（メインオーケストレーター）                │
│   ├─ ヒアリング → ゴール最適化 → クエリ分解              │
│   │                                                 │
│   ├─ PaperSearchAgent（検索＆並列分析）                │
│   │   ├─ Qdrant RAG 検索                            │
│   │   ├─ OpenAI Embeddings + コサイン類似度リランキング  │
│   │   ├─ pdf-parse でローカル PDF→テキスト変換          │
│   │   └─ PaperAnalyzerAgent × N（並列論文分析）        │
│   │       └─ セクション選択 → 十分性チェック → 要約      │
│   │                                                 │
│   ├─ タスク評価（不足なら再検索）                       │
│   └─ 最終レポート生成（GPT-4o）                        │
└─────────────────────────────────────────────────────┘
```

## 技術スタック

| コンポーネント | 技術 |
|------------|------|
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Embedding | OpenAI text-embedding-3-small |
| ベクトルDB | Qdrant（docker-compose） |
| リランキング | OpenAI Embeddings + コサイン類似度 |
| PDF変換 | pdf-parse（ローカル） |
| フレームワーク | LangGraph |

## セットアップ

### 1. 環境変数

```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Qdrant サーバー起動

```bash
cd chapter6-biorxiv
docker compose up -d
```

### 3. 依存パッケージ（プロジェクトルートで実行）

```bash
pnpm install
```

## 使い方

### Step 1: bioRxiv 論文メタデータの取得

bioRxiv API から bioinformatics 分野の論文メタデータを取得し、JSONL ファイルとして逐次保存する。ページ取得ごとにファイルに追記するため、大量データでもメモリを圧迫しない。

```bash
# 直近1週間分を取得する例
npx tsx chapter6-biorxiv/rag/biorxiv-fetcher.ts --start 2025-03-01 --end 2025-03-07

# カテゴリや出力先を指定する場合
npx tsx chapter6-biorxiv/rag/biorxiv-fetcher.ts --start 2025-01-01 --end 2025-03-28 --category bioinformatics --output storage/biorxiv-tmp

# エラーで中断した場合、--resume で前回の続きから再開
npx tsx chapter6-biorxiv/rag/biorxiv-fetcher.ts --start 2025-01-01 --end 2025-03-28 --resume

# 既存 JSONL に別の日付範囲を追加取得
npx tsx chapter6-biorxiv/rag/biorxiv-fetcher.ts --start 2025-03-28 --end 2025-04-10 \
  --append storage/biorxiv-tmp/biorxiv_2025-01-01_2025-03-28_*.jsonl
```

JSONL ファイルは `storage/biorxiv-tmp/` に保存される（1行1論文の JSON Lines 形式）。bioRxiv API は 100 件/リクエストでページネーションされる。bioinformatics カテゴリは約 40,000 件以上あるため、まずは短い日付範囲から始めることを推奨。

- **自動リトライ**: 429（レート制限）や 5xx エラー時にエクスポネンシャルバックオフで自動リトライ
- **レジューム**: プロセスが中断された場合、`--resume` でプログレスファイルから再開可能
- **追記取得**: `--append <file>` で既存 JSONL に別の日付範囲のデータを追加取得可能

### Step 1.5: JSONL の重複除去（任意）

bioRxiv API は同じ DOI の複数バージョンを返す場合がある。Qdrant 投入前に各 DOI の最新バージョンのみ残して重複を除去できる。

```bash
# 別ファイルに出力（*_dedup.jsonl）
npx tsx chapter6-biorxiv/rag/dedup-jsonl.ts storage/biorxiv-tmp/biorxiv_2021-03-01_2025-03-27_*.jsonl

# 元ファイルを直接置き換え
npx tsx chapter6-biorxiv/rag/dedup-jsonl.ts storage/biorxiv-tmp/biorxiv_2021-03-01_2025-03-27_*.jsonl --in-place

# 出力先を指定
npx tsx chapter6-biorxiv/rag/dedup-jsonl.ts --input input.jsonl --output clean.jsonl
```

### Step 2: Qdrant にデータ投入

Step 1 で保存した JSONL ファイルを行単位でストリーム読み込みし、Qdrant ベクトルDB に投入する。大量データでもメモリを圧迫しない。

```bash
# JSONL ファイルを指定して投入
npx tsx chapter6-biorxiv/rag/qdrant-loader.ts --input storage/biorxiv-tmp/biorxiv_2021-03-01_2025-03-27_2026-03-28T00-50-45-897Z_dedup.jsonl

# 既存データを上書き upsert
npx tsx chapter6-biorxiv/rag/qdrant-loader.ts --force --input storage/biorxiv-tmp/biorxiv_2021-03-01_2025-03-27_2026-03-28T00-50-45-897Z_dedup.jsonl

# バッチサイズを指定（デフォルト: 50）
npx tsx chapter6-biorxiv/rag/qdrant-loader.ts --input storage/biorxiv-tmp/biorxiv_2025-03-01_2025-03-07_*.jsonl --batch-size 30
```

重複チェック付きなので、同じ JSONL を再投入しても二重登録されない。

### Step 3: リサーチエージェント実行

```bash
# CLI で実行（ヒアリングあり）
npx tsx chapter6-biorxiv/agent/research-agent.ts "single-cell RNA-seq解析の最新手法について調べる"

# ヒアリングをスキップして即実行
npx tsx chapter6-biorxiv/agent/research-agent.ts "CRISPR スクリーニングのバイオインフォマティクス解析" --skip-feedback
```

### Step 4: FT 学習データ生成（任意）

クエリ拡張モデルのファインチューニング用学習データを自動生成する。Qdrant に格納済みの論文データから、合成ユーザークエリ＋理想検索クエリのペアを作成する。

```bash
# テスト実行（10 論文分 + 品質検証、デフォルト: gpt-5-nano・5 並列・Embedding 検証スキップ）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --limit 10 --validate

# 本番データ生成（全論文対象）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts

# モデルを指定（コストと品質のトレードオフ）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --model gpt-5-nano      # 最安（~$3.5/1.8万件）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --model gpt-4.1-nano    # 安価（~$5/1.8万件）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --model gpt-5.4-nano    # バランス（~$17/1.8万件）

# 並列数を調整（デフォルト: 5、レート制限が頻発する場合は下げる）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --concurrency 3

# Embedding 品質検証を有効化（デフォルト: スキップ。品質重視の場合に使用）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --embedding-check

# 特定キーワードに関連する論文だけで学習データを作成（FT テスト用）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts \
  --query "generative AI genomics" \
  --query "Alternative Polyadenylation APA"

# キーワード検索のヒット数を増やす（デフォルト: 100 件/キーワード）
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts \
  --query "single-cell RNA-seq" --top-k 200

# 中断した場合は --resume で前回の続きから再開
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --resume

# 既存データの品質検証のみ
npx tsx chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts --validate-only storage/ft-training-data/training_2026-04-01.jsonl
```

**実行結果の例（--query 指定時）:**

```text
[generate-ft-data] Step 1: Searching papers by queries: generative AI genomics, Alternative Polyadenylation APA
[paper-extractor] Searching for: "generative AI genomics" (top 100)
[paper-extractor] Found 100 papers, unique so far: 100
[paper-extractor] Searching for: "Alternative Polyadenylation APA" (top 100)
[paper-extractor] Found 100 papers, unique so far: 187
[paper-extractor] Total unique papers for queries: 187
[generate-ft-data] Using model: gpt-5-nano (temperature fixed by API)
[generate-ft-data] Concurrency: 5, Embedding check: skip
[generate-ft-data] Processing batch 1 (papers 1-5/187)
...

Done! Generated 561 training examples.
  Training data: storage/ft-training-data/training_2026-04-02.jsonl
  Metadata: storage/ft-training-data/training_2026-04-02_metadata.jsonl
  Processed: 187, Failed: 0
```

**実行結果の例（全件）:**

```text
[generate-ft-data] Step 1: Extracting all papers from Qdrant...
[paper-extractor] Total papers extracted: 18234
[generate-ft-data] Using model: gpt-5-nano (temperature fixed by API)
[generate-ft-data] Concurrency: 5, Embedding check: skip
[generate-ft-data] Processing batch 1 (papers 1-5/18234)
[ideal-query-generator] Generated (no validation): "CycSim context-aware simulator Bayesian optimization long-read..."
[query-synthesizer] Generated 3 synthetic queries for: Context-aware simulation enables sys...
[generate-ft-data] Processing batch 2 (papers 6-10/18234)
...
[training-data-formatter] Written 54702 training examples to storage/ft-training-data/training_2026-04-02.jsonl

Done! Generated 54702 training examples.
  Training data: storage/ft-training-data/training_2026-04-02.jsonl
  Metadata: storage/ft-training-data/training_2026-04-02_metadata.jsonl
  Processed: 18234, Failed: 0
```

**出力ファイル:**

- `storage/ft-training-data/training_YYYY-MM-DD.jsonl` — OpenAI FT 用学習データ（JSONL）
- `storage/ft-training-data/training_YYYY-MM-DD_metadata.jsonl` — DOI 紐付けメタデータ（検証用）

**パイプラインの流れ:**

1. Qdrant から全論文を抽出（scroll API）
2. 論文ごとに LLM で日本語 3 種の合成クエリを生成（キーワード / 質問 / タスク記述）
3. 論文ごとに理想の英語検索クエリを生成（15 語以内）
4. OpenAI FT 形式の JSONL にストリーミング書き出し
5. 品質検証（理想クエリで Qdrant 検索 → 対象論文ヒット率を測定）

**高速化オプション:**

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--concurrency <n>` | 5 | バッチ並列数。合成クエリ+理想クエリも論文内で並列生成 |
| `--embedding-check` | スキップ | 有効にすると理想クエリの Embedding 品質検証を実施（低速） |

**エラーハンドリング:**

- バッチごとにプログレスを自動保存。中断時は `--resume` で再開可能
- レート制限（429）検知時はエクスポネンシャルバックオフで最大 3 回リトライ（30s → 60s → 90s）
- 3 回リトライしても失敗した論文はスキップして処理を継続

### Step 5: ファインチューニングの実行（任意）

Step 4 で生成した学習データを使って、OpenAI Fine-tuning API でモデルを学習させる。

```bash
# 基本的な実行（デフォルト: gpt-4.1-nano）
npx tsx chapter6-biorxiv/rag/ft-pipeline/run-fine-tuning.ts \
  --training-file storage/ft-training-data/training_2026-04-01.jsonl

# モデルとサフィックスを指定
npx tsx chapter6-biorxiv/rag/ft-pipeline/run-fine-tuning.ts \
  --training-file storage/ft-training-data/training_2026-04-01.jsonl \
  --model gpt-4.1-nano-2025-04-14 \
  --suffix biorxiv-query

# 検証データ付きで実行
npx tsx chapter6-biorxiv/rag/ft-pipeline/run-fine-tuning.ts \
  --training-file storage/ft-training-data/training_2026-04-01.jsonl \
  --validation-file storage/ft-training-data/validation.jsonl

# 既存ジョブの状態確認
npx tsx chapter6-biorxiv/rag/ft-pipeline/run-fine-tuning.ts --status ftjob-xxxxxxxx

# ジョブのキャンセル
npx tsx chapter6-biorxiv/rag/ft-pipeline/run-fine-tuning.ts --cancel ftjob-xxxxxxxx
```

**実行結果の例:**

```text
[run-fine-tuning] Step 1: Uploading training file: storage/ft-training-data/training_2026-04-01.jsonl
[run-fine-tuning] Training file uploaded: file-abc123
[run-fine-tuning] Step 2: Creating fine-tuning job (model: gpt-4.1-nano-2025-04-14)
[run-fine-tuning] Job created: ftjob-xyz789

Fine-tuning job started!
  Job ID: ftjob-xyz789
  Model: gpt-4.1-nano-2025-04-14
  Training file: file-abc123

Polling every 30s. Press Ctrl+C to stop (job continues on OpenAI side).

[run-fine-tuning] Status changed: validating_files → running
[run-fine-tuning]   [info] Training started
...
[run-fine-tuning] Status changed: running → succeeded

=== Fine-tuning completed! ===
  Fine-tuned model: ft:gpt-4.1-nano-2025-04-14:your-org:biorxiv-query:xxxxxxxx
  Trained tokens: 4500000

To use this model, set the environment variable:
  OPENAI_FAST_MODEL=ft:gpt-4.1-nano-2025-04-14:your-org:biorxiv-query:xxxxxxxx
```

完了後は環境変数 `OPENAI_FAST_MODEL` に FT 済みモデル名を設定するだけで、エージェントのクエリ拡張に反映される。

### Step 6: LangGraph Studio で実行

```bash
cd chapter6-biorxiv
npx @langchain/langgraph-cli dev
```

## ディレクトリ構成

```
chapter6-biorxiv/
├── docker-compose.yml           # Qdrant サーバー
├── models.ts                    # BiorxivPaper 等の型定義（Zod）
├── configs.ts                   # 設定 & LLM ファクトリ
├── custom-logger.ts             # ロガー
├── langgraph.json               # LangGraph Studio 設定
├── agent/
│   ├── research-agent.ts        # メインオーケストレーター
│   ├── paper-search-agent.ts    # RAG 検索 + 並列分析
│   └── paper-analyzer-agent.ts  # 個別論文分析
├── chains/
│   ├── hearing-chain.ts         # ヒアリング
│   ├── goal-optimizer-chain.ts  # ゴール最適化
│   ├── query-decomposer-chain.ts # クエリ分解
│   ├── paper-processor-chain.ts # RAG 検索 → PDF 変換
│   ├── reading-chains.ts        # 論文読解
│   ├── task-evaluator-chain.ts  # タスク評価
│   ├── reporter-chain.ts        # レポート生成
│   ├── utils.ts                 # ユーティリティ
│   └── prompts/                 # プロンプトテンプレート（10ファイル）
├── rag/
│   ├── biorxiv-fetcher.ts       # Step A: bioRxiv API → JSONL 保存（逐次追記）
│   ├── dedup-jsonl.ts           # JSONL 重複除去（DOI ごとに最新バージョンのみ残す）
│   ├── qdrant-loader.ts         # Step B: JSONL → Qdrant 投入（ストリーム読み込み）
│   ├── qdrant-store.ts          # Qdrant クライアント
│   ├── rag-searcher.ts          # RAG 検索 + リランキング
│   └── ft-pipeline/             # FT 学習データ生成 & ファインチューニング
│       ├── generate-ft-data.ts  # 学習データ生成 CLI
│       ├── run-fine-tuning.ts   # ファインチューニング実行 CLI
│       ├── paper-extractor.ts   # Qdrant から論文抽出（全件 / キーワード検索）
│       ├── query-synthesizer.ts # 合成クエリ生成（LLM）
│       ├── ideal-query-generator.ts # 理想クエリ生成（Embedding 検証付き）
│       ├── training-data-formatter.ts # JSONL 整形
│       └── validation.ts        # 品質検証
├── searcher/
│   └── searcher.ts              # Searcher インターフェース
├── service/
│   ├── pdf-to-text.ts           # pdf-parse でローカル PDF 変換
│   ├── markdown-storage.ts      # ファイル I/O
│   └── markdown-parser.ts       # セクション抽出
└── storage/
    ├── biorxiv-tmp/             # Step 1 で取得した JSONL の保存先
    └── markdown/                # 変換済みテキスト保存先
```

## chapter6（arXiv版）との主な違い

| 項目 | chapter6（arXiv） | chapter6-biorxiv |
|------|------------------|------------------|
| 論文ソース | arXiv API（キーワード検索） | bioRxiv API → Qdrant RAG |
| 検索方式 | arXiv API の全文検索 | ベクトル類似度検索（RAG） |
| リランキング | Cohere rerank API | OpenAI Embeddings + コサイン類似度 |
| PDF 変換 | Jina Reader API | pdf-parse（ローカル） |
| LLM | OpenAI + Claude Sonnet 4 | OpenAI のみ |
| 必要な API キー | OpenAI, Cohere, Jina | OpenAI のみ |
| 事前データ取り込み | 不要 | 必要（Qdrant へ格納） |

## 環境変数一覧

| 変数名 | 必須 | デフォルト | 説明 |
|--------|------|-----------|------|
| `OPENAI_API_KEY` | ✅ | - | OpenAI API キー |
| `OPENAI_SMART_MODEL` | - | `gpt-4o` | 高品質推論用モデル |
| `OPENAI_FAST_MODEL` | - | `gpt-4o-mini` | 高速処理用モデル |
| `OPENAI_REPORTER_MODEL` | - | `gpt-4o` | レポート生成用モデル |
| `EMBEDDING_MODEL` | - | `text-embedding-3-small` | エンベディングモデル |
| `QDRANT_URL` | - | `http://localhost:6333` | Qdrant サーバー URL |
| `QDRANT_COLLECTION_NAME` | - | `biorxiv-bioinformatics` | Qdrant コレクション名 |
| `BIORXIV_CATEGORY` | - | `bioinformatics` | bioRxiv カテゴリフィルタ |
| `MAX_SEARCH_RESULTS` | - | `20` | RAG 検索の取得件数 |
| `MAX_PAPERS` | - | `3` | 深掘り分析する論文数 |
| `DEBUG` | - | `false` | デバッグログ出力 |
