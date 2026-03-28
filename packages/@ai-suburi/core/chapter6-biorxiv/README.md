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

### Step 2: Qdrant にデータ投入

Step 1 で保存した JSONL ファイルを行単位でストリーム読み込みし、Qdrant ベクトルDB に投入する。大量データでもメモリを圧迫しない。

```bash
# JSONL ファイルを指定して投入
npx tsx chapter6-biorxiv/rag/qdrant-loader.ts --input storage/biorxiv-tmp/biorxiv_2025-03-01_2025-03-07_*.jsonl

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

### Step 4: LangGraph Studio で実行

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
│   ├── qdrant-loader.ts         # Step B: JSONL → Qdrant 投入（ストリーム読み込み）
│   ├── qdrant-store.ts          # Qdrant クライアント
│   └── rag-searcher.ts          # RAG 検索 + リランキング
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
