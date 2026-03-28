# Chapter 6-bioRxiv: bioRxiv 論文 RAG 知識ベース + リサーチ AI エージェント

bioRxiv の bioinformatics 分野の論文を RAG（Retrieval-Augmented Generation）で知識ベース化し、質問に応じて論文を検索・分析・レポート生成する AI エージェント。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│              データ取り込みパイプライン（2ステップ）         │
│                                                     │
│  Step A: bioRxiv API ──→ JSON ファイル保存（tmp）      │
│          (日付+カテゴリ)    (タイトル+Abstract+メタデータ)  │
│                                                     │
│  Step B: JSON 読み込み ──→ OpenAI Embeddings          │
│                           ──→ Chroma 格納             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                リサーチ AI エージェント（3層構造）          │
│                                                     │
│  ResearchAgent（メインオーケストレーター）                │
│   ├─ ヒアリング → ゴール最適化 → クエリ分解              │
│   │                                                 │
│   ├─ PaperSearchAgent（検索＆並列分析）                │
│   │   ├─ Chroma RAG 検索                            │
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
| ベクトルDB | Chroma（ローカル） |
| リランキング | OpenAI Embeddings + コサイン類似度 |
| PDF変換 | pdf-parse（ローカル） |
| フレームワーク | LangGraph |

## セットアップ

### 1. 環境変数

```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Chroma サーバー起動

```bash
docker run -d -p 8000:8000 chromadb/chroma
```

### 3. 依存パッケージ（プロジェクトルートで実行）

```bash
pnpm install
```

## 使い方

### Step 1: bioRxiv 論文メタデータの取得

bioRxiv API から bioinformatics 分野の論文メタデータを取得し、JSON ファイルとして保存する。

```bash
# 直近1週間分を取得する例
npx tsx chapter6-biorxiv/rag/biorxiv-fetcher.ts --start 2025-03-01 --end 2025-03-07

# カテゴリや出力先を指定する場合
npx tsx chapter6-biorxiv/rag/biorxiv-fetcher.ts --start 2025-01-01 --end 2025-03-28 --category bioinformatics --output storage/biorxiv-tmp
```

JSON ファイルは `storage/biorxiv-tmp/` に保存される。bioRxiv API は 100 件/リクエストでページネーションされる。bioinformatics カテゴリは約 40,000 件以上あるため、まずは短い日付範囲から始めることを推奨。

### Step 2: Chroma にデータ投入

Step 1 で保存した JSON ファイルを読み込み、Chroma ベクトルDB に投入する。

```bash
# JSON ファイルを指定して投入
npx tsx chapter6-biorxiv/rag/chroma-loader.ts --input storage/biorxiv-tmp/biorxiv_2025-03-01_2025-03-07_*.json

# バッチサイズを指定（デフォルト: 50）
npx tsx chapter6-biorxiv/rag/chroma-loader.ts --input storage/biorxiv-tmp/biorxiv_2025-03-01_2025-03-07_*.json --batch-size 30
```

重複チェック付きなので、同じ JSON を再投入しても二重登録されない。

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
│   ├── biorxiv-fetcher.ts       # Step A: bioRxiv API → JSON 保存
│   ├── chroma-loader.ts         # Step B: JSON → Chroma 投入
│   ├── chroma-store.ts          # Chroma クライアント
│   └── rag-searcher.ts          # RAG 検索 + リランキング
├── searcher/
│   └── searcher.ts              # Searcher インターフェース
├── service/
│   ├── pdf-to-text.ts           # pdf-parse でローカル PDF 変換
│   ├── markdown-storage.ts      # ファイル I/O
│   └── markdown-parser.ts       # セクション抽出
└── storage/
    ├── biorxiv-tmp/             # Step 1 で取得した JSON の保存先
    └── markdown/                # 変換済みテキスト保存先
```

## chapter6（arXiv版）との主な違い

| 項目 | chapter6（arXiv） | chapter6-biorxiv |
|------|------------------|------------------|
| 論文ソース | arXiv API（キーワード検索） | bioRxiv API → Chroma RAG |
| 検索方式 | arXiv API の全文検索 | ベクトル類似度検索（RAG） |
| リランキング | Cohere rerank API | OpenAI Embeddings + コサイン類似度 |
| PDF 変換 | Jina Reader API | pdf-parse（ローカル） |
| LLM | OpenAI + Claude Sonnet 4 | OpenAI のみ |
| 必要な API キー | OpenAI, Cohere, Jina | OpenAI のみ |
| 事前データ取り込み | 不要 | 必要（Chroma へ格納） |

## 環境変数一覧

| 変数名 | 必須 | デフォルト | 説明 |
|--------|------|-----------|------|
| `OPENAI_API_KEY` | ✅ | - | OpenAI API キー |
| `OPENAI_SMART_MODEL` | - | `gpt-4o` | 高品質推論用モデル |
| `OPENAI_FAST_MODEL` | - | `gpt-4o-mini` | 高速処理用モデル |
| `OPENAI_REPORTER_MODEL` | - | `gpt-4o` | レポート生成用モデル |
| `EMBEDDING_MODEL` | - | `text-embedding-3-small` | エンベディングモデル |
| `CHROMA_COLLECTION_NAME` | - | `biorxiv-bioinformatics` | Chroma コレクション名 |
| `CHROMA_PERSIST_DIRECTORY` | - | `storage/chroma` | Chroma 永続化ディレクトリ |
| `BIORXIV_CATEGORY` | - | `bioinformatics` | bioRxiv カテゴリフィルタ |
| `MAX_SEARCH_RESULTS` | - | `20` | RAG 検索の取得件数 |
| `MAX_PAPERS` | - | `3` | 深掘り分析する論文数 |
| `DEBUG` | - | `false` | デバッグログ出力 |
