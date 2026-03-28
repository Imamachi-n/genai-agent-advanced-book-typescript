# Chapter 6-bioRxiv: bioRxiv 論文 RAG 知識ベース + リサーチ AI エージェント

bioRxiv の bioinformatics 分野の論文を RAG（Retrieval-Augmented Generation）で知識ベース化し、質問に応じて論文を検索・分析・レポート生成する AI エージェント。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│                  データ取り込みパイプライン                │
│                                                     │
│  bioRxiv API ──→ メタデータ取得 ──→ OpenAI Embeddings │
│  (日付+カテゴリ)   (タイトル+Abstract)  ──→ Chroma 格納  │
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

### Step 1: bioRxiv 論文の取り込み

bioRxiv API から bioinformatics 分野の論文を取得し、Chroma に格納する。

```bash
# 直近1週間分を取り込む例
npx tsx chapter6-biorxiv/rag/biorxiv-ingester.ts --start 2025-03-01 --end 2025-03-07

# カテゴリを指定する場合（デフォルト: bioinformatics）
npx tsx chapter6-biorxiv/rag/biorxiv-ingester.ts --start 2025-01-01 --end 2025-03-28 --category bioinformatics
```

bioRxiv API は 100 件/リクエストでページネーションされる。bioinformatics カテゴリは約 40,000 件以上あるため、まずは短い日付範囲から始めることを推奨。

### Step 2: リサーチエージェント実行

```bash
# CLI で実行（ヒアリングあり）
npx tsx chapter6-biorxiv/agent/research-agent.ts "single-cell RNA-seq解析の最新手法について調べる"

# ヒアリングをスキップして即実行
npx tsx chapter6-biorxiv/agent/research-agent.ts "CRISPR スクリーニングのバイオインフォマティクス解析" --skip-feedback
```

### Step 3: LangGraph Studio で実行

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
│   ├── biorxiv-ingester.ts      # bioRxiv API → Chroma 取り込み
│   ├── chroma-store.ts          # Chroma クライアント
│   └── rag-searcher.ts          # RAG 検索 + リランキング
├── searcher/
│   └── searcher.ts              # Searcher インターフェース
├── service/
│   ├── pdf-to-text.ts           # pdf-parse でローカル PDF 変換
│   ├── markdown-storage.ts      # ファイル I/O
│   └── markdown-parser.ts       # セクション抽出
└── storage/
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
