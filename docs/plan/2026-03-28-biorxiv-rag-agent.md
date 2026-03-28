# Chapter6-bioRxiv: bioRxiv 論文 RAG 知識ベース + リサーチ AI エージェント

## Context

現在の chapter6 は arXiv 論文を検索・分析する AI エージェント。ユーザーは bioRxiv（特に bioinformatics 分野）の論文を対象に、RAG 知識ベースを構築して AI の知識として組み込みたい。

**bioRxiv API の制約**: キーワード検索機能がない（日付範囲 + カテゴリフィルタのみ）。そのため、bioRxiv API で bioinformatics 論文を全件取得 → タイトル＋アブストラクトを Chroma（ベクトルDB）に格納 → RAG で検索するアプローチを採用する。

**設計方針**: 外部サービス依存を最小限にし、OpenAI API のみ使用。Reranking・PDF変換もローカルで自前実装する。

## 技術スタック

| コンポーネント | 選定 |
|------------|------|
| ベクトルDB | **Chroma**（ローカル、chromadb パッケージ） |
| Embedding | **OpenAI text-embedding-3-small** |
| LLM | **GPT-4o**（高品質推論）/ **GPT-4o-mini**（高速処理・レポート生成含む全て） |
| Reranking | **OpenAI Embeddings + コサイン類似度**（ローカル計算、Cohere 不使用） |
| PDF変換 | **pdf-parse**（ローカル、Jina Reader 不使用） |
| フレームワーク | LangGraph（chapter6 と同じ） |

## ディレクトリ構成

```
packages/@ai-suburi/core/chapter6-biorxiv/
├── models.ts                          # BiorxivPaper, RAG関連の型定義
├── configs.ts                         # 設定 & LLMファクトリ（OpenAIのみ）
├── custom-logger.ts                   # ロガー（chapter5から再利用）
├── langgraph.json                     # LangGraph Studio設定
├── agent/
│   ├── research-agent.ts              # メインオーケストレーター
│   ├── paper-search-agent.ts          # RAG検索 + 並列分析
│   └── paper-analyzer-agent.ts        # 個別論文分析（chapter6と同一）
├── chains/
│   ├── hearing-chain.ts               # ヒアリング（chapter6と同一）
│   ├── goal-optimizer-chain.ts        # ゴール最適化（プロンプト変更）
│   ├── query-decomposer-chain.ts      # クエリ分解（chapter6と同一）
│   ├── paper-processor-chain.ts       # RAG検索→PDF変換パイプライン ★変更大
│   ├── reading-chains.ts              # 論文読解（chapter6と同一）
│   ├── task-evaluator-chain.ts        # 評価（chapter6と同一）
│   ├── reporter-chain.ts              # レポート生成（OpenAI GPT-4oに変更）
│   ├── utils.ts                       # ユーティリティ
│   └── prompts/                       # プロンプトテンプレート（3ファイル変更）
├── rag/                               # ★新規：RAGモジュール
│   ├── biorxiv-ingester.ts            # bioRxiv API → Chroma 取り込みパイプライン
│   ├── chroma-store.ts                # Chroma クライアントラッパー
│   └── rag-searcher.ts               # RAG検索（Searcher インターフェース実装）
├── searcher/
│   └── searcher.ts                    # Searcher インターフェース定義
├── service/
│   ├── pdf-to-text.ts                 # ★変更：pdf-parse でローカルPDF→テキスト変換
│   ├── markdown-storage.ts            # ファイルI/O（chapter6と同一）
│   └── markdown-parser.ts             # セクション抽出（chapter6と同一）
├── storage/                           # テキスト出力ディレクトリ
└── test/                              # テスト
```

## 実装の全体フロー

### フロー1: データ取り込みパイプライン（事前実行）

```
bioRxiv API (details/biorxiv/{date-range}?category=bioinformatics)
    ↓ 100件ずつページネーション
    ↓ 全 bioinformatics 論文を取得
    ↓
各論文のメタデータ取得
    ├── DOI, タイトル, 著者, アブストラクト, カテゴリ, 公開日
    ↓
OpenAI Embeddings (text-embedding-3-small)
    ├── タイトル + アブストラクト を結合してエンベディング
    ↓
Chroma に格納
    ├── embedding: ベクトル
    ├── document: タイトル + アブストラクト
    └── metadata: { doi, title, authors, published, category, pdfLink }
```

### フロー2: リサーチ AI エージェント（ユーザーが質問時）

```
ユーザーの質問
    ↓
[ResearchAgent: user_hearing] → ヒアリング
    ↓
[goal_setting] → ゴール最適化
    ↓
[decompose_query] → 3-5個のサブタスクに分解
    ↓
[paper_search_agent] (PaperSearchAgent)
    ├── [search_papers] (PaperProcessor)
    │   ├── 各サブタスク → RAG検索（Chroma類似度検索）
    │   ├── OpenAI Embeddings + コサイン類似度でリランキング
    │   ├── 上位論文のPDFを pdf-parse でテキスト変換
    │   └── ReadingResult 作成
    │
    ├── [analyze_paper] (並列: PaperAnalyzerAgent × N)
    │   ├── テキストセクション選択 → 十分性チェック → 要約生成
    │   └── chapter6 と完全に同じフロー
    │
    └── [organize_results] → 関連論文のみフィルタ
    ↓
[evaluate_task] → 情報十分性チェック（不足なら再検索）
    ↓
[generate_report] → 最終レポート生成（GPT-4o）
```

## 実装フェーズ

### Phase 1: 基盤（モデル・設定・インターフェース）

**1-1. `models.ts` を作成**
- `ArxivPaper` → `BiorxivPaper` に置き換え
  - `id` → `doi` (string)
  - `link` → `https://doi.org/{doi}`
  - `pdfLink` → `https://www.biorxiv.org/content/{doi}v{version}.full.pdf`
  - `categories: string[]` → `category: string`（bioRxiv は単一カテゴリ）
  - `relevanceScore` はコサイン類似度スコアに変更
- `ReadingResult` の `paper` フィールドを `BiorxivPaper` に変更
- `biorxivPaperToXml()` ヘルパー追加
- 他の型（Section, Hearing, DecomposedTasks 等）はそのまま

参照: `/packages/@ai-suburi/core/chapter6/models.ts`

**1-2. `configs.ts` を作成**
- chapter6 からコピー、以下を変更・追加:
  - `createReporterLlm()`: Claude Sonnet 4 → **GPT-4o** に変更（OpenAIのみ方針）
  - `chromaCollectionName`: `'biorxiv-bioinformatics'`
  - `chromaPersistDirectory`: `'./storage/chroma'`
  - `embeddingModel`: `'text-embedding-3-small'`
  - `biorxivCategory`: `'bioinformatics'`
  - `ingestionBatchSize`: `100`（bioRxiv API の1ページサイズ）
  - Cohere 関連の設定を全て削除

参照: `/packages/@ai-suburi/core/chapter6/configs.ts`

**1-3. `searcher/searcher.ts` を作成**
- `Searcher` インターフェース: `run(goal, query) => Promise<BiorxivPaper[]>`

参照: `/packages/@ai-suburi/core/chapter6/searcher/searcher.ts`

### Phase 2: RAG モジュール（★コア新規実装）

**2-1. `rag/chroma-store.ts` を作成（~100行）**
- Chroma クライアント初期化（永続化ディレクトリ指定）
- コレクション取得/作成
- `addDocuments(papers: BiorxivPaper[])`: OpenAI Embeddings でエンベディング → Chroma に格納
- `search(query: string, topK: number)`: Chroma 類似度検索で候補取得
- `rerank(query: string, papers: BiorxivPaper[], topN: number)`: OpenAI Embeddings でクエリと各論文のコサイン類似度を計算してリランキング
- コサイン類似度計算ユーティリティ関数

**2-2. `rag/biorxiv-ingester.ts` を作成（~150行）**
- bioRxiv API クライアント:
  - エンドポイント: `https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}`
  - カテゴリフィルタ: `?category=bioinformatics`
  - ページネーション: cursor を 0, 100, 200... とインクリメント
  - レスポンス JSON パース → `BiorxivPaper[]` に変換
  - 1リクエスト/秒のウェイト（レート制限対策）
- 取り込みオーケストレーション:
  - 日付範囲を指定して全件取得
  - バッチで Chroma に格納
  - 進捗ログ出力
  - 重複チェック（DOI ベース）
- CLI エントリポイント: `npx tsx rag/biorxiv-ingester.ts --start 2024-01-01 --end 2024-12-31`

**2-3. `rag/rag-searcher.ts` を作成（~120行）**
- `Searcher` インターフェース実装
- 検索フロー:
  1. LLM（GPT-4o-mini）でクエリ最適化（ゴール + サブタスク → 検索クエリ）
  2. Chroma 類似度検索（topK=20）
  3. OpenAI Embeddings + コサイン類似度でリランキング（上位 3 件に絞り込み）
  4. `BiorxivPaper[]` を返却
- リトライロジック（結果 0 件時にクエリ再生成）

### Phase 3: サービス（PDF変換のローカル化）

**3-1. `service/pdf-to-text.ts` を作成（~80行）★新規**
- chapter6 の `pdf-to-markdown.ts`（Jina Reader）を置き換え
- `pdf-parse` パッケージを使用:
  - PDFのURLからダウンロード（`fetch` でバイナリ取得）
  - `pdf-parse` でテキスト抽出
  - テキストをMarkdown風に整形（段落分割）
- ローカルキャッシュ機能（同じPDFの再変換を防止）
- Jina Reader API キー不要

**3-2. chapter6 からそのままコピー**
- `service/markdown-storage.ts`
- `service/markdown-parser.ts`

### Phase 4: チェーン＆プロンプト

**4-1. chapter6 からコピー（変更なし）**
- `hearing-chain.ts`
- `reading-chains.ts`（SetSection, CheckSufficiency, Summarizer）
- `task-evaluator-chain.ts`
- `query-decomposer-chain.ts`
- `utils.ts`

**4-2. chapter6 からコピー（変更あり）**
- `goal-optimizer-chain.ts`: 型名のみ変更
- `reporter-chain.ts`: `createReporterLlm()` が GPT-4o を返すので、Anthropic SDK の呼び出しを OpenAI SDK に変更
- `paper-processor-chain.ts`: ★変更大
  - `ArxivSearcher` → `RagSearcher` に差し替え
  - `ArxivPaper` → `BiorxivPaper`
  - PDF変換: `PdfToMarkdown`（Jina）→ `PdfToText`（pdf-parse）に差し替え

**4-3. プロンプト変更（3ファイル）**
- `goal_optimizer_conversation.prompt`: "arXiv" → "bioRxiv（bioinformatics分野）"
- `goal_optimizer_search.prompt`: 同上
- `reporter_user.prompt`: "arXiv論文ID" → "bioRxiv DOI"

参照: `/packages/@ai-suburi/core/chapter6/chains/prompts/`

### Phase 5: エージェント

**5-1. chapter6 からコピー（軽微な変更）**
- `research-agent.ts`: `ArxivSearcher` → `RagSearcher` に差し替え、thread_id 変更
- `paper-search-agent.ts`: 型名変更のみ
- `paper-analyzer-agent.ts`: 変更なし

参照: `/packages/@ai-suburi/core/chapter6/agent/`

### Phase 6: その他

- `custom-logger.ts`: chapter6 からコピー
- `langgraph.json`: graph パス更新
- `README.md`: セットアップ手順（Chroma 設定、データ取り込み方法、必要な環境変数）

## 変更が必要な既存ファイル

なし（chapter6 は一切変更しない）。全て新規作成。

## 必要な新規パッケージ

```
chromadb          # Chroma ベクトルDB クライアント
pdf-parse         # ローカルPDFテキスト抽出
```

※ OpenAI SDK は既に依存関係にあるため追加不要。
※ Cohere SDK は不要（削除対象）。
※ Jina Reader API キーは不要。

## 必要な環境変数

```
OPENAI_API_KEY    # OpenAI API（LLM + Embeddings 共通）
```

※ chapter6 で必要だった `COHERE_API_KEY`、`JINA_API_KEY` は不要。

## 検証方法

1. **データ取り込みテスト**: `npx tsx rag/biorxiv-ingester.ts --start 2025-03-01 --end 2025-03-07` で1週間分を取り込み、Chroma にデータが格納されることを確認
2. **RAG 検索テスト**: 「single-cell RNA-seq analysis methods」等のクエリで類似度検索し、関連論文が返ることを確認
3. **リランキングテスト**: コサイン類似度スコアが適切に計算され、関連度順にソートされることを確認
4. **PDF変換テスト**: bioRxiv の論文PDFを pdf-parse でテキスト抽出し、セクション分割が正しく動作することを確認
5. **エージェント E2E テスト**: LangGraph Studio でリサーチエージェントを起動し、bioinformatics に関する質問を投げてレポートが生成されることを確認

## 注意点

- bioRxiv API はページあたり100件で、bioinformatics カテゴリは約40,000件以上ある → 全件取り込みは時間がかかるため、まずは直近数ヶ月分など範囲を絞って開始
- bioRxiv API のレート制限は明文化されていないが、1リクエスト/秒程度のウェイトを入れるのが安全
- PDF リンクの version は bioRxiv API レスポンスから取得可能
- pdf-parse はテキスト抽出のみで、画像・表は取得できない → 学術論文では多くの情報がテキストに含まれるため実用上問題は少ない
- OpenAI Embeddings のバッチサイズに注意（一度に大量送信しない）
