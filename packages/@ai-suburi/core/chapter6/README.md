# 現場で活用するためのAIエージェント実践入門 - Chapter 6（TypeScript版）

このディレクトリは、書籍「現場で活用するためのAIエージェント実践入門」（講談社）の第6章に関連するTypeScript版のソースコードとリソースを含んでいます。

## 前提条件

このプロジェクトを実行するには、以下の準備が必要です：

- Node.js 20 以上
- OpenAI APIキー
- Cohere APIキー
- Jina Reader APIキー
- LangGraph CLI（LangGraph Studioを使用する場合）

また、依存関係は `packages/@ai-suburi/core/package.json` に記載されています。

## 環境変数の設定

プロジェクトルートの `.envrc.sample` ファイルを `.envrc` にコピーして、必要な環境変数を設定します：

```bash
cp .envrc.sample .envrc
```

`.envrc` ファイルで以下の項目を設定してください：

- `OPENAI_API_KEY`: OpenAI API キー
- `COHERE_API_KEY`: Cohere API キー
- `JINA_API_KEY`: Jina Reader API キー

### オプションの環境変数

| 環境変数 | 説明 | デフォルト値 |
|---------|------|------------|
| `OPENAI_SMART_MODEL` | 複雑なタスク用のOpenAIモデル | `gpt-4o` |
| `OPENAI_FAST_MODEL` | 高速・軽量なタスク用のOpenAIモデル | `gpt-4o-mini` |
| `OPENAI_REPORTER_MODEL` | レポート生成用のOpenAIモデル | `gpt-4o` |
| `COHERE_RERANK_MODEL` | リランキング用のCohereモデル | `rerank-multilingual-v3.0` |
| `TEMPERATURE` | 生成時の温度パラメータ | `0` |
| `MAX_EVALUATION_RETRY_COUNT` | 追加情報取得の最大リトライ回数 | `3` |
| `MIN_DECOMPOSED_TASKS` | タスク分解時の最小タスク数 | `3` |
| `MAX_DECOMPOSED_TASKS` | タスク分解時の最大タスク数 | `5` |
| `MAX_SEARCH_RETRIES` | 検索失敗時の最大リトライ回数 | `3` |
| `MAX_SEARCH_RESULTS` | 1回の検索で取得する最大論文数 | `10` |
| `MAX_PAPERS` | 詳細分析する最大論文数 | `3` |
| `MAX_WORKERS` | 並列処理数 | `3` |
| `MAX_RECURSION_LIMIT` | ノードの最大実行回数制限 | `1000` |

## LangGraph Studioでの実行（オプション）

LangGraph Studio は LangGraph 公式のビジュアルデバッグツールで、以下の機能をローカル環境で利用できます：

- グラフ構造のビジュアル表示（ノードとエッジの図）
- ノードの実行順をリアルタイム追跡
- タイムトラベルデバッグ（任意のノードに巻き戻して再実行）
- コード変更のホットリロード

Docker 不要で、完全にローカルで動作します。

### 前準備

LangGraph Studio の Web UI を利用するには、LangSmith の無料アカウントが必要です。

1. [smith.langchain.com](https://smith.langchain.com) でアカウントを作成（Google / GitHub ログイン対応）
2. Settings > API Keys から API キーを取得し、`.envrc` に設定

```bash
export LANGSMITH_API_KEY="lsv2_..."
```

### 起動方法

chapter6 ディレクトリに移動して `langgraph dev` コマンドを実行します：

```bash
cd packages/@ai-suburi/core/chapter6
npx @langchain/langgraph-cli dev
```

ブラウザに LangGraph Studio の Web UI が自動的に開きます。初回はLangSmithへのログインが求められるので、作成したアカウントでログインしてください。

グラフ構造がビジュアル表示され、UI 上からリサーチゴールを入力してエージェントを実行すると、各ノードの実行状態をリアルタイムで追跡できます。

## ディレクトリ構成

```
chapter6/
├── README.md                          # このファイル
├── langgraph.json                     # LangGraph Studio設定ファイル
├── configs.ts                         # 設定管理・LLMインスタンス生成
├── models.ts                          # Zodスキーマ定義
├── custom-logger.ts                   # ロガー設定
├── agent/                             # エージェント実装
│   ├── research-agent.ts              # リサーチエージェント（メイン）
│   ├── paper-search-agent.ts          # 論文検索エージェント
│   └── paper-analyzer-agent.ts        # 論文分析エージェント
├── chains/                            # LangChainチェーン実装
│   ├── hearing-chain.ts               # ヒアリングチェーン
│   ├── goal-optimizer-chain.ts        # ゴール最適化チェーン
│   ├── query-decomposer-chain.ts      # クエリ分解チェーン
│   ├── task-evaluator-chain.ts        # タスク評価チェーン
│   ├── paper-processor-chain.ts       # 論文処理チェーン
│   ├── reading-chains.ts             # 論文読解チェーン
│   ├── reporter-chain.ts             # レポート生成チェーン
│   ├── utils.ts                       # ユーティリティ関数
│   └── prompts/                       # プロンプトテンプレート
│       ├── check_sufficiency.prompt
│       ├── goal_optimizer_conversation.prompt
│       ├── goal_optimizer_search.prompt
│       ├── hearing.prompt
│       ├── query_decomposer.prompt
│       ├── reporter_system.prompt
│       ├── reporter_user.prompt
│       ├── set_section.prompt
│       ├── summarize.prompt
│       └── task_evaluator.prompt
├── searcher/                          # 検索機能
│   ├── arxiv-searcher.ts              # arXiv検索実装
│   └── searcher.ts                    # 検索インターフェース
├── service/                           # サービス層
│   ├── markdown-parser.ts             # Markdownパーサー
│   ├── markdown-storage.ts            # Markdownストレージ
│   └── pdf-to-markdown.ts            # PDF→Markdown変換
├── test/                              # テストファイル
│   ├── models.test.ts                 # モデルテスト
│   └── markdown-parser.test.ts        # パーサーテスト
├── fixtures/                          # テスト用データ
│   ├── 2408.14317.md                  # サンプル論文（Markdown形式）
│   ├── sample_context.txt             # サンプルコンテキスト
│   └── sample_goal.txt                # サンプルゴール
└── storage/                           # ストレージ
    └── markdown/                      # 論文のMarkdownファイル保存先
```

## 主要な機能

### arXiv論文リサーチャー

このエージェントは、arXivから関連論文を検索し、PDFファイルをMarkdown形式に変換して構造化された分析を行います。ヒアリング機能でユーザーの研究目的を対話的に明確化し、収集した情報を統合してレポートを生成します。

## 動作確認で使用している論文

```bibtex
@article{dmonte2024claim,
  title={Claim Verification in the Age of Large Language Models: A Survey},
  author={Dmonte, Alphaeus and Oruche, Roland and Zampieri, Marcos and Calyam, Prasad and Augenstein, Isabelle},
  journal={arXiv preprint arXiv:2408.14317},
  year={2024}
}
```

## APIキーの取得方法

### OpenAI APIキーの取得方法

[platform.openai.com](https://platform.openai.com)にアクセスしてアカウントを作成します。ログイン後、Dashboard > API keysからAPIキーを生成できます。APIを使用するには支払い方法の設定とクレジットの購入が必要です。

### Cohere APIキーの取得方法

[dashboard.cohere.com/welcome/register](https://dashboard.cohere.com/welcome/register)でアカウントを作成します。ログイン後、[dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)でAPIキーを確認できます。トライアルキーは無料で使用できますが、レート制限があります。

### Jina Reader APIキーの取得方法

[jina.ai/api-dashboard/](https://jina.ai/api-dashboard/)でアカウントを作成すると、自動的にAPIキーが生成されます。新規ユーザーには1,000万トークンの無料トライアルが提供されます。
