# TypeScript版 現場で活用するためのAIエージェント実践入門

> 📖 本リポジトリは [genai-agent-advanced-book](https://github.com/masamasa59/genai-agent-advanced-book) のサンプルコードを Python から TypeScript へ変換・再実装したものです。

## ディレクトリ構成

本プロジェクトはpnpmワークスペースによるモノレポ構成です。

```plaintext
/
├── packages/
│   └── @ai-suburi/
│       ├── core/                    # サンプルコード (@ai-suburi/core)
│       │   ├── chapter3/            # 第3章のサンプル
│       │   ├── chapter4/            # 第4章のサンプル
│       │   ├── chapter5/            # 第5章のサンプル
│       │   ├── chapter6/            # 第6章のサンプル
│       │   ├── package.json
│       │   └── tsconfig.json
│       ├── docs/                    # Docusaurus ドキュメント (@ai-suburi/docs)
│       │   ├── docs/
│       │   ├── src/
│       │   ├── docusaurus.config.ts
│       │   └── package.json
│       └── bedrock-agentcore-cdk/   # AgentCore CDK サンプル (@ai-suburi/bedrock-agentcore-cdk)
│           ├── agent/               # エージェントアプリ（Docker コンテナ）
│           ├── lib/
│           ├── bin/
│           └── package.json
├── pnpm-workspace.yaml              # ワークスペース設定
├── package.json                     # ルート設定
└── tsconfig.json                    # 共通TypeScript設定
```

## 開発環境

本プロジェクトでは以下のツールを使用しています。

| カテゴリ | ツール | 説明 |
| --- | --- | --- |
| 共通 | [pnpm](https://pnpm.io/) | パッケージマネージャー（モノレポ対応） |
| | [tsx](https://www.npmjs.com/package/tsx) | TypeScript ファイルの直接実行 |
| | [Biome](https://biomejs.dev/) | リンター・フォーマッター |
| | [secretlint](https://github.com/secretlint/secretlint) | シークレット検出ツール（API キーの誤コミット防止） |
| | [husky](https://typicode.github.io/husky/) | Git hooks 管理（pre-commit で secretlint を自動実行） |
| core | [OpenAI SDK](https://www.npmjs.com/package/openai) | OpenAI API クライアント |
| | [Anthropic SDK](https://www.npmjs.com/package/@anthropic-ai/sdk) | Anthropic Claude API クライアント |
| | [Google Gen AI SDK](https://www.npmjs.com/package/@google/genai) | Google Gemini API クライアント |
| | [LangChain](https://js.langchain.com/) | LLM アプリケーション開発フレームワーク |
| | [Tavily](https://tavily.com/) | AI エージェント向け Web 検索 API |
| | [Zod](https://zod.dev/) | スキーマバリデーションライブラリ |
| | [E2B Code Interpreter](https://e2b.dev/) | クラウドサンドボックスでの Python コード実行 |
| | [LangGraph](https://langchain-ai.github.io/langgraphjs/) | グラフベースのワークフロー制御 |
| | [Cohere](https://cohere.com/) | 論文リランキング API（chapter6） |
| | [Jina Reader](https://jina.ai/reader/) | PDF → Markdown 変換 API（chapter6） |
| | [LangGraph Studio](https://studio.langchain.com/) | グラフ構造の可視化・リアルタイムデバッグ（chapter6） |
| docs | [Docusaurus](https://docusaurus.io/) | ドキュメントサイト |
| | [Rspack](https://rspack.rs/) | Rust 製の高速バンドラ |
| | [SWC](https://swc.rs/) | Rust 製の高速トランスパイラ・ミニファイア |
| | [Lightning CSS](https://lightningcss.dev/) | Rust 製の高速 CSS パーサー・ミニファイア |
| bedrock-agentcore-cdk | [AWS CDK](https://aws.amazon.com/cdk/) | AWS インフラをコードで定義・デプロイ |
| | [esbuild](https://esbuild.github.io/) | 高速 JavaScript/TypeScript バンドラ |
| | [Hono](https://hono.dev/) | 軽量 Web フレームワーク（エージェントアプリで使用） |

## セットアップ

### direnvのインストール

direnvを使って環境変数を管理します。

#### macOS (Homebrew)

```zsh
brew install direnv
```

シェルにhookを追加します（zshの場合）。

```zsh
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
```

#### `.envrc` の設定

プロジェクトルートに `.envrc` ファイルを作成し、必要な環境変数を記述します。

```zsh
cp .envrc.sample .envrc
direnv allow
```

`.envrc` ファイルには以下の環境変数を設定します。

```zsh
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
export E2B_API_KEY="your-key"
export COHERE_API_KEY="your-key"
export JINA_API_KEY="your-key"
```

#### APIキーの取得方法

##### OPENAI_API_KEY

OpenAI の API キーは以下の手順で取得できます。

1. [OpenAI Platform](https://platform.openai.com/) にアクセスし、アカウントを作成またはログイン
2. 右上のアイコンから **Dashboard** に移動
3. 左メニューの **API keys** をクリック
4. **Create new secret key** をクリックしてキーを生成
5. 生成されたキー（`sk-proj-...` の形式）をコピーして `.envrc` の `OPENAI_API_KEY` に設定

> ⚠️ API キーは作成時に一度しか表示されません。必ずコピーして安全な場所に保管してください。
>
> ⚠️ APIの利用にはクレジットの購入（有料）が必要です。[Billing](https://platform.openai.com/settings/organization/billing/overview) ページからクレジットを追加してください。

##### GOOGLE_API_KEY

Google Gemini API のキーは以下の手順で取得できます。

1. [Google AI Studio](https://aistudio.google.com/) にアクセスし、Google アカウントでログイン
2. 左メニューまたはヘッダーの **Get API key** をクリック
3. **Create API key** をクリックしてキーを生成
4. 生成されたキーをコピーして `.envrc` の `GOOGLE_API_KEY` に設定

> ℹ️ 無料枠が用意されており、一定のレート制限内であれば無料で利用できます。

##### ANTHROPIC_API_KEY

Anthropic（Claude）の API キーは以下の手順で取得できます。

1. [Anthropic Console](https://console.anthropic.com/) にアクセスし、アカウントを作成またはログイン
2. 左メニューの **API Keys** をクリック
3. **Create Key** をクリックしてキーを生成
4. 生成されたキー（`sk-ant-...` の形式）をコピーして `.envrc` の `ANTHROPIC_API_KEY` に設定

> ⚠️ API キーは作成時に一度しか表示されません。必ずコピーして安全な場所に保管してください。
>
> ⚠️ API の利用にはクレジットの購入（有料）が必要です。[Plans & Billing](https://console.anthropic.com/settings/plans) ページからクレジットを追加してください。

##### TAVILY_API_KEY

Tavily は AI エージェント向けの Web 検索 API です。以下の手順でキーを取得できます。

1. [Tavily](https://tavily.com/) にアクセスし、アカウントを作成またはログイン
2. ログイン後、ダッシュボードに API キーが表示される
3. API キー（`tvly-...` の形式）をコピーして `.envrc` の `TAVILY_API_KEY` に設定

> ℹ️ 無料プラン（Free）では月 1,000 リクエストまで利用可能です。

##### E2B_API_KEY

E2B は AI エージェントが生成したコードをクラウド上のサンドボックスで安全に実行するサービスです。以下の手順でキーを取得できます。

1. [E2B](https://e2b.dev/) にアクセスし、アカウントを作成またはログイン
2. ダッシュボードの **API Keys** セクションで API キーを確認
3. API キーをコピーして `.envrc` の `E2B_API_KEY` に設定

> ℹ️ 無料枠が用意されており、一定時間のサンドボックス利用が無料で可能です。

##### COHERE_API_KEY

Cohere は自然言語処理 API を提供するサービスで、chapter6 の arXiv 論文リサーチャーでは論文のリランキング（関連度順の並び替え）に使用しています。以下の手順でキーを取得できます。

1. [Cohere Dashboard](https://dashboard.cohere.com/) にアクセスし、アカウントを作成またはログイン
2. 左メニューの **API Keys** をクリック
3. デフォルトで Trial key が発行されているので、そのキーをコピーして `.envrc` の `COHERE_API_KEY` に設定

> ℹ️ Trial キーではレート制限付きで無料利用が可能です。本番利用には Production キーへのアップグレードが必要です。

##### JINA_API_KEY

Jina AI は Reader API を提供しており、chapter6 の arXiv 論文リサーチャーでは PDF 論文を Markdown テキストに変換するために使用しています。以下の手順でキーを取得できます。

1. [Jina AI](https://jina.ai/) にアクセスし、アカウントを作成またはログイン
2. ダッシュボードの **API Keys** セクションで API キーを確認
3. API キー（`jina_...` の形式）をコピーして `.envrc` の `JINA_API_KEY` に設定

> ℹ️ 無料枠が用意されており、一定のリクエスト数まで無料で利用可能です。

##### LangGraph Studio（ビジュアルデバッグ / オプション）

LangGraph Studio は LangGraph 公式のビジュアルデバッグツールで、chapter6 のエージェントのグラフ構造をビジュアル表示し、ノードの実行順をリアルタイムで追跡できます。Docker 不要で、完全にローカルで動作します。

Web UI の利用には LangSmith の無料アカウントが必要です。[smith.langchain.com](https://smith.langchain.com) でアカウントを作成し、API キーを `.envrc` に設定してください。

```zsh
export LANGSMITH_API_KEY="lsv2_..."
```

```zsh
cd packages/@ai-suburi/core/chapter6
npx @langchain/langgraph-cli dev
```

ブラウザに LangGraph Studio の Web UI が自動的に開き、グラフ構造が表示されます。

> ℹ️ LangGraph Studio の利用は任意です。未使用でもエージェントは正常に動作します。

### pnpmのインストール

#### Homebrew

```zsh
brew install pnpm
```

#### npm

```zsh
npm install -g pnpm
```

### npmパッケージのインストール

```zsh
pnpm install
```

#### AgentCore エージェントアプリの依存インストール

`agent/` ディレクトリは Docker ビルド用に独立した `package.json` を持っているため、ワークスペースとは別にインストールが必要です。

```zsh
pnpm agent:install
```

## 使用方法

### ドキュメントサイト

```zsh
# 開発サーバー起動
pnpm dev:docs

# ビルド
pnpm build:docs
```

### サンプルコードの実行

```zsh
pnpm tsx chapter3/test3-1-chat-completions-api.ts
```

### CDK（Bedrock AgentCore）

```zsh
# CDK コマンド実行
pnpm cdk

# スタック合成
pnpm cdk:synth

# デプロイ
pnpm cdk:deploy

# 削除
pnpm cdk:destroy
```

### 特定パッケージでのコマンド実行

```zsh
# @ai-suburi/core パッケージ
pnpm --filter @ai-suburi/core <command>

# @ai-suburi/docs パッケージ
pnpm --filter @ai-suburi/docs <command>
```

## シークレット検出（secretlint）

API キーなどのシークレットが誤ってコミットされるのを防ぐため、[secretlint](https://github.com/secretlint/secretlint) を導入しています。

- `git commit` 時に husky の pre-commit hook 経由で自動実行される
- OpenAI / AWS / GCP / GitHub / Slack / npm など主要サービスの API キーパターンを検出

```zsh
# 手動でシークレットスキャンを実行
pnpm lint:secret
```

## Claude Code Skills

本プロジェクトでは、開発を効率化するための [Claude Code スキル](.claude/skills/) を用意しています。

| スキル | コマンド例 | 概要 |
| --- | --- | --- |
| [add-doc](.claude/skills/add-doc/SKILL.md) | `/add-doc <ソースコードパス>` | ソースコードからドキュメントセクションを自動生成・追記 |
| [review-doc](.claude/skills/review-doc/SKILL.md) | `/review-doc chapter3` | ドキュメントの正確性・整合性チェック＆修正 |
| [brushup-doc](.claude/skills/brushup-doc/SKILL.md) | `/brushup-doc chapter3` | ドキュメントの文章品質向上＆内容充実化 |
| [cleanup-code](.claude/skills/cleanup-code/SKILL.md) | `/cleanup-code <TSファイルパス>` | TypeScriptコードの型エラー修正・非推奨API置換・未使用import削除・JSDoc追加 |
| [commit](.claude/skills/commit/SKILL.md) | `/commit` | 変更を分析して日本語 Conventional Commits 形式でコミット |
| [sync-readme](.claude/skills/sync-readme/SKILL.md) | `/sync-readme` | プロジェクトの実態に合わせてルート README.md を同期 |

### 推奨ワークフロー

```plaintext
1. ソースコードを新規作成
       ↓
  (optional) /cleanup-code で型エラー修正・import整理・JSDoc追加
       ↓
2. /add-doc でドキュメントセクションを自動生成
       ↓
  (optional) /review-doc でコードとの整合性をチェック
       ↓
3. /brushup-doc で文章品質・内容を仕上げ
```

各スキルの詳細は [.claude/skills/README.md](.claude/skills/README.md) を参照してください。
