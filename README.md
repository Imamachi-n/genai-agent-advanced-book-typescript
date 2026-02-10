# genai-agent-advanced-book-typescript

TypeScript版 現場で活用するためのAIエージェント実践入門

> 📖 本リポジトリは [genai-agent-advanced-book](https://github.com/masamasa59/genai-agent-advanced-book) のサンプルコードを Python から TypeScript へ変換・再実装したものです。

## ディレクトリ構成

本プロジェクトはpnpmワークスペースによるモノレポ構成です。

```plaintext
/
├── packages/
│   └── @ai-suburi/
│       ├── core/            # サンプルコード (@ai-suburi/core)
│       │   ├── chapter3/    # 第3章のサンプル
│       │   ├── package.json
│       │   └── tsconfig.json
│       └── docs/            # Docusaurus ドキュメント (@ai-suburi/docs)
│           ├── docs/
│           ├── src/
│           ├── docusaurus.config.ts
│           └── package.json
├── pnpm-workspace.yaml      # ワークスペース設定
├── package.json             # ルート設定
└── tsconfig.json            # 共通TypeScript設定
```

## 開発環境

本プロジェクトでは以下のツールを使用しています。

| ツール | 説明 |
| --- | --- |
| [pnpm](https://pnpm.io/) | パッケージマネージャー（モノレポ対応） |
| [tsx](https://www.npmjs.com/package/tsx) | TypeScript ファイルの直接実行 |
| [Biome](https://biomejs.dev/) | リンター・フォーマッター |
| [Docusaurus](https://docusaurus.io/) | ドキュメントサイト |

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
cp .envrc.example .envrc  # テンプレートがある場合
direnv allow
```

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

### 特定パッケージでのコマンド実行

```zsh
# @ai-suburi/core パッケージ
pnpm --filter @ai-suburi/core <command>

# @ai-suburi/docs パッケージ
pnpm --filter @ai-suburi/docs <command>
```

## Claude Code Skills

本プロジェクトでは、ドキュメント管理を効率化するための [Claude Code スキル](.claude/skills/) を用意しています。

| スキル | コマンド例 | 概要 |
| --- | --- | --- |
| [add-doc](.claude/skills/add-doc/SKILL.md) | `/add-doc <ソースコードパス>` | ソースコードからドキュメントセクションを自動生成・追記 |
| [review-doc](.claude/skills/review-doc/SKILL.md) | `/review-doc chapter3` | ドキュメントの正確性・整合性チェック＆修正 |
| [brushup-doc](.claude/skills/brushup-doc/SKILL.md) | `/brushup-doc chapter3` | ドキュメントの文章品質向上＆内容充実化 |

### 推奨ワークフロー

```plaintext
1. ソースコードを新規作成
       ↓
2. /add-doc でドキュメントセクションを自動生成
       ↓
3. /review-doc でコードとの整合性をチェック
       ↓
4. /brushup-doc で文章品質・内容を仕上げ
```

各スキルの詳細は [.claude/skills/README.md](.claude/skills/README.md) を参照してください。
