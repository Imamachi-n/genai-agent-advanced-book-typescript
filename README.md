# genai-agent-advanced-book-typescript

TypeScript版 現場で活用するためのAIエージェント実践入門

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

## direnvのインストール

direnvを使って環境変数を管理します。

### macOS (Homebrew)

```zsh
brew install direnv
```

シェルにhookを追加します（zshの場合）。

```zsh
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
```

### `.envrc` の設定

プロジェクトルートに `.envrc` ファイルを作成し、必要な環境変数を記述します。

```zsh
cp .envrc.example .envrc  # テンプレートがある場合
direnv allow
```

## pnpmのインストール

### Homebrew

```zsh
brew install pnpm
```

### npm

```zsh
npm install -g pnpm
```

## npmパッケージのインストール

```zsh
pnpm install
```

## 使用方法

### サンプルコードの実行

```zsh
pnpm tsx chapter3/test3-1-chat-completions-api.ts
```

### ドキュメントサイト

```zsh
# 開発サーバー起動
pnpm dev:docs

# ビルド
pnpm build:docs
```

### 特定パッケージでのコマンド実行

```zsh
# @ai-suburi/core パッケージ
pnpm --filter @ai-suburi/core <command>

# @ai-suburi/docs パッケージ
pnpm --filter @ai-suburi/docs <command>
```
