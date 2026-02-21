# agent/ ディレクトリについて

このディレクトリは AgentCore Runtime 上で動作するエージェントアプリのソースコードです。
CDK デプロイ時に Docker イメージとしてビルドされ、コンテナとして実行されます。

## package.json が独立して存在する理由

本プロジェクトは pnpm-workspace でモノレポ管理していますが、このディレクトリには独自の `package.json` と `pnpm-lock.yaml` が存在します。

これは **Dockerfile 内で `pnpm install` を実行するため**です。

```dockerfile
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --prod --frozen-lockfile
```

Docker ビルドコンテキストは `agent/` ディレクトリに限定されるため、親ディレクトリの `node_modules` やワークスペースの依存解決を利用できません。そのため、Docker 内で依存関係を解決するための `package.json` と `pnpm-lock.yaml` がこのディレクトリに必要になります。

## ローカル開発時の注意

- 依存パッケージを追加・更新した場合は、このディレクトリの `package.json` を直接編集してください
- ロックファイルの再生成は `pnpm install --ignore-workspace` で実行してください
