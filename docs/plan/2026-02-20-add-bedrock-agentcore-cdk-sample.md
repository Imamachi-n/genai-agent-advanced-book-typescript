# Bedrock AgentCore CDK サンプルコード実装プラン

## Context

既存リポジトリに Amazon Bedrock AgentCore のサンプルコードを追加する。`bedrock-agentcore/index.md` で概念説明は完了済みなので、次のステップとして **AWS CDK を使った実際のデプロイコード** と **解説ドキュメント** を作成する。

最初のサンプルは「最もシンプルな構成」として **Runtime のみ** をカバーし、基本的なエージェントアプリ + CDK インフラコードの両方を含める。

---

## 1. 新規パッケージ作成

### パッケージ: `packages/@ai-suburi/bedrock-agentcore-cdk/`

```
packages/@ai-suburi/bedrock-agentcore-cdk/
├── bin/
│   └── app.ts                       # CDK アプリエントリポイント
├── lib/
│   └── agentcore-runtime-stack.ts   # CDK スタック定義（AgentCore Runtime）
├── agent/
│   ├── index.ts                     # エージェントアプリ（Express サーバー）
│   ├── package.json                 # エージェント側の依存関係
│   ├── tsconfig.json                # エージェント側の TS 設定
│   └── Dockerfile                   # Runtime デプロイ用コンテナ
├── package.json                     # CDK プロジェクトの依存関係
├── tsconfig.json                    # CDK 側の TS 設定
└── cdk.json                         # CDK 設定ファイル
```

### 1-1. エージェントアプリ (`agent/`)

**目的:** AgentCore Runtime 上で動作するシンプルな HTTP サーバー

- **フレームワーク:** Express（広く知られており教育用途に最適）
- **ポート:** 8080（Runtime の固定要件）
- **エンドポイント:**
  - `GET /ping` → ヘルスチェック（200 OK を返す）
  - `POST /invocations` → ユーザーのプロンプトを受け取り、Bedrock（Claude）で処理して応答を返す
- **依存パッケージ:**
  - `express` + `@types/express`
  - `@aws-sdk/client-bedrock-runtime`（Bedrock モデル呼び出し用）
  - `typescript`, `tsx`

### 1-2. CDK スタック (`lib/agentcore-runtime-stack.ts`)

**目的:** AgentCore Runtime をデプロイする CDK スタック

- `@aws-cdk/aws-bedrock-agentcore-alpha` の L2 コンストラクトを使用
- Docker イメージを `agent/` ディレクトリからビルド
- Bedrock モデルへのアクセス権限を IAM で設定
- 教育用のコメントを豊富に記載

### 1-3. CDK アプリ (`bin/app.ts`)

- `AgentCoreRuntimeStack` をインスタンス化
- リージョン・アカウント設定

### 1-4. package.json（CDK プロジェクト）

**主要依存パッケージ:**
- `aws-cdk-lib`
- `constructs`
- `@aws-cdk/aws-bedrock-agentcore-alpha`
- `aws-cdk`（devDependencies）
- `typescript`, `tsx`

---

## 2. ドキュメント作成

### ファイル: `packages/@ai-suburi/docs/docs/bedrock-agentcore/chapter1.md`

**sidebar_position:** 2（index.md の次に配置）

**内容構成:**

1. **タイトル:** AgentCore Runtime を CDK でデプロイする
2. **導入:** なぜ CDK を使うのか、このサンプルで何ができるか
3. **:::note この章で学ぶこと** — 学習目標をリスト化
4. **前提条件** — AWS CLI, CDK CLI, Docker, Node.js の準備
5. **アーキテクチャ概要** — Mermaid ダイアグラムでデプロイ構成を図示
6. **エージェントアプリの実装** — `agent/index.ts` のコード解説
7. **CDK スタックの実装** — `lib/agentcore-runtime-stack.ts` のコード解説
8. **デプロイ手順** — `cdk bootstrap` → `cdk deploy` の流れ
9. **動作確認** — `agentcore invoke` またはAPIでの呼び出しテスト
10. **クリーンアップ** — `cdk destroy` でリソース削除
11. **参考資料** — 公式ドキュメントへのリンク

**スタイル:** 既存ドキュメントに合わせて、です/ます調、Mermaid 図、:::note/:::tip/:::caution を使用

---

## 3. モノレポ統合

### 変更ファイル:

| ファイル | 変更内容 |
| --- | --- |
| `pnpm-workspace.yaml` | `packages/@ai-suburi/bedrock-agentcore-cdk` を追加 |

---

## 4. 実装順序

1. `packages/@ai-suburi/bedrock-agentcore-cdk/` ディレクトリ作成
2. `package.json`, `tsconfig.json`, `cdk.json` を作成
3. `agent/` 配下にエージェントアプリ（`index.ts`, `Dockerfile`, `package.json`, `tsconfig.json`）を作成
4. `lib/agentcore-runtime-stack.ts` で CDK スタック作成
5. `bin/app.ts` で CDK アプリ作成
6. `pnpm-workspace.yaml` を更新
7. `pnpm install` で依存関係インストール
8. ドキュメント `chapter1.md` を作成

---

## 5. 検証方法

- `cd packages/@ai-suburi/bedrock-agentcore-cdk && npx tsc --noEmit` で型チェック通過を確認
- `cd packages/@ai-suburi/bedrock-agentcore-cdk && npx cdk synth` で CloudFormation テンプレート生成を確認
- `pnpm dev:docs` でドキュメントサイトにアクセスし、新しいページが表示されることを確認
