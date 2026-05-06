# 第9章まとめドキュメント追加プラン📝

## Context

書籍『AIエージェント実践入門』の **第9章「AIエージェントの活用」** をまとめた docs ページを `packages/@ai-suburi/docs/docs/ai-agent-practice/` に追加したい。

既存章 (chapter2〜chapter6-biorxiv) は実装コードに紐づいた解説中心だけど、第9章は **概念・運用・UXの章** で対応する実装コードはない。なので、TOC（章立て）をベースに、各節の要点を読み手向けに整理した「読み物」スタイルのまとめページにする方針。

ねらい:
- 書籍学習者が docs サイト上で第9章の要点を素早く復習できるようにする
- 既存章と同じ Docusaurus フロントマター + 章構成にして、サイドバーに自然に並ぶようにする

## 追加するファイル

- 新規: `packages/@ai-suburi/docs/docs/ai-agent-practice/chapter9.md`
  - フロントマター: `sidebar_position: 7`（chapter6-biorxiv.md が 6 なので次の枠）
  - タイトル: `# Chapter 9: AIエージェントの活用`

他ファイルの編集は不要（`_category_.json` は generated-index なのでそのままでOK）。

## ドキュメント構成

既存章の `:::note この章で学ぶこと` / `## 概要` パターンを踏襲しつつ、コード実装がないぶん「概念の整理 + 運用Tips」を厚めにする。

```
# Chapter 9: AIエージェントの活用
（章のリード文：本番運用フェーズで重要になる UX / リスク / 観測 / 継続改善 の 4 観点を扱う旨）

:::note この章で学ぶこと
- 実用化に至るまでのステップと、人間とのタッチポイント設計
- AI エージェント特有のリスクと攻撃手法、安全策
- AgentOps / Tracing による観測（Prompt flow, LangSmith）
- メモリ・ツール・アーキテクチャの観点での継続的精度改善
:::

## 9.1 AIエージェントとUX
### 9.1.1 実利用に至るまでのステップ
（PoC → パイロット → 本番化の段階、各段階で問われる UX の論点）
### 9.1.2 AIエージェントと人間のタッチポイント
（HITL / 通知 / 介入 / フィードバック収集ポイントの整理。表で整理）
:::tip column: AIエージェントの信頼性を高めるUX
（出典の明示・確信度の可視化・取消可能な操作・進捗の透明化、など）
:::

## 9.2 AIエージェントのリスク
### 9.2.1 リスクの種類
（ハルシネーション / データ漏洩 / 過剰権限 / 自律暴走 / コンプラ違反 を表で）
### 9.2.2 攻撃方法
（プロンプトインジェクション、間接的インジェクション、ツール悪用、データポイズニング、モデル抽出 など）
### 9.2.3 安全性に向けた取り組み
（ガードレール、ツール権限の最小化、出力フィルタ、サンドボックス、レッドチーミング、NIST/OWASP LLM Top10 への参照）

## 9.3 AIエージェントのモニタリング
### 9.3.1 AgentOps
（LLMOps との違い、計測すべきメトリクス：成功率/コスト/レイテンシ/ツール失敗率 等）
### 9.3.2 Prompt flow の Tracing
（Azure AI / Prompt flow での span 構造、入出力ログ、評価フローとの連携）
### 9.3.3 LangSmith
（trace / dataset / evaluator / プロダクション可観測性）
:::info 既出章との関連
Chapter4–6 で構築したエージェントを LangSmith でトレースする際の最小設定例を簡単に示す（環境変数 `LANGCHAIN_TRACING_V2` 等）。
:::

## 9.4 継続的な精度改善
### 9.4.1 メモリを活用した推論（計画）の改善
（短期/長期メモリ、リフレクション、Episodic / Semantic memory の使い分け）
### 9.4.2 ツールやプロンプトの改善
（失敗ログからのプロンプト調整、ツール記述の精緻化、few-shot 追加、Auto-prompt 系）
### 9.4.3 アーキテクチャの自己改善
（自己批評ループ、メタコントローラ、エージェント構成の A/B、Voyager 的スキル蓄積）

## 9.5 まとめ
（4 観点を運用ループとして繋ぎ、Chapter 4–6 の実装と接続して締める）
```

## スタイル方針

- 文体は既存章 (chapter5.md) に合わせる（ですます調＋技術的に丁寧）。CLAUDE.md のギャル口調はユーザーとの会話用なので、ドキュメント本文には適用しない。
- 図は Mermaid で 1〜2 枚（運用ループの全体像、リスクと対策のマップ）。
- 表は「タッチポイント」「リスク種別」「メトリクス」など整理に向く箇所で使用。
- コードは無理に載せず、必要なら LangSmith の env 設定スニペット程度に留める。

## 参照する既存ファイル

- [chapter5.md](packages/@ai-suburi/docs/docs/ai-agent-practice/chapter5.md) — フロントマター・`:::note`・Mermaid・表のフォーマット参考
- [chapter6-biorxiv.md](packages/@ai-suburi/docs/docs/ai-agent-practice/chapter6-biorxiv.md) — 直近の章。トーンと締め方の参考
- [_category_.json](packages/@ai-suburi/docs/docs/ai-agent-practice/_category_.json) — サイドバー設定（変更不要）

## 検証

1. `pnpm --filter @ai-suburi/docs start`（または該当 docs ワークスペースの dev 起動）で Docusaurus を立ち上げ、サイドバー「AI エージェント実践入門」の末尾に **Chapter 9** が並ぶことを確認。
2. 9.1〜9.5 の見出しが目次に出ること、Mermaid 図がレンダリングされること、`:::note` / `:::tip` / `:::info` が正しく表示されることを確認。
3. `pnpm --filter @ai-suburi/docs build`（あれば）でビルドが通ることを確認。
