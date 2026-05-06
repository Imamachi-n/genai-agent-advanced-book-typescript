# Plan: Claude Code ベストプラクティス（Agent Skills 編）章の追加

## Context

- `packages/@ai-suburi/docs` は Docusaurus 3.9 ベースの書籍ドキュメントサイト。既存セクションは [`ai-agent-practice/`](../../packages/@ai-suburi/docs/docs/ai-agent-practice/) と [`bedrock-agentcore/`](../../packages/@ai-suburi/docs/docs/bedrock-agentcore/) の 2 つで、サイドバーは `_category_.json` の `position` と各 md の `sidebar_position` から自動生成される（[`sidebars.ts`](../../packages/@ai-suburi/docs/sidebars.ts)）。
- 本書 9.4.2「実践例：Claude Code の Agent Skill 設計に応用する」（[`chapter9.md`](../../packages/@ai-suburi/docs/docs/ai-agent-practice/chapter9.md)）で Agent Skill の設計サイクルには触れているが、Skill そのものの仕様・実装・運用ベスプラを体系化した章は未整備。
- ユーザは Claude Code のベスプラ集を新セクションとして育てたく、第 1 弾として「Agent Skills のベストプラクティス」をコンパクト 1 章で書き起こしたい。情報源は Anthropic 公式ドキュメント（[code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills)）に準拠。

## アプローチ

新規ディレクトリ `packages/@ai-suburi/docs/docs/claude-code-best-practices/` を切り、サイドバー `position: 3`（既存 2 セクションの後ろ）に並べる。1 章完結のコンパクト構成（Skill とは → 構造 → 設計指針 → 運用改善 まで）。スタイルは [`ai-agent-practice/chapter9.md`](../../packages/@ai-suburi/docs/docs/ai-agent-practice/chapter9.md) に揃える（`:::note この章で学ぶこと`、Mermaid、テーブル、`:::tip`、コードブロックの `title=`）。

## 追加・変更するファイル

| パス | 種別 | 内容 |
|---|---|---|
| `packages/@ai-suburi/docs/docs/claude-code-best-practices/_category_.json` | 新規 | `{ "label": "Claude Code ベストプラクティス", "position": 3, "link": { "type": "generated-index", "description": "Claude Code を活用したコーディングのベストプラクティス集。" } }` |
| `packages/@ai-suburi/docs/docs/claude-code-best-practices/chapter1.md` | 新規 | 本章本体（`sidebar_position: 1`）。下記の節構成で執筆。 |

既存ファイルは編集しない。9.4.2 への相互参照は本章側からのリンクで完結させる。

## 章立て（chapter1.md）

```
---
sidebar_position: 1
---

# Chapter 1: Agent Skills のベストプラクティス

:::note この章で学ぶこと
- Agent Skill とは何か、CLAUDE.md やスラッシュコマンドとの使い分け
- SKILL.md の構造とフロントマター主要フィールド
- description 設計の勘所（progressive disclosure / 1,536 文字キャップ）
- 起動制御マトリクス（disable-model-invocation × user-invocable）
- 動的コンテキスト注入と context: fork による副エージェント実行
- 観測ログから Skill を継続改善するループ（本書 9.4.2 との対応）
:::
```

1. **概要**：Skill = `SKILL.md` + 補助ファイル群。CLAUDE.md と異なり「使われる時だけ読み込まれる」のがコア。スラッシュコマンドが Skill に統合されたという公式の経緯も触れる。
2. **Skill の構造**：ディレクトリレイアウト（`SKILL.md` 必須、`reference.md` / `examples/` / `scripts/` 任意）。フロントマター主要フィールド表（`description` / `when_to_use` / `disable-model-invocation` / `user-invocable` / `allowed-tools` / `model` / `effort` / `context` / `agent` / `paths` / `arguments`）。
3. **配置場所と優先度**：Enterprise / Personal (`~/.claude/skills/`) / Project (`.claude/skills/`) / Plugin の 4 階層と優先順位。live change detection、モノレポでの nested discovery、`--add-dir` の挙動。
4. **description 設計のベストプラクティス**：description が「Claude が起動判断に使う唯一の手がかり」であること。1,536 文字キャップ／key use case を先頭に／自然な発話に出る語彙を入れる、を Tips としてまとめる。Anti-pattern として「曖昧 description で誤起動」「機能羅列で 1,536 文字を超え末尾切れ」。
5. **progressive disclosure**：SKILL.md は 500 行以下、詳細は `reference.md` / `examples.md` に分離、`scripts/` は実行専用（読み込ませない）。Anthropic 公式の階層モデル（メタデータ→本文→補助）を Mermaid で可視化。
6. **起動制御パターン**：「Claude が呼ぶ／ユーザだけが呼ぶ／ユーザは隠す」の 4 マトリクスを表で提示。`/deploy` `/commit` 系は `disable-model-invocation: true`、背景知識は `user-invocable: false`、という典型例を `:::tip` で添える。
7. **動的コンテキスト注入と引数**：`` !`cmd` `` / フェンス `` ```! `` ブロック、`${CLAUDE_SKILL_DIR}` / `${CLAUDE_SESSION_ID}`、`$ARGUMENTS` `$0/$1`、`arguments:` 名前付き引数。`pr-summary` 風 Skill のサンプルを置く。
8. **context: fork とサブエージェント実行**：`context: fork` + `agent: Explore` / `Plan` の使いどころ、`Skill (context: fork)` と `Subagent (skills: ...)` の対応表。Warning（タスクなし Skill を fork すると無意味）も明記。
9. **継続改善ループ**：本書 9.4.2 の「ツールやプロンプトの改善」「自己改善エージェント」の文脈に接続。観測 → description/本文を更新 → live change detection で即反映、というショートサイクルを Mermaid で図示。
10. **アンチパターン & トラブルシュート**：「起動しすぎる／しない」「description 切れ」「auto-compaction 後に効かない（25k token 共有予算）」「`allowed-tools` を広げすぎてセキュリティリスク」など、公式 Troubleshooting と engineering blog の指摘をベースに整理。
11. **まとめ**：「最初に作るべき Skill 候補」を選ぶフローチャート（Mermaid）で締め。9.4.2 への相互リンク、公式ドキュ・関連節（subagents / hooks / plugins）への参照リスト。

## 参考にする既存資産（再利用）

- ai-agent-practice の章構造（`:::note この章で学ぶこと`、`:::tip`、Mermaid 図、code block の `title=` 規約）→ そのまま踏襲
- Tabs / TabItem は今回の章では用途が薄いので原則使わず、必要時のみ chapter3.md のパターンを参照
- 9.4.2 内の「Skill 設計サイクル」記述は重複を避けて、本章 §9 から本書 9.4.2 を参照する形で連携

## 検証方法

1. `pnpm --filter @ai-suburi/docs start`（Docusaurus dev server）でローカル起動し、サイドバーに「Claude Code ベストプラクティス > Chapter 1: Agent Skills のベストプラクティス」が `position: 3` の位置に表示されることを確認
2. ページ遷移して、Mermaid 図 / テーブル / `:::note` `:::tip` admonition / コードブロックのシンタックスハイライトが崩れずレンダリングされること
3. 本章から 9.4.2 への相対リンク（`../ai-agent-practice/chapter9#942-...`）がクリックで遷移すること
4. `pnpm --filter @ai-suburi/docs build` を通し、ビルドエラー（broken anchor / mdx parse）が出ないことを確認
5. 内容面：公式ドキュ ([code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills)) のフロントマター仕様表と本章の表を突き合わせ、漏れ・誤記がないか最終チェック
