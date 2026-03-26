# Chapter 6 ドキュメント追記プラン：プロンプトファイルの追加

## Context

chapter6.md（2217行）はすでにメインの実装ファイル19個をカバーしているが、LLMチェーンで使用する**プロンプトファイル10個**がドキュメントに含まれていない。プロンプトはエージェントの振る舞いを決定する重要コンポーネントであり、ドキュメントに追記する価値が高い。

## 追記対象

### プロンプトファイル（10個）

各チェーンのセクションに対応するプロンプトを追記する：

| プロンプトファイル | 対応セクション | 対応チェーン |
| --- | --- | --- |
| `hearing.prompt` | 6-6 | HearingChain |
| `goal_optimizer_conversation.prompt` | 6-6 | GoalOptimizer |
| `goal_optimizer_search.prompt` | 6-6 | GoalOptimizer |
| `query_decomposer.prompt` | 6-6 | QueryDecomposer |
| `set_section.prompt` | 6-7 | SetSection |
| `check_sufficiency.prompt` | 6-7 | CheckSufficiency |
| `summarize.prompt` | 6-7 | Summarizer |
| `task_evaluator.prompt` | 6-9 | TaskEvaluator |
| `reporter_system.prompt` | 6-10 | Reporter |
| `reporter_user.prompt` | 6-10 | Reporter |

### テストファイル（2個）- 任意

| テストファイル | 内容 |
| --- | --- |
| `test/models.test.ts` | 13スキーマのバリデーションテスト |
| `test/markdown-parser.test.ts` | セクション解析テスト |

## 追記方針

### プロンプトファイルの追記方法

各セクション（6-6, 6-7, 6-9, 6-10）のチェーンコードの直後に、対応するプロンプトファイルを `<details>` タグで折りたたみ表示する。

```markdown
<details>
<summary>chapter6/chains/prompts/hearing.prompt（クリックで展開）</summary>

\```text title="chapter6/chains/prompts/hearing.prompt"
（プロンプト全文）
\```

</details>
```

### テストファイルの追記方法（任意）

新しいセクション「6-12. テスト」を参考文献の直前（`---` の前）に追加する。

## 修正対象ファイル

- `packages/@ai-suburi/docs/docs/ai-agent-practice/chapter6.md`
  - セクション 6-6 にプロンプト4個追記
  - セクション 6-7 にプロンプト3個追記
  - セクション 6-9 にプロンプト1個追記
  - セクション 6-10 にプロンプト2個追記
  - （任意）セクション 6-12 テスト追加
  - 学習の流れテーブル更新不要（セクション数変更なしの場合）

## 検証方法

1. `pnpm --filter @ai-suburi/docs build` でドキュメントビルドが通ることを確認
2. ブラウザでプレビューして表示崩れがないことを確認
