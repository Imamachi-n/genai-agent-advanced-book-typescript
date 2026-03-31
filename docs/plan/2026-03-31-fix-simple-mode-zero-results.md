# 簡易モードで検索結果が0件になる問題の改善プラン

## Context

簡易モード（タイトル+アブストラクトのみ）で論文検索を実行したところ、3件の論文がヒット（関連度 0.40〜0.43）したにもかかわらず、SimpleAnalyzer が全て `isRelated: false` と判定し、readingResults が 0 件になった。さらに TaskEvaluator の早期終了ロジックにより、リトライせず即レポート生成に進んでしまい、中身のないレポートが出力された。

```
論文フィルタリング: 全3件 → 関連あり0件（除外3件）
読み取り結果が0件のため、現状の情報でレポート生成に進みます
```

---

## 改善点 1: SimpleAnalyzer プロンプトの関連度判定を緩和

**問題**: 簡易モードはアブストラクトしか情報がないのに、厳密な関連度判定をしてしまい、関連度 0.40〜0.43 の論文を全て除外してしまう。

**対策**: `simple_analyze.prompt` の指示を緩和し、「部分的にでも関連があれば関連あり」とする。

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/prompts/simple_analyze.prompt`

`<instructions>` の 1 番を以下に変更:

```
1. この論文が調査タスクに関連しているかどうかを判定してください。
   - 簡易モードのため、厳密な判定ではなく寛容に判定してください。
   - アブストラクトの内容がタスクで求められている情報に部分的にでも関連する場合は関連ありと判断してください。
   - タスクのテーマと全く無関係な場合にのみ関連なしと判断してください。
```

---

## 改善点 2: TaskEvaluator の 0 件時早期終了をリトライ可能に変更

**問題**: `readingResults.length === 0` の場合、リトライカウントを確認せず即 `generate_report` に遷移する。せっかくのリトライループが無駄になる。

**対策**: 0 件の場合もリトライ上限に達していなければ `decompose_query` に戻し、別角度の検索を試みる。

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/task-evaluator-chain.ts`

現在のコード（L44-59）:
```typescript
if (readingResults.length === 0) {
  // 即 generate_report に遷移
}
```

変更後:
```typescript
if (readingResults.length === 0) {
  currentRetryCount++;
  if (currentRetryCount >= this.maxRetryCount) {
    // リトライ上限到達 → generate_report へ
  } else {
    // リトライ → decompose_query へ（0件だった旨をフィードバック）
  }
}
```

evaluation の content に「前回の検索では関連論文が 0 件でした。より一般的な検索キーワードや異なるアプローチのサブタスクを生成してください。」を設定して decompose_query に戻す。

---

## 改善点 3: サブタスク分解プロンプトでメタタスクを禁止

**問題**: サブタスク1「文献の収集とフィルタリング」のようなメタタスク（検索プロセスそのものの説明）が生成され、検索クエリとして機能しない。

**対策**: `query_decomposer.prompt` のルールに「メタタスク禁止」を追加。

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/prompts/query_decomposer.prompt`

`<rules>` セクションにルール 8 を追加:

```
8. サブタスクは検索クエリとして直接機能する具体的な調査内容でなければなりません：
   - 「文献を収集する」「情報を抽出する」「著者を特定する」のような検索プロセスの手順ではなく、具体的な調査対象を記述してください
   - NG例: 「生成AIに関する文献を収集しフィルタリングする」
   - OK例: 「生成AIを用いたゲノムアセンブリの精度改善手法」
   - 各サブタスクは「〇〇について調査する」「〇〇を分析する」という形で、具体的な技術・手法・応用を含めてください
```

---

## 変更対象ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `chains/prompts/simple_analyze.prompt` | 関連度判定基準を緩和 |
| `chains/task-evaluator-chain.ts` | 0 件時にリトライ可能にする |
| `chains/prompts/query_decomposer.prompt` | メタタスク禁止ルール追加 |

## 検証方法

1. 同じクエリ「生成AIを用いたゲノム解析の最新動向」で簡易モード実行
2. SimpleAnalyzer が少なくとも 1 件以上 `isRelated: true` を返すことを確認
3. 万が一 0 件でもリトライが発動し `decompose_query` に戻ることを確認
4. サブタスクに「文献を収集する」のようなメタタスクが含まれないことを確認
