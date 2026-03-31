# 簡易版レポートモード（タイトル+アブストラクトのみ）追加のプラン

## Context

現在の RAG 論文検索エージェントは、検索でヒットした全論文の PDF をダウンロード → Markdown 変換 → セクション分析 → 要約 という重い処理を行っている。タイトルとアブストラクトだけでざっくり調べたいケースに対応するため、PDF 取得をスキップする「簡易版モード」を追加する。

**モード選択は AI との対話（HearingChain）の中でユーザーに確認する。**

---

## 設計方針

- BiorxivPaper には `abstract` が Qdrant から取得済み → PDF 不要で分析可能
- 簡易版では `PaperAnalyzerAgent`（SetSection → CheckSufficiency → Summarizer ループ）を完全バイパス
- `SimpleAnalyzer` で 1 回の LLM コールで関連度判定 + 回答生成
- モード選択は `HearingChain` の対話フローで行い、`Hearing` スキーマ経由で State に伝播
- `ReadingResult` 型はそのまま維持（簡易版では `markdownPath: ''`）

---

## Step 1: `Hearing` スキーマにモード選択を追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/models.ts`

```typescript
export const hearingSchema = z.object({
  is_need_human_feedback: z.boolean().describe('追加の質問が必要かどうか'),
  additional_question: z.string().describe('追加の質問'),
  analysis_mode: z.enum(['simple', 'detailed']).describe(
    '分析モード。simple: タイトルとアブストラクトのみで簡易レポート、detailed: PDF全文を取得して詳細レポート'
  ),
});
```

- ヒアリング完了時に LLM がユーザーの希望に基づいてモードを判定
- `is_need_human_feedback: true` の間は暫定値でOK

---

## Step 2: ヒアリングプロンプトにモード選択を追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/prompts/hearing.prompt`

`<key_areas>` にモード選択の項目を追加:

```
- 調査の深さ（簡易版: タイトルとアブストラクトのみで素早く概要把握 / 詳細版: PDF全文を取得して深く分析）
```

`<rules>` に追加:

```
5. 調査の深さについて、簡易版（タイトル・アブストラクトのみ）と詳細版（PDF全文分析）のどちらを希望するか確認してください。
   ユーザーが明示的に選択しない場合は、詳細版（detailed）をデフォルトとしてください。
```

---

## Step 3: `ResearchAgentAnnotation` に `analysisMode` を追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/agent/research-agent.ts`

State に `analysisMode` を追加:

```typescript
analysisMode: Annotation<'simple' | 'detailed'>({
  reducer: (_prev, next) => next,
  default: () => 'detailed',
}),
```

`HearingChain.invoke()` の結果から `analysisMode` を State に反映する（`hearing.analysis_mode`）。

---

## Step 4: `HearingChain` から `analysisMode` を State に伝播

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/hearing-chain.ts`

`invoke()` の Command update に `analysisMode` を追加:

```typescript
return new Command({
  goto: nextNode,
  update: {
    hearing,
    messages: message,
    analysisMode: hearing.analysis_mode,
  },
});
```

---

## Step 5: `configs.ts` にデフォルト `analysisMode` 設定を追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/configs.ts`

```typescript
// Settings interface に追加
analysisMode: 'simple' | 'detailed';

// loadSettings() に追加
analysisMode: (process.env.ANALYSIS_MODE as 'simple' | 'detailed') ?? 'detailed',
```

環境変数 `ANALYSIS_MODE` でもオーバーライド可能にしておく（テスト用途）。

---

## Step 6: `simpleAnalysisSchema` と `SimpleAnalyzer` チェーンを作成

### 6a: スキーマ追加
**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/models.ts`

```typescript
export const simpleAnalysisSchema = z.object({
  is_related: z.boolean().describe('論文がタスクに関連しているかどうか'),
  answer: z.string().describe('アブストラクトに基づく簡潔な回答、または無関係の理由'),
});
```

### 6b: プロンプト作成
**ファイル（新規）**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/prompts/simple_analyze.prompt`

入力: タイトル、著者、アブストラクト、ゴール、タスク
出力: `is_related` (boolean) + `answer` (string)

### 6c: チェーンクラス作成
**ファイル（新規）**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/simple-analyzer-chain.ts`

- `loadPrompt('simple_analyze')` + `llm.withStructuredOutput(simpleAnalysisSchema)`
- state から `goal` と `readingResult` を受け取り、paper.abstract を使って分析
- `{ readingResult: updatedResult }` を返す

---

## Step 7: `PaperProcessor` を簡易モード対応に

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/paper-processor-chain.ts`

### コンストラクタ
```typescript
constructor(searcher, maxWorkers, analysisMode: 'simple' | 'detailed' = 'detailed')
```

### `invoke()` - Send 先を分岐
```typescript
const targetNode = this.analysisMode === 'simple' ? 'simple_analyze_paper' : 'analyze_paper';
```

### `run()` - 簡易モード時は PDF 変換をスキップ
```typescript
if (this.analysisMode === 'detailed') {
  const markdownPaths = await this.convertPdfs(uniquePapersList);
  // 既存ロジック
} else {
  // markdownPath: '' で ReadingResult 構築
}
```

---

## Step 8: `PaperSearchAgent` に `simple_analyze_paper` ノードを追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/agent/paper-search-agent.ts`

### コンストラクタ
```typescript
constructor(llm, searcher, options: { recursionLimit?, maxWorkers?, analysisMode? })
```

### グラフに新ノード追加
```typescript
.addNode('search_papers', ..., { ends: ['analyze_paper', 'simple_analyze_paper'] })
.addNode('simple_analyze_paper', async (state) => {
  const result = await this.simpleAnalyzer.invoke(state);
  const readingResult = result.readingResult as ReadingResult | undefined;
  return { processingReadingResults: readingResult ? [readingResult] : [] };
})
.addEdge('simple_analyze_paper', 'organize_results')
```

---

## Step 9: `ResearchAgent` からモードを伝播

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/agent/research-agent.ts`

### 方式: 対話でモード確定後、`invokePaperSearchAgent` で State から読み取り

`analysisMode` は State に入っているので、`invokePaperSearchAgent` で State から取得して `PaperSearchAgent` に渡す。

ただし `PaperSearchAgent` はグラフ構築時にモードが必要（ノード構成に影響するため）なので、**両方のノードを常に登録しておき、`PaperProcessor` の Send 先で分岐する** 方式を採用。

→ `PaperSearchAgent` は `analysisMode` をコンストラクタで受け取らず、`PaperProcessor` が State から `analysisMode` を読む形にする。

### `PaperProcessor.invoke()` の修正

```typescript
async invoke(state: Record<string, unknown>): Promise<Command> {
  const analysisMode = (state.analysisMode as 'simple' | 'detailed') ?? this.defaultAnalysisMode;
  // ... targetNode を analysisMode で分岐
}
```

---

## 変更対象ファイル一覧

| ファイル | Action | 変更内容 |
|----------|--------|----------|
| `models.ts` | 変更 | `hearingSchema` に `analysis_mode` 追加、`simpleAnalysisSchema` 追加 |
| `chains/prompts/hearing.prompt` | 変更 | モード選択の質問項目追加 |
| `chains/hearing-chain.ts` | 変更 | `analysisMode` を State に伝播 |
| `agent/research-agent.ts` | 変更 | State に `analysisMode` 追加 |
| `configs.ts` | 変更 | `analysisMode` 設定追加 |
| `chains/prompts/simple_analyze.prompt` | **新規** | 簡易分析プロンプト |
| `chains/simple-analyzer-chain.ts` | **新規** | SimpleAnalyzer チェーン |
| `chains/paper-processor-chain.ts` | 変更 | PDF スキップ + Send 先分岐 |
| `agent/paper-search-agent.ts` | 変更 | `simple_analyze_paper` ノード追加 |

## 変更不要

- `task-evaluator-chain.ts`, `reporter-chain.ts`: markdownPath 除外で動作済み
- `searcher/`, `rag/`: 検索ロジック変更なし

---

## 検証方法

1. **対話テスト**: エージェント起動後、ヒアリングで「簡易版で」と答え、PDF ダウンロードが発生しないことを確認
2. **詳細版テスト**: 「詳細版で」と答え、従来通り PDF 分析が動作することを確認
3. **デフォルト確認**: モード指定なしの場合に `detailed` がデフォルトになることを確認
4. **TypeScript**: `npx tsc --noEmit` でエラーなし
5. **ログ**: 簡易モード時に `simple_analyze_paper` ノードログが出力されること
