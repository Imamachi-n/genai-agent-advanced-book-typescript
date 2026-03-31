# RAG ベース論文検索エージェントの改善プラン

## Context

RAG ベースの bioRxiv 論文検索エージェントで、ユーザークエリを 3-5 個のサブタスクに分解し、各サブタスクごとにベクトル DB を検索して論文を取得している。しかし、以下の問題がログから明らかになった:

1. **サブタスク間で同一論文が大量に重複** (例: 5 サブタスク中 5 つで同じ論文がヒット)
2. **関連度スコアが低い** (全て 0.40-0.49)
3. **サブタスク分解がセマンティックに近すぎる** → 同じベクトル空間をサーチしてしまう
4. **リトライループが無効** → 同じ DB に同じようなクエリを投げるだけで新情報が得られない
5. **フィルタリング損失が不透明** → isRelated フィルタでどれだけ落ちてるか見えない

---

## Phase 1: サブタスク間の DOI 重複排除 (HIGH)

**目的**: タスク N の検索時に、タスク 1..N-1 で既に見つかった DOI を Qdrant のフィルタで除外し、各タスクが異なる論文を取得するようにする。

### Step 1.1: `Searcher` インターフェース拡張

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/searcher/searcher.ts`

```typescript
export interface Searcher {
  run(goalSetting: string, query: string, excludeDois?: string[]): Promise<BiorxivPaper[]>;
}
```

- オプショナルパラメータなので後方互換性あり

### Step 1.2: `QdrantStore.search` にフィルタ追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/rag/qdrant-store.ts`

`search` メソッドに `excludeDois` パラメータを追加:

```typescript
async search(query: string, topK: number = 20, excludeDois: string[] = []): Promise<BiorxivPaper[]>
```

- `excludeDois.length > 0` の場合、`client.search()` に `filter: { must_not: [...] }` を渡す
- `doi` フィールドは既にキーワードインデックスが作成済み (L47-50)

### Step 1.3: `RagSearcher.run` に除外 DOI を伝播

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/rag/rag-searcher.ts`

```typescript
async run(goalSetting: string, query: string, excludeDois: string[] = []): Promise<BiorxivPaper[]>
```

- L152 の `this.store.search()` に `excludeDois` を渡す

### Step 1.4: `PaperProcessor.run` で DOI を蓄積しながら検索

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/paper-processor-chain.ts`

L68-77 のタスクループを変更:

```typescript
const allFoundDois: string[] = [];
for (const task of state.tasks) {
  const searchedPapers = await this.searcher.run(state.goal, task, allFoundDois);
  taskPapers.set(task, searchedPapers.map((paper) => paper.doi));
  for (const paper of searchedPapers) {
    uniquePapers.set(paper.doi, paper);
    allFoundDois.push(paper.doi);
  }
}
```

- DOI 蓄積は `PaperProcessor.run` のローカル変数で完結 → LangGraph State の変更不要

---

## Phase 2: サブタスク分解の多様性向上 (HIGH)

**目的**: decomposer プロンプトを改善し、各サブタスクが異なるキーワード空間をターゲットにするようにする。

### Step 2.1: decomposer プロンプト更新

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/prompts/query_decomposer.prompt`

`<rules>` セクションにルール 6 を追加:

```
6. 各サブタスクは異なる検索キーワード空間をターゲットにする必要があります：
   - サブタスク間で主要な検索キーワードが重複しないようにしてください
   - 例えば、手法・アルゴリズム、応用分野・対象データ、ベンチマーク・評価、理論的背景、実用的課題・限界 のように異なる観点から分解してください
   - 各サブタスクには、他のサブタスクと重複しない固有のキーワード（技術名、手法名、概念名）を少なくとも1つ含めてください
```

---

## Phase 3: リトライ時のコンテキスト強化 (MEDIUM)

**目的**: リトライ時に前回の検索結果と使用サブタスクを decomposer に渡し、異なる角度からの分解を促す。

### Step 3.1: `QueryDecomposer.invoke` に前回コンテキストを追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/query-decomposer-chain.ts`

`invoke` メソッドで `searchedPaperList` と前回の `tasks` を state から取得:

```typescript
async invoke(state: Record<string, unknown>): Promise<Command> {
  const evaluation = state.evaluation as TaskEvaluation | undefined;
  const content = evaluation?.content ?? (state.goal as string) ?? '';
  const searchedPaperList = (state.searchedPaperList as string) ?? '';
  const previousTasks = (state.tasks as string[]) ?? [];

  const decomposedTasks = await this.run(content, searchedPaperList, previousTasks);
  return new Command({
    goto: 'paper_search_agent',
    update: { tasks: decomposedTasks.tasks },
  });
}
```

### Step 3.2: `QueryDecomposer.run` にリトライコンテキストを渡す

```typescript
async run(
  query: string,
  searchedPaperList: string = '',
  previousTasks: string[] = [],
): Promise<DecomposedTasks>
```

- リトライコンテキスト文字列を組み立てて `{retry_context}` テンプレート変数として渡す
- 初回呼び出し時は空文字列になるので影響なし

### Step 3.3: decomposer プロンプトにリトライセクション追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/prompts/query_decomposer.prompt`

末尾に追加:

```
{retry_context}
```

ルール 7 を追加:

```
7. リトライ時（前回のサブタスクが提供された場合）は、前回とは完全に異なるアプローチのサブタスクを生成してください：
   - 前回のサブタスクと同じキーワードや観点を使用しないでください
   - まだ調査されていない新しい角度からクエリを分解してください
```

---

## Phase 4: ログ改善 (MEDIUM)

**目的**: フィルタリング損失やサブタスクごとの検索結果件数を可視化する。

### Step 4.1: `PaperSearchAgent.organizeResults` にフィルタ統計ログ追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/agent/paper-search-agent.ts`

L121-135 の `organizeResults` に追加:

```typescript
const totalCount = processingReadingResults.length;
const relatedCount = readingResults.length;
logger.info(`論文フィルタリング: 全${totalCount}件 → 関連あり${relatedCount}件（除外${totalCount - relatedCount}件）`);
```

### Step 4.2: `PaperProcessor.run` にタスクごとの検索ログ追加

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/paper-processor-chain.ts`

- logger をインポートし、各タスク検索後にログ出力:

```typescript
logger.info(`Task: ${searchedPapers.length}件取得（除外DOI: ${allFoundDois.length}件）`);
```

- ループ完了後:

```typescript
logger.info(`ユニーク論文数: ${uniquePapers.size}件 / タスク数: ${state.tasks.length}`);
```

---

## 実装順序

```
Phase 1 (Step 1.1 → 1.2 → 1.3 → 1.4)  -- 順序依存あり
  ↓
Phase 2 (Step 2.1)                       -- Phase 1 と並行可能
  ↓
Phase 3 (Step 3.1 → 3.2 → 3.3)          -- Phase 2 と同じプロンプトファイルを編集するので Phase 2 後
  ↓
Phase 4 (Step 4.1, 4.2)                  -- いつでも可能
```

## 変更対象ファイル一覧

| ファイル | Phase | 変更内容 |
|----------|-------|----------|
| `searcher/searcher.ts` | 1 | `excludeDois?` パラメータ追加 |
| `rag/qdrant-store.ts` | 1 | `search` に `must_not` フィルタ追加 |
| `rag/rag-searcher.ts` | 1 | `run` に `excludeDois` 伝播 |
| `chains/paper-processor-chain.ts` | 1, 4 | DOI 蓄積ロジック + ログ追加 |
| `chains/prompts/query_decomposer.prompt` | 2, 3 | 多様性ルール + リトライコンテキスト |
| `chains/query-decomposer-chain.ts` | 3 | リトライコンテキスト渡し |
| `agent/paper-search-agent.ts` | 4 | フィルタ統計ログ |

## 検証方法

1. **ユニットレベル**: `QdrantStore.search` に `excludeDois` を渡して、フィルタされた結果が返ることを確認
2. **統合テスト**: 同じクエリで `PaperProcessor.run` を実行し、タスク間で DOI が重複しないことを確認
3. **E2E**: `research-agent.ts` の `main()` を実行し、以下を確認:
   - サブタスクごとに異なる論文がヒットする
   - リトライ時に前回と異なるサブタスクが生成される
   - ログに論文フィルタリング統計が出力される
