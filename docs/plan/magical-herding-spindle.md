# chapter6_python → TypeScript 変換プラン

## Context

`packages/@ai-suburi/core/chapter6_python/` にある arXiv 論文リサーチエージェント（Python）を TypeScript に変換する。
3層の LangGraph エージェント構成（ResearchAgent → PaperSearchAgent → PaperAnalyzerAgent）で、ユーザーの研究要件から arXiv 論文を自動検索・分析・レポート生成するシステム。

既存の chapter4/chapter5 の TypeScript パターン（Annotation, Zod, Command, setupLogger, loadSettings）を踏襲しつつ、`@langchain/openai` の `withStructuredOutput()` を使って Python 版と同じ LangChain ベースの実装にする。

## 決定事項

- **LLM**: `@langchain/openai` の ChatOpenAI + `@langchain/anthropic` の ChatAnthropic
- **プロンプト**: `.prompt` ファイルを維持（10ファイルをそのままコピー）
- **ディレクトリ名**: `chapter6`
- **テスト**: 主要コンポーネントの基本テストを作成

## 新規追加 npm パッケージ

ルートの `package.json` に追加:

| パッケージ | 用途 |
|-----------|------|
| `cohere-ai` | Cohere Reranking API |
| `fast-xml-parser` | arXiv API の Atom フィード解析 |
| `@langchain/anthropic` | ChatAnthropic（レポート生成用） |

※ `@langchain/openai`, `@langchain/langgraph`, `zod` は既に存在

## ディレクトリ構造

```
packages/@ai-suburi/core/chapter6/
├── configs.ts                    # Settings インターフェース + loadSettings()
├── custom-logger.ts              # chapter5 と同じ setupLogger() を再エクスポート
├── models.ts                     # Zod スキーマ: ArxivPaper, ReadingResult, Section, etc.
├── chains/
│   ├── utils.ts                  # loadPrompt(), dictToXmlStr()
│   ├── hearing-chain.ts          # HearingChain
│   ├── goal-optimizer-chain.ts   # GoalOptimizer
│   ├── query-decomposer-chain.ts # QueryDecomposer
│   ├── paper-processor-chain.ts  # PaperProcessor (Send 使用)
│   ├── task-evaluator-chain.ts   # TaskEvaluator
│   ├── reading-chains.ts         # SetSection, CheckSufficiency, Summarizer
│   ├── reporter-chain.ts         # Reporter (ChatAnthropic)
│   └── prompts/                  # Python 版からそのままコピー
│       ├── hearing.prompt
│       ├── goal_optimizer_conversation.prompt
│       ├── goal_optimizer_search.prompt
│       ├── query_decomposer.prompt
│       ├── task_evaluator.prompt
│       ├── set_section.prompt
│       ├── check_sufficiency.prompt
│       ├── summarize.prompt
│       ├── reporter_system.prompt
│       └── reporter_user.prompt
├── searcher/
│   ├── searcher.ts               # Searcher 抽象インターフェース
│   └── arxiv-searcher.ts         # ArxivSearcher (fast-xml-parser + Cohere)
├── service/
│   ├── markdown-parser.ts        # MarkdownParser
│   ├── pdf-to-markdown.ts        # JinaApiClient + PdfToMarkdown
│   └── markdown-storage.ts       # MarkdownStorage
├── agent/
│   ├── research-agent.ts         # ResearchAgent（最上層グラフ）
│   ├── paper-search-agent.ts     # PaperSearchAgent（中層グラフ）
│   └── paper-analyzer-agent.ts   # PaperAnalyzerAgent（下層グラフ）
├── test/
│   ├── models.test.ts            # Zod スキーマのテスト
│   ├── markdown-parser.test.ts   # MarkdownParser のテスト
│   └── arxiv-searcher.test.ts    # ArxivSearcher のテスト
├── fixtures/                     # Python 版からコピー
│   ├── 2408.14317.md
│   ├── sample_goal.txt
│   └── sample_context.txt
└── storage/
    └── markdown/                 # 論文 Markdown 保存先（.gitkeep）
```

## ファイル別実装詳細

### 1. `configs.ts` — 設定管理

Python の `settings.py` に対応。環境変数からの読み込み + LLM インスタンス生成。

```typescript
// 再利用: chapter5/configs.ts のパターン
export interface Settings {
  openaiApiKey: string;
  anthropicApiKey: string;
  cohereApiKey: string;
  jinaApiKey: string;
  debug: boolean;
  // モデル設定
  openaiSmartModel: string;   // default: "gpt-4o"
  openaiFastModel: string;    // default: "gpt-4o-mini"
  anthropicModel: string;     // default: "claude-sonnet-4-20250514"
  cohereRerankModel: string;  // default: "rerank-multilingual-v3.0"
  temperature: number;        // default: 0.0
  // エージェント設定
  maxEvaluationRetryCount: number;  // default: 3
  minDecomposedTasks: number;       // default: 3
  maxDecomposedTasks: number;       // default: 5
  maxSearchRetries: number;         // default: 3
  maxSearchResults: number;         // default: 10
  maxPapers: number;                // default: 3
  maxWorkers: number;               // default: 3
  maxRecursionLimit: number;        // default: 1000
}

export function loadSettings(): Settings { ... }
// LLM インスタンス生成ヘルパー
export function createLlm(settings: Settings): ChatOpenAI { ... }
export function createFastLlm(settings: Settings): ChatOpenAI { ... }
export function createReporterLlm(settings: Settings): ChatAnthropic { ... }
```

### 2. `custom-logger.ts` — ロガー

```typescript
// chapter5/custom-logger.ts と同一の setupLogger を再エクスポート
export { setupLogger, type Logger } from '../chapter5/custom-logger.js';
```

### 3. `models.ts` — Zod スキーマ + TypeScript 型

Python の `models/` 内の3つの Pydantic モデルを Zod スキーマに変換。

```typescript
// ArxivPaper
export const arxivPaperSchema = z.object({
  id: z.string(), title: z.string(), link: z.string(),
  pdfLink: z.string(), abstract: z.string(),
  published: z.string(), updated: z.string(),  // ISO string
  version: z.number(), authors: z.array(z.string()),
  categories: z.array(z.string()),
  relevanceScore: z.number().nullable().optional(),
});
export type ArxivPaper = z.infer<typeof arxivPaperSchema>;
export function arxivPaperToXml(paper: ArxivPaper): string { ... }

// ReadingResult
export const readingResultSchema = z.object({
  id: z.number(), task: z.string(),
  paper: arxivPaperSchema,
  markdownPath: z.string(),
  answer: z.string().default(""),
  isRelated: z.boolean().nullable().optional(),
});
export type ReadingResult = z.infer<typeof readingResultSchema>;

// Section
export const sectionSchema = z.object({
  header: z.string(), content: z.string(), charCount: z.number(),
});
export type Section = z.infer<typeof sectionSchema>;

// LLM 構造化出力用の Zod スキーマ
export const hearingSchema = z.object({
  is_need_human_feedback: z.boolean(),
  additional_question: z.string(),
});
export const decomposedTasksSchema = z.object({
  tasks: z.array(z.string()),
});
export const taskEvaluationSchema = z.object({
  need_more_information: z.boolean(),
  reason: z.string(),
  content: z.string(),
});
export const sufficiencySchema = z.object({
  is_sufficient: z.boolean(),
  reason: z.string(),
});
export const arxivFieldsSchema = z.object({
  values: z.array(z.string()),
});
export const arxivTimeRangeSchema = z.object({
  start: z.string().nullable().optional(),
  end: z.string().nullable().optional(),
});
```

### 4. `chains/utils.ts` — ユーティリティ

```typescript
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

export function loadPrompt(name: string): string {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const promptPath = path.join(__dirname, 'prompts', `${name}.prompt`);
  return fs.readFileSync(promptPath, 'utf-8').trim();
}

export function dictToXmlStr(data: Record<string, unknown>, excludeKeys: string[] = []): string {
  // Python 版と同じ XML 変換ロジック
}
```

### 5. `chains/hearing-chain.ts` — ユーザーヒアリング

- Python: `HearingChain.__call__` → TS: `hearingChain(state)` 関数
- `ChatPromptTemplate.fromTemplate()` + `llm.withStructuredOutput(hearingSchema)` を使用
- `Command` で `human_feedback` or `goal_setting` へ遷移

### 6. `chains/goal-optimizer-chain.ts` — ゴール最適化

- `ChatPromptTemplate.fromTemplate()` + `llm.pipe(new StringOutputParser())` でテキスト出力
- mode に応じて `goal_optimizer_conversation.prompt` / `goal_optimizer_search.prompt` を使い分け
- `Command` で `decompose_query` へ遷移

### 7. `chains/query-decomposer-chain.ts` — クエリ分解

- `llm.withStructuredOutput(decomposedTasksSchema)` で構造化出力
- `Command` で `paper_search_agent` へ遷移

### 8. `chains/paper-processor-chain.ts` — 論文処理 + Send

- `Send()` で各 `ReadingResult` を `analyze_paper` ノードへ並列送信
- `Promise.all()` で PDF→Markdown 並列変換（Python の `ThreadPoolExecutor` 相当）
- arXiv 検索 → 重複排除 → PDF 変換 → ReadingResult 生成

### 9. `chains/task-evaluator-chain.ts` — タスク評価

- `llm.withStructuredOutput(taskEvaluationSchema)` で構造化出力
- `Command` で `decompose_query`（リトライ）or `generate_report` へ遷移

### 10. `chains/reading-chains.ts` — 論文分析チェーン

3クラスを1ファイルに:
- **SetSection**: セクション番号選択 → `check_sufficiency` へ
- **CheckSufficiency**: 十分性判定 → `summarize` / `set_section`（ループ）/ `mark_as_not_related` へ
- **Summarizer**: サマリー生成 → `reading_result` 更新

### 11. `chains/reporter-chain.ts` — レポート生成

- `ChatAnthropic` を使用（Python 版と同じ）
- `ChatPromptTemplate.fromMessages()` で system + user プロンプト
- `StringOutputParser()` でテキスト出力

### 12. `searcher/searcher.ts` — 抽象インターフェース

```typescript
export interface Searcher {
  run(goalSetting: string, query: string): Promise<ArxivPaper[]>;
}
```

### 13. `searcher/arxiv-searcher.ts` — arXiv 検索実装

- `fast-xml-parser` で arXiv Atom API レスポンスを解析（feedparser 代替）
- `fetch()` で arXiv API にリクエスト
- `cohere-ai` SDK で Reranking
- LLM による日付範囲選択・クエリ拡張（`withStructuredOutput` 使用）
- リトライロジック（while ループ）

### 14. `service/markdown-parser.ts` — Markdown 解析

- Python 版と同じ正規表現ベースのセクション分割
- `parseSections()`, `formatAsXml()`, `getSectionsOverview()`, `getSelectedSections()`

### 15. `service/pdf-to-markdown.ts` — PDF→Markdown 変換

- `fetch()` で Jina Reader API を呼び出し（Python の `requests` 代替）
- MarkdownStorage でキャッシュ

### 16. `service/markdown-storage.ts` — Markdown ストレージ

- `fs.writeFileSync()` / `fs.readFileSync()` でファイル I/O
- `storage/markdown/` にファイル保存

### 17. `agent/research-agent.ts` — ResearchAgent（最上層）

状態定義（Annotation.Root）:
```typescript
const ResearchAgentAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({ reducer: messagesStateReducer, default: () => [] }),
  hearing: Annotation<Hearing>,
  goal: Annotation<string>,
  tasks: Annotation<string[]>({ reducer: (_p, n) => n, default: () => [] }),
  readingResults: Annotation<ReadingResult[]>({ reducer: (_p, n) => n, default: () => [] }),
  evaluation: Annotation<TaskEvaluation>,
  finalOutput: Annotation<string>,
  retryCount: Annotation<number>({ reducer: (_p, n) => n, default: () => 0 }),
});
```

グラフ構築:
- ノード: `user_hearing` → `human_feedback`（interrupt）→ `goal_setting` → `decompose_query` → `paper_search_agent`（サブグラフ invoke）→ `evaluate_task` → `generate_report`
- `MemorySaver` でチェックポイント
- input/output 型の分離

### 18. `agent/paper-search-agent.ts` — PaperSearchAgent（中層）

状態定義:
```typescript
const PaperSearchAgentAnnotation = Annotation.Root({
  goal: Annotation<string>,
  tasks: Annotation<string[]>,
  processingReadingResults: Annotation<ReadingResult[]>({
    reducer: (prev, next) => [...prev, ...next],  // operator.add 相当
    default: () => [],
  }),
  readingResults: Annotation<ReadingResult[]>({ reducer: (_p, n) => n, default: () => [] }),
});
```

グラフ構築:
- `search_papers` → `analyze_paper`（Send で並列）→ `organize_results`
- `organize_results` で `isRelated === true` のみフィルタ

### 19. `agent/paper-analyzer-agent.ts` — PaperAnalyzerAgent（下層）

状態定義:
```typescript
const PaperAnalyzerAgentAnnotation = Annotation.Root({
  goal: Annotation<string>,
  readingResult: Annotation<ReadingResult>,
  selectedSectionIndices: Annotation<number[]>({ reducer: (_p, n) => n, default: () => [] }),
  sufficiency: Annotation<Sufficiency>,
  checkCount: Annotation<number>({ reducer: (_p, n) => n, default: () => 0 }),
});
```

グラフ構築:
- `set_section` → `check_sufficiency` → `summarize` / `mark_as_not_related` / `set_section`（ループ）

## 実装順序

依存関係に基づいた実装順:

1. **npm パッケージ追加**: `cohere-ai`, `fast-xml-parser`, `@langchain/anthropic` を `pnpm add`
2. **基盤ファイル**: `configs.ts`, `custom-logger.ts`, `models.ts`
3. **ユーティリティ**: `chains/utils.ts` + `chains/prompts/` (コピー)
4. **サービス層**: `markdown-storage.ts` → `markdown-parser.ts` → `pdf-to-markdown.ts`
5. **検索層**: `searcher/searcher.ts` → `searcher/arxiv-searcher.ts`
6. **チェーン層**: `hearing-chain.ts` → `goal-optimizer-chain.ts` → `query-decomposer-chain.ts` → `task-evaluator-chain.ts` → `reading-chains.ts` → `reporter-chain.ts` → `paper-processor-chain.ts`
7. **エージェント層**: `paper-analyzer-agent.ts` → `paper-search-agent.ts` → `research-agent.ts`
8. **テスト**: `models.test.ts`, `markdown-parser.test.ts`, `arxiv-searcher.test.ts`
9. **フィクスチャ**: `fixtures/` をコピー + `storage/markdown/.gitkeep`

## Python → TypeScript 変換マッピング

| Python | TypeScript |
|--------|-----------|
| `Pydantic BaseModel` | Zod スキーマ + `z.infer` |
| `TypedDict` | Annotation.Root |
| `Annotated[list, operator.add]` | `Annotation<T[]>({ reducer: (p, n) => [...p, ...n] })` |
| `ChatPromptTemplate.from_template()` | `ChatPromptTemplate.fromTemplate()` |
| `llm.with_structured_output(Model)` | `llm.withStructuredOutput(zodSchema)` |
| `StrOutputParser()` | `new StringOutputParser()` |
| `Command(goto=..., update={...})` | `new Command({ goto: ..., update: {...} })` |
| `Send("node", state)` | `new Send("node", state)` |
| `interrupt(value)` | `interrupt(value)` |
| `StateGraph(state, input, output)` | `new StateGraph(Annotation)` |
| `workflow.set_entry_point("node")` | `.addEdge('__start__', 'node')` |
| `workflow.set_finish_point("node")` | `.addEdge('node', '__end__')` 等 |
| `feedparser.parse(url)` | `fetch(url)` + `XMLParser.parse()` |
| `concurrent.futures.ThreadPoolExecutor` | `Promise.all()` |
| `cohere.Client.rerank()` | `CohereClient.v2.rerank()` |
| `requests.get()` | `fetch()` |
| `Path(__file__).parent` | `path.dirname(fileURLToPath(import.meta.url))` |
| `os.makedirs()` | `fs.mkdirSync(dir, { recursive: true })` |

## 再利用する既存コード

| ファイル | 再利用内容 |
|---------|-----------|
| `chapter5/custom-logger.ts` | `setupLogger()`, `Logger` インターフェース |
| `chapter5/configs.ts` | `loadSettings()` パターン |
| `chapter5/graph/state.ts` | Annotation.Root パターン |
| `chapter5/graph/data-analysis.ts` | StateGraph 構築 + Command パターン |
| `chapter3/test3-11-text-to-sql-langchain.ts:161` | `withStructuredOutput(zodSchema)` パターン |

## 検証方法

### 型チェック
```bash
npx tsc --noEmit
```

### 基本テスト
```bash
npx tsx packages/@ai-suburi/core/chapter6/test/models.test.ts
npx tsx packages/@ai-suburi/core/chapter6/test/markdown-parser.test.ts
```

### エージェント単体実行
```bash
# PaperAnalyzerAgent（fixtures 使用）
npx tsx packages/@ai-suburi/core/chapter6/agent/paper-analyzer-agent.ts fixtures/2408.14317.md

# PaperSearchAgent
npx tsx packages/@ai-suburi/core/chapter6/agent/paper-search-agent.ts

# ResearchAgent（フルフロー）
npx tsx packages/@ai-suburi/core/chapter6/agent/research-agent.ts
```

### 必要な環境変数
```
OPENAI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY, JINA_API_KEY
```
