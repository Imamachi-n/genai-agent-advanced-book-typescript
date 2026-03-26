# Chapter 6 ドキュメント追記プラン

## Context

Chapter 6（arXiv 論文リサーチ AI エージェント）のソースコードは `packages/@ai-suburi/core/chapter6/` に実装済みだが、対応するドキュメントが存在しない。Chapter 5 のドキュメントスタイルに準拠した `chapter6.md` を新規作成し、フッターリンクも追加する。

## 作成するファイル

### 1. `packages/@ai-suburi/docs/docs/ai-agent-practice/chapter6.md`（新規作成）

#### ドキュメント構成

```
---
sidebar_position: 5
---

# Chapter 6: arXiv 論文リサーチ AI エージェントの実装
```

**導入文（2〜3段落）:**
- 研究者が arXiv 論文を調査する際の課題（大量の論文、時間がかかる、関連論文の見落とし）
- Chapter 5 との接続（データ分析 → 論文リサーチへ対象を拡大）
- このエージェントで組み合わせる技術（LangGraph マルチエージェント、arXiv API、Cohere Rerank、Jina Reader API、Structured Outputs）

**:::note この章で学ぶこと:**
- LangGraph による**マルチエージェント**アーキテクチャ（メインエージェント + サブエージェント 2 層）
- arXiv API を使った**学術論文検索**と LLM によるクエリ最適化
- **Cohere Rerank** によるセマンティックリランキング
- **Jina Reader API** による PDF → Markdown 変換
- LangGraph の **Send API** による動的な並列タスク分配
- `interrupt()` による **human-in-the-loop**（ヒアリングフェーズ）
- **Structured Outputs**（Zod スキーマ）による型安全な LLM 出力
- 調査結果の自動評価と**再検索ループ**（TaskEvaluator）

#### セクション一覧（学習の流れ）

| セクション | 内容 | キーワード |
| --- | --- | --- |
| 6-1 | 型定義と Zod スキーマ（ArxivPaper, ReadingResult, Hearing 等） | Zod, Structured Outputs |
| 6-2 | 設定管理と LLM インスタンス生成（OpenAI + Anthropic + Cohere） | configs, マルチプロバイダ |
| 6-3 | arXiv 論文検索とクエリ最適化（フィールド選択・日付絞り込み・リトライ） | arXiv API, Cohere Rerank |
| 6-4 | PDF → Markdown 変換とストレージ管理 | Jina Reader API, キャッシュ |
| 6-5 | Markdown パーサーとセクション抽出 | MarkdownParser, XML フォーマット |
| 6-6 | ヒアリング・ゴール最適化・クエリ分解チェーン | HearingChain, GoalOptimizer, QueryDecomposer |
| 6-7 | 論文分析エージェント（セクション選択 → 十分性チェック → 要約） | PaperAnalyzerAgent, SetSection, CheckSufficiency |
| 6-8 | 論文検索エージェント（検索 → PDF 変換 → 分析の並列実行） | PaperSearchAgent, Send API, PaperProcessor |
| 6-9 | タスク評価と再検索ループ | TaskEvaluator, 再分解 |
| 6-10 | レポート生成（Claude Sonnet 4） | Reporter, ChatAnthropic |
| 6-11 | メインエージェント（ResearchAgent）と LangGraph ワークフロー全体 | StateGraph, interrupt, human-in-the-loop |

**:::info 前提条件:**
- `OPENAI_API_KEY`（OpenAI）
- `ANTHROPIC_API_KEY`（Anthropic）
- `COHERE_API_KEY`（Cohere）
- `JINA_API_KEY`（Jina Reader）
- `@langchain/langgraph`, `@langchain/openai`, `@langchain/anthropic`, `cohere-ai`, `fast-xml-parser` がインストール済み

**サンプルコードの実行方法:**
```bash
pnpm tsx chapter6/agent/research-agent.ts "LLMエージェントの評価方法について調べる"
```

**エージェントの構成ファイルテーブル:** 全ファイルの一覧と役割

**Mermaid 図:**

1. **メインワークフロー図**: ResearchAgent の全体フロー
   - user_hearing → human_feedback（interrupt）→ goal_setting → decompose_query → paper_search_agent → evaluate_task → generate_report
   - evaluate_task から decompose_query への再検索ループ

2. **PaperSearchAgent サブグラフ図**:
   - search_papers → analyze_paper（Send で並列）→ organize_results

3. **PaperAnalyzerAgent サブグラフ図**:
   - set_section → check_sufficiency → summarize or mark_as_not_related
   - check_sufficiency から set_section への再試行ループ

**各セクションの詳細:**
- セクションごとにソースコード全文を `typescript title="chapter6/..."` で掲載
- 各ファイルの説明（概要 + ポイント）
- 実行方法がある場合は `pnpm tsx` コマンドを記載

**参考文献:**
- LangChain/LangGraph 公式ドキュメント
- arXiv API ドキュメント
- Cohere Rerank API
- Jina Reader API
- Zod v4

### 2. `packages/@ai-suburi/docs/docusaurus.config.ts`（フッターリンク追加）

「AI エージェント実践入門」カテゴリの `items` 末尾に追加:

```typescript
{
  label: 'Chapter 6: arXiv論文リサーチAIエージェントの実装',
  to: '/docs/ai-agent-practice/chapter6',
},
```

## 対象ファイル一覧

| 操作 | ファイルパス |
| --- | --- |
| 新規作成 | `packages/@ai-suburi/docs/docs/ai-agent-practice/chapter6.md` |
| 編集 | `packages/@ai-suburi/docs/docusaurus.config.ts`（フッターリンク追加） |

## ソースコードファイル（ドキュメントに掲載するコード）

| ファイル | セクション |
| --- | --- |
| `chapter6/models.ts` | 6-1 |
| `chapter6/configs.ts` | 6-2 |
| `chapter6/custom-logger.ts` | 6-2 |
| `chapter6/searcher/searcher.ts` | 6-3 |
| `chapter6/searcher/arxiv-searcher.ts` | 6-3 |
| `chapter6/service/pdf-to-markdown.ts` | 6-4 |
| `chapter6/service/markdown-storage.ts` | 6-4 |
| `chapter6/service/markdown-parser.ts` | 6-5 |
| `chapter6/chains/utils.ts` | 6-6 |
| `chapter6/chains/hearing-chain.ts` | 6-6 |
| `chapter6/chains/goal-optimizer-chain.ts` | 6-6 |
| `chapter6/chains/query-decomposer-chain.ts` | 6-6 |
| `chapter6/chains/reading-chains.ts` | 6-7 |
| `chapter6/chains/paper-processor-chain.ts` | 6-8 |
| `chapter6/agent/paper-analyzer-agent.ts` | 6-7 |
| `chapter6/agent/paper-search-agent.ts` | 6-8 |
| `chapter6/chains/task-evaluator-chain.ts` | 6-9 |
| `chapter6/chains/reporter-chain.ts` | 6-10 |
| `chapter6/agent/research-agent.ts` | 6-11 |

## 検証方法

1. `pnpm --filter @ai-suburi/docs build` でドキュメントがビルドできることを確認
2. サイドバーに Chapter 6 が表示されることを確認
3. フッターに Chapter 6 のリンクが表示されることを確認
4. Mermaid 図が正しくレンダリングされることを確認
5. コードブロックのシンタックスハイライトが正しいことを確認
