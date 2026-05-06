# FTモデル改善プラン: bioRxiv RAG検索クエリ変換

## Context

bioRxiv論文のRAG検索システムで、日本語クエリ→英語検索クエリ変換をファインチューニング（FT）で改善しようとしたが、デフォルトモデル（gpt-4o-mini）との差が感じられなかった。コードとデータを分析した結果、**学習データの設計に7つの構造的問題**を特定した。

---

## 特定した問題点

### 問題1: goalの不一致（影響度: 最大）
- **学習データ**: `${category}分野の研究動向を調査する` で常に固定（[training-data-formatter.ts:109](packages/@ai-suburi/core/chapter6-biorxiv/rag/ft-pipeline/training-data-formatter.ts#L109)）
- **推論時**: ユーザーが入力する具体的な目標（例: "生成AIを用いたゲノム解析の最新動向"）
- FTモデルは汎用goalしか見たことがなく、具体的goalに対応できていない

### 問題2: 全クエリタイプに同一のidealQuery
- 1論文につきidealQueryは1つだけ（[generate-ft-data.ts:76-79](packages/@ai-suburi/core/chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts#L76-L79)）
- keyword型もquestion型もtask型も同じ出力を学習 → クエリ意図の違いを反映できない

### 問題3: Embedding品質検証がデフォルトでスキップ
- `skipEmbeddingCheck = true`（[generate-ft-data.ts:108](packages/@ai-suburi/core/chapter6-biorxiv/rag/ft-pipeline/generate-ft-data.ts#L108)）
- 理想クエリが本当にその論文をヒットできるかの品質保証がない

### 問題4: Synthetic-to-Synthetic問題
- 合成クエリも理想クエリもLLM生成 → 人間の実際のクエリパターンを反映していない

### 問題5: ハードネガティブ/困難例の欠如
- ベースモデルが既に正しく変換できる簡単なケースばかり学習している
- FTの付加価値が出ない

### 問題6: 体系的なA/B評価がない
- 「差を感じない」という主観的判断しかできない状態

### 問題7: リランキングとexpandQueryの整合性問題
- rerankは元の日本語クエリ+goalでスコアリング（[rag-searcher.ts:164](packages/@ai-suburi/core/chapter6-biorxiv/rag/rag-searcher.ts#L164)）
- expandQueryの改善がrerank後の結果に直接反映されない

---

## 改善プラン（優先度順）

### Phase 1: 基盤整備（効果: High / コスト: Low）

#### 1-A. A/B評価基盤の構築
**なぜ最初か**: 以降の改善の効果を定量的に測定するため

- 新規 `ft-pipeline/evaluation.ts` を作成
- 評価データセット `storage/eval-queries.jsonl` を手動で20-30件作成（日本語クエリ + 期待DOI）
- 評価指標: Recall@3, Recall@10, MRR（Mean Reciprocal Rank）
- CLIオプション: `--model-a` `--model-b` `--eval-data`

#### 1-B. goal多様化
**最もコスパが高い改善**

- 新規 `ft-pipeline/goal-synthesizer.ts` を作成
- LLMで論文ごとに3-5パターンのgoalを生成（汎用型/具体型/応用型/課題解決型）
- `training-data-formatter.ts` の `writeEntry()` を修正してgoalを多様化
- 1論文あたりの学習データ量: 3クエリ × 1goal → 3クエリ × 3-5goal = 9-15例

### Phase 2: データ品質改善（効果: High / コスト: Medium）

#### 2-A. クエリタイプ別idealQuery生成
- `ideal-query-generator.ts` のインターフェースを変更: `(paper, syntheticQuery) => Promise<string>`
- プロンプトにユーザーの合成クエリ内容とタイプを含める
- `generate-ft-data.ts` のフロー変更: 合成クエリ生成 → 各クエリに対して個別にidealQuery生成

#### 2-B. Embedding品質検証のデフォルト有効化
- `skipEmbeddingCheck` のデフォルトを `false` に変更
- 論文EmbeddingのキャッシュMap追加で速度低下を抑制
- CLIフラグ名を `--skip-embedding-check`（明示スキップ）に変更

### Phase 3: 高度な改善（効果: Medium / コスト: Medium-High）

#### 3-A. rerank整合性改善
- `rag-searcher.ts` の `run()` で、rerankクエリにexpandedQueryを組み合わせる
- 変更: `const searchQuery = \`${expandedQuery}\n${goalSetting}: ${query}\``

#### 3-B. ハードネガティブ学習
- Phase 1の評価基盤でベースモデルの弱点（Recall漏れ）を特定
- 困難例に対してGPT-4クラスで高品質idealQueryを生成
- 困難例を全体の20-30%混入

### Phase 4: 長期課題（効果: Medium-Low / コスト: High）

#### 4-A. Synthetic-to-Synthetic問題の緩和
- 合成クエリにノイズ注入（タイポ、省略表現、ひらがな化）
- 実際のユーザーログ収集 → 学習データへの反映

---

## 改善効果の見込み

| Phase | 改善 | 効果 | コスト | 主な変更ファイル |
|-------|------|------|--------|-----------------|
| 1-A | A/B評価基盤 | High | Low | 新規: evaluation.ts |
| 1-B | goal多様化 | High | Low | training-data-formatter.ts, 新規: goal-synthesizer.ts |
| 2-A | クエリ別idealQuery | High | Medium | ideal-query-generator.ts, generate-ft-data.ts |
| 2-B | Embedding検証有効化 | Medium | Low | generate-ft-data.ts, ideal-query-generator.ts |
| 3-A | rerank整合性 | Medium | Medium | rag-searcher.ts |
| 3-B | ハードネガティブ | Medium | Med-High | 新規: hard-example-miner.ts |
| 4-A | ノイズ注入 | Med-Low | High | query-synthesizer.ts |

---

## 検証方法

1. Phase 1-Aの評価基盤で、改善前のベースライン指標を取得
2. 各Phase実施後にFTモデルを再学習し、同じ評価セットでRecall@3/10を比較
3. ログに出力される検索クエリの質を目視で確認（ログ例と比較）
4. 実際のユーザーシナリオ（"生成AIを用いたゲノム解析の最新動向"等）で動作確認
