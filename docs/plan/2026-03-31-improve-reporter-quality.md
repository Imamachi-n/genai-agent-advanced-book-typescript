# レポート品質改善のプラン（具体性向上 + 論文 URL 表示）

## Context

簡易モードでレポートを生成したところ、2 つの問題が発生:

1. **具体性に欠ける**: 「論文ID: 0」のような抽象的な参照のみで、論文タイトルや具体的内容が薄い
2. **論文 URL がない**: 参考文献リストに実際の論文 URL（bioRxiv リンク）が含まれていない

### 根本原因

`reporter-chain.ts` で ReadingResult を XML に変換する際に `dictToXmlStr()` を使用しているが、この関数はネストされたオブジェクトを `${value}` でシリアライズするため、`paper` フィールド（BiorxivPaper オブジェクト）が **`[object Object]`** になってしまう。

結果として、Reporter LLM にタイトル・DOI・URL・著者・アブストラクト等の論文メタデータが一切渡されていない。

---

## 改善点 1: Reporter のコンテキスト生成で論文メタデータを正しくシリアライズ

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/reporter-chain.ts`

既に `models.ts` に `biorxivPaperToXml()` 関数が存在する。これを使って ReadingResult の `paper` フィールドを正しく XML 化する。

現在のコード (L22-27):
```typescript
const context = results
  .map((item) => {
    const { markdownPath: _mp, ...rest } = item;
    return dictToXmlStr(rest as unknown as Record<string, unknown>);
  })
  .join('\n');
```

変更後: ReadingResult を専用のシリアライズ関数で変換する。`paper` フィールドを `biorxivPaperToXml()` で展開し、その他のフィールドと組み合わせて XML を構築する。

```typescript
const context = results
  .map((item) => {
    return `<item>
<id>${item.id}</id>
<task>${item.task}</task>
${biorxivPaperToXml(item.paper)}
<answer>${item.answer}</answer>
<is_related>${item.isRelated}</is_related>
</item>`;
  })
  .join('\n');
```

これにより Reporter LLM が `<paper>` 内の `<link>`, `<doi>`, `<title>`, `<authors>`, `<abstract>` 等に直接アクセスできるようになる。

---

## 改善点 2: Reporter プロンプトに論文 URL 表示と具体性の指示を強化

**ファイル**: `packages/@ai-suburi/core/chapter6-biorxiv/chains/prompts/reporter_user.prompt`

### 2a: 参考文献リストの URL 表示を明確化

`<quality_checklist>` セクションの項目 6 を具体化:

```
6. すべての引用された作品の完全な参考文献リストを末尾に作成してください。各論文は以下の形式で記載してください：
   - タイトル、著者、公開日、DOI、URL（paper タグ内の link フィールドを使用）
   - 例: [論文タイトル](https://doi.org/xxx) - 著者名 (公開日)
```

### 2b: 具体性の向上指示を追加

`<evaluations>` セクションに追加:

```
4. 各論文の具体的な手法名、技術名、データセット名、結果の数値を引用していると高評価です。
5. 論文のタイトルとDOIを明示的に引用していると高評価です（論文IDだけの参照は低評価です）。
```

---

## 変更対象ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `chains/reporter-chain.ts` | `biorxivPaperToXml()` を使ったコンテキスト生成に変更 |
| `chains/prompts/reporter_user.prompt` | 論文 URL 表示・具体性の指示を強化 |

## 検証方法

1. 簡易モードで実行し、レポートに論文タイトル・DOI・URL が含まれることを確認
2. 参考文献リストが末尾に `[タイトル](URL)` 形式で表示されることを確認
3. 「論文ID: 0」のような抽象的な参照ではなく具体的な論文名で引用されていることを確認
