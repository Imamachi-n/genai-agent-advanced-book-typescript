---
name: cleanup-code
description: 指定されたTypeScriptファイルの型エラー修正・非推奨APIの置換・未使用importの削除・JSDocの追加を行い、コードを整理する。
allowed-tools: Read, Edit, Glob, Grep, Bash
argument-hint: <TypeScriptファイルのパス>
---

# コード整理スキル

TypeScript ファイルのパスを受け取り、型エラー修正・非推奨 API の置換・未使用 import 削除・JSDoc 追加の 4 ステップでコードを整理する。

## 手順

### Step 1: 対象ファイルの読み込み

1. `$ARGUMENTS` のファイルを読み込み、内容を把握する
2. ファイルが存在しない場合はエラーを報告して終了する

### Step 2: 型エラーの修正

1. `npx tsc --noEmit --skipLibCheck --pretty 2>&1 | grep -E "<対象ファイル名>"` でプロジェクト全体をビルドし、対象ファイルのエラーのみ抽出する
   - 単一ファイル指定（`npx tsc --noEmit <ファイルパス>`）はプロジェクトの tsconfig が無視される場合があるため使わない
   - `--skipLibCheck` で `node_modules` 内のエラーを除外する
2. 型エラーがある場合、エラー内容を分析して修正する
3. 修正パターンの例:
   - `Object is possibly 'undefined'`（TS2532）→ オプショナルチェーン（`?.`）+ Nullish Coalescing（`??`）で対応する
   - ユニオン型でプロパティにアクセスできない（TS2339）→ 判別プロパティで型を絞り込む（例: `role === 'assistant'`、`type === 'function'`）
   - 引数の型不一致 → 正しい型への修正
   - 存在しないプロパティへのアクセス → プロパティ名の修正
   - `import type` と `import` の使い分け誤り → 適切な import 形式に修正
4. 修正後に再度型チェックコマンドで型エラーが解消されたことを確認する
5. 型変更により呼び出し元（テストファイル等）にもエラーが波及する場合がある。ディレクトリ全体でも grep して確認する

### Step 3: 非推奨 API の検出・置換

1. ファイル内で `@deprecated` 警告が出ているメソッド・プロパティを検出する
2. 検出方法:
   - IDE の diagnostics（TS6387: deprecated）を確認する
   - 使用しているライブラリの型定義（`.d.ts`）を `Grep` で調べ、`@deprecated` コメントと推奨される代替 API を特定する
3. 代替 API が明確な場合は置換する
   - 同期 → 非同期への変更（例: `getGraph()` → `await getGraphAsync()`）の場合、呼び出し元が `async` 関数内であることを確認する
   - 引数・戻り値の型が変わる場合は呼び出し側も合わせて修正する
4. 代替 API が不明確な場合はユーザーに確認を取る

### Step 4: 未使用 import の削除

1. 修正後のファイルを読み込む
2. import されているシンボルがファイル内で実際に使用されているか確認する
3. 未使用の import があれば削除する
   - 名前付き import の一部が未使用 → その名前だけを削除する
   - import 文全体が未使用 → import 文ごと削除する
4. 削除後、空行が連続しないよう整形する

### Step 5: JSDoc の追加

1. ファイル内のすべての関数宣言（`function`）を検出する
2. 既に JSDoc がある関数はスキップする
3. JSDoc がない関数に以下の形式で追加する:

```typescript
/**
 * <関数の役割を1文で説明>
 * @param <引数名> - <引数の説明>
 * @returns <戻り値の説明>
 */
```

4. JSDoc 記述のルール:
   - 説明文は日本語で記述する
   - `@param` は関数に引数がある場合のみ記述する
   - `@returns` は戻り値がある場合のみ記述する（`void` / `Promise<void>` の場合は省略可）
   - 既存の `//` コメントがある場合、その内容を参考にして JSDoc を作成し、冗長な `//` コメントは削除する

### Step 6: 最終確認

1. `npx tsc --noEmit --skipLibCheck --pretty 2>&1 | grep -E "<対象ファイル名>"` で対象ファイルの型エラーがないことを最終確認する
2. 呼び出し元のファイル（テストファイル等）にもエラーが波及していないか、ディレクトリ全体で確認する
3. ファイルの整形が崩れていないか確認する

## 修正時のルール

- ロジックの変更は行わない（型アノテーションや import の整理のみ）
- 既存のコードスタイル（インデント・クォート・セミコロン等）を維持する
- 不確実な修正は行わず、ユーザーに確認を取る
- `any` 型は使わない。代わりにライブラリが提供する型（例: `StructuredToolInterface`）や `unknown` + 型ガードを使う
- 型アサーション（`as X`）は使わない。代わりに判別プロパティによる型の絞り込み（discriminated union narrowing）を使う
- 非 null アサーション（感嘆符）は使わない。代わりにオプショナルチェーン（`?.`）や Nullish Coalescing（`??`）を使う
- `undefined` の可能性がある値に対して `throw` によるガード節は使わない。`?.` と `??` で対応する

```typescript
// NG: any 型
type Foo = { invoke(input: any): Promise<any> };

// OK: ライブラリ型 or unknown
import type { StructuredToolInterface } from '@langchain/core/tools';
type Foo = { invoke(input: Record<string, unknown>): Promise<unknown> };

// NG: 型アサーション
(lastMessage as any).tool_calls
openaiTools as any

// OK: 判別プロパティで型を絞り込む
if (lastMessage?.role === 'assistant' && 'tool_calls' in lastMessage) {
  lastMessage.tool_calls; // 型が絞り込まれる
}
if (toolCall.type === 'function') {
  toolCall.function.name; // 型が絞り込まれる
}

// NG: 非 null アサーション / throw ガード
response.choices[0]!.message.content
const choice = response.choices[0];
if (!choice) throw new Error('...');

// OK: オプショナルチェーン + Nullish Coalescing
response.choices[0]?.message.content ?? ''
state.plan[state.currentStep] ?? ''
```

## SKILL.md 記述時の注意

このファイルの内容はシェル（zsh）で評価される場合がある。以下の文字はコードブロックの外では使わないこと。

- `**` → zsh の再帰 glob として展開される（マークダウンの太字には使わない）
- 感嘆符 → zsh の history expansion として解釈される（コードブロック内のみ可）
- `$(...)` → コマンド置換として実行される
- バッククォート連続 → コマンド置換として実行される場合がある

## 出力

整理完了後、以下の形式でサマリーを出力する:

```
## コード整理結果サマリー

### 型エラー修正
- [ 修正内容1 ]
- [ 修正内容2 ]
- （型エラーなしの場合は「型エラーなし ✅」）

### 非推奨 API の置換
- [ 置換した API ]
- （非推奨 API なしの場合は「非推奨 API なし ✅」）

### 未使用 import 削除
- [ 削除した import ]
- （未使用 import なしの場合は「未使用 import なし ✅」）

### JSDoc 追加
- [ JSDoc を追加した関数名 ]
- （全関数に JSDoc ありの場合は「追加不要 ✅」）

### 最終確認
- 型チェック: ✅ / ❌
```
