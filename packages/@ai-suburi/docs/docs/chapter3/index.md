---
sidebar_position: 1
---

# Chapter 3: AIエージェントの開発準備

この章では、AI エージェントを構築するための土台となる OpenAI API の基本的な使い方を、実際のコードを動かしながら学んでいきます。
[Chapter 2](../chapter2/index.md) で解説した「プロフィール」「メモリ」「ツール」「プランニング」の各コンポーネントが、API レベルではどのように実現されるのかを体感できる内容になっています。

:::note この章で学ぶこと

- **Chat Completions API** の基本的な呼び出し方とトークン使用量の確認
- **JSON モード**と **Structured Outputs** による構造化された出力の取得
- **Function Calling** を使った外部ツールとの連携パターン
- **Tavily API** を使った AI エージェント向けの Web 検索
- **LangChain** の `tool` ヘルパーによるカスタム Tool 定義
- **DuckDuckGo** を使った無料の Web 検索とページ取得

:::

## 概要

以下のトピックを扱います。セクション番号は参考元の書籍に合わせています。

| セクション | 内容 |
| --- | --- |
| 3-1 | Chat Completions API の基本的な使い方 |
| 3-3 | JSON モードによる構造化された出力 |
| 3-4 | Zod スキーマを用いた Structured Outputs |
| 3-6 | Function Calling による外部ツール連携 |
| 3-7 | Tavily API を使った Web 検索 |
| 3-8 | LangChain カスタム Tool 定義 |
| 3-9 | DuckDuckGo Web 検索 |

:::info 前提条件

- 環境変数 `OPENAI_API_KEY` に OpenAI の API キーが設定されていること
- 3-7 のみ、環境変数 `TAVILY_API_KEY` に Tavily の API キーが設定されていること

:::

### サンプルコードの実行方法

各サンプルは、リポジトリのルートディレクトリから以下のコマンドで実行できます。

```bash
# ルートディレクトリで実行（pnpm tsx は @ai-suburi/core パッケージ内で tsx を実行するエイリアス）
pnpm tsx chapter3/<ファイル名>.ts
```

## 3-1. Chat Completions API

Chat Completions API は、OpenAI のチャットモデルと対話するための最も基本的な API です。
AI エージェントを構築する際、すべての対話はこの API を通じて行われるため、まずここでの基本操作をしっかり押さえておくことが重要です。

`messages` 配列にロール（`system`, `user`, `assistant`）とメッセージを渡すことで、モデルからの応答を取得できます。

このサンプルでは以下を行います。

- `gpt-4o` モデルへのメッセージ送信
- 応答テキストの取得
- トークン使用量（プロンプト / 生成 / 合計）の確認

:::info トークンとは？
トークンは、モデルがテキストを処理する際の最小単位です。英語では 1 単語が約 1 トークン、日本語ではひらがな 1 文字が約 1 トークンに相当します。API の利用料金はトークン数に基づいて計算されるため、使用量の確認は重要です。
:::

```typescript title="chapter3/test3-1-chat-completions-api.ts"
import OpenAI from "openai";

// クライアントを定義
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Chat Completion APIの呼び出し例
async function main() {
  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "user", content: "こんにちは、今日はどんな天気ですか？" },
    ],
  });

  // 応答内容を出力
  console.log("Response:", response.choices[0]?.message.content, "\n");

  // 消費されたトークン数の表示
  const tokensUsed = response.usage;
  console.log("Prompt Tokens:", tokensUsed?.prompt_tokens);
  console.log("Completion Tokens:", tokensUsed?.completion_tokens);
  console.log("Total Tokens:", tokensUsed?.total_tokens);
  console.log(
    "Completion_tokens_details:",
    tokensUsed?.completion_tokens_details,
  );
  console.log("Prompt_tokens_details:", tokensUsed?.prompt_tokens_details);
}

main();
```

**実行方法:**

```bash
pnpm tsx chapter3/test3-1-chat-completions-api.ts
```

**実行結果の例:**

```text
Response: こんにちは！私はAIなので天気を直接確認する手段はありませんが、...

Prompt Tokens: 25
Completion Tokens: 80
Total Tokens: 105
Completion_tokens_details: { reasoning_tokens: 0, ... }
Prompt_tokens_details: { cached_tokens: 0, ... }
```

## 3-3. JSON Outputs

JSON モードを使うと、モデルの出力を有効な JSON 形式に制約できます。
`response_format: { type: "json_object" }` を指定することで、モデルは必ず JSON として解析可能な文字列を返します。

このサンプルでは以下のポイントを示しています。

- `response_format` に `json_object` を指定して JSON 出力を強制
- `system` メッセージで JSON 出力を指示
- `assistant` メッセージで出力スキーマのヒントを提供（期待する JSON の構造を例示することで、モデルが同じキー名・構造で応答するよう誘導できます）

:::tip
JSON モードを使用する際は、システムメッセージで「JSON を出力してください」と明示的に指示する必要があります。指示がない場合、モデルが無限にホワイトスペースを生成する可能性があります。
:::

```typescript title="chapter3/test3-3-json-outputs.ts"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function main() {
  const response = await client.chat.completions.create({
    model: "gpt-4o",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "あなたは JSON を出力するように設計された便利なアシスタントです。",
      },
      { role: "assistant", content: '{"winner": "String"}' },
      {
        role: "user",
        content: "2020 年のワールド シリーズの優勝者は誰ですか?",
      },
    ],
  });

  console.log(response.choices[0]?.message.content);
}

main();
```

**実行方法:**

```bash
pnpm tsx chapter3/test3-3-json-outputs.ts
```

**実行結果の例:**

```text
{"winner": "ロサンゼルス・ドジャース"}
```

:::info JSON モードの制約
JSON モードでは出力が有効な JSON であることは保証されますが、特定のスキーマに従うことは保証されません。たとえば、キー名が `"winner"` ではなく `"champion"` になる可能性があります。スキーマへの厳密な準拠が必要な場合は、次のセクション（3-4）の Structured Outputs を使用してください。
:::

## 3-4. Structured Outputs

Structured Outputs は、3-3 で紹介した JSON モードの進化版です。
JSON モードではスキーマの遵守が保証されませんでしたが、Structured Outputs では [Zod](https://zod.dev/)（TypeScript ファーストのスキーマバリデーションライブラリ）でスキーマを定義し、`zodResponseFormat` ヘルパーを使うことで、モデルの出力がスキーマに 100% 準拠することが保証されます。
型安全なデータ抽出や、構造化された情報の取得に最適です。

このサンプルでは以下のポイントを示しています。

- Zod でレシピのスキーマ（名前・人数・材料・手順）を定義
- `client.chat.completions.parse()` で型安全なパース結果を取得
- `response.choices[0].message.parsed` から直接型付きオブジェクトにアクセス

:::tip
JSON モード（3-3）との違いは、Structured Outputs ではスキーマに厳密に従った出力が保証される点です。JSON モードでは出力が JSON であることは保証されますが、スキーマの遵守は保証されません。
:::

### temperature パラメータとは？

`temperature` は、モデルの出力の **ランダム性（創造性）** を制御するパラメータで、`0` から `2` の範囲で指定します。

| 値 | 特徴 | ユースケース |
| --- | --- | --- |
| `0` | 最も決定的（同じ入力に対してほぼ同じ出力） | 構造化データ抽出、分類、事実に基づく回答 |
| `0.5〜0.7` | バランスの取れた出力 | 一般的な会話、要約 |
| `1.0`（デフォルト） | 適度なランダム性 | 創作、ブレインストーミング |
| `1.5〜2.0` | 非常にランダム（予測しにくい出力） | 実験的な用途 |

内部的には、モデルが次のトークン（単語の断片）を選ぶ際の確率分布を調整しています。`temperature` が低いほど確率の高いトークンが選ばれやすくなり、高いほど確率の低いトークンも選ばれる可能性が増します。

このサンプルでは `temperature: 0` を指定しているため、毎回ほぼ同じレシピが生成されます。レシピの構造化データを安定して取得したい場合に適した設定です。

```typescript title="chapter3/test3-4-structured-outputs.ts"
import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import { z } from "zod/v4";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Zodスキーマを定義
const Recipe = z.object({
  name: z.string(),
  servings: z.number().int(),
  ingredients: z.array(z.string()),
  steps: z.array(z.string()),
});

async function main() {
  // Structured Outputsに対応するZodスキーマを指定して呼び出し
  const response = await client.chat.completions.parse({
    model: "gpt-4o",
    messages: [{ role: "user", content: "タコライスのレシピを教えてください" }],
    temperature: 0,
    response_format: zodResponseFormat(Recipe, "Recipe"),
  });

  // 生成されたレシピ情報の表示
  const recipe = response.choices[0]?.message.parsed;

  console.log("Recipe Name:", recipe?.name);
  console.log("Servings:", recipe?.servings);
  console.log("Ingredients:", recipe?.ingredients);
  console.log("Steps:", recipe?.steps);
}

main();
```

**実行方法:**

```bash
pnpm tsx chapter3/test3-4-structured-outputs.ts
```

**実行結果の例:**

```text
Recipe Name: タコライス
Servings: 2
Ingredients: [ 'ひき肉 200g', 'レタス 2枚', 'トマト 1個', 'チーズ 適量', ... ]
Steps: [ '1. ひき肉をフライパンで炒める', '2. タコスシーズニングを加える', ... ]
```

## 3-6. Function Calling

Function Calling は、モデルが外部の関数（ツール）を呼び出せるようにする仕組みです。
[Chapter 2](../chapter2/index.md) で解説した「ツール」コンポーネントを API レベルで実現するための中核的な機能であり、AI エージェント開発において最も重要な概念の 1 つです。

モデル自体が関数を実行するわけではなく、「この関数をこの引数で呼ぶべき」という指示を JSON 形式で返します。アプリケーション側で実際の関数を実行し、その結果をモデルに返すことで、外部データを活用した応答を生成できます。

### tools パラメータの構造

`tools` パラメータには、モデルに提供する関数の定義を配列で渡します。各関数の定義は以下の構造を持ちます。

| フィールド | 説明 |
| --- | --- |
| `type` | ツールの種類。現在は `"function"` のみ |
| `function.name` | 関数名。モデルが呼び出す際の識別子として使用 |
| `function.description` | 関数の説明。モデルがどの関数を呼ぶか判断する際に参照される |
| `function.parameters` | JSON Schema 形式で定義する引数のスキーマ |

`description` はモデルの判断精度に大きく影響します。関数が「いつ・何のために使われるか」を具体的に記述することが重要です。

### 処理の流れ

このサンプルでは以下の流れを実装しています。

1. **ツールの定義** - `get_weather` 関数のスキーマを `tools` パラメータで定義
2. **初回リクエスト** - ユーザーメッセージを送信し、モデルが `tool_calls` を返す
3. **関数の実行** - モデルが指定した引数で `getWeather()` を実行
4. **結果の返却** - 関数の実行結果を `role: "tool"` メッセージとして返す
5. **最終応答** - モデルが関数の結果を踏まえた自然言語の応答を生成

:::tip
`tool_choice: "auto"` を指定すると、モデルが関数を呼ぶかどうかを自動で判断します。`"required"` にすると必ずいずれかの関数を呼び、`{"type": "function", "function": {"name": "get_weather"}}` のように指定すると特定の関数を強制的に呼ばせることができます。
:::

```typescript title="chapter3/test3-6-function-calling.ts"
import OpenAI from "openai";

// クライアントを定義
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// 天気情報を取得するダミー関数
function getWeather(location: string): string {
  const weatherInfo: Record<string, string> = {
    Tokyo: "晴れ、気温25度",
    Osaka: "曇り、気温22度",
    Kyoto: "雨、気温18度",
  };
  return weatherInfo[location] ?? "天気情報が見つかりません";
}

// モデルに提供するToolの定義
const tools: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "指定された場所の天気情報を取得します",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "都市名（例: Tokyo）",
          },
        },
        required: ["location"],
      },
    },
  },
];

async function main() {
  // 初回のユーザーメッセージ
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "user", content: "東京の天気を教えてください" },
  ];

  // モデルへの最初のAPIリクエスト
  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages,
    temperature: 0,
    tools,
    tool_choice: "auto",
  });

  // モデルの応答を処理
  const responseMessage = response.choices[0]!.message;
  messages.push(responseMessage);

  console.log("モデルからの応答:");
  console.log(responseMessage);

  // 関数呼び出しを処理
  if (responseMessage.tool_calls) {
    for (const toolCall of responseMessage.tool_calls) {
      if (
        toolCall.type === "function" &&
        toolCall.function.name === "get_weather"
      ) {
        const functionArgs = JSON.parse(toolCall.function.arguments);
        console.log("関数の引数:", functionArgs);
        const weatherResponse = getWeather(functionArgs.location);
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: weatherResponse,
        });
      }
    }
  } else {
    console.log("モデルによるツール呼び出しはありませんでした。");
  }

  // モデルへの最終的なAPIリクエスト
  const finalResponse = await client.chat.completions.create({
    model: "gpt-4o",
    messages,
    temperature: 0,
  });

  console.log("Final Response:", finalResponse.choices[0]?.message.content);
}

main();
```

**実行方法:**

```bash
pnpm tsx chapter3/test3-6-function-calling.ts
```

**実行結果の例:**

```text
モデルからの応答:
{
  role: 'assistant',
  content: null,
  tool_calls: [
    {
      id: 'call_xxx',
      type: 'function',
      function: { name: 'get_weather', arguments: '{"location":"Tokyo"}' }
    }
  ]
}
関数の引数: { location: 'Tokyo' }
Final Response: 東京の天気は晴れで、気温は25度です。
```

## 3-7. Tavily Search

[Tavily](https://tavily.com/) は、AI エージェント向けに最適化された Web 検索 API です。
通常の検索エンジンとは異なり、検索結果のスニペットだけでなく、ページの主要コンテンツを抽出して返すため、LLM が直接活用しやすい形式になっています。

### なぜ Tavily が必要なのか？

AI エージェントが Web 検索を行う場合、Google 検索のような一般的な検索エンジンでは以下の課題があります。

- 検索結果は URL とスニペット（短い抜粋）のみで、詳細な情報を得るには各ページをスクレイピングする必要がある
- 広告やナビゲーションなどのノイズが多く、LLM に渡す情報として最適化されていない
- API の利用料金が高く、レート制限も厳しい

Tavily はこれらの課題を解決するために設計されており、**1 回の API コールで検索結果とページコンテンツの両方を取得**できます。

### Tavily の主な特徴

| 特徴 | 説明 |
| --- | --- |
| **コンテンツ抽出** | 検索結果の各ページから本文を自動抽出し、広告やナビゲーションなどのノイズを除去 |
| **検索深度の選択** | `basic`（高速）と `advanced`（高精度）の 2 つの検索モードを提供 |
| **トピック指定** | `general`、`news` などのトピックを指定して検索対象を絞り込み可能 |
| **ドメインフィルタ** | `includeDomains` / `excludeDomains` で検索対象のドメインを制御 |
| **日付フィルタ** | `days` パラメータで指定日数以内の結果に限定可能 |

### search メソッドの主なオプション

```typescript
const response = await client.search(query, {
  searchDepth: "basic",    // "basic" | "advanced"（デフォルト: "basic"）
  topic: "general",        // "general" | "news"（デフォルト: "general"）
  maxResults: 5,           // 取得する検索結果の最大件数（デフォルト: 5）
  includeDomains: [],      // 検索対象に含めるドメインのリスト
  excludeDomains: [],      // 検索対象から除外するドメインのリスト
  days: 7,                 // 指定日数以内の結果に限定
});
```

### サンプルの内容

このサンプルでは以下を行います。

- Tavily クライアントの初期化（API キーの設定）
- 検索クエリの実行（`maxResults: 3` で上位 3 件を取得）
- 検索結果（タイトル・URL・コンテンツ）の表示

:::note
Tavily API を利用するには、[Tavily](https://tavily.com/) でアカウントを作成し、API キーを取得する必要があります。無料プランでは月 1,000 回の API コールが利用できます。
:::

```typescript title="chapter3/test3-7-tavily-search.ts"
import { tavily } from "@tavily/core";

// Tavily検索クライアントを初期化
const client = tavily({ apiKey: process.env.TAVILY_API_KEY ?? "" });

// 検索の実行例
const query = "AIエージェント 実践本";
const response = await client.search(query, { maxResults: 3 });
const results = response.results;

console.log(`検索クエリ: ${query}`);
console.log(`検索結果数: ${results.length}`);
console.log("\n検索結果:");
results.forEach((result, i) => {
  console.log(`\n${i + 1}. タイトル: ${result.title ?? "N/A"}`);
  console.log(`   URL: ${result.url ?? "N/A"}`);
  console.log(`   内容: ${(result.content ?? "N/A").slice(0, 100)}...`);
});
```

**実行方法:**

```bash
pnpm tsx chapter3/test3-7-tavily-search.ts
```

**実行結果の例:**

```text
検索クエリ: AIエージェント 実践本
検索結果数: 3

検索結果:

1. タイトル: AIエージェント入門 ― 実践ガイド
   URL: https://example.com/ai-agent-book
   内容: AIエージェントの基礎から実践までを解説した書籍...
```

## 3-8. LangChain カスタム Tool 定義

[LangChain](https://js.langchain.com/) の `tool` ヘルパーを使うと、AI エージェントが利用できるカスタム Tool を簡潔に定義できます。
3-6 の Function Calling では OpenAI API の JSON Schema 形式でスキーマ定義を直接記述しましたが、LangChain では Zod スキーマと関数本体をまとめて定義できます。これにより、型安全性を維持しながら宣言的に Tool を作成でき、定義の重複やミスを減らせます。

### LangChain の Tool とは？

LangChain の Tool は、AI エージェントが外部の機能を呼び出すための統一的なインターフェースです。`@langchain/core/tools` が提供する `tool` ヘルパー関数を使うことで、以下の要素を 1 つにまとめて定義できます。

| 要素 | 説明 |
| --- | --- |
| `name` | Tool の名前。エージェントが呼び出す際の識別子 |
| `description` | Tool の説明。エージェントがどの Tool を使うか判断する際に参照される |
| `schema` | Zod スキーマで定義する引数の型とバリデーション |
| 関数本体 | 実際に実行される処理（非同期関数） |

### サンプルで行うこと

このサンプルでは以下を行います。

- Zod で引数スキーマ（2 つの整数）を定義
- `tool` ヘルパーで加算 Tool を作成
- `invoke()` メソッドで Tool を実行
- Tool に関連付けられた属性（`name`、`description`、`schema`）の確認

```typescript title="chapter3/test3-8-custom-tool-definition.ts"
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// 引数スキーマを定義
const AddArgs = z.object({
  a: z.number().int().describe("加算する最初の整数。"),
  b: z.number().int().describe("加算する2つ目の整数。"),
});

// Tool定義
const add = tool(
  async ({ a, b }): Promise<string> => {
    return String(a + b);
  },
  {
    name: "add",
    description: [
      "このToolは2つの整数を引数として受け取り、それらの合計を返します。",
      "",
      "使用例:",
      "  例:",
      '    入力: {"a": 3, "b": 5}',
      "    出力: 8",
    ].join("\n"),
    schema: AddArgs,
  },
);

// 実行例
const args = { a: 5, b: 10 };
const result = await add.invoke(args); // Toolを呼び出す
console.log(`Result: ${result}`); // Result: 15

// Toolに関連付けられている属性の確認
console.log(add.name);
console.log(add.description);
console.log(add.schema);
```

**実行方法:**

```bash
pnpm tsx chapter3/test3-8-custom-tool-definition.ts
```

**実行結果の例:**

```text
Result: 15
add
このToolは2つの整数を引数として受け取り、それらの合計を返します。

使用例:
  例:
    入力: {"a": 3, "b": 5}
    出力: 8
ZodObject { ... }
```

## 3-9. DuckDuckGo Web 検索

[duck-duck-scrape](https://www.npmjs.com/package/duck-duck-scrape) は、DuckDuckGo の検索結果をプログラムから取得できるライブラリです。
3-7 の Tavily とは異なり、**API キー不要・無料**で利用できるため、手軽に Web 検索機能を組み込みたい場合に便利です。

### Tavily との比較

| 項目 | Tavily（3-7） | DuckDuckGo（3-9） |
| --- | --- | --- |
| API キー | 必要 | 不要 |
| 料金 | 無料枠あり（月 1,000 回） | 完全無料 |
| コンテンツ抽出 | 自動抽出（ノイズ除去済み） | なし（HTML を自前で取得・パースする必要あり） |
| LLM 向け最適化 | あり | なし |

### サンプルの処理ステップ

このサンプルでは以下の 2 ステップを実装しています。

1. **DuckDuckGo 検索** - `search()` 関数でキーワード検索を実行し、上位 3 件の結果（タイトル・概要・URL）を表示
2. **Web ページ取得** - 最初の検索結果の URL に対して `fetch` で HTTP リクエストを送り、HTML コンテンツのサイズと冒頭部分を表示

:::tip
DuckDuckGo 検索は API キーが不要なため、環境変数の設定なしですぐに試せます。ただし、Tavily のようなコンテンツ抽出機能はないため、取得した HTML から必要な情報を抽出するには別途パース処理（例: [cheerio](https://www.npmjs.com/package/cheerio) などの HTML パーサー）が必要です。
:::

:::caution レート制限について
`duck-duck-scrape` は DuckDuckGo の Web ページをスクレイピングして検索結果を取得しています。そのため、短時間に連続してリクエストを送ると **「DDG detected an anomaly in the request」** エラーが発生することがあります。このエラーが発生した場合は、しばらく時間を空けてから再実行してください。サンプルコードにはリトライ機能（最大 3 回、指数バックオフ）を組み込んでいますが、それでも失敗する場合があります。
:::

```typescript title="chapter3/test3-9-duckduckgo-search.ts"
import { SafeSearchType, search } from "duck-duck-scrape";

// リトライ付きで検索を実行する関数
async function searchWithRetry(
  query: string,
  maxRetries = 3,
  baseDelay = 2000,
) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await search(query, {
        safeSearch: SafeSearchType.OFF,
        locale: "ja-JP",
      });
    } catch (e) {
      if (attempt === maxRetries) throw e;
      const delay = baseDelay * attempt;
      console.log(
        `検索リクエストがブロックされました。${delay}ms 待機してリトライします... (${attempt}/${maxRetries})`,
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw new Error("検索に失敗しました");
}

// DuckDuckGo検索を実行（リトライ付き）
const searchQuery = "AIエージェント 実践本";
const searchResponse = await searchWithRetry(searchQuery);
const searchResults = searchResponse.results.slice(0, 3);

// 検索結果を表示
console.log("\n検索結果:");
searchResults.forEach((result, i) => {
  console.log(`\n${i + 1}. ${result.title}`);
  console.log(`   概要: ${(result.description ?? "").slice(0, 100)}...`);
  console.log(`   URL: ${result.url}`);
});

// 最初の検索結果のURLを取得
if (searchResults.length > 0) {
  const url = searchResults[0]?.url ?? "";
  console.log(`\n最初の検索結果のURLにアクセスしています: ${url}`);

  // Webページを取得
  try {
    const response = await fetch(url);
    const htmlContent = await response.text();
    console.log(`\nHTTPステータスコード: ${response.status}`);
    console.log(
      `\nHTMLコンテンツの大きさ: ${new Blob([htmlContent]).size} bytes`,
    );
    console.log(
      `\nHTMLコンテンツの最初の部分: \n${htmlContent.slice(0, 500)}...`,
    );
  } catch (e) {
    console.log(`\nエラーが発生しました: ${e}`);
  }
} else {
  console.log("\n検索結果はありませんでした");
}
```

**実行方法:**

```bash
pnpm tsx chapter3/test3-9-duckduckgo-search.ts
```

**実行結果の例:**

```text
検索結果:

1. AIエージェント入門 ― 実践ガイド
   概要: AIエージェントの基礎から実践までを解説...
   URL: https://example.com/ai-agent-book

最初の検索結果のURLにアクセスしています: https://example.com/ai-agent-book

HTTPステータスコード: 200

HTMLコンテンツの大きさ: 45678 bytes

HTMLコンテンツの最初の部分:
<!DOCTYPE html><html lang="ja">...
```

---

## 参考文献

- OpenAI. [Chat Completions API](https://platform.openai.com/docs/guides/text-generation) - Chat Completions API の公式ガイド
- OpenAI. [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) - JSON モードおよび Structured Outputs の公式ドキュメント
- OpenAI. [Function Calling](https://platform.openai.com/docs/guides/function-calling) - Function Calling の公式ドキュメント
- [Zod](https://zod.dev/) - TypeScript ファーストのスキーマバリデーションライブラリ
- [Tavily](https://docs.tavily.com/) - AI エージェント向け Web 検索 API の公式ドキュメント
- [LangChain Tools](https://js.langchain.com/docs/how_to/custom_tools/) - LangChain カスタム Tool の公式ドキュメント
- [duck-duck-scrape](https://www.npmjs.com/package/duck-duck-scrape) - DuckDuckGo 検索結果を取得する npm パッケージ
- [cheerio](https://www.npmjs.com/package/cheerio) - サーバーサイドで HTML をパース・操作するための軽量ライブラリ
