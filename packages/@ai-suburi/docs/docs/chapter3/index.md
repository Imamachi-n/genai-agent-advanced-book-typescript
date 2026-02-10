---
sidebar_position: 1
---

# Chapter 3: AIエージェントの開発準備

第3章のサンプルコード解説です。

## 概要

この章では、OpenAI API を使用した基本的な実装について解説します。
以下のトピックを扱います。

| セクション | 内容 |
| --- | --- |
| 3-1 | Chat Completions API の基本的な使い方 |
| 3-3 | JSON モードによる構造化された出力 |
| 3-4 | Zod スキーマを用いた Structured Outputs |
| 3-6 | Function Calling による外部ツール連携 |
| 3-7 | Tavily API を使った Web 検索 |

:::info 前提条件

- 環境変数 `OPENAI_API_KEY` に OpenAI の API キーが設定されていること
- 3-7 のみ、環境変数 `TAVILY_API_KEY` に Tavily の API キーが設定されていること

:::

## 3-1. Chat Completions API

Chat Completions API は、OpenAI のチャットモデルと対話するための最も基本的な API です。
`messages` 配列にロール（`system`, `user`, `assistant`）とメッセージを渡すことで、モデルからの応答を取得できます。

このサンプルでは以下を行います。

- `gpt-4o` モデルへのメッセージ送信
- 応答テキストの取得
- トークン使用量（プロンプト / 生成 / 合計）の確認

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

## 3-3. JSON Outputs

JSON モードを使うと、モデルの出力を有効な JSON 形式に制約できます。
`response_format: { type: "json_object" }` を指定することで、モデルは必ず JSON として解析可能な文字列を返します。

このサンプルでは以下のポイントを示しています。

- `response_format` に `json_object` を指定して JSON 出力を強制
- `system` メッセージで JSON 出力を指示
- `assistant` メッセージで出力スキーマのヒントを提供

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

## 3-4. Structured Outputs

Structured Outputs は、JSON モードの進化版です。
[Zod](https://zod.dev/) スキーマを定義し、`zodResponseFormat` ヘルパーを使うことで、モデルの出力がスキーマに 100% 準拠することが保証されます。

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

## 3-6. Function Calling

Function Calling は、モデルが外部の関数（ツール）を呼び出せるようにする仕組みです。
モデル自体が関数を実行するわけではなく、「この関数をこの引数で呼ぶべき」という指示を返します。アプリケーション側で実際の関数を実行し、その結果をモデルに返すことで、外部データを活用した応答を生成できます。

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
