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
