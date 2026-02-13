import type { Content } from '@google/genai';
import { GoogleGenAI, Type } from '@google/genai';

// クライアントを定義
const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY ?? '' });

/**
 * 指定された都市の天気情報を返すダミー関数
 * @param location - 都市名（例: Tokyo, Osaka, Kyoto）
 * @returns 天気情報の文字列
 */
function getWeather(location: string): string {
  const weatherInfo: Record<string, string> = {
    Tokyo: '晴れ、気温25度',
    Osaka: '曇り、気温22度',
    Kyoto: '雨、気温18度',
  };
  return weatherInfo[location] ?? '天気情報が見つかりません';
}

// モデルに提供するToolの定義
const getWeatherDeclaration = {
  name: 'get_weather',
  description: '指定された場所の天気情報を取得します',
  parameters: {
    type: Type.OBJECT,
    properties: {
      location: {
        type: Type.STRING,
        description: '都市名（例: Tokyo）',
      },
    },
    required: ['location'],
  },
};

const tools = [{ functionDeclarations: [getWeatherDeclaration] }];

/**
 * Gemini APIのFunction Callingを使って天気情報を取得する
 */
async function main() {
  // モデルへの最初のAPIリクエスト
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: '東京の天気を教えてください',
    config: {
      temperature: 0,
      tools,
    },
  });

  console.log('モデルからの応答:');

  // 関数呼び出しを処理
  const functionCalls = response.functionCalls;
  if (functionCalls && functionCalls.length > 0) {
    const functionCall = functionCalls[0];
    if (!functionCall) return;
    console.log('関数名:', functionCall.name);
    console.log('関数の引数:', functionCall.args);

    const weatherResponse = getWeather(functionCall.args?.location as string);

    // 関数の実行結果をモデルに返す
    const contents: Content[] = [
      { role: 'user', parts: [{ text: '東京の天気を教えてください' }] },
      {
        role: 'model',
        parts: [
          {
            functionCall: {
              name: functionCall.name ?? '',
              args: functionCall.args ?? {},
            },
          },
        ],
      },
      {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: functionCall.name ?? '',
              response: { result: weatherResponse },
            },
          },
        ],
      },
    ];
    const finalResponse = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents,
      config: {
        temperature: 0,
        tools,
      },
    });

    console.log('Final Response:', finalResponse.text);
  } else {
    console.log('モデルによるツール呼び出しはありませんでした。');
    console.log('Response:', response.text);
  }
}

main();
