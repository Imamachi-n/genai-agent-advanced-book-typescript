import Anthropic from '@anthropic-ai/sdk';

// クライアントを定義
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

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
const tools: Anthropic.Messages.Tool[] = [
  {
    name: 'get_weather',
    description: '指定された場所の天気情報を取得します',
    input_schema: {
      type: 'object' as const,
      properties: {
        location: {
          type: 'string',
          description: '都市名（例: Tokyo）',
        },
      },
      required: ['location'],
    },
  },
];

/**
 * Claude APIのFunction Callingを使って天気情報を取得する
 */
async function main() {
  // 初回のユーザーメッセージ
  const messages: Anthropic.Messages.MessageParam[] = [
    { role: 'user', content: '東京の天気を教えてください' },
  ];

  // モデルへの最初のAPIリクエスト
  const response = await client.messages.create({
    model: 'claude-sonnet-4-5-20250929',
    max_tokens: 1024,
    messages,
    tools,
  });

  console.log('モデルからの応答:');
  console.log('stop_reason:', response.stop_reason);

  // 関数呼び出しを処理
  if (response.stop_reason === 'tool_use') {
    const toolUseBlock = response.content.find(
      (block) => block.type === 'tool_use',
    );

    if (toolUseBlock && toolUseBlock.type === 'tool_use') {
      console.log('関数名:', toolUseBlock.name);
      console.log('関数の引数:', toolUseBlock.input);

      const weatherResponse = getWeather(
        (toolUseBlock.input as { location: string }).location,
      );

      // 関数の実行結果をモデルに返す
      const finalResponse = await client.messages.create({
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 1024,
        messages: [
          ...messages,
          { role: 'assistant', content: response.content },
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: toolUseBlock.id,
                content: weatherResponse,
              },
            ],
          },
        ],
        tools,
      });

      const textBlock = finalResponse.content.find(
        (block) => block.type === 'text',
      );
      console.log(
        'Final Response:',
        textBlock?.type === 'text' ? textBlock.text : '',
      );
    }
  } else {
    console.log('モデルによるツール呼び出しはありませんでした。');
  }
}

main();
