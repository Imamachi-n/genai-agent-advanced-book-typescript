import Anthropic from '@anthropic-ai/sdk';

// クライアントを定義
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

/**
 * Claude Messages APIを使ってテキスト生成を行い、トークン使用量を表示する
 */
async function main() {
  const response = await client.messages.create({
    model: 'claude-sonnet-4-5-20250929',
    max_tokens: 1024,
    messages: [
      { role: 'user', content: 'こんにちは、今日はどんな天気ですか？' },
    ],
  });

  // 応答内容を出力
  const textBlock = response.content[0];
  console.log(
    'Response:',
    textBlock?.type === 'text' ? textBlock.text : '',
    '\n',
  );

  // 消費されたトークン数の表示
  console.log('Input Tokens:', response.usage.input_tokens);
  console.log('Output Tokens:', response.usage.output_tokens);
}

main();
