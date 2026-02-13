import OpenAI from 'openai';

// クライアントを定義
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function main() {
  // 1. Responses API で最初のメッセージを送信
  const response1 = await client.responses.create({
    model: 'gpt-4o',
    instructions:
      'あなたは親切な料理アドバイザーです。ユーザーの質問に対して、簡潔で実用的なアドバイスを日本語で提供してください。',
    input: '初心者でも作れる簡単なパスタのレシピを教えてください',
    store: true, // 会話履歴をサーバー側に保存（マルチターンに必要）
  });

  console.log('Response ID:', response1.id);
  console.log('ステータス:', response1.status);
  console.log('\nアシスタントの応答:');
  console.log(response1.output_text);

  // --- マルチターン会話 ---
  console.log('\n--- マルチターン会話 ---\n');

  // 2. previous_response_id を指定して会話を継続
  const response2 = await client.responses.create({
    model: 'gpt-4o',
    instructions:
      'あなたは親切な料理アドバイザーです。ユーザーの質問に対して、簡潔で実用的なアドバイスを日本語で提供してください。',
    input: 'そのパスタに合うサラダも教えてください',
    previous_response_id: response1.id, // 前の会話を参照
    store: true,
  });

  console.log('Response ID:', response2.id);
  console.log('ステータス:', response2.status);
  console.log('\nアシスタントの応答:');
  console.log(response2.output_text);

  // 3. トークン使用量の確認
  console.log('\n--- トークン使用量 ---');
  console.log('1回目:', response1.usage);
  console.log('2回目:', response2.usage);
}

main();
