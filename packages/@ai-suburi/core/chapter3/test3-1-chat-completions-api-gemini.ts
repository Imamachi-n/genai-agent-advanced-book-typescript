import { GoogleGenAI } from '@google/genai';

// クライアントを定義
const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY ?? '' });

/**
 * Gemini APIを使ってテキスト生成を行い、トークン使用量を表示する
 */
async function main() {
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: 'こんにちは、今日はどんな天気ですか？',
  });

  // 応答内容を出力
  console.log('Response:', response.text, '\n');

  // 消費されたトークン数の表示
  const usage = response.usageMetadata;
  console.log('Prompt Tokens:', usage?.promptTokenCount);
  console.log('Completion Tokens:', usage?.candidatesTokenCount);
  console.log('Total Tokens:', usage?.totalTokenCount);
}

main();
