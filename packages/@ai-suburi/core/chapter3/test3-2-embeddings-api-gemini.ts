import { GoogleGenAI } from '@google/genai';

// クライアントを定義
const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY ?? '' });

/**
 * 2つのベクトル間のコサイン類似度を計算する
 * @param vecA - 比較元のベクトル
 * @param vecB - 比較先のベクトル
 * @returns コサイン類似度（-1〜1の値）
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error('ベクトルの次元数が一致しません');
  }
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += (vecA[i] ?? 0) * (vecB[i] ?? 0);
    normA += (vecA[i] ?? 0) * (vecA[i] ?? 0);
    normB += (vecB[i] ?? 0) * (vecB[i] ?? 0);
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Gemini Embeddings APIでテキストをベクトル化し、コサイン類似度を比較する
 */
async function main() {
  // --- 1. 基本的な Embeddings API 呼び出し ---
  const response = await ai.models.embedContent({
    model: 'gemini-embedding-001',
    contents: 'AIエージェントは自律的にタスクを実行するシステムです',
  });

  const embedding = response.embeddings?.[0]?.values ?? [];
  console.log('--- Embeddings API 基本呼び出し ---');
  console.log('ベクトルの次元数:', embedding.length);
  console.log('ベクトルの先頭5要素:', embedding.slice(0, 5));
  console.log();

  // --- 2. 複数テキストの埋め込みとコサイン類似度 ---
  const texts = [
    'AIエージェントは自律的にタスクを実行するプログラムです', // 類似テキスト
    '機械学習モデルを使って自動化されたワークフローを構築する', // やや類似
    '今日の東京の天気は晴れで、気温は25度です', // 無関係なテキスト
  ];

  console.log('--- コサイン類似度の比較 ---');
  console.log(
    '基準テキスト: "AIエージェントは自律的にタスクを実行するシステムです"',
  );
  console.log();

  for (let i = 0; i < texts.length; i++) {
    const batchResponse = await ai.models.embedContent({
      model: 'gemini-embedding-001',
      contents: texts[i] ?? '',
    });
    const similarity = cosineSimilarity(
      embedding,
      batchResponse.embeddings?.[0]?.values ?? [],
    );
    console.log(`テキスト: "${texts[i]}"`);
    console.log(`類似度:   ${similarity.toFixed(4)}`);
    console.log();
  }
}

main();
