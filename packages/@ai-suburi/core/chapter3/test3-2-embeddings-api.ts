import OpenAI from 'openai';

// クライアントを定義
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// コサイン類似度を計算する関数
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error('ベクトルの次元数が一致しません');
  }
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i]! * vecB[i]!;
    normA += vecA[i]! * vecA[i]!;
    normB += vecB[i]! * vecB[i]!;
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main() {
  // --- 1. 基本的な Embeddings API 呼び出し ---
  const response = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: 'AIエージェントは自律的にタスクを実行するシステムです',
  });

  const embedding = response.data[0]!.embedding;
  console.log('--- Embeddings API 基本呼び出し ---');
  console.log('モデル:', response.model);
  console.log('ベクトルの次元数:', embedding.length);
  console.log('ベクトルの先頭5要素:', embedding.slice(0, 5));
  console.log('トークン使用量:', response.usage);
  console.log();

  // --- 2. 複数テキストの埋め込みとコサイン類似度 ---
  const texts = [
    'AIエージェントは自律的にタスクを実行するプログラムです', // 類似テキスト
    '機械学習モデルを使って自動化されたワークフローを構築する', // やや類似
    '今日の東京の天気は晴れで、気温は25度です', // 無関係なテキスト
  ];

  const batchResponse = await client.embeddings.create({
    model: 'text-embedding-3-small',
    input: texts,
  });

  console.log('--- コサイン類似度の比較 ---');
  console.log(
    '基準テキスト: "AIエージェントは自律的にタスクを実行するシステムです"',
  );
  console.log();
  for (let i = 0; i < batchResponse.data.length; i++) {
    const similarity = cosineSimilarity(
      embedding,
      batchResponse.data[i]!.embedding,
    );
    console.log(`テキスト: "${texts[i]}"`);
    console.log(`類似度:   ${similarity.toFixed(4)}`);
    console.log();
  }
}

main();
