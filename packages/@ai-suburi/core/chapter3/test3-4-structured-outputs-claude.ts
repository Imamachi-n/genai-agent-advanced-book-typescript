import Anthropic from '@anthropic-ai/sdk';

// クライアントを定義
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

/**
 * Claude APIのStructured Outputsを使ってJSON形式でレシピ情報を取得する
 */
async function main() {
  // Structured Outputsに対応するJSON Schemaを指定して呼び出し
  const response = await client.messages.create({
    model: 'claude-sonnet-4-5-20250929',
    max_tokens: 1024,
    messages: [{ role: 'user', content: 'タコライスのレシピを教えてください' }],
    output_config: {
      format: {
        type: 'json_schema',
        schema: {
          type: 'object',
          properties: {
            name: { type: 'string', description: 'レシピ名' },
            servings: { type: 'integer', description: '何人前か' },
            ingredients: {
              type: 'array',
              items: { type: 'string' },
              description: '材料リスト',
            },
            steps: {
              type: 'array',
              items: { type: 'string' },
              description: '手順リスト',
            },
          },
          required: ['name', 'servings', 'ingredients', 'steps'],
        },
      },
    },
  });

  // 生成されたレシピ情報の表示
  const textBlock = response.content[0];
  const recipe = JSON.parse(
    textBlock?.type === 'text' ? textBlock.text : '{}',
  );

  console.log('Recipe Name:', recipe.name);
  console.log('Servings:', recipe.servings);
  console.log('Ingredients:', recipe.ingredients);
  console.log('Steps:', recipe.steps);
}

main();
