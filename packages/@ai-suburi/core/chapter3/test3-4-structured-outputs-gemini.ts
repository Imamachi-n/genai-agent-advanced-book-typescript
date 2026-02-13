import { GoogleGenAI, Type } from '@google/genai';

// クライアントを定義
const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY ?? '' });

/**
 * Gemini APIのStructured Outputsを使ってJSON形式でレシピ情報を取得する
 */
async function main() {
  // Structured Outputsに対応するスキーマを指定して呼び出し
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: 'タコライスのレシピを教えてください',
    config: {
      temperature: 0,
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING, description: 'レシピ名' },
          servings: { type: Type.INTEGER, description: '何人前か' },
          ingredients: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '材料リスト',
          },
          steps: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '手順リスト',
          },
        },
        required: ['name', 'servings', 'ingredients', 'steps'],
      },
    },
  });

  // 生成されたレシピ情報の表示
  const recipe = JSON.parse(response.text ?? '{}');

  console.log('Recipe Name:', recipe.name);
  console.log('Servings:', recipe.servings);
  console.log('Ingredients:', recipe.ingredients);
  console.log('Steps:', recipe.steps);
}

main();
