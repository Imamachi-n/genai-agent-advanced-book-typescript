import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import { z } from "zod/v4";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Zodスキーマを定義
const Recipe = z.object({
  name: z.string(),
  servings: z.number().int(),
  ingredients: z.array(z.string()),
  steps: z.array(z.string()),
});

async function main() {
  // Structured Outputsに対応するZodスキーマを指定して呼び出し
  const response = await client.chat.completions.parse({
    model: "gpt-4o",
    messages: [{ role: "user", content: "タコライスのレシピを教えてください" }],
    temperature: 0,
    response_format: zodResponseFormat(Recipe, "Recipe"),
  });

  // 生成されたレシピ情報の表示
  const recipe = response.choices[0]?.message.parsed;

  console.log("Recipe Name:", recipe?.name);
  console.log("Servings:", recipe?.servings);
  console.log("Ingredients:", recipe?.ingredients);
  console.log("Steps:", recipe?.steps);
}

main();
