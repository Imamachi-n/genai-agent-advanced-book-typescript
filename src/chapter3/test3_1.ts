import OpenAI from "openai";

// クライアントを定義
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Chat Completion APIの呼び出し例
async function main() {
  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "user", content: "こんにちは、今日はどんな天気ですか？" },
    ],
  });

  // 応答内容を出力
  console.log("Response:", response.choices[0]?.message.content, "\n");

  // 消費されたトークン数の表示
  const tokensUsed = response.usage;
  console.log("Prompt Tokens:", tokensUsed?.prompt_tokens);
  console.log("Completion Tokens:", tokensUsed?.completion_tokens);
  console.log("Total Tokens:", tokensUsed?.total_tokens);
  console.log(
    "Completion_tokens_details:",
    tokensUsed?.completion_tokens_details,
  );
  console.log("Prompt_tokens_details:", tokensUsed?.prompt_tokens_details);
}

main();
