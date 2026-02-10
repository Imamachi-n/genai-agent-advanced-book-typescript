import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function main() {
  const response = await client.chat.completions.create({
    model: "gpt-4o",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "あなたは JSON を出力するように設計された便利なアシスタントです。",
      },
      { role: "assistant", content: '{"winner": "String"}' },
      {
        role: "user",
        content: "2020 年のワールド シリーズの優勝者は誰ですか?",
      },
    ],
  });

  console.log(response.choices[0]?.message.content);
}

main();
