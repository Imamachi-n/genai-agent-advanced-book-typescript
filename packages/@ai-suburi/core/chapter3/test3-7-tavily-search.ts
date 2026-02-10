import { tavily } from "@tavily/core";

// Tavily検索クライアントを初期化
const client = tavily({ apiKey: process.env.TAVILY_API_KEY ?? "" });

// 検索の実行例
const query = "AIエージェント 実践本";
const response = await client.search(query, { maxResults: 3 });
const results = response.results;

console.log(`検索クエリ: ${query}`);
console.log(`検索結果数: ${results.length}`);
console.log("\n検索結果:");
results.forEach((result, i) => {
  console.log(`\n${i + 1}. タイトル: ${result.title ?? "N/A"}`);
  console.log(`   URL: ${result.url ?? "N/A"}`);
  console.log(`   内容: ${(result.content ?? "N/A").slice(0, 100)}...`);
});
