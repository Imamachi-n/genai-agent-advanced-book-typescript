import { searchXyzQa } from '../tools/search-xyz-qa/search-xyz-qa.js';

// テスト用のクエリでベクトル検索を実行
const query = 'パスワードを間違えてロックされました';
console.log(`=== searchXyzQa テスト ===`);
console.log(`検索クエリ: ${query}\n`);

const results = await searchXyzQa.invoke({ query });

console.log(`\n=== 検索結果: ${results.length} 件 ===`);
for (const [i, result] of results.entries()) {
  console.log(`\n--- ${i + 1}. ${result.fileName} ---`);
  console.log(result.content);
}
