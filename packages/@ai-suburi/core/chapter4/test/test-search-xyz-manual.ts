import { searchXyzManual } from '../tools/search-xyz-manual/search-xyz-manual.js';

// テスト用のキーワードで検索を実行
const keyword = 'オンラインヘルプセンター';
console.log(`=== searchXyzManual テスト ===`);
console.log(`検索キーワード: ${keyword}\n`);

const results = await searchXyzManual.invoke({ keywords: keyword });

console.log(`\n=== 検索結果: ${results.length} 件 ===`);
for (const [i, result] of results.entries()) {
  console.log(`\n--- ${i + 1}. ${result.fileName} ---`);
  console.log(result.content.slice(0, 200));
}
