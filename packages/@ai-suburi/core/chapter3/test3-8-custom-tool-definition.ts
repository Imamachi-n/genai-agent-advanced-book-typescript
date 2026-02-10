import { tool } from "@langchain/core/tools";
import { z } from "zod";

// 引数スキーマを定義
const AddArgs = z.object({
  a: z.number().int().describe("加算する最初の整数。"),
  b: z.number().int().describe("加算する2つ目の整数。"),
});

// Tool定義
const add = tool(
  async ({ a, b }): Promise<string> => {
    return String(a + b);
  },
  {
    name: "add",
    description: [
      "このToolは2つの整数を引数として受け取り、それらの合計を返します。",
      "",
      "使用例:",
      "  例:",
      '    入力: {"a": 3, "b": 5}',
      "    出力: 8",
    ].join("\n"),
    schema: AddArgs,
  },
);

// 実行例
const args = { a: 5, b: 10 };
const result = await add.invoke(args); // Toolを呼び出す
console.log(`Result: ${result}`); // Result: 15

// Toolに関連付けられている属性の確認
console.log(add.name);
console.log(add.description);
console.log(add.schema);
