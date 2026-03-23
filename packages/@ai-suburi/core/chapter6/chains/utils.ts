import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function loadPrompt(name: string): string {
  const promptPath = path.join(__dirname, 'prompts', `${name}.prompt`);
  return fs.readFileSync(promptPath, 'utf-8').trim();
}

export function dictToXmlStr(
  data: Record<string, unknown>,
  excludeKeys: string[] = [],
): string {
  let xmlStr = '<item>';
  for (const [key, value] of Object.entries(data)) {
    if (!excludeKeys.includes(key)) {
      xmlStr += `<${key}>${value}</${key}>`;
    }
  }
  xmlStr += '</item>';
  return xmlStr;
}
