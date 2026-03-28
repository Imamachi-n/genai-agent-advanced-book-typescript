import { createRequire } from 'node:module';

import { setupLogger } from '../custom-logger.js';
import { MarkdownStorage } from './markdown-storage.js';

const require = createRequire(import.meta.url);
// pdf-parse には型定義がないため require で読み込む
const pdfParse = require('pdf-parse') as (
  buffer: Buffer,
) => Promise<{ text: string; numpages: number }>;

const logger = setupLogger('pdf-to-text');

export class PdfToText {
  private pdfUrl: string;
  private storage: MarkdownStorage;

  constructor(pdfUrl: string) {
    this.pdfUrl = pdfUrl;
    this.storage = new MarkdownStorage();
  }

  async convert(fileName?: string): Promise<string> {
    let _fileName = fileName ?? this.pdfUrl.split('/').pop() ?? 'unknown';
    _fileName = `${_fileName}.md`;

    // 既存のファイルがあれば、それを読み込んで返す（キャッシュ）
    try {
      return this.storage.read(_fileName);
    } catch {
      // 新規変換の場合
      logger.info(`Downloading PDF: ${this.pdfUrl}`);
      const response = await fetch(this.pdfUrl);
      if (!response.ok) {
        throw new Error(
          `PDF download failed: ${response.status} ${response.statusText} - ${this.pdfUrl}`,
        );
      }

      const arrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);

      const parsed = await pdfParse(buffer);
      const text = this.formatAsMarkdown(parsed.text);

      this.storage.write(_fileName, text);
      logger.info(`Converted and cached: ${_fileName}`);
      return text;
    }
  }

  private formatAsMarkdown(rawText: string): string {
    // テキストを段落に分割してMarkdown風に整形
    const lines = rawText.split('\n');
    const formatted: string[] = [];
    let currentParagraph: string[] = [];

    for (const line of lines) {
      const trimmed = line.trim();

      if (!trimmed) {
        if (currentParagraph.length > 0) {
          formatted.push(currentParagraph.join(' '));
          currentParagraph = [];
        }
        continue;
      }

      // ヘッダー候補の検出（全大文字の短い行、または数字で始まるセクション）
      if (
        (trimmed.length < 100 && trimmed === trimmed.toUpperCase() && /[A-Z]/.test(trimmed)) ||
        /^\d+\.?\s+[A-Z]/.test(trimmed)
      ) {
        if (currentParagraph.length > 0) {
          formatted.push(currentParagraph.join(' '));
          currentParagraph = [];
        }
        formatted.push(`\n## ${trimmed}\n`);
        continue;
      }

      currentParagraph.push(trimmed);
    }

    if (currentParagraph.length > 0) {
      formatted.push(currentParagraph.join(' '));
    }

    return formatted.join('\n\n');
  }
}
