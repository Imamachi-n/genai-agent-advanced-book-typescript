import { loadSettings } from '../configs.js';
import { MarkdownStorage } from './markdown-storage.js';

class JinaApiClient {
  private headers: Record<string, string>;
  private baseUrl: string;

  constructor(apiKey: string) {
    this.headers = { Authorization: `Bearer ${apiKey}` };
    this.baseUrl = 'https://r.jina.ai';
  }

  async convertPdfToMarkdown(pdfUrl: string): Promise<string> {
    const jinaUrl = `${this.baseUrl}/${pdfUrl}`;

    const response = await fetch(jinaUrl, { headers: this.headers });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(
        `JINA API error: ${response.status} - ${text}`,
      );
    }

    return response.text();
  }
}

export class PdfToMarkdown {
  private pdfPathOrUrl: string;
  private jinaClient: JinaApiClient;
  private storage: MarkdownStorage;

  constructor(pdfPathOrUrl: string, jinaApiKey?: string) {
    this.pdfPathOrUrl = pdfPathOrUrl;
    const apiKey = jinaApiKey ?? loadSettings().jinaApiKey;
    this.jinaClient = new JinaApiClient(apiKey);
    this.storage = new MarkdownStorage();
  }

  async convert(fileName?: string): Promise<string> {
    let _fileName = fileName ?? this.pdfPathOrUrl.split('/').pop() ?? 'unknown';
    _fileName = `${_fileName}.md`;

    // 既存のmarkdownファイルがあれば、それを読み込んで返す
    try {
      return this.storage.read(_fileName);
    } catch {
      // 新規変換の場合、JINA Reader APIを使用
      const markdown = await this.jinaClient.convertPdfToMarkdown(
        this.pdfPathOrUrl,
      );
      this.storage.write(_fileName, markdown);
      return markdown;
    }
  }
}
