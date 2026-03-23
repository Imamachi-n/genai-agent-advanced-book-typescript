import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { ChatOpenAI } from '@langchain/openai';
import { CohereClient } from 'cohere-ai';
import { XMLParser } from 'fast-xml-parser';

import { setupLogger } from '../custom-logger.js';
import {
  type ArxivFields,
  type ArxivPaper,
  type ArxivTimeRange,
  arxivFieldsSchema,
  arxivTimeRangeSchema,
  formatTimeRange,
} from '../models.js';
import type { Searcher } from './searcher.js';

const logger = setupLogger('arxiv-searcher');

const FIELD_SELECTOR_PROMPT = `\
Determine the arXiv categories that need to be searched based on the user's query.
Select one or more category names, separated by commas.
Reply only with the exact category names (e.g., cs.AI, math.CO).

User Query: {query}`;

const DATE_SELECTOR_PROMPT = `\
Determine the time range to be retrieved based on the user's query and the current system time.
Use the format YYMM-YYMM (e.g., 2203-2402 for March 2022 to February 2024).
If no time range is specified, reply with "NONE".

Current Date: {current_date}
User Query: {query}`;

const EXPAND_QUERY_PROMPT = `\
<system>
あなたは、与えられた単一のサブクエリから効果的なarXiv検索クエリを生成する専門家です。あなたの役割は、学術的な文脈を理解し、arXivの検索システムで直接使用できる最適な検索クエリを作成することです。

{feedback}
</system>

## 主要タスク

1. 提供されたサブクエリを分析する
2. サブクエリから重要なキーワードを抽出する
3. 抽出したキーワードを使用して、arXivで直接使用可能な効果的な検索クエリを構築する

## 詳細な指示

<instructions>
1. サブクエリを注意深く読み、主要な概念や専門用語を特定してください。
2. 学術的文脈に適した具体的なキーワードを選択してください。
3. 同義語や関連する用語も考慮に入れてください。
4. arXivの検索構文を適切に使用して、効果的な検索クエリを作成してください。
5. 検索結果が適切に絞り込まれるよう、必要に応じてフィールド指定子を使用してください。
6. 生成したクエリがarXivの検索ボックスに直接コピー＆ペーストできることを確認してください。
</instructions>

## 重要なルール

<rules>
1. クエリには1〜2つの主要なキーワードまたはフレーズを含めてください。
2. 一般的すぎる用語や非学術的な用語は避けてください。
3. 検索クエリは20文字以内に収めてください。
4. クエリの前後に余分な空白や引用符を付けないでください。
5. 説明や理由付けは含めず、純粋な検索クエリのみを出力してください。
6. 最大キーワード数は2つまでにすること。
7. OR検索はしないこと。
</rules>

## arXiv検索の構文ヒント

<arxiv_syntax>
- AND: 複数の用語を含む文書を検索（例：quantum AND computing）
- OR: いずれかの用語を含む文書を検索（例：neural OR quantum）
- 引用符: フレーズ検索（例："quantum computing"）
- フィールド指定子: ti:（タイトル）, au:（著者）, abs:（要約）
- マイナス記号: 特定の用語を除外（例：quantum -classical）
- ワイルドカード: 部分一致検索（例：neuro*）
</arxiv_syntax>

<keywords>
- 研究的なキーワードの例: RL, Optimization, LLM, etc.
- サーベイ論文について検索する場合は次のキーワードを利用する: Survey, Review
- データセットについて検索する場合は次のキーワードを利用する: Benchmark
- 論文名が分かっている場合は論文名で検索する
</keywords>

## 例

<example>
クエリ: 量子コンピューティングにおける最近の進歩に関する情報を取得する。

arXiv検索クエリ:
ti:"quantum computing"
</example>

<example>
クエリ: 深層強化学習の金融市場への応用に関する最新の研究を見つける。

arXiv検索クエリ:
"deep reinforcement learning" AND "financial markets"
</example>

## 入力フォーマット

<input_format>
目標: {goal_setting}
クエリ: {query}
</input_format>

REMEMBER: rulesタグの内容に必ず従うこと`;

interface ArxivEntry {
  id: string;
  title: string;
  link: string | { '@_href': string; '@_type'?: string }[] | { '@_href': string };
  summary: string;
  published: string;
  updated: string;
  author: { name: string }[] | { name: string };
  category: { '@_term': string }[] | { '@_term': string };
}

export class ArxivSearcher implements Searcher {
  static readonly RELEVANCE_SCORE_THRESHOLD = 0.7;

  private llm: ChatOpenAI;
  private cohereClient: CohereClient;
  private currentDate: string;
  private maxSearchResults: number;
  private maxPapers: number;
  private maxRetries: number;
  private debug: boolean;
  private cohereRerankModel: string;

  constructor(
    llm: ChatOpenAI,
    options: {
      cohereApiKey: string;
      cohereRerankModel?: string;
      maxSearchResults?: number;
      maxPapers?: number;
      maxRetries?: number;
      debug?: boolean;
    },
  ) {
    this.llm = llm;
    this.cohereClient = new CohereClient({ token: options.cohereApiKey });
    this.currentDate = new Date().toISOString().split('T')[0]!;
    this.maxSearchResults = options.maxSearchResults ?? 10;
    this.maxPapers = options.maxPapers ?? 3;
    this.maxRetries = options.maxRetries ?? 3;
    this.debug = options.debug ?? true;
    this.cohereRerankModel =
      options.cohereRerankModel ?? 'rerank-multilingual-v3.0';
  }

  private async fieldSelector(query: string): Promise<ArxivFields> {
    const prompt = ChatPromptTemplate.fromTemplate(FIELD_SELECTOR_PROMPT);
    const chain = prompt.pipe(
      this.llm.withStructuredOutput(arxivFieldsSchema),
    );
    return chain.invoke({ query });
  }

  private async dateSelector(query: string): Promise<ArxivTimeRange> {
    const prompt = ChatPromptTemplate.fromTemplate(DATE_SELECTOR_PROMPT);
    const chain = prompt.pipe(
      this.llm.withStructuredOutput(arxivTimeRangeSchema),
    );
    return chain.invoke({
      current_date: this.currentDate,
      query,
    });
  }

  private async expandQuery(
    goalSetting: string,
    query: string,
    feedback: string = '',
  ): Promise<string> {
    const prompt = ChatPromptTemplate.fromTemplate(EXPAND_QUERY_PROMPT);
    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());
    return chain.invoke({
      goal_setting: goalSetting,
      query,
      feedback,
    });
  }

  private parseArxivResponse(xmlText: string): ArxivPaper[] {
    const parser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '@_',
      isArray: (name) => name === 'entry' || name === 'author' || name === 'category' || name === 'link',
    });
    const parsed = parser.parse(xmlText);
    const entries: ArxivEntry[] = parsed?.feed?.entry ?? [];

    return entries.map((entry) => {
      const id = typeof entry.id === 'string' ? entry.id : String(entry.id);
      const arxivId = id.split('/').pop()?.split('v')[0] ?? '';
      const version = Number.parseInt(
        id.split('/').pop()?.split('v').pop() ?? '1',
        10,
      );

      // PDF リンクの抽出
      let pdfLink = '';
      if (Array.isArray(entry.link)) {
        const pdfEntry = entry.link.find(
          (l) =>
            typeof l === 'object' &&
            '@_type' in l &&
            l['@_type'] === 'application/pdf',
        );
        pdfLink =
          pdfEntry && typeof pdfEntry === 'object' && '@_href' in pdfEntry
            ? pdfEntry['@_href']
            : '';
      }

      // リンクの抽出
      let link = '';
      if (Array.isArray(entry.link)) {
        const htmlEntry = entry.link.find(
          (l) =>
            typeof l === 'object' &&
            (!('@_type' in l) || l['@_type'] !== 'application/pdf'),
        );
        link =
          htmlEntry && typeof htmlEntry === 'object' && '@_href' in htmlEntry
            ? htmlEntry['@_href']
            : '';
      } else if (typeof entry.link === 'string') {
        link = entry.link;
      }

      // 著者の抽出
      const authors = Array.isArray(entry.author)
        ? entry.author.map((a) => a.name ?? '')
        : [entry.author?.name ?? ''];

      // カテゴリの抽出
      const categories = Array.isArray(entry.category)
        ? entry.category.map((c) => c['@_term'] ?? '')
        : [entry.category?.['@_term'] ?? ''];

      return {
        id: arxivId,
        title: typeof entry.title === 'string' ? entry.title.trim() : '',
        link,
        pdfLink,
        abstract:
          typeof entry.summary === 'string'
            ? entry.summary.replace(/\n/g, ' ').trim()
            : '',
        published: entry.published ?? '',
        updated: entry.updated ?? '',
        version,
        authors,
        categories,
        relevanceScore: null,
      };
    });
  }

  async run(goalSetting: string, query: string): Promise<ArxivPaper[]> {
    const baseUrl = 'https://export.arxiv.org/api/query?search_query=';
    let retryCount = 0;
    let feedback = '';
    let papers: ArxivPaper[] = [];

    while (retryCount < this.maxRetries) {
      const arxivTimeRange = await this.dateSelector(query);
      const queryFilterDate = formatTimeRange(arxivTimeRange);

      const expandedQuery = await this.expandQuery(
        goalSetting,
        query,
        feedback,
      );

      const searchQuery = `all:${expandedQuery}`;
      const encodedSearchQuery = encodeURIComponent(searchQuery);

      let fullUrl = `${baseUrl}${encodedSearchQuery}&sortBy=relevance&max_results=${this.maxSearchResults}`;
      if (queryFilterDate) {
        fullUrl += `&submittedDate=${queryFilterDate}`;
      }
      logger.info(`Searching for papers: ${fullUrl}`);

      const response = await fetch(fullUrl);
      const xmlText = await response.text();
      papers = this.parseArxivResponse(xmlText);

      if (this.debug) {
        logger.info(`Found ${papers.length} papers.`);
      }

      if (papers.length > 0) {
        logger.info('Papers found. Exiting retry loop.');
        break;
      }

      retryCount++;
      if (retryCount < this.maxRetries) {
        feedback =
          '検索結果が0件でした。クエリをより一般的なものや関連するキーワードに調整してください。';
        logger.info(
          `No papers found. Retrying with adjusted query. Attempt ${retryCount}/${this.maxRetries}`,
        );
      } else {
        logger.info('Max retries reached. No results found.');
        break;
      }
    }

    if (papers.length > 0) {
      const reranked = await this.cohereClient.v2.rerank({
        model: this.cohereRerankModel,
        query: `${goalSetting}\n${query}`,
        documents: papers.map(
          (paper) => `${paper.title}\n${paper.abstract}`,
        ),
        topN: Math.min(this.maxPapers, papers.length),
      });

      const rerankedPapers: ArxivPaper[] = [];
      for (const result of reranked.results) {
        const paper = papers[result.index]!;
        paper.relevanceScore = result.relevanceScore;
        rerankedPapers.push(paper);
      }

      papers = rerankedPapers.filter(
        (paper) =>
          paper.relevanceScore != null &&
          paper.relevanceScore >= ArxivSearcher.RELEVANCE_SCORE_THRESHOLD,
      );
    }

    return papers;
  }
}
