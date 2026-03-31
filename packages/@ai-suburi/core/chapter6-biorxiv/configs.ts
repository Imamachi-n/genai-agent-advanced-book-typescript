import { ChatOpenAI } from '@langchain/openai';

export interface Settings {
  openaiApiKey: string;
  debug: boolean;
  // モデル設定
  openaiSmartModel: string;
  openaiFastModel: string;
  openaiReporterModel: string;
  embeddingModel: string;
  temperature: number;
  // エージェント設定
  maxEvaluationRetryCount: number;
  minDecomposedTasks: number;
  maxDecomposedTasks: number;
  maxSearchResults: number;
  maxPapers: number;
  maxWorkers: number;
  maxRecursionLimit: number;
  // 分析モード
  analysisMode: 'simple' | 'detailed';
  // RAG設定
  qdrantCollectionName: string;
  qdrantUrl: string;
  biorxivCategory: string;
  ingestionBatchSize: number;
}

export function loadSettings(): Settings {
  const openaiApiKey = process.env.OPENAI_API_KEY;

  if (!openaiApiKey) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }

  return {
    openaiApiKey,
    debug: process.env.DEBUG === 'true',
    // モデル設定
    openaiSmartModel: process.env.OPENAI_SMART_MODEL ?? 'gpt-4o',
    openaiFastModel: process.env.OPENAI_FAST_MODEL ?? 'gpt-4o-mini',
    openaiReporterModel: process.env.OPENAI_REPORTER_MODEL ?? 'gpt-4o',
    embeddingModel: process.env.EMBEDDING_MODEL ?? 'text-embedding-3-small',
    temperature: Number(process.env.TEMPERATURE ?? '0'),
    // エージェント設定
    maxEvaluationRetryCount: Number(
      process.env.MAX_EVALUATION_RETRY_COUNT ?? '3',
    ),
    minDecomposedTasks: Number(process.env.MIN_DECOMPOSED_TASKS ?? '3'),
    maxDecomposedTasks: Number(process.env.MAX_DECOMPOSED_TASKS ?? '5'),
    maxSearchResults: Number(process.env.MAX_SEARCH_RESULTS ?? '20'),
    maxPapers: Number(process.env.MAX_PAPERS ?? '3'),
    maxWorkers: Number(process.env.MAX_WORKERS ?? '3'),
    maxRecursionLimit: Number(process.env.MAX_RECURSION_LIMIT ?? '1000'),
    // 分析モード
    analysisMode:
      (process.env.ANALYSIS_MODE as 'simple' | 'detailed') ?? 'detailed',
    // RAG設定
    qdrantCollectionName:
      process.env.QDRANT_COLLECTION_NAME ?? 'biorxiv-bioinformatics',
    qdrantUrl:
      process.env.QDRANT_URL ?? 'http://localhost:6333',
    biorxivCategory: process.env.BIORXIV_CATEGORY ?? 'bioinformatics',
    ingestionBatchSize: Number(process.env.INGESTION_BATCH_SIZE ?? '100'),
  };
}

export function createLlm(settings: Settings): ChatOpenAI {
  return new ChatOpenAI({
    model: settings.openaiSmartModel,
    temperature: settings.temperature,
  });
}

export function createFastLlm(settings: Settings): ChatOpenAI {
  return new ChatOpenAI({
    model: settings.openaiFastModel,
    temperature: settings.temperature,
  });
}

export function createReporterLlm(settings: Settings): ChatOpenAI {
  return new ChatOpenAI({
    model: settings.openaiReporterModel,
    temperature: settings.temperature,
    maxTokens: 8192,
  });
}
