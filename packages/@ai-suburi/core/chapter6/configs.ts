import { ChatOpenAI } from '@langchain/openai';

export interface Settings {
  openaiApiKey: string;
  cohereApiKey: string;
  jinaApiKey: string;
  debug: boolean;
  // モデル設定
  openaiSmartModel: string;
  openaiFastModel: string;
  openaiReporterModel: string;
  cohereRerankModel: string;
  temperature: number;
  // エージェント設定
  maxEvaluationRetryCount: number;
  minDecomposedTasks: number;
  maxDecomposedTasks: number;
  maxSearchRetries: number;
  maxSearchResults: number;
  maxPapers: number;
  maxWorkers: number;
  maxRecursionLimit: number;
}

export function loadSettings(): Settings {
  const openaiApiKey = process.env.OPENAI_API_KEY;
  const cohereApiKey = process.env.COHERE_API_KEY;
  const jinaApiKey = process.env.JINA_API_KEY;

  if (!openaiApiKey) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }
  if (!cohereApiKey) {
    throw new Error('COHERE_API_KEY environment variable is required');
  }
  if (!jinaApiKey) {
    throw new Error('JINA_API_KEY environment variable is required');
  }

  return {
    openaiApiKey,
    cohereApiKey,
    jinaApiKey,
    debug: process.env.DEBUG === 'true',
    // モデル設定
    openaiSmartModel: process.env.OPENAI_SMART_MODEL ?? 'gpt-4o',
    openaiFastModel: process.env.OPENAI_FAST_MODEL ?? 'gpt-4o-mini',
    openaiReporterModel: process.env.OPENAI_REPORTER_MODEL ?? 'gpt-4o',
    cohereRerankModel:
      process.env.COHERE_RERANK_MODEL ?? 'rerank-multilingual-v3.0',
    temperature: Number(process.env.TEMPERATURE ?? '0'),
    // エージェント設定
    maxEvaluationRetryCount: Number(
      process.env.MAX_EVALUATION_RETRY_COUNT ?? '3',
    ),
    minDecomposedTasks: Number(process.env.MIN_DECOMPOSED_TASKS ?? '3'),
    maxDecomposedTasks: Number(process.env.MAX_DECOMPOSED_TASKS ?? '5'),
    maxSearchRetries: Number(process.env.MAX_SEARCH_RETRIES ?? '3'),
    maxSearchResults: Number(process.env.MAX_SEARCH_RESULTS ?? '10'),
    maxPapers: Number(process.env.MAX_PAPERS ?? '3'),
    maxWorkers: Number(process.env.MAX_WORKERS ?? '3'),
    maxRecursionLimit: Number(process.env.MAX_RECURSION_LIMIT ?? '1000'),
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
