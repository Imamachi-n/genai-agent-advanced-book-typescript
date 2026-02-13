export interface Settings {
  openaiApiKey: string;
  openaiApiBase: string;
  openaiModel: string;
}

export function loadSettings(): Settings {
  const openaiApiKey = process.env.OPENAI_API_KEY;
  const openaiApiBase =
    process.env.OPENAI_API_BASE ?? 'https://api.openai.com/v1';
  const openaiModel = process.env.OPENAI_MODEL ?? 'gpt-4o';

  if (!openaiApiKey) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }

  return { openaiApiKey, openaiApiBase, openaiModel };
}
