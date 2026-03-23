import type { ArxivPaper } from '../models.js';

export interface Searcher {
  run(goalSetting: string, query: string): Promise<ArxivPaper[]>;
}
