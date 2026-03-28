import type { BiorxivPaper } from '../models.js';

export interface Searcher {
  run(goalSetting: string, query: string): Promise<BiorxivPaper[]>;
}
