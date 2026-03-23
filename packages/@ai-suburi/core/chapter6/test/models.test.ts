import { z } from 'zod/v4';
import {
  arxivFieldsSchema,
  arxivPaperSchema,
  arxivPaperToXml,
  arxivTimeRangeSchema,
  decomposedTasksSchema,
  formatTimeRange,
  hearingSchema,
  readingResultSchema,
  sectionSchema,
  sufficiencySchema,
  taskEvaluationSchema,
} from '../models.js';

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`FAIL: ${message}`);
  }
}

// --- ArxivPaper ---
console.log('Testing ArxivPaper schema...');

const validPaper = {
  id: '2408.14317',
  title: 'Test Paper',
  link: 'https://arxiv.org/abs/2408.14317',
  pdfLink: 'https://arxiv.org/pdf/2408.14317',
  abstract: 'This is a test abstract.',
  published: '2024-08-26',
  updated: '2024-08-26',
  version: 1,
  authors: ['Author A', 'Author B'],
  categories: ['cs.AI'],
  relevanceScore: 0.95,
};

const parsed = arxivPaperSchema.parse(validPaper);
assert(parsed.id === '2408.14317', 'ArxivPaper id');
assert(parsed.authors.length === 2, 'ArxivPaper authors');
assert(parsed.relevanceScore === 0.95, 'ArxivPaper relevanceScore');

// null の relevanceScore
const paperWithNull = { ...validPaper, relevanceScore: null };
const parsedNull = arxivPaperSchema.parse(paperWithNull);
assert(parsedNull.relevanceScore === null, 'ArxivPaper null relevanceScore');

// XML 変換
const xml = arxivPaperToXml(parsed);
assert(xml.includes('<id>2408.14317</id>'), 'arxivPaperToXml id');
assert(xml.includes('<title>Test Paper</title>'), 'arxivPaperToXml title');

console.log('  ArxivPaper: PASS');

// --- ReadingResult ---
console.log('Testing ReadingResult schema...');

const validResult = {
  id: 0,
  task: 'Test task',
  paper: validPaper,
  markdownPath: 'storage/markdown/2408.14317.md',
  answer: 'Test answer',
  isRelated: true,
};

const parsedResult = readingResultSchema.parse(validResult);
assert(parsedResult.id === 0, 'ReadingResult id');
assert(parsedResult.task === 'Test task', 'ReadingResult task');
assert(parsedResult.isRelated === true, 'ReadingResult isRelated');

// デフォルト値
const minimalResult = {
  id: 1,
  task: 'Minimal',
  paper: validPaper,
  markdownPath: 'test.md',
};
const parsedMinimal = readingResultSchema.parse(minimalResult);
assert(parsedMinimal.answer === '', 'ReadingResult default answer');

console.log('  ReadingResult: PASS');

// --- Section ---
console.log('Testing Section schema...');

const validSection = {
  header: 'Introduction',
  content: 'This is the intro.',
  charCount: 18,
};
const parsedSection = sectionSchema.parse(validSection);
assert(parsedSection.header === 'Introduction', 'Section header');

console.log('  Section: PASS');

// --- LLM schemas ---
console.log('Testing LLM schemas...');

const hearing = hearingSchema.parse({
  is_need_human_feedback: true,
  additional_question: 'What specific area?',
});
assert(hearing.is_need_human_feedback === true, 'Hearing is_need_human_feedback');

const tasks = decomposedTasksSchema.parse({
  tasks: ['task1', 'task2', 'task3'],
});
assert(tasks.tasks.length === 3, 'DecomposedTasks length');

const evaluation = taskEvaluationSchema.parse({
  need_more_information: false,
  reason: 'Sufficient',
  content: '',
});
assert(evaluation.need_more_information === false, 'TaskEvaluation');

const sufficiency = sufficiencySchema.parse({
  is_sufficient: true,
  reason: 'Enough info',
});
assert(sufficiency.is_sufficient === true, 'Sufficiency');

console.log('  LLM schemas: PASS');

// --- formatTimeRange ---
console.log('Testing formatTimeRange...');

assert(
  formatTimeRange({ start: '2024-01-01', end: '2024-12-31' }) ===
    '20240101+TO+20241231',
  'formatTimeRange both',
);
assert(
  formatTimeRange({ start: '2024-01-01', end: null }) ===
    '20240101+TO+LATEST',
  'formatTimeRange start only',
);
assert(
  formatTimeRange({ start: null, end: '2024-12-31' }) ===
    'EARLIEST+TO+20241231',
  'formatTimeRange end only',
);
assert(
  formatTimeRange({ start: null, end: null }) === null,
  'formatTimeRange none',
);

console.log('  formatTimeRange: PASS');

console.log('\nAll model tests passed!');
