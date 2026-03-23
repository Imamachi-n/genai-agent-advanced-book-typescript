import { MarkdownParser } from '../service/markdown-parser.js';

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`FAIL: ${message}`);
  }
}

const parser = new MarkdownParser();

// --- parseSections ---
console.log('Testing MarkdownParser.parseSections...');

const markdown = `# Introduction
This is the introduction.
Some more content here.

## Methods
Description of methods.

### Sub Methods
Sub method details.

## Results
The results are great.
Very significant findings.

## Conclusion
We conclude that this works.
`;

const sections = parser.parseSections(markdown);
assert(sections.length === 5, `Expected 5 sections, got ${sections.length}`);
assert(sections[0]!.header === 'Introduction', 'First section header');
assert(
  sections[0]!.content.includes('This is the introduction'),
  'First section content',
);
assert(sections[1]!.header === 'Methods', 'Second section header');
assert(sections[2]!.header === 'Sub Methods', 'Third section header (sub)');
assert(sections[3]!.header === 'Results', 'Fourth section header');
assert(sections[4]!.header === 'Conclusion', 'Fifth section header');

console.log('  parseSections: PASS');

// --- formatAsXml ---
console.log('Testing MarkdownParser.formatAsXml...');

const xml = parser.formatAsXml(sections);
assert(xml.includes('<items>'), 'XML has items tag');
assert(xml.includes('<index>1</index>'), 'XML has index 1');
assert(xml.includes('<header>Introduction</header>'), 'XML has header');
assert(xml.includes('</items>'), 'XML closes items');

console.log('  formatAsXml: PASS');

// --- getSectionsOverview ---
console.log('Testing MarkdownParser.getSectionsOverview...');

const overview = parser.getSectionsOverview(markdown);
assert(overview.includes('<items>'), 'Overview has items');
assert(overview.includes('<index>5</index>'), 'Overview has all indices');

console.log('  getSectionsOverview: PASS');

// --- getSelectedSections ---
console.log('Testing MarkdownParser.getSelectedSections...');

const selected = parser.getSelectedSections(markdown, [1, 4]);
assert(selected.includes('<header>Introduction</header>'), 'Selected has intro');
assert(selected.includes('<header>Results</header>'), 'Selected has results');
assert(!selected.includes('<header>Methods</header>'), 'Selected excludes methods');

console.log('  getSelectedSections: PASS');

// --- empty markdown ---
console.log('Testing empty markdown...');

const emptySections = parser.parseSections('');
assert(emptySections.length === 0, 'Empty markdown returns 0 sections');

const noHeaderSections = parser.parseSections('Just some text without headers');
assert(
  noHeaderSections.length === 0,
  'No header markdown returns 0 sections',
);

console.log('  Empty markdown: PASS');

// --- out of range indices ---
console.log('Testing out of range indices...');

const outOfRange = parser.getSelectedSections(markdown, [0, 100]);
assert(outOfRange === '', 'Out of range returns empty');

console.log('  Out of range: PASS');

console.log('\nAll MarkdownParser tests passed!');
