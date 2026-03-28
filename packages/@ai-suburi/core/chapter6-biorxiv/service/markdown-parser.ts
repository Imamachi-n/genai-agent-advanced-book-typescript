import type { Section } from '../models.js';

export class MarkdownParser {
  parseSections(text: string): Section[] {
    const sections: Section[] = [];
    const lines = text.split('\n');
    let currentHeader: string | null = null;
    const sectionContent: string[] = [];

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }

      const headerMatch = line.trim().match(/^(#+)\s+(.+)$/);

      if (headerMatch) {
        if (currentHeader) {
          const sectionText = sectionContent.join('\n');
          sections.push({
            header: currentHeader,
            content: sectionText,
            charCount: sectionText.length,
          });
          sectionContent.length = 0;
        }
        currentHeader = headerMatch[2]!;
      } else if (currentHeader) {
        sectionContent.push(line);
      }
    }

    // 最後のセクションを追加
    if (currentHeader) {
      const sectionText = sectionContent.join('\n');
      sections.push({
        header: currentHeader,
        content: sectionText,
        charCount: sectionText.length,
      });
    }

    return sections;
  }

  formatAsXml(sections: Section[]): string {
    const output: string[] = [];
    output.push('<items>');
    for (let i = 0; i < sections.length; i++) {
      const section = sections[i]!;
      const firstLine = section.content.split('\n')[0]?.trim().slice(0, 200) ?? '';
      output.push('  <item>');
      output.push(`    <index>${i + 1}</index>`);
      output.push(`    <header>${section.header}</header>`);
      output.push(`    <first_line>${firstLine}</first_line>`);
      output.push(`    <char_count>${section.charCount}</char_count>`);
      output.push('  </item>');
    }
    output.push('</items>');
    return output.join('\n');
  }

  getSectionsOverview(text: string): string {
    const sections = this.parseSections(text);
    return this.formatAsXml(sections);
  }

  getSelectedSections(text: string, sectionIndices: number[]): string {
    const sections = this.parseSections(text);
    const selectedSections: string[] = [];
    for (const sectionIndex of sectionIndices) {
      if (sectionIndex >= 1 && sectionIndex <= sections.length) {
        const section = sections[sectionIndex - 1]!;
        selectedSections.push(
          `<section>\n<header>${section.header}</header>\n<content>${section.content}</content>\n</section>`,
        );
      }
    }
    return selectedSections.join('\n');
  }
}
