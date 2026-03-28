import * as fs from 'node:fs';
import * as path from 'node:path';

export class MarkdownStorage {
  private baseDir: string;

  constructor(baseDir: string = 'storage/markdown') {
    this.baseDir = baseDir;
    fs.mkdirSync(baseDir, { recursive: true });
  }

  write(filename: string, content: string): string {
    const filepath = path.join(this.baseDir, filename);
    fs.writeFileSync(filepath, content, 'utf-8');
    return path.join(this.baseDir, filename);
  }

  read(filePath: string): string {
    const resolvedPath = path.isAbsolute(filePath)
      ? filePath
      : path.join(process.cwd(), filePath);
    return fs.readFileSync(resolvedPath, 'utf-8');
  }
}
