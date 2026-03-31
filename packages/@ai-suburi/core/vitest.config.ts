import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    include: ['chapter6-biorxiv/test/**/*.test.ts'],
  },
});
