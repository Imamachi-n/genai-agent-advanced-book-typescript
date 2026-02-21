FROM --platform=linux/arm64 node:24-slim

WORKDIR /app

# pnpm をインストール
RUN npm install -g pnpm

# 依存関係を先にコピーしてキャッシュを活用
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --prod --frozen-lockfile

# ソースコードをコピーしてビルド
COPY . .
RUN pnpm build

EXPOSE 8080

CMD ["node", "dist/index.js"]
