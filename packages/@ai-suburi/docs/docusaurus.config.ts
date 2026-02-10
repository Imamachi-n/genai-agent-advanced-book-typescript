import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import { themes as prismThemes } from "prism-react-renderer";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "現場で活用するためのAIエージェント実践入門（TypeScript 版）",
  tagline: "TypeScript で学ぶ AI エージェント開発の実践知",
  favicon: "img/top-icon.png",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  markdown: {
    mermaid: true,
  },
  themes: ["@docusaurus/theme-mermaid"],

  // Set the production url of your site here
  url: "https://Imamachi-n.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/genai-agent-advanced-book-typescript/",

  // GitHub pages deployment config.
  organizationName: "Imamachi-n",
  projectName: "genai-agent-advanced-book-typescript",

  onBrokenLinks: "throw",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            "https://github.com/Imamachi-n/genai-agent-advanced-book-typescript/tree/main/packages/create-docusaurus/templates/shared/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: "img/social-card.png",
    colorMode: {
      respectPrefersColorScheme: true,
    },
    mermaid: {
      theme: {
        light: "base",
        dark: "dark",
      },
      options: {
        themeVariables: {
          primaryColor: "#DBEAFE",
          primaryTextColor: "#1E3A5F",
          primaryBorderColor: "#3B82F6",
          lineColor: "#94A3B8",
          secondaryColor: "#EDE9FE",
          tertiaryColor: "#F0F9FF",
          fontFamily: "inherit",
        },
      },
    },
    navbar: {
      title: "現場で活用するためのAIエージェント実践入門（TypeScript 版）",
      logo: {
        alt: "現場で活用するためのAIエージェント実践入門（TypeScript 版） Logo",
        src: "img/top-icon.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docsSidebar",
          position: "left",
          label: "Docs",
        },
        {
          href: "https://github.com/Imamachi-n/genai-agent-advanced-book-typescript",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Chapter 2: AIエージェントの構成",
              to: "/docs/chapter2",
            },
            {
              label: "Chapter 3: AIエージェントの開発準備",
              to: "/docs/chapter3",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/Imamachi-n/genai-agent-advanced-book-typescript",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Naoto Imamachi`,
    },
    prism: {
      theme: prismThemes.nightOwl,
      darkTheme: prismThemes.nightOwl,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
