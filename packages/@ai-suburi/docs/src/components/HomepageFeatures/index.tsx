import clsx from "clsx";
import type { ReactNode } from "react";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  imgSrc: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: "AI エージェントを知る",
    imgSrc: require("@site/static/img/1-icon.png").default,
    description: (
      <>
        AI
        エージェントの定義やビジネスでの位置づけから、ツール呼び出し・計画・自己修正・メモリといった内部構成まで、
        全体像を体系的に理解できます。
      </>
    ),
  },
  {
    title: "AI エージェントを作る",
    imgSrc: require("@site/static/img/2-icon.png").default,
    description: (
      <>
        OpenAI API と TypeScript を使い、ヘルプデスク支援・データ分析・
        情報収集・マーケティング支援など、目的別の AI
        エージェントを実際に構築していきます。
      </>
    ),
  },
  {
    title: "AI エージェントを現場で使う",
    imgSrc: require("@site/static/img/3-icon.png").default,
    description: (
      <>
        評価指標の設計やエラー分析、UX・リスク管理・モニタリングなど、AI
        エージェントを本番環境で運用するための実践的なノウハウを企業の実用化事例とともに学べます。
      </>
    ),
  },
];

function Feature({ title, imgSrc, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className={styles.featureCard}>
        <div className={styles.featureCardAccent} />
        <div className={styles.featureIconWrapper}>
          <img className={styles.featureSvg} src={imgSrc} alt={title} />
        </div>
        <h3 className={styles.featureTitle}>{title}</h3>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <h2 className={styles.sectionTitle}>このドキュメントで学べること</h2>
        <p className={styles.sectionSubtitle}>
          AI エージェントの理解から実装・運用まで、3 つのステップで学べます
        </p>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
