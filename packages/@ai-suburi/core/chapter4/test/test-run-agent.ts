import { HelpDeskAgent } from '../agent.js';
import { loadSettings } from '../configs.js';
import { searchXyzManual } from '../tools/search-xyz-manual/search-xyz-manual.js';
import { searchXyzQa } from '../tools/search-xyz-qa/search-xyz-qa.js';

const settings = loadSettings();

const agent = new HelpDeskAgent(settings, [searchXyzManual, searchXyzQa]);

// const question = `
// お世話になっております。
//
// 現在、XYZシステムの利用を検討しており、以下の2点についてご教示いただければと存じます。
//
// 1. パスワードに利用可能な文字の制限について
// 当該システムにてパスワードを設定する際、使用可能な文字の範囲（例：英数字、記号、文字数制限など）について詳しい情報をいただけますでしょうか。安全かつシステムでの認証エラーを防ぐため、具体的な仕様を確認したいと考えております。
//
// 2. 最新リリースの取得方法について
// 最新のアップデート情報をどのように確認・取得できるかについてもお教えいただけますと幸いです。
//
// お忙しいところ恐縮ですが、ご対応のほどよろしくお願い申し上げます。
// `;

const question = `
お世話になっております。

現在、XYZシステムを利用しており、以下の点についてご教示いただければと存じます。

1. 二段階認証の設定について
SMS認証が使えない環境のため、認証アプリを利用した二段階認証の設定手順を教えていただけますでしょうか。

2. バックアップ失敗時の通知について
バックアップ監視機能で通知を設定しているにもかかわらず、バックアップ失敗時に通知が届きません。確認すべき箇所を教えていただけますでしょうか。

お忙しいところ恐縮ですが、ご対応のほどよろしくお願い申し上げます。
`;

const result = await agent.runAgent(question);
// 回答
console.log(result.answer);
