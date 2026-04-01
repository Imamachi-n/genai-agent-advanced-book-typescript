import * as fs from 'node:fs';
import OpenAI from 'openai';
import { loadSettings } from '../../configs.js';
import { setupLogger } from '../../custom-logger.js';

const logger = setupLogger('run-fine-tuning');

const DEFAULT_BASE_MODEL = 'gpt-4.1-nano-2025-04-14';
const POLL_INTERVAL_MS = 30000;

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let trainingFile = '';
  let validationFile: string | undefined;
  let baseModel = DEFAULT_BASE_MODEL;
  let suffix: string | undefined;
  let nEpochs: number | 'auto' = 'auto';
  let statusOnly: string | undefined;
  let cancelJob: string | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--training-file' && args[i + 1]) {
      trainingFile = args[i + 1]!;
      i++;
    } else if (args[i] === '--validation-file' && args[i + 1]) {
      validationFile = args[i + 1]!;
      i++;
    } else if (args[i] === '--model' && args[i + 1]) {
      baseModel = args[i + 1]!;
      i++;
    } else if (args[i] === '--suffix' && args[i + 1]) {
      suffix = args[i + 1]!;
      i++;
    } else if (args[i] === '--n-epochs' && args[i + 1]) {
      const val = args[i + 1]!;
      nEpochs = val === 'auto' ? 'auto' : Number.parseInt(val, 10);
      i++;
    } else if (args[i] === '--status' && args[i + 1]) {
      statusOnly = args[i + 1]!;
      i++;
    } else if (args[i] === '--cancel' && args[i + 1]) {
      cancelJob = args[i + 1]!;
      i++;
    }
  }

  const settings = loadSettings();
  const openai = new OpenAI({ apiKey: settings.openaiApiKey });

  // --- ジョブ状態確認モード ---
  if (statusOnly) {
    await showJobStatus(openai, statusOnly);
    return;
  }

  // --- ジョブキャンセルモード ---
  if (cancelJob) {
    const cancelled = await openai.fineTuning.jobs.cancel(cancelJob);
    logger.info(`Job cancelled: ${cancelled.id} (status: ${cancelled.status})`);
    return;
  }

  // --- メインフロー: ファイルアップロード → ジョブ作成 → 完了待ち ---
  if (!trainingFile) {
    printUsage();
    process.exit(1);
  }

  if (!fs.existsSync(trainingFile)) {
    console.error(`File not found: ${trainingFile}`);
    process.exit(1);
  }

  // Step 1: 学習データのアップロード
  logger.info(`Step 1: Uploading training file: ${trainingFile}`);
  const uploadedTraining = await openai.files.create({
    file: fs.createReadStream(trainingFile),
    purpose: 'fine-tune',
  });
  logger.info(`Training file uploaded: ${uploadedTraining.id}`);

  // 検証データのアップロード（任意）
  let uploadedValidationId: string | undefined;
  if (validationFile) {
    if (!fs.existsSync(validationFile)) {
      console.error(`Validation file not found: ${validationFile}`);
      process.exit(1);
    }
    logger.info(`Uploading validation file: ${validationFile}`);
    const uploadedValidation = await openai.files.create({
      file: fs.createReadStream(validationFile),
      purpose: 'fine-tune',
    });
    uploadedValidationId = uploadedValidation.id;
    logger.info(`Validation file uploaded: ${uploadedValidationId}`);
  }

  // Step 2: ファインチューニングジョブの作成
  logger.info(`Step 2: Creating fine-tuning job (model: ${baseModel})`);
  const job = await openai.fineTuning.jobs.create({
    model: baseModel,
    training_file: uploadedTraining.id,
    ...(uploadedValidationId ? { validation_file: uploadedValidationId } : {}),
    ...(suffix ? { suffix } : {}),
    hyperparameters: {
      n_epochs: nEpochs,
    },
  });

  logger.info(`Job created: ${job.id}`);
  logger.info(`Status: ${job.status}`);
  console.log(`\nFine-tuning job started!`);
  console.log(`  Job ID: ${job.id}`);
  console.log(`  Model: ${baseModel}`);
  console.log(`  Training file: ${uploadedTraining.id}`);
  if (uploadedValidationId) {
    console.log(`  Validation file: ${uploadedValidationId}`);
  }

  // Step 3: 完了をポーリングで待機
  logger.info('Step 3: Waiting for completion...');
  console.log(`\nPolling every ${POLL_INTERVAL_MS / 1000}s. Press Ctrl+C to stop (job continues on OpenAI side).\n`);

  let lastStatus = job.status;
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));

    const current = await openai.fineTuning.jobs.retrieve(job.id);

    if (current.status !== lastStatus) {
      logger.info(`Status changed: ${lastStatus} → ${current.status}`);
      lastStatus = current.status;
    }

    // イベントログを表示
    const events = await openai.fineTuning.jobs.listEvents(job.id, { limit: 5 });
    for (const event of events.data.reverse()) {
      logger.info(`  [${event.level}] ${event.message}`);
    }

    if (current.status === 'succeeded') {
      console.log('\n=== Fine-tuning completed! ===');
      console.log(`  Fine-tuned model: ${current.fine_tuned_model}`);
      console.log(`  Trained tokens: ${current.trained_tokens}`);
      console.log(`\nTo use this model, set the environment variable:`);
      console.log(`  OPENAI_FAST_MODEL=${current.fine_tuned_model}`);
      break;
    }

    if (current.status === 'failed') {
      console.error('\n=== Fine-tuning failed! ===');
      console.error(`  Error: ${current.error?.message ?? 'Unknown error'}`);
      process.exit(1);
    }

    if (current.status === 'cancelled') {
      console.log('\n=== Fine-tuning cancelled ===');
      break;
    }
  }
}

async function showJobStatus(openai: OpenAI, jobId: string): Promise<void> {
  const job = await openai.fineTuning.jobs.retrieve(jobId);
  console.log(`\nJob: ${job.id}`);
  console.log(`  Status: ${job.status}`);
  console.log(`  Model: ${job.model}`);
  console.log(`  Fine-tuned model: ${job.fine_tuned_model ?? '(not yet)'}`);
  console.log(`  Trained tokens: ${job.trained_tokens ?? '(not yet)'}`);
  console.log(`  Created at: ${new Date((job.created_at ?? 0) * 1000).toISOString()}`);

  if (job.error?.message) {
    console.log(`  Error: ${job.error.message}`);
  }

  // 最新イベント
  console.log('\nRecent events:');
  const events = await openai.fineTuning.jobs.listEvents(jobId, { limit: 10 });
  for (const event of events.data.reverse()) {
    console.log(`  [${event.level}] ${event.message}`);
  }
}

function printUsage(): void {
  console.error('Usage:');
  console.error(
    '  npx tsx chapter6-biorxiv/rag/ft-pipeline/run-fine-tuning.ts --training-file <path> [options]',
  );
  console.error('\nOptions:');
  console.error(
    `  --training-file <path>    学習データ JSONL ファイル（必須）`,
  );
  console.error(
    '  --validation-file <path>  検証データ JSONL ファイル（任意）',
  );
  console.error(
    `  --model <name>            ベースモデル（default: ${DEFAULT_BASE_MODEL}）`,
  );
  console.error(
    '  --suffix <name>           FT モデル名のサフィックス（例: biorxiv-query）',
  );
  console.error(
    '  --n-epochs <n|auto>       エポック数（default: auto）',
  );
  console.error(
    '  --status <job-id>         既存ジョブの状態を確認',
  );
  console.error(
    '  --cancel <job-id>         既存ジョブをキャンセル',
  );
}

main().catch((error) => {
  console.error(`\nError: ${(error as Error).message}`);
  process.exit(1);
});
