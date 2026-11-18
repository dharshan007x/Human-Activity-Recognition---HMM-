import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';

const ARTIFACT_DIR = 'C:\\Users\\padma\\.gemini\\antigravity\\brain\\865c359b-8dad-4e72-a6d8-a380a2e3b40f';
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE_URL = 'http://127.0.0.1:5173';

const routes = [
  { path: '/', name: '01_overview', title: 'Overview & Monograph' },
  { path: '/dataset', name: '02_dataset', title: 'Dataset Explorer' },
  { path: '/preprocessing', name: '03_preprocessing', title: 'Signal Preprocessing' },
  { path: '/sensor-explorer', name: '04_sensor_explorer', title: '3D Sensor Explorer' },
  { path: '/model-parameters', name: '05_model_parameters', title: 'HMM Parameters' },
  { path: '/forward', name: '06_forward', title: 'Forward Algorithm' },
  { path: '/backward', name: '07_backward', title: 'Backward Algorithm' },
  { path: '/posterior', name: '08_posterior', title: 'Posterior Decoding' },
  { path: '/recognition', name: '09_recognition', title: 'Live Activity Recognition' },
  { path: '/evaluation', name: '10_evaluation', title: 'Model Evaluation' },
  { path: '/about', name: '11_viva_defense', title: 'Viva Defense & Theory' },
];

async function runAudit() {
  console.log('--- Launching Headless Microsoft Edge QA Audit ---');
  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1400,900'],
    defaultViewport: { width: 1400, height: 900 },
  });

  const page = await browser.newPage();
  const consoleErrors = [];
  let currentRoute = 'initial';

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(`[Browser Console Error on ${currentRoute}] ${msg.text()}`);
    }
  });

  page.on('pageerror', (err) => {
    const errorStr = `[Uncaught Page Error on ${currentRoute}] ${err.stack || err.toString()}`;
    console.error(errorStr);
    consoleErrors.push(errorStr);
  });

  page.on('dialog', async (dialog) => {
    console.log(`[Dialog Triggered on ${currentRoute}] ${dialog.message()}`);
    await dialog.dismiss();
  });

  const results = [];

  for (const r of routes) {
    currentRoute = `${r.title} (${r.path})`;
    const url = `${BASE_URL}${r.path}`;
    console.log(`Navigating to ${r.title} (${url})...`);

    try {
      await page.goto(url, { waitUntil: 'networkidle0', timeout: 15000 });
      // Wait extra 1000ms for Three.js WebGL & D3 animations to settle
      await new Promise((res) => setTimeout(res, 1000));

      // If recognition page, click "Load Built-in PAMAP2 Validation Sample" button
      if (r.path === '/recognition') {
        try {
          const sampleBtn = await page.waitForSelector('button ::-p-text(Load Built-in)', { timeout: 3000 });
          if (sampleBtn) {
            console.log('Clicking "Load Built-in PAMAP2 Validation Sample"...');
            await sampleBtn.click();
            await new Promise((res) => setTimeout(res, 2000));
          }
        } catch (e) {
          console.log('Sample button click note:', e.message);
        }
      }

      const screenshotPath = path.join(ARTIFACT_DIR, `${r.name}.png`);
      await page.screenshot({ path: screenshotPath, fullPage: false });

      console.log(`✓ Captured screenshot: ${screenshotPath}`);
      results.push({ ...r, status: 'SUCCESS', screenshot: `${r.name}.png` });
    } catch (err) {
      console.error(`✗ Error on ${r.title}:`, err.message);
      results.push({ ...r, status: 'ERROR', error: err.message });
    }
  }

  await browser.close();

  console.log('\n--- QA Audit Summary ---');
  console.log(`Total Pages Inspected: ${results.length}`);
  console.log(`Success: ${results.filter((r) => r.status === 'SUCCESS').length}`);
  console.log(`Errors Encountered: ${consoleErrors.length}`);
  if (consoleErrors.length > 0) {
    consoleErrors.forEach((e) => console.error(e));
  }

  fs.writeFileSync(
    path.join(ARTIFACT_DIR, 'qa_audit_results.json'),
    JSON.stringify({ results, consoleErrors, timestamp: new Date().toISOString() }, null, 2)
  );
  console.log('Results saved to qa_audit_results.json');
}

runAudit().catch((err) => {
  console.error('Audit Script Failed:', err);
  process.exit(1);
});
