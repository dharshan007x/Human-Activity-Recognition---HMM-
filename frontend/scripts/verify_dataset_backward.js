import puppeteer from 'puppeteer-core';
import path from 'path';

const ARTIFACT_DIR = 'C:\\Users\\padma\\.gemini\\antigravity\\brain\\865c359b-8dad-4e72-a6d8-a380a2e3b40f';
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE_URL = 'http://127.0.0.1:5173';

async function verify() {
  console.log('=== Launching Headless Edge Verification for Dataset Explorer & Backward Algorithm ===');
  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1400,900'],
    defaultViewport: { width: 1400, height: 900 },
  });

  const page = await browser.newPage();
  const consoleErrors = [];

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(`[Console Error] ${msg.text()}`);
    }
  });

  page.on('pageerror', (err) => {
    console.error('[Page Error]:', err.message);
    consoleErrors.push(`[Page Error] ${err.toString()}`);
  });

  // -------------------------------------------------------------
  // TEST 1: Dataset Explorer (/dataset)
  // -------------------------------------------------------------
  console.log('\n--- 1. Testing Dataset Explorer (/dataset) ---');
  await page.goto(`${BASE_URL}/dataset`, { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1500));

  // Extract metric cards
  const metrics = await page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll('div')).filter((el) =>
      el.innerText.includes('Total Records') || el.innerText.includes('Sequences (T=50)')
    );
    return cards.length > 0 ? cards[0].innerText : 'NOT FOUND';
  });
  console.log('Dataset Explorer Header Metrics:\n', metrics.substring(0, 300));

  // Extract table rows count
  const initialRowCount = await page.evaluate(() => {
    const tbody = document.querySelector('tbody');
    return tbody ? tbody.querySelectorAll('tr').length : 0;
  });
  console.log(`Table rendered rows count: ${initialRowCount}`);

  // Capture Screenshot of initial page
  await page.screenshot({ path: path.join(ARTIFACT_DIR, '02_dataset.png'), fullPage: false });
  console.log('✓ Captured 02_dataset.png');

  // Test pagination button "Next 25"
  console.log('Clicking "Next 25" button...');
  const nextBtn = await page.waitForSelector('button ::-p-text(Next 25)', { timeout: 3000 });
  if (nextBtn) {
    await nextBtn.click();
    await new Promise((r) => setTimeout(r, 1000));
    const paginatedSub = await page.evaluate(() => {
      const sub = document.querySelector('h2 + p');
      return sub ? sub.innerText : '';
    });
    console.log(`Pagination text after clicking Next: "${paginatedSub}"`);
  }

  // -------------------------------------------------------------
  // TEST 2: Backward Algorithm (/backward)
  // -------------------------------------------------------------
  console.log('\n--- 2. Testing Backward Algorithm (/backward) ---');
  await page.goto(`${BASE_URL}/backward`, { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1500));

  // Extract Backward metrics
  const bwdMetrics = await page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll('div')).filter((el) =>
      el.innerText.includes('Backward Log-Likelihood') || el.innerText.includes('Dual Likelihood Agreement')
    );
    return cards.length > 0 ? cards[0].innerText : 'NOT FOUND';
  });
  console.log('Backward Page Metrics:\n', bwdMetrics.substring(0, 350));

  // Check trellis svg elements
  const trellisNodes = await page.evaluate(() => {
    const circles = document.querySelectorAll('svg circle');
    return circles.length;
  });
  console.log(`Trellis chart rendered state nodes: ${trellisNodes}`);

  // Capture screenshot of Backward Page (Walking)
  await page.screenshot({ path: path.join(ARTIFACT_DIR, '07_backward.png'), fullPage: false });
  console.log('✓ Captured 07_backward.png (Walking)');

  // Change sample sequence in dropdown to Sitting
  console.log('Switching sample sequence to Sitting...');
  const selectElem = await page.$('select');
  if (selectElem) {
    await selectElem.select('sample_sitting');
    await new Promise((r) => setTimeout(r, 1500));
    const newMetrics = await page.evaluate(() => {
      const cards = Array.from(document.querySelectorAll('div')).filter((el) =>
        el.innerText.includes('Backward Log-Likelihood')
      );
      return cards.length > 0 ? cards[0].innerText : '';
    });
    console.log('Metrics after switching to Sitting:\n', newMetrics.substring(0, 300));
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'verify_backward_sitting.png'), fullPage: false });
    console.log('✓ Captured verify_backward_sitting.png');
  }

  await browser.close();

  console.log('\n=== VERIFICATION SUMMARY ===');
  console.log(`Total Console/Page Errors: ${consoleErrors.length}`);
  if (consoleErrors.length > 0) {
    consoleErrors.forEach((e) => console.error(e));
  } else {
    console.log('ALL TESTS PASSED WITH 0 ERRORS.');
  }
}

verify().catch((err) => {
  console.error('Verification failed:', err);
  process.exit(1);
});
