// Playwright E2E tests for dashboard
const { test, expect } = require('@playwright/test');

const DASHBOARD_URL = 'http://localhost:8000';

test.describe('Dashboard End-to-End', () => {
  test('Dashboard loads and displays main sections', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await expect(page.locator('.topbar')).toBeVisible();
    await expect(page.locator('.stat-cards')).toBeVisible();
    await expect(page.locator('.card.right-panel')).toBeVisible();
    await expect(page.locator('.logs-section')).toBeVisible();
    await expect(page.locator('#candle-chart')).toBeVisible();
    await expect(page.locator('#perf-chart')).toBeVisible();
  });

  test('Order book and metrics update with backend data', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    // Wait for polling to update
    await page.waitForTimeout(2500);
    const ask = await page.locator('.badge.ask').textContent();
    const bid = await page.locator('.badge.bid').textContent();
    expect(ask).not.toContain('--');
    expect(bid).not.toContain('--');
  });

  test('Start and stop simulation updates UI', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await page.click('#start-btn');
    await expect(page.locator('#start-btn')).toHaveText(/Running/);
    await page.waitForTimeout(2000);
    await page.click('#stop-btn');
    await expect(page.locator('#start-btn')).toHaveText(/Start Simulation/);
  });

  test('Clear session data resets charts and logs', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await page.click('#clear-btn');
    // Attendre dynamiquement que le tag du chart soit bien Waiting ou --
    await expect.poll(async () => {
      return await page.locator('#chart-tag').textContent();
    }, { timeout: 4000 }).toMatch(/Waiting|--/);
    const tradeRows = await page.locator('#trade-log-body tr').count();
    expect(tradeRows).toBe(0);
  });

  test('Switching product triggers backend and updates UI', async ({ page }) => {
    await page.goto(DASHBOARD_URL);
    await page.selectOption('#product-select', { label: 'BTCUSDT' });
    // Attendre dynamiquement que le sous-titre affiche le bon symbol
    await expect.poll(async () => {
      return await page.locator('.brand-text p').textContent();
    }, { timeout: 4000 }).toContain('BTCUSDT');
  });
});
