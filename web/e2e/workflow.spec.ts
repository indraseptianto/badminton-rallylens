import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", { configurable: true, value: { requestAdapter: async () => ({}) } });
    if (!("VideoDecoder" in window)) Object.defineProperty(window, "VideoDecoder", { value: class {} });
    if (!("VideoEncoder" in window)) Object.defineProperty(window, "VideoEncoder", { value: class {} });
    if (!("OffscreenCanvas" in window)) Object.defineProperty(window, "OffscreenCanvas", { value: class {} });
  });
});

test("opens the private video source step", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /See the match/ })).toBeVisible();
  await expect(page.getByText("Runs entirely on this device")).toBeVisible();
});

test("rejects platform page URLs before network access", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("https://example.com/match.mp4").fill("https://www.youtube.com/watch?v=not-a-direct-file");
  await page.getByRole("button", { name: "Use URL" }).click();
  await expect(page.getByRole("alert")).toContainText("PLATFORM URL UNSUPPORTED");
});
