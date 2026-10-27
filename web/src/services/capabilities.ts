export interface CapabilityReport {
  supported: boolean;
  reasons: string[];
  features: { chrome: boolean; webgpu: boolean; webcodecs: boolean; workers: boolean };
}

export async function checkCapabilities(): Promise<CapabilityReport> {
  const match = navigator.userAgent.match(/(?:Chrome|Chromium)\/(\d+)/);
  const chrome = !!match && Number(match[1]) >= 148 && !/(Edg|OPR)\//.test(navigator.userAgent);
  const webgpu = "gpu" in navigator && !!(await navigator.gpu?.requestAdapter().catch(() => null));
  const webcodecs = "VideoDecoder" in window && "VideoEncoder" in window;
  const workers = "Worker" in window && "OffscreenCanvas" in window;
  const reasons: string[] = [];
  if (!chrome) reasons.push("RallyLens requires desktop Google Chrome 148 or newer.");
  if (!webgpu) reasons.push("WebGPU is unavailable. Enable hardware acceleration and update your GPU driver.");
  if (!webcodecs) reasons.push("WebCodecs is unavailable in this browser context.");
  if (!workers) reasons.push("Worker or OffscreenCanvas support is unavailable.");
  return { supported: reasons.length === 0, reasons, features: { chrome, webgpu, webcodecs, workers } };
}
