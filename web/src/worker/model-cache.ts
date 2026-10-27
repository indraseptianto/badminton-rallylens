import { AnalysisError } from "../domain/errors";
import type { ModelManifestEntry } from "../domain/types";

const CACHE_NAME = "rallylens-models-v1";

function bytesToHex(bytes: ArrayBuffer): string {
  return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function sha256(buffer: ArrayBuffer): Promise<string> {
  return bytesToHex(await crypto.subtle.digest("SHA-256", buffer));
}

export async function getVerifiedModel(entry: ModelManifestEntry, onDownload?: (loaded: number, total?: number) => void): Promise<ArrayBuffer> {
  if (!entry.license.redistributionVerified) throw new AnalysisError("download", "MODEL_UNAVAILABLE", `${entry.id} has not passed redistribution review.`, "Use a release containing verified model assets; do not copy unverified checkpoints into the app.");
  if (!/^[a-f0-9]{64}$/i.test(entry.sha256)) throw new AnalysisError("download", "MODEL_MANIFEST_INVALID", `${entry.id} has an invalid SHA-256.`, "Fix the release model manifest.");
  const cache = await caches.open(CACHE_NAME);
  const key = new Request(new URL(`./${entry.id}-${entry.version}-${entry.sha256}.onnx`, self.location.href));
  let response = await cache.match(key);
  if (!response) {
    const remote = await fetch(entry.url);
    if (!remote.ok) throw new AnalysisError("download", "MODEL_UNAVAILABLE", `Could not download ${entry.id} (HTTP ${remote.status}).`, "Check your connection and retry.");
    const total = Number(remote.headers.get("content-length")) || undefined;
    if (!remote.body) throw new AnalysisError("download", "MODEL_UNAVAILABLE", `Empty response for ${entry.id}.`, "Retry the model download.");
    const reader = remote.body.getReader(); const chunks: Uint8Array[] = []; let loaded = 0;
    while (true) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); loaded += value.byteLength; onDownload?.(loaded, total); }
    const merged = new Uint8Array(loaded); let offset = 0; for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.byteLength; }
    response = new Response(merged, { headers: { "content-type": "application/octet-stream" } });
    await cache.put(key, response.clone());
  }
  const buffer = await response.arrayBuffer();
  if ((await sha256(buffer)).toLowerCase() !== entry.sha256.toLowerCase()) { await cache.delete(key); throw new AnalysisError("download", "MODEL_INTEGRITY_FAILED", `Checksum mismatch for ${entry.id}.`, "Retry. If it persists, report the release asset as corrupt."); }
  return buffer;
}

export async function pruneModelCache(entries: ModelManifestEntry[]): Promise<void> {
  const cache = await caches.open(CACHE_NAME); const active = new Set(entries.map((e) => `${e.id}-${e.version}-${e.sha256}.onnx`));
  for (const request of await cache.keys()) if (![...active].some((key) => request.url.endsWith(key))) await cache.delete(request);
}
