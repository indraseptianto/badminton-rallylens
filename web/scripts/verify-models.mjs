import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(await readFile(resolve(root, "public/models/manifest.json"), "utf8"));
if (manifest.schemaVersion !== "1" || !Array.isArray(manifest.models)) throw new Error("Invalid model manifest");
const ids = new Set();
for (const model of manifest.models) {
  if (ids.has(model.id)) throw new Error(`Duplicate model id: ${model.id}`); ids.add(model.id);
  if (!model.license?.redistributionVerified || !model.license?.source) throw new Error(`Unverified model license: ${model.id}`);
  if (!/^[a-f0-9]{64}$/i.test(model.sha256)) throw new Error(`Invalid SHA-256: ${model.id}`);
  const shape = JSON.stringify(model.input?.shape);
  const validShape = model.task === "player" ? shape === "[1,3,640,640]" || shape === "[1,3,960,960]" : shape === "[1,27,288,512]";
  if (!validShape) throw new Error(`Unexpected static input shape: ${model.id}`);
  if (/^https:/i.test(model.url)) continue;
  const bytes = await readFile(resolve(root, "public", model.url.replace(/^\.?\//, "")));
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== model.sha256.toLowerCase()) throw new Error(`Checksum mismatch: ${model.id}`);
}
console.log(`Verified ${manifest.models.length} model entries.`);
