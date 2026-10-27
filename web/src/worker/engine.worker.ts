/// <reference lib="webworker" />
import { ALL_FORMATS, BlobSource, Input, UrlSource, VideoSampleSink } from "mediabunny";
import * as ort from "onnxruntime-web/webgpu";
import { recomputeMetrics, createCourtProjection } from "../analysis/metrics";
import { AnalysisError, toAnalysisError } from "../domain/errors";
import type { AnalysisConfig, AnalysisProgress, AnalysisResult, CourtCorners, InferenceProfileId, MappingSegment, ModelManifest, ModelManifestEntry, PlayerDetection, PlayerLabel, ShuttlePoint, VideoInput } from "../domain/types";
import { RESULT_SCHEMA_VERSION } from "../domain/types";
import { getVerifiedModel, pruneModelCache } from "./model-cache";
import type { WorkerCommand, WorkerEvent, WorkerPayload } from "./protocol";
import { ByteTracker, decodeYoloPose, heatmapCentroid } from "./vision";

const scope = self as DedicatedWorkerGlobalScope;
let manifest: ModelManifest | undefined;
let cancelled = false;
let selectedProfile: InferenceProfileId | undefined;

function post(event: WorkerEvent): void { scope.postMessage(event); }
function success(id: string, payload: WorkerPayload): void { post({ id, type: "success", payload }); }
function progress(id: string, value: AnalysisProgress): void { post({ id, type: "progress", progress: value }); }
function ensureActive(): void { if (cancelled) throw new AnalysisError("decode", "CANCELLED", "Analysis was cancelled.", "Choose Analyze when you are ready to start again."); }

function validateManifest(value: unknown): ModelManifest {
  if (!value || typeof value !== "object" || (value as ModelManifest).schemaVersion !== "1" || !Array.isArray((value as ModelManifest).models)) throw new AnalysisError("download", "MODEL_MANIFEST_INVALID", "The model manifest is invalid.", "Use model assets from a verified RallyLens release.");
  const parsed = value as ModelManifest;
  for (const model of parsed.models) if (!model.id || !model.version || !model.url || !model.input?.name || !model.outputNames?.length) throw new AnalysisError("download", "MODEL_MANIFEST_INVALID", "A model manifest entry is incomplete.", "Regenerate the release model manifest.");
  return parsed;
}

async function initialize(id: string, url: string): Promise<void> {
  if (!("gpu" in navigator) || !(await navigator.gpu.requestAdapter())) throw new AnalysisError("capability", "WEBGPU_UNAVAILABLE", "No WebGPU adapter is available to the analysis worker.", "Enable Chrome hardware acceleration, restart Chrome, and retry.");
  const response = await fetch(url, { cache: "no-cache" }); if (!response.ok) throw new AnalysisError("download", "MODEL_MANIFEST_INVALID", `Model manifest returned HTTP ${response.status}.`, "Deploy public/models/manifest.json with the web app.");
  manifest = validateManifest(await response.json()); manifest = { ...manifest, models: manifest.models.map((model) => ({ ...model, url: new URL(model.url, response.url).href })) }; await pruneModelCache(manifest.models); success(id, { kind: "initialized", manifest });
}

async function makeSession(entry: ModelManifestEntry, id: string): Promise<ort.InferenceSession> {
  const bytes = await getVerifiedModel(entry, (loaded, total) => progress(id, { stage: "download", completedFrames: loaded, totalFrames: total ?? loaded, message: `Downloading ${entry.id}` }));
  return ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
}

async function benchmark(id: string): Promise<void> {
  if (!manifest) throw new AnalysisError("capability", "MODEL_MANIFEST_INVALID", "The worker is not initialized.", "Reload RallyLens and retry.");
  const candidates = manifest.models.filter((m) => m.task === "player" && m.profile).sort((a, b) => (b.input.shape.at(-1) ?? 0) - (a.input.shape.at(-1) ?? 0));
  if (!candidates.length) throw new AnalysisError("download", "MODEL_UNAVAILABLE", "This release contains no verified player model.", "Install a release whose model manifest passed golden-set and license checks.");
  const shuttle = manifest.models.find((m) => m.task === "shuttle"); if (!shuttle) throw new AnalysisError("download", "MODEL_UNAVAILABLE", "This release contains no verified shuttle model.", "Install a complete model release.");
  const shuttleSession = await makeSession(shuttle, id); const shuttleTensor = new ort.Tensor("float32", new Float32Array(shuttle.input.shape.reduce((a, b) => a * b, 1)), shuttle.input.shape); const shuttleStart = performance.now(); await shuttleSession.run({ [shuttle.input.name]: shuttleTensor }); const shuttleMsPerFrame = (performance.now() - shuttleStart) / 8; await shuttleSession.release();
  for (const entry of candidates) {
    ensureActive(); const session = await makeSession(entry, id); const size = entry.input.shape.reduce((a, b) => a * b, 1); const tensor = new ort.Tensor("float32", new Float32Array(size), entry.input.shape); const start = performance.now();
    await session.run({ [entry.input.name]: tensor }); const elapsed = performance.now() - start + shuttleMsPerFrame; await session.release(); const framesPerSecond = 1000 / Math.max(elapsed, 1); const realtimeRatioAt60Fps = framesPerSecond / 60;
    if (realtimeRatioAt60Fps >= 1 / 3) { selectedProfile = entry.profile; success(id, { kind: "benchmark", profile: entry.profile!, estimatedSpeedRatio: realtimeRatioAt60Fps }); return; }
  }
  throw new AnalysisError("player", "BENCHMARK_TOO_SLOW", "This GPU is not predicted to finish a 60-second clip within 180 seconds.", "Try a shorter clip or a desktop with a faster WebGPU adapter.");
}

function mediaInput(input: VideoInput): Input {
  return new Input({ formats: ALL_FORMATS, source: input.kind === "file" ? new BlobSource(input.file) : new UrlSource(input.url) });
}

function canvasPixels(sample: Awaited<ReturnType<VideoSampleSink["getSample"]>>, width: number, height: number, letterbox = false): ImageData {
  if (!sample) throw new Error("Missing decoded sample"); const canvas = new OffscreenCanvas(width, height); const context = canvas.getContext("2d", { willReadFrequently: true }); if (!context) throw new Error("2D canvas unavailable"); context.fillStyle = "#000"; context.fillRect(0, 0, width, height);
  if (letterbox) { const scale = Math.min(width / sample.displayWidth, height / sample.displayHeight); const w = sample.displayWidth * scale, h = sample.displayHeight * scale; sample.draw(context, (width - w) / 2, (height - h) / 2, w, h); }
  else sample.draw(context, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
}

function imageToRgbTensor(image: ImageData): Float32Array {
  const pixels = image.width * image.height; const out = new Float32Array(pixels * 3); for (let i = 0; i < pixels; i++) { out[i] = image.data[i * 4] / 255; out[pixels + i] = image.data[i * 4 + 1] / 255; out[pixels * 2 + i] = image.data[i * 4 + 2] / 255; } return out;
}

async function analyzePlayers(id: string, config: AnalysisConfig, entry: ModelManifestEntry): Promise<PlayerDetection[]> {
  const media = mediaInput(config.input); const session = await makeSession(entry, id); const output: PlayerDetection[] = []; const tracker = new ByteTracker();
  try { const track = await media.getPrimaryVideoTrack(); if (!track) throw new Error("No video track"); const sink = new VideoSampleSink(track); const size = entry.input.shape.at(-1)!; const total = Math.ceil((config.clip.endSeconds - config.clip.startSeconds) * 30); let frame = 0;
    for await (const sample of sink.samples(config.clip.startSeconds, config.clip.endSeconds)) { ensureActive(); const pixels = canvasPixels(sample, size, size, true); const tensor = new ort.Tensor("float32", imageToRgbTensor(pixels), entry.input.shape); const values = await session.run({ [entry.input.name]: tensor }); const first = values[entry.outputNames[0]]; const candidates = decodeYoloPose(first.data as Float32Array, first.dims, sample.displayWidth, sample.displayHeight, size); const tracked = tracker.update(candidates); for (const item of tracked) output.push({ frame, timestampSeconds: sample.timestamp, sourceTrackId: item.trackId, box: item.box, confidence: item.confidence, keypoints: item.keypoints }); frame++; sample.close(); if (frame % 5 === 0) progress(id, { stage: "player", completedFrames: frame, totalFrames: total, estimatedSecondsRemaining: Math.max(0, (total - frame) / 3), message: "Tracking players" }); }
  } finally { await session.release(); media.dispose(); }
  return output;
}

async function analyzeShuttle(id: string, config: AnalysisConfig, entry: ModelManifestEntry, sourceWidth: number, sourceHeight: number): Promise<ShuttlePoint[]> {
  const media = mediaInput(config.input); const session = await makeSession(entry, id); const output: ShuttlePoint[] = []; const inputFrames = entry.input.shape[1] / 3; const outputFrames = entry.input.shape[1] === 27 ? 8 : inputFrames; const queue: Array<{ image: ImageData; timestamp: number; frame: number }> = [];
  try { const track = await media.getPrimaryVideoTrack(); if (!track) throw new Error("No video track"); const sink = new VideoSampleSink(track); let frame = 0; const total = Math.ceil((config.clip.endSeconds - config.clip.startSeconds) * 30);
    for await (const sample of sink.samples(config.clip.startSeconds, config.clip.endSeconds)) { ensureActive(); queue.push({ image: canvasPixels(sample, 512, 288), timestamp: sample.timestamp, frame }); sample.close(); frame++; if (queue.length < inputFrames) continue; const pixels = 512 * 288; const inputData = new Float32Array(inputFrames * 3 * pixels); queue.forEach((item, f) => inputData.set(imageToRgbTensor(item.image), f * 3 * pixels)); const result = await session.run({ [entry.input.name]: new ort.Tensor("float32", inputData, entry.input.shape) }); const heatmaps = result[entry.outputNames[0]].data as Float32Array;
      for (let j = 0; j < outputFrames; j++) { const point = heatmapCentroid(heatmaps, j * pixels); const meta = queue[j + (inputFrames - outputFrames)]; output.push({ frame: meta.frame, timestampSeconds: meta.timestamp, x: point ? point.x / 512 * sourceWidth : 0, y: point ? point.y / 288 * sourceHeight : 0, confidence: point?.confidence ?? 0, visible: !!point }); }
      queue.splice(0, outputFrames); progress(id, { stage: "shuttle", completedFrames: frame, totalFrames: total, estimatedSecondsRemaining: Math.max(0, (total - frame) / 3), message: "Tracking shuttle" }); }
  } finally { await session.release(); media.dispose(); }
  return output;
}

function buildMapping(players: PlayerDetection[], corners: CourtCorners, mode: AnalysisConfig["mode"], endFrame: number): MappingSegment[] {
  const project = createCourtProjection(corners); const stats = new Map<number, { count: number; inside: number; points: Array<{ x: number; y: number }> }>();
  for (const d of players) { const point = project({ x: d.box.x + d.box.width / 2, y: d.box.y + d.box.height }); const stat = stats.get(d.sourceTrackId) ?? { count: 0, inside: 0, points: [] }; stat.count++; if (point.x >= -0.5 && point.x <= 6.6 && point.y >= -0.5 && point.y <= 13.9) stat.inside++; stat.points.push(point); stats.set(d.sourceTrackId, stat); }
  const count = mode === "singles" ? 2 : 4; const selected = [...stats].sort((a, b) => (b[1].inside / b[1].count) * b[1].count - (a[1].inside / a[1].count) * a[1].count).slice(0, count).map(([id, stat]) => ({ id, x: stat.points.reduce((n, p) => n + p.x, 0) / stat.points.length, y: stat.points.reduce((n, p) => n + p.y, 0) / stat.points.length })).sort((a, b) => a.y - b.y || a.x - b.x);
  const labels = (mode === "singles" ? ["P1", "P2"] : ["P1", "P2", "P3", "P4"]) as PlayerLabel[]; const mapping: MappingSegment = { startFrame: 0, endFrame, players: {}, teams: {} }; selected.forEach((track, index) => { const label = labels[index]; mapping.players[label] = track.id; if (mode === "doubles") mapping.teams[label] = index < 2 ? "A" : "B"; }); return [mapping];
}

async function analyze(id: string, config: AnalysisConfig, corners: CourtCorners): Promise<void> {
  if (!manifest) throw new AnalysisError("capability", "MODEL_MANIFEST_INVALID", "The worker is not initialized.", "Reload the page."); cancelled = false;
  const profile = config.profile === "auto" ? selectedProfile : config.profile; if (!profile) throw new AnalysisError("player", "BENCHMARK_TOO_SLOW", "No inference profile was selected.", "Run the startup benchmark again.");
  const playerModel = manifest.models.find((m) => m.task === "player" && m.profile === profile); const shuttleModel = manifest.models.find((m) => m.task === "shuttle"); if (!playerModel || !shuttleModel) throw new AnalysisError("download", "MODEL_UNAVAILABLE", "The selected release is missing a required verified model.", "Use a complete RallyLens model release.");
  const media = mediaInput(config.input); const track = await media.getPrimaryVideoTrack(); if (!track) throw new AnalysisError("decode", "DECODE_FAILED", "No video track was found.", "Select an MP4 or WebM video."); const [width, height, fps] = await Promise.all([track.getDisplayWidth(), track.getDisplayHeight(), track.computeFrameRateMetrics()]); media.dispose();
  const players = await analyzePlayers(id, config, playerModel); const shuttle = await analyzeShuttle(id, config, shuttleModel, width, height); const frameCount = Math.ceil((config.clip.endSeconds - config.clip.startSeconds) * fps.bestGuessFrameRate); const mappings = buildMapping(players, corners, config.mode, frameCount - 1);
  let result: AnalysisResult = { schemaVersion: RESULT_SCHEMA_VERSION, createdAt: new Date().toISOString(), models: [playerModel, shuttleModel].map((m) => ({ id: m.id, version: m.version, sha256: m.sha256 })), video: { width, height, fps: fps.bestGuessFrameRate, durationSeconds: config.clip.endSeconds - config.clip.startSeconds, analyzedStartSeconds: config.clip.startSeconds, analyzedEndSeconds: config.clip.endSeconds }, mode: config.mode, court: corners, mappings, players, shuttle, metrics: { players: [], teams: [] } };
  progress(id, { stage: "metrics", completedFrames: frameCount, totalFrames: frameCount, message: "Computing court metrics" }); result = { ...result, metrics: recomputeMetrics(result) }; success(id, { kind: "result", result });
}

scope.onmessage = async ({ data }: MessageEvent<WorkerCommand>) => {
  const { id } = data;
  try {
    if (data.command === "cancel") { cancelled = true; success(id, { kind: "cancelled" }); return; }
    if (data.command === "dispose") { manifest = undefined; selectedProfile = undefined; success(id, { kind: "disposed" }); scope.close(); return; }
    if (data.command === "initialize") await initialize(id, data.manifestUrl);
    else if (data.command === "benchmark") await benchmark(id);
    else if (data.command === "analyze") await analyze(id, data.config, data.corners);
  } catch (error) { post({ id, type: "error", error: toAnalysisError(error, data.command === "analyze" ? "player" : "capability").serialize() }); }
};
