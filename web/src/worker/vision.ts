import type { Box, Keypoint, Point } from "../domain/types";

export interface PoseCandidate { box: Box; confidence: number; keypoints: Keypoint[] }
export function intersectionOverUnion(a: Box, b: Box): number {
  const left = Math.max(a.x, b.x), top = Math.max(a.y, b.y), right = Math.min(a.x + a.width, b.x + b.width), bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  return intersection / Math.max(1e-9, a.width * a.height + b.width * b.height - intersection);
}

export function nonMaximumSuppression(candidates: PoseCandidate[], threshold = 0.45): PoseCandidate[] {
  const sorted = [...candidates].sort((a, b) => b.confidence - a.confidence); const kept: PoseCandidate[] = [];
  while (sorted.length) { const best = sorted.shift()!; kept.push(best); for (let i = sorted.length - 1; i >= 0; i--) if (intersectionOverUnion(best.box, sorted[i].box) > threshold) sorted.splice(i, 1); }
  return kept;
}

export function decodeYoloPose(data: Float32Array, dims: readonly number[], sourceWidth: number, sourceHeight: number, inputSize: number, confidenceThreshold = 0.3): PoseCandidate[] {
  const features = dims[1] <= 128 ? dims[1] : dims[2]; const count = dims[1] <= 128 ? dims[2] : dims[1]; const featureAt = (feature: number, index: number) => dims[1] <= 128 ? data[feature * count + index] : data[index * features + feature];
  const scale = Math.min(inputSize / sourceWidth, inputSize / sourceHeight); const padX = (inputSize - sourceWidth * scale) / 2, padY = (inputSize - sourceHeight * scale) / 2; const out: PoseCandidate[] = [];
  for (let i = 0; i < count; i++) {
    const confidence = featureAt(4, i); if (confidence < confidenceThreshold) continue;
    const cx = featureAt(0, i), cy = featureAt(1, i), w = featureAt(2, i), h = featureAt(3, i);
    const keypoints: Keypoint[] = []; for (let k = 5; k + 2 < features; k += 3) keypoints.push({ x: (featureAt(k, i) - padX) / scale, y: (featureAt(k + 1, i) - padY) / scale, confidence: featureAt(k + 2, i) });
    out.push({ box: { x: (cx - w / 2 - padX) / scale, y: (cy - h / 2 - padY) / scale, width: w / scale, height: h / scale }, confidence, keypoints });
  }
  return nonMaximumSuppression(out);
}

interface Track { id: number; box: Box; misses: number; hits: number }
export class ByteTracker {
  private tracks: Track[] = []; private nextId = 1;
  update(detections: PoseCandidate[]): Array<PoseCandidate & { trackId: number }> {
    const pairs: Array<{ ti: number; di: number; score: number }> = []; this.tracks.forEach((t, ti) => detections.forEach((d, di) => pairs.push({ ti, di, score: intersectionOverUnion(t.box, d.box) }))); pairs.sort((a, b) => b.score - a.score);
    const usedTracks = new Set<number>(), usedDetections = new Set<number>(), assigned = new Map<number, number>();
    for (const pair of pairs) if (pair.score >= 0.25 && !usedTracks.has(pair.ti) && !usedDetections.has(pair.di)) { usedTracks.add(pair.ti); usedDetections.add(pair.di); assigned.set(pair.di, pair.ti); }
    this.tracks.forEach((track, i) => { if (!usedTracks.has(i)) track.misses++; });
    detections.forEach((detection, di) => { const ti = assigned.get(di); if (ti === undefined) { this.tracks.push({ id: this.nextId++, box: detection.box, misses: 0, hits: 1 }); assigned.set(di, this.tracks.length - 1); } else { this.tracks[ti].box = detection.box; this.tracks[ti].misses = 0; this.tracks[ti].hits++; } });
    const output = detections.map((detection, di) => ({ ...detection, trackId: this.tracks[assigned.get(di)!].id })); this.tracks = this.tracks.filter((track) => track.misses <= 30); return output;
  }
}

export function heatmapCentroid(data: Float32Array, offset: number, width = 512, height = 288, threshold = 0.5): (Point & { confidence: number }) | undefined {
  let weight = 0, x = 0, y = 0, peak = 0;
  for (let py = 0; py < height; py++) for (let px = 0; px < width; px++) { const value = data[offset + py * width + px]; if (value >= threshold) { weight += value; x += px * value; y += py * value; peak = Math.max(peak, value); } }
  return weight ? { x: x / weight, y: y / weight, confidence: peak } : undefined;
}

export function rgbaToChw(image: ImageData, channels: number, normalize = true): Float32Array {
  const pixels = image.width * image.height; const out = new Float32Array(pixels * channels); const frames = channels / 3;
  for (let f = 0; f < frames; f++) for (let i = 0; i < pixels; i++) for (let c = 0; c < 3; c++) out[f * pixels * 3 + c * pixels + i] = image.data[i * 4 + c] / (normalize ? 255 : 1);
  return out;
}
