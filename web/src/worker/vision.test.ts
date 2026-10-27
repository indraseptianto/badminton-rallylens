import { describe, expect, it } from "vitest";
import { ByteTracker, heatmapCentroid, intersectionOverUnion, nonMaximumSuppression } from "./vision";

const pose = (x: number, confidence = 0.9) => ({ box: { x, y: 0, width: 10, height: 20 }, confidence, keypoints: [] });
describe("browser vision post-processing", () => {
  it("computes IoU and suppresses overlapping boxes", () => {
    expect(intersectionOverUnion(pose(0).box, pose(5).box)).toBeCloseTo(1 / 3);
    expect(nonMaximumSuppression([pose(0), pose(1, 0.8), pose(40)]).length).toBe(2);
  });
  it("keeps an identity across nearby detections", () => {
    const tracker = new ByteTracker(); const first = tracker.update([pose(0)])[0]; const second = tracker.update([pose(2)])[0]; expect(second.trackId).toBe(first.trackId);
  });
  it("recovers a weighted shuttle centroid", () => {
    const map = new Float32Array(12); map[5] = 1; map[6] = 1; expect(heatmapCentroid(map, 0, 4, 3)).toMatchObject({ x: 1.5, y: 1 });
  });
});
