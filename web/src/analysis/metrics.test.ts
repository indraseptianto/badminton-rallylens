import { describe, expect, it } from "vitest";
import { createCourtProjection, recomputeMetrics } from "./metrics";
import type { AnalysisResult } from "../domain/types";

const court = { topLeft: { x: 0, y: 0 }, topRight: { x: 610, y: 0 }, bottomLeft: { x: 0, y: 1340 }, bottomRight: { x: 610, y: 1340 } };
describe("court metrics", () => {
  it("maps image corners to regulation dimensions", () => { const project = createCourtProjection(court); const point = project({ x: 610, y: 1340 }); expect(point.x).toBeCloseTo(6.1); expect(point.y).toBeCloseTo(13.4); });
  it("recalculates labels after an ID correction", () => {
    const result: AnalysisResult = { schemaVersion: "1", createdAt: "2026-01-01T00:00:00Z", models: [], mode: "singles", court, video: { width: 610, height: 1340, fps: 10, durationSeconds: 1, analyzedStartSeconds: 0, analyzedEndSeconds: 1 }, mappings: [{ startFrame: 0, endFrame: 9, players: { P1: 1, P2: 2 }, teams: {} }], players: [{ frame: 0, timestampSeconds: 0, sourceTrackId: 1, confidence: 1, keypoints: [], box: { x: 95, y: 90, width: 10, height: 10 } }, { frame: 1, timestampSeconds: .1, sourceTrackId: 1, confidence: 1, keypoints: [], box: { x: 195, y: 90, width: 10, height: 10 } }], shuttle: [], metrics: { players: [], teams: [] } };
    const original = recomputeMetrics(result); const corrected = recomputeMetrics(result, [{ ...result.mappings[0], players: { P1: 2, P2: 1 } }]); expect(original.players[0].totalDistanceM).toBeCloseTo(1); expect(corrected.players[1].totalDistanceM).toBeCloseTo(1);
  });
});
