import { describe, expect, it } from "vitest";
import { initialState, reducer } from "./machine";

describe("analysis state machine", () => {
  it("does not advance when required capabilities are missing", () => {
    const next = reducer(initialState, { type: "CAPABILITY", supported: false, reasons: ["No WebGPU"] });
    expect(next.step).toBe("capability"); expect(next.capability.reasons).toEqual(["No WebGPU"]);
  });

  it("sets a bounded initial clip after source inspection", () => {
    const file = new File(["video"], "match.mp4", { type: "video/mp4" });
    const next = reducer({ ...initialState, step: "source" }, { type: "SOURCE", input: { kind: "file", file }, video: { width: 1920, height: 1080, duration: 91, fps: 60 } });
    expect(next.step).toBe("mode"); expect(next.clip).toEqual({ startSeconds: 0, endSeconds: 60 });
  });
});
