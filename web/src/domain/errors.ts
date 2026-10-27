import type { AnalysisStage } from "./types";

export type AnalysisErrorCode =
  | "UNSUPPORTED_BROWSER" | "WEBGPU_UNAVAILABLE" | "INVALID_SOURCE"
  | "PLATFORM_URL_UNSUPPORTED" | "CORS_UNAVAILABLE" | "RANGE_UNAVAILABLE"
  | "CODEC_UNSUPPORTED" | "CLIP_TOO_LONG" | "MODEL_MANIFEST_INVALID"
  | "MODEL_UNAVAILABLE" | "MODEL_INTEGRITY_FAILED" | "BENCHMARK_TOO_SLOW"
  | "DECODE_FAILED" | "INFERENCE_FAILED" | "CANCELLED" | "REPORT_UNAVAILABLE"
  | "RENDER_FAILED" | "UNKNOWN";

export interface SerializedAnalysisError {
  name: "AnalysisError";
  stage: AnalysisStage;
  code: AnalysisErrorCode;
  message: string;
  recovery: string;
}

export class AnalysisError extends Error {
  readonly name = "AnalysisError";
  constructor(
    readonly stage: AnalysisStage,
    readonly code: AnalysisErrorCode,
    message: string,
    readonly recovery: string,
  ) { super(message); }

  serialize(): SerializedAnalysisError {
    return { name: this.name, stage: this.stage, code: this.code, message: this.message, recovery: this.recovery };
  }
}

export function toAnalysisError(value: unknown, stage: AnalysisStage = "capability"): AnalysisError {
  if (value instanceof AnalysisError) return value;
  if (typeof value === "object" && value !== null && "name" in value && value.name === "AnalysisError") {
    const error = value as SerializedAnalysisError;
    return new AnalysisError(error.stage, error.code, error.message, error.recovery);
  }
  return new AnalysisError(stage, "UNKNOWN", value instanceof Error ? value.message : String(value), "Retry the operation. If it fails again, download the anonymous diagnostics file.");
}
