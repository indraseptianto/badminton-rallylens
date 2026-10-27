import type { AnalysisConfig, AnalysisProgress, AnalysisResult, CourtCorners, InferenceProfileId, ModelManifest } from "../domain/types";
import type { SerializedAnalysisError } from "../domain/errors";

export type WorkerCommand =
  | { id: string; command: "initialize"; manifestUrl: string }
  | { id: string; command: "benchmark" }
  | { id: string; command: "analyze"; config: AnalysisConfig; corners: CourtCorners }
  | { id: string; command: "cancel" }
  | { id: string; command: "dispose" };

export type WorkerPayload =
  | { kind: "initialized"; manifest: ModelManifest }
  | { kind: "benchmark"; profile: InferenceProfileId; estimatedSpeedRatio: number }
  | { kind: "result"; result: AnalysisResult }
  | { kind: "cancelled" }
  | { kind: "disposed" };

export type WorkerEvent =
  | { id: string; type: "progress"; progress: AnalysisProgress }
  | { id: string; type: "success"; payload: WorkerPayload }
  | { id: string; type: "error"; error: SerializedAnalysisError };
