import type { AnalysisConfig, AnalysisProgress, AnalysisResult, CourtCorners, MappingSegment, MatchMode, VideoInput } from "../domain/types";
import type { SerializedAnalysisError } from "../domain/errors";

export type Step = "capability" | "source" | "mode" | "clip" | "calibration" | "analysis" | "identity" | "results";
export interface AppState {
  step: Step;
  capability: { checked: boolean; supported: boolean; reasons: string[] };
  input?: VideoInput;
  sourceUrl: string;
  video?: { width: number; height: number; duration: number; fps: number };
  mode: MatchMode;
  clip: { startSeconds: number; endSeconds: number };
  corners?: CourtCorners;
  config?: AnalysisConfig;
  progress?: AnalysisProgress;
  result?: AnalysisResult;
  mappings: MappingSegment[];
  error?: SerializedAnalysisError;
}

export const initialState: AppState = {
  step: "capability",
  capability: { checked: false, supported: false, reasons: [] },
  sourceUrl: "",
  mode: "singles",
  clip: { startSeconds: 0, endSeconds: 60 },
  mappings: [],
};

export type Action =
  | { type: "CAPABILITY"; supported: boolean; reasons: string[] }
  | { type: "SET_URL"; value: string }
  | { type: "SOURCE"; input: VideoInput; video: AppState["video"] }
  | { type: "MODE"; mode: MatchMode }
  | { type: "EDIT_CLIP"; startSeconds: number; endSeconds: number }
  | { type: "CLIP"; startSeconds: number; endSeconds: number }
  | { type: "CORNERS"; corners: CourtCorners }
  | { type: "START"; config: AnalysisConfig }
  | { type: "PROGRESS"; progress: AnalysisProgress }
  | { type: "ANALYZED"; result: AnalysisResult }
  | { type: "MAPPINGS"; mappings: MappingSegment[]; result: AnalysisResult }
  | { type: "FINISH" }
  | { type: "ERROR"; error: SerializedAnalysisError }
  | { type: "BACK" }
  | { type: "RESET" };

const order: Step[] = ["capability", "source", "mode", "clip", "calibration", "analysis", "identity", "results"];

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "CAPABILITY": return { ...state, step: action.supported ? "source" : "capability", capability: { checked: true, supported: action.supported, reasons: action.reasons } };
    case "SET_URL": return { ...state, sourceUrl: action.value, error: undefined };
    case "SOURCE": return { ...state, input: action.input, video: action.video, clip: { startSeconds: 0, endSeconds: Math.min(60, action.video?.duration ?? 60) }, step: "mode", error: undefined };
    case "MODE": return { ...state, mode: action.mode, step: "clip" };
    case "EDIT_CLIP": return { ...state, clip: { startSeconds: action.startSeconds, endSeconds: action.endSeconds }, error: undefined };
    case "CLIP": return { ...state, clip: { startSeconds: action.startSeconds, endSeconds: action.endSeconds }, step: "calibration" };
    case "CORNERS": return { ...state, corners: action.corners, error: undefined };
    case "START": return { ...state, config: action.config, step: "analysis", error: undefined };
    case "PROGRESS": return { ...state, progress: action.progress };
    case "ANALYZED": return { ...state, result: action.result, mappings: action.result.mappings, step: "identity", progress: undefined };
    case "MAPPINGS": return { ...state, mappings: action.mappings, result: action.result };
    case "FINISH": return { ...state, step: "results" };
    case "ERROR": return { ...state, error: action.error, progress: undefined, step: state.step === "analysis" ? "calibration" : state.step };
    case "BACK": return { ...state, step: order[Math.max(1, order.indexOf(state.step) - 1)], error: undefined };
    case "RESET": return { ...initialState, capability: state.capability, step: state.capability.supported ? "source" : "capability" };
  }
}
