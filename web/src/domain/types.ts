export const RESULT_SCHEMA_VERSION = "1" as const;
export const MAX_CLIP_SECONDS = 60;

export type MatchMode = "singles" | "doubles";
export type InferenceProfileId = "yolo11n-pose-640" | "yolo11s-pose-960";
export type AnalysisStage =
  | "capability" | "download" | "decode" | "player" | "shuttle"
  | "calibration" | "metrics" | "render" | "report";

export interface LocalInput { kind: "file"; file: File }
export interface UrlInput { kind: "url"; url: string }
export type VideoInput = LocalInput | UrlInput;

export interface AnalysisConfig {
  input: VideoInput;
  clip: { startSeconds: number; endSeconds: number };
  mode: MatchMode;
  profile: "auto" | InferenceProfileId;
}

export interface AnalysisProgress {
  stage: AnalysisStage;
  completedFrames: number;
  totalFrames: number;
  estimatedSecondsRemaining?: number;
  message: string;
}

export interface Point { x: number; y: number }
export interface CourtCorners { topLeft: Point; topRight: Point; bottomLeft: Point; bottomRight: Point }
export interface Box { x: number; y: number; width: number; height: number }
export interface Keypoint extends Point { confidence: number }
export interface PlayerDetection {
  frame: number;
  timestampSeconds: number;
  sourceTrackId: number;
  box: Box;
  confidence: number;
  keypoints: Keypoint[];
}
export interface ShuttlePoint extends Point {
  frame: number;
  timestampSeconds: number;
  confidence: number;
  visible: boolean;
}

export type PlayerLabel = "P1" | "P2" | "P3" | "P4";
export type TeamLabel = "A" | "B";
export interface MappingSegment {
  startFrame: number;
  endFrame: number;
  players: Partial<Record<PlayerLabel, number>>;
  teams: Partial<Record<PlayerLabel, TeamLabel>>;
}

export interface PlayerMetrics {
  player: PlayerLabel;
  detectionFrameCount: number;
  totalDistanceM: number;
  averageSpeedMps: number;
  maxSpeedMps: number;
  courtCoverageM2: number;
  shotCount: number;
}
export interface TeamMetrics {
  team: TeamLabel;
  members: PlayerLabel[];
  totalDistanceM: number;
  shotCount: number;
  combinedCoverageM2: number;
  movementShare: Partial<Record<PlayerLabel, number>>;
  shotShare: Partial<Record<PlayerLabel, number>>;
}

export interface AnalysisResult {
  schemaVersion: typeof RESULT_SCHEMA_VERSION;
  createdAt: string;
  models: Array<{ id: string; version: string; sha256: string }>;
  video: { width: number; height: number; fps: number; durationSeconds: number; analyzedStartSeconds: number; analyzedEndSeconds: number };
  mode: MatchMode;
  court: CourtCorners;
  mappings: MappingSegment[];
  players: PlayerDetection[];
  shuttle: ShuttlePoint[];
  metrics: { players: PlayerMetrics[]; teams: TeamMetrics[] };
  report?: string;
}

export interface ModelManifestEntry {
  id: string;
  task: "player" | "shuttle";
  profile?: InferenceProfileId;
  version: string;
  url: string;
  sha256: string;
  input: { name: string; shape: number[] };
  outputNames: string[];
  license: { name: string; source: string; redistributionVerified: boolean };
}
export interface ModelManifest { schemaVersion: "1"; models: ModelManifestEntry[] }
