import { toAnalysisError } from "../domain/errors";
import type { AnalysisConfig, AnalysisProgress, AnalysisResult, CourtCorners, InferenceProfileId } from "../domain/types";
import type { WorkerCommand, WorkerEvent, WorkerPayload } from "../worker/protocol";

export class AnalysisWorkerClient {
  private readonly worker = new Worker(new URL("../worker/engine.worker.ts", import.meta.url), { type: "module" });
  private pending = new Map<string, { resolve: (value: WorkerPayload) => void; reject: (reason: unknown) => void; progress?: (value: AnalysisProgress) => void }>();
  constructor() { this.worker.onmessage = ({ data }: MessageEvent<WorkerEvent>) => { const request = this.pending.get(data.id); if (!request) return; if (data.type === "progress") request.progress?.(data.progress); else { this.pending.delete(data.id); if (data.type === "success") request.resolve(data.payload); else request.reject(toAnalysisError(data.error)); } }; this.worker.onerror = (event) => { for (const request of this.pending.values()) request.reject(event.error ?? new Error(event.message)); this.pending.clear(); }; }
  private send(command: CommandWithoutId, onProgress?: (value: AnalysisProgress) => void): Promise<WorkerPayload> { const id = crypto.randomUUID(); return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject, progress: onProgress }); this.worker.postMessage({ ...command, id } as WorkerCommand); }); }
  async initialize(manifestUrl: string): Promise<void> { await this.send({ command: "initialize", manifestUrl }); }
  async benchmark(): Promise<{ profile: InferenceProfileId; estimatedSpeedRatio: number }> { const value = await this.send({ command: "benchmark" }); if (value.kind !== "benchmark") throw new Error("Unexpected worker response"); return value; }
  async analyze(config: AnalysisConfig, corners: CourtCorners, onProgress: (value: AnalysisProgress) => void): Promise<AnalysisResult> { const value = await this.send({ command: "analyze", config, corners }, onProgress); if (value.kind !== "result") throw new Error("Unexpected worker response"); return value.result; }
  cancel(): void { void this.send({ command: "cancel" }); }
  dispose(): void { void this.send({ command: "dispose" }).finally(() => this.worker.terminate()); }
}

type CommandWithoutId = WorkerCommand extends infer Command ? Command extends WorkerCommand ? Omit<Command, "id"> : never : never;
