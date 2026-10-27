import { AnalysisError } from "../domain/errors";
import type { AnalysisResult } from "../domain/types";

export type LanguageModelStatus = "unavailable" | "downloadable" | "downloading" | "available";
export async function getLanguageModelStatus(): Promise<LanguageModelStatus> { return typeof LanguageModel === "undefined" ? "unavailable" : LanguageModel.availability(); }

export async function generateReport(result: AnalysisResult, onDownloadProgress?: (fraction: number) => void): Promise<string> {
  if (typeof LanguageModel === "undefined") throw new AnalysisError("report", "REPORT_UNAVAILABLE", "Chrome's built-in LanguageModel API is unavailable.", "The analysis and downloads remain available. Install the on-device model in a supported Chrome build to enable the report.");
  const status = await LanguageModel.availability(); if (status === "unavailable") throw new AnalysisError("report", "REPORT_UNAVAILABLE", "Chrome's on-device language model is unavailable.", "The rest of the result is unaffected.");
  const session = await LanguageModel.create({ initialPrompts: [{ role: "system", content: "You are a badminton performance analyst. Write a concise, evidence-based English report. Use only supplied metrics, distinguish observations from interpretation, and never invent events." }], monitor(monitor) { monitor.addEventListener("downloadprogress", (event) => onDownloadProgress?.(event.loaded)); } });
  try {
    const safeMetrics = { mode: result.mode, durationSeconds: result.video.durationSeconds, players: result.metrics.players, teams: result.metrics.teams };
    return await session.prompt(`Analyze these corrected RallyLens metrics. Include: summary, player comparison, movement and coverage, shot distribution, and two cautious coaching observations. Metrics JSON:\n${JSON.stringify(safeMetrics)}`);
  } finally { session.destroy(); }
}
