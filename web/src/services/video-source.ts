import { AnalysisError } from "../domain/errors";
import type { VideoInput } from "../domain/types";
import { ALL_FORMATS, BlobSource, Input, UrlSource } from "mediabunny";

const PLATFORM_HOSTS = /(^|\.)(youtube\.com|youtu\.be|vimeo\.com|twitch\.tv|dailymotion\.com)$/i;
const VIDEO_TYPES = /^(video\/(mp4|webm|quicktime)|application\/octet-stream)(;|$)/i;

export interface VideoMetadata { width: number; height: number; duration: number; fps: number }

export function validateClip(startSeconds: number, endSeconds: number, duration: number): void {
  if (!Number.isFinite(startSeconds) || !Number.isFinite(endSeconds) || startSeconds < 0 || endSeconds <= startSeconds || endSeconds > duration + 0.01) {
    throw new AnalysisError("decode", "INVALID_SOURCE", "The selected time range is invalid.", "Choose a range contained within the video.");
  }
  if (endSeconds - startSeconds > 60.001) {
    throw new AnalysisError("decode", "CLIP_TOO_LONG", "A clip can be at most 60 seconds.", "Shorten the selected range to 60 seconds or less.");
  }
}

export async function validateDirectVideoUrl(raw: string): Promise<string> {
  let url: URL;
  try { url = new URL(raw); } catch { throw new AnalysisError("download", "INVALID_SOURCE", "Enter a valid HTTPS video URL.", "Use a direct .mp4 or .webm HTTPS URL."); }
  if (url.protocol !== "https:") throw new AnalysisError("download", "INVALID_SOURCE", "Only HTTPS video URLs are accepted.", "Use an HTTPS direct link or select a local file.");
  if (PLATFORM_HOSTS.test(url.hostname)) throw new AnalysisError("download", "PLATFORM_URL_UNSUPPORTED", "Streaming platform pages are not direct video files.", "Download a video you are permitted to use, then select the local MP4 or WebM file.");

  let response: Response;
  try { response = await fetch(url, { method: "GET", headers: { Range: "bytes=0-1" } }); }
  catch { throw new AnalysisError("download", "CORS_UNAVAILABLE", "The video host did not allow browser access.", "Use a host with CORS enabled or select the file locally."); }
  if (!(response.ok || response.status === 206)) throw new AnalysisError("download", "INVALID_SOURCE", `The video host returned HTTP ${response.status}.`, "Check the link or select a local file.");
  if (response.type === "opaque") throw new AnalysisError("download", "CORS_UNAVAILABLE", "The video response is opaque to RallyLens.", "Configure Access-Control-Allow-Origin on the host.");
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType && !VIDEO_TYPES.test(contentType)) throw new AnalysisError("download", "INVALID_SOURCE", `Expected MP4 or WebM, received ${contentType}.`, "Use a direct URL whose response Content-Type is video/mp4 or video/webm.");
  if (response.status !== 206 || !response.headers.get("content-range")) throw new AnalysisError("download", "RANGE_UNAVAILABLE", "The server does not support byte-range requests.", "Enable byte ranges on the host or select the file locally.");
  return url.href;
}

export async function readVideoMetadata(input: VideoInput): Promise<VideoMetadata> {
  const source = input.kind === "file" ? new BlobSource(input.file) : new UrlSource(input.url);
  const media = new Input({ formats: ALL_FORMATS, source });
  try {
    if (!(await media.canRead())) throw new Error("Container is not MP4 or WebM");
    const track = await media.getPrimaryVideoTrack();
    if (!track || !(await track.canDecode())) throw new AnalysisError("decode", "CODEC_UNSUPPORTED", "Chrome cannot decode the video's primary track.", "Transcode it to H.264 MP4 or VP9 WebM.");
    const [width, height, duration, frameRate] = await Promise.all([track.getDisplayWidth(), track.getDisplayHeight(), media.computeDuration([track]), track.computeFrameRateMetrics()]);
    if (width > 1920 || height > 1080) throw new AnalysisError("decode", "CODEC_UNSUPPORTED", "4K input is outside the supported profile.", "Use a video at 1920×1080 or lower.");
    return { width, height, duration, fps: frameRate.bestGuessFrameRate };
  } catch (error) {
    if (error instanceof AnalysisError) throw error;
    throw new AnalysisError("decode", "DECODE_FAILED", error instanceof Error ? error.message : "Chrome could not read this video.", "Use a valid H.264 MP4 or VP9 WebM file.");
  } finally { media.dispose(); }
}
