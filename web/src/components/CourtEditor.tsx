import { useEffect, useMemo, useRef, useState } from "react";
import type { CourtCorners, Point, VideoInput } from "../domain/types";

const names = ["topLeft", "topRight", "bottomLeft", "bottomRight"] as const;
export function CourtEditor({ input, width, height, timestamp, value, onChange }: { input: VideoInput; width: number; height: number; timestamp: number; value: CourtCorners; onChange: (value: CourtCorners) => void }) {
  const video = useRef<HTMLVideoElement>(null); const [dragging, setDragging] = useState<(typeof names)[number]>();
  const url = useMemo(() => input.kind === "file" ? URL.createObjectURL(input.file) : input.url, [input]);
  useEffect(() => () => { if (input.kind === "file") URL.revokeObjectURL(url); }, [input, url]);
  const point = (event: React.PointerEvent<SVGSVGElement>): Point => { const rect = event.currentTarget.getBoundingClientRect(); return { x: Math.max(0, Math.min(width, (event.clientX - rect.left) / rect.width * width)), y: Math.max(0, Math.min(height, (event.clientY - rect.top) / rect.height * height)) }; };
  const polygon = names.map((name) => `${value[name].x},${value[name].y}`).join(" ");
  return <div className="court-editor">
    <video ref={video} src={url} crossOrigin="anonymous" muted playsInline onLoadedMetadata={(e) => { e.currentTarget.currentTime = timestamp; }} />
    <svg viewBox={`0 0 ${width} ${height}`} onPointerMove={(e) => dragging && onChange({ ...value, [dragging]: point(e) })} onPointerUp={() => setDragging(undefined)} onPointerLeave={() => setDragging(undefined)}>
      <polygon points={polygon} />
      {names.map((name, index) => <g key={name} className="corner" transform={`translate(${value[name].x} ${value[name].y})`} onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setDragging(name); }}><circle r={Math.max(10, width / 80)} /><text y={-18}>{index + 1}</text></g>)}
    </svg>
  </div>;
}
