import { recomputeMetrics } from "../analysis/metrics";
import type { AnalysisResult, MappingSegment, PlayerLabel, TeamLabel } from "../domain/types";

export function IdentityEditor({ result, mappings, onChange }: { result: AnalysisResult; mappings: MappingSegment[]; onChange: (mappings: MappingSegment[], result: AnalysisResult) => void }) {
  const labels: PlayerLabel[] = result.mode === "singles" ? ["P1", "P2"] : ["P1", "P2", "P3", "P4"];
  const trackIds = [...new Set(result.players.map((p) => p.sourceTrackId))];
  const update = (index: number, next: MappingSegment) => { const updated = mappings.map((item, i) => i === index ? next : item); onChange(updated, { ...result, mappings: updated, metrics: recomputeMetrics(result, updated) }); };
  const split = () => { const last = mappings.at(-1)!; if (last.endFrame - last.startFrame < 2) return; const cut = Math.floor((last.startFrame + last.endFrame) / 2); const updated = [...mappings.slice(0, -1), { ...last, endFrame: cut }, { ...last, startFrame: cut + 1 }]; onChange(updated, { ...result, mappings: updated, metrics: recomputeMetrics(result, updated) }); };
  return <div className="stack"><div className="section-heading"><div><span className="eyebrow">Identity review</span><h2>Correct track switches before metrics are final.</h2></div><button className="secondary" onClick={split}>Split last segment</button></div>
    {mappings.map((segment, index) => <div className="segment" key={`${segment.startFrame}-${index}`}><div className="segment-title">Frames {segment.startFrame}–{segment.endFrame}</div><div className="mapping-grid">{labels.map((label) => <label key={label}><span>{label}</span><select value={segment.players[label] ?? ""} onChange={(e) => update(index, { ...segment, players: { ...segment.players, [label]: Number(e.target.value) } })}><option value="">Unassigned</option>{trackIds.map((id) => <option key={id} value={id}>Track {id}</option>)}</select>{result.mode === "doubles" && <select aria-label={`${label} team`} value={segment.teams[label] ?? ""} onChange={(e) => update(index, { ...segment, teams: { ...segment.teams, [label]: e.target.value as TeamLabel } })}><option value="A">Team A</option><option value="B">Team B</option></select>}</label>)}</div></div>)}
  </div>;
}
