import type { AnalysisResult, CourtCorners, MappingSegment, PlayerDetection, PlayerLabel, PlayerMetrics, Point, ShuttlePoint, TeamLabel, TeamMetrics } from "../domain/types";

const COURT_WIDTH_M = 6.1;
const COURT_LENGTH_M = 13.4;

function solve(matrix: number[][], values: number[]): number[] {
  const a = matrix.map((row, i) => [...row, values[i]]);
  for (let col = 0; col < values.length; col++) {
    let pivot = col;
    for (let row = col + 1; row < a.length; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    if (Math.abs(divisor) < 1e-9) throw new Error("Degenerate court calibration");
    for (let j = col; j <= values.length; j++) a[col][j] /= divisor;
    for (let row = 0; row < a.length; row++) if (row !== col) {
      const factor = a[row][col];
      for (let j = col; j <= values.length; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map((row) => row[values.length]);
}

export function createCourtProjection(corners: CourtCorners): (point: Point) => Point {
  const pairs: Array<[Point, Point]> = [
    [corners.topLeft, { x: 0, y: 0 }], [corners.topRight, { x: COURT_WIDTH_M, y: 0 }],
    [corners.bottomLeft, { x: 0, y: COURT_LENGTH_M }], [corners.bottomRight, { x: COURT_WIDTH_M, y: COURT_LENGTH_M }],
  ];
  const m: number[][] = []; const b: number[] = [];
  for (const [source, target] of pairs) {
    m.push([source.x, source.y, 1, 0, 0, 0, -target.x * source.x, -target.x * source.y]); b.push(target.x);
    m.push([0, 0, 0, source.x, source.y, 1, -target.y * source.x, -target.y * source.y]); b.push(target.y);
  }
  const h = solve(m, b);
  return ({ x, y }) => { const d = h[6] * x + h[7] * y + 1; return { x: (h[0] * x + h[1] * y + h[2]) / d, y: (h[3] * x + h[4] * y + h[5]) / d }; };
}

function labelAt(frame: number, trackId: number, mappings: MappingSegment[]): PlayerLabel | undefined {
  const segment = mappings.find((item) => frame >= item.startFrame && frame <= item.endFrame);
  return (Object.entries(segment?.players ?? {}) as Array<[PlayerLabel, number]>).find(([, id]) => id === trackId)?.[0];
}

function hullArea(points: Point[]): number {
  if (points.length < 3) return 0;
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o: Point, a: Point, b: Point) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const half = (items: Point[]) => { const out: Point[] = []; for (const p of items) { while (out.length >= 2 && cross(out.at(-2)!, out.at(-1)!, p) <= 0) out.pop(); out.push(p); } return out; };
  const hull = [...half(sorted).slice(0, -1), ...half(sorted.reverse()).slice(0, -1)];
  return Math.abs(hull.reduce((sum, p, i) => sum + p.x * hull[(i + 1) % hull.length].y - p.y * hull[(i + 1) % hull.length].x, 0)) / 2;
}

function inferShots(shuttle: ShuttlePoint[], detections: PlayerDetection[], mappings: MappingSegment[], project: (p: Point) => Point): Map<PlayerLabel, number> {
  const shots = new Map<PlayerLabel, number>();
  let lastHit = -Infinity;
  for (let i = 2; i < shuttle.length - 2; i++) {
    const a = shuttle[i - 2], b = shuttle[i], c = shuttle[i + 2];
    if (!a.visible || !b.visible || !c.visible) continue;
    const v1 = { x: b.x - a.x, y: b.y - a.y }; const v2 = { x: c.x - b.x, y: c.y - b.y };
    const turn = v1.x * v2.x + v1.y * v2.y;
    if (turn >= 0 || b.frame - lastHit < 6) continue;
    const candidates = detections.filter((d) => Math.abs(d.frame - b.frame) <= 1).map((d) => ({ detection: d, label: labelAt(b.frame, d.sourceTrackId, mappings) })).filter((v): v is { detection: PlayerDetection; label: PlayerLabel } => !!v.label);
    let best: { label: PlayerLabel; distance: number } | undefined;
    for (const item of candidates) { const foot = project({ x: item.detection.box.x + item.detection.box.width / 2, y: item.detection.box.y + item.detection.box.height }); const ball = project(b); const distance = Math.hypot(foot.x - ball.x, foot.y - ball.y); if (!best || distance < best.distance) best = { label: item.label, distance }; }
    if (best) { shots.set(best.label, (shots.get(best.label) ?? 0) + 1); lastHit = b.frame; }
  }
  return shots;
}

export function recomputeMetrics(result: AnalysisResult, mappings = result.mappings): AnalysisResult["metrics"] {
  const project = createCourtProjection(result.court);
  const labels: PlayerLabel[] = result.mode === "singles" ? ["P1", "P2"] : ["P1", "P2", "P3", "P4"];
  const shots = inferShots(result.shuttle, result.players, mappings, project);
  const players: PlayerMetrics[] = labels.map((player) => {
    const points = result.players.filter((d) => labelAt(d.frame, d.sourceTrackId, mappings) === player).map((d) => ({ frame: d.frame, point: project({ x: d.box.x + d.box.width / 2, y: d.box.y + d.box.height }) })).sort((a, b) => a.frame - b.frame);
    let distance = 0; let maxSpeed = 0;
    for (let i = 1; i < points.length; i++) { const frameDelta = points[i].frame - points[i - 1].frame; if (frameDelta <= 0 || frameDelta > result.video.fps) continue; const step = Math.hypot(points[i].point.x - points[i - 1].point.x, points[i].point.y - points[i - 1].point.y); if (step < 2.5) { distance += step; maxSpeed = Math.max(maxSpeed, step / (frameDelta / result.video.fps)); } }
    const duration = Math.max(0.001, result.video.analyzedEndSeconds - result.video.analyzedStartSeconds);
    return { player, detectionFrameCount: points.length, totalDistanceM: distance, averageSpeedMps: distance / duration, maxSpeedMps: maxSpeed, courtCoverageM2: hullArea(points.map((v) => v.point)), shotCount: shots.get(player) ?? 0 };
  });
  const lastSegment = mappings.at(-1); const teams: TeamMetrics[] = result.mode === "doubles" ? (["A", "B"] as TeamLabel[]).map((team) => {
    const members = labels.filter((p) => lastSegment?.teams[p] === team); const memberMetrics = players.filter((p) => members.includes(p.player)); const totalDistanceM = memberMetrics.reduce((n, p) => n + p.totalDistanceM, 0); const shotCount = memberMetrics.reduce((n, p) => n + p.shotCount, 0);
    return { team, members, totalDistanceM, shotCount, combinedCoverageM2: Math.min(COURT_WIDTH_M * COURT_LENGTH_M, memberMetrics.reduce((n, p) => n + p.courtCoverageM2, 0)), movementShare: Object.fromEntries(memberMetrics.map((p) => [p.player, totalDistanceM ? p.totalDistanceM / totalDistanceM : 0])), shotShare: Object.fromEntries(memberMetrics.map((p) => [p.player, shotCount ? p.shotCount / shotCount : 0])) };
  }) : [];
  return { players, teams };
}
