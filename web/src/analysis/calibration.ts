import type { CourtCorners } from "../domain/types";

export function suggestedCorners(width: number, height: number): CourtCorners {
  return {
    topLeft: { x: width * 0.34, y: height * 0.28 },
    topRight: { x: width * 0.66, y: height * 0.28 },
    bottomLeft: { x: width * 0.08, y: height * 0.92 },
    bottomRight: { x: width * 0.92, y: height * 0.92 },
  };
}
