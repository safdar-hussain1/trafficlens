/** Road-plane homography: maps image pixels to real-world metres so speed can
 * be reported in km/h. Mirrors the projection half of
 * `trafficlens.core.homography`.
 *
 * Policy -- the reason this module exists at all: **an uncalibrated camera
 * reports no speed, ever -- never a pixel-derived guess.** `NO_CALIBRATION`
 * (an alias for `null`) is the sentinel a caller passes wherever a `RoadPlane`
 * is expected but no survey has been done for this camera. `SpeedEstimator`
 * takes a `RoadPlane | null` and returns `null` for every speed when it is
 * handed `NO_CALIBRATION`, rather than falling back to a raw pixel-per-frame
 * estimate dressed up as a speed.
 *
 * Scope, and why it is narrower than the Python module: fitting a homography
 * from surveyed correspondences (`RoadPlane.from_correspondences`) and
 * validating it (`RoadPlane.validate`, `_dlt_condition_number`) are NOT
 * mirrored here. Both go through `cv2.findHomography` and `np.linalg.svd`,
 * whose float64 output no hand-written TypeScript reproduces bit for bit; a
 * browser-side fit would be a second, slightly different calibration wearing
 * the same name, which is exactly the sort of quiet divergence the parity work
 * exists to prevent. Calibration is a survey step that happens once, offline,
 * in the Python tool; the browser receives the fitted 3x3 matrix and projects
 * with it. If a browser-side calibration UI is wanted later, it should be
 * added deliberately, with its own parity measurement. */

import type { Point } from "./geometry";
import { hypot } from "./numeric";

/** An uncalibrated camera reports no speed, ever -- never a pixel-derived
 * guess. Pass this (or plain null -- they are the same value) wherever a
 * RoadPlane is expected but this camera has not been surveyed. */
export const NO_CALIBRATION = null;

export interface ReprojectionError {
  meanM: number;
  maxM: number;
  perPointM: number[];
}

/** A calibrated mapping from image pixels to real-world metres on one road
 * plane, built from an already-computed image -> world homography. */
export class RoadPlane {
  private readonly h: number[][];

  constructor(imageToWorld: readonly (readonly number[])[]) {
    if (imageToWorld.length !== 3) {
      throw new Error(
        `image_to_world must be a 3x3 matrix, got ${imageToWorld.length} rows`,
      );
    }
    this.h = imageToWorld.map((row, i) => {
      if (row.length !== 3) {
        throw new Error(
          `image_to_world row ${i} has ${row.length} entries, expected 3`,
        );
      }
      return [row[0] as number, row[1] as number, row[2] as number];
    });
  }

  /** Map one image pixel to its real-world (metres) position on this road
   * plane. The three products are summed in the same order numpy's matrix-
   * vector product accumulates them. */
  toWorld(p: Point): Point {
    const x = p[0];
    const y = p[1];
    const row0 = this.h[0] as number[];
    const row1 = this.h[1] as number[];
    const row2 = this.h[2] as number[];
    const u = (row0[0] as number) * x + (row0[1] as number) * y + (row0[2] as number) * 1.0;
    const v = (row1[0] as number) * x + (row1[1] as number) * y + (row1[2] as number) * 1.0;
    const w = (row2[0] as number) * x + (row2[1] as number) * y + (row2[2] as number) * 1.0;
    return [u / w, v / w];
  }

  /** Measure this plane's error, in metres, against a set of known
   * image/world correspondences -- deliberately in metres, not pixels, so a
   * person can read the number and judge for themselves whether it is good
   * enough for their deployment. */
  reprojectionError(
    imagePts: readonly Point[],
    worldPts: readonly Point[],
  ): ReprojectionError {
    if (imagePts.length !== worldPts.length) {
      throw new Error(
        `imagePts and worldPts must have the same length, got ` +
          `${imagePts.length} and ${worldPts.length}`,
      );
    }
    const perPointM: number[] = [];
    for (let i = 0; i < imagePts.length; i += 1) {
      const [wx, wy] = this.toWorld(imagePts[i] as Point);
      const worldPt = worldPts[i] as Point;
      perPointM.push(hypot(wx - worldPt[0], wy - worldPt[1]));
    }

    let sum = 0.0;
    for (const value of perPointM) {
      sum += value;
    }
    const meanM = perPointM.length > 0 ? sum / perPointM.length : 0.0;
    const maxM = perPointM.length > 0 ? Math.max(...perPointM) : 0.0;
    return { meanM, maxM, perPointM };
  }
}
