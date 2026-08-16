/** The three things the control room can point at, and what is true about each.
 *
 * Every field here is a claim the interface will make, so the honest ones are
 * the important ones. `calibrated` is false on all three and that is not an
 * omission: no camera reachable from this page has an independent along-road
 * survey, so the engine returns `null` for every speed and the interface says
 * `no speed` rather than showing a pixel-derived guess. The motorway clip is
 * the case worth naming -- a lane-marking survey exists for it, but its
 * along-road scale rests on an assumed 18 m dash period with no independent
 * corroboration, so no km/h figure derived from that clip appears anywhere on
 * this page.
 *
 * `speedLimitKmh` is null for the same family of reasons. A limit sign is
 * visible on the median of the motorway clip and its digits are not legible at
 * this resolution, so there is no posted limit to compare against; a limit here
 * would be user-set, and the interface would have to say so. */

import type { Point } from "../engine/geometry";

export interface GateSpec {
  readonly name: string;
  /** Normalized [0, 1] coordinates against the source's own frame. */
  readonly start: Point;
  readonly end: Point;
  readonly labelPositive: string;
  readonly labelNegative: string;
  readonly expectedDirection: string | null;
}

export interface SourceSpec {
  readonly id: string;
  readonly label: string;
  readonly kind: "clip" | "camera";
  /** Relative to the page, so the site works from a project subpath. */
  readonly url: string | null;
  /** Frame rate of the source, used for the detection cadence and the speed
   * estimator's memory bound -- not for any timing the page reports. */
  readonly fps: number;
  readonly classes: readonly (readonly [number, string])[];
  readonly gate: GateSpec;
  readonly calibrated: boolean;
  /** Why there is no speed, in the interface's own words. */
  readonly speedNote: string;
  /** One line under the video saying what the visitor is looking at. */
  readonly caption: string;
}

/** COCO ids, matching the exported graph's class ordering. */
const PERSON = [0, "person"] as const;
const BICYCLE = [1, "bicycle"] as const;
const CAR = [2, "car"] as const;
const MOTORCYCLE = [3, "motorcycle"] as const;
const BUS = [5, "bus"] as const;
const TRUCK = [7, "truck"] as const;

export const SOURCES: readonly SourceSpec[] = [
  {
    id: "motorway",
    label: "Motorway",
    kind: "clip",
    url: "clips/motorway-a40.mp4",
    fps: 30,
    classes: [CAR, MOTORCYCLE, BUS, TRUCK],
    // The inbound carriageway only. A single gate drawn across both
    // carriageways would count the opposing flow as wrong-way traffic, and the
    // half it leaves out is the useful half of the picture: those vehicles
    // cross the gate's infinite line without ever meeting the segment, which is
    // the difference between a line and a gate, drawn.
    gate: {
      name: "inbound",
      start: [0.06, 0.8],
      end: [0.46, 0.8],
      labelPositive: "away",
      labelNegative: "toward",
      expectedDirection: "toward",
    },
    calibrated: false,
    speedNote:
      "This camera has no independent along-road survey, so the engine reports no speed rather than a pixel-derived guess.",
    caption: "German A40, filmed from an overpass. Three lanes per carriageway.",
  },
  {
    id: "street",
    label: "Street",
    kind: "clip",
    url: "clips/street-aisle.mp4",
    fps: 12,
    classes: [PERSON, BICYCLE, CAR],
    gate: {
      name: "aisle",
      start: [0.02, 0.55],
      end: [0.98, 0.55],
      labelPositive: "in",
      labelNegative: "out",
      expectedDirection: null,
    },
    calibrated: false,
    speedNote: "This camera has not been surveyed, so the engine reports no speed.",
    caption: "An open paved area from above. A car, a cyclist and a pedestrian cross the aisle.",
  },
  {
    id: "webcam",
    label: "Webcam",
    kind: "camera",
    url: null,
    fps: 30,
    classes: [PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK],
    gate: {
      name: "midline",
      start: [0.0, 0.5],
      end: [1.0, 0.5],
      labelPositive: "in",
      labelNegative: "out",
      expectedDirection: null,
    },
    calibrated: false,
    speedNote: "An arbitrary camera view has no surveyed geometry, so there is no speed to report.",
    caption: "Your own camera. The frames are read, detected and discarded in this tab.",
  },
];

export function sourceById(id: string): SourceSpec {
  const found = SOURCES.find((source) => source.id === id);
  if (found === undefined) {
    throw new Error(`unknown source ${JSON.stringify(id)}`);
  }
  return found;
}

export function keepClassesOf(source: SourceSpec): Map<number, string> {
  return new Map(source.classes.map(([id, name]) => [id, name]));
}
