/** The per-frame session loop, in one place.
 *
 * `trafficlens.pipeline.run_session` is the Python original: track, observe,
 * count, reap. Until now the TypeScript side had that loop written out inside
 * `parity.test.ts`, which meant the loop proven to agree with Python and the
 * loop the shipped page runs were two different pieces of code that happened to
 * look alike. They are the same object now: the parity suite drives this class,
 * the live control room drives this class, and `?selftest=1` drives this class
 * in the visitor's own browser.
 *
 * The reaping rule is the pipeline's own and it matters: a confirmed track
 * survives while `timeSinceUpdate <= maxAge` and may re-associate at exactly
 * `maxAge`, so a track unseen for STRICTLY MORE than `maxAge` frames is the
 * first moment it is provably gone. Reaping one frame early would forget a gate
 * counter's `_counted` entry while the track could still return, and the same
 * vehicle would be counted twice. */

import { Gate, GateCounter } from "./gate";
import type { CrossingEvent } from "./gate";
import type { Point } from "./geometry";
import type { RoadPlane } from "./homography";
import { SpeedEstimator } from "./speed";
import { Tracker } from "./tracker";
import type { Detection, TrackerOptions } from "./tracker";

/** gate name -> class name -> direction label -> count. */
export type Counts = Record<string, Record<string, Record<string, number>>>;

export interface PipelineOptions {
  readonly gates: readonly Gate[];
  /** `null` is `NO_CALIBRATION`: every speed is then `null`, always. */
  readonly plane: RoadPlane | null;
  readonly fps: number;
  readonly speedLimitKmh: number | null;
  readonly tracker?: TrackerOptions | undefined;
}

/** One live track, as the drawing and panel code needs it. */
export interface TrackView {
  readonly trackId: number;
  readonly className: string;
  readonly box: readonly [number, number, number, number];
  readonly anchor: Point;
  readonly speedKmh: number | null;
}

export interface StepResult {
  readonly frameIndex: number;
  readonly timestamp: number;
  readonly tracks: readonly TrackView[];
  readonly events: readonly CrossingEvent[];
}

export class SessionPipeline {
  readonly fps: number;
  readonly speedLimitKmh: number | null;

  private readonly trackerOptions: TrackerOptions;
  private readonly plane: RoadPlane | null;
  private tracker: Tracker;
  private speed: SpeedEstimator;
  private gateList: Gate[];
  private counters: Map<string, GateCounter>;
  private previousAnchor = new Map<number, Point>();
  private lastSeen = new Map<number, number>();
  private allocated = 0;
  private countingSince: number | null = null;

  constructor(options: PipelineOptions) {
    this.trackerOptions = options.tracker ?? {};
    this.plane = options.plane;
    this.fps = options.fps;
    this.speedLimitKmh = options.speedLimitKmh;
    this.tracker = new Tracker(this.trackerOptions);
    this.speed = new SpeedEstimator(options.plane, options.fps);
    this.gateList = [...options.gates];
    this.counters = new Map(this.gateList.map((gate) => [gate.name, new GateCounter(gate)]));
  }

  /** True when this session can produce a speed at all. The interface prints
   * `no speed` when it is false rather than hiding the readout. */
  get calibrated(): boolean {
    return this.plane !== null;
  }

  get gates(): readonly Gate[] {
    return this.gateList;
  }

  get tracksAllocated(): number {
    return this.allocated;
  }

  /** How many track ids this pipeline is still holding state for -- gate
   * `_counted` entries, previous anchors and speed history.
   *
   * Exposed because reaping is otherwise invisible from outside, and reaping is
   * exactly what a non-monotonic frame clock breaks: `step()` reaps on
   * `frameIndex - lastSeen > maxAge`, so a caller that restarts its frame
   * counter at 0 while this pipeline survives makes that difference negative
   * and nothing is ever released. `web/src/ui/app.ts` used to do that on
   * stop -> start. */
  get retainedTracks(): number {
    return this.lastSeen.size;
  }

  /** The clip timestamp the current counts started accumulating from: `null`
   * while they cover the whole session, and the moment of the change once the
   * gate has been moved. The interface says which. */
  get countingSinceTimestamp(): number | null {
    return this.countingSince;
  }

  /** Advance one frame. */
  step(
    detections: readonly Detection[],
    frameIndex: number,
    timestamp: number,
  ): StepResult {
    const tracks = this.tracker.update(detections, frameIndex);
    const views: TrackView[] = [];
    const events: CrossingEvent[] = [];

    for (const track of tracks) {
      const anchor = track.anchor;
      this.lastSeen.set(track.trackId, frameIndex);
      this.allocated = Math.max(this.allocated, track.trackId);

      this.speed.observe(track.trackId, anchor, timestamp);
      const speedKmh = this.speed.speedKmh(track.trackId);
      views.push({
        trackId: track.trackId,
        className: track.className,
        box: [track.box[0], track.box[1], track.box[2], track.box[3]],
        anchor,
        speedKmh,
      });

      const previous = this.previousAnchor.get(track.trackId);
      if (previous !== undefined) {
        for (const gate of this.gateList) {
          const event = (this.counters.get(gate.name) as GateCounter).update(
            track.trackId,
            track.className,
            previous,
            anchor,
            frameIndex,
            timestamp,
            speedKmh,
            this.speedLimitKmh,
          );
          if (event !== null) {
            events.push(event);
          }
        }
      }
      this.previousAnchor.set(track.trackId, anchor);
    }

    for (const [trackId, seen] of [...this.lastSeen]) {
      if (frameIndex - seen > this.tracker.maxAge) {
        this.lastSeen.delete(trackId);
        for (const counter of this.counters.values()) {
          counter.forget(trackId);
        }
        this.speed.forget(trackId);
        this.previousAnchor.delete(trackId);
      }
    }

    return { frameIndex, timestamp, tracks: views, events };
  }

  counts(): Counts {
    const out: Counts = {};
    for (const [name, counter] of this.counters) {
      for (const [className, directions] of counter.totals) {
        for (const [direction, count] of directions) {
          ((out[name] ??= {})[className] ??= {})[direction] = count;
        }
      }
    }
    return out;
  }

  total(): number {
    let sum = 0;
    for (const counter of this.counters.values()) {
      sum += counter.total();
    }
    return sum;
  }

  /** Swap the gates, keeping the tracks.
   *
   * Counts start again from this moment, and they have to: a count is a
   * property of a gate, so totals gathered against a gate that is no longer
   * there would be a number about geometry the visitor can no longer see. The
   * tracks, their speeds and their previous anchors survive, so a vehicle
   * halfway across the frame can cross the new gate on the very next frame
   * rather than waiting to be re-detected. */
  replaceGates(gates: readonly Gate[], timestamp: number): void {
    this.gateList = [...gates];
    this.counters = new Map(this.gateList.map((gate) => [gate.name, new GateCounter(gate)]));
    this.countingSince = timestamp;
  }

  /** Back to a freshly constructed session: no tracks, no counts, ids from 1. */
  reset(): void {
    this.tracker = new Tracker(this.trackerOptions);
    this.speed = new SpeedEstimator(this.plane, this.fps);
    this.counters = new Map(this.gateList.map((gate) => [gate.name, new GateCounter(gate)]));
    this.previousAnchor.clear();
    this.lastSeen.clear();
    this.allocated = 0;
    this.countingSince = null;
  }
}

/** Build gates from normalized coordinates against a frame size. */
export function gatesFromNormalized(
  specs: readonly {
    name: string;
    start: Point;
    end: Point;
    labelPositive?: string;
    labelNegative?: string;
    expectedDirection?: string | null;
  }[],
  width: number,
  height: number,
): Gate[] {
  return specs.map((spec) =>
    Gate.fromNormalized(spec.name, spec.start, spec.end, width, height, {
      ...(spec.labelPositive === undefined ? {} : { labelPositive: spec.labelPositive }),
      ...(spec.labelNegative === undefined ? {} : { labelNegative: spec.labelNegative }),
      ...(spec.expectedDirection === undefined
        ? {}
        : { expectedDirection: spec.expectedDirection }),
    }),
  );
}
