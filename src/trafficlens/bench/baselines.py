"""The standard failure modes of naive traffic counting, implemented
faithfully so the benchmark can measure what they cost.

Every one of these is a well-known way to count vehicles that is still
widely used because it is easy to write and looks right on a demo clip.
None of them is anyone's earlier work; they are the field's default
answers, implemented here at their best so the engine has something real
to be measured against. A benchmark won against a deliberately weakened
contender measures nothing, so each class below keeps the obvious guards
a competent implementation would have (bounded gates, cross-class bars,
track lifetimes) and differs from the engine only in the one decision
that names it.

Two families, two interfaces
----------------------------
The four baselines are NOT one interface. Two of them are counting
rules, two are trackers, and they sit at different points of the
pipeline:

- **Counting rules** -- ``BandCounter`` and ``PerFrameCounter`` -- share
  ``trafficlens.core.gate.GateCounter``'s surface exactly:
  ``update(track_id, class_name, prev, curr, frame_index, timestamp,
  speed_kmh=None, speed_limit_kmh=None) -> CrossingEvent | None``, plus
  ``forget(track_id)``, ``totals`` and ``total()``. They consume anchor
  positions and emit ``CrossingEvent``.
- **Trackers** -- ``CentroidTracker`` and ``GreedyIoUTracker`` -- share
  ``trafficlens.track.tracker.Tracker``'s surface:
  ``update(detections, frame_index) -> list[Track]``, plus ``reset()``.
  They consume ``Detection`` and emit the engine's own ``Track`` objects.

The benchmark harness composes {tracker} x {counting rule}: with three
trackers and three counting rules, a count error can be attributed to
the association step or to the counting rule separately, because
swapping one leaves the other's input identical. Holding the tracker
fixed and swapping only the rule is a RULE comparison; holding the rule
fixed and swapping only the tracker is a TRACKER comparison. Collapsing
all four into one interface would have made that impossible.

What these share with the engine, stated plainly
-------------------------------------------------
A baseline that shared geometry with its own scorer would be graded 1.0
by construction, so what is shared and what is not is written out here
and in each class's docstring rather than left to be discovered:

- **The ground truth is shared with nothing.** Counting accuracy is
  scored against the slit-scan labels of ``trafficlens.bench.slitscan``,
  which are read from raw pixels by a human and share no detector, no
  tracker, no threshold with either the engine or these baselines. That
  is the one comparison in the benchmark where nothing is shared.
- **The gate segment and the anchor policy ARE shared.** Both counting
  rules take the same ``Gate`` object the engine takes, and both consume
  the bottom-centre anchor that ``trafficlens.core.geometry.box_anchor``
  defines (via ``Track.anchor``). Band-vs-gate is therefore a comparison
  of two counting RULES applied to identical tracker output over
  identical geometry -- it is not evidence about anything else, and it
  must not be reported as though it were.
- **Detection input is shared.** All three trackers see the same
  ``Detection`` list per frame, and all three read scores with the same
  meaning.
- **The IoU metric is shared.** ``GreedyIoUTracker`` computes overlap
  with ``trafficlens.track.associate.iou_matrix``, the engine's own
  function. What differs is the matching POLICY (greedy, one stage), not
  the measurement.
- **The track lifecycle is shared.** Both baseline trackers use the
  engine's ``TRACK_MAX_AGE``, ``TRACK_MIN_HITS`` and ``TRACK_HIGH_CONF``,
  the engine's rule that a tentative track dies on its first miss, the
  engine's cross-class association bar, and the engine's output
  convention (below). A measured difference therefore cannot be a
  difference in how long tracks live or when they confirm.
- **The Kalman filter and the Hungarian assignment are NOT shared.**
  Neither baseline has a motion model, so neither predicts a box, gates
  by Mahalanobis distance, or smooths an anchor; and neither solves for a
  globally optimal assignment. Those are the differences under test.

Output convention (trackers)
----------------------------
Both baseline trackers follow ``Tracker.update``'s convention exactly:
the returned list holds only CONFIRMED tracks that were updated by a
real detection in THIS frame (``time_since_update == 0``), ascending by
``track_id``. Tentative tracks and coasting confirmed tracks stay
internal. This is what lets the harness drop either one in where the
engine's tracker was without changing a line of the consuming code.
``Track.history`` likewise grows only on detection-backed frames.

Unlike the engine, ``Track.box`` here is the last OBSERVED detection box,
never a predicted one -- with no motion model there is nothing to predict
with, so a coasting track's box simply goes stale. That staleness is not
an implementation shortcut; it is the failure mode.

numpy + stdlib + ``trafficlens.core`` / ``.detect`` / ``.track`` only.
"""

from __future__ import annotations

import math

import numpy as np

from trafficlens.core.constants import (
    BASELINE_BAND_PX,
    BASELINE_CENTROID_MAX_DISTANCE_PX,
    BASELINE_GREEDY_IOU_THRESH,
    GEOMETRY_EPS,
    TRACK_HIGH_CONF,
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
)
from trafficlens.core.gate import CrossingEvent, Gate, is_over_limit
from trafficlens.core.geometry import Point
from trafficlens.detect.base import Detection
from trafficlens.track.associate import iou_matrix
from trafficlens.track.tracker import STATE_CONFIRMED, STATE_TENTATIVE, Track


# -- band geometry, shared by both counting rules ----------------------------


def _signed_offset(gate: Gate, p: Point) -> tuple[float, float]:
    """Return ``(offset, t)`` for point ``p`` against ``gate``.

    ``offset`` is the SIGNED perpendicular distance in pixels from the
    gate's infinite line, using the identical cross-product expression
    (and therefore the identical sign convention) as
    ``trafficlens.core.geometry.side_of_line``: positive on the +1 side
    the gate calls ``label_positive``, negative on the -1 side.

    ``t`` is the position of ``p``'s foot of perpendicular along the gate
    as a fraction of its length: 0 at ``start``, 1 at ``end``, outside
    [0, 1] when ``p`` is past an end.

    ``Gate.__post_init__`` rejects a zero-length gate, so the division is
    always safe.
    """
    ax, ay = gate.start
    bx, by = gate.end
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    cross = dy * (p[0] - ax) - dx * (p[1] - ay)
    offset = cross / math.sqrt(length_sq)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq
    return offset, t


def _band_offset(gate: Gate, p: Point, band_px: float) -> float | None:
    """The signed offset of ``p`` if it lies inside the gate's band, else
    ``None``.

    The band is the closed rectangle of half-width ``band_px`` around the
    gate SEGMENT: ``abs(offset) <= band_px`` AND the foot of perpendicular
    lies between the gate's two endpoints. The second condition is the
    guard every competent band implementation has, and it is here for the
    same reason the engine bounds its segment -- without it, a vehicle on
    the opposite carriageway crossing the gate's infinite line hundreds of
    pixels past its end would be counted, and the baseline would be a
    straw man rather than the rule people actually deploy.

    The rectangle is not extended by ``band_px`` past the endpoints (no
    rounded caps): the band tracks the drawn gate's extent exactly, so a
    band rule and the engine's segment test agree about which vehicles are
    even eligible, and disagree only about the crossing test itself.
    """
    offset, t = _signed_offset(gate, p)
    if t < -GEOMETRY_EPS or t > 1.0 + GEOMETRY_EPS:
        return None
    if abs(offset) > band_px:
        return None
    return offset


class _BandRule:
    """Shared machinery of the two band-based counting rules: the gate,
    the band test, the direction test and the ``GateCounter`` bookkeeping
    surface (``totals``, ``total()``, ``forget()``).

    Direction, for both rules, is the sign of the PERPENDICULAR
    displacement from ``prev`` to ``curr``: moving toward the gate's +1
    side gives ``label_positive``, toward the -1 side ``label_negative``.
    A band rule cannot use the engine's side-change test -- the whole
    point is that it fires while the object is still *inside* the band,
    where its side may be either or neither -- so the direction of travel
    is the only evidence available. When the perpendicular displacement is
    exactly zero (motion purely ALONG the gate, or no motion at all) there
    is no direction evidence and no event is emitted. That guard, if
    anything, understates both rules' error rates.
    """

    def __init__(self, gate: Gate, band_px: float = BASELINE_BAND_PX) -> None:
        self.gate = gate
        self.band_px = float(band_px)
        self.totals: dict[str, dict[str, int]] = {}

    def _direction(self, prev: Point, curr: Point) -> int:
        """+1 / -1 for perpendicular travel toward the gate's positive /
        negative side, 0 when the step is parallel to the gate."""
        offset_prev, _ = _signed_offset(self.gate, prev)
        offset_curr, _ = _signed_offset(self.gate, curr)
        delta = offset_curr - offset_prev
        if delta > GEOMETRY_EPS:
            return 1
        if delta < -GEOMETRY_EPS:
            return -1
        return 0

    def _emit(
        self,
        track_id: int,
        class_name: str,
        signed: int,
        curr: Point,
        frame_index: int,
        timestamp: float,
        speed_kmh: float | None,
        speed_limit_kmh: float | None,
    ) -> CrossingEvent:
        """Build the event and fold it into ``totals``.

        ``crossing_x`` / ``crossing_y`` are the object's own in-band
        position, NOT an intersection with the gate line: a band rule
        never computes one. They are therefore accurate only to within
        ``band_px``, where the engine's are the exact sub-pixel segment
        intersection. Anything downstream that measures crossing POSITION
        (a sub-frame timestamp, a lane assignment) inherits that error.
        """
        direction = (
            self.gate.label_positive if signed == 1 else self.gate.label_negative
        )
        class_totals = self.totals.setdefault(class_name, {})
        class_totals[direction] = class_totals.get(direction, 0) + 1
        return CrossingEvent(
            track_id=track_id,
            class_name=class_name,
            gate=self.gate.name,
            direction=direction,
            signed_direction=signed,
            frame_index=frame_index,
            timestamp=timestamp,
            crossing_x=curr[0],
            crossing_y=curr[1],
            speed_kmh=speed_kmh,
            is_violation=is_over_limit(speed_kmh, speed_limit_kmh),
        )

    def forget(self, track_id: int) -> None:
        """Clear all memory of ``track_id`` (see each subclass)."""
        raise NotImplementedError

    def total(self) -> int:
        return sum(sum(directions.values()) for directions in self.totals.values())


class BandCounter(_BandRule):
    """The pixel-band rule: count a track the first time its anchor is
    seen INSIDE a band of ``band_px`` around the gate line.

    Shared with the engine: the ``Gate`` object (same segment, same
    endpoints, same direction labels), the bottom-centre anchor policy,
    the once-per-track bookkeeping, and the whole ``GateCounter`` surface.
    A band-vs-gate benchmark run is therefore a comparison of two counting
    RULES on identical tracker output over identical geometry, and nothing
    more; it says nothing about detection or tracking.

    Not shared: the crossing test itself. The engine asks whether the
    swept path from ``prev`` to ``curr`` intersects the bounded gate
    segment. This rule asks whether ``curr`` is close to the line. Those
    two questions differ in both directions, which is exactly why this
    baseline exists:

    **Miss.** An object that steps further than ``band_px`` in one frame
    passes clean over the band and is never counted. Fast vehicles, small
    bands and low frame rates all make this worse, and the three
    compound: the anchor of a vehicle near the camera can move tens of
    pixels per frame. The engine is immune to it, because a swept segment
    has no gaps regardless of how long it is.

    **Phantom.** An object that enters the band and leaves the same way it
    came -- a lane change along the gate line, an anchor jittering as a
    box's lower edge is clipped, a vehicle pulling up short of the line --
    is counted as a crossing that never happened. The engine refuses it,
    because both positions are on the same side of the gate.

    Widening the band trades the first error for the second and narrowing
    it trades back; no value of ``band_px`` removes either, which is the
    finding the benchmark is meant to quantify.
    """

    def __init__(self, gate: Gate, band_px: float = BASELINE_BAND_PX) -> None:
        super().__init__(gate, band_px)
        self._counted: set[int] = set()

    def update(
        self,
        track_id: int,
        class_name: str,
        prev: Point,
        curr: Point,
        frame_index: int,
        timestamp: float,
        speed_kmh: float | None = None,
        speed_limit_kmh: float | None = None,
    ) -> CrossingEvent | None:
        if track_id in self._counted:
            return None
        if _band_offset(self.gate, curr, self.band_px) is None:
            return None
        signed = self._direction(prev, curr)
        if signed == 0:
            return None
        self._counted.add(track_id)
        return self._emit(
            track_id,
            class_name,
            signed,
            curr,
            frame_index,
            timestamp,
            speed_kmh,
            speed_limit_kmh,
        )

    def forget(self, track_id: int) -> None:
        """Clear all memory of ``track_id``, so a recycled tracker ID can
        be counted again -- same contract as ``GateCounter.forget``."""
        self._counted.discard(track_id)


class PerFrameCounter(_BandRule):
    """No tracking at all: count EVERY frame on which a box sits on the
    gate.

    Shared with the engine: the ``Gate``, the anchor policy and the
    ``GateCounter`` surface, exactly as ``BandCounter`` shares them.

    Not shared: any notion of object identity. ``track_id`` is carried
    into the emitted event so the harness can attribute a count, but it is
    never used to decide whether to count. Nor is ``prev`` used for that
    decision -- the band test reads ``curr`` alone. ``prev`` supplies only
    the direction of travel, the same way it does for ``BandCounter``, so
    that the events this rule emits are shaped like every other rule's.

    The failure mode is the one every naive occupancy counter has: a
    vehicle inside the band for N frames is counted N times, so the total
    is not a vehicle count at all but a dwell-weighted one. It over-counts
    slow and stopped traffic in exact proportion to how slowly it moves --
    which is to say it is at its most wrong precisely during congestion,
    the condition a traffic count is most often commissioned to measure.
    A queue creeping over the gate at walking pace can multiply a single
    vehicle by dozens.

    This is the reason a counter needs a tracker at all, and stating that
    is what this baseline is here for.
    """

    def update(
        self,
        track_id: int,
        class_name: str,
        prev: Point,
        curr: Point,
        frame_index: int,
        timestamp: float,
        speed_kmh: float | None = None,
        speed_limit_kmh: float | None = None,
    ) -> CrossingEvent | None:
        if _band_offset(self.gate, curr, self.band_px) is None:
            return None
        signed = self._direction(prev, curr)
        if signed == 0:
            return None
        return self._emit(
            track_id,
            class_name,
            signed,
            curr,
            frame_index,
            timestamp,
            speed_kmh,
            speed_limit_kmh,
        )

    def forget(self, track_id: int) -> None:
        """Accepted for interface parity and does nothing: this rule holds
        no per-track memory to clear. That absence IS the failure mode,
        not an oversight -- see the class docstring."""
        del track_id


# -- baseline trackers -------------------------------------------------------


class _BaselineTracker:
    """Shared lifecycle of both baseline trackers, so that the ONLY thing
    separating them from each other -- and from the engine -- is the
    association step ``_associate`` implements.

    Everything here is the engine's own policy, deliberately: the
    confidence floor for starting a track, ``min_hits`` to confirm, the
    rule that a tentative track dies on its first miss, death once
    ``time_since_update > max_age``, the cross-class bar, ascending ID
    allocation in detection order, and the output convention (confirmed
    and detector-backed this frame only, ascending by ``track_id``). A
    benchmark difference therefore cannot be attributed to any of them.

    What is absent, in both subclasses, is a motion model. No prediction
    step runs, so a track's box is always the last box a detector actually
    produced, and a coasting track's box stands still while the object
    does not. There is likewise no second association stage: a single
    confidence threshold divides detections into "used" and "discarded",
    with no band of low-confidence boxes kept back to keep an occluded
    track alive.
    """

    def __init__(
        self,
        max_age: int = TRACK_MAX_AGE,
        min_hits: int = TRACK_MIN_HITS,
        conf_thresh: float = TRACK_HIGH_CONF,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.conf_thresh = conf_thresh
        self._tracks: list[Track] = []  # creation order == ascending track_id
        self._next_id = 1

    def reset(self) -> None:
        """Drop every track and restart the ID counter at 1, restoring the
        exact state of a freshly constructed tracker."""
        self._tracks = []
        self._next_id = 1

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Advance one frame. See the class docstring for the lifecycle
        and the module docstring for the return convention."""
        del frame_index  # caller's clock only; association is call-clocked

        dets = [d for d in detections if d.score >= self.conf_thresh]

        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1
            # No prediction: track.box stays at the last observed box.

        matches, unmatched_dets = self._associate(self._tracks, dets)
        for track_idx, det_idx in matches:
            self._apply_detection(self._tracks[track_idx], dets[det_idx])
        for det_idx in unmatched_dets:
            self._start_track(dets[det_idx])

        self._tracks = [
            track
            for track in self._tracks
            if track.time_since_update == 0
            or (
                track.state == STATE_CONFIRMED
                and track.time_since_update <= self.max_age
            )
        ]

        return [
            track
            for track in self._tracks
            if track.state == STATE_CONFIRMED and track.time_since_update == 0
        ]

    def _associate(
        self, tracks: list[Track], dets: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Return ``(matches, unmatched_detection_indices)`` with matches
        as ``(track_index, detection_index)`` pairs. Implemented by each
        subclass; it is the only thing that differs between them."""
        raise NotImplementedError

    @staticmethod
    def _greedy(
        candidates: list[tuple[float, int, int]], n_dets: int
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Take eligible pairs best-first, first come first served.

        ``candidates`` is ``(rank, track_index, detection_index)`` for
        every pair that clears its subclass's threshold, where a smaller
        ``rank`` is a better pair. Sorting on the full triple makes the
        result independent of the order pairs were generated in and
        breaks exact ties toward the lowest track index then the lowest
        detection index, so the same detection sequence always yields the
        same IDs.

        This is the greedy step itself, and it is the point: a pair that
        looks best in isolation is taken even when giving it up would have
        let two other pairs match better overall. The engine solves the
        whole frame's assignment at once instead, so it can refuse a
        locally-attractive pair.
        """
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, track_idx, det_idx in sorted(candidates):
            if track_idx in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_idx)
            used_dets.add(det_idx)
            matches.append((track_idx, det_idx))
        matches.sort()
        unmatched = [j for j in range(n_dets) if j not in used_dets]
        return matches, unmatched

    def _apply_detection(self, track: Track, det: Detection) -> None:
        """Fold a matched detection into a track. The box becomes the
        detection's box verbatim -- no filter, no smoothing, so detector
        jitter passes straight through to the anchor and to any speed
        estimated from it."""
        track.box = (det.x1, det.y1, det.x2, det.y2)
        track.score = det.score
        track.hits += 1
        track.time_since_update = 0
        if track.state == STATE_TENTATIVE and track.hits >= self.min_hits:
            track.state = STATE_CONFIRMED
        track.history.append(track.anchor)

    def _start_track(self, det: Detection) -> None:
        """Spawn a tentative track from an unmatched detection; its class
        is fixed here, forever, exactly as the engine fixes it."""
        track = Track(
            track_id=self._next_id,
            class_name=det.class_name,
            box=(det.x1, det.y1, det.x2, det.y2),
            score=det.score,
            age=1,
            hits=1,
            time_since_update=0,
            state=STATE_CONFIRMED if 1 >= self.min_hits else STATE_TENTATIVE,
        )
        self._next_id += 1
        track.history.append(track.anchor)
        self._tracks.append(track)


class CentroidTracker(_BaselineTracker):
    """Nearest-centroid association: match each detection to the track
    whose last observed box has the closest centre, within
    ``max_distance_px``.

    Shared with the engine: ``Detection`` input, the ``Track`` output
    type and output convention, the cross-class bar, ID allocation and
    the whole track lifecycle (see ``_BaselineTracker``).

    Not shared: everything about motion. There is no Kalman filter, so
    there is no predicted box to measure against and no Mahalanobis gate
    to bar an implausible pair; association compares a detection against
    where the object WAS, not where it should now be. Association is also
    greedy rather than a global assignment.

    The failure mode is identity swapping. When two objects pass each
    other, there is an instant at which each one is nearer to the other's
    previous position than to its own -- and with no velocity estimate,
    nothing distinguishes the two. The tracker takes the swap, and from
    then on every downstream product of identity is wrong for both
    vehicles: their crossings are attributed to each other, their speeds
    are computed across a jump, and a per-class count can flip if the two
    objects are different classes (the cross-class bar prevents that last
    one here, which is why the swap this baseline exhibits is same-class).

    Centroid, not anchor, is deliberate: the box CENTRE is what a
    centroid tracker associates on, and that is what this associates on.
    Downstream the ``Track.anchor`` property still returns the shared
    bottom-centre point, so a counting rule sees the same kind of position
    whichever tracker produced it.
    """

    def __init__(
        self,
        max_distance_px: float = BASELINE_CENTROID_MAX_DISTANCE_PX,
        max_age: int = TRACK_MAX_AGE,
        min_hits: int = TRACK_MIN_HITS,
        conf_thresh: float = TRACK_HIGH_CONF,
    ) -> None:
        super().__init__(max_age=max_age, min_hits=min_hits, conf_thresh=conf_thresh)
        self.max_distance_px = float(max_distance_px)

    @staticmethod
    def _centres(boxes: np.ndarray) -> np.ndarray:
        return np.stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0],
            axis=1,
        )

    def _associate(
        self, tracks: list[Track], dets: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        if not tracks or not dets:
            return [], list(range(len(dets)))

        track_centres = self._centres(
            np.array([t.box for t in tracks], dtype=np.float64)
        )
        det_centres = self._centres(
            np.array([(d.x1, d.y1, d.x2, d.y2) for d in dets], dtype=np.float64)
        )
        distances = np.linalg.norm(
            track_centres[:, None, :] - det_centres[None, :, :], axis=2
        )

        candidates = [
            (float(distances[i, j]), i, j)
            for i in range(len(tracks))
            for j in range(len(dets))
            if distances[i, j] <= self.max_distance_px
            and dets[j].class_name == tracks[i].class_name
        ]
        return self._greedy(candidates, len(dets))


class GreedyIoUTracker(_BaselineTracker):
    """Greedy IoU association: repeatedly take the highest-IoU
    track/detection pair that still clears ``iou_thresh``, first come
    first served.

    Shared with the engine: ``Detection`` input, the ``Track`` output type
    and output convention, the cross-class bar, ID allocation, the whole
    track lifecycle, and the IoU measurement itself -- overlap is computed
    by ``trafficlens.track.associate.iou_matrix``, the engine's own
    function, so the two differ in matching POLICY and not in what a
    number means.

    Not shared: the Kalman filter, the Mahalanobis gate, the globally
    optimal assignment, and -- the one that matters most here -- the
    second association stage. The engine keeps back detections in the
    low-confidence band and offers them to already-confirmed tracks that
    stage one could not match, which is how a track survives the
    confidence dip an occlusion causes. This baseline has a single
    threshold and a single stage: a detection is either good enough to
    use for everything, including starting a new track, or it is thrown
    away.

    The failure mode is fragmentation. A vehicle passing behind a lorry
    or a sign gantry has its detection score dip below the threshold for
    a stretch of frames. With no low-confidence stage those frames are
    lost, and with no motion model the track's box freezes where the
    object last was, so by the time a confident detection returns the two
    boxes no longer overlap enough to match. The track is cut in two and
    a fresh ID is issued -- which downstream is a second vehicle counted,
    a broken speed history, and a stopped-vehicle timer reset.

    ``iou_thresh`` defaults far below the engine's IoU floor for an honest
    reason spelled out at ``BASELINE_GREEDY_IOU_THRESH``: without a
    prediction to compare against, a strict floor would reject ordinary
    motion, so the strictness that works for the engine would make this
    baseline a straw man.
    """

    def __init__(
        self,
        iou_thresh: float = BASELINE_GREEDY_IOU_THRESH,
        max_age: int = TRACK_MAX_AGE,
        min_hits: int = TRACK_MIN_HITS,
        conf_thresh: float = TRACK_HIGH_CONF,
    ) -> None:
        super().__init__(max_age=max_age, min_hits=min_hits, conf_thresh=conf_thresh)
        self.iou_thresh = float(iou_thresh)

    def _associate(
        self, tracks: list[Track], dets: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        if not tracks or not dets:
            return [], list(range(len(dets)))

        ious = iou_matrix(
            np.array([t.box for t in tracks], dtype=np.float64),
            np.array([(d.x1, d.y1, d.x2, d.y2) for d in dets], dtype=np.float64),
        )
        # Rank by -IoU so the shared greedy step (which takes the smallest
        # rank first) takes the HIGHEST overlap first.
        candidates = [
            (-float(ious[i, j]), i, j)
            for i in range(len(tracks))
            for j in range(len(dets))
            if ious[i, j] >= self.iou_thresh
            and dets[j].class_name == tracks[i].class_name
        ]
        return self._greedy(candidates, len(dets))
