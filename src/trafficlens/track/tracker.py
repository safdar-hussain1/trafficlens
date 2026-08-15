"""Two-stage multi-object tracker over the constant-velocity Kalman filter.

ByteTrack-style association adapted to per-class traffic tracking. One
``Tracker.update()`` call is one video frame; the tracker owns all state
(the ``KalmanBoxFilter`` itself is stateless). Every rule below is exact
and deterministic -- the TypeScript mirror reproduces each one verbatim,
and downstream counting/speed tasks build against them.

Association algorithm (per ``update()`` call)
---------------------------------------------
1. Every live track is predicted one step; its ``age`` and
   ``time_since_update`` advance and its ``box`` becomes the predicted box.
2. Stage 1: HIGH-confidence detections (``score >= high_thresh``) against
   ALL live tracks, confirmed and tentative. Cost is ``1 - IoU`` between
   the predicted and detected boxes; a pair is barred (cost ``inf``) when
   its squared Mahalanobis gating distance exceeds the chi-square 95%
   gate ``KALMAN_GATING_CHI2_95_4DOF``, or when the classes differ.
3. Stage 2: tracks left unmatched by stage 1 that are CONFIRMED, against
   LOW-confidence detections (``low_thresh <= score < high_thresh``).
   IoU-only cost, no Mahalanobis bar (an occluded box is exactly the box
   the filter is most unsure about); the cross-class bar still applies.
4. Unmatched HIGH-confidence detections start tentative tracks; unmatched
   low-confidence detections are discarded and can never start a track.
5. Lifecycle: a tentative track that misses even ONE frame dies
   immediately (only an unbroken run of hits can confirm it, which keeps
   one-frame NMS ghosts out of the output); a tentative track with
   ``hits >= min_hits`` becomes confirmed; a confirmed track dies when
   ``time_since_update > max_age``, so a detection gap of up to exactly
   ``max_age`` frames survives on prediction alone and may re-associate.

Cost / threshold semantics
--------------------------
``match_thresh`` is an IoU FLOOR: a pair may match only when
``IoU >= match_thresh``. Both stages implement it as
``assign(cost, max_cost)`` with ``cost = 1 - IoU`` and
``max_cost = 1 - match_thresh``, and ``assign`` drops every assigned pair
with ``cost > max_cost``. The mirror must evaluate the same float64
expressions so borderline IoUs resolve identically.

Class policy
------------
Association never crosses classes -- implemented by barring cross-class
pairs in the cost matrix, NOT by per-class sub-trackers, so ID allocation
stays a single global deterministic sequence. A track's ``class_name`` is
FIXED at creation (its first detection's class) and never renamed:
detector class flicker must not rename an existing track, otherwise the
class-consistency metric measured against ground truth later would be
meaningless. A flickered detection that cannot match anything simply
starts (or fails to start) its own track under the usual rules.

Output policy
-------------
``update()`` returns only CONFIRMED tracks that were updated with a real
detection in THIS frame (``time_since_update == 0``), in ascending
track_id order. Tentative tracks are internal, and coasting confirmed
tracks stay internal while they coast: a frame's output is exactly the
detector-backed state of that frame. The returned ``Track`` objects are
the tracker's own live records (not copies); they keep mutating on later
``update()`` calls, which lets downstream consumers hold a reference and
watch ``history`` grow.

``Track.box`` always reflects the Kalman state -- the corrected state on
matched frames, the predicted state while coasting -- so anchors move
smoothly through occlusions instead of jumping. ``Track.history`` gets the
anchor appended ONLY on frames with a real detection update (creation
included), never while coasting: downstream speed estimation must average
over measured motion, not over the filter's own extrapolation, or a long
coast would fabricate distance travelled.

ID allocation
-------------
IDs start at 1 and increase monotonically for the tracker's lifetime; new
IDs are assigned in ascending detection-index order among the unmatched
high-confidence detections of the frame. ``reset()`` clears all tracks
AND restarts the counter at 1. The same detection sequence therefore
always yields the same IDs, which is asserted by test.

``frame_index`` is the caller's frame clock, carried for interface parity
with downstream event stamping; association itself is clocked purely by
successive ``update()`` calls (one call = one frame).

numpy + stdlib only here; the Hungarian solver (scipy) is confined to
``trafficlens.track.associate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trafficlens.core.constants import (
    KALMAN_GATING_CHI2_95_4DOF,
    TRACK_HIGH_CONF,
    TRACK_LOW_CONF,
    TRACK_MATCH_IOU,
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
)
from trafficlens.core.geometry import Point, box_anchor
from trafficlens.detect.base import Detection
from trafficlens.track.associate import assign, iou_matrix
from trafficlens.track.kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah

STATE_TENTATIVE = "tentative"
STATE_CONFIRMED = "confirmed"


@dataclass
class Track:
    """One tracked object's public state (see module docstring for the
    exact box/history semantics)."""

    track_id: int
    class_name: str  # fixed at creation; never renamed by class flicker
    box: tuple[float, float, float, float]  # xyxy, Kalman-state-derived
    score: float  # score of the most recent matched detection
    age: int  # frames since creation, inclusive
    hits: int  # total matched-detection updates, creation included
    time_since_update: int  # frames since the last matched detection
    state: str  # STATE_TENTATIVE or STATE_CONFIRMED
    history: list[Point] = field(default_factory=list)  # anchor per updated frame

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the current box: where the object meets the
        road (see ``trafficlens.core.geometry.box_anchor``)."""
        return box_anchor(*self.box)


class _Live:
    """Internal record pairing a public Track with its Kalman state."""

    __slots__ = ("track", "mean", "cov")

    def __init__(self, track: Track, mean: np.ndarray, cov: np.ndarray) -> None:
        self.track = track
        self.mean = mean
        self.cov = cov


class Tracker:
    """Two-stage per-class multi-object tracker. All tunables default to
    the values in ``trafficlens.core.constants`` (the source of truth the
    TypeScript mirror also reads)."""

    def __init__(
        self,
        high_thresh: float = TRACK_HIGH_CONF,
        low_thresh: float = TRACK_LOW_CONF,
        match_thresh: float = TRACK_MATCH_IOU,
        max_age: int = TRACK_MAX_AGE,
        min_hits: int = TRACK_MIN_HITS,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self._kf = KalmanBoxFilter()
        self._live: list[_Live] = []  # creation order == ascending track_id
        self._next_id = 1

    def reset(self) -> None:
        """Drop every track and restart the ID counter at 1, restoring the
        exact state of a freshly constructed Tracker."""
        self._live = []
        self._next_id = 1

    # -- per-frame step ----------------------------------------------------

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Advance one frame: predict, associate in two stages, spawn and
        retire tracks. Returns the confirmed tracks updated by a detection
        this frame, ascending by track_id (see module docstring)."""
        del frame_index  # caller's clock only; association is call-clocked

        high = [d for d in detections if d.score >= self.high_thresh]
        low = [
            d for d in detections if self.low_thresh <= d.score < self.high_thresh
        ]

        # 1. Predict every live track one step; ageing happens here so an
        # unmatched track needs no second pass.
        for rec in self._live:
            rec.mean, rec.cov = self._kf.predict(rec.mean, rec.cov)
            rec.track.age += 1
            rec.track.time_since_update += 1
            rec.track.box = self._state_box(rec.mean)

        max_cost = 1.0 - self.match_thresh

        # 2. Stage 1: high-confidence detections vs ALL tracks.
        cost = self._stage1_cost(self._live, high)
        matches, unmatched_track_idx, unmatched_high_idx = assign(cost, max_cost)
        for t_idx, d_idx in matches:
            self._apply_detection(self._live[t_idx], high[d_idx])

        # 3. Stage 2: still-unmatched CONFIRMED tracks vs low-confidence
        # detections, IoU-only (no Mahalanobis bar), cross-class bar kept.
        stage2 = [
            self._live[i]
            for i in unmatched_track_idx
            if self._live[i].track.state == STATE_CONFIRMED
        ]
        cost = self._stage2_cost(stage2, low)
        matches, _, _ = assign(cost, max_cost)
        for t_idx, d_idx in matches:
            self._apply_detection(stage2[t_idx], low[d_idx])

        # 4. Unmatched high-confidence detections start tentative tracks,
        # in ascending detection-index order (ID determinism).
        for d_idx in unmatched_high_idx:
            self._start_track(high[d_idx])

        # 5. Retire: tentative tracks die on their first miss; confirmed
        # tracks die once their gap exceeds max_age.
        survivors = []
        for rec in self._live:
            tr = rec.track
            if tr.time_since_update == 0:
                survivors.append(rec)
            elif tr.state == STATE_CONFIRMED and tr.time_since_update <= self.max_age:
                survivors.append(rec)
        self._live = survivors

        # _live is creation-ordered, so this is ascending by track_id.
        return [
            rec.track
            for rec in self._live
            if rec.track.state == STATE_CONFIRMED and rec.track.time_since_update == 0
        ]

    # -- cost matrices -----------------------------------------------------

    def _stage1_cost(self, recs: list[_Live], dets: list[Detection]) -> np.ndarray:
        """(len(recs), len(dets)) cost matrix: 1 - IoU, with cross-class
        pairs and pairs outside the chi-square gate barred at inf."""
        if not recs or not dets:
            return np.zeros((len(recs), len(dets)))
        track_boxes = np.array([rec.track.box for rec in recs])
        det_boxes = np.array([(d.x1, d.y1, d.x2, d.y2) for d in dets])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)

        measurements = np.array([xyxy_to_xyah(b) for b in det_boxes])
        for i, rec in enumerate(recs):
            gating = self._kf.gating_distance(rec.mean, rec.cov, measurements)
            cost[i, gating > KALMAN_GATING_CHI2_95_4DOF] = np.inf
            for j, det in enumerate(dets):
                if det.class_name != rec.track.class_name:
                    cost[i, j] = np.inf
        return cost

    def _stage2_cost(self, recs: list[_Live], dets: list[Detection]) -> np.ndarray:
        """(len(recs), len(dets)) cost matrix: 1 - IoU with the cross-class
        bar only -- no Mahalanobis gate (see module docstring)."""
        if not recs or not dets:
            return np.zeros((len(recs), len(dets)))
        track_boxes = np.array([rec.track.box for rec in recs])
        det_boxes = np.array([(d.x1, d.y1, d.x2, d.y2) for d in dets])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        for i, rec in enumerate(recs):
            for j, det in enumerate(dets):
                if det.class_name != rec.track.class_name:
                    cost[i, j] = np.inf
        return cost

    # -- track mutations ---------------------------------------------------

    @staticmethod
    def _state_box(mean: np.ndarray) -> tuple[float, float, float, float]:
        b = xyah_to_xyxy(mean[:4])
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    def _apply_detection(self, rec: _Live, det: Detection) -> None:
        """Fold a matched detection into a track: Kalman correction, box,
        score, counters, confirmation, history."""
        measurement = xyxy_to_xyah(np.array([det.x1, det.y1, det.x2, det.y2]))
        rec.mean, rec.cov = self._kf.update(rec.mean, rec.cov, measurement)
        tr = rec.track
        tr.box = self._state_box(rec.mean)
        tr.score = det.score
        tr.hits += 1
        tr.time_since_update = 0
        if tr.state == STATE_TENTATIVE and tr.hits >= self.min_hits:
            tr.state = STATE_CONFIRMED
        tr.history.append(tr.anchor)

    def _start_track(self, det: Detection) -> None:
        """Spawn a tentative track from an unmatched high-confidence
        detection; its class is fixed here, forever."""
        measurement = xyxy_to_xyah(np.array([det.x1, det.y1, det.x2, det.y2]))
        mean, cov = self._kf.initiate(measurement)
        track = Track(
            track_id=self._next_id,
            class_name=det.class_name,
            box=self._state_box(mean),
            score=det.score,
            age=1,
            hits=1,
            time_since_update=0,
            state=STATE_CONFIRMED if 1 >= self.min_hits else STATE_TENTATIVE,
        )
        self._next_id += 1
        track.history.append(track.anchor)
        self._live.append(_Live(track, mean, cov))
