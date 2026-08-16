"""A synthetic traffic scene whose ground truth is exact, for Tier-1 speed
validation.

Why this module exists
----------------------
Speed is the one analytic in this product whose absolute correctness cannot
be checked against the flagship clip. That clip's along-road scale has no
independent anchor (see ``reports/speed_real.json``), so no km/h derived
from it is publishable. Here the ground plane is KNOWN, so the scale is
known, so absolute km/h error is meaningful. This is the product's only
absolute speed claim.

What is truth, and why it is independent
----------------------------------------
The truth is an INPUT. ``simulate_scene`` is handed ``speeds_kmh``; vehicle
``i`` travels at exactly ``speeds_kmh[i % len(speeds_kmh)]`` along a
straight line on the road plane, and its exact world position at frame
``k`` is ``start_y - speed * k / fps`` -- arithmetic in the caller's own
numbers, computed before anything is projected and never revised
afterwards. Nothing the estimator produces feeds back into it, and the
truth is invariant under the seed, the noise level and the box model --
every knob that changes what the estimator is shown. ``tests/
test_simulate.py::test_the_truth_is_an_input_and_is_untouched_by_everything_the_estimator_sees``
asserts exactly that, because a simulator scored against numbers it
derived from the estimator would agree perfectly while measuring nothing.

The camera is a plain pinhole with known focal length, principal point,
height above the road and downward pitch. The ``RoadPlane`` the estimator
uses is NOT the camera's analytic inverse: it is fitted, by
``RoadPlane.from_correspondences``, from surveyed dash centroids on two
divider lines a lane apart -- the same shape of survey a deployment does,
and the same shape the real motorway config used. The fit is exact to
float precision because the true mapping road-plane-to-image IS a
homography, which is the point: it removes calibration error from the
measurement so what remains is the tracking -> homography -> speed chain.

Box models
----------
``box_model="footprint"`` (the default) emits a box whose SIZE is the
projected extent of the 3-D vehicle but whose bottom-centre is exactly the
projection of the vehicle's ground reference point. That isolates the
chain under test: how a detector places its box is a property of the
DETECTOR, and this tier explicitly does not measure the detector.

``box_model="solid"`` emits the true axis-aligned bounding box of the
projected 3-D vehicle, so the bottom-centre anchor carries the real
geometric offset a detector's box would. It is provided so that offset is
measured rather than hidden; on the geometries measured here it moves the
settled error by under 0.1 km/h, because a near-constant world offset does
not change a fitted slope.

Noise
-----
``box_noise_px`` perturbs (centre x, centre y, width, height) with
independent zero-mean Gaussians. It is a four-component vector; a scalar
broadens to all four. The published sweep takes its components from
``reports/detection_noise.json`` via ``noise_from_detection_report`` --
that report labels itself a PROXY for detector box noise and is
heavy-tailed (std exceeds p95 on the centres, because a few large
excursions dominate the variance), so both statistics are offered and
neither is silently preferred. Neither is a measurement of the detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Sequence

import numpy as np

from trafficlens.analytics.speed import SpeedEstimator, time_of_flight_kmh
from trafficlens.core.constants import SPEED_WINDOW_S
from trafficlens.core.geometry import Point
from trafficlens.core.homography import RoadPlane
from trafficlens.detect.base import Detection
from trafficlens.track.tracker import Tracker

#: The statistics ``reports/detection_noise.json`` publishes per residual
#: that can serve as a sigma. ``mae_px`` is deliberately absent: a mean
#: absolute deviation is not a standard deviation and feeding it to a
#: Gaussian would understate the spread by a factor of ~1.25.
MEASURED_NOISE_STATISTICS: tuple[str, ...] = ("std_px", "p95_abs_px")

#: Residual names in that report, in the order the noise vector uses:
#: centre x, centre y, box width, box height.
_NOISE_COMPONENTS: tuple[str, ...] = (
    "centre_x",
    "centre_y",
    "box_width",
    "box_height",
)

#: Box parameters may not be perturbed below this many pixels: a zero- or
#: negative-size box is rejected by ``xyxy_to_xyah`` and is not something a
#: detector emits.
_MIN_BOX_PX = 4.0

#: m/s -> km/h.
_MPS_TO_KMH = 3.6


class SimulationError(RuntimeError):
    """A scene could not be produced in a form that could be scored
    honestly -- most often because two vehicles' boxes overlap, which lets
    the tracker swap identities, and a speed measured across an identity
    swap is not a speed."""


def noise_from_detection_report(
    report: dict, statistic: str
) -> tuple[float, float, float, float]:
    """The (centre x, centre y, width, height) sigma vector, in pixels,
    from a ``reports/detection_noise.json``-shaped dict.

    ``statistic`` must be one of ``MEASURED_NOISE_STATISTICS``. Read
    ``p95_abs_px`` for the typical case and ``std_px`` for the tail -- that
    report's distribution is heavy-tailed, so on the centres the std is the
    LARGER of the two. Treat neither as a measurement of the detector; the
    report labels itself a proxy.
    """
    if statistic not in MEASURED_NOISE_STATISTICS:
        raise ValueError(
            f"statistic must be one of {MEASURED_NOISE_STATISTICS}, got "
            f"{statistic!r}"
        )
    residuals = report["residuals"]
    return tuple(  # type: ignore[return-value]
        float(residuals[name][statistic]) for name in _NOISE_COMPONENTS
    )


def _as_noise_vector(value) -> tuple[float, float, float, float]:
    if isinstance(value, (int, float)):
        return (float(value),) * 4
    components = tuple(float(v) for v in value)
    if len(components) != 4:
        raise ValueError(
            f"box_noise_px must be a scalar or 4 components (centre x, "
            f"centre y, width, height), got {len(components)}"
        )
    if any(c < 0.0 for c in components):
        raise ValueError(f"box_noise_px components must be >= 0, got {components}")
    return components  # type: ignore[return-value]


@dataclass(frozen=True)
class ScenePlane:
    """A known road plane and the known pinhole camera looking down it.

    World frame, matching the engine's convention: ``x`` is metres across
    the road, ``y`` is metres along it away from the camera, ``z`` is
    metres up from the road surface. The camera sits at ``(0, 0,
    height_m)`` looking along ``+y``, pitched ``pitch_deg`` DOWN from
    horizontal. Defaults describe a camera on an overpass above a
    motorway, which is the deployment this product is built for.
    """

    focal_px: float = 1400.0
    principal_x: float = 640.0
    principal_y: float = 360.0
    height_m: float = 8.0
    pitch_deg: float = 12.0
    frame_width: int = 1280
    frame_height: int = 720

    def _rotation(self) -> np.ndarray:
        """World -> camera rotation. Camera axes are the image convention:
        +x right, +y DOWN the image, +z forward along the optical axis."""
        pitch = math.radians(self.pitch_deg)
        sin_p, cos_p = math.sin(pitch), math.cos(pitch)
        return np.array(
            [
                [1.0, 0.0, 0.0],  # right
                [0.0, -sin_p, -cos_p],  # down
                [0.0, cos_p, -sin_p],  # forward
            ]
        )

    def project(self, x: float, y: float, z: float = 0.0) -> Point | None:
        """Image pixel for a world point, or ``None`` when it is behind the
        camera or outside the frame."""
        camera = self._rotation() @ np.array(
            [x - 0.0, y - 0.0, z - self.height_m]
        )
        if camera[2] <= 1e-6:
            return None
        u = self.focal_px * camera[0] / camera[2] + self.principal_x
        v = self.focal_px * camera[1] / camera[2] + self.principal_y
        if not (0.0 <= u <= self.frame_width and 0.0 <= v <= self.frame_height):
            return None
        return (float(u), float(v))

    # -- the survey the estimator's RoadPlane is fitted from ----------------

    #: Dash centroids on two divider lines a lane apart -- the same SHAPE
    #: of survey the real motorway config used, and deliberately not the
    #: camera's analytic inverse. Two lines is the minimum: correspondences
    #: on a single line are collinear and determine no homography at all.
    _SURVEY_WORLD: ClassVar[tuple[Point, ...]] = (
        (0.0, 20.0),
        (0.0, 38.0),
        (0.0, 56.0),
        (3.75, 20.0),
        (3.75, 38.0),
        (3.75, 56.0),
    )
    _HOLDOUT_WORLD: ClassVar[tuple[Point, ...]] = ((0.0, 74.0), (3.75, 74.0))

    def _surveyed(self, world_pts) -> tuple[list[Point], list[Point]]:
        image_pts, kept = [], []
        for x, y in world_pts:
            pixel = self.project(x, y, 0.0)
            if pixel is None:
                raise SimulationError(
                    f"survey point {(x, y)} is not visible in this camera"
                )
            image_pts.append(pixel)
            kept.append((x, y))
        return image_pts, kept

    def survey_points(self) -> tuple[list[Point], list[Point]]:
        """The fit correspondences: (image pixels, world metres)."""
        return self._surveyed(self._SURVEY_WORLD)

    def holdout_points(self) -> tuple[list[Point], list[Point]]:
        """Surveyed points deliberately kept out of the fit, for the
        genuine out-of-sample check ``RoadPlane.validate`` wants."""
        return self._surveyed(self._HOLDOUT_WORLD)

    def road_plane(self) -> RoadPlane:
        """The fitted, VALIDATED ``RoadPlane`` the estimator runs on."""
        image_pts, world_pts = self.survey_points()
        hold_img, hold_world = self.holdout_points()
        plane = RoadPlane.from_correspondences(image_pts, world_pts)
        plane.validate(
            holdout_image_pts=hold_img, holdout_world_pts=hold_world
        )
        return plane


@dataclass(frozen=True)
class Vehicle:
    """One simulated vehicle, travelling TOWARD the camera (decreasing
    world y) at an exactly known constant speed."""

    speed_kmh: float
    lane_x_m: float
    start_y_m: float
    length_m: float = 4.5
    width_m: float = 1.8
    height_m: float = 1.5
    class_name: str = "car"

    release_s: float = 0.0

    def world_y(self, seconds: float) -> float:
        """Exact world y at a time, from the caller's own speed. This is
        the ground truth; it is arithmetic, not a measurement."""
        return self.start_y_m - (self.speed_kmh / _MPS_TO_KMH) * (
            seconds - self.release_s
        )


@dataclass(frozen=True)
class SimulatedScene:
    """A finished scene: the detections, and the exact truth beside them."""

    plane: ScenePlane
    fps: float
    seed: int
    box_noise_px: tuple[float, float, float, float]
    box_model: str
    vehicles: tuple[Vehicle, ...]
    frames: tuple[tuple[Detection, ...], ...]
    #: Per frame, the vehicle index that produced each detection, in the
    #: same order. Used only to attribute a scored track to a vehicle --
    #: never to produce a speed.
    owners: tuple[tuple[int, ...], ...]
    visible_y_m: tuple[float, float]

    def truth_speed_kmh(self, vehicle_index: int) -> float:
        return self.vehicles[vehicle_index].speed_kmh

    def truth_world(self, vehicle_index: int, frame_index: int) -> Point:
        vehicle = self.vehicles[vehicle_index]
        return (vehicle.lane_x_m, vehicle.world_y(frame_index / self.fps))

    def truth_release_s(self, vehicle_index: int) -> float:
        return self.vehicles[vehicle_index].release_s


def _boxes_overlap(a: Detection, b: Detection) -> bool:
    return (
        a.x1 < b.x2 and b.x1 < a.x2 and a.y1 < b.y2 and b.y1 < a.y2
    )


def _project_extent(
    plane: ScenePlane, vehicle: Vehicle, y: float
) -> tuple[float, float, float, float] | None:
    """Axis-aligned image bounding box of the vehicle's eight 3-D corners,
    or ``None`` if any corner leaves the frame."""
    xs, ys = [], []
    for dx in (-vehicle.width_m / 2.0, vehicle.width_m / 2.0):
        for dy in (-vehicle.length_m / 2.0, vehicle.length_m / 2.0):
            for dz in (0.0, vehicle.height_m):
                pixel = plane.project(vehicle.lane_x_m + dx, y + dy, dz)
                if pixel is None:
                    return None
                xs.append(pixel[0])
                ys.append(pixel[1])
    return (min(xs), min(ys), max(xs), max(ys))


def _vehicle_box(
    plane: ScenePlane, vehicle: Vehicle, y: float, box_model: str
) -> tuple[float, float, float, float] | None:
    extent = _project_extent(plane, vehicle, y)
    if extent is None:
        return None
    if box_model == "solid":
        return extent
    if box_model != "footprint":
        raise ValueError(
            f"box_model must be 'footprint' or 'solid', got {box_model!r}"
        )
    ground = plane.project(vehicle.lane_x_m, y, 0.0)
    if ground is None:
        return None
    x1, y1, x2, y2 = extent
    shift_x = ground[0] - (x1 + x2) / 2.0
    shift_y = ground[1] - y2
    return (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)


def simulate_scene(
    plane: ScenePlane,
    speeds_kmh: Sequence[float],
    n_vehicles: int,
    fps: float,
    seed: int,
    box_noise_px,
    *,
    box_model: str = "footprint",
    lane_offsets_m: Sequence[float] = (1.875, 5.625, 9.375),
    lane_release_gap_s: float | None = None,
    visible_y_m: tuple[float, float] = (30.0, 140.0),
    start_y_m: float = 140.0,
) -> SimulatedScene:
    """Place ``n_vehicles`` on ``plane``, move them at exactly the given
    speeds, project them, and emit ``Detection`` objects perturbed by
    ``box_noise_px``.

    Vehicle ``i`` takes ``speeds_kmh[i % len(speeds_kmh)]`` and lane
    ``lane_offsets_m[i % len(lane_offsets_m)]``; each time the lanes wrap,
    the next occupant of a lane is RELEASED only once the previous one has
    left, rather than started further up the road. Staggering in time, not
    in distance, is what makes the guarantee hold for any speed mix: a
    distance gap that separates a 30 km/h leader from a 130 km/h follower
    has to be several hundred metres, and the follower still closes it.
    Each release waits on the LEADER IN THAT LANE specifically -- the gap
    is that leader's own crossing time plus a second -- so one slow vehicle
    does not stretch every other lane's schedule.
    ``lane_release_gap_s`` overrides the computed gap with a fixed one.
    A detection is emitted only
    while the vehicle's ground point lies inside ``visible_y_m`` AND its
    whole box is inside the frame. The scene runs until every vehicle has
    left the near end of that band.

    Raises ``SimulationError`` if any two boxes in a frame overlap: that
    scene cannot be scored honestly (see the class docstring).
    """
    if n_vehicles < 1:
        raise ValueError(f"n_vehicles must be >= 1, got {n_vehicles}")
    if not speeds_kmh:
        raise ValueError("speeds_kmh must name at least one speed")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    sigma = _as_noise_vector(box_noise_px)
    near_y, far_y = min(visible_y_m), max(visible_y_m)

    assigned = [float(speeds_kmh[i % len(speeds_kmh)]) for i in range(n_vehicles)]
    n_lanes = len(lane_offsets_m)

    def crossing_s(speed_kmh: float) -> float:
        return (start_y_m - near_y) / (speed_kmh / _MPS_TO_KMH)

    releases: list[float] = []
    for i in range(n_vehicles):
        if i < n_lanes:
            releases.append(0.0)
            continue
        leader = i - n_lanes  # the previous occupant of this same lane
        gap = (
            crossing_s(assigned[leader]) + 1.0
            if lane_release_gap_s is None
            else float(lane_release_gap_s)
        )
        releases.append(releases[leader] + gap)

    vehicles = tuple(
        Vehicle(
            speed_kmh=assigned[i],
            lane_x_m=float(lane_offsets_m[i % n_lanes]),
            start_y_m=start_y_m,
            release_s=releases[i],
        )
        for i in range(n_vehicles)
    )

    # Long enough for every vehicle to cross the whole band: a fixed frame
    # count would silently truncate a slow one and score it on a shorter
    # track than the fast ones.
    n_frames = int(
        math.ceil(
            max(v.release_s + crossing_s(v.speed_kmh) for v in vehicles) * fps
        )
    ) + 1

    rng = np.random.default_rng(seed)
    frames: list[tuple[Detection, ...]] = []
    owners: list[tuple[int, ...]] = []

    for frame_index in range(n_frames):
        seconds = frame_index / fps
        frame: list[Detection] = []
        frame_owners: list[int] = []
        for index, vehicle in enumerate(vehicles):
            if seconds < vehicle.release_s:
                continue
            y = vehicle.world_y(seconds)
            if not (near_y <= y <= far_y):
                continue
            box = _vehicle_box(plane, vehicle, y, box_model)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if any(s > 0.0 for s in sigma):
                cx = (x1 + x2) / 2.0 + rng.normal(0.0, sigma[0])
                cy = (y1 + y2) / 2.0 + rng.normal(0.0, sigma[1])
                w = max(_MIN_BOX_PX, (x2 - x1) + rng.normal(0.0, sigma[2]))
                h = max(_MIN_BOX_PX, (y2 - y1) + rng.normal(0.0, sigma[3]))
                x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
            frame.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=0.9,
                    class_id=2,
                    class_name=vehicle.class_name,
                )
            )
            frame_owners.append(index)

        for i in range(len(frame)):
            for j in range(i + 1, len(frame)):
                if _boxes_overlap(frame[i], frame[j]):
                    raise SimulationError(
                        f"vehicles {frame_owners[i]} and {frame_owners[j]} "
                        f"overlap in frame {frame_index}: an overlapping pair "
                        f"lets the tracker swap identities, and a speed "
                        f"measured across an identity swap is not a speed"
                    )
        frames.append(tuple(frame))
        owners.append(tuple(frame_owners))

    return SimulatedScene(
        plane=plane,
        fps=float(fps),
        seed=int(seed),
        box_noise_px=sigma,
        box_model=box_model,
        vehicles=vehicles,
        frames=tuple(frames),
        owners=tuple(owners),
        visible_y_m=(near_y, far_y),
    )


# -- scoring -------------------------------------------------------------------


@dataclass(frozen=True)
class SpeedSample:
    """One reported speed, beside the exact truth it is measured against."""

    vehicle: int
    frame_index: int
    age_s: float
    truth_kmh: float
    estimate_kmh: float
    world_y_m: float

    @property
    def error_kmh(self) -> float:
        return self.estimate_kmh - self.truth_kmh

    @property
    def relative_percent(self) -> float:
        return 100.0 * self.error_kmh / self.truth_kmh

    @property
    def settled(self) -> bool:
        """The estimator's window is full. Below this age it fits a slope
        over a partly-filled window while the Kalman filter's velocity is
        still converging from its zero-velocity initialisation, which is a
        start-up transient rather than a steady-state accuracy."""
        return self.age_s >= SPEED_WINDOW_S


@dataclass(frozen=True)
class SceneScore:
    """Everything one scored scene produced."""

    samples: tuple[SpeedSample, ...]
    matched_vehicles: tuple[int, ...]
    bypass_tracker: bool
    tracks_seen: int
    #: Vehicles the tracker never produced a confirmed track for. Recorded
    #: rather than dropped: at high noise the association fails before the
    #: speed chain does, and that is a result about the tracker, not a
    #: missing row.
    lost_vehicles: tuple[int, ...] = field(default_factory=tuple)

    @property
    def settled_samples(self) -> tuple[SpeedSample, ...]:
        return tuple(s for s in self.samples if s.settled)


def _nearest_vehicle(detection: Detection, frame, frame_owners) -> int | None:
    """Attribute a tracked box to the vehicle whose simulated detection it
    overlaps most in this frame. Boxes never overlap each other (the
    simulator refuses such scenes), so this is unambiguous."""
    best, best_iou = None, 0.0
    for detection_index, owner in enumerate(frame_owners):
        other = frame[detection_index]
        ix1 = max(detection.x1, other.x1)
        iy1 = max(detection.y1, other.y1)
        ix2 = min(detection.x2, other.x2)
        iy2 = min(detection.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            continue
        union = (
            (detection.x2 - detection.x1) * (detection.y2 - detection.y1)
            + (other.x2 - other.x1) * (other.y2 - other.y1)
            - inter
        )
        iou = inter / union if union > 0.0 else 0.0
        if iou > best_iou:
            best, best_iou = owner, iou
    return best


def score_scene(scene: SimulatedScene, *, bypass_tracker: bool = False) -> SceneScore:
    """Run the estimator over a scene and pair every reported speed with
    the exact truth.

    ``bypass_tracker=True`` feeds each simulated detection's own
    bottom-centre anchor straight to the ``SpeedEstimator``, skipping the
    Kalman filter. That isolates the homography -> speed half of the chain,
    which is what makes the tracker's contribution attributable rather
    than merely present.
    """
    road = scene.plane.road_plane()
    estimator = SpeedEstimator(road, scene.fps)
    tracker = None if bypass_tracker else Tracker()

    samples: list[SpeedSample] = []
    first_seen: dict[int, float] = {}
    owner_of_track: dict[int, int] = {}
    matched: set[int] = set()

    for frame_index, (frame, frame_owners) in enumerate(
        zip(scene.frames, scene.owners)
    ):
        seconds = frame_index / scene.fps

        if bypass_tracker:
            observations = [
                (owner, ((d.x1 + d.x2) / 2.0, d.y2))
                for d, owner in zip(frame, frame_owners)
            ]
        else:
            observations = []
            for track in tracker.update(list(frame), frame_index):
                box = Detection(
                    *track.box, score=track.score, class_id=0,
                    class_name=track.class_name,
                )
                owner = owner_of_track.get(track.track_id)
                if owner is None:
                    owner = _nearest_vehicle(box, frame, frame_owners)
                    if owner is None:
                        continue
                    owner_of_track[track.track_id] = owner
                observations.append((owner, track.anchor))

        for owner, anchor in observations:
            first_seen.setdefault(owner, seconds)
            estimator.observe(owner, anchor, seconds)
            estimate = estimator.speed_kmh(owner)
            if estimate is None:
                continue
            matched.add(owner)
            samples.append(
                SpeedSample(
                    vehicle=owner,
                    frame_index=frame_index,
                    age_s=seconds - first_seen[owner],
                    truth_kmh=scene.truth_speed_kmh(owner),
                    estimate_kmh=estimate,
                    world_y_m=scene.truth_world(owner, frame_index)[1],
                )
            )

    return SceneScore(
        samples=tuple(samples),
        matched_vehicles=tuple(sorted(matched)),
        bypass_tracker=bypass_tracker,
        tracks_seen=len(first_seen) if bypass_tracker else len(owner_of_track),
        lost_vehicles=tuple(
            i for i in range(len(scene.vehicles)) if i not in matched
        ),
    )


# -- Check C: the two estimators, checked against each other --------------------


@dataclass(frozen=True)
class TimeOfFlightScore:
    """One vehicle's two independent speeds, and the exact truth."""

    vehicle: int
    truth_kmh: float
    time_of_flight_kmh: float
    homography_kmh: float
    t_far_s: float
    t_near_s: float
    gate_separation_m: float

    @property
    def difference_kmh(self) -> float:
        return self.time_of_flight_kmh - self.homography_kmh


def time_of_flight_scores(
    scene: SimulatedScene, *, gate_far_y_m: float, gate_near_y_m: float
) -> list[TimeOfFlightScore]:
    """Check C on the SIMULATED scene, as an instrument check.

    Two gates a known ground distance apart give a time-of-flight speed per
    vehicle that shares no per-frame displacement with the homography
    estimate: it is ``(distance / (t_near - t_far)) * 3.6`` from two
    crossing instants and the surveyed separation, nothing else. The
    crossing instants come from the plane-projected track position, linearly
    interpolated between the two frames that straddle the gate.

    This is NOT run on real footage, and cannot be: there, the ground
    distance between two gates is itself known only through the along-road
    scale that ``reports/speed_real.json`` shows has no independent anchor,
    so the check would be calibrated by the quantity it was meant to check.
    On the simulated scene both estimators share a KNOWN scale, so their
    agreement tests the two estimators against each other, which is a real
    result and the one Check C was wanted for.
    """
    if gate_far_y_m <= gate_near_y_m:
        raise ValueError(
            f"gate_far_y_m must be further from the camera than gate_near_y_m, "
            f"got {gate_far_y_m} and {gate_near_y_m}"
        )
    separation = gate_far_y_m - gate_near_y_m

    road = scene.plane.road_plane()
    tracker = Tracker()
    estimator = SpeedEstimator(road, scene.fps)
    owner_of_track: dict[int, int] = {}
    trace: dict[int, list[tuple[float, float]]] = {}
    last_estimate: dict[int, float] = {}

    for frame_index, (frame, frame_owners) in enumerate(
        zip(scene.frames, scene.owners)
    ):
        seconds = frame_index / scene.fps
        for track in tracker.update(list(frame), frame_index):
            box = Detection(
                *track.box, score=track.score, class_id=0,
                class_name=track.class_name,
            )
            owner = owner_of_track.get(track.track_id)
            if owner is None:
                owner = _nearest_vehicle(box, frame, frame_owners)
                if owner is None:
                    continue
                owner_of_track[track.track_id] = owner
            _, world_y = road.to_world(track.anchor)
            trace.setdefault(owner, []).append((seconds, world_y))
            estimator.observe(owner, track.anchor, seconds)
            estimate = estimator.speed_kmh(owner)
            if estimate is not None:
                last_estimate[owner] = estimate

    def crossing(points, gate_y) -> float | None:
        """Time the track's world y passes ``gate_y``, linearly
        interpolated between the straddling pair. Vehicles approach, so y
        decreases."""
        for (t0, y0), (t1, y1) in zip(points, points[1:]):
            if y0 >= gate_y >= y1 and y0 != y1:
                return t0 + (y0 - gate_y) / (y0 - y1) * (t1 - t0)
        return None

    results: list[TimeOfFlightScore] = []
    for owner in sorted(trace):
        points = trace[owner]
        t_far = crossing(points, gate_far_y_m)
        t_near = crossing(points, gate_near_y_m)
        if t_far is None or t_near is None or owner not in last_estimate:
            continue
        results.append(
            TimeOfFlightScore(
                vehicle=owner,
                truth_kmh=scene.truth_speed_kmh(owner),
                time_of_flight_kmh=time_of_flight_kmh(t_far, t_near, separation),
                homography_kmh=last_estimate[owner],
                t_far_s=t_far,
                t_near_s=t_near,
                gate_separation_m=separation,
            )
        )
    return results
