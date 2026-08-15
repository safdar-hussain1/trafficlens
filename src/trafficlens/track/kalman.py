"""Constant-velocity Kalman filter over bounding boxes in xyah space.

State vector (8,): ``[cx, cy, a, h, vx, vy, va, vh]`` -- box centre x/y,
aspect ratio ``a = w/h``, height ``h``, and their per-frame velocities.
Measurement vector (4,): ``[cx, cy, a, h]`` -- the same box parameters as
observed by the detector (see ``xyxy_to_xyah`` for the conversion from a
``Detection``'s corner format).

Design rules, all of which the multi-object tracker and the later
TypeScript mirror depend on:

- ``KalmanBoxFilter`` holds NO per-track state. It carries only the
  constant transition/measurement matrices; ``initiate``, ``predict``,
  ``update`` and ``gating_distance`` are pure functions of their
  ``(mean, cov)`` arguments and always return new arrays. The tracker owns
  one ``(mean, cov)`` pair per track.
- All noise magnitudes scale with the box height ``h``: a taller box is a
  nearer object, which moves more pixels per frame, so its uncertainty in
  pixels is proportionally larger. Every scale factor is a named constant
  in ``trafficlens.core.constants`` -- this module contains no numeric
  tunables of its own.
- Deterministic: no randomness anywhere; the same input sequence produces
  bit-identical output on every run.
- numpy + standard library only. No scipy: linear systems are solved with
  ``np.linalg.solve`` / ``np.linalg.cholesky``, both of which the
  TypeScript mirror can reproduce with a small hand-written LU/Cholesky.
"""

from __future__ import annotations

import numpy as np

from trafficlens.core.constants import (
    KALMAN_ASPECT_MEASUREMENT_STD,
    KALMAN_ASPECT_STD,
    KALMAN_ASPECT_VELOCITY_STD,
    KALMAN_INIT_POSITION_STD_FACTOR,
    KALMAN_INIT_VELOCITY_STD_FACTOR,
    KALMAN_STD_WEIGHT_POSITION,
    KALMAN_STD_WEIGHT_VELOCITY,
)

# State/measurement dimensions (structural, not tunable).
_NDIM = 4  # measured components: cx, cy, a, h
_DT = 1.0  # one frame per step; frame index is the filter's clock


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    """Convert a corner-format box ``[x1, y1, x2, y2]`` to measurement
    format ``[cx, cy, a, h]`` with ``a = w/h``.

    Raises ``ValueError`` on a zero- or negative-area box (``x2 <= x1`` or
    ``y2 <= y1``). Fail fast is deliberate: a detector emitting a
    degenerate box is an upstream bug, and silently computing ``a = w/h``
    with ``h <= 0`` would push NaN/inf (or a nonsense negative aspect)
    into the filter and poison the tracker invisibly.
    """
    box = np.asarray(box, dtype=np.float64)
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1
    if w <= 0.0 or h <= 0.0:
        raise ValueError(
            f"degenerate box: width={w}, height={h} (from xyxy={box.tolist()}); "
            "both must be strictly positive"
        )
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h])


def xyah_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, a, h]`` back to ``[x1, y1, x2, y2]``.

    Exact inverse of ``xyxy_to_xyah`` up to float64 rounding (the pair is
    tested to round-trip within 1e-9).
    """
    box = np.asarray(box, dtype=np.float64)
    cx, cy, a, h = box[0], box[1], box[2], box[3]
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def _symmetrize(cov: np.ndarray) -> np.ndarray:
    """Force exact symmetry: ``(P + P.T) / 2``.

    Floating-point matrix products make ``P[i, j]`` and ``P[j, i]`` drift
    apart by a few ULPs per step; left alone, that asymmetry compounds and
    is the classic silent Kalman failure mode (a covariance that is no
    longer a valid covariance, and Cholesky factorizations that start to
    disagree between platforms). Averaging with the transpose restores
    exact ``P == P.T`` after every step; the TypeScript mirror must do the
    same or the two implementations diverge bit by bit.
    """
    return (cov + cov.T) / 2.0


class KalmanBoxFilter:
    """Stateless constant-velocity Kalman filter for xyah boxes.

    Instances hold only the constant transition matrix ``F`` (identity plus
    dt on the position->velocity diagonal) and measurement matrix ``H``
    (selects the first four state components). All per-track state flows
    through the method arguments.
    """

    def __init__(self) -> None:
        self._motion_mat = np.eye(2 * _NDIM)
        for i in range(_NDIM):
            self._motion_mat[i, _NDIM + i] = _DT
        self._update_mat = np.eye(_NDIM, 2 * _NDIM)

    # -- noise profiles ----------------------------------------------------

    @staticmethod
    def _std_position(h: float) -> list[float]:
        """Per-component std of the position block [cx, cy, a, h] at box
        height ``h``: pixel components scale with h, aspect is absolute."""
        return [
            KALMAN_STD_WEIGHT_POSITION * h,
            KALMAN_STD_WEIGHT_POSITION * h,
            KALMAN_ASPECT_STD,
            KALMAN_STD_WEIGHT_POSITION * h,
        ]

    @staticmethod
    def _std_velocity(h: float) -> list[float]:
        """Per-component std of the velocity block [vx, vy, va, vh]."""
        return [
            KALMAN_STD_WEIGHT_VELOCITY * h,
            KALMAN_STD_WEIGHT_VELOCITY * h,
            KALMAN_ASPECT_VELOCITY_STD,
            KALMAN_STD_WEIGHT_VELOCITY * h,
        ]

    # -- filter steps ------------------------------------------------------

    def initiate(self, box_xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a new track from an unassociated measurement.

        Returns ``(mean, cov)``: mean (8,) with the measured box and zero
        velocities; cov (8, 8) diagonal, with position variances inflated
        by ``KALMAN_INIT_POSITION_STD_FACTOR`` (a single detection is less
        trustworthy than a tracked state) and velocity variances inflated
        by ``KALMAN_INIT_VELOCITY_STD_FACTOR`` (zero velocity is pure
        ignorance, so the first measurements must dominate it).
        """
        box_xyah = np.asarray(box_xyah, dtype=np.float64)
        mean = np.zeros(2 * _NDIM)
        mean[:_NDIM] = box_xyah

        h = float(box_xyah[3])
        std = np.array(
            [KALMAN_INIT_POSITION_STD_FACTOR * s for s in self._std_position(h)]
            + [KALMAN_INIT_VELOCITY_STD_FACTOR * s for s in self._std_velocity(h)]
        )
        cov = np.diag(std * std)
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One constant-velocity step: ``x' = F x``, ``P' = F P F^T + Q``.

        The process noise ``Q`` is diagonal and re-derived each step from
        the CURRENT height ``mean[3]``, so a box growing as it approaches
        the camera automatically gets a wider motion envelope.
        """
        h = float(mean[3])
        std = np.array(self._std_position(h) + self._std_velocity(h))
        process_noise = np.diag(std * std)

        new_mean = self._motion_mat @ mean
        new_cov = self._motion_mat @ cov @ self._motion_mat.T + process_noise
        # Symmetrized every step (see _symmetrize) so predict/update chains
        # keep P exactly symmetric no matter how they interleave.
        return new_mean, _symmetrize(new_cov)

    def _project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space:
        ``(H x, H P H^T + R)`` with height-scaled measurement noise R."""
        h = float(mean[3])
        std = np.array(
            [
                KALMAN_STD_WEIGHT_POSITION * h,
                KALMAN_STD_WEIGHT_POSITION * h,
                KALMAN_ASPECT_MEASUREMENT_STD,
                KALMAN_STD_WEIGHT_POSITION * h,
            ]
        )
        measurement_noise = np.diag(std * std)

        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ cov @ self._update_mat.T + measurement_noise
        return proj_mean, _symmetrize(proj_cov)

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement_xyah: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standard Kalman correction with an xyah measurement.

        The gain is computed with ``np.linalg.solve`` on the innovation
        covariance ``S`` -- NEVER an explicit matrix inverse -- because
        solving the linear system directly is better conditioned and is
        the numerical approach the TypeScript mirror must reproduce
        (a small LU solve) for bit-comparable results.
        """
        measurement_xyah = np.asarray(measurement_xyah, dtype=np.float64)
        proj_mean, proj_cov = self._project(mean, cov)

        # Kalman gain K = P H^T S^{-1}, obtained by solving S K^T = (P H^T)^T.
        b = cov @ self._update_mat.T  # (8, 4)
        kalman_gain = np.linalg.solve(proj_cov, b.T).T  # (8, 4)

        innovation = measurement_xyah - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = cov - kalman_gain @ proj_cov @ kalman_gain.T
        # Enforce exact symmetry: the subtraction above loses a few ULPs of
        # P == P.T per step, and that drift compounding silently is the
        # classic Kalman covariance failure (see _symmetrize).
        return new_mean, _symmetrize(new_cov)

    def gating_distance(
        self, mean: np.ndarray, cov: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Squared Mahalanobis distance of N xyah measurements from the
        state's predicted measurement distribution.

        ``measurements`` is (N, 4); returns (N,). Distances follow a
        chi-square distribution with 4 degrees of freedom, so the tracker
        gates associations at ``KALMAN_GATING_CHI2_95_4DOF`` (9.4877, the
        95% quantile): any pair scoring above it is rejected as having
        under a 5% chance of being the same object.

        Computed via Cholesky (``S = L L^T``) and a triangular solve --
        never by inverting S -- so each distance is a plain sum of squares
        ``||L^{-1} d||^2``; the TypeScript mirror uses the same
        factorization.
        """
        measurements = np.asarray(measurements, dtype=np.float64)
        proj_mean, proj_cov = self._project(mean, cov)

        d = measurements - proj_mean  # (N, 4)
        chol = np.linalg.cholesky(proj_cov)  # lower-triangular L
        # Solve L y = d^T column-wise; squared Mahalanobis is sum(y^2).
        y = np.linalg.solve(chol, d.T)  # (4, N)
        return np.sum(y * y, axis=0)
