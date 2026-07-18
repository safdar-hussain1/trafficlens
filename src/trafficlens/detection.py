"""Detection + tracking via Ultralytics YOLO with ByteTrack/BoT-SORT.

Class names are resolved against ``model.names`` at load time — the
model's own vocabulary is the source of truth, never a hardcoded list
(hardcoded COCO lists drift: index 3 is "motorcycle", and a list that
says "bike" there silently filters the wrong class).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from trafficlens.config import DetectorConfig
from trafficlens.geometry import Point


@dataclass(frozen=True)
class Observation:
    """One tracked object on one frame."""

    track_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 pixels

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the box — the object's ground-contact point.

        Counting and speed both operate on the road plane, so the anchor
        must be the point where the object meets the road, not the box
        centre (which sits mid-air and shifts with perspective).
        """
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)


def resolve_device(requested: str = "auto") -> str:
    """Pick the best available compute device, honouring an explicit choice."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    """YOLO detector with persistent multi-object tracking."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.model = YOLO(config.model)
        self.names: dict[int, str] = dict(self.model.names)
        name_to_id = {v: k for k, v in self.names.items()}
        unknown = [c for c in config.classes if c not in name_to_id]
        if unknown:
            raise ValueError(
                f"classes {unknown} not in model vocabulary; available: "
                f"{sorted(name_to_id)}"
            )
        self.class_ids = [name_to_id[c] for c in config.classes]

    def track(self, frame: np.ndarray) -> list[Observation]:
        """Run detection+tracking on one frame, filtered to configured classes."""
        results = self.model.track(
            frame,
            persist=True,
            conf=self.config.confidence,
            classes=self.class_ids,
            imgsz=self.config.imgsz,
            tracker=f"{self.config.tracker}.yaml",
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        out: list[Observation] = []
        ids = boxes.id.int().tolist()
        clss = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()
        for tid, cid, conf, box in zip(ids, clss, confs, xyxy):
            out.append(
                Observation(
                    track_id=int(tid),
                    class_name=self.names[int(cid)],
                    confidence=float(conf),
                    box=tuple(float(v) for v in box),
                )
            )
        return out

    def reset_tracker(self) -> None:
        """Start tracking fresh (e.g. when a video source loops or changes)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
