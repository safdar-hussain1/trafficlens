"""Video input/output with fail-fast validation.

Sources: video file path, webcam index ("0"), or RTSP/HTTP URL — the
same string a traffic-camera deployment would put in a config file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class SourceInfo:
    width: int
    height: int
    fps: float
    frame_count: int | None  # None for live sources
    is_live: bool


class VideoSource:
    """Wraps cv2.VideoCapture with validation and clean iteration."""

    def __init__(self, source: str):
        self.source = source
        self._is_stream = source.startswith(("rtsp://", "http://", "https://"))
        self._is_camera = source.isdigit()
        if not self._is_stream and not self._is_camera and not Path(source).exists():
            raise FileNotFoundError(
                f"video source not found: {source!r} — expected a file path, "
                "a webcam index like '0', or an rtsp:// / http(s):// URL"
            )
        self._cap = cv2.VideoCapture(int(source) if self._is_camera else source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video source {source!r}")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            self._cap.release()
            raise RuntimeError(f"source {source!r} opened but reports invalid size {w}x{h}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info = SourceInfo(
            width=w,
            height=h,
            # Live cameras often report 0 fps; timestamps then come from
            # the wall clock instead of frame_index / fps.
            fps=fps if fps and fps > 0 else 0.0,
            frame_count=n if n > 0 else None,
            is_live=self._is_stream or self._is_camera,
        )

    def frames(self):
        """Yield (frame_index, frame) until the source ends."""
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return
            yield idx, frame
            idx += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoWriter:
    """MP4 writer that inherits the source's real fps (not a magic 15.0)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fps <= 0:
            fps = 30.0  # sensible default for live sources that report 0
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer at {path}")
        self.path = path
        self._size = (width, height)

    def write(self, frame) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            raise ValueError(f"frame size {w}x{h} does not match writer size {self._size}")
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
