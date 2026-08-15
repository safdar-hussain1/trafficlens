"""Video frame sources: files, webcams and network streams behind one
iterator interface.

``VideoSource.open(spec)`` decides what a spec means with ``classify_spec``:
an int or all-digit string is a webcam index, a URL with an rtsp/http/https
scheme is a network stream, anything else is a file path that must exist.
Iterating a source yields ``(frame_index, timestamp_s, frame)`` tuples.

Timestamp policy, per source kind:

- **file**: ``timestamp_s = frame_index / fps`` with fps read from the
  container at open time. A variable-frame-rate file is therefore
  approximated by its container's nominal rate -- the honest limit of what
  cv2 exposes portably. A file whose container reports no frame rate at
  all is refused at open (``SourceError``), because every downstream
  timestamp would be a guess.
- **webcam**: wall-clock seconds since the first frame (0.0 for the first
  frame), because a live camera has no container rate worth trusting and
  wall time is the ground truth of when each frame actually arrived.
- **stream**: container fps when the stream reports a positive one
  (``frame_index / fps``, same VFR approximation as files), wall clock
  otherwise.

Errors are ``SourceError`` with actionable messages: a missing file names
the path (and points at ``trafficlens fetch-samples`` when it lives under
``data/samples/``), an existing file cv2 cannot decode names the codec
possibility, a webcam index or stream that will not open says exactly which
one failed.
"""

import time
from pathlib import Path

import cv2


class SourceError(RuntimeError):
    """A video source could not be opened or read. The message says which
    source, and what to do about it."""


def classify_spec(spec) -> tuple[str, object]:
    """Decide what a source spec means, without touching any hardware.

    Returns one of ``("webcam", index: int)``, ``("stream", url: str)`` or
    ``("file", path: str)``. Split out from ``VideoSource.open`` so the
    dispatch rule is testable with no camera, network or file present.
    """
    if isinstance(spec, int) and not isinstance(spec, bool):
        return ("webcam", spec)
    text = str(spec)
    if text.isdigit():
        return ("webcam", int(text))
    lowered = text.lower()
    if lowered.startswith(("rtsp://", "http://", "https://")):
        return ("stream", text)
    return ("file", text)


class VideoSource:
    """One opened video source. Build with ``VideoSource.open``, iterate
    for ``(frame_index, timestamp_s, frame)``, and close -- ideally via
    ``with`` -- when done."""

    def __init__(self, capture: cv2.VideoCapture, kind: str, spec: str) -> None:
        self._capture = capture
        self.kind = kind
        self.spec = spec
        self._closed = False
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._fps: float | None = fps if fps > 0 else None

    @classmethod
    def open(cls, spec) -> "VideoSource":
        """Open a webcam index, stream URL or file path (see
        ``classify_spec`` for how a spec is read)."""
        kind, value = classify_spec(spec)
        if kind == "webcam":
            capture = cv2.VideoCapture(value)
            if not capture.isOpened():
                capture.release()
                raise SourceError(
                    f"webcam index {value} did not open. Is a camera "
                    f"connected, is the index right (0 is the first "
                    f"camera), and does this process have camera permission?"
                )
            return cls(capture, kind, str(value))

        if kind == "stream":
            capture = cv2.VideoCapture(value)
            if not capture.isOpened():
                capture.release()
                raise SourceError(
                    f"stream {value} did not open. Check that the URL is "
                    f"reachable from this machine, that any credentials are "
                    f"embedded in it, and that the server is up."
                )
            return cls(capture, kind, value)

        path = Path(value)
        if not path.is_file():
            hint = ""
            if "data/samples" in path.as_posix():
                hint = " Run `trafficlens fetch-samples` to download the sample clips."
            raise SourceError(f"video file not found: {value}.{hint}")
        capture = cv2.VideoCapture(str(path))
        opened = capture.isOpened()
        if opened:
            # Some builds report isOpened() for a container they cannot
            # actually decode; probe one frame so the failure surfaces here,
            # at open, instead of as a silently empty iteration.
            ok, _ = capture.read()
            if ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            opened = ok
        if not opened:
            capture.release()
            raise SourceError(
                f"cv2 could not decode {value}. The file exists but its "
                f"container or codec is unsupported by this OpenCV build; "
                f"transcoding it (e.g. with ffmpeg) to H.264 MP4 or VP9 "
                f"WebM usually fixes this."
            )
        source = cls(capture, kind, str(path))
        if source._fps is None:
            capture.release()
            raise SourceError(
                f"{value} reports no frame rate in its container, so frame "
                f"timestamps cannot be derived. Remux or transcode the file "
                f"so it carries one."
            )
        return source

    # --- properties -----------------------------------------------------------

    @property
    def fps(self) -> float | None:
        """Container frame rate; None when the source does not report one
        (common for webcams and some streams). Always a positive float for
        files -- open() refuses a file without one."""
        return self._fps

    @property
    def width(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int | None:
        """Total frames for a file; None for webcams and streams, whose
        length is unknowable up front."""
        if self.kind != "file":
            return None
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None

    # --- iteration ------------------------------------------------------------

    def __iter__(self):
        """Yield ``(frame_index, timestamp_s, frame)`` until the source
        ends. See the module docstring for the per-kind timestamp policy."""
        if self._closed:
            raise SourceError(
                f"source {self.spec} is closed; open a new VideoSource to "
                f"read it again"
            )
        use_wall_clock = self.kind == "webcam" or self._fps is None
        first_frame_monotonic: float | None = None
        frame_index = 0
        while True:
            if self._closed:
                raise SourceError(
                    f"source {self.spec} was closed mid-iteration"
                )
            ok, frame = self._capture.read()
            if not ok:
                return
            if use_wall_clock:
                now = time.monotonic()
                if first_frame_monotonic is None:
                    first_frame_monotonic = now
                timestamp_s = now - first_frame_monotonic
            else:
                timestamp_s = frame_index / self._fps
            yield frame_index, timestamp_s, frame
            frame_index += 1

    # --- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
