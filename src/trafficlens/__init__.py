"""TrafficLens — real-time traffic analytics from any camera.

Detects, tracks, counts, and speed-estimates objects (vehicles, people,
bicycles — any COCO class) in video files, webcams, or RTSP streams.
Counting uses segment-intersection geometry against user-defined virtual
gates; speeds come from a homography that maps the image road plane to
real-world metres.
"""

__version__ = "1.0.0"

from trafficlens.config import AppConfig, CalibrationConfig, GateConfig, load_config
from trafficlens.counting import CrossingEvent, Gate, GateCounter
from trafficlens.speed import SpeedEstimator
from trafficlens.pipeline import Pipeline, FrameResult

__all__ = [
    "AppConfig",
    "CalibrationConfig",
    "GateConfig",
    "load_config",
    "CrossingEvent",
    "Gate",
    "GateCounter",
    "SpeedEstimator",
    "Pipeline",
    "FrameResult",
    "__version__",
]
