"""Detection layer: shared preprocessing/decoding plus detector adapters.

``trafficlens.detect.base`` (letterbox, NMS, decode_yolo, the ``Detection``
dataclass and ``Detector`` protocol) is standard-library + numpy + OpenCV
only. The two adapter modules, ``trafficlens.detect.ultralytics_yolo`` and
``trafficlens.detect.onnx_yolo``, import their heavy backend (torch via
ultralytics, or onnxruntime) lazily inside the adapter class's
``__init__``, never at module import time -- so importing this package
itself, or either adapter module, never pulls in torch, ultralytics, or
onnxruntime. This file intentionally re-exports nothing, so that importing
it can never accidentally import an adapter module either.
"""
