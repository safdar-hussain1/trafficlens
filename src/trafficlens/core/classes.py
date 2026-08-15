"""COCO class names and the vehicle subset trafficlens counts by default.

``COCO_CLASSES`` is the 80-name tuple in the exact index order YOLO11 (and
every other ultralytics COCO-pretrained checkpoint) outputs class scores
in -- index ``i`` in a decoded score row *is* ``COCO_CLASSES[i]``. It is
transcribed here rather than imported from ultralytics (this module must
stay dependency-free, standard library only, so it can be imported without
torch/ultralytics present) but is verified byte-for-byte against a loaded
checkpoint's own ``model.names`` mapping in
``tests/test_detect.py::test_coco_classes_matches_ultralytics_names_mapping``,
not typed from memory.

``VEHICLE_CLASSES`` is the subset of ``COCO_CLASSES`` this project treats
as road traffic to count and track: wheeled vehicles a road camera actually
sees crossing a gate. It deliberately excludes ``"train"``, ``"airplane"``
and ``"boat"`` -- COCO categories that are technically vehicles but never
appear as road traffic in a gate-counting scene -- so a camera pointed at a
level crossing or a bridge doesn't start counting trains as cars.

This module imports nothing beyond the standard library.
"""

COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

VEHICLE_CLASSES: tuple[str, ...] = ("bicycle", "car", "motorcycle", "bus", "truck")


def class_ids(names) -> set[int]:
    """Resolve an iterable of COCO class names to their ``COCO_CLASSES``
    indices.

    Raises ``ValueError`` naming the offending class -- and listing every
    valid option -- the moment an unrecognised name is seen, rather than
    silently returning a set that quietly detects nothing. A typo in a
    config's class list (e.g. ``"trucks"`` instead of ``"truck"``) is a
    configuration bug that must fail loudly at load time, not a
    detector that mysteriously never fires.
    """
    lookup = {name: index for index, name in enumerate(COCO_CLASSES)}
    ids: set[int] = set()
    for name in names:
        if name not in lookup:
            raise ValueError(
                f"Unknown class name {name!r}. Valid class names are: "
                f"{', '.join(COCO_CLASSES)}"
            )
        ids.add(lookup[name])
    return ids
