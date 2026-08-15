"""Tracking layer: per-box Kalman filtering and multi-object association.

``trafficlens.track.kalman`` (the constant-velocity box filter and the
xyxy/xyah conversions) imports only the standard library and numpy so it
can be mechanically mirrored to TypeScript for the browser engine. The
association step (Hungarian assignment) lives in a separate module and is
the only place scipy enters the tracking layer.
"""
