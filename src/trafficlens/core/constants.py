"""Machine-parseable constants shared across the Python core and, later, a
generated TypeScript mirror for the browser engine.

This file is parsed with Python's ``ast`` module by a separate script, so it
must contain only module-level ``UPPER_CASE = <literal>`` assignments (int,
float, or str literals) plus comments and this docstring. No imports, no
computed expressions, no tuples, no f-strings. Anything that needs to be
computed belongs at its point of use, not here.
"""

# Tolerance for geometric comparisons (cross-product sign, denominator
# magnitude) in trafficlens.core.geometry. Coordinates are pixel positions,
# so 1e-9 is far below any sub-pixel noise a detector or tracker could ever
# produce -- it only guards against floating-point rounding, not real
# ambiguity in the input.
GEOMETRY_EPS = 1e-9

# Maximum acceptable condition number of the (Hartley-normalized) DLT design
# matrix used by trafficlens.core.homography.RoadPlane.validate() to detect
# a degenerate or near-degenerate correspondence set (collinear points, a
# duplicated point, or any configuration close to one). Measured on real
# synthetic configurations: a healthy trapezoid of 4 surveyed road points
# gives a condition number of about 8; nudging one point 1mm toward
# collinear with the other three pushes it to about 56000; exactly collinear
# or duplicated points push it above 1e16. 10000.0 sits comfortably between
# the healthy and near-degenerate measurements (three orders of magnitude of
# margin on each side), so it rejects genuine near-degeneracy without ever
# rejecting a well-spread survey.
HOMOGRAPHY_MAX_CONDITION_NUMBER = 10000.0

# Default maximum acceptable mean reprojection error, in metres, for
# RoadPlane.validate() to accept a calibration. 0.5m mirrors the threshold a
# corrupted single-point survey error is expected to exceed: a mean error
# this large means an object's reported plane position -- and therefore its
# speed -- cannot be trusted to better than half a metre, which is too
# coarse for confident lane-level speed enforcement.
HOMOGRAPHY_MAX_MEAN_ERROR_M = 0.5
