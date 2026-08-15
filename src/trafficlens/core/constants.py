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
