"""Benchmarking and ground-truth tooling.

Two kinds of module live here and the distinction is load-bearing.

**Yardsticks** -- ``slitscan`` -- supply the truth a measurement is made
against, so they may depend on nothing the engine judges: no detector, no
tracker, no counter. A yardstick cut from the thing it measures reports
only that the thing agrees with itself.

**Contenders** -- ``baselines`` -- are alternative engines, measured by
those yardsticks alongside the real one. A contender is *supposed* to
share the parts of the pipeline that are not under test, so that a
measured difference can be attributed to the part that is; ``baselines``
therefore imports the engine's gate geometry, anchor policy and track
types on purpose, and its module docstring writes out exactly what each
baseline shares and what it does not. The rule is that no contender may
share anything with the YARDSTICK, which none of them does.
"""
