"""Smoke test: verify Dedalus system can be instantiated from Python.

This test checks that:
  1. The dedalus Python package is importable
  2. The LLE.DedalusPy class can be constructed
  3. The to_vector() method returns a non-empty numpy array

Exit code 0 = pass, non-zero = fail.
"""
import sys

try:
    from LLE import DedalusPy

    dd = DedalusPy()
    vec = dd.to_vector()
    assert len(vec) > 0, "to_vector() returned empty array"

    print("dedalus_python_smoke: PASS")
    sys.exit(0)

except Exception as e:
    print(f"dedalus_python_smoke: FAIL -- {e}", file=sys.stderr)
    sys.exit(1)
