#!/usr/bin/env bash
# Smoke test: verify the C++ -> Python -> Dedalus pipeline works.
#
# Runs findsoln with -Nn 1 (1 Newton iteration) to exercise the full chain:
#   C++ findsoln -> Py_Initialize() -> PyImport_Import("LLE") -> Dedalus time integration
#
# Required environment:
#   FINDSOLN         - path to findsoln binary
#   RANDOMDEDAFIELD  - path to randomdedafield binary
#   PYTHONPATH       - must include the dedalus/ directory with LLE.py
#
# Exit code 0 = pass, non-zero = fail.

set -euo pipefail

FINDSOLN="${FINDSOLN:?ERROR: FINDSOLN not set — point to findsoln binary}"
RANDOMDEDAFIELD="${RANDOMDEDAFIELD:?ERROR: RANDOMDEDAFIELD not set — point to randomdedafield binary}"

export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

# Create a temporary sil.nc file
PSI_NC="psi.nc"
trap "rm -f $PSI_NC" EXIT

echo "dedalus_findsoln_smoke: creating test data with randomdedafield -Nd 1 -Nvar 1 -N 512 -L 3 $PSI_NC"
${RANDOMDEDAFIELD} -Nd 1 -Nvar 1 -N 512 -L 3 "$PSI_NC"

echo "dedalus_findsoln_smoke: running findsoln -eqb -T 1 -Nn 1 -sys LLE $PSI_NC"
OUTPUT=$(${FINDSOLN} -eqb -T 1 -Nn 1 -sys LLE "$PSI_NC" 2>&1)
echo "$OUTPUT"

# Verify Newton iteration actually ran (not just a silent failure)
if ! echo "$OUTPUT" | grep -q "L2Norm"; then
    echo "dedalus_findsoln_smoke: FAIL -- no L2Norm output (Newton iteration did not run)" >&2
    exit 1
fi

# Check for NaN in the output (indicates numerical blowup)
if echo "$OUTPUT" | grep -qi "nan"; then
    echo "dedalus_findsoln_smoke: FAIL -- NaN detected in output" >&2
    exit 1
fi

echo "dedalus_findsoln_smoke: PASS"
