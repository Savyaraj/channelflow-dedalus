#!/usr/bin/env bash
# install.sh — Automated installer for channelflow-dedalus
#
# Usage:
#   mamba create -n channelflow -c conda-forge --strict-channel-priority python=3.10 dedalus compilers eigen cmake libnetcdf netcdf4
#   conda activate channelflow
#   bash scripts/install.sh
#
# Options:
#   --no-test     Skip running tests after build
#   --clean       Remove build directory before configuring

set -euo pipefail

# ---- Parse arguments ----
RUN_TESTS=true
CLEAN_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --no-test)  RUN_TESTS=false ;;
        --clean)    CLEAN_BUILD=true ;;
        --help|-h)
            echo "Usage: bash scripts/install.sh [--no-test] [--clean]"
            echo "  --no-test  Skip running tests after build"
            echo "  --clean    Remove build directory before configuring"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ---- Paths ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
INSTALL_PREFIX="${PROJECT_DIR}/install"
NPROC="$(nproc 2>/dev/null || echo 4)"

echo "============================================"
echo "  channelflow-dedalus installer"
echo "============================================"
echo ""

# ---- Check conda environment ----
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: No conda environment detected."
    echo ""
    echo "Setup:"
    echo "  mamba create -n channelflow -c conda-forge --strict-channel-priority python=3.10 dedalus compilers eigen cmake libnetcdf netcdf4"
    echo "  conda activate channelflow"
    echo "  bash scripts/install.sh"
    echo ""
    exit 1
fi

echo "==> Detected conda environment: $CONDA_PREFIX"

# Verify Dedalus is installed
if ! python3 -c "import dedalus" 2>/dev/null; then
    echo ""
    echo "ERROR: Dedalus not found in conda environment."
    echo "Install it first:"
    echo "  conda install -c conda-forge dedalus"
    echo ""
    exit 1
fi

# Use conda's compilers and libraries
CC="${CONDA_PREFIX}/bin/mpicc"
CXX="${CONDA_PREFIX}/bin/mpicxx"
CMAKE_PREFIX="${CONDA_PREFIX}"
PYTHON_EXE="${CONDA_PREFIX}/bin/python3"

# Check if mpicc exists in conda env
if [ ! -f "$CC" ]; then
    echo "ERROR: mpicc not found in conda env. Install MPI compilers:"
    echo "  conda install -c conda-forge compilers"
    exit 1
fi

# Ensure compilers, eigen3 and cmake are available
echo "==> Checking for compilers, eigen3 and cmake in conda env..."
conda install -y -c conda-forge compilers eigen cmake libnetcdf netcdf4 2>/dev/null || {
    echo "ERROR: Could not install required packages via conda."
    exit 1
}

echo ""
echo "==> CC:     ${CC}"
echo "    CXX:    ${CXX}"
echo "    Python: ${PYTHON_EXE}"
echo ""

# ---- Clean build if requested ----
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "==> Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# ---- Configure ----
echo "==> Configuring with CMake..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CC="$CC" CXX="$CXX" cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DUSE_MPI=ON \
    -DWITH_NSOLVER=ON \
    -DWITH_NETCDF=Serial \
    -DWITH_HDF5CXX=OFF \
    -DWITH_PYTHON=OFF \
    -DPython3_EXECUTABLE="$PYTHON_EXE" \
    "$PROJECT_DIR"

# ---- Build ----
echo ""
echo "==> Building channelflow (${NPROC} cores)..."
make -j"$NPROC"

# ---- Test ----
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "==> Running Dedalus smoke tests..."
    ctest -L dedalus -V || echo "WARNING: Some tests failed. Check output above."
fi

# ---- Install ----
echo ""
echo "==> Installing to ${INSTALL_PREFIX}..."
make install

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "Binaries installed to: ${INSTALL_PREFIX}/bin"
echo ""
echo "To run channelflow programs:"
echo "  scripts/run-channelflow.sh <command> [args...]"
echo ""
echo "Example:"
echo "  scripts/run-channelflow.sh findsoln -sys active_matter -T 10 ubest.nc"
echo ""
