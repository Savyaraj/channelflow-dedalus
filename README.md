
# Channelflow-Dedalus

This repository augments channelflow/nsolver by replacing the time-integration step in nonlinear solves with a flexible Python + Dedalus interface.

## What this fork adds

- Keep channelflow's C++ nonlinear algorithms (`findsoln`, `continuesoln`, `findeigenvals`).
- Delegate time marching to Dedalus systems implemented in Python modules under `dedalus/`.
- Select systems at runtime using `-sys <module_name>` (for example `-sys LLE`).
- Support rapid experimentation with timesteppers/physics in Python without rewriting C++ solvers.

## Coupling architecture

At runtime, `findsoln` calls the C++ bridge (`dedaDSI`) which embeds Python, imports your module, and instantiates `DedalusPy`:

```text
findsoln (C++)
  -> dedaDSI / DedaField bridge (C++)
  -> Py_Initialize + import <sys_module>
  -> DedalusPy object (Python)
  -> advance(T, u) / diff / observable / updateMu
```

This architecture keeps Newton/continuation in C++ while making the PDE timestepper fully programmable in Python.

## Quick start

```bash
# 1) Create environment (recommended)
mamba create -n channelflow -c conda-forge --strict-channel-priority python=3.10 dedalus compilers eigen cmake libnetcdf netcdf4

# If mamba is unavailable, replace `mamba create` with `conda create`
conda activate channelflow

# 2) Build and install
git clone <repo-url> channelflow-dedalus && cd channelflow-dedalus
bash scripts/install.sh
```

## Running the interface

Use the wrapper so `PATH` and `PYTHONPATH` are set correctly:

```bash
scripts/run-channelflow.sh findsoln -sys LLE -T 10 ubest.nc
```

First coupled smoke-style run:

```bash
# create a minimal field file
scripts/run-channelflow.sh randomdedafield -Nd 1 -Nvar 1 -N 512 -L 3 psi.nc

# run one Newton iteration through C++ -> Python -> Dedalus
scripts/run-channelflow.sh findsoln -eqb -T 1 -Nn 1 -sys LLE psi.nc
```

## Flexible timestepping model

The Python side controls timestepping behavior:

- `self.dt` sets the base timestep.
- `self.timestepper` selects Dedalus timestepper (for example `de.RK222`, `de.RK443`, `de.SBDF2`).
- `advance(T, u_init)` evolves to exactly `T` using one or more internal steps.
- For Jacobian approximations, the interface returns $(u(T)-u(0))/T$ when `u_init` is provided.

This lets you tune integration strategy per system while reusing the same C++ continuation/Newton framework.

## Implementing a new system (recommended pattern)

Create `dedalus/<my_system>.py` with a class named exactly `DedalusPy` inheriting `DedalusInterface`.

### Required class name and module mapping

- File name: `dedalus/my_system.py`
- Class name: `DedalusPy`
- Run from C++ with: `-sys my_system`

### Required setup sequence

Before calling `super().__init__()`, set:

- `self.dt`
- `self.N`
- `self.parameters` (dictionary)
- `self.Niter`
- `self.timestepper`
- `self.N_Lc`

Then implement domain/problem/solver setup hooks.

### Required interface methods

Your class must provide (directly or via base hooks):

- `advance(T, u_init=None)` (usually inherited)
- `add_perturbations(mag, decay)`
- `observable(x)`
- `diff(x, dir)`
- `read_h5(filename, iter)`
- `updateMu(mu, muName)` (usually inherited)

In addition, implement these base-class abstract hooks:

- `system_name`
- `_get_initial_param()`
- `_get_default_mu_name()`
- `_get_vector_size()`

And implement conversion helpers used by the above methods:

- `to_field(x)`
- `to_vector()`

### Minimal template

```python
from dedalus_interface import DedalusInterface
from dedalus import public as de

class DedalusPy(DedalusInterface):
   def __init__(self):
      self.dt = 1e-2
      self.N = 256
      self.N_Lc = 1
      self.Niter = 1000
      self.parameters = {"a": 0.0}
      self.timestepper = de.RK222
      super().__init__()

   @property
   def system_name(self):
      return "my_system"

   def _get_initial_param(self):
      return self.parameters["a"]

   def _get_default_mu_name(self):
      return "a"

   def _get_vector_size(self):
      return 2 * (self.N + 1)

   def domain_setup(self, N_Lc):
      ...

   def problem_setup(self, mu=-1, muName='a'):
      ...

   def build_solver(self, n_iter=1000):
      ...

   def init_problem(self):
      ...

   def to_field(self, x):
      ...

   def to_vector(self):
      ...

   def add_perturbations(self, mag, decay):
      ...

   def observable(self, x):
      ...

   def diff(self, x, dir):
      ...

   def read_h5(self, filename, iter):
      ...
```

### Reference implementations

- `dedalus/LLE.py`
- `dedalus/coupled_LLE.py`
- `dedalus/trimer_LLE.py`

## Interface contract expected by C++ bridge

The C++ side (`dedaDSI`) expects the Python object to expose methods with these names:

- `advance`
- `diff`
- `observable`
- `add_perturbations`
- `read_h5`
- `updateMu`

If method names/signatures do not match, bridge calls fail at runtime.

## Validation and smoke tests

From the build directory:

```bash
ctest -L dedalus -V
```

Manual smoke test script:

```bash
export FINDSOLN=$PWD/build/programs/findsoln
export RANDOMDEDAFIELD=$PWD/build/tools/randomdedafield
export PYTHONPATH=$PWD/dedalus:${PYTHONPATH}
bash tests/dedalus/test_findsoln_smoke.sh
```

Expected pass signal: output contains `L2Norm` and does not contain `nan`.

## MPI and runtime notes

- The current Dedalus bridge path uses `DedaField` vector representation for Python coupling.
- Keep `OMP_NUM_THREADS=1` for stable smoke-test behavior unless you intentionally tune threading.
- Prefer `scripts/run-channelflow.sh` so `install/bin` and `dedalus/` are visible in `PATH`/`PYTHONPATH`.

## Legacy channelflow utilities

The repository still includes the full channelflow toolchain (DNS, continuation, post-processing tools).

## Troubleshooting

- Install issues: verify the conda/mamba environment is activated before running `scripts/install.sh`.
- Python import issues: ensure `PYTHONPATH` includes the repository `dedalus/` folder.
- Runtime interface issues: verify your module defines class `DedalusPy` and required methods above.

For broader compilation/use questions, see the channelflow forum: [discourse.channelflow.ch](https://discourse.channelflow.ch/).

## Bugs report

Please report issues via GitHub Issues.

## License

Channelflow is released under the [GNU GPL version 2](./LICENSE)

