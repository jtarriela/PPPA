# PPPA

PPPA is a C++17/OpenMP hypervolume approximation library packaged as the Python module `fast_hv`.
It is intended for Linux and WSL2. Native Windows builds are not supported.

## What it does

Use `fast_hv.compute(...)` to approximate hypervolume for a set of objective vectors and a reference point.
The package uses pybind11 for the Python bridge and scikit-build-core/CMake for the build.

## Install

From the repository root on Linux or inside WSL2:

```bash
python -m pip install -U pip
python -m pip install -e .
```

For a standard install:

```bash
python -m pip install .
```

## Windows Guidance

If you are on Windows, use WSL2 and run the Linux commands above from the WSL shell.
Native Windows is blocked intentionally by CMake so you do not get a partial or misleading build.

## Manual CMake Build on Linux

If you need to configure and build directly with CMake, use a Linux shell:

```bash
python -m pip install -U pip
python -m pip install scikit-build-core pybind11 numpy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build -j
```

## Smoke Test

Run the bundled diagnostic script:

```bash
python tests/quick_test.py
```

## Minimal Python Example

```python
import numpy as np
import fast_hv

points = np.array([[0.2, 0.3], [0.4, 0.1]], dtype=np.float64)
ref = np.array([1.0, 1.0], dtype=np.float64)

hv = fast_hv.compute(points, ref, max_depth=4, n_samples=1000)
print(hv)
```
