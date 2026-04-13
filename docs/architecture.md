# PPPA Hypervolume Module - Architecture & Implementation Plan

## Overview

Build a production-grade C++17 PPPA (Partial Precision and Partial Approximation) hypervolume library as a pip-installable Python package, integrated with gdwh_fsm's genetic algorithm optimizer and pymoo. Supported development targets are Linux and WSL2; native Windows is not supported.

**Target Environment:** Linux HPC head node or WSL2 on Windows, Intel Skylake / AMD EPYC (x86-64-v3)
**Parallelism:** OpenMP only (shared memory, no MPI)
**Python Integration:** pybind11 with zero-copy NumPy array access

### Automatic Algorithm Selection (Key Feature)

gdwh_fsm will **automatically switch** between hypervolume algorithms based on objective count:

| Objectives (M) | Algorithm | Library | When to Use |
|:--------------:|:---------:|---------|-------------|
| **M <= 8** | HSO (exact) | `pymoo.indicators.hv` (moocore) | Fast, exact results |
| **M > 8** | PPPA (approximate) | `fast_hv` (this build) | 10x faster, ~0.1% error |

This mirrors the existing pattern in `gdwh/optimization/pareto.py` which auto-selects between pygmo (M<=4) and Jensen's algorithm (M>4) for non-dominated sorting.

### Reference

[Tang, W., Liu, H.-L., Chen, L., Tan, K.C., & Cheung, Y.-m. (2020). *Fast hypervolume approximation scheme based on a segmentation strategy.* Information Sciences, 509, 320-342.](https://www.sciencedirect.com/science/article/pii/S0020025519301690?via%3Dihub)

---

## 1. Architecture Overview

We will build a **Shared Object Library (`.so`)** using **C++17** that hooks directly into Python memory.

* **Input:** Zero-copy view of your Pymoo population (NumPy array)
* **Engine:** OpenMP-based Task Parallelism (for recursion) + Data Parallelism (for sampling)
* **Output:** Double precision Hypervolume scalar

### The "Hybrid Parallel" Strategy

To saturate 48 cores with a recursive algorithm, we need two types of parallelism:

1. **Macro (Task) Parallelism:** Handled by `omp task`. As the algorithm recursively splits the geometric space, it spawns tasks. Free cores "steal" these tasks.
2. **Micro (Data) Parallelism:** Handled by `omp parallel for`. When the recursion bottoms out and switches to Monte Carlo approximation, we blast the sampling loop across all available threads.

---

## 2. Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | C++17 | Strictly typed, STL algorithms, `std::mt19937_64` |
| **Build System** | CMake + scikit-build-core | Standard for HPC, pip-installable |
| **Python Bridge** | pybind11 | Lowest overhead, NumPy integration |
| **Parallelism** | OpenMP | Native to GCC/LLVM, shared memory |
| **Target Arch** | x86-64-v3 | AVX2/FMA, common to Skylake & Zen2+ |
| **Data Structures** | `std::vector<double>` | Contiguous memory, cache-friendly |

---

## 3. Package Structure

```
PPPA/
├── pyproject.toml           # PEP 517 build configuration
├── CMakeLists.txt           # Main build configuration
├── src/
│   └── fast_hv/
│       ├── __init__.py      # Python package init
│       ├── pppa.hpp         # Header with class definition
│       ├── pppa.cpp         # Core PPPA algorithm (refactored)
│       ├── monte_carlo.hpp  # Monte Carlo approximation header
│       ├── monte_carlo.cpp  # Monte Carlo approximation kernel
│       └── bindings.cpp     # pybind11 module definition
├── tests/
│   ├── test_accuracy.py     # Compare against pymoo exact HV
│   ├── test_performance.py  # Benchmark vs FPRAS/moocore
│   └── conftest.py
└── docs/
    ├── architecture.md      # This file
    └── translated_code/     # Original reference implementation
```

---

## 4. C++ Implementation

### 4.1 Refactoring from Original Code

| Original (MEX/Global State) | Refactored (Production C++17) |
|----------------------------|-------------------------------|
| `LinkList` for point sets | `std::vector<double>` (contiguous, cache-friendly) |
| Global `numK`, `Samp` | Class members in `HypervolumeCalculator` |
| Single-threaded recursion | OpenMP task parallelism + data parallelism |
| Raw pointers | Pointer + size pairs with bounds checking |
| `rand()` | `std::mt19937_64` per-thread PRNG (thread-safe) |

### 4.2 Class Design

```cpp
// pppa.hpp
#pragma once
#include <vector>
#include <random>

class HypervolumeCalculator {
public:
    HypervolumeCalculator(int max_depth = 6, int n_samples = 10'000'000);

    // Main API - accepts flattened row-major array
    double compute(const double* points, int n_points, int n_dim,
                   const double* ref_point);

private:
    int max_depth_;      // Segmentation recursion depth (k in paper)
    int n_samples_;      // Monte Carlo samples per unit volume (rho)

    // Recursive segmentation (Algorithm 1: PDCH)
    double recursive_hv(std::vector<double>& points, int n, int dim,
                        std::vector<double>& bounds, int depth);

    // Monte Carlo approximation (Algorithm 2: leaf nodes)
    double monte_carlo_approx(const std::vector<double>& points, int n, int dim,
                              const std::vector<double>& bounds);

    // Pivot selection: argmax_{a in F} HV({a}, B)
    int find_pivot(const std::vector<double>& points, int n, int dim,
                   const std::vector<double>& bounds);
};
```

### 4.3 Core Algorithm (Task Parallelism)

```cpp
double HypervolumeCalculator::recursive_hv(
    std::vector<double>& flat_points, int n_points, int dim,
    std::vector<double>& bounds, int depth) {

    // 1. Base Case (Monte Carlo)
    if (depth >= max_depth_ || n_points <= 1) {
        return monte_carlo_approx(flat_points, n_points, dim, bounds);
    }

    // 2. Find pivot (point with maximum volume product)
    int pivot_idx = find_pivot(flat_points, n_points, dim, bounds);
    double pivot_volume = 1.0;
    for (int d = 0; d < dim; ++d) {
        pivot_volume *= (bounds[d] - flat_points[pivot_idx * dim + d]);
    }

    // 3. Partition into d subsets
    std::vector<std::vector<double>> sub_pops(dim);
    // ... partitioning logic per Tang et al. Algorithm 1 ...

    double total_hv = pivot_volume;
    std::vector<double> sub_results(dim, 0.0);

    // 4. Parallel Recursion (Task Parallelism)
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int d = 0; d < dim; ++d) {
                #pragma omp task shared(sub_results) firstprivate(d)
                {
                    if (!sub_pops[d].empty()) {
                        std::vector<double> new_bounds = bounds;
                        new_bounds[d] = flat_points[pivot_idx * dim + d];
                        int sub_n = sub_pops[d].size() / dim;
                        sub_results[d] = recursive_hv(sub_pops[d], sub_n, dim,
                                                       new_bounds, depth + 1);
                    }
                }
            }
        }
    }
    #pragma omp taskwait

    for (double val : sub_results) total_hv += val;
    return total_hv;
}
```

### 4.4 Monte Carlo Approximation (Data Parallelism)

```cpp
double HypervolumeCalculator::monte_carlo_approx(
    const std::vector<double>& points, int n, int dim,
    const std::vector<double>& bounds) {

    if (n == 0) return 0.0;

    // Compute bounding box volume
    double box_volume = 1.0;
    std::vector<double> ideal(dim);
    for (int d = 0; d < dim; ++d) {
        double min_val = bounds[d];
        for (int i = 0; i < n; ++i) {
            min_val = std::min(min_val, points[i * dim + d]);
        }
        ideal[d] = min_val;
        box_volume *= (bounds[d] - ideal[d]);
    }

    if (box_volume <= 0.0) return 0.0;

    int total_samples = static_cast<int>(n_samples_ * box_volume);
    total_samples = std::max(total_samples, 1000);  // Minimum samples

    long long dominated_count = 0;

    #pragma omp parallel reduction(+:dominated_count)
    {
        // Thread-local PRNG
        std::mt19937_64 rng(std::random_device{}() + omp_get_thread_num());
        std::vector<std::uniform_real_distribution<double>> dists(dim);
        for (int d = 0; d < dim; ++d) {
            dists[d] = std::uniform_real_distribution<double>(ideal[d], bounds[d]);
        }

        #pragma omp for schedule(static)
        for (int s = 0; s < total_samples; ++s) {
            // Generate random sample point
            std::vector<double> sample(dim);
            for (int d = 0; d < dim; ++d) {
                sample[d] = dists[d](rng);
            }

            // Check if any point dominates this sample
            for (int i = 0; i < n; ++i) {
                bool dominated = true;
                for (int d = 0; d < dim; ++d) {
                    if (points[i * dim + d] > sample[d]) {
                        dominated = false;
                        break;
                    }
                }
                if (dominated) {
                    ++dominated_count;
                    break;
                }
            }
        }
    }

    return box_volume * static_cast<double>(dominated_count) / total_samples;
}
```

---

## 5. Build System

### 5.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(fast_hv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# x86-64-v3: AVX2, FMA - common to Skylake and Zen2+
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=x86-64-v3 -fPIC -ffast-math")

# OpenMP
find_package(OpenMP REQUIRED)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Build the module
pybind11_add_module(fast_hv
    src/fast_hv/pppa.cpp
    src/fast_hv/monte_carlo.cpp
    src/fast_hv/bindings.cpp
)

target_include_directories(fast_hv PRIVATE src/fast_hv)
target_link_libraries(fast_hv PRIVATE OpenMP::OpenMP_CXX)

# Install
install(TARGETS fast_hv LIBRARY DESTINATION fast_hv)
```

### 5.2 pyproject.toml

```toml
[build-system]
requires = ["scikit-build-core>=0.5", "pybind11>=2.11"]
build-backend = "scikit_build_core.build"

[project]
name = "fast-hv"
version = "0.1.0"
description = "Fast hypervolume approximation using PPPA algorithm (Tang et al. 2020)"
requires-python = ">=3.9"
dependencies = ["numpy>=1.20"]

[project.optional-dependencies]
test = ["pytest", "pymoo>=0.6"]
dev = ["pytest", "pymoo>=0.6", "black", "mypy"]
```

### 5.3 Installation

```bash
# From the PPPA directory on Linux or WSL2
pip install .

# Or in development mode
pip install -e .

# Verify installation
python -c "import fast_hv; print(fast_hv.compute.__doc__)"
```

On native Windows, stop at the CMake guard and use Linux or WSL2 instead.

---

## 6. pybind11 Bindings

### bindings.cpp

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pppa.hpp"

namespace py = pybind11;

double compute(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<double, py::array::c_style | py::array::forcecast> ref_point,
    int max_depth = 6,
    int n_samples = 10'000'000) {

    auto pts = points.unchecked<2>();  // (N, M) matrix
    auto ref = ref_point.unchecked<1>();

    int n_points = pts.shape(0);
    int n_dim = pts.shape(1);

    if (ref.shape(0) != n_dim) {
        throw std::runtime_error("ref_point dimension must match points");
    }

    HypervolumeCalculator calc(max_depth, n_samples);

    // Release GIL for parallel C++ execution
    double result;
    {
        py::gil_scoped_release release;
        result = calc.compute(pts.data(0, 0), n_points, n_dim, ref.data(0));
    }

    return result;
}

PYBIND11_MODULE(fast_hv, m) {
    m.doc() = "Fast hypervolume approximation using PPPA (Tang et al. 2020)";

    m.def("compute", &compute,
          py::arg("points"),
          py::arg("ref_point"),
          py::arg("max_depth") = 6,
          py::arg("n_samples") = 10'000'000,
          R"doc(
Compute hypervolume indicator for a set of points.

Parameters
----------
points : ndarray of shape (N, M)
    Objective values for N solutions with M objectives.
    Must be C-contiguous float64.
ref_point : ndarray of shape (M,)
    Reference point that dominates all points.
max_depth : int, default=6
    Maximum segmentation recursion depth.
n_samples : int, default=10000000
    Monte Carlo samples per unit volume.

Returns
-------
float
    Hypervolume indicator value.

Examples
--------
>>> import numpy as np
>>> import fast_hv
>>> F = np.array([[0.1, 0.9], [0.3, 0.5], [0.7, 0.2]])
>>> ref = np.array([1.1, 1.1])
>>> hv = fast_hv.compute(F, ref)
)doc");
}
```

---

## 7. gdwh_fsm Integration

### 7.1 Convergence Utilities Module

**File:** `gdwh/utils/convergence.py`

```python
"""Hypervolume computation with automatic algorithm selection."""
import numpy as np
from typing import Optional


def compute_hypervolume(
    F: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    n_objectives_threshold: int = 8,
) -> Optional[float]:
    """
    Compute hypervolume with auto-selection between exact and approximate.

    Algorithm Selection (mirrors pareto.py pattern):
    - M <= threshold: Use Pymoo's moocore (HSO exact algorithm)
    - M > threshold: Use fast_hv PPPA (Monte Carlo approximation)

    Parameters
    ----------
    F : ndarray of shape (N, M)
        Objective values for N feasible solutions with M objectives.
    ref_point : ndarray of shape (M,), optional
        Reference point. If None, computed as nadir * 1.1.
    n_objectives_threshold : int, default=8
        Switch to PPPA approximation when M > threshold.

    Returns
    -------
    float or None
        Hypervolume value, or None if computation not possible.
    """
    if F.size == 0:
        return None

    n_points, n_obj = F.shape

    # Compute reference point if not provided: nadir + 10% margin
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)

    if ref_point is None:
        # Avoid division by zero
        if np.any(nadir <= ideal):
            return None
        ref_point = nadir * 1.1

    # Algorithm selection based on objective count
    if n_obj <= n_objectives_threshold:
        # Exact computation via Pymoo
        try:
            from pymoo.indicators.hv import Hypervolume

            metric = Hypervolume(
                ref_point=ref_point,
                zero_to_one=True,
                ideal=ideal,
                nadir=nadir,
            )
            return float(metric.do(F))
        except ImportError:
            pass  # Fall through to PPPA

    # Approximate computation via PPPA (M > threshold or pymoo unavailable)
    try:
        import fast_hv

        F_contiguous = np.ascontiguousarray(F, dtype=np.float64)
        ref_contiguous = np.ascontiguousarray(ref_point, dtype=np.float64)
        return fast_hv.compute(F_contiguous, ref_contiguous, max_depth=6)
    except ImportError:
        # Neither library available
        return None
```

### 7.2 Update optimizer.py CheckpointCallback

Replace `hypervolume = None` stub (~line 142) with:

```python
# Inside CheckpointCallback.notify()
from gdwh.utils.convergence import compute_hypervolume

# Get feasible population objectives
F = algorithm.pop.get("F")
feasible_mask = algorithm.pop.get("feasible")
if feasible_mask is not None:
    feasible_mask = feasible_mask.flatten()
    F_feasible = F[feasible_mask]
else:
    F_feasible = F

# Compute hypervolume (auto-selects algorithm based on M)
hypervolume = compute_hypervolume(F_feasible) if len(F_feasible) > 0 else None
```

### 7.3 ADR-010 Update

Add to `gdwh_fsm/docs/ADR-010-Convergence-Metrics-and-Hypervolume.md`:

```markdown
## PPPA Integration (M > 8 Objectives)

For many-objective problems (M > 8), exact hypervolume computation becomes prohibitive.
The [fast-hv](../../../PPPA) package provides the PPPA algorithm (Tang et al. 2020).

### Installation

```bash
pip install /path/to/PPPA
```

### Algorithm Selection (Automatic)

gdwh automatically selects the optimal HV algorithm based on objective count:

| Objectives | Algorithm | Library | Complexity |
|:----------:|:---------:|---------|------------|
| M <= 8 | HSO (exact) | `pymoo.indicators.hv` | O(N * 2^(M-1)) |
| M > 8 | PPPA (approx) | `fast_hv` | O(dnρ(VQ)^(1+ε)) |

### Configuration

The threshold can be adjusted via `compute_hypervolume()` parameter:

```python
from gdwh.utils.convergence import compute_hypervolume

# Default: switch at M > 8
hv = compute_hypervolume(F, n_objectives_threshold=8)

# Force PPPA for M > 6 (faster, slight accuracy loss)
hv = compute_hypervolume(F, n_objectives_threshold=6)

# Always use exact (slow for M > 10)
hv = compute_hypervolume(F, n_objectives_threshold=99)
```
```

---

## 8. Verification Plan

### 8.1 Unit Tests (PPPA package)

```bash
cd /home/jd/proj/PPPA
pytest tests/ -v
```

**test_accuracy.py:**
- Compare PPPA vs pymoo exact HV for M=3,5,8
- Assert relative error < 0.1%

**test_performance.py:**
- Benchmark PPPA vs moocore for M=10,15
- Expect 10x+ speedup for M > 8

**test_parallel.py:**
- Verify OpenMP scaling: OMP_NUM_THREADS=1,4,16,48

### 8.2 Integration Tests (gdwh_fsm)

```bash
cd /home/jd/proj/gdwh_fsm
pytest tests/test_convergence_metrics.py -v
```

- Verify `compute_hypervolume()` returns non-None for valid inputs
- Verify algorithm selection logic (pymoo for M<=8, fast_hv for M>8)
- Verify HV is stored in database after generation

### 8.3 End-to-End Test

```python
# Run short optimization and verify HV tracking
from gdwh.ga.optimizer import run_optimization

result = run_optimization(config_path="test_config.yaml", n_gen=10)

# Query database
import sqlite3
conn = sqlite3.connect(result.db_path)
hvs = conn.execute(
    "SELECT gen_num, hypervolume FROM generations WHERE campaign_id=? ORDER BY gen_num",
    (result.campaign_id,)
).fetchall()

# Verify HV is populated and generally increasing
assert all(hv is not None for _, hv in hvs)
print("HV progression:", [hv for _, hv in hvs])
```

---

## 9. Implementation Checklist

| Step | Task | Status |
|------|------|--------|
| 1 | Create pyproject.toml | Pending |
| 2 | Create CMakeLists.txt | Pending |
| 3 | Refactor pppa.hpp/cpp (remove globals, use std::vector) | Pending |
| 4 | Implement monte_carlo.hpp/cpp | Pending |
| 5 | Add OpenMP parallelization | Pending |
| 6 | Create bindings.cpp (pybind11 with GIL release) | Pending |
| 7 | Create src/fast_hv/__init__.py | Pending |
| 8 | Build and test (`pip install .`) | Pending |
| 9 | Create gdwh/utils/convergence.py | Pending |
| 10 | Update optimizer.py CheckpointCallback | Pending |
| 11 | Update ADR-010 documentation | Pending |
| 12 | Run integration tests | Pending |

---

## 10. Files Summary

### PPPA Repository

| File | Action |
|------|--------|
| `pyproject.toml` | Create |
| `CMakeLists.txt` | Create |
| `src/fast_hv/__init__.py` | Create |
| `src/fast_hv/pppa.hpp` | Create |
| `src/fast_hv/pppa.cpp` | Create (refactor from docs/translated_code/) |
| `src/fast_hv/monte_carlo.hpp` | Create |
| `src/fast_hv/monte_carlo.cpp` | Create |
| `src/fast_hv/bindings.cpp` | Create |
| `tests/test_accuracy.py` | Create |
| `tests/test_performance.py` | Create |

### gdwh_fsm Repository

| File | Action |
|------|--------|
| `gdwh/utils/convergence.py` | Create |
| `gdwh/ga/optimizer.py` | Modify (~line 142) |
| `docs/ADR-010-*.md` | Update (add PPPA section) |
| `tests/test_convergence_metrics.py` | Create |
