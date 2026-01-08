#!/usr/bin/env python3
"""Quick diagnostic test for fast_hv."""

import sys
import time
import numpy as np

print("Starting quick test...", flush=True)

print("Importing fast_hv...", flush=True)
import fast_hv
print("Import successful", flush=True)

# Test 1: Single point
print("\n[TEST 1] Single point, 2D", flush=True)
points = np.array([[0.2, 0.3]], dtype=np.float64)
ref = np.array([1.0, 1.0], dtype=np.float64)
print(f"  Points: {points}", flush=True)
print(f"  Ref: {ref}", flush=True)
print("  Calling compute...", flush=True)
start = time.time()
hv = fast_hv.compute(points, ref, max_depth=2, n_samples=100)
print(f"  HV = {hv:.6f}, time = {time.time()-start:.3f}s", flush=True)
print(f"  Expected: {0.8 * 0.7:.6f}", flush=True)

# Test 2: 3 points 2D
print("\n[TEST 2] 3 points, 2D", flush=True)
points = np.array([[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]], dtype=np.float64)
ref = np.array([1.0, 1.0], dtype=np.float64)
print("  Calling compute...", flush=True)
start = time.time()
hv = fast_hv.compute(points, ref, max_depth=4, n_samples=1000)
print(f"  HV = {hv:.6f}, time = {time.time()-start:.3f}s", flush=True)

# Test 3: 10 points 5D
print("\n[TEST 3] 10 points, 5D", flush=True)
np.random.seed(42)
points = np.random.rand(10, 5) * 0.5
points = np.ascontiguousarray(points, dtype=np.float64)
ref = np.ones(5, dtype=np.float64)
print("  Calling compute...", flush=True)
start = time.time()
hv = fast_hv.compute(points, ref, max_depth=4, n_samples=1000)
print(f"  HV = {hv:.6f}, time = {time.time()-start:.3f}s", flush=True)

# Test 4: 50 points 5D (compare with pymoo)
print("\n[TEST 4] 50 points, 5D (pymoo comparison)", flush=True)
np.random.seed(42)
points = np.random.rand(50, 5) * 0.5
points = np.ascontiguousarray(points, dtype=np.float64)
ref = np.ones(5, dtype=np.float64)

print("  Calling fast_hv.compute...", flush=True)
start = time.time()
hv_fast = fast_hv.compute(points, ref, max_depth=6, n_samples=10000)
time_fast = time.time() - start
print(f"  fast_hv: HV = {hv_fast:.6f}, time = {time_fast:.3f}s", flush=True)

try:
    print("  Importing pymoo...", flush=True)
    from pymoo.indicators.hv import Hypervolume
    print("  Calling pymoo Hypervolume...", flush=True)
    metric = Hypervolume(ref_point=ref)
    start = time.time()
    hv_pymoo = metric.do(points)
    time_pymoo = time.time() - start
    print(f"  pymoo:   HV = {hv_pymoo:.6f}, time = {time_pymoo:.3f}s", flush=True)
    rel_err = abs(hv_fast - hv_pymoo) / hv_pymoo * 100
    print(f"  Relative error: {rel_err:.2f}%", flush=True)
except ImportError:
    print("  pymoo not available, skipping comparison", flush=True)

# Test 5: Main test - 100 points 13D
print("\n[TEST 5] 100 points, 13D (main test)", flush=True)
np.random.seed(42)

# Generate spherical instance like testCpp.m
n_points = 100
n_dim = 13
C = np.zeros((n_points, n_dim))
M = 0.0
for i in range(n_dim - 1):
    C[:, i] = np.sqrt(1 - M) * np.random.rand(n_points)
    M = M + C[:, i] ** 2
C[:, n_dim - 1] = np.sqrt(1 - M)
bounds = np.ones(n_dim)
C = bounds - C  # Flip

points = np.ascontiguousarray(C, dtype=np.float64)
ref = np.ascontiguousarray(bounds, dtype=np.float64)

print(f"  Points shape: {points.shape}", flush=True)
print(f"  Points range: [{points.min():.4f}, {points.max():.4f}]", flush=True)
print("  Calling fast_hv.compute (this may take a moment)...", flush=True)

start = time.time()
hv = fast_hv.compute(points, ref, max_depth=6, n_samples=10000)
elapsed = time.time() - start

print(f"  HV = {hv:.10f}", flush=True)
print(f"  Time = {elapsed:.3f}s", flush=True)

print("\n" + "="*60, flush=True)
print("All tests completed successfully!", flush=True)
print("="*60, flush=True)
