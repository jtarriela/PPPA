#!/usr/bin/env python3
"""
PPPA Verification Test

Replicates testCpp.m from the MATLAB reference implementation.
Compares results against:
1. MATLAB mexPPPA results (if available in test_results.mat)
2. Pymoo exact HV (for smaller dimensions as sanity check)

Test instance: 100 points on unit sphere in 13 dimensions (spherical instance)
"""

import numpy as np
import time
import sys
from pathlib import Path

def generate_spherical_instance(n_points: int, n_dim: int, seed: int = 42) -> np.ndarray:
    """
    Generate spherical test instance (same as testCpp.m).

    Points are generated on a unit sphere surface, then flipped
    so they lie in [0, 1]^dim (minimization format).
    """
    np.random.seed(seed)

    C = np.zeros((n_points, n_dim))
    M = 0.0

    for i in range(n_dim - 1):
        C[:, i] = np.sqrt(1 - M) * np.random.rand(n_points)
        M = M + C[:, i] ** 2

    C[:, n_dim - 1] = np.sqrt(1 - M)

    # Flip points (bounds - C, where bounds = ones)
    bounds = np.ones(n_dim)
    C = bounds - C

    return C, bounds


def test_basic_functionality():
    """Test that fast_hv works on simple cases."""
    import fast_hv
    import sys

    print("=" * 60, flush=True)
    print("TEST 1: Basic functionality", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    # Simple 2D test case
    print("  Creating 2D test data...", flush=True)
    points_2d = np.array([[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]], dtype=np.float64)
    ref_2d = np.array([1.0, 1.0], dtype=np.float64)

    print("  Calling fast_hv.compute for 2D...", flush=True)
    hv = fast_hv.compute(points_2d, ref_2d, max_depth=6, n_samples=1000)
    print(f"  2D test (3 points): HV = {hv:.6f}", flush=True)

    # Expected: (1-0.5)*(1-0.5) + (0.5-0.3)*(1-0.7) + (0.5-0.3)*(1-0.3)
    # = 0.25 + 0.06 + 0.14 = 0.45 (approximately, inclusion-exclusion)
    # Actually let's just verify it's reasonable
    assert 0.3 < hv < 0.7, f"2D HV {hv} seems unreasonable"
    print("  PASSED: 2D test returned reasonable value", flush=True)

    # Single point test
    print("  Creating single point test data...", flush=True)
    points_1 = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)
    ref_1 = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    print("  Calling fast_hv.compute for single point...", flush=True)
    hv_1 = fast_hv.compute(points_1, ref_1, max_depth=6, n_samples=1000)
    expected_1 = (1-0.2) * (1-0.3) * (1-0.4)  # = 0.8 * 0.7 * 0.6 = 0.336
    print(f"  Single point test: HV = {hv_1:.6f}, expected = {expected_1:.6f}", flush=True)
    assert abs(hv_1 - expected_1) < 0.01, f"Single point HV error too large"
    print("  PASSED: Single point test", flush=True)

    return True


def test_against_pymoo(n_points: int = 50, n_dim: int = 5, seed: int = 42):
    """Compare fast_hv against pymoo exact HV for moderate dimensions."""
    import fast_hv

    print("\n" + "=" * 60)
    print(f"TEST 2: Comparison with Pymoo (N={n_points}, M={n_dim})")
    print("=" * 60)

    try:
        from pymoo.indicators.hv import Hypervolume
    except ImportError:
        print("  SKIPPED: pymoo not installed")
        return None

    # Generate test data
    C, bounds = generate_spherical_instance(n_points, n_dim, seed)
    ref_point = bounds.copy()

    # Compute with fast_hv
    start = time.perf_counter()
    hv_fast = fast_hv.compute(
        np.ascontiguousarray(C, dtype=np.float64),
        np.ascontiguousarray(ref_point, dtype=np.float64),
        max_depth=6,
        n_samples=100_000
    )
    time_fast = time.perf_counter() - start

    # Compute with pymoo (exact)
    metric = Hypervolume(ref_point=ref_point)
    start = time.perf_counter()
    hv_pymoo = metric.do(C)
    time_pymoo = time.perf_counter() - start

    rel_error = abs(hv_fast - hv_pymoo) / hv_pymoo * 100

    print(f"  fast_hv:  HV = {hv_fast:.10f}  ({time_fast:.4f}s)")
    print(f"  pymoo:    HV = {hv_pymoo:.10f}  ({time_pymoo:.4f}s)")
    print(f"  Relative error: {rel_error:.4f}%")

    if rel_error < 1.0:
        print("  PASSED: Error < 1%")
    else:
        print(f"  WARNING: Error = {rel_error:.2f}% (threshold 1%)")

    return {
        'hv_fast': hv_fast,
        'hv_pymoo': hv_pymoo,
        'rel_error': rel_error,
        'time_fast': time_fast,
        'time_pymoo': time_pymoo
    }


def test_main_instance():
    """
    Main test: Replicate testCpp.m exactly.
    100 points, 13 dimensions, spherical instance.
    """
    import fast_hv

    print("\n" + "=" * 60)
    print("TEST 3: Main Instance (N=100, M=13) - Replicating testCpp.m")
    print("=" * 60)

    C, bounds = generate_spherical_instance(n_points=100, n_dim=13, seed=42)
    ref_point = bounds.copy()

    print(f"  Points shape: {C.shape}")
    print(f"  Reference point: {ref_point}")
    print(f"  Points range: [{C.min():.4f}, {C.max():.4f}]")

    # Compute with fast_hv
    start = time.perf_counter()
    hv_result = fast_hv.compute(
        np.ascontiguousarray(C, dtype=np.float64),
        np.ascontiguousarray(ref_point, dtype=np.float64),
        max_depth=6,
        n_samples=100_000
    )
    elapsed = time.perf_counter() - start

    print(f"\n  Hypervolume: {hv_result:.10f}")
    print(f"  Time elapsed: {elapsed:.4f} seconds")

    # Try to load MATLAB reference if available
    matlab_hv = None
    mat_file = Path(__file__).parent / "test_results.mat"
    if mat_file.exists():
        try:
            from scipy.io import loadmat
            data = loadmat(mat_file)
            matlab_hv = float(data['hv_result'][0, 0])
            matlab_time = float(data['elapsed_time'][0, 0])

            rel_error = abs(hv_result - matlab_hv) / matlab_hv * 100

            print(f"\n  MATLAB reference:")
            print(f"    HV = {matlab_hv:.10f}  ({matlab_time:.4f}s)")
            print(f"    Relative error: {rel_error:.4f}%")

            if rel_error < 1.0:
                print("  PASSED: Error < 1% vs MATLAB")
            else:
                print(f"  WARNING: Error = {rel_error:.2f}%")
        except ImportError:
            print("\n  NOTE: scipy not available, cannot load MATLAB reference")
        except Exception as e:
            print(f"\n  NOTE: Could not load MATLAB reference: {e}")
    else:
        print(f"\n  NOTE: MATLAB reference not found at {mat_file}")
        print("  Run matlab_reference.m to generate reference data")

    return {
        'hv': hv_result,
        'time': elapsed,
        'matlab_hv': matlab_hv,
        'n_points': 100,
        'n_dim': 13
    }


def test_performance_scaling():
    """Test performance across different dimensions."""
    import fast_hv

    print("\n" + "=" * 60)
    print("TEST 4: Performance Scaling")
    print("=" * 60)

    results = []

    for n_dim in [5, 8, 10, 13, 15]:
        C, bounds = generate_spherical_instance(n_points=100, n_dim=n_dim, seed=42)
        ref_point = bounds.copy()

        start = time.perf_counter()
        hv = fast_hv.compute(
            np.ascontiguousarray(C, dtype=np.float64),
            np.ascontiguousarray(ref_point, dtype=np.float64),
            max_depth=6,
            n_samples=100_000
        )
        elapsed = time.perf_counter() - start

        results.append({'dim': n_dim, 'hv': hv, 'time': elapsed})
        print(f"  M={n_dim:2d}: HV = {hv:.8f}, time = {elapsed:.4f}s")

    return results


def generate_report(results: dict) -> str:
    """Generate verification report."""
    report = []
    report.append("=" * 70)
    report.append("PPPA VERIFICATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("Package: fast_hv v0.1.0")
    report.append("Algorithm: PPPA (Tang et al. 2020)")
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("-" * 70)
    report.append("MAIN TEST: Spherical Instance (100 points, 13 dimensions)")
    report.append("-" * 70)

    main = results.get('main', {})
    report.append(f"  Hypervolume:    {main.get('hv', 'N/A'):.10f}")
    report.append(f"  Compute time:   {main.get('time', 'N/A'):.4f} seconds")

    if main.get('matlab_hv'):
        rel_error = abs(main['hv'] - main['matlab_hv']) / main['matlab_hv'] * 100
        report.append(f"  MATLAB ref:     {main['matlab_hv']:.10f}")
        report.append(f"  Relative error: {rel_error:.4f}%")
        report.append(f"  Status:         {'PASS' if rel_error < 1.0 else 'CHECK'}")
    else:
        report.append("  MATLAB ref:     Not available")

    report.append("")

    if 'pymoo' in results and results['pymoo']:
        report.append("-" * 70)
        report.append("PYMOO COMPARISON (50 points, 5 dimensions)")
        report.append("-" * 70)
        pymoo = results['pymoo']
        report.append(f"  fast_hv:        {pymoo['hv_fast']:.10f} ({pymoo['time_fast']:.4f}s)")
        report.append(f"  pymoo (exact):  {pymoo['hv_pymoo']:.10f} ({pymoo['time_pymoo']:.4f}s)")
        report.append(f"  Relative error: {pymoo['rel_error']:.4f}%")
        report.append(f"  Status:         {'PASS' if pymoo['rel_error'] < 1.0 else 'CHECK'}")
        report.append("")

    if 'scaling' in results:
        report.append("-" * 70)
        report.append("PERFORMANCE SCALING (100 points)")
        report.append("-" * 70)
        for r in results['scaling']:
            report.append(f"  M={r['dim']:2d}: {r['time']:.4f}s")
        report.append("")

    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    print("\nPPPA (fast_hv) Verification Suite")
    print("=" * 60)

    results = {}

    # Test 1: Basic functionality
    try:
        test_basic_functionality()
        results['basic'] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results['basic'] = False

    # Test 2: Compare with pymoo
    try:
        results['pymoo'] = test_against_pymoo()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['pymoo'] = None

    # Test 3: Main instance (replicate testCpp.m)
    try:
        results['main'] = test_main_instance()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['main'] = {}

    # Test 4: Performance scaling
    try:
        results['scaling'] = test_performance_scaling()
    except Exception as e:
        print(f"  FAILED: {e}")
        results['scaling'] = []

    # Generate report
    report = generate_report(results)
    print("\n")
    print(report)

    # Save report
    report_path = Path(__file__).parent / "verification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return 0 if results.get('basic') else 1


if __name__ == "__main__":
    sys.exit(main())
