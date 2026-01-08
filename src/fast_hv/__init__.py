"""Fast hypervolume approximation using PPPA algorithm.

This package provides a high-performance C++17/OpenMP implementation
of the PPPA (Partial Precision and Partial Approximation) algorithm
for computing hypervolume indicators in multi-objective optimization.

Reference:
    Tang, W., Liu, H.-L., Chen, L., Tan, K.C., & Cheung, Y.-m. (2020).
    Fast hypervolume approximation scheme based on a segmentation strategy.
    Information Sciences, 509, 320-342.
"""

from ._fast_hv import compute

__all__ = ["compute"]
__version__ = "0.1.0"
