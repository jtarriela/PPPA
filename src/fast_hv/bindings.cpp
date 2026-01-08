#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pppa.hpp"

namespace py = pybind11;

double compute(py::array_t<double, py::array::c_style | py::array::forcecast> points,
               py::array_t<double, py::array::c_style | py::array::forcecast> ref_point,
               int max_depth = 6,
               int n_samples = 10000000) {

    auto pts = points.unchecked<2>();  // (N, M) matrix
    auto ref = ref_point.unchecked<1>();

    int n_points = static_cast<int>(pts.shape(0));
    int n_dim = static_cast<int>(pts.shape(1));

    if (n_dim != static_cast<int>(ref.shape(0))) {
        throw std::runtime_error("Reference point dimension must match points dimension");
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

PYBIND11_MODULE(_fast_hv, m) {
    m.doc() = "Fast hypervolume approximation using PPPA (Tang et al. 2020)";

    m.def("compute", &compute,
          py::arg("points"),
          py::arg("ref_point"),
          py::arg("max_depth") = 6,
          py::arg("n_samples") = 10000000,
          R"doc(
Compute hypervolume indicator for a set of points.

Parameters
----------
points : ndarray of shape (N, M)
    Objective values for N solutions with M objectives.
    Must be C-contiguous float64. Points should be in minimization
    format (lower is better).
ref_point : ndarray of shape (M,)
    Reference point that is dominated by all points (i.e., worse than
    all points in all objectives for minimization).
max_depth : int, default=6
    Maximum segmentation recursion depth. Higher values give more
    exact results but take longer.
n_samples : int, default=10000000
    Monte Carlo samples for approximation at leaf nodes.

Returns
-------
float
    Hypervolume indicator value.

Notes
-----
This implements the PPPA algorithm from Tang et al. (2020):
"Fast hypervolume approximation scheme based on a segmentation strategy"
Information Sciences, 509, 320-342.
)doc");
}
