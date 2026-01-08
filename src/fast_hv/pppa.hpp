#pragma once
#ifndef PPPA_HPP
#define PPPA_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <omp.h>

/**
 * PPPA (Partial Precision and Partial Approximation) Hypervolume Calculator
 *
 * Based on: Tang, W., Liu, H.-L., Chen, L., Tan, K.C., & Cheung, Y.-m. (2020).
 * "Fast hypervolume approximation scheme based on a segmentation strategy."
 * Information Sciences, 509, 320-342.
 *
 * This implementation:
 * - Uses std::vector instead of linked lists (cache-friendly)
 * - Thread-safe with per-thread PRNGs
 * - OpenMP parallelized for multi-core systems
 */
class HypervolumeCalculator {
public:
    /**
     * Constructor
     * @param max_depth Maximum segmentation recursion depth (k in paper)
     * @param n_samples Monte Carlo samples per unit volume (rho in paper)
     */
    HypervolumeCalculator(int max_depth = 6, int n_samples = 10000000);

    /**
     * Compute hypervolume indicator
     * @param points Pointer to flattened row-major (N x M) array of objective values
     * @param n_points Number of points (N)
     * @param n_dim Number of dimensions/objectives (M)
     * @param ref_point Pointer to reference point array of size M
     * @return Hypervolume value
     *
     * Note: Points are assumed to be in minimization form (lower is better).
     *       The algorithm internally works with "distance to reference" which
     *       converts to a maximization problem.
     */
    double compute(const double* points, int n_points, int n_dim,
                   const double* ref_point);

private:
    int max_depth_;
    int n_samples_;

    /**
     * Recursive segmentation (Algorithm 1: PDCH from paper)
     * Works on normalized points where each coordinate is distance from ref_point
     */
    double recursive_hv(std::vector<std::vector<double>>& points,
                        int n_dim, int depth);

    /**
     * Monte Carlo approximation (Algorithm 2 from paper)
     * Called when depth >= max_depth or points.size() is small
     */
    double monte_carlo_approx(const std::vector<std::vector<double>>& points,
                              int n_dim);

    /**
     * Find pivot point with maximum volume product
     */
    int find_pivot(const std::vector<std::vector<double>>& points, int n_dim);

    /**
     * Check if point a dominates point b (for maximization)
     */
    static bool dominates(const std::vector<double>& a,
                          const std::vector<double>& b, int n_dim);
};

#endif // PPPA_HPP
