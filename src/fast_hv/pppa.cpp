#include "pppa.hpp"
#include <limits>
#include <numeric>
#include <algorithm>

HypervolumeCalculator::HypervolumeCalculator(int max_depth, int n_samples)
    : max_depth_(max_depth), n_samples_(n_samples) {
    // Set default number of threads to 6 if not already set
    if (omp_get_max_threads() > 6) {
        omp_set_num_threads(6);
    }
}

double HypervolumeCalculator::compute(const double* points, int n_points, int n_dim,
                                       const double* ref_point) {
    if (n_points == 0 || n_dim == 0) {
        return 0.0;
    }

    // Convert to internal format: each point stores (ref_point[i] - point[i])
    // This transforms minimization to maximization and normalizes to [0, ref-point]
    std::vector<std::vector<double>> normalized_points(n_points, std::vector<double>(n_dim));

    for (int i = 0; i < n_points; ++i) {
        bool valid = true;
        for (int j = 0; j < n_dim; ++j) {
            double val = ref_point[j] - points[i * n_dim + j];
            if (val <= 0) {
                valid = false;
                break;
            }
            normalized_points[i][j] = val;
        }
        if (!valid) {
            // Point is not dominated by reference point, skip it
            normalized_points[i].assign(n_dim, 0.0);
        }
    }

    // Remove invalid points (all zeros)
    std::vector<std::vector<double>> valid_points;
    valid_points.reserve(n_points);
    for (auto& p : normalized_points) {
        bool all_zero = true;
        for (int j = 0; j < n_dim; ++j) {
            if (p[j] > 0) {
                all_zero = false;
                break;
            }
        }
        if (!all_zero) {
            valid_points.push_back(std::move(p));
        }
    }

    if (valid_points.empty()) {
        return 0.0;
    }

    return recursive_hv(valid_points, n_dim, 0);
}

double HypervolumeCalculator::recursive_hv(std::vector<std::vector<double>>& points,
                                            int n_dim, int depth) {
    int n_points = static_cast<int>(points.size());

    if (n_points == 0) {
        return 0.0;
    }

    // Special cases for small point sets (exact computation)
    if (n_points == 1) {
        double vol = 1.0;
        for (int j = 0; j < n_dim; ++j) {
            vol *= points[0][j];
        }
        return vol;
    }

    if (n_points == 2) {
        double vol_a = 1.0, vol_b = 1.0, vol_min = 1.0;
        for (int j = 0; j < n_dim; ++j) {
            vol_a *= points[0][j];
            vol_b *= points[1][j];
            vol_min *= std::min(points[0][j], points[1][j]);
        }
        return vol_a + vol_b - vol_min;
    }

    // Base case: use Monte Carlo approximation when deep enough or too many points
    if (depth >= max_depth_ || (depth >= 3 && n_points > 20)) {
        return monte_carlo_approx(points, n_dim);
    }

    // Find pivot point (point with maximum volume)
    int pivot_idx = find_pivot(points, n_dim);
    std::vector<double> pivot = points[pivot_idx];  // Copy pivot

    // Compute pivot volume
    double pivot_vol = 1.0;
    for (int j = 0; j < n_dim; ++j) {
        pivot_vol *= pivot[j];
    }

    double total_hv = pivot_vol;

    // Process each dimension
    for (int d = 0; d < n_dim; ++d) {
        std::vector<std::vector<double>> sub_points;
        sub_points.reserve(n_points);

        // Partition points
        for (int i = 0; i < n_points; ++i) {
            if (points[i][d] > pivot[d]) {
                // Create sub-problem point
                std::vector<double> sub_point(n_dim);
                for (int k = 0; k < n_dim; ++k) {
                    sub_point[k] = points[i][k];
                }
                sub_point[d] = points[i][d] - pivot[d];

                // Update original point for subsequent dimensions
                points[i][d] = pivot[d];

                sub_points.push_back(std::move(sub_point));
            }
        }

        if (!sub_points.empty()) {
            total_hv += recursive_hv(sub_points, n_dim, depth + 1);
        }
    }

    return total_hv;
}

double HypervolumeCalculator::monte_carlo_approx(
    const std::vector<std::vector<double>>& points, int n_dim) {

    int n_points = static_cast<int>(points.size());
    if (n_points == 0) {
        return 0.0;
    }

    // Find bounding box
    std::vector<double> upper_bound(n_dim, 0.0);
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < n_dim; ++j) {
            upper_bound[j] = std::max(upper_bound[j], points[i][j]);
        }
    }

    // Compute bounding box volume
    double box_vol = 1.0;
    for (int j = 0; j < n_dim; ++j) {
        box_vol *= upper_bound[j];
    }

    if (box_vol <= 0.0) {
        return 0.0;
    }

    // Monte Carlo sampling: count how many random points are dominated
    int dominated_count = 0;

    // Limit samples based on problem size
    int actual_samples = std::min(n_samples_, 100000);

    #pragma omp parallel reduction(+:dominated_count)
    {
        std::mt19937_64 rng(std::random_device{}() + omp_get_thread_num() * 12345);
        std::vector<std::uniform_real_distribution<double>> dists(n_dim);
        for (int d = 0; d < n_dim; ++d) {
            dists[d] = std::uniform_real_distribution<double>(0.0, upper_bound[d]);
        }

        std::vector<double> sample(n_dim);

        #pragma omp for schedule(static)
        for (int s = 0; s < actual_samples; ++s) {
            // Generate random sample in bounding box
            for (int d = 0; d < n_dim; ++d) {
                sample[d] = dists[d](rng);
            }

            // Check if sample is dominated by any point
            for (int i = 0; i < n_points; ++i) {
                if (dominates(points[i], sample, n_dim)) {
                    ++dominated_count;
                    break;
                }
            }
        }
    }

    // Estimate hypervolume
    return box_vol * (static_cast<double>(dominated_count) / actual_samples);
}

int HypervolumeCalculator::find_pivot(const std::vector<std::vector<double>>& points,
                                       int n_dim) {
    int n_points = static_cast<int>(points.size());
    int max_idx = 0;
    double max_vol = 0.0;

    for (int i = 0; i < n_points; ++i) {
        double vol = 1.0;
        for (int j = 0; j < n_dim; ++j) {
            vol *= points[i][j];
        }
        if (vol > max_vol) {
            max_vol = vol;
            max_idx = i;
        }
    }

    return max_idx;
}

bool HypervolumeCalculator::dominates(const std::vector<double>& a,
                                       const std::vector<double>& b, int n_dim) {
    // a dominates b if a[i] >= b[i] for all i (maximization)
    for (int i = 0; i < n_dim; ++i) {
        if (a[i] < b[i]) {
            return false;
        }
    }
    return true;
}
