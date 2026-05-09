// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// C++ mirror of `lean/Curvenet/ChebyshevAccel.lean`.
//
// Wang 2015's Chebyshev semi-iterative acceleration. Wraps any base
// iteration `step(x) -> Δ` in a 3-term recurrence that contracts
// the error 2-3× faster per outer iter using only the spectral
// radius `ρ` of the iteration matrix.
//
// Recurrence:
//   ω_1   = 2 / (2 − ρ²)
//   ω_k+1 = 4 / (4 − ρ² · ω_k)            for k ≥ 1
//   x_k+1 = ω_k+1 · (γ · Δ_k + x_k − x_{k-1}) + x_{k-1}
//
// In our deployment: `step` is one outer Schwarz sweep (or one
// Jacobi/Richardson sweep). On the deformer's per-meshlet outer
// Schwarz that's been converging at 0.97/iter, Chebyshev with
// ρ ≈ 0.97 is expected to push the effective rate to ≈ 0.7/iter.

#ifndef CURVENET_CHEBYSHEV_ACCEL_H
#define CURVENET_CHEBYSHEV_ACCEL_H

#include <cstddef>
#include <functional>
#include <vector>

namespace curvenet {
namespace chebyshev_accel {

inline double omega1(double rho) {
    return 2.0 / (2.0 - rho * rho);
}

inline double omega_next(double rho, double omega) {
    return 4.0 / (4.0 - rho * rho * omega);
}

// Generic Chebyshev-accelerated iteration. `step` returns
// Δ = (the underlying base iteration's update at x). Caller
// supplies γ (typically 1.0) and ρ (spectral radius of the
// iteration matrix).
inline std::vector<double>
accel(double rho, double gamma,
       std::function<std::vector<double>(const std::vector<double> &)> step,
       const std::vector<double> &x0,
       std::size_t n_iter) {
    const std::size_t n = x0.size();
    std::vector<double> x_prev = x0;
    std::vector<double> x      = x0;
    double omega = 1.0;

    for (std::size_t k = 0; k < n_iter; ++k) {
        const std::vector<double> delta = step(x);
        if (k == 0) {
            std::vector<double> x_next(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                x_next[i] = x[i] + gamma * delta[i];
            }
            omega = omega1(rho);
            x_prev = x;
            x      = std::move(x_next);
        } else {
            omega = omega_next(rho, omega);
            std::vector<double> x_next(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                const double acc_i = gamma * delta[i] + x[i] - x_prev[i];
                x_next[i] = omega * acc_i + x_prev[i];
            }
            x_prev = x;
            x      = std::move(x_next);
        }
    }
    return x;
}

} // namespace chebyshev_accel
} // namespace curvenet

#endif
