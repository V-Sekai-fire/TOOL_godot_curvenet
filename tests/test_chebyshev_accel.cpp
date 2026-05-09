// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of lean/Curvenet/ChebyshevAccel.lean's four
// native_decide proofs. Same Chebyshev coefficient identities and
// Richardson-on-2x2-SPD convergence test.

#include "curvenet/chebyshev_accel.h"
#include <rapidcheck.h>
#include <cmath>
#include <vector>

namespace cb = curvenet::chebyshev_accel;

int main() {
    bool ok = true;

    ok &= rc::check("omega1(0.9) matches analytical 2/(2-0.81)", [] {
        const double w = cb::omega1(0.9);
        const double expected = 2.0 / (2.0 - 0.81);
        RC_ASSERT(std::fabs(w - expected) < 1e-12);
    });

    ok &= rc::check("recurrence identity: omega_k+1 (1 - rho²/4 omega_k) = 1", [] {
        const double rho = 0.9;
        const double w1 = cb::omega1(rho);
        const double w2 = cb::omega_next(rho, w1);
        const double lhs = w2 * (1.0 - rho * rho * 0.25 * w1);
        RC_ASSERT(std::fabs(lhs - 1.0) < 1e-12);
    });

    ok &= rc::check("omega_k -> 4/(2 + 2 sqrt(1-rho²)) asymptote at k=30", [] {
        const double rho = 0.9;
        double w = cb::omega1(rho);
        for (int i = 1; i < 30; ++i) w = cb::omega_next(rho, w);
        const double limit = 4.0 / (2.0 + 2.0 * std::sqrt(1.0 - rho * rho));
        RC_ASSERT(std::fabs(w - limit) < 1e-3);
    });

    ok &= rc::check("Chebyshev-Richardson on 2x2 SPD reaches answer", [] {
        // A = [[2,1],[1,2]], spec(A) = {3, 1}, lambda_max = 3.
        // A · x = (5, 1) ⇒ x = (3, -1).
        const double A_diag = 2.0;
        const double A_off  = 1.0;
        const double lambda_max = 3.0;
        const std::vector<double> b = { 5.0, 1.0 };
        auto step = [&](const std::vector<double> &x) -> std::vector<double> {
            return {
                (b[0] - (A_diag * x[0] + A_off  * x[1])) / lambda_max,
                (b[1] - (A_off  * x[0] + A_diag * x[1])) / lambda_max,
            };
        };
        const std::vector<double> x0 = { 0.0, 0.0 };
        const auto x = cb::accel(2.0 / 3.0, 1.0, step, x0, 12);
        RC_ASSERT(std::fabs(x[0] - 3.0) < 1e-2);
        RC_ASSERT(std::fabs(x[1] + 1.0) < 1e-2);
    });

    return ok ? 0 : 1;
}
