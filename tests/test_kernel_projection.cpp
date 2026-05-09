// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of KernelProjection.lean.

#include "curvenet/kernel_projection.h"

#include <rapidcheck.h>
#include <cmath>
#include <vector>

namespace kp = curvenet::kernel_projection;

int main() {
    bool ok = true;

    ok &= rc::check("[1,2,3,4] zero_mean = [-1.5,-0.5,0.5,1.5]", [] {
        const auto z = kp::zero_mean({ 1.0, 2.0, 3.0, 4.0 });
        RC_ASSERT(std::fabs(z[0] - (-1.5)) < 1e-12);
        RC_ASSERT(std::fabs(z[1] - (-0.5)) < 1e-12);
        RC_ASSERT(std::fabs(z[2] -   0.5)  < 1e-12);
        RC_ASSERT(std::fabs(z[3] -   1.5)  < 1e-12);
        RC_ASSERT(std::fabs(kp::mean(z))   < 1e-12);
    });

    ok &= rc::check("constant vector projects to zero", [] {
        const auto z = kp::zero_mean({ 7.0, 7.0, 7.0, 7.0, 7.0 });
        for (double x : z) RC_ASSERT(std::fabs(x) < 1e-12);
    });

    ok &= rc::check("zero-mean vector unchanged", [] {
        const std::vector<double> v = { -1.5, -0.5, 0.5, 1.5 };
        const auto z = kp::zero_mean(v);
        for (std::size_t i = 0; i < v.size(); ++i) {
            RC_ASSERT(std::fabs(z[i] - v[i]) < 1e-12);
        }
    });

    ok &= rc::check("idempotent: zero_mean ∘ zero_mean = zero_mean", [] {
        const std::vector<double> v = { 2.5, -3.0, 1.0, 4.5, -7.0 };
        const auto z  = kp::zero_mean(v);
        const auto zz = kp::zero_mean(z);
        for (std::size_t i = 0; i < v.size(); ++i) {
            RC_ASSERT(std::fabs(zz[i] - z[i]) < 1e-12);
        }
    });

    ok &= rc::check("empty input is empty output", [] {
        const auto z = kp::zero_mean(std::vector<double>{});
        RC_ASSERT(z.empty());
    });

    return ok ? 0 : 1;
}
