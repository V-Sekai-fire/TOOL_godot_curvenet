// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of DirectDeltaMush.lean and DirectDeltaMushBind.lean.

#include "curvenet/direct_delta_mush.h"

#include <rapidcheck.h>
#include <cmath>
#include <vector>

namespace ddm = curvenet::direct_delta_mush;

namespace {

bool vclose(const std::array<double, 3> &a, const std::array<double, 3> &b, double eps) {
    return std::fabs(a[0] - b[0]) < eps
        && std::fabs(a[1] - b[1]) < eps
        && std::fabs(a[2] - b[2]) < eps;
}

} // namespace

int main() {
    bool ok = true;

    // ----- runtime kernel -----

    ok &= rc::check("identity transform preserves rest pose", [&] {
        const ddm::Influences infl = { {0, 1.0} };
        const std::vector<ddm::Mat4> T = { ddm::mat4_identity() };
        const auto out = ddm::lbs_matvec(T, infl, 1.5, -2.0, 0.75);
        RC_ASSERT(vclose(out, {1.5, -2.0, 0.75}, 1e-12));
    });

    ok &= rc::check("translation transform shifts uniformly", [&] {
        const ddm::Influences infl = { {0, 1.0} };
        const std::vector<ddm::Mat4> T = { ddm::mat4_translation(2.0, -3.0, 1.0) };
        const auto out = ddm::lbs_matvec(T, infl, 1.0, 1.0, 1.0);
        RC_ASSERT(vclose(out, {3.0, -2.0, 2.0}, 1e-12));
    });

    ok &= rc::check("two-handle equal blend with both identity = rest", [&] {
        const ddm::Influences infl = { {0, 0.5}, {1, 0.5} };
        const std::vector<ddm::Mat4> T = { ddm::mat4_identity(), ddm::mat4_identity() };
        const auto out = ddm::lbs_matvec(T, infl, 4.0, -1.0, 2.0);
        RC_ASSERT(vclose(out, {4.0, -1.0, 2.0}, 1e-12));
    });

    ok &= rc::check("0.7/0.3 blend of opposing translations averages", [&] {
        const ddm::Influences infl = { {0, 0.7}, {1, 0.3} };
        const std::vector<ddm::Mat4> T = {
            ddm::mat4_translation(1.0, 0.0, 0.0),
            ddm::mat4_translation(-1.0, 0.0, 0.0)
        };
        const auto out = ddm::lbs_matvec(T, infl, 0.0, 0.0, 0.0);
        // 0.7 * (+1) + 0.3 * (-1) = +0.4
        RC_ASSERT(vclose(out, {0.4, 0.0, 0.0}, 1e-12));
    });

    ok &= rc::check("empty influences yield zero vector", [&] {
        const ddm::Influences infl = {};
        const std::vector<ddm::Mat4> T = { ddm::mat4_identity() };
        const auto out = ddm::lbs_matvec(T, infl, 1.0, 1.0, 1.0);
        RC_ASSERT(vclose(out, {0.0, 0.0, 0.0}, 1e-12));
    });

    ok &= rc::check("4-handle uniform 1/4 blend with all identity = rest", [&] {
        const ddm::Influences infl = { {0, 0.25}, {1, 0.25}, {2, 0.25}, {3, 0.25} };
        const std::vector<ddm::Mat4> T(4, ddm::mat4_identity());
        const auto out = ddm::lbs_matvec(T, infl, 7.0, 8.0, 9.0);
        RC_ASSERT(vclose(out, {7.0, 8.0, 9.0}, 1e-12));
    });

    // ----- bind-time post-processing -----

    // Path-4 graph adjacency: 0-1-2-3.
    const ddm::Adjacency path4 = {
        {1},      // vertex 0's neighbors
        {0, 2},   // vertex 1's neighbors
        {1, 3},   // vertex 2's neighbors
        {2}       // vertex 3's neighbors
    };

    // Identity weights: each vertex has its own handle with weight 1.
    const ddm::WeightMatrix identityW = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    ok &= rc::check("identity W has partition of unity", [&] {
        RC_ASSERT(ddm::partition_of_unity(identityW, 1e-12));
    });

    // All-half weights: every vertex has equal influence on the two handles.
    const ddm::WeightMatrix halfW = {
        {0.5, 0.5},
        {0.5, 0.5},
        {0.5, 0.5},
        {0.5, 0.5}
    };

    ok &= rc::check("uniform-half W has partition of unity", [&] {
        RC_ASSERT(ddm::partition_of_unity(halfW, 1e-12));
    });

    ok &= rc::check("smoothing preserves uniform-half W (Laplacian eigen 0)", [&] {
        const ddm::WeightMatrix W1 = ddm::smooth_weights(halfW, path4, 5, 0.5);
        RC_ASSERT(ddm::partition_of_unity(W1, 1e-9));
        for (const auto &row : W1) {
            RC_ASSERT(std::fabs(row[0] - 0.5) < 1e-9);
            RC_ASSERT(std::fabs(row[1] - 0.5) < 1e-9);
        }
    });

    ok &= rc::check("smoothing preserves partition of unity on identityW", [&] {
        const ddm::WeightMatrix W1 = ddm::smooth_weights(identityW, path4, 2, 0.5);
        RC_ASSERT(ddm::partition_of_unity(W1, 1e-9));
    });

    // Top-K sparsification on a single-row [0.4, 0.3, 0.2, 0.1] with K = 2.
    // Expected: keep 0.4 and 0.3, renormalize: 0.4/0.7, 0.3/0.7 ≈ 4/7, 3/7.
    const ddm::WeightMatrix skewW = { {0.4, 0.3, 0.2, 0.1} };

    ok &= rc::check("sparsifyTopK keeps K entries", [&] {
        const auto sp = ddm::sparsify_top_k(skewW, 2);
        RC_ASSERT(sp.size() == 1u);
        RC_ASSERT(sp[0].size() == 2u);
    });

    ok &= rc::check("sparsifyTopK renormalizes to sum 1", [&] {
        const auto sp = ddm::sparsify_top_k(skewW, 2);
        RC_ASSERT(std::fabs(ddm::sparse_row_sum(sp[0]) - 1.0) < 1e-12);
    });

    ok &= rc::check("sparsifyTopK retains the largest entries", [&] {
        const auto sp = ddm::sparsify_top_k(skewW, 2);
        RC_ASSERT(std::fabs(sp[0][0].second - 4.0 / 7.0) < 1e-12);
        RC_ASSERT(std::fabs(sp[0][1].second - 3.0 / 7.0) < 1e-12);
        RC_ASSERT(sp[0][0].first == 0);
        RC_ASSERT(sp[0][1].first == 1);
    });

    ok &= rc::check("composition: sparsify(smooth(identityW)) preserves PoU", [&] {
        const ddm::WeightMatrix W1 = ddm::smooth_weights(identityW, path4, 2, 0.5);
        const auto sp = ddm::sparsify_top_k(W1, 2);
        for (const auto &row : sp) {
            RC_ASSERT(std::fabs(ddm::sparse_row_sum(row) - 1.0) < 1e-9);
        }
    });

    return ok ? 0 : 1;
}
