// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// RapidCheck mirror of MultiLevelSchwarz.lean's six native_decide
// proofs. 4-level toy hierarchy: 8 fine → 4 → 2 → 1.

#include "curvenet/multi_level_schwarz.h"

#include <rapidcheck.h>
#include <cmath>
#include <vector>

namespace mls = curvenet::multi_level_schwarz;

namespace {

bool vec_close(const std::vector<double> &a, const std::vector<double> &b, double eps) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) >= eps) return false;
    }
    return true;
}

mls::Hierarchy make_hier_8421() {
    mls::Hierarchy h;
    h.cmaps = {
        { 0, 0, 1, 1, 2, 2, 3, 3 },   // 8 → 4
        { 0, 0, 1, 1 },                // 4 → 2
        { 0, 0 },                      // 2 → 1
    };
    h.num_coarse_at = { 8, 4, 2, 1 };
    return h;
}

} // namespace

int main() {
    bool ok = true;
    const auto h = make_hier_8421();

    ok &= rc::check("fine_size = 8, top_size = 1, num_levels = 4", [&] {
        RC_ASSERT(mls::fine_size(h) == 8);
        RC_ASSERT(mls::top_size(h) == 1);
        RC_ASSERT(mls::num_levels(h) == 4);
    });

    ok &= rc::check("restrict_through sums all fine values", [&] {
        const std::vector<double> f = { 1, 2, 3, 4, 5, 6, 7, 8 };
        const auto r = mls::restrict_through(h, f);
        RC_ASSERT(vec_close(r, { 36.0 }, 1e-12));
    });

    ok &= rc::check("prolong_through broadcasts top to all fine", [&] {
        const std::vector<double> c = { 36.0 };
        const auto p = mls::prolong_through(h, c);
        const std::vector<double> expected(8, 36.0);
        RC_ASSERT(vec_close(p, expected, 1e-12));
    });

    ok &= rc::check("restrict ∘ prolong scales by total fine count", [&] {
        const std::vector<double> c = { 5.0 };
        const auto back = mls::restrict_through(h, mls::prolong_through(h, c));
        RC_ASSERT(vec_close(back, { 40.0 }, 1e-12));   // 5 * 8 fine verts
    });

    ok &= rc::check("ones vector through restrict = total fine count", [&] {
        const std::vector<double> ones(8, 1.0);
        const auto r = mls::restrict_through(h, ones);
        RC_ASSERT(vec_close(r, { 8.0 }, 1e-12));
    });

    ok &= rc::check("intermediate-level restrict captures partial sums", [&] {
        const std::vector<double> f = { 1, 2, 3, 4, 5, 6, 7, 8 };
        // Apply only the first cmap: 8 → 4. Pairs sum: (3, 7, 11, 15).
        const auto v = curvenet::two_level_schwarz::restrict_fine(
            h.cmaps[0], h.num_coarse_at[1], f);
        RC_ASSERT(vec_close(v, { 3.0, 7.0, 11.0, 15.0 }, 1e-12));
    });

    return ok ? 0 : 1;
}
