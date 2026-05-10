// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Loop 100/9, cycle 1: measure how non-Delaunay the Mire 5k mesh
// is. For each interior edge with two adjacent triangles forming
// a quadrilateral with opposite angles alpha, beta:
//
//   * Delaunay iff alpha + beta <= pi  iff  cot(alpha) + cot(beta) >= 0
//   * Non-Delaunay edges are sliver indicators that contribute
//     negative weights to the cot-Laplacian, blowing up kappa.
//
// If the fraction of non-Delaunay edges is small (<5%), edge flips
// won't help much. If large (>20%), Sharp & Crane intrinsic
// Delaunay flipping can drop kappa significantly.

#include "curvenet/halfedge_builder.h"
#include "curvenet/vec3.h"
#include "mire_body_data.h"

#include <cmath>
#include <cstdio>
#include <vector>

using curvenet::Vec3;

namespace {

double dot(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 sub(const Vec3 &a, const Vec3 &b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return { a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x };
}

double norm(const Vec3 &a) { return std::sqrt(dot(a, a)); }

// Cotangent of the angle at vertex w in triangle (u, v, w),
// opposite the edge (u, v). Standard formula:
//   cot(theta_w) = ((u-w) . (v-w)) / |((u-w) x (v-w))|.
double cot_at(const Vec3 &u, const Vec3 &v, const Vec3 &w) {
    const Vec3 a = sub(u, w);
    const Vec3 b = sub(v, w);
    const double n = norm(cross(a, b));
    return n == 0.0 ? 0.0 : dot(a, b) / n;
}

} // namespace

int main() {
    std::vector<Vec3> positions;
    positions.reserve(mire_body::n_verts);
    for (int i = 0; i < mire_body::n_verts; ++i) {
        positions.push_back({
            static_cast<double>(mire_body::positions[i * 3 + 0]),
            static_cast<double>(mire_body::positions[i * 3 + 1]),
            static_cast<double>(mire_body::positions[i * 3 + 2]),
        });
    }
    std::vector<int> tris(mire_body::tris,
                            mire_body::tris + mire_body::n_tris * 3);
    const std::size_t nv = positions.size();
    const std::size_t nt = tris.size() / 3;
    std::printf("Mire body 5k: nv=%zu, nt=%zu\n", nv, nt);

    // Build an edge -> opposite-vertex list. For interior edges we
    // get exactly 2 opposites; boundary edges get 1.
    // Edge key = (min(a,b), max(a,b)).
    struct Edge {
        int a, b;
        int opp[2];   // up to 2 opposite vertices
        int opp_count;
    };
    std::vector<Edge> edges;
    edges.reserve(nt * 3 / 2);
    // We use a flat scan (O(nt^2) lookup is fine at 10k tris).
    auto find_or_add = [&](int a, int b, int opp) {
        const int lo = std::min(a, b);
        const int hi = std::max(a, b);
        for (auto &e : edges) {
            if (e.a == lo && e.b == hi) {
                if (e.opp_count < 2) e.opp[e.opp_count++] = opp;
                return;
            }
        }
        edges.push_back({ lo, hi, { opp, -1 }, 1 });
    };
    for (std::size_t t = 0; t < nt; ++t) {
        const int v[3] = { tris[t * 3], tris[t * 3 + 1], tris[t * 3 + 2] };
        find_or_add(v[0], v[1], v[2]);
        find_or_add(v[1], v[2], v[0]);
        find_or_add(v[0], v[2], v[1]);
    }
    std::printf("edges: %zu total\n", edges.size());

    // Classify each interior edge.
    std::size_t n_interior = 0;
    std::size_t n_nondel = 0;
    double min_cotsum = 1e300;
    double max_cotsum = -1e300;
    std::size_t n_negweight = 0;
    for (const auto &e : edges) {
        if (e.opp_count != 2) continue;
        ++n_interior;
        const Vec3 &u = positions[e.a];
        const Vec3 &v = positions[e.b];
        const Vec3 &w0 = positions[e.opp[0]];
        const Vec3 &w1 = positions[e.opp[1]];
        const double c0 = cot_at(u, v, w0);
        const double c1 = cot_at(u, v, w1);
        const double sum = c0 + c1;
        if (sum < min_cotsum) min_cotsum = sum;
        if (sum > max_cotsum) max_cotsum = sum;
        if (sum < 0.0) ++n_nondel;
        // The cot-Laplacian off-diagonal is -(cot α + cot β)/2;
        // negative cotsum = positive off-diagonal weight (wrong sign)
        // = "negative weight" in the M-matrix sense.
        if (sum < 0.0) ++n_negweight;
    }
    std::printf("interior edges: %zu  (%.1f%% of total)\n",
                  n_interior, 100.0 * n_interior / edges.size());
    std::printf("non-Delaunay edges (cot-sum < 0): %zu  (%.2f%% of interior)\n",
                  n_nondel, 100.0 * n_nondel / n_interior);
    std::printf("cot-sum range: [%.4e, %.4e]\n", min_cotsum, max_cotsum);
    if (n_nondel > 0) {
        std::printf("worst non-Delaunay (most negative cot-sum): %.4e\n",
                      min_cotsum);
    }
    return 0;
}
