// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// C++ mirror of `lean/Curvenet/RobustLaplacian.lean`. Sharp & Crane
// 2020 intrinsic mollification: pad every edge length by ε to
// guarantee the strict triangle inequality holds with margin δ at
// every corner, then compute cot Laplacian from the mollified lengths
// via law of cosines + Heron — finite even on near-degenerate input.
//
// Diagnosed problem: the embedding-based cot in `polygon_laplacian.h`
// produces ±∞ off-diagonal entries on real character meshes (Mire
// Quest body, 5485 verts), because the cross product `|a × b|` in
// `cot θ = (a · b) / |a × b|` underflows on near-collinear vertex
// pairs. CG then fails to converge: ~5.8 s/frame on the body.
//
// References:
//   * Sharp & Crane 2020, "A Laplacian for Nonmanifold Triangle
//     Meshes" (CGF / SGP 2020). Algorithm description in §4
//     (intrinsic mollification) and §5 (intrinsic Delaunay flips).
//     Bibtex key: `Sharp2020Nonmanifold` in references.bib.
//
// Scope: this file implements §4 only — intrinsic mollification.
// Intrinsic Delaunay flipping (§5) is a follow-up that gives
// nonneg edge weights even on extreme meshes; mollification alone
// fixes the divide-by-zero / inf problem and is enough to unblock
// the deformer's CG.

#ifndef CURVENET_ROBUST_LAPLACIAN_H
#define CURVENET_ROBUST_LAPLACIAN_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "polygon_laplacian.h"
#include "vec3.h"

namespace curvenet {
namespace robust_laplacian {

// Three side lengths of a triangle, labeled by the vertex they
// don't touch:
//   eA = |b - c|, opposite vertex a
//   eB = |a - c|, opposite vertex b
//   eC = |a - b|, opposite vertex c
struct EdgeLengths {
    double eA;
    double eB;
    double eC;
};

inline EdgeLengths edge_lengths_from_triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    auto dist = [](const Vec3 &u, const Vec3 &v) {
        const double dx = u.x - v.x;
        const double dy = u.y - v.y;
        const double dz = u.z - v.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    return EdgeLengths{ dist(b, c), dist(a, c), dist(a, b) };
}

// Smallest non-negative ε such that mollifying every edge of this
// triangle by ε gives  a + b > c + δ  at every corner.
inline double mollify_triangle(double delta, const EdgeLengths &e) {
    const double req_a = delta + e.eA - e.eB - e.eC;
    const double req_b = delta + e.eB - e.eA - e.eC;
    const double req_c = delta + e.eC - e.eA - e.eB;
    return std::max(0.0, std::max(req_a, std::max(req_b, req_c)));
}

inline EdgeLengths apply_epsilon(double eps, const EdgeLengths &e) {
    return EdgeLengths{ e.eA + eps, e.eB + eps, e.eC + eps };
}

// Cotangent of the angle opposite edge `e_opp`, using law of cosines
// and Heron:
//   cos θ = (e1² + e2² - e_opp²) / (2 · e1 · e2)
//   sin θ = (2A) / (e1 · e2)
//   cot θ = (e1² + e2² - e_opp²) / (4A)
// `A` is positive when the triangle inequality holds with positive
// margin (which mollification guarantees), so the result is finite.
inline double cot_from_lengths(double e_opp, double e_adj1, double e_adj2) {
    const double s = (e_opp + e_adj1 + e_adj2) * 0.5;
    const double area = std::sqrt(s * (s - e_opp) * (s - e_adj1) * (s - e_adj2));
    return (e_adj1 * e_adj1 + e_adj2 * e_adj2 - e_opp * e_opp) / (4.0 * area);
}

// 3×3 cot Laplacian for a triangle from mollified edge lengths.
// Same row/col convention as `polygon_laplacian::triangle_cot_laplacian`:
// rows are (a=0, b=1, c=2).
inline std::vector<double> triangle_cot_laplacian_from_lengths(const EdgeLengths &e) {
    const double cotA = cot_from_lengths(e.eA, e.eB, e.eC); // angle at a
    const double cotB = cot_from_lengths(e.eB, e.eA, e.eC); // angle at b
    const double cotC = cot_from_lengths(e.eC, e.eA, e.eB); // angle at c
    const double w_bc = cotA * 0.5;
    const double w_ac = cotB * 0.5;
    const double w_ab = cotC * 0.5;

    std::vector<double> m(9, 0.0);
    polygon_laplacian::set_at(m, 3, 0, 1, -w_ab);
    polygon_laplacian::set_at(m, 3, 1, 0, -w_ab);
    polygon_laplacian::set_at(m, 3, 0, 2, -w_ac);
    polygon_laplacian::set_at(m, 3, 2, 0, -w_ac);
    polygon_laplacian::set_at(m, 3, 1, 2, -w_bc);
    polygon_laplacian::set_at(m, 3, 2, 1, -w_bc);
    polygon_laplacian::set_at(m, 3, 0, 0, w_ab + w_ac);
    polygon_laplacian::set_at(m, 3, 1, 1, w_ab + w_bc);
    polygon_laplacian::set_at(m, 3, 2, 2, w_ac + w_bc);
    return m;
}

// Convenience: the full triangle path (embedded → lengths → mollify
// → Laplacian). Returns the same 3×3 row-major matrix shape as
// `polygon_laplacian::triangle_cot_laplacian`.
inline std::vector<double> robust_triangle_cot_laplacian(
        double delta, const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    const EdgeLengths e   = edge_lengths_from_triangle(a, b, c);
    const double eps      = mollify_triangle(delta, e);
    const EdgeLengths em  = apply_epsilon(eps, e);
    return triangle_cot_laplacian_from_lengths(em);
}

// Two-pass robust polygon Laplacian:
//   pass 1: walk every fan-triangle of every polygon, find the
//           global ε* = max over all corners of the per-corner
//           required mollification.
//   pass 2: assemble the polygon Laplacian using ε* applied
//           uniformly. A single global ε keeps the resulting
//           matrix consistent across shared edges (each edge is
//           in two triangles; both must agree on ε).
//
// `polygons` is a list of polygons (each a vector of Vec3). The
// per-polygon output indexing matches `polygon_cot_laplacian`. The
// caller is responsible for stitching the per-polygon matrices into
// the global cut-mesh Laplacian (this is what
// `cut_mesh_laplacian::assemble_lh_csr` does).
inline double mollify_global(double delta, const std::vector<std::vector<Vec3>> &polygons) {
    double eps_global = 0.0;
    for (const auto &poly : polygons) {
        const std::size_t n = poly.size();
        if (n < 3) continue;
        // Same fan triangulation as polygon_cot_laplacian.
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const EdgeLengths e =
                edge_lengths_from_triangle(poly[0], poly[i], poly[i + 1]);
            const double per_tri = mollify_triangle(delta, e);
            if (per_tri > eps_global) eps_global = per_tri;
        }
    }
    return eps_global;
}

// Build the Laplacian for a single polygon using a globally-determined
// ε. Same fan-triangulation strategy as `polygon_cot_laplacian` so the
// row sums vanish (constants are in the kernel) and the matrix is
// symmetric.
inline std::vector<double> polygon_cot_laplacian_with_epsilon(
        const std::vector<Vec3> &poly, double eps_global) {
    const std::size_t n = poly.size();
    std::vector<double> m(n * n, 0.0);
    if (n < 3) return m;

    for (std::size_t i = 1; i + 1 < n; ++i) {
        const EdgeLengths e =
            edge_lengths_from_triangle(poly[0], poly[i], poly[i + 1]);
        const EdgeLengths em = apply_epsilon(eps_global, e);
        const std::vector<double> tri = triangle_cot_laplacian_from_lengths(em);
        const std::size_t idx[3] = { 0, i, i + 1 };
        for (std::size_t li = 0; li < 3; ++li) {
            for (std::size_t lj = 0; lj < 3; ++lj) {
                polygon_laplacian::add_at(m, n, idx[li], idx[lj],
                    polygon_laplacian::get_at(tri, 3, li, lj));
            }
        }
    }
    return m;
}

} // namespace robust_laplacian
} // namespace curvenet

#endif
