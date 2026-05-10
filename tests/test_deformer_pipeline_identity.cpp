// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Correctness test for the per-frame solve sequence used by
// `CurveNetDeformer3D::apply_deformation`. Three asserts on a
// tiny known mesh:
//
//   1. Identity deformation (Fc = I, Xc = rest sample positions)
//      produces output Xv equal to rest mesh positions.
//
//   2. Pure translation (Fc = I, Xc = rest + (dx, dy, dz)) produces
//      output Xv = rest + (dx, dy, dz) at every vertex.
//
//   3. Single-sample translation (move sample 0 by 0.1 in x) does
//      NOT produce wild output — every vertex stays within a bounded
//      distance of the rest mesh.
//
// Per the user's loop-100/6 note: the synthetic-Laplacian benches
// have been measuring the linear-solve subroutine in isolation, but
// nothing has asserted that the full pipeline (compute_c_fc_matrix
// -> sparse PCG -> compute_fh -> average -> compute_yh -> sparse PCG)
// produces sensible geometry on a known input. This test fills that
// gap before any further solver-perf work.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/deform_solve.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/harmonic_solve.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"

#include <rapidcheck.h>
#include <cmath>
#include <vector>

using curvenet::Vec3;
namespace cm  = curvenet::cut_mesh;
namespace cml = curvenet::cut_mesh_laplacian;
namespace ds  = curvenet::deform_solve;
namespace hs  = curvenet::harmonic_solve;
namespace sp  = curvenet::sparse;

namespace {

// 3x3 grid (9 verts, 8 tris) in the xz-plane. The 4 corners
// (0, 2, 6, 8) are promoted to samples; the remaining 5 vertices
// (edges + centre) stay as mesh-vertex. Without at least one
// mesh-vertex column the pipeline degenerates to zero, so 4 verts
// is too small.
struct QuadMesh {
    std::vector<Vec3> positions;
    std::vector<int>  tris;
};

QuadMesh make_quad() {
    QuadMesh m;
    // 3x3 grid: index = z_row * 3 + x_col, x and z in {-1, 0, 1}.
    for (int z = 0; z < 3; ++z) {
        for (int x = 0; x < 3; ++x) {
            m.positions.push_back({
                -1.0 + static_cast<double>(x),
                 0.0,
                -1.0 + static_cast<double>(z)
            });
        }
    }
    // Two tris per cell, 4 cells.
    for (int z = 0; z < 2; ++z) {
        for (int x = 0; x < 2; ++x) {
            const int a = z * 3 + x;
            const int b = a + 1;
            const int c = a + 3;
            const int d = c + 1;
            m.tris.push_back(a); m.tris.push_back(b); m.tris.push_back(d);
            m.tris.push_back(a); m.tris.push_back(d); m.tris.push_back(c);
        }
    }
    return m;
}

int sample_col(int curve_id, int /*sample_idx*/, bool /*side*/) {
    return curve_id;
}

// Run the full §4.3 two-stage solve on a CutMesh with 4 samples.
// Mirrors `CurveNetDeformer3D::apply_deformation` lines 388-466.
std::vector<Vec3> run_pipeline(const cm::CutMesh &c,
                                  const std::vector<Vec3> &positions,
                                  const std::vector<int>  &tris,
                                  const std::vector<double> &Fc,
                                  const std::vector<double> &Xc) {
    const std::size_t nv = positions.size();
    const std::size_t nh = c.base.he_count();

    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR Lh =
        cml::assemble_lh_csr_robust(c, positions, mol_delta);
    const sp::SparseMatrixCSR LhsM =
        cml::assemble_vt_lh_v_csr_robust(c, positions, mol_delta);

    auto apply_vt = [&](const std::vector<double> &Y_he, std::size_t k) {
        std::vector<double> out(nv * k, 0.0);
        for (std::size_t h = 0; h < nh; ++h) {
            const int col = cm::v_column_of(c, h);
            if (col < 0) continue;
            for (std::size_t kk = 0; kk < k; ++kk) {
                out[static_cast<std::size_t>(col) * k + kk] += Y_he[h * k + kk];
            }
        }
        return out;
    };

    const std::size_t cg_max_iter = std::max<std::size_t>(50, nv * 4);
    const double tol = 1e-9;

    // Stage 1: Fv solve.
    const std::vector<double> CFc = hs::compute_c_fc_matrix(c, sample_col, Fc, 9);
    const std::vector<double> Lh_CFc = sp::spmv_multi(Lh, CFc, 9);
    const std::vector<double> Vt_Lh_CFc = apply_vt(Lh_CFc, 9);
    std::vector<double> rhs_a(nv * 9, 0.0);
    for (std::size_t i = 0; i < nv * 9; ++i) rhs_a[i] = -Vt_Lh_CFc[i];
    std::vector<double> Fv(nv * 9, 0.0);
    for (std::size_t k = 0; k < 9; ++k) {
        std::vector<double> b_col(nv);
        for (std::size_t i = 0; i < nv; ++i) b_col[i] = rhs_a[i * 9 + k];
        const std::vector<double> x = sp::cg(LhsM, b_col, cg_max_iter, tol);
        for (std::size_t i = 0; i < nv; ++i) Fv[i * 9 + k] = x[i];
    }

    // Bridge.
    const std::vector<double> Fh = ds::compute_fh(c, sample_col, Fv, Fc, 9);
    const std::vector<double> Ff = ds::average_over_faces(c, Fh, 9);
    const std::vector<double> yh = ds::compute_yh(c, positions, Ff);

    // Stage 2: Xv solve.
    const std::vector<double> CXc = hs::compute_c_fc_matrix(c, sample_col, Xc, 3);
    std::vector<double> diff(nh * 3, 0.0);
    for (std::size_t i = 0; i < nh * 3; ++i) diff[i] = CXc[i] - yh[i];
    const std::vector<double> Lh_diff = sp::spmv_multi(Lh, diff, 3);
    const std::vector<double> Vt_Lh_diff = apply_vt(Lh_diff, 3);
    std::vector<double> rhs_b(nv * 3, 0.0);
    for (std::size_t i = 0; i < nv * 3; ++i) rhs_b[i] = -Vt_Lh_diff[i];
    std::vector<double> Xv(nv * 3, 0.0);
    for (std::size_t k = 0; k < 3; ++k) {
        std::vector<double> b_col(nv);
        for (std::size_t i = 0; i < nv; ++i) b_col[i] = rhs_b[i * 3 + k];
        const std::vector<double> x = sp::cg(LhsM, b_col, cg_max_iter, tol);
        for (std::size_t i = 0; i < nv; ++i) Xv[i * 3 + k] = x[i];
    }

    // Compose the deformed mesh exactly as
    // `CurveNetDeformer3D::apply_deformation` does: sample vertices
    // take their position from Xc (the constraint), mesh-vertex
    // positions come from the Xv solve. Without this composition
    // sample positions are read as 0 (since LhsM has zero rows on
    // sample slots) and the test would always see "no movement".
    std::vector<Vec3> out(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        const auto &kind = c.vertex_kind[i];
        if (kind.tag == cm::CutVertexKindTag::sample) {
            const int col = kind.curve_id;
            out[i] = { Xc[col * 3 + 0], Xc[col * 3 + 1], Xc[col * 3 + 2] };
        } else {
            out[i] = { Xv[i * 3 + 0], Xv[i * 3 + 1], Xv[i * 3 + 2] };
        }
    }
    return out;
}

double max_error(const std::vector<Vec3> &got, const std::vector<Vec3> &want) {
    double m = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        m = std::max(m, std::fabs(got[i].x - want[i].x));
        m = std::max(m, std::fabs(got[i].y - want[i].y));
        m = std::max(m, std::fabs(got[i].z - want[i].z));
    }
    return m;
}

double max_norm(const std::vector<Vec3> &v) {
    double m = 0.0;
    for (const auto &p : v) {
        m = std::max(m, std::fabs(p.x));
        m = std::max(m, std::fabs(p.y));
        m = std::max(m, std::fabs(p.z));
    }
    return m;
}

cm::CutMesh promote_4_corners(const QuadMesh &m) {
    const std::size_t nv = m.positions.size();
    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, m.tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    // Promote the 4 grid corners (indices 0, 2, 6, 8) as sample
    // columns 0..3. Edges + centre (1, 3, 4, 5, 7) stay mesh-vertex.
    c.vertex_kind[0] = cm::CutVertexKind::sample_kind(0, 0, false);
    c.vertex_kind[2] = cm::CutVertexKind::sample_kind(1, 0, false);
    c.vertex_kind[6] = cm::CutVertexKind::sample_kind(2, 0, false);
    c.vertex_kind[8] = cm::CutVertexKind::sample_kind(3, 0, false);
    return c;
}

// For asserting on Xc that's per-sample-column, the test build
// needs sample column 0..3 to map to vertex indices 0, 2, 6, 8.
constexpr int CORNER_VERTS[4] = { 0, 2, 6, 8 };

} // namespace

int main() {
    bool ok = true;
    const QuadMesh m = make_quad();
    const cm::CutMesh c = promote_4_corners(m);

    // Identity Fc per sample (3x3 identity packed row-major into 9 values).
    std::vector<double> Fc_id(4 * 9, 0.0);
    for (int s = 0; s < 4; ++s) {
        Fc_id[s * 9 + 0] = 1.0;
        Fc_id[s * 9 + 4] = 1.0;
        Fc_id[s * 9 + 8] = 1.0;
    }

    // 1. Identity input -> identity output (within solve tolerance).
    ok &= rc::check("identity Fc + rest Xc -> output equals rest", [&] {
        std::vector<double> Xc_rest(4 * 3);
        for (int s = 0; s < 4; ++s) {
            const Vec3 &p = m.positions[CORNER_VERTS[s]];
            Xc_rest[s * 3 + 0] = p.x;
            Xc_rest[s * 3 + 1] = p.y;
            Xc_rest[s * 3 + 2] = p.z;
        }
        const auto Xv = run_pipeline(c, m.positions, m.tris, Fc_id, Xc_rest);
        const double err = max_error(Xv, m.positions);
        std::printf("  identity error: %.3e\n", err);
        RC_ASSERT(err < 1e-6);
    });

    // 2. Pure translation: every sample shifted by (0.5, 0.3, -0.2).
    ok &= rc::check("identity Fc + translated Xc -> output is rest + translation", [&] {
        const Vec3 t { 0.5, 0.3, -0.2 };
        std::vector<double> Xc_t(4 * 3);
        std::vector<Vec3>   want(m.positions.size());
        for (int s = 0; s < 4; ++s) {
            const Vec3 &p = m.positions[CORNER_VERTS[s]];
            Xc_t[s * 3 + 0] = p.x + t.x;
            Xc_t[s * 3 + 1] = p.y + t.y;
            Xc_t[s * 3 + 2] = p.z + t.z;
        }
        for (std::size_t i = 0; i < m.positions.size(); ++i) {
            want[i] = { m.positions[i].x + t.x,
                          m.positions[i].y + t.y,
                          m.positions[i].z + t.z };
        }
        const auto Xv = run_pipeline(c, m.positions, m.tris, Fc_id, Xc_t);
        const double err = max_error(Xv, want);
        std::printf("  translation error: %.3e\n", err);
        RC_ASSERT(err < 1e-6);
    });

    // 3. Bounded output under single-sample perturbation. Sample 0
    //    (corner vertex 0) moves by (0.1, 0, 0). Output must NOT
    //    explode — every vertex stays within a few units of rest.
    ok &= rc::check("single-sample tug -> bounded output", [&] {
        std::vector<double> Xc_tug(4 * 3);
        for (int s = 0; s < 4; ++s) {
            const Vec3 &p = m.positions[CORNER_VERTS[s]];
            Xc_tug[s * 3 + 0] = p.x;
            Xc_tug[s * 3 + 1] = p.y;
            Xc_tug[s * 3 + 2] = p.z;
        }
        Xc_tug[0 * 3 + 0] += 0.1;   // sample 0: x += 0.1
        const auto Xv = run_pipeline(c, m.positions, m.tris, Fc_id, Xc_tug);
        const double max_v = max_norm(Xv);
        const double max_input = max_norm(m.positions);
        std::printf("  tug max |Xv|: %.3e   (input bound: %.3e)\n",
                      max_v, max_input);
        // Output should stay within 5x the input bound. The historical
        // Coons-patch implementation produced 25538 units on this same
        // 2x2 input — a 12000x explosion — so this is the regression
        // gate against THAT class of bug.
        RC_ASSERT(max_v < max_input * 5.0);
    });

    return ok ? 0 : 1;
}
