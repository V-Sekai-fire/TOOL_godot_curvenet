// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_HARMONIC_SOLVE_H
#define CURVENET_HARMONIC_SOLVE_H

// C++ mirror of `lean/Curvenet/HarmonicSolve.lean`. Scalar and
// matrix-valued (k-column) harmonic interpolation on a cut-mesh:
//   (VᵀLₕV) X_v = − Vᵀ Lₕ (C F_c)
// Used as the §4.3 Eq. (6a) deformation-gradient stage and as a
// stripped-down Eq. (6b) when y_h is folded into Fc.

#include <cstddef>
#include <vector>

#include "cut_mesh.h"
#include "cut_mesh_laplacian.h"
#include "dense_linalg.h"
#include "vec3.h"

namespace curvenet {
namespace harmonic_solve {

// Per-halfedge scalar vector built from per-sample scalar boundary
// values. fc has length nc (the total sample-column count).
template <typename SampleColumnFn>
inline std::vector<double> compute_cfc(const cut_mesh::CutMesh &m,
                                         SampleColumnFn &&pack,
                                         const std::vector<double> &fc) {
	const std::size_t nh = m.he_count();
	std::vector<double> out(nh, 0.0);
	for (std::size_t h = 0; h < nh; ++h) {
		const int c = cut_mesh::c_column_of(m, h, pack);
		if (c >= 0) {
			out[h] = fc[c];
		}
	}
	return out;
}

// Per-halfedge nh × k matrix built from a per-sample nc × k matrix Fc.
template <typename SampleColumnFn>
inline std::vector<double> compute_c_fc_matrix(const cut_mesh::CutMesh &m,
                                                SampleColumnFn &&pack,
                                                const std::vector<double> &Fc,
                                                std::size_t k) {
	const std::size_t nh = m.he_count();
	std::vector<double> out(nh * k, 0.0);
	for (std::size_t h = 0; h < nh; ++h) {
		const int c = cut_mesh::c_column_of(m, h, pack);
		if (c >= 0) {
			for (std::size_t j = 0; j < k; ++j) {
				out[h * k + j] = Fc[static_cast<std::size_t>(c) * k + j];
			}
		}
	}
	return out;
}

// Scalar harmonic solve: (VᵀLₕV) x_v = − Vᵀ Lₕ (C f_c). Returns x_v
// of size m.vertex_count(). Promoted-vertex slots come back zero
// (degenerate Gaussian elim row), which the caller can overlay with
// sample values if needed.
template <typename SampleColumnFn>
inline std::vector<double> solve_scalar(const cut_mesh::CutMesh &m,
                                          const std::vector<Vec3> &positions,
                                          SampleColumnFn &&pack,
                                          const std::vector<double> &fc) {
	const std::size_t nh = m.he_count();
	const std::size_t nv = m.vertex_count();
	const std::vector<double> Lh = cut_mesh_laplacian::assemble_lh(m, positions);
	const std::vector<double> V  = cut_mesh_laplacian::assemble_v(m);
	const std::vector<double> Vt = dense::transpose(nh, nv, V);
	const std::vector<double> LV = dense::mat_mul(nh, nh, nv, Lh, V);
	const std::vector<double> lhs = dense::mat_mul(nv, nh, nv, Vt, LV);
	const std::vector<double> cfc = compute_cfc(m, pack, fc);
	const std::vector<double> Lh_cfc = dense::mat_vec(nh, nh, Lh, cfc);
	const std::vector<double> Vt_Lh_cfc = dense::mat_vec(nv, nh, Vt, Lh_cfc);
	std::vector<double> rhs(nv, 0.0);
	for (std::size_t i = 0; i < nv; ++i) {
		rhs[i] = -Vt_Lh_cfc[i];
	}
	return dense::solve(nv, lhs, rhs);
}

// Multi-column harmonic solve. Same pipeline, k right-hand-side columns.
template <typename SampleColumnFn>
inline std::vector<double> solve_multi(const cut_mesh::CutMesh &m,
                                         const std::vector<Vec3> &positions,
                                         SampleColumnFn &&pack,
                                         const std::vector<double> &Fc,
                                         std::size_t k) {
	const std::size_t nh = m.he_count();
	const std::size_t nv = m.vertex_count();
	const std::vector<double> Lh = cut_mesh_laplacian::assemble_lh(m, positions);
	const std::vector<double> V  = cut_mesh_laplacian::assemble_v(m);
	const std::vector<double> Vt = dense::transpose(nh, nv, V);
	const std::vector<double> LV = dense::mat_mul(nh, nh, nv, Lh, V);
	const std::vector<double> lhs = dense::mat_mul(nv, nh, nv, Vt, LV);
	const std::vector<double> CFc = compute_c_fc_matrix(m, pack, Fc, k);
	const std::vector<double> Lh_CFc = dense::mat_mul(nh, nh, k, Lh, CFc);
	const std::vector<double> Vt_Lh_CFc = dense::mat_mul(nv, nh, k, Vt, Lh_CFc);
	std::vector<double> rhs(nv * k, 0.0);
	for (std::size_t i = 0; i < nv * k; ++i) {
		rhs[i] = -Vt_Lh_CFc[i];
	}
	return dense::solve_multi(nv, k, lhs, rhs);
}

} // namespace harmonic_solve
} // namespace curvenet

#endif
