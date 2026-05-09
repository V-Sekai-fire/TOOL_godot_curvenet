// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_DEFORM_SOLVE_H
#define CURVENET_DEFORM_SOLVE_H

// C++ mirror of `lean/Curvenet/DeformSolve.lean`. Full DeGoes22 §4.3
// two-stage solve: Eq. (6a) for f_v, then bridge via Eq. (3) +
// per-face F_f average + y_h transform, then Eq. (6b) for x_v.

#include <cmath>
#include <cstddef>
#include <vector>

#include "cut_mesh.h"
#include "cut_mesh_laplacian.h"
#include "dense_linalg.h"
#include "halfedge.h"
#include "harmonic_solve.h"
#include "vec3.h"

namespace curvenet {
namespace deform_solve {

// Eq. (3): f_h = V f_v + C f_c. For each halfedge:
//   target is mesh-vertex   → f_h[h] = f_v[target, :]
//   target is sample        → f_h[h] = f_c[sample_col, :]
//   target is edge intersection → 0 (not a load-bearing case here)
template <typename SampleColumnFn>
inline std::vector<double>
compute_fh(const cut_mesh::CutMesh &m, SampleColumnFn &&pack,
           const std::vector<double> &Fv, const std::vector<double> &Fc,
           std::size_t k) {
	const std::size_t nh = m.he_count();
	std::vector<double> Fh(nh * k, 0.0);
	for (std::size_t h = 0; h < nh; ++h) {
		const int v_col = cut_mesh::v_column_of(m, h);
		const int c_col = cut_mesh::c_column_of(m, h, pack);
		if (v_col >= 0) {
			for (std::size_t j = 0; j < k; ++j) {
				Fh[h * k + j] = Fv[static_cast<std::size_t>(v_col) * k + j];
			}
		} else if (c_col >= 0) {
			for (std::size_t j = 0; j < k; ++j) {
				Fh[h * k + j] = Fc[static_cast<std::size_t>(c_col) * k + j];
			}
		}
	}
	return Fh;
}

// Average f_h over each cut-face's halfedges → faceCount × k matrix.
inline std::vector<double> average_over_faces(const cut_mesh::CutMesh &m,
                                                const std::vector<double> &Fh,
                                                std::size_t k) {
	const std::size_t n_faces = m.base.face_count;
	std::vector<double> Ff(n_faces * k, 0.0);
	for (std::size_t f = 0; f < n_faces; ++f) {
		const std::vector<std::size_t> halfedges = cut_mesh_laplacian::face_loop(m, f);
		const std::size_t nf = halfedges.size();
		if (nf == 0) {
			continue;
		}
		const double inv_n = 1.0 / static_cast<double>(nf);
		for (std::size_t j = 0; j < k; ++j) {
			double s = 0.0;
			for (std::size_t i = 0; i < nf; ++i) {
				s += Fh[halfedges[i] * k + j];
			}
			Ff[f * k + j] = s * inv_n;
		}
	}
	return Ff;
}

// y_h[h, :] = positions[target(h)] · F_f^T for the F_f belonging to
// h's face. F_f stored row-major as 9 entries per face. Boundary
// halfedges yield zeros.
inline std::vector<double> compute_yh(const cut_mesh::CutMesh &m,
                                        const std::vector<Vec3> &positions,
                                        const std::vector<double> &Ff) {
	const std::size_t nh = m.he_count();
	std::vector<double> yh(nh * 3, 0.0);
	for (std::size_t h = 0; h < nh; ++h) {
		const Halfedge &he = m.base.halfedges[h];
		if (!he.face.has_value()) {
			continue;
		}
		const std::size_t f = static_cast<std::size_t>(he.face.unwrap());
		const Vec3 pos = positions[he.target];
		const double f00 = Ff[f * 9 + 0];
		const double f01 = Ff[f * 9 + 1];
		const double f02 = Ff[f * 9 + 2];
		const double f10 = Ff[f * 9 + 3];
		const double f11 = Ff[f * 9 + 4];
		const double f12 = Ff[f * 9 + 5];
		const double f20 = Ff[f * 9 + 6];
		const double f21 = Ff[f * 9 + 7];
		const double f22 = Ff[f * 9 + 8];
		yh[h * 3 + 0] = pos.x * f00 + pos.y * f01 + pos.z * f02;
		yh[h * 3 + 1] = pos.x * f10 + pos.y * f11 + pos.z * f12;
		yh[h * 3 + 2] = pos.x * f20 + pos.y * f21 + pos.z * f22;
	}
	return yh;
}

// Full §4.3 deformation solve. Returns nv × 3 vertex positions.
//
// Inputs:
//   m          — cut-mesh (slice 3 layout)
//   positions  — rest 3D positions, indexed by m.vertex_count
//   pack       — packs (curveId, sampleIdx, side) → C-column index
//   Fc         — nc × 9 sample deformation gradients (flattened 3×3)
//   Xc         — nc × 3 sample target positions
template <typename SampleColumnFn>
inline std::vector<double>
solve_deformation(const cut_mesh::CutMesh &m,
                   const std::vector<Vec3> &positions,
                   SampleColumnFn &&pack,
                   const std::vector<double> &Fc,
                   const std::vector<double> &Xc) {
	const std::size_t nh = m.he_count();
	const std::size_t nv = m.vertex_count();

	// Stage 1 (Eq. 6a): solve for f_v.
	const std::vector<double> Fv =
		harmonic_solve::solve_multi(m, positions, pack, Fc, 9);

	// Bridge (Eq. 3 + per-face average): build F_f.
	const std::vector<double> Fh = compute_fh(m, pack, Fv, Fc, 9);
	const std::vector<double> Ff = average_over_faces(m, Fh, 9);

	// y_h from rest polygons + F_f.
	const std::vector<double> yh = compute_yh(m, positions, Ff);

	// Stage 2 (Eq. 6b): assemble RHS = − Vᵀ Lₕ (C X_c − y_h).
	const std::vector<double> Lh = cut_mesh_laplacian::assemble_lh(m, positions);
	const std::vector<double> V  = cut_mesh_laplacian::assemble_v(m);
	const std::vector<double> Vt = dense::transpose(nh, nv, V);
	const std::vector<double> CXc =
		harmonic_solve::compute_c_fc_matrix(m, pack, Xc, 3);
	std::vector<double> diff(nh * 3, 0.0);
	for (std::size_t i = 0; i < nh * 3; ++i) {
		diff[i] = CXc[i] - yh[i];
	}
	const std::vector<double> Lh_diff = dense::mat_mul(nh, nh, 3, Lh, diff);
	const std::vector<double> Vt_Lh_diff = dense::mat_mul(nv, nh, 3, Vt, Lh_diff);
	std::vector<double> rhs(nv * 3, 0.0);
	for (std::size_t i = 0; i < nv * 3; ++i) {
		rhs[i] = -Vt_Lh_diff[i];
	}
	const std::vector<double> LV = dense::mat_mul(nh, nh, nv, Lh, V);
	const std::vector<double> lhs = dense::mat_mul(nv, nh, nv, Vt, LV);
	return dense::solve_multi(nv, 3, lhs, rhs);
}

} // namespace deform_solve
} // namespace curvenet

#endif
