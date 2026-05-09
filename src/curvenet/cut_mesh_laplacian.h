// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_CUT_MESH_LAPLACIAN_H
#define CURVENET_CUT_MESH_LAPLACIAN_H

// C++ mirror of `lean/Curvenet/CutMeshLaplacian.lean`. Builds the
// halfedge-corner Laplacian Lₕ from per-cut-face polygon Laplacians,
// the V map from cut-mesh halfedges to mesh-vertex unknowns, and the
// shared LHS VᵀLₕV for the §4.3 solve.

#include <cstddef>
#include <vector>

#include "cut_mesh.h"
#include "dense_linalg.h"
#include "halfedge.h"
#include "polygon_laplacian.h"
#include "vec3.h"

namespace curvenet {
namespace cut_mesh_laplacian {

inline std::vector<std::size_t> face_loop(const cut_mesh::CutMesh &m, std::size_t face_id) {
	const std::size_t nh = m.he_count();
	std::size_t start = nh;
	for (std::size_t h = 0; h < nh; ++h) {
		const OptionalIndex f = m.base.halfedges[h].face;
		if (f.has_value() && static_cast<std::size_t>(f.unwrap()) == face_id) {
			start = h;
			break;
		}
	}
	if (start == nh) {
		return {};
	}
	std::vector<std::size_t> acc = { start };
	std::size_t cur = m.base.halfedges[start].next;
	std::size_t steps = 0;
	while (cur != start && steps < nh) {
		acc.push_back(cur);
		cur = m.base.halfedges[cur].next;
		++steps;
	}
	return acc;
}

inline std::vector<Vec3> face_polygon(const cut_mesh::CutMesh &m,
                                        const std::vector<Vec3> &positions,
                                        std::size_t face_id) {
	const std::vector<std::size_t> halfedges = face_loop(m, face_id);
	std::vector<Vec3> poly;
	poly.reserve(halfedges.size());
	for (const std::size_t h : halfedges) {
		poly.push_back(positions[m.base.halfedges[h].target]);
	}
	return poly;
}

inline std::vector<double> assemble_lh(const cut_mesh::CutMesh &m,
                                        const std::vector<Vec3> &positions) {
	const std::size_t nh = m.he_count();
	std::vector<double> L(nh * nh, 0.0);
	for (std::size_t f = 0; f < m.base.face_count; ++f) {
		const std::vector<std::size_t> halfedges = face_loop(m, f);
		const std::vector<Vec3> poly = face_polygon(m, positions, f);
		const std::size_t nf = poly.size();
		if (nf < 3) {
			continue;
		}
		const std::vector<double> Lf = polygon_laplacian::polygon_cot_laplacian(poly);
		for (std::size_t li = 0; li < nf; ++li) {
			for (std::size_t lj = 0; lj < nf; ++lj) {
				const std::size_t gi = halfedges[li];
				const std::size_t gj = halfedges[lj];
				dense::set_at(L, nh, gi, gj,
				              dense::get_at(L, nh, gi, gj) +
				              polygon_laplacian::get_at(Lf, nf, li, lj));
			}
		}
	}
	return L;
}

inline std::vector<double> assemble_v(const cut_mesh::CutMesh &m) {
	const std::size_t nh = m.he_count();
	const std::size_t nv = m.vertex_count();
	std::vector<double> V(nh * nv, 0.0);
	for (std::size_t h = 0; h < nh; ++h) {
		const int col = cut_mesh::v_column_of(m, h);
		if (col >= 0) {
			dense::set_at(V, nv, h, static_cast<std::size_t>(col), 1.0);
		}
	}
	return V;
}

inline std::vector<double> assemble_vt_lh_v(const cut_mesh::CutMesh &m,
                                              const std::vector<Vec3> &positions) {
	const std::size_t nh = m.he_count();
	const std::size_t nv = m.vertex_count();
	const std::vector<double> L  = assemble_lh(m, positions);
	const std::vector<double> V  = assemble_v(m);
	const std::vector<double> Vt = dense::transpose(nh, nv, V);
	const std::vector<double> LV = dense::mat_mul(nh, nh, nv, L, V);
	return dense::mat_mul(nv, nh, nv, Vt, LV);
}

} // namespace cut_mesh_laplacian
} // namespace curvenet

#endif
