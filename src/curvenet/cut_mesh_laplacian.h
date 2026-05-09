// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
#ifndef CURVENET_CUT_MESH_LAPLACIAN_H
#define CURVENET_CUT_MESH_LAPLACIAN_H

// C++ mirror of `lean/Curvenet/CutMeshLaplacian.lean`. Builds the
// halfedge-corner Laplacian Lₕ from per-cut-face polygon Laplacians,
// the V map from cut-mesh halfedges to mesh-vertex unknowns, and the
// shared LHS VᵀLₕV for the §4.3 solve.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cut_mesh.h"
#include "dense_linalg.h"
#include "halfedge.h"
#include "polygon_laplacian.h"
#include "robust_laplacian.h"
#include "sparse_linalg.h"
#include "vec3.h"

namespace curvenet {
namespace cut_mesh_laplacian {

// Flat (row, col, value) triple used for COO accumulation. Replaces
// `std::map<std::pair<int,int>, double>` in the assembly path —
// at 1.4M entries on an 81k-vert mesh, the map's per-insert tree
// allocation costs ~100 seconds; sort-and-merge on a flat vector is
// 1-2 seconds.
struct CooEntry {
    int    row;
    int    col;
    double val;
};

inline bool coo_less(const CooEntry &a, const CooEntry &b) {
    if (a.row != b.row) return a.row < b.row;
    return a.col < b.col;
}

// Sort the vector by (row, col), merge duplicates by addition,
// and emit the result as a CSR matrix of size `rows × cols`.
inline sparse::SparseMatrixCSR coo_to_csr(std::vector<CooEntry> &&entries,
                                              std::size_t rows, std::size_t cols) {
    std::sort(entries.begin(), entries.end(), coo_less);

    // In-place de-dup: collapse runs of equal (row, col) by summing
    // their `val` into the first occurrence.
    std::size_t out_idx = 0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (out_idx > 0 &&
            entries[out_idx - 1].row == entries[i].row &&
            entries[out_idx - 1].col == entries[i].col) {
            entries[out_idx - 1].val += entries[i].val;
        } else {
            entries[out_idx] = entries[i];
            ++out_idx;
        }
    }
    entries.resize(out_idx);

    sparse::SparseMatrixCSR out;
    out.rows = rows;
    out.cols = cols;
    out.row_ptr.assign(rows + 1, 0);
    for (const auto &e : entries) {
        ++out.row_ptr[static_cast<std::size_t>(e.row) + 1];
    }
    for (std::size_t i = 0; i < rows; ++i) {
        out.row_ptr[i + 1] += out.row_ptr[i];
    }
    out.col_idx.resize(entries.size());
    out.values.resize(entries.size());
    std::vector<int> cursor = out.row_ptr;
    for (const auto &e : entries) {
        const int idx = cursor[e.row]++;
        out.col_idx[idx] = e.col;
        out.values[idx]  = e.val;
    }
    return out;
}

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

// Precompute face_loop for *every* face in O(nh) total. Replaces
// the O(nh*face_count) cost of calling face_loop once per face on
// large meshes (e.g. 81k verts: face_count = 159k, nh = 482k →
// 76 billion ops, ~100 s; with this, ~500k ops, ~1 ms).
//
// Returns a vector indexed by face_id, each element is the loop
// of halfedge indices for that face.
inline std::vector<std::vector<std::size_t>>
all_face_loops(const cut_mesh::CutMesh &m) {
	const std::size_t nh = m.he_count();
	const std::size_t nf = m.base.face_count;
	std::vector<std::size_t> face_start(nf, nh);
	for (std::size_t h = 0; h < nh; ++h) {
		const OptionalIndex f = m.base.halfedges[h].face;
		if (!f.has_value()) continue;
		const std::size_t fid = static_cast<std::size_t>(f.unwrap());
		if (fid < nf && face_start[fid] == nh) {
			face_start[fid] = h;
		}
	}
	std::vector<std::vector<std::size_t>> loops(nf);
	for (std::size_t fid = 0; fid < nf; ++fid) {
		const std::size_t start = face_start[fid];
		if (start == nh) continue;
		loops[fid].push_back(start);
		std::size_t cur = m.base.halfedges[start].next;
		std::size_t steps = 0;
		while (cur != start && steps < nh) {
			loops[fid].push_back(cur);
			cur = m.base.halfedges[cur].next;
			++steps;
		}
	}
	return loops;
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
	const auto loops = all_face_loops(m);
	std::vector<double> L(nh * nh, 0.0);
	for (std::size_t f = 0; f < m.base.face_count; ++f) {
		const auto &halfedges = loops[f];
		std::vector<Vec3> poly;
		poly.reserve(halfedges.size());
		for (std::size_t h : halfedges) {
			poly.push_back(positions[m.base.halfedges[h].target]);
		}
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

// ============================================================
// Sparse equivalents for the perf-critical path.
// ============================================================
//
// `assemble_lh` builds Lₕ as a dense `nh × nh` matrix that's mostly
// zero — for triangle inputs each row has at most ~3 nonzeros from the
// per-face contribution. Storing it as CSR pulls the assembly memory
// from O(nh²) to O(nnz(Lₕ)) ≈ O(nh) and unlocks O(nnz)-per-iter
// `spmv` for the per-frame solve.

// Compute the global mollification ε across every fan-triangle of
// every cut-face. Pass-1 helper for the robust assembly path.
// `delta` is the safety margin Sharp & Crane recommend, scaled to a
// small fraction of the mean edge length of the input mesh.
inline double mollify_epsilon(const cut_mesh::CutMesh &m,
                                const std::vector<Vec3> &positions,
                                double delta) {
    const auto loops = all_face_loops(m);
    std::vector<std::vector<Vec3>> polys;
    polys.reserve(m.base.face_count);
    for (std::size_t f = 0; f < m.base.face_count; ++f) {
        std::vector<Vec3> poly;
        poly.reserve(loops[f].size());
        for (std::size_t h : loops[f]) {
            poly.push_back(positions[m.base.halfedges[h].target]);
        }
        polys.push_back(std::move(poly));
    }
    return robust_laplacian::mollify_global(delta, polys);
}

// Like `mollify_epsilon` but takes precomputed face loops so callers
// who already paid the `all_face_loops` cost don't pay it again.
inline double mollify_epsilon_with_loops(const cut_mesh::CutMesh &m,
                                            const std::vector<Vec3> &positions,
                                            const std::vector<std::vector<std::size_t>> &loops,
                                            double delta) {
    std::vector<std::vector<Vec3>> polys;
    polys.reserve(loops.size());
    for (const auto &loop : loops) {
        std::vector<Vec3> poly;
        poly.reserve(loop.size());
        for (std::size_t h : loop) {
            poly.push_back(positions[m.base.halfedges[h].target]);
        }
        polys.push_back(std::move(poly));
    }
    return robust_laplacian::mollify_global(delta, polys);
}

// Default δ as a fraction of mean edge length. Sharp & Crane suggest
// 1e-5; we use that. The result is then applied uniformly to every
// edge of every fan-triangle of every cut-face.
inline double default_mollify_delta(const std::vector<Vec3> &positions,
                                       const std::vector<int> &tri_indices) {
    double total = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i + 2 < tri_indices.size(); i += 3) {
        const Vec3 &a = positions[tri_indices[i + 0]];
        const Vec3 &b = positions[tri_indices[i + 1]];
        const Vec3 &c = positions[tri_indices[i + 2]];
        auto dist = [](const Vec3 &u, const Vec3 &v) {
            const double dx = u.x - v.x;
            const double dy = u.y - v.y;
            const double dz = u.z - v.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };
        total += dist(a, b) + dist(b, c) + dist(a, c);
        n += 3;
    }
    const double mean = (n > 0) ? (total / static_cast<double>(n)) : 1.0;
    return 1e-5 * mean;
}

// Robust Lₕ assembly. Two passes:
//   1. Walk every fan-triangle, gather edge lengths, find global ε
//      via `robust_laplacian::mollify_triangle`.
//   2. Build per-cut-face polygon Laplacian using mollified lengths,
//      stitch into a CSR `nh × nh` matrix.
// Output is finite even on input meshes with degenerate triangles —
// the embedding-based path produces ±∞ in the same situations.
inline sparse::SparseMatrixCSR assemble_lh_csr_robust(const cut_mesh::CutMesh &m,
                                                          const std::vector<Vec3> &positions,
                                                          double delta) {
    const std::size_t nh = m.he_count();
    const auto loops = all_face_loops(m);
    const double eps = mollify_epsilon_with_loops(m, positions, loops, delta);

    std::vector<CooEntry> entries;
    for (std::size_t f = 0; f < m.base.face_count; ++f) {
        const auto &halfedges = loops[f];
        std::vector<Vec3> poly;
        poly.reserve(halfedges.size());
        for (std::size_t h : halfedges) {
            poly.push_back(positions[m.base.halfedges[h].target]);
        }
        const std::size_t nf = poly.size();
        if (nf < 3) continue;
        const std::vector<double> Lf =
            robust_laplacian::polygon_cot_laplacian_with_epsilon(poly, eps);
        for (std::size_t li = 0; li < nf; ++li) {
            for (std::size_t lj = 0; lj < nf; ++lj) {
                const double v = polygon_laplacian::get_at(Lf, nf, li, lj);
                if (v != 0.0) {
                    entries.push_back({
                        static_cast<int>(halfedges[li]),
                        static_cast<int>(halfedges[lj]),
                        v
                    });
                }
            }
        }
    }
    return coo_to_csr(std::move(entries), nh, nh);
}

// Robust LhsM = Vᵀ·Lₕ·V assembly. Same fan-triangulation +
// mollification as `assemble_lh_csr_robust`, but only includes
// meshVertex × meshVertex entries (sample-promoted slots stay zero
// so the constraint can be applied via overlay).
inline sparse::SparseMatrixCSR assemble_vt_lh_v_csr_robust(const cut_mesh::CutMesh &m,
                                                              const std::vector<Vec3> &positions,
                                                              double delta) {
    const std::size_t nv = m.vertex_count();
    const auto loops = all_face_loops(m);
    const double eps = mollify_epsilon_with_loops(m, positions, loops, delta);

    std::vector<CooEntry> entries;
    for (std::size_t f = 0; f < m.base.face_count; ++f) {
        const auto &halfedges = loops[f];
        std::vector<Vec3> poly;
        poly.reserve(halfedges.size());
        for (std::size_t h : halfedges) {
            poly.push_back(positions[m.base.halfedges[h].target]);
        }
        const std::size_t nf = poly.size();
        if (nf < 3) continue;
        const std::vector<double> Lf =
            robust_laplacian::polygon_cot_laplacian_with_epsilon(poly, eps);
        for (std::size_t li = 0; li < nf; ++li) {
            const int va = cut_mesh::v_column_of(m, halfedges[li]);
            if (va < 0) continue;
            for (std::size_t lj = 0; lj < nf; ++lj) {
                const int vb = cut_mesh::v_column_of(m, halfedges[lj]);
                if (vb < 0) continue;
                const double v = polygon_laplacian::get_at(Lf, nf, li, lj);
                if (v != 0.0) {
                    entries.push_back({ va, vb, v });
                }
            }
        }
    }
    // Make sure every diagonal entry exists so the Jacobi preconditioner
    // can read it. Adding val=0 here is a no-op for existing entries
    // (sort+merge sums them) and ensures missing diagonals are present.
    for (std::size_t i = 0; i < nv; ++i) {
        entries.push_back({ static_cast<int>(i), static_cast<int>(i), 0.0 });
    }
    return coo_to_csr(std::move(entries), nv, nv);
}

// Build Lₕ in CSR. Same per-cut-face polygon-Laplacian sum as
// `assemble_lh`, but the sum is accumulated into a (row, col) → value
// map and converted to CSR at the end. Output rows = cols = nh.
inline sparse::SparseMatrixCSR assemble_lh_csr(const cut_mesh::CutMesh &m,
                                                  const std::vector<Vec3> &positions) {
	const std::size_t nh = m.he_count();
	const auto loops = all_face_loops(m);
	std::vector<CooEntry> entries;
	for (std::size_t f = 0; f < m.base.face_count; ++f) {
		const auto &halfedges = loops[f];
		std::vector<Vec3> poly;
		poly.reserve(halfedges.size());
		for (std::size_t h : halfedges) {
			poly.push_back(positions[m.base.halfedges[h].target]);
		}
		const std::size_t nf = poly.size();
		if (nf < 3) {
			continue;
		}
		const std::vector<double> Lf = polygon_laplacian::polygon_cot_laplacian(poly);
		for (std::size_t li = 0; li < nf; ++li) {
			for (std::size_t lj = 0; lj < nf; ++lj) {
				const double v = polygon_laplacian::get_at(Lf, nf, li, lj);
				if (v != 0.0) {
					entries.push_back({
						static_cast<int>(halfedges[li]),
						static_cast<int>(halfedges[lj]),
						v
					});
				}
			}
		}
	}
	return coo_to_csr(std::move(entries), nh, nh);
}

// Build the shared LHS Vᵀ·Lₕ·V directly in CSR. Each per-face
// polygon-Laplacian contribution L_f[li, lj] is added to the LHS at
// `(target(li), target(lj))` *only* when both targets are mesh-vertices
// (sample-promoted vertices contribute to the C operator instead and
// produce zero rows in the dense LHS — here those rows are simply
// absent from the CSR, which the caller handles by overlaying sample
// positions on the deformed-vertex output).
//
// The CSR is `nv × nv` (full vertex space) with the promoted-vertex
// rows empty. Diagonal entries always exist (zero-valued for promoted
// rows) so `sparse::cg`'s Jacobi preconditioner doesn't divide by
// missing values.
inline sparse::SparseMatrixCSR assemble_vt_lh_v_csr(const cut_mesh::CutMesh &m,
                                                       const std::vector<Vec3> &positions) {
	const std::size_t nv = m.vertex_count();
	const auto loops = all_face_loops(m);
	std::vector<CooEntry> entries;
	for (std::size_t f = 0; f < m.base.face_count; ++f) {
		const auto &halfedges = loops[f];
		std::vector<Vec3> poly;
		poly.reserve(halfedges.size());
		for (std::size_t h : halfedges) {
			poly.push_back(positions[m.base.halfedges[h].target]);
		}
		const std::size_t nf = poly.size();
		if (nf < 3) {
			continue;
		}
		const std::vector<double> Lf = polygon_laplacian::polygon_cot_laplacian(poly);
		for (std::size_t li = 0; li < nf; ++li) {
			const int va = cut_mesh::v_column_of(m, halfedges[li]);
			if (va < 0) {
				continue;
			}
			for (std::size_t lj = 0; lj < nf; ++lj) {
				const int vb = cut_mesh::v_column_of(m, halfedges[lj]);
				if (vb < 0) {
					continue;
				}
				const double v = polygon_laplacian::get_at(Lf, nf, li, lj);
				if (v != 0.0) {
					entries.push_back({ va, vb, v });
				}
			}
		}
	}
	for (std::size_t i = 0; i < nv; ++i) {
		entries.push_back({ static_cast<int>(i), static_cast<int>(i), 0.0 });
	}
	return coo_to_csr(std::move(entries), nv, nv);
}

} // namespace cut_mesh_laplacian
} // namespace curvenet

#endif
