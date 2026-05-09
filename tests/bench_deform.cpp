// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Wall-clock benchmark for the DeGoes22 deformer at a few mesh sizes.
// Compares the original dense LU factor-once path against the sparse
// CSR + Jacobi-preconditioned CG path the runtime actually uses today.
// See `docs/PERF_BASELINE.md` for the recorded numbers.
//
// Build with the Makefile target `make -C tests bench` (uses -O3,
// not the -O0 the property tests run under).
#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/deform_solve.h"
#include "curvenet/dense_linalg.h"
#include "curvenet/halfedge.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/harmonic_solve.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using curvenet::Vec3;
namespace cm  = curvenet::cut_mesh;
namespace cml = curvenet::cut_mesh_laplacian;
namespace dl  = curvenet::dense;
namespace sp  = curvenet::sparse;
namespace hs  = curvenet::harmonic_solve;
namespace ds  = curvenet::deform_solve;

namespace {

struct GridMesh {
	std::vector<Vec3> positions;
	std::vector<int>  tris;
};

// NxN regular grid in [-1, 1]² × {0}, with 2·(N-1)² triangles.
GridMesh make_plane(int n) {
	GridMesh m;
	m.positions.reserve(static_cast<std::size_t>(n) * n);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			const double x = -1.0 + 2.0 * i / (n - 1);
			const double z = -1.0 + 2.0 * j / (n - 1);
			m.positions.push_back({ x, 0.0, z });
		}
	}
	m.tris.reserve(static_cast<std::size_t>(n - 1) * (n - 1) * 6);
	for (int j = 0; j < n - 1; ++j) {
		for (int i = 0; i < n - 1; ++i) {
			const int a = j * n + i;
			const int b = a + 1;
			const int c = a + n;
			const int d = c + 1;
			m.tris.push_back(a); m.tris.push_back(b); m.tris.push_back(d);
			m.tris.push_back(a); m.tris.push_back(d); m.tris.push_back(c);
		}
	}
	return m;
}

double now_ms() {
	using namespace std::chrono;
	return duration_cast<duration<double, std::milli>>(
		steady_clock::now().time_since_epoch()).count();
}

int sample_col(int curve_id, int /*sample_idx*/, bool /*side*/) {
	return curve_id;
}

struct BenchRow {
	int    n;
	std::size_t nv;
	std::size_t nh;
	double bind_ms;
	double frame_ms_avg;
};

cm::CutMesh build_grid_cut(const GridMesh &g, int n) {
	const std::size_t nv = g.positions.size();
	const curvenet::HalfedgeMesh hm =
		curvenet::halfedge_builder::from_triangles(nv, g.tris);
	cm::CutMesh c;
	c.base = hm;
	c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
	c.segment_of_halfedge.assign(hm.he_count(), -1);
	c.vertex_kind[0]                    = cm::CutVertexKind::sample_kind(0, 0, false);
	c.vertex_kind[n - 1]                = cm::CutVertexKind::sample_kind(1, 0, false);
	c.vertex_kind[(n - 1) * n]          = cm::CutVertexKind::sample_kind(2, 0, false);
	c.vertex_kind[n * n - 1]            = cm::CutVertexKind::sample_kind(3, 0, false);
	return c;
}

std::vector<double> identity_fc() {
	std::vector<double> Fc(4 * 9, 0.0);
	for (int s = 0; s < 4; ++s) {
		Fc[s * 9 + 0] = 1.0;
		Fc[s * 9 + 4] = 1.0;
		Fc[s * 9 + 8] = 1.0;
	}
	return Fc;
}

std::vector<double> drift_xc(int f) {
	const double dy = 0.4 + 0.01 * f;
	return {
		-1.0, dy, -1.0,
		 1.0, dy, -1.0,
		 1.0, dy,  1.0,
		-1.0, dy,  1.0
	};
}

BenchRow run_dense(int n, int frames) {
	const GridMesh g = make_plane(n);
	const std::size_t nv = g.positions.size();
	const cm::CutMesh c = build_grid_cut(g, n);
	const std::size_t nh = c.base.he_count();

	const double t_bind = now_ms();
	const std::vector<double> Lh   = cml::assemble_lh(c, g.positions);
	const std::vector<double> V    = cml::assemble_v(c);
	const std::vector<double> Vt   = dl::transpose(nh, nv, V);
	const std::vector<double> LhV  = dl::mat_mul(nh, nh, nv, Lh, V);
	const std::vector<double> LhsM = dl::mat_mul(nv, nh, nv, Vt, LhV);
	const dl::LUFactor factor = dl::factorize_lu(nv, LhsM);
	const double bind_ms = now_ms() - t_bind;

	const std::vector<double> Fc = identity_fc();

	double frame_total_ms = 0.0;
	for (int f = 0; f < frames; ++f) {
		const std::vector<double> Xc = drift_xc(f);

		const double t0 = now_ms();
		const std::vector<double> CFc =
			hs::compute_c_fc_matrix(c, sample_col, Fc, 9);
		const std::vector<double> Lh_CFc =
			dl::mat_mul(nh, nh, 9, Lh, CFc);
		const std::vector<double> Vt_Lh_CFc =
			dl::mat_mul(nv, nh, 9, Vt, Lh_CFc);
		std::vector<double> rhs_a(nv * 9, 0.0);
		for (std::size_t i = 0; i < nv * 9; ++i) {
			rhs_a[i] = -Vt_Lh_CFc[i];
		}
		const std::vector<double> Fv =
			dl::solve_multi_with_lu(factor, 9, rhs_a);

		const std::vector<double> Fh =
			ds::compute_fh(c, sample_col, Fv, Fc, 9);
		const std::vector<double> Ff =
			ds::average_over_faces(c, Fh, 9);
		const std::vector<double> yh =
			ds::compute_yh(c, g.positions, Ff);

		const std::vector<double> CXc =
			hs::compute_c_fc_matrix(c, sample_col, Xc, 3);
		std::vector<double> diff(nh * 3, 0.0);
		for (std::size_t i = 0; i < nh * 3; ++i) {
			diff[i] = CXc[i] - yh[i];
		}
		const std::vector<double> Lh_diff =
			dl::mat_mul(nh, nh, 3, Lh, diff);
		const std::vector<double> Vt_Lh_diff =
			dl::mat_mul(nv, nh, 3, Vt, Lh_diff);
		std::vector<double> rhs_b(nv * 3, 0.0);
		for (std::size_t i = 0; i < nv * 3; ++i) {
			rhs_b[i] = -Vt_Lh_diff[i];
		}
		const std::vector<double> Xv =
			dl::solve_multi_with_lu(factor, 3, rhs_b);
		(void)Xv;

		frame_total_ms += now_ms() - t0;
	}

	BenchRow row;
	row.n            = n;
	row.nv           = nv;
	row.nh           = nh;
	row.bind_ms      = bind_ms;
	row.frame_ms_avg = frame_total_ms / frames;
	return row;
}

BenchRow run_sparse(int n, int frames) {
	const GridMesh g = make_plane(n);
	const std::size_t nv = g.positions.size();
	const cm::CutMesh c = build_grid_cut(g, n);
	const std::size_t nh = c.base.he_count();

	const double t_bind = now_ms();
	const sp::SparseMatrixCSR Lh_csr   = cml::assemble_lh_csr(c, g.positions);
	const sp::SparseMatrixCSR LhsM_csr = cml::assemble_vt_lh_v_csr(c, g.positions);
	const double bind_ms = now_ms() - t_bind;

	const std::vector<double> Fc = identity_fc();

	auto apply_vt = [&](const std::vector<double> &Y_he, std::size_t k) {
		std::vector<double> out(nv * k, 0.0);
		for (std::size_t h = 0; h < nh; ++h) {
			const int col = cm::v_column_of(c, h);
			if (col < 0) continue;
			for (std::size_t cc = 0; cc < k; ++cc) {
				out[static_cast<std::size_t>(col) * k + cc] += Y_he[h * k + cc];
			}
		}
		return out;
	};

	const std::size_t cg_max_iter = std::max<std::size_t>(50, nv * 2);
	auto cg_multi = [&](std::size_t k, const std::vector<double> &rhs) {
		std::vector<double> X(nv * k, 0.0);
		std::vector<double> b_col(nv, 0.0);
		for (std::size_t cc = 0; cc < k; ++cc) {
			for (std::size_t i = 0; i < nv; ++i) {
				b_col[i] = rhs[i * k + cc];
			}
			const std::vector<double> x_col =
				sp::cg(LhsM_csr, b_col, cg_max_iter, 1e-8);
			for (std::size_t i = 0; i < nv; ++i) {
				X[i * k + cc] = x_col[i];
			}
		}
		return X;
	};

	double frame_total_ms = 0.0;
	for (int f = 0; f < frames; ++f) {
		const std::vector<double> Xc = drift_xc(f);

		const double t0 = now_ms();
		const std::vector<double> CFc =
			hs::compute_c_fc_matrix(c, sample_col, Fc, 9);
		const std::vector<double> Lh_CFc = sp::spmv_multi(Lh_csr, CFc, 9);
		const std::vector<double> Vt_Lh_CFc = apply_vt(Lh_CFc, 9);
		std::vector<double> rhs_a(nv * 9, 0.0);
		for (std::size_t i = 0; i < nv * 9; ++i) {
			rhs_a[i] = -Vt_Lh_CFc[i];
		}
		const std::vector<double> Fv = cg_multi(9, rhs_a);

		const std::vector<double> Fh =
			ds::compute_fh(c, sample_col, Fv, Fc, 9);
		const std::vector<double> Ff =
			ds::average_over_faces(c, Fh, 9);
		const std::vector<double> yh =
			ds::compute_yh(c, g.positions, Ff);

		const std::vector<double> CXc =
			hs::compute_c_fc_matrix(c, sample_col, Xc, 3);
		std::vector<double> diff(nh * 3, 0.0);
		for (std::size_t i = 0; i < nh * 3; ++i) {
			diff[i] = CXc[i] - yh[i];
		}
		const std::vector<double> Lh_diff = sp::spmv_multi(Lh_csr, diff, 3);
		const std::vector<double> Vt_Lh_diff = apply_vt(Lh_diff, 3);
		std::vector<double> rhs_b(nv * 3, 0.0);
		for (std::size_t i = 0; i < nv * 3; ++i) {
			rhs_b[i] = -Vt_Lh_diff[i];
		}
		const std::vector<double> Xv = cg_multi(3, rhs_b);
		(void)Xv;

		frame_total_ms += now_ms() - t0;
	}

	BenchRow row;
	row.n            = n;
	row.nv           = nv;
	row.nh           = nh;
	row.bind_ms      = bind_ms;
	row.frame_ms_avg = frame_total_ms / frames;
	return row;
}

void print_table(const char *title, const std::vector<BenchRow> &rows) {
	std::printf("%s\n\n", title);
	std::printf("%-6s %-7s %-7s %-12s %-12s %-12s\n",
	             "N×N", "nv", "nh", "bind ms", "frame ms", "frames/s");
	std::printf("--------------------------------------------------------------\n");
	for (const auto &row : rows) {
		const double fps = (row.frame_ms_avg > 0.0)
			? (1000.0 / row.frame_ms_avg) : 0.0;
		std::printf("%-6d %-7zu %-7zu %-12.2f %-12.2f %-12.1f\n",
		             row.n * row.n, row.nv, row.nh,
		             row.bind_ms, row.frame_ms_avg, fps);
	}
	std::printf("\n");
}

} // namespace

int main(int argc, char ** /*argv*/) {
	(void)argc;

	const int frames = 30;

	std::vector<BenchRow> dense_rows;
	for (const int n : {10, 15, 20, 25, 30}) {
		dense_rows.push_back(run_dense(n, frames));
	}
	print_table("DeGoes22 deformer — dense LU factor-once path (baseline)",
	             dense_rows);

	std::vector<BenchRow> sparse_rows;
	for (const int n : {10, 15, 20, 25, 30, 40, 50, 70}) {
		sparse_rows.push_back(run_sparse(n, frames));
	}
	print_table("DeGoes22 deformer — sparse CSR + Jacobi-PCG path (current)",
	             sparse_rows);

	return 0;
}
