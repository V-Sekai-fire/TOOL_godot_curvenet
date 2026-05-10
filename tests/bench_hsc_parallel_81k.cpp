// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// 12-RHS shared-nothing bench: HSC's cg_hsc_with_guess takes only
// const references and produces a fresh x from stack-local scratch.
// 12 solves on the same A + Hierarchy run independently — no
// shared mutable state, no sync. Use std::thread to parallelise.
//
// Hard 5 s wall cap.

#include "curvenet/cut_mesh.h"
#include "curvenet/cut_mesh_laplacian.h"
#include "curvenet/halfedge_builder.h"
#include "curvenet/hierarchical_sparsify.h"
#include "curvenet/sparse_linalg.h"
#include "curvenet/vec3.h"
#include "mire_body_70k_data.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

using curvenet::Vec3;
namespace cm   = curvenet::cut_mesh;
namespace cml  = curvenet::cut_mesh_laplacian;
namespace hscn = curvenet::hsc;
namespace sp   = curvenet::sparse;

namespace {
double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
}
} // namespace

int main() {
    std::vector<Vec3> positions;
    positions.reserve(mire_body_70k::n_verts);
    for (int i = 0; i < mire_body_70k::n_verts; ++i) {
        positions.push_back({
            static_cast<double>(mire_body_70k::positions[i * 3 + 0]),
            static_cast<double>(mire_body_70k::positions[i * 3 + 1]),
            static_cast<double>(mire_body_70k::positions[i * 3 + 2]),
        });
    }
    std::vector<int> tris(mire_body_70k::tris,
                            mire_body_70k::tris + mire_body_70k::n_tris * 3);
    const std::size_t nv = positions.size();
    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(
                                     c, positions, mol_delta);
    std::printf("Mire 81k: nv=%zu nnz=%zu\n", nv, A.values.size());

    const hscn::Graph g = hscn::csr_to_graph(A);
    const hscn::Hierarchy h = hscn::build_hierarchy(g);

    // Build 12 distinct right-hand sides (mimicking Fv 9 + Xv 3
    // columns from the deformer's actual workload).
    constexpr std::size_t k = 12;
    std::vector<std::vector<double>> bs(k);
    for (std::size_t c_ = 0; c_ < k; ++c_) {
        std::vector<double> y(nv);
        const double phase = 0.001 + 0.0003 * static_cast<double>(c_);
        for (std::size_t i = 0; i < nv; ++i) {
            y[i] = std::sin(phase * static_cast<double>(i + 1));
        }
        bs[c_] = sp::spmv(A, y);
    }
    const std::vector<double> x0(nv, 0.0);

    auto solve_one = [&](const std::vector<double> &b) {
        return hscn::cg_hsc_with_guess(A, h, b, x0, 1000, 1e-8);
    };

    // Sequential baseline.
    std::vector<std::vector<double>> xs_seq(k);
    const double t_seq = now_ms();
    for (std::size_t c_ = 0; c_ < k; ++c_) xs_seq[c_] = solve_one(bs[c_]);
    const double seq_ms = now_ms() - t_seq;

    // Shared-nothing parallel: spawn k threads, each solves one RHS.
    // Hierarchy and A are captured by const-reference; no shared
    // mutable state. Each thread's stack-local scratch buffers
    // are private. Standard data-parallel pattern.
    std::vector<std::vector<double>> xs_par(k);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const double t_par = now_ms();
    {
        std::vector<std::thread> ts;
        ts.reserve(k);
        for (std::size_t c_ = 0; c_ < k; ++c_) {
            ts.emplace_back([&, c_] {
                xs_par[c_] = solve_one(bs[c_]);
            });
        }
        for (auto &t : ts) t.join();
    }
    const double par_ms = now_ms() - t_par;

    // Verify: parallel == sequential.
    double max_diff = 0.0;
    for (std::size_t c_ = 0; c_ < k; ++c_) {
        for (std::size_t i = 0; i < nv; ++i) {
            const double d = std::fabs(xs_seq[c_][i] - xs_par[c_][i]);
            if (d > max_diff) max_diff = d;
        }
    }

    std::printf("\n--- 12-RHS shared-nothing bench (5k, hw=%u cores) ---\n", hw);
    std::printf("  sequential : %8.2f ms\n", seq_ms);
    std::printf("  parallel   : %8.2f ms  (speedup %.2fx)\n",
                  par_ms, seq_ms / par_ms);
    std::printf("  max |seq - par|: %.3e (correctness check)\n", max_diff);
    return 0;
}
