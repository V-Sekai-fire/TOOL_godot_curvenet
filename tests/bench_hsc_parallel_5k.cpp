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
#include "mire_body_data.h"

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
    const curvenet::HalfedgeMesh hm =
        curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c;
    c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const double mol_delta = cml::default_mollify_delta(positions, tris);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(
                                     c, positions, mol_delta);
    std::printf("Mire 5k: nv=%zu nnz=%zu\n", nv, A.values.size());

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

    // Each thread holds its own VCycleScratch (shared-nothing).
    auto solve_one = [&](const std::vector<double> &b,
                            hscn::VCycleScratch &scr) {
        return hscn::cg_hsc_with_guess_scratch(A, h, b, x0, 1000, 1e-8, scr);
    };

    // Sequential baseline (single shared scratch).
    std::vector<std::vector<double>> xs_seq(k);
    hscn::VCycleScratch scr_seq = hscn::make_scratch(h);
    const double t_seq = now_ms();
    for (std::size_t c_ = 0; c_ < k; ++c_) xs_seq[c_] = solve_one(bs[c_], scr_seq);
    const double seq_ms = now_ms() - t_seq;

    // Shared-nothing parallel: each thread holds its own scratch.
    // Hierarchy + A captured by const-ref; everything else thread-local.
    std::vector<std::vector<double>> xs_par(k);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const double t_par = now_ms();
    {
        std::vector<std::thread> ts;
        ts.reserve(k);
        for (std::size_t c_ = 0; c_ < k; ++c_) {
            ts.emplace_back([&, c_] {
                hscn::VCycleScratch scr = hscn::make_scratch(h);
                xs_par[c_] = solve_one(bs[c_], scr);
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
