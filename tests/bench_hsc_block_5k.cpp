// TOMBSTONE [post-100-loops cleanup, 2026-05-09]
// Block V-cycle was net-negative at 5k (1-thread block 135 ms vs
// 12-thread shared-nothing 42 ms — at 5k working set fits cache so
// memory-amortisation across RHS doesn't help, while doing 12x
// arithmetic in 1 thread costs more). Bench preserved for trajectory
// record; see PERF_BASELINE.md "Trajectory" section. Do not regress
// to this measurement as a baseline. Architecture is now DDM at
// runtime + HSC at bind time; this bench measures a retired path.
//
// Block (multi-RHS) HSC at 5k. Compares:
//  - 12 sequential single-RHS solves
//  - 12-thread shared-nothing (one solve per thread)
//  - Block-12-RHS in 1 thread (memory-amortised across all RHS)
//  - Block split: 3 threads x 4 RHS each (memory-amortised x parallel)
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

using curvenet::Vec3;
namespace cm   = curvenet::cut_mesh;
namespace cml  = curvenet::cut_mesh_laplacian;
namespace hscn = curvenet::hsc;
namespace sp   = curvenet::sparse;

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    std::vector<Vec3> positions;
    positions.reserve(mire_body::n_verts);
    for (int i = 0; i < mire_body::n_verts; ++i) {
        positions.push_back({ static_cast<double>(mire_body::positions[i*3+0]),
                                static_cast<double>(mire_body::positions[i*3+1]),
                                static_cast<double>(mire_body::positions[i*3+2]) });
    }
    std::vector<int> tris(mire_body::tris, mire_body::tris + mire_body::n_tris*3);
    const std::size_t nv = positions.size();
    const curvenet::HalfedgeMesh hm = curvenet::halfedge_builder::from_triangles(nv, tris);
    cm::CutMesh c; c.base = hm;
    c.vertex_kind.assign(nv, cm::CutVertexKind::mesh_vertex_kind());
    c.segment_of_halfedge.assign(hm.he_count(), -1);
    const sp::SparseMatrixCSR A = cml::assemble_vt_lh_v_csr_robust(
        c, positions, cml::default_mollify_delta(positions, tris));
    const hscn::Graph g = hscn::csr_to_graph(A);
    const hscn::Hierarchy h = hscn::build_hierarchy(g);

    constexpr std::size_t k = 12;

    // Build B (n × k row-major) with k distinct RHS.
    std::vector<double> B(nv * k);
    for (std::size_t cc = 0; cc < k; ++cc) {
        std::vector<double> y(nv);
        const double phase = 0.001 + 0.0003 * static_cast<double>(cc);
        for (std::size_t i = 0; i < nv; ++i) y[i] = std::sin(phase * static_cast<double>(i + 1));
        const auto b_col = sp::spmv(A, y);
        for (std::size_t i = 0; i < nv; ++i) B[i * k + cc] = b_col[i];
    }

    // Mode 1: Block-12 in 1 thread.
    {
        hscn::BlockVCycleScratch s = hscn::make_scratch_block(h, k);
        std::vector<std::size_t> iters;
        const double t = now_ms();
        const auto X = hscn::cg_hsc_block(A, h, B, k, 1000, 1e-8, s, &iters);
        const double ms = now_ms() - t;
        (void)X;
        std::size_t mn = 99999, mx = 0, sm = 0;
        for (auto v : iters) { mn = std::min(mn, v); mx = std::max(mx, v); sm += v; }
        std::printf("block-12 (1 thread): %.2f ms  iters min=%zu avg=%zu max=%zu\n",
                      ms, mn, sm / k, mx);
        std::fflush(stdout);
    }

    // Mode 2: 12-thread shared-nothing single-RHS (existing).
    {
        std::vector<double> bs[k];
        for (std::size_t cc = 0; cc < k; ++cc) {
            bs[cc].resize(nv);
            for (std::size_t i = 0; i < nv; ++i) bs[cc][i] = B[i * k + cc];
        }
        const std::vector<double> x0(nv, 0.0);
        const double t = now_ms();
        std::vector<std::thread> ts;
        for (std::size_t cc = 0; cc < k; ++cc) {
            ts.emplace_back([&, cc] {
                hscn::VCycleScratch scr = hscn::make_scratch(h);
                hscn::cg_hsc_with_guess_scratch(A, h, bs[cc], x0, 1000, 1e-8, scr);
            });
        }
        for (auto &t2 : ts) t2.join();
        std::printf("12-thread single (existing): %.2f ms\n", now_ms() - t);
        std::fflush(stdout);
    }

    // Mode 3: Block split: T threads x (k/T) RHS each.
    for (std::size_t T : { 2, 3, 4, 6 }) {
        if (k % T != 0) continue;
        const std::size_t per = k / T;
        // Build per-thread B blocks.
        std::vector<std::vector<double>> Bs(T);
        for (std::size_t t_id = 0; t_id < T; ++t_id) {
            Bs[t_id].resize(nv * per);
            for (std::size_t cc = 0; cc < per; ++cc) {
                const std::size_t orig = t_id * per + cc;
                for (std::size_t i = 0; i < nv; ++i) {
                    Bs[t_id][i * per + cc] = B[i * k + orig];
                }
            }
        }
        const double t = now_ms();
        std::vector<std::thread> ts;
        for (std::size_t t_id = 0; t_id < T; ++t_id) {
            ts.emplace_back([&, t_id] {
                hscn::BlockVCycleScratch scr = hscn::make_scratch_block(h, per);
                hscn::cg_hsc_block(A, h, Bs[t_id], per, 1000, 1e-8, scr);
            });
        }
        for (auto &t2 : ts) t2.join();
        std::printf("split %zu threads x %zu RHS: %.2f ms\n", T, per, now_ms() - t);
        std::fflush(stdout);
    }
    return 0;
}
