// Realistic interactive bench: warm-start across frames. Frame 0
// is cold. Frame N>0 starts from frame N-1's solution with a
// realistic small perturbation (handle drag = b' = b + A·δx with
// |δx| small). 5 s wall cap.
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
    constexpr std::size_t N_FRAMES = 30;
    constexpr double drift_amp = 1e-4;   // realistic per-frame handle drag

    // 12 base RHS from base y_seeds.
    std::vector<std::vector<double>> y_base(k), b_base(k);
    for (std::size_t cc = 0; cc < k; ++cc) {
        y_base[cc].resize(nv);
        const double phase = 0.001 + 0.0003 * static_cast<double>(cc);
        for (std::size_t i = 0; i < nv; ++i)
            y_base[cc][i] = std::sin(phase * static_cast<double>(i + 1));
        b_base[cc] = sp::spmv(A, y_base[cc]);
    }

    // Each frame f: y[f] = y_base + drift * sin(0.0017 * i + 0.05 * f).
    // b[f] = A * y[f]. The true x for frame f is y[f].
    // PCG warm-starts from x[f-1] (frame 0 is cold).
    std::vector<std::vector<double>> x_prev(k);
    for (auto &x : x_prev) x.assign(nv, 0.0);

    auto solve_frame = [&](std::size_t f) {
        std::vector<std::vector<double>> bs(k);
        for (std::size_t cc = 0; cc < k; ++cc) {
            std::vector<double> yf(nv);
            for (std::size_t i = 0; i < nv; ++i) {
                yf[i] = y_base[cc][i] +
                          drift_amp * std::sin(0.0017 * static_cast<double>(i)
                                               + 0.05 * static_cast<double>(f));
            }
            bs[cc] = sp::spmv(A, yf);
        }
        // 12-thread shared-nothing solve, each warm-started.
        std::vector<std::vector<double>> xs(k);
        std::vector<std::thread> ts;
        for (std::size_t cc = 0; cc < k; ++cc) {
            ts.emplace_back([&, cc] {
                hscn::VCycleScratch scr = hscn::make_scratch(h);
                xs[cc] = hscn::cg_hsc_with_guess_scratch(
                    A, h, bs[cc], x_prev[cc], 30, 1e-7, scr);
            });
        }
        for (auto &t : ts) t.join();
        x_prev = xs;
    };

    // Frame 0 cold (warm-start from zeros).
    const double t0 = now_ms();
    solve_frame(0);
    const double frame0_ms = now_ms() - t0;

    // Frames 1..N-1 warm.
    std::vector<double> warm_ms;
    for (std::size_t f = 1; f < N_FRAMES; ++f) {
        const double t = now_ms();
        solve_frame(f);
        warm_ms.push_back(now_ms() - t);
    }
    std::sort(warm_ms.begin(), warm_ms.end());
    const double median = warm_ms[warm_ms.size() / 2];
    double avg = 0;
    for (double v : warm_ms) avg += v;
    avg /= warm_ms.size();

    std::printf("Mire 5k 12-RHS interactive (drift=%.0e per frame):\n", drift_amp);
    std::printf("  frame 0 (cold)         : %8.2f ms\n", frame0_ms);
    std::printf("  warm steady-state min  : %8.2f ms\n", warm_ms.front());
    std::printf("  warm steady-state med  : %8.2f ms\n", median);
    std::printf("  warm steady-state avg  : %8.2f ms\n", avg);
    std::printf("  warm steady-state max  : %8.2f ms\n", warm_ms.back());
    std::printf("  warm FPS (median)      : %8.1f Hz\n", 1000.0 / median);
    std::printf("  5 ms target            : %8s (gap %.1fx)\n",
                  median <= 5.0 ? "HIT" : "MISS",
                  median / 5.0);
    return 0;
}
