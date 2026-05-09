// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Wall-clock benchmark: GPU CG vs CPU CG on identical 2D Laplacian
// systems of growing size. The GPU path uses the standalone Vulkan
// solver with df32 dot scalars; CPU is curvenet::sparse::cg with
// Jacobi preconditioning and warm-started variant.
//
// Run with `make -C tests gpu_bench`.
//
// Caveats: on macOS this dispatches through MoltenVK, which translates
// each Vulkan command buffer into Metal under the hood. Per-dispatch
// overhead is much higher than on native Vulkan (Steam Deck) — these
// numbers are an upper bound on the GPU cost, not a deployment-target
// estimate. The same SPIR-V will run faster on RDNA 2 and Adreno
// where the loader -> driver path has fewer translation layers.

#include "gpu_cg_solver.h"
#include "curvenet/sparse_linalg.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace sp = curvenet::sparse;
using namespace curvenet_gpu_test;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
}

sp::SparseMatrixCSR grid_laplacian(int n) {
    const int nv = n * n;
    std::vector<std::vector<std::pair<int, double>>> rows(nv);
    auto idx = [n](int i, int j) { return j * n + i; };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int v = idx(i, j);
            const int neigh[4][2] = { {i+1,j}, {i-1,j}, {i,j+1}, {i,j-1} };
            for (auto &nb : neigh) {
                if (nb[0] < 0 || nb[0] >= n || nb[1] < 0 || nb[1] >= n) continue;
                rows[v].push_back({ idx(nb[0], nb[1]), -1.0 });
            }
            rows[v].push_back({ v, 4.0 });
        }
    }
    sp::SparseMatrixCSR A;
    A.rows = nv; A.cols = nv;
    A.row_ptr.assign(nv + 1, 0);
    for (int i = 0; i < nv; ++i) A.row_ptr[i + 1] = A.row_ptr[i] + rows[i].size();
    A.col_idx.resize(A.row_ptr[nv]);
    A.values.resize(A.row_ptr[nv]);
    for (int i = 0; i < nv; ++i) {
        auto &r = rows[i];
        std::sort(r.begin(), r.end(),
                    [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                        return a.first < b.first;
                    });
        for (std::size_t k = 0; k < r.size(); ++k) {
            A.col_idx[A.row_ptr[i] + k] = r[k].first;
            A.values [A.row_ptr[i] + k] = r[k].second;
        }
    }
    return A;
}

struct Row {
    int    n;        // grid edge length
    std::size_t nv;
    double bind_ms;
    double cpu_cg_ms;
    std::size_t cpu_iters_est;
    double gpu_cg_ms;
    std::size_t gpu_iters;
    double gpu_warm_ms;
    std::size_t gpu_warm_iters;
};

Row run_one(GpuCgSolver &gpu, int n) {
    const sp::SparseMatrixCSR A = grid_laplacian(n);
    const std::size_t nv = A.rows;

    std::mt19937_64 rng(0xb1ade ^ static_cast<std::uint64_t>(n));
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> b(nv);
    for (auto &v : b) v = d(rng);

    Row row{};
    row.n = n; row.nv = nv;

    // Bind time: prepare_matrix uploads CSR + creates descriptor sets.
    const double t_bind = now_ms();
    gpu.prepare_matrix(A);
    row.bind_ms = now_ms() - t_bind;

    // CPU CG (cold).
    const double t_cpu = now_ms();
    const auto x_cpu = sp::cg(A, b, 5000, 1e-8);
    row.cpu_cg_ms = now_ms() - t_cpu;
    row.cpu_iters_est = 0; // sparse::cg doesn't return the iter count

    // GPU CG (cold).
    const double t_gpu = now_ms();
    std::size_t gpu_iters = 0;
    const auto x_gpu = gpu.solve(b, 5000, 1e-8, &gpu_iters);
    row.gpu_cg_ms = now_ms() - t_gpu;
    row.gpu_iters = gpu_iters;

    // GPU CG (warm-start with x_cpu as the seed; mimics per-frame loop).
    const double t_warm = now_ms();
    std::size_t warm_iters = 0;
    const auto x_warm = gpu.solve_with_guess(b, x_cpu, 5000, 1e-8, &warm_iters);
    row.gpu_warm_ms = now_ms() - t_warm;
    row.gpu_warm_iters = warm_iters;
    (void)x_warm;
    return row;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    VulkanCompute vk;
    vk.init();

    GpuCgSolver gpu;
    gpu.init(vk,
              load_spv("bin/spmv.spv"),
              load_spv("bin/dot_reduce.spv"),
              load_spv("bin/axpy.spv"),
              load_spv("bin/jacobi.spv"),
              load_spv("bin/saxpby.spv"));

    std::printf("\nGPU CG vs CPU CG (2D grid Laplacian, random RHS, tol=1e-8)\n\n");
    std::printf("%-6s %-7s %-9s %-12s %-13s %-12s %-13s\n",
                  "N", "nv", "bind ms",
                  "cpu_cg ms",
                  "gpu_cg ms (it)",
                  "gpu_warm ms",
                  "gpu/cpu");
    std::printf("------------------------------------------------------------------------\n");

    for (const int n : {10, 20, 30, 50, 70, 100}) {
        const Row r = run_one(gpu, n);
        const double ratio = (r.cpu_cg_ms > 0.0) ? (r.gpu_cg_ms / r.cpu_cg_ms) : 0.0;
        char gpu_cell[64];
        std::snprintf(gpu_cell, sizeof(gpu_cell), "%.2f (%zu)", r.gpu_cg_ms, r.gpu_iters);
        char warm_cell[64];
        std::snprintf(warm_cell, sizeof(warm_cell), "%.2f (%zu)", r.gpu_warm_ms, r.gpu_warm_iters);
        std::printf("%-6d %-7zu %-9.2f %-12.2f %-13s %-12s %-13.2fx\n",
                      r.n, r.nv, r.bind_ms,
                      r.cpu_cg_ms,
                      gpu_cell,
                      warm_cell,
                      ratio);
    }

    gpu.shutdown();
    vk.shutdown();
    return 0;
}
