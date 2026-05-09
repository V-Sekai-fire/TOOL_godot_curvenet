// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Bench: multi-RHS GPU CG vs k separate single-RHS GPU CG calls.
// Both paths solve A·X = B for the same B at the same k. The multi-
// RHS path bundles all k columns into one nv*k working buffer; the
// single-RHS path makes k separate solve() calls. Both share the
// same A bind, so bind time is excluded from the loop.
//
// Run with `make -C tests gpu_multi_bench`.
//
// On dispatch-bound hardware (MoltenVK on M2 Pro, mobile GPUs in
// general) the multi-RHS path is the dominant per-frame win: the
// dispatch + readback overhead per CG iter is amortised across k
// columns instead of paid k times. The deformer's per-frame solve
// pattern is exactly k=9 (Fv stage) and k=3 (Xv stage), so this
// bench reflects the actual hot-path.

#include "gpu_cg_multi_solver.h"
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
        for (std::size_t j = 0; j < r.size(); ++j) {
            A.col_idx[A.row_ptr[i] + j] = r[j].first;
            A.values [A.row_ptr[i] + j] = r[j].second;
        }
    }
    return A;
}

struct Row {
    int n;
    std::size_t nv;
    std::uint32_t k;
    double single_ms;     // sum of k single-RHS solves
    std::size_t single_iters_total;
    double multi_ms;      // one multi-RHS solve
    std::size_t multi_iters;
};

Row run_one(GpuCgMultiSolver &gpu_multi, GpuCgSolver &gpu_single,
             int n, std::uint32_t k) {
    const sp::SparseMatrixCSR A = grid_laplacian(n);
    const std::size_t nv = A.rows;

    std::mt19937_64 rng(0xb1ade ^ (static_cast<std::uint64_t>(n) << 16) ^ k);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> B(nv * k);
    for (auto &v : B) v = d(rng);

    Row row{};
    row.n = n; row.nv = nv; row.k = k;

    // Single-RHS path: prepare once, solve k times.
    gpu_single.prepare_matrix(A);
    const double t_single = now_ms();
    for (std::uint32_t c = 0; c < k; ++c) {
        std::vector<double> b_c(nv);
        for (std::size_t i = 0; i < nv; ++i) b_c[i] = B[i * k + c];
        std::size_t iters = 0;
        const auto x_c = gpu_single.solve(b_c, 5000, 1e-8, &iters);
        row.single_iters_total += iters;
        (void)x_c;
    }
    row.single_ms = now_ms() - t_single;

    // Multi-RHS path: prepare for k, solve once.
    gpu_multi.prepare(A, k);
    const double t_multi = now_ms();
    std::size_t multi_iters = 0;
    const auto X = gpu_multi.solve(B, 5000, 1e-8, &multi_iters);
    row.multi_ms = now_ms() - t_multi;
    row.multi_iters = multi_iters;
    (void)X;

    return row;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    VulkanCompute vk;
    vk.init();

    GpuCgMultiSolver gpu_multi;
    gpu_multi.init(vk,
                    load_spv("bin/spmv_multi.spv"),
                    load_spv("bin/dot_reduce_multi.spv"),
                    load_spv("bin/axpy_multi.spv"),
                    load_spv("bin/jacobi_multi.spv"),
                    load_spv("bin/saxpby_multi.spv"));

    GpuCgSolver gpu_single;
    gpu_single.init(vk,
                     load_spv("bin/spmv.spv"),
                     load_spv("bin/dot_reduce.spv"),
                     load_spv("bin/axpy.spv"),
                     load_spv("bin/jacobi.spv"),
                     load_spv("bin/saxpby.spv"));

    std::printf("\nMulti-RHS GPU CG vs k separate single-RHS solves\n");
    std::printf("(2D grid Laplacian, random RHS, tol=1e-8)\n\n");
    std::printf("%-6s %-7s %-3s %-15s %-15s %-10s\n",
                  "N", "nv", "k",
                  "single ms (it)",
                  "multi  ms (it)",
                  "speedup");
    std::printf("---------------------------------------------------------------\n");

    struct Plan { int n; std::uint32_t k; };
    const Plan plans[] = {
        { 10,  3 }, { 10,  9 }, { 10, 12 },
        { 20,  3 }, { 20,  9 }, { 20, 12 },
        { 30,  9 }, { 30, 12 },
        { 50,  9 }, { 50, 12 },
        { 70,  9 }, { 70, 12 },
    };
    for (const auto &p : plans) {
        const Row r = run_one(gpu_multi, gpu_single, p.n, p.k);
        const double speedup = (r.multi_ms > 0.0) ? (r.single_ms / r.multi_ms) : 0.0;
        char single_cell[64];
        std::snprintf(single_cell, sizeof(single_cell), "%.2f (%zu)",
                        r.single_ms, r.single_iters_total);
        char multi_cell[64];
        std::snprintf(multi_cell, sizeof(multi_cell), "%.2f (%zu)",
                        r.multi_ms, r.multi_iters);
        std::printf("%-6d %-7zu %-3u %-15s %-15s %-9.2fx\n",
                      r.n, r.nv, r.k, single_cell, multi_cell, speedup);
    }

    gpu_single.shutdown();
    gpu_multi.shutdown();
    vk.shutdown();
    return 0;
}
