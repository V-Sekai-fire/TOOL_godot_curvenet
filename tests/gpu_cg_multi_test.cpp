// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// End-to-end correctness test for the multi-RHS GPU CG solver.
// Verifies that GpuCgMultiSolver.solve(B, k) produces the same per-
// column result as k separate GpuCgSolver.solve(b_c) calls, within
// the fp32 floor of the GPU path.
//
// Run with `make -C tests gpu`.

#include "gpu_cg_multi_solver.h"
#include "gpu_cg_solver.h"
#include "curvenet/sparse_linalg.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace sp = curvenet::sparse;
using namespace curvenet_gpu_test;

namespace {

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

double max_abs_residual(const sp::SparseMatrixCSR &A,
                          const std::vector<double> &x,
                          const std::vector<double> &b) {
    const std::vector<double> Ax = sp::spmv(A, x);
    double m = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        m = std::max(m, std::fabs(Ax[i] - b[i]));
    }
    return m;
}

int run_case(const char *name, const sp::SparseMatrixCSR &A,
              const std::vector<double> &B_rowmajor, std::uint32_t k,
              std::size_t max_iter, double tol,
              GpuCgMultiSolver &gpu_multi, GpuCgSolver &gpu_single) {
    const std::size_t nv = A.rows;

    // Multi-RHS solve.
    gpu_multi.prepare(A, k);
    std::size_t multi_iters = 0;
    const auto X_multi = gpu_multi.solve(B_rowmajor, max_iter, tol, &multi_iters);

    // Single-RHS reference: k separate solves.
    gpu_single.prepare_matrix(A);
    std::vector<double> X_single(nv * k, 0.0);
    std::size_t single_iters_total = 0;
    for (std::uint32_t c = 0; c < k; ++c) {
        std::vector<double> b_c(nv);
        for (std::size_t i = 0; i < nv; ++i) b_c[i] = B_rowmajor[i * k + c];
        std::size_t iters = 0;
        const auto x_c = gpu_single.solve(b_c, max_iter, tol, &iters);
        single_iters_total += iters;
        for (std::size_t i = 0; i < nv; ++i) X_single[i * k + c] = x_c[i];
    }

    // Per-column residual check on the multi-RHS result.
    double worst_res = 0.0;
    for (std::uint32_t c = 0; c < k; ++c) {
        std::vector<double> x_c(nv), b_c(nv);
        for (std::size_t i = 0; i < nv; ++i) {
            x_c[i] = X_multi    [i * k + c];
            b_c[i] = B_rowmajor[i * k + c];
        }
        const double res = max_abs_residual(A, x_c, b_c);
        if (res > worst_res) worst_res = res;
    }

    const bool ok = (worst_res <= 1e-3);
    std::printf("  %-26s n=%-5zu k=%-2u multi_it=%-3zu single_total_it=%-4zu worst_res=%.3e %s\n",
                  name, nv, k, multi_iters, single_iters_total, worst_res,
                  ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
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

    int failures = 0;

    // k=3 small grid, hand-built RHS columns.
    {
        const auto A = grid_laplacian(5);
        const std::size_t nv = A.rows;
        std::vector<double> B(nv * 3, 0.0);
        B[12 * 3 + 0] = 1.0;
        for (std::size_t i = 0; i < nv; ++i) B[i * 3 + 1] = 0.1 * (i + 1);
        for (std::size_t i = 0; i < nv; ++i) B[i * 3 + 2] = (i & 1) ? 1.0 : -1.0;
        failures += run_case("grid_lap(5) k=3", A, B, 3, 500, 1e-9,
                              gpu_multi, gpu_single);
    }

    // k=9 (Fv-shaped) random RHS at n=100.
    {
        const auto A = grid_laplacian(10);
        const std::size_t nv = A.rows;
        std::mt19937_64 rng(0xfeedULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> B(nv * 9);
        for (auto &v : B) v = d(rng);
        failures += run_case("grid_lap(10) k=9 random", A, B, 9, 1000, 1e-9,
                              gpu_multi, gpu_single);
    }

    // k=12 random RHS at n=400, the deformer's worst-case fan-out.
    {
        const auto A = grid_laplacian(20);
        const std::size_t nv = A.rows;
        std::mt19937_64 rng(0xc0deULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> B(nv * 12);
        for (auto &v : B) v = d(rng);
        failures += run_case("grid_lap(20) k=12 random", A, B, 12, 2000, 1e-9,
                              gpu_multi, gpu_single);
    }

    gpu_single.shutdown();
    gpu_multi.shutdown();
    vk.shutdown();
    std::printf("\n%s: %d failing case(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
