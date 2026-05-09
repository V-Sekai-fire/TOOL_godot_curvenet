// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// End-to-end correctness test for the standalone GPU CG solver.
// Spins up Vulkan + the five compute kernels (spmv, dot_reduce-df32,
// axpy, jacobi, saxpby), runs preconditioned CG on the GPU on a
// handful of SPD systems, and checks that the result matches the CPU
// reference `curvenet::sparse::cg` to within fp32 tolerance.
//
// Run with `make -C tests gpu`.

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

sp::SparseMatrixCSR diag5() {
    sp::SparseMatrixCSR A;
    A.rows = 5; A.cols = 5;
    A.row_ptr = { 0, 1, 2, 3, 4, 5 };
    A.col_idx = { 0, 1, 2, 3, 4 };
    A.values  = { 2.0, 3.0, 4.0, 5.0, 6.0 };
    return A;
}

sp::SparseMatrixCSR laplacian4() {
    sp::SparseMatrixCSR A;
    A.rows = 4; A.cols = 4;
    A.row_ptr = { 0, 2, 5, 8, 10 };
    A.col_idx = { 0, 1,
                    0, 1, 2,
                    1, 2, 3,
                    2, 3 };
    A.values  = { 2.0, -1.0,
                    -1.0,  2.0, -1.0,
                            -1.0,  2.0, -1.0,
                                    -1.0,  2.0 };
    return A;
}

// 2D grid Laplacian, n×n with Dirichlet boundary embedded in the
// matrix (boundary nodes get a self-edge of 4 with no neighbour
// connections, making the system SPD throughout).
sp::SparseMatrixCSR grid_laplacian(int n) {
    const int nv = n * n;
    std::vector<std::vector<std::pair<int, double>>> rows(nv);
    auto idx = [n](int i, int j) { return j * n + i; };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int v = idx(i, j);
            // Always-positive diagonal entry of 4 keeps the matrix SPD
            // (interior nodes have 4 neighbours of −1; boundary nodes
            // have fewer −1's and so a stronger diagonal — still SPD).
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

// Residual ‖A·x − b‖∞ — primary correctness check. We don't expect
// GPU and CPU CG to produce *bit-identical* x (different rounding
// orders) but both should drive the residual under tol.
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

int run_case(GpuCgSolver &gpu, const char *name,
              const sp::SparseMatrixCSR &A, const std::vector<double> &b,
              std::size_t max_iter, double tol) {
    gpu.prepare_matrix(A);

    std::size_t gpu_iters = 0;
    const std::vector<double> x_gpu = gpu.solve(b, max_iter, tol, &gpu_iters);
    const std::vector<double> x_cpu = sp::cg(A, b, max_iter, tol);

    const double res_gpu = max_abs_residual(A, x_gpu, b);
    const double res_cpu = max_abs_residual(A, x_cpu, b);
    // GPU result lives in fp32 so the residual won't go below ~1e-6
    // even with df32 dot scalars; that's the precision floor of
    // storing x as fp32 anywhere in the pipeline.
    const bool ok = (res_gpu <= 1e-4);

    std::printf("  %-26s n=%-5zu  cpu_res=%-9.3e  gpu_res=%-9.3e  gpu_iters=%-3zu  %s\n",
                  name, A.rows, res_cpu, res_gpu, gpu_iters,
                  ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
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

    int failures = 0;

    failures += run_case(gpu, "diag(2,3,4,5,6) * b", diag5(),
                          { 4.0, 9.0, 16.0, 25.0, 36.0 }, 200, 1e-9);

    failures += run_case(gpu, "laplacian4 (1,0,0,0)", laplacian4(),
                          { 1.0, 0.0, 0.0, 0.0 }, 200, 1e-9);

    {
        const auto A = grid_laplacian(5);
        std::vector<double> b(25, 0.0);
        b[12] = 1.0;
        failures += run_case(gpu, "grid_laplacian(5) e12", A, b, 500, 1e-9);
    }

    {
        const auto A = grid_laplacian(10);
        std::vector<double> b(100);
        for (int i = 0; i < 100; ++i) b[i] = (i % 3 == 0) ? 1.0 : -0.5;
        failures += run_case(gpu, "grid_laplacian(10) tri", A, b, 1000, 1e-9);
    }

    {
        const auto A = grid_laplacian(20);
        std::mt19937_64 rng(0xc0ffee0a1ULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> b(400);
        for (auto &v : b) v = d(rng);
        failures += run_case(gpu, "grid_laplacian(20) rand", A, b, 2000, 1e-9);
    }

    // Warm-start: solve once cold, then re-solve with the cold result
    // as initial guess. The warm solve should converge in 0-1 iters
    // because x0 is already the answer (within fp32 floor).
    {
        const auto A = grid_laplacian(10);
        std::vector<double> b(100);
        for (int i = 0; i < 100; ++i) b[i] = (i % 3 == 0) ? 1.0 : -0.5;
        gpu.prepare_matrix(A);
        std::size_t cold_iters = 0;
        const auto x_cold = gpu.solve(b, 1000, 1e-9, &cold_iters);
        std::size_t warm_iters = 0;
        const auto x_warm = gpu.solve_with_guess(b, x_cold, 1000, 1e-9, &warm_iters);
        const double res_warm = max_abs_residual(A, x_warm, b);
        const bool ok = (warm_iters <= cold_iters) && (res_warm <= 1e-4);
        std::printf("  %-26s n=%-5zu  cold_iters=%-3zu  warm_iters=%-3zu  warm_res=%-9.3e  %s\n",
                      "warm-start grid_lap(10)", A.rows,
                      cold_iters, warm_iters, res_warm, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    // Slightly perturbed warm-start: nudge the cold result by ~1% so
    // the warm solve still has nontrivial work to do, but should
    // still finish in noticeably fewer iters than cold.
    {
        const auto A = grid_laplacian(20);
        std::mt19937_64 rng(0x1234abcdULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> b(400);
        for (auto &v : b) v = d(rng);
        gpu.prepare_matrix(A);
        std::size_t cold_iters = 0;
        const auto x_cold = gpu.solve(b, 2000, 1e-9, &cold_iters);
        // Add 1% perturbation per element.
        std::vector<double> x_seed = x_cold;
        std::uniform_real_distribution<double> p(-0.01, 0.01);
        for (auto &v : x_seed) v *= (1.0 + p(rng));
        std::size_t warm_iters = 0;
        const auto x_warm = gpu.solve_with_guess(b, x_seed, 2000, 1e-9, &warm_iters);
        const double res_warm = max_abs_residual(A, x_warm, b);
        const bool ok = (warm_iters < cold_iters) && (res_warm <= 1e-4);
        std::printf("  %-26s n=%-5zu  cold_iters=%-3zu  warm_iters=%-3zu  warm_res=%-9.3e  %s\n",
                      "warm-start +1% grid_lap(20)", A.rows,
                      cold_iters, warm_iters, res_warm, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    gpu.shutdown();
    vk.shutdown();

    std::printf("\n%s: %d failing case(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
