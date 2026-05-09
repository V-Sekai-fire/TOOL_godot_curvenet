// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Standalone Vulkan-compute correctness test for the multi-RHS
// kernels. The deformer solves 9 RHS columns (Fv stage) and 3 RHS
// columns (Xv stage) per frame; bundling them into one nv × k
// dispatch instead of k separate ones amortises the dispatch
// overhead k×, which on MoltenVK / dispatch-bound regimes is the
// single largest win available.
//
// This test currently covers spmv_multi against k separate
// curvenet::sparse::spmv calls. Subsequent shaders (axpy, jacobi,
// saxpby, dot_reduce, all _multi variants) will land here too.
//
// Run with `make -C tests gpu`.

#include "curvenet/sparse_linalg.h"
#include "gpu_compute_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace sp = curvenet::sparse;
using namespace curvenet_gpu_test;

namespace {

curvenet::sparse::SparseMatrixCSR grid_laplacian(int n) {
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

struct SpmvMultiPipeline {
    ComputeKernel kernel;

    void init(VkDevice device, const std::vector<std::uint32_t> &spv) {
        kernel.init(device, spv, 5);
    }
    void shutdown(VkDevice device) {
        kernel.shutdown(device);
    }
};

// gpu_spmv_multi: dispatch one shader call to compute Y = A · X for
// k columns at once.
std::vector<double> gpu_spmv_multi(VulkanCompute &vk,
                                     SpmvMultiPipeline &pipe,
                                     VkDescriptorPool pool,
                                     const sp::SparseMatrixCSR &A,
                                     const std::vector<double> &X_rowmajor,
                                     std::uint32_t k) {
    const std::uint32_t rows = static_cast<std::uint32_t>(A.rows);
    const std::size_t   nnz  = A.values.size();

    struct UB { std::uint32_t rows; std::uint32_t k; } ub{ rows, k };

    Buffer params  = make_buffer(vk, sizeof(UB), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    Buffer row_ptr = make_buffer(vk, A.row_ptr.size() * sizeof(int),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer col_idx = make_buffer(vk, A.col_idx.size() * sizeof(int),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer values  = make_buffer(vk, nnz * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer x_buf   = make_buffer(vk, X_rowmajor.size() * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer y_buf   = make_buffer(vk, rows * k * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::memcpy(params.mapped,  &ub,              sizeof(UB));
    std::memcpy(row_ptr.mapped, A.row_ptr.data(), A.row_ptr.size() * sizeof(int));
    std::memcpy(col_idx.mapped, A.col_idx.data(), A.col_idx.size() * sizeof(int));
    {
        std::vector<float> v32(nnz);
        for (std::size_t i = 0; i < nnz; ++i) v32[i] = static_cast<float>(A.values[i]);
        std::memcpy(values.mapped, v32.data(), nnz * sizeof(float));
    }
    {
        std::vector<float> x32(X_rowmajor.size());
        for (std::size_t i = 0; i < X_rowmajor.size(); ++i)
            x32[i] = static_cast<float>(X_rowmajor[i]);
        std::memcpy(x_buf.mapped, x32.data(), x32.size() * sizeof(float));
    }
    std::memset(y_buf.mapped, 0, rows * k * sizeof(float));

    VkDescriptorSet ds = alloc_and_bind(vk.device, pool, pipe.kernel.dsl, params,
                                         { row_ptr, col_idx, values, x_buf, y_buf });
    run_compute_once(vk, pipe.kernel.pipeline, pipe.kernel.layout, ds, (rows + 63) / 64);

    std::vector<double> Y(rows * k, 0.0);
    {
        std::vector<float> y32(rows * k);
        std::memcpy(y32.data(), y_buf.mapped, rows * k * sizeof(float));
        for (std::size_t i = 0; i < y32.size(); ++i) Y[i] = static_cast<double>(y32[i]);
    }

    destroy_buffer(vk.device, y_buf);
    destroy_buffer(vk.device, x_buf);
    destroy_buffer(vk.device, values);
    destroy_buffer(vk.device, col_idx);
    destroy_buffer(vk.device, row_ptr);
    destroy_buffer(vk.device, params);
    return Y;
}

// CPU multi-RHS spmv (reference): k separate spmvs stacked into one
// row-major nv × k buffer.
std::vector<double> cpu_spmv_multi(const sp::SparseMatrixCSR &A,
                                     const std::vector<double> &X_rowmajor,
                                     std::uint32_t k) {
    const std::size_t nv = A.rows;
    std::vector<double> Y(nv * k, 0.0);
    for (std::size_t i = 0; i < nv; ++i) {
        const int rs = A.row_ptr[i];
        const int re = A.row_ptr[i + 1];
        for (int p = rs; p < re; ++p) {
            const double aij = A.values[p];
            const std::size_t j = static_cast<std::size_t>(A.col_idx[p]);
            for (std::uint32_t c = 0; c < k; ++c) {
                Y[i * k + c] += aij * X_rowmajor[j * k + c];
            }
        }
    }
    return Y;
}

int run_spmv_multi_case(VulkanCompute &vk, SpmvMultiPipeline &pipe,
                          VkDescriptorPool pool,
                          const char *name,
                          const sp::SparseMatrixCSR &A,
                          const std::vector<double> &X, std::uint32_t k) {
    const auto Y_cpu = cpu_spmv_multi(A, X, k);
    const auto Y_gpu = gpu_spmv_multi(vk, pipe, pool, A, X, k);

    // Combined absolute + relative tolerance: |a-b| <= abs_tol + rel_tol·(|a|+|b|).
    // Without the absolute floor, near-zero results (where both CPU
    // fp64 noise ≈ 1e-16 and GPU fp32 noise ≈ 1e-7 register) report
    // a meaningless rel_err of 1.0.
    const double abs_tol = 1e-5;
    const double rel_tol = 1e-5;
    double worst_rel = 0.0;
    bool ok = (Y_cpu.size() == Y_gpu.size());
    for (std::size_t i = 0; ok && i < Y_cpu.size(); ++i) {
        const double d = std::fabs(Y_cpu[i] - Y_gpu[i]);
        const double m = std::fabs(Y_cpu[i]) + std::fabs(Y_gpu[i]) + 1e-30;
        const double rel = d / m;
        if (rel > worst_rel) worst_rel = rel;
        if (d > abs_tol + rel_tol * m) ok = false;
    }
    std::printf("  %-30s  n=%-5zu  k=%-2u  worst_rel=%.3e  %s\n",
                  name, A.rows, k, worst_rel, ok ? "OK" : "FAIL");
    if (!ok) {
        std::size_t worst_i = 0;
        double w = 0.0;
        for (std::size_t i = 0; i < Y_cpu.size(); ++i) {
            const double d = std::fabs(Y_cpu[i] - Y_gpu[i]);
            const double m = std::fabs(Y_cpu[i]) + std::fabs(Y_gpu[i]) + 1e-30;
            const double r = d / m;
            if (r > w) { w = r; worst_i = i; }
        }
        const std::size_t lo = (worst_i > 2) ? worst_i - 2 : 0;
        const std::size_t hi = std::min(Y_cpu.size(), worst_i + 4);
        std::printf("    worst at i=%zu (i/k=%zu, c=%zu)\n",
                      worst_i, worst_i / k, worst_i % k);
        std::printf("    cpu[%zu..%zu]:", lo, hi);
        for (std::size_t i = lo; i < hi; ++i) std::printf(" %.4g", Y_cpu[i]);
        std::printf("\n    gpu[%zu..%zu]:", lo, hi);
        for (std::size_t i = lo; i < hi; ++i) std::printf(" %.4g", Y_gpu[i]);
        std::printf("\n");
    }
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    VulkanCompute vk;
    vk.init();

    SpmvMultiPipeline pipe;
    pipe.init(vk.device, load_spv("bin/spmv_multi.spv"));

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 8;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 40;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = 8;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_OR_DIE(vkCreateDescriptorPool(vk.device, &dpci, nullptr, &pool));

    int failures = 0;

    // k=3 trivial: A = grid_lap(5), X = three different RHS stacked.
    {
        const auto A = grid_laplacian(5);
        const std::size_t nv = A.rows;
        std::vector<double> X(nv * 3, 0.0);
        // column 0: indicator at center
        X[12 * 3 + 0] = 1.0;
        // column 1: ramp
        for (std::size_t i = 0; i < nv; ++i) X[i * 3 + 1] = 0.1 * i;
        // column 2: alternating ±1
        for (std::size_t i = 0; i < nv; ++i) X[i * 3 + 2] = (i & 1) ? 1.0 : -1.0;
        failures += run_spmv_multi_case(vk, pipe, pool, "grid_lap(5) k=3", A, X, 3);
    }

    // k=9 (Fv-shaped): A = grid_lap(10), X = 9 random columns.
    {
        const auto A = grid_laplacian(10);
        const std::size_t nv = A.rows;
        std::mt19937_64 rng(0xab12cd34ULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> X(nv * 9);
        for (auto &v : X) v = d(rng);
        failures += run_spmv_multi_case(vk, pipe, pool, "grid_lap(10) k=9 random", A, X, 9);
    }

    // k=12 (max practical): A = grid_lap(20), X = 12 random columns.
    {
        const auto A = grid_laplacian(20);
        const std::size_t nv = A.rows;
        std::mt19937_64 rng(0xfeedfaceULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> X(nv * 12);
        for (auto &v : X) v = d(rng);
        failures += run_spmv_multi_case(vk, pipe, pool, "grid_lap(20) k=12 random", A, X, 12);
    }

    // k=1 sanity: should match the single-RHS spmv kernel exactly.
    {
        const auto A = grid_laplacian(7);
        const std::size_t nv = A.rows;
        std::vector<double> X(nv);
        for (std::size_t i = 0; i < nv; ++i) X[i] = 0.1 * i;
        failures += run_spmv_multi_case(vk, pipe, pool, "grid_lap(7) k=1", A, X, 1);
    }

    vkDestroyDescriptorPool(vk.device, pool, nullptr);
    pipe.shutdown(vk.device);
    vk.shutdown();
    std::printf("\n%s: %d failing case(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
