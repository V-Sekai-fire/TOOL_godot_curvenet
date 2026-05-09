// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// End-to-end GPU CG solver. Stitches the five compute kernels
// (spmv, dot_reduce-df32, axpy, jacobi, saxpby) into the same
// preconditioned conjugate gradient iteration that lives on CPU in
// `curvenet::sparse::cg`. The iteration loop runs on the host; each
// kernel is a single dispatch + readback when a scalar (α, β,
// residual norm) is needed.
//
// Lives under tests/ for now because it depends on the test-only
// Vulkan helpers. Once the algorithm is locked in we'll move the
// solver into src/curvenet/gpu_sparse_solve.{h,cpp} with a pImpl
// wrapper hiding the Vulkan types from the deformer header (see
// todos/08 phase 7).
//
// Precision strategy (locked in via earlier shaders):
//   * spmv / axpy / jacobi / saxpby: fp32 throughout, ~few ulp error.
//   * dot reduction: df32, ~48-bit mantissa, fp64-quality scalars.
//   * Host α/β math: fp64.
// At 100k vertices this is sufficient to drive CG to its true residual
// tolerance without the iterative-refinement-on-CPU roundtrip; see
// docs/PERF_BASELINE.md and todos/08_gpu_compute_solver.md.

#ifndef CURVENET_GPU_CG_SOLVER_H
#define CURVENET_GPU_CG_SOLVER_H

#include "curvenet/sparse_linalg.h"
#include "gpu_compute_helpers.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace curvenet_gpu_test {

namespace sp = curvenet::sparse;

class GpuCgSolver {
public:
    // Lifecycle
    void init(VulkanCompute &vk_in,
               const std::vector<std::uint32_t> &spmv_spv,
               const std::vector<std::uint32_t> &dot_spv,
               const std::vector<std::uint32_t> &axpy_spv,
               const std::vector<std::uint32_t> &jacobi_spv,
               const std::vector<std::uint32_t> &saxpby_spv);
    void prepare_matrix(const sp::SparseMatrixCSR &A);
    std::vector<double> solve(const std::vector<double> &b,
                                std::size_t max_iter, double tol,
                                std::size_t *iters_used = nullptr);
    // Warm-started CG: starts from `x0` instead of zero. Mathematically
    // identical fixed point to `solve(b, ...)`, but converges in
    // far fewer iterations when x0 is close to the true solution —
    // exactly the situation in the deformer's per-frame loop after
    // the first frame.
    std::vector<double> solve_with_guess(const std::vector<double> &b,
                                            const std::vector<double> &x0,
                                            std::size_t max_iter, double tol,
                                            std::size_t *iters_used = nullptr);
    void shutdown();

private:
    // Dispatch helpers — one per kernel. Each takes the descriptor set
    // already bound to the right buffers and the dispatch group count.
    void dispatch_spmv  (VkDescriptorSet ds);
    void dispatch_dot   (VkDescriptorSet ds);
    void dispatch_axpy  (VkDescriptorSet ds);
    void dispatch_jacobi(VkDescriptorSet ds);
    void dispatch_saxpby(VkDescriptorSet ds);

    // Read back the df32 (hi, lo) pair from `dot_pair` and fold to fp64.
    double read_dot_scalar() const;

    void write_axpy_uniform  (Buffer &u, std::uint32_t n, float alpha);
    void write_saxpby_uniform(Buffer &u, std::uint32_t n,
                                float alpha, float beta);

    VulkanCompute *vk = nullptr;

    ComputeKernel k_spmv;
    ComputeKernel k_dot;
    ComputeKernel k_axpy;
    ComputeKernel k_jacobi;
    ComputeKernel k_saxpby;

    VkDescriptorPool pool = VK_NULL_HANDLE;

    // Persistent matrix-side buffers (rebuilt on prepare_matrix).
    Buffer A_row_ptr;
    Buffer A_col_idx;
    Buffer A_values;
    Buffer A_diag;

    // Per-solve working buffers.
    Buffer b_buf;
    Buffer x_buf;
    Buffer r_buf;
    Buffer z_buf;
    Buffer p_buf;
    Buffer Ap_buf;
    Buffer dot_pair;     // 2 floats: df32 (hi, lo)

    // Uniforms.
    Buffer u_spmv;       // {uint rows;}
    Buffer u_n;          // {uint n;}                — shared by dot/jacobi
    Buffer u_axpy_x;     // {uint n; float alpha;}   — x ← x + α·p
    Buffer u_axpy_r;     // {uint n; float alpha;}   — r ← r + (-α)·Ap
    Buffer u_saxpby;     // {uint n; float α; float β;}

    // Descriptor sets (one per logical use — buffer bindings change
    // per use, so we can't share sets across uses without rebinding).
    VkDescriptorSet ds_spmv      = VK_NULL_HANDLE;
    VkDescriptorSet ds_dot_pAp   = VK_NULL_HANDLE;
    VkDescriptorSet ds_dot_rr    = VK_NULL_HANDLE;
    VkDescriptorSet ds_dot_rz    = VK_NULL_HANDLE;
    VkDescriptorSet ds_axpy_x    = VK_NULL_HANDLE;
    VkDescriptorSet ds_axpy_r    = VK_NULL_HANDLE;
    VkDescriptorSet ds_jacobi_b  = VK_NULL_HANDLE; // initial z ← M⁻¹·b
    VkDescriptorSet ds_jacobi_r  = VK_NULL_HANDLE; // per-iter z ← M⁻¹·r
    VkDescriptorSet ds_saxpby    = VK_NULL_HANDLE;

    std::uint32_t n_rows = 0;
};

inline void GpuCgSolver::init(VulkanCompute &vk_in,
                                  const std::vector<std::uint32_t> &spmv_spv,
                                  const std::vector<std::uint32_t> &dot_spv,
                                  const std::vector<std::uint32_t> &axpy_spv,
                                  const std::vector<std::uint32_t> &jacobi_spv,
                                  const std::vector<std::uint32_t> &saxpby_spv) {
    vk = &vk_in;
    k_spmv  .init(vk->device, spmv_spv,   5);
    k_dot   .init(vk->device, dot_spv,    3);
    k_axpy  .init(vk->device, axpy_spv,   2);
    k_jacobi.init(vk->device, jacobi_spv, 3);
    k_saxpby.init(vk->device, saxpby_spv, 3);

    // Pool sized for: 9 descriptor sets, each with 1 uniform + storage.
    // Total uniforms = 9, total storage = 5+3+2+3+3+2+3+3+3 = 27 (estimate).
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 16;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 64;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 16;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = sizes;
    VK_OR_DIE(vkCreateDescriptorPool(vk->device, &dpci, nullptr, &pool));
}

inline void GpuCgSolver::shutdown() {
    if (!vk) return;
    auto kill = [this](Buffer &b) { destroy_buffer(vk->device, b); };
    kill(A_row_ptr); kill(A_col_idx); kill(A_values); kill(A_diag);
    kill(b_buf); kill(x_buf); kill(r_buf); kill(z_buf);
    kill(p_buf); kill(Ap_buf); kill(dot_pair);
    kill(u_spmv); kill(u_n); kill(u_axpy_x); kill(u_axpy_r); kill(u_saxpby);
    if (pool) vkDestroyDescriptorPool(vk->device, pool, nullptr);
    pool = VK_NULL_HANDLE;
    k_spmv.shutdown(vk->device);
    k_dot.shutdown(vk->device);
    k_axpy.shutdown(vk->device);
    k_jacobi.shutdown(vk->device);
    k_saxpby.shutdown(vk->device);
    vk = nullptr;
}

inline void GpuCgSolver::prepare_matrix(const sp::SparseMatrixCSR &A) {
    n_rows = static_cast<std::uint32_t>(A.rows);
    const std::size_t nnz = A.values.size();

    // Free any previous matrix buffers so prepare_matrix can be called
    // again with a different A.
    auto reset = [this](Buffer &b) {
        if (b.handle) destroy_buffer(vk->device, b);
    };
    reset(A_row_ptr); reset(A_col_idx); reset(A_values); reset(A_diag);
    reset(b_buf); reset(x_buf); reset(r_buf); reset(z_buf);
    reset(p_buf); reset(Ap_buf); reset(dot_pair);
    reset(u_spmv); reset(u_n); reset(u_axpy_x); reset(u_axpy_r); reset(u_saxpby);
    // Pool reset clears all descriptor sets allocated so far.
    if (pool) vkResetDescriptorPool(vk->device, pool, 0);

    A_row_ptr = make_buffer(*vk, A.row_ptr.size() * sizeof(int),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    A_col_idx = make_buffer(*vk, A.col_idx.size() * sizeof(int),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    A_values  = make_buffer(*vk, nnz * sizeof(float),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    A_diag    = make_buffer(*vk, n_rows * sizeof(float),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::memcpy(A_row_ptr.mapped, A.row_ptr.data(), A.row_ptr.size() * sizeof(int));
    std::memcpy(A_col_idx.mapped, A.col_idx.data(), A.col_idx.size() * sizeof(int));
    {
        std::vector<float> v32(nnz);
        for (std::size_t i = 0; i < nnz; ++i) v32[i] = static_cast<float>(A.values[i]);
        std::memcpy(A_values.mapped, v32.data(), nnz * sizeof(float));
    }
    {
        const std::vector<double> diag = sp::diagonal(A);
        std::vector<float> d32(n_rows);
        for (std::uint32_t i = 0; i < n_rows; ++i) d32[i] = static_cast<float>(diag[i]);
        std::memcpy(A_diag.mapped, d32.data(), n_rows * sizeof(float));
    }

    b_buf  = make_buffer(*vk, n_rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    x_buf  = make_buffer(*vk, n_rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_buf  = make_buffer(*vk, n_rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    z_buf  = make_buffer(*vk, n_rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    p_buf  = make_buffer(*vk, n_rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Ap_buf = make_buffer(*vk, n_rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dot_pair = make_buffer(*vk, 2 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    u_spmv   = make_buffer(*vk, sizeof(std::uint32_t),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    u_n      = make_buffer(*vk, sizeof(std::uint32_t),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    u_axpy_x = make_buffer(*vk, sizeof(std::uint32_t) + sizeof(float),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    u_axpy_r = make_buffer(*vk, sizeof(std::uint32_t) + sizeof(float),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    u_saxpby = make_buffer(*vk, sizeof(std::uint32_t) + 2 * sizeof(float),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Write the static uniform contents (n / rows) up-front.
    std::memcpy(u_spmv.mapped, &n_rows, sizeof(std::uint32_t));
    std::memcpy(u_n.mapped,    &n_rows, sizeof(std::uint32_t));
    write_axpy_uniform(u_axpy_x, n_rows, 0.0f);
    write_axpy_uniform(u_axpy_r, n_rows, 0.0f);
    write_saxpby_uniform(u_saxpby, n_rows, 1.0f, 0.0f);

    // ----------------------------------------------------------------
    // Bind descriptor sets — one per logical use.
    // ----------------------------------------------------------------
    ds_spmv = alloc_and_bind(vk->device, pool, k_spmv.dsl, u_spmv,
                              { A_row_ptr, A_col_idx, A_values, p_buf, Ap_buf });
    ds_dot_pAp = alloc_and_bind(vk->device, pool, k_dot.dsl, u_n,
                                  { p_buf, Ap_buf, dot_pair });
    ds_dot_rr  = alloc_and_bind(vk->device, pool, k_dot.dsl, u_n,
                                  { r_buf, r_buf, dot_pair });
    ds_dot_rz  = alloc_and_bind(vk->device, pool, k_dot.dsl, u_n,
                                  { r_buf, z_buf, dot_pair });
    ds_axpy_x = alloc_and_bind(vk->device, pool, k_axpy.dsl, u_axpy_x,
                                  { p_buf, x_buf });
    ds_axpy_r = alloc_and_bind(vk->device, pool, k_axpy.dsl, u_axpy_r,
                                  { Ap_buf, r_buf });
    ds_jacobi_b = alloc_and_bind(vk->device, pool, k_jacobi.dsl, u_n,
                                   { A_diag, b_buf, z_buf });
    ds_jacobi_r = alloc_and_bind(vk->device, pool, k_jacobi.dsl, u_n,
                                   { A_diag, r_buf, z_buf });
    ds_saxpby = alloc_and_bind(vk->device, pool, k_saxpby.dsl, u_saxpby,
                                  { z_buf, p_buf, p_buf });
}

inline void GpuCgSolver::write_axpy_uniform(Buffer &u, std::uint32_t n, float alpha) {
    std::memcpy(u.mapped,
                  &n, sizeof(std::uint32_t));
    std::memcpy(static_cast<char *>(u.mapped) + sizeof(std::uint32_t),
                  &alpha, sizeof(float));
}

inline void GpuCgSolver::write_saxpby_uniform(Buffer &u, std::uint32_t n,
                                                 float alpha, float beta) {
    char *p = static_cast<char *>(u.mapped);
    std::memcpy(p + 0, &n,     sizeof(std::uint32_t));
    std::memcpy(p + 4, &alpha, sizeof(float));
    std::memcpy(p + 8, &beta,  sizeof(float));
}

inline void GpuCgSolver::dispatch_spmv  (VkDescriptorSet ds) {
    run_compute_once(*vk, k_spmv  .pipeline, k_spmv  .layout, ds, (n_rows + 63) / 64);
}
inline void GpuCgSolver::dispatch_dot   (VkDescriptorSet ds) {
    // Single workgroup grid-strided over the input.
    run_compute_once(*vk, k_dot   .pipeline, k_dot   .layout, ds, 1);
}
inline void GpuCgSolver::dispatch_axpy  (VkDescriptorSet ds) {
    run_compute_once(*vk, k_axpy  .pipeline, k_axpy  .layout, ds, (n_rows + 63) / 64);
}
inline void GpuCgSolver::dispatch_jacobi(VkDescriptorSet ds) {
    run_compute_once(*vk, k_jacobi.pipeline, k_jacobi.layout, ds, (n_rows + 63) / 64);
}
inline void GpuCgSolver::dispatch_saxpby(VkDescriptorSet ds) {
    run_compute_once(*vk, k_saxpby.pipeline, k_saxpby.layout, ds, (n_rows + 63) / 64);
}

inline double GpuCgSolver::read_dot_scalar() const {
    float pair[2];
    std::memcpy(pair, dot_pair.mapped, 2 * sizeof(float));
    return static_cast<double>(pair[0]) + static_cast<double>(pair[1]);
}

inline std::vector<double> GpuCgSolver::solve_with_guess(
        const std::vector<double> &b, const std::vector<double> &x0,
        std::size_t max_iter, double tol, std::size_t *iters_used) {
    // Upload b and x0 (fp32).
    {
        std::vector<float> b32(n_rows);
        for (std::uint32_t i = 0; i < n_rows; ++i) b32[i] = static_cast<float>(b[i]);
        std::memcpy(b_buf.mapped, b32.data(), n_rows * sizeof(float));
        std::vector<float> x32(n_rows);
        for (std::uint32_t i = 0; i < n_rows; ++i) x32[i] = static_cast<float>(x0[i]);
        std::memcpy(x_buf.mapped, x32.data(), n_rows * sizeof(float));
        std::memcpy(p_buf.mapped, x32.data(), n_rows * sizeof(float));
    }

    // Ap ← A · p   (with p == x0).
    dispatch_spmv(ds_spmv);

    // r ← b   then   r ← r + (-1)·Ap   ⇒   r = b − A·x0.
    std::memcpy(r_buf.mapped, b_buf.mapped, n_rows * sizeof(float));
    write_axpy_uniform(u_axpy_r, n_rows, -1.0f);
    dispatch_axpy(ds_axpy_r);

    // z ← M⁻¹ · r.
    dispatch_jacobi(ds_jacobi_r);
    std::memcpy(p_buf.mapped, z_buf.mapped, n_rows * sizeof(float));

    dispatch_dot(ds_dot_rz);
    double rz_old = read_dot_scalar();

    const double tol_sq = tol * tol;
    std::size_t iter = 0;
    for (; iter < max_iter; ++iter) {
        dispatch_spmv(ds_spmv);
        dispatch_dot(ds_dot_pAp);
        const double pAp = read_dot_scalar();
        if (pAp == 0.0) break;
        const double alpha = rz_old / pAp;
        write_axpy_uniform(u_axpy_x, n_rows, static_cast<float>(alpha));
        dispatch_axpy(ds_axpy_x);
        write_axpy_uniform(u_axpy_r, n_rows, static_cast<float>(-alpha));
        dispatch_axpy(ds_axpy_r);
        dispatch_dot(ds_dot_rr);
        const double rr = read_dot_scalar();
        if (rr < tol_sq) { ++iter; break; }
        dispatch_jacobi(ds_jacobi_r);
        dispatch_dot(ds_dot_rz);
        const double rz_new = read_dot_scalar();
        const double beta   = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);
        write_saxpby_uniform(u_saxpby, n_rows, 1.0f, static_cast<float>(beta));
        dispatch_saxpby(ds_saxpby);
        rz_old = rz_new;
    }
    if (iters_used) *iters_used = iter;

    std::vector<double> out(n_rows);
    {
        std::vector<float> x32(n_rows);
        std::memcpy(x32.data(), x_buf.mapped, n_rows * sizeof(float));
        for (std::uint32_t i = 0; i < n_rows; ++i) out[i] = static_cast<double>(x32[i]);
    }
    return out;
}

inline std::vector<double> GpuCgSolver::solve(const std::vector<double> &b,
                                                  std::size_t max_iter, double tol,
                                                  std::size_t *iters_used) {
    // Upload b (fp32) and zero x.
    {
        std::vector<float> b32(n_rows);
        for (std::uint32_t i = 0; i < n_rows; ++i) b32[i] = static_cast<float>(b[i]);
        std::memcpy(b_buf.mapped, b32.data(), n_rows * sizeof(float));
    }
    std::memset(x_buf.mapped, 0, n_rows * sizeof(float));

    // r ← b   (since x₀ = 0).
    std::memcpy(r_buf.mapped, b_buf.mapped, n_rows * sizeof(float));

    // z ← M⁻¹ · b   (uses ds_jacobi_b which reads b → writes z).
    dispatch_jacobi(ds_jacobi_b);

    // p ← z   (one host-side memcpy is cheaper than a saxpby dispatch).
    std::memcpy(p_buf.mapped, z_buf.mapped, n_rows * sizeof(float));

    // ρ_old = r·z.
    dispatch_dot(ds_dot_rz);
    double rz_old = read_dot_scalar();

    const double tol_sq = tol * tol;
    std::size_t iter = 0;
    for (; iter < max_iter; ++iter) {
        // Ap ← A · p.
        dispatch_spmv(ds_spmv);

        // pAp = p · Ap.
        dispatch_dot(ds_dot_pAp);
        const double pAp = read_dot_scalar();
        if (pAp == 0.0) break;

        const double alpha = rz_old / pAp;
        write_axpy_uniform(u_axpy_x, n_rows, static_cast<float>(alpha));
        dispatch_axpy(ds_axpy_x);   // x ← x + α·p
        write_axpy_uniform(u_axpy_r, n_rows, static_cast<float>(-alpha));
        dispatch_axpy(ds_axpy_r);   // r ← r + (−α)·Ap

        // ‖r‖² = r · r.
        dispatch_dot(ds_dot_rr);
        const double rr = read_dot_scalar();
        if (rr < tol_sq) {
            ++iter;
            break;
        }

        // z ← M⁻¹ · r.
        dispatch_jacobi(ds_jacobi_r);

        // ρ_new = r · z.
        dispatch_dot(ds_dot_rz);
        const double rz_new = read_dot_scalar();
        const double beta   = (rz_old == 0.0) ? 0.0 : (rz_new / rz_old);

        // p ← 1·z + β·p.
        write_saxpby_uniform(u_saxpby, n_rows, 1.0f, static_cast<float>(beta));
        dispatch_saxpby(ds_saxpby);

        rz_old = rz_new;
    }
    if (iters_used) *iters_used = iter;

    // Read back x as fp64 for the host caller.
    std::vector<double> out(n_rows);
    {
        std::vector<float> x32(n_rows);
        std::memcpy(x32.data(), x_buf.mapped, n_rows * sizeof(float));
        for (std::uint32_t i = 0; i < n_rows; ++i) out[i] = static_cast<double>(x32[i]);
    }
    return out;
}

} // namespace curvenet_gpu_test

#endif
