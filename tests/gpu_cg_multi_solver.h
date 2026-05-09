// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Multi-RHS GPU CG: solves A · X = B for B of shape nv × k in one
// host-orchestrated CG iteration loop. The deformer's per-frame
// pattern is a 9-RHS Fv solve plus a 3-RHS Xv solve; bundling the
// columns into one nv*k working buffer means the same A is read
// once per spmv across all k columns, and the dispatch overhead is
// amortised k×.
//
// Same algorithmic structure as GpuCgSolver (preconditioned CG,
// pre-recorded command buffers, df32 dot products) but with the
// _multi shader variants. Per-iter scalars (alpha, beta, pAp, rr,
// rz_old, rz_new) are length-k arrays now; the host computes them
// from the df32 pair output and writes them into per-column storage
// buffers that the GPU reads on the next dispatch.
//
// Convergence: terminates when max_c (rr[c]) < tol². All-columns
// criterion would let cheap columns wait while a hard column
// finishes; for the deformer's 9-RHS / 3-RHS uses the columns are
// similarly conditioned and this is fine.

#ifndef CURVENET_GPU_CG_MULTI_SOLVER_H
#define CURVENET_GPU_CG_MULTI_SOLVER_H

#include "curvenet/sparse_linalg.h"
#include "gpu_compute_helpers.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace curvenet_gpu_test {

namespace sp = curvenet::sparse;

class GpuCgMultiSolver {
public:
    void init(VulkanCompute &vk_in,
               const std::vector<std::uint32_t> &spmv_multi_spv,
               const std::vector<std::uint32_t> &dot_multi_spv,
               const std::vector<std::uint32_t> &axpy_multi_spv,
               const std::vector<std::uint32_t> &jacobi_multi_spv,
               const std::vector<std::uint32_t> &saxpby_multi_spv);
    void prepare(const sp::SparseMatrixCSR &A, std::uint32_t k);
    // Solve A·X = B for k stacked columns. B is row-major nv × k.
    // Returns X as row-major nv × k. Optionally reports iter count.
    std::vector<double> solve(const std::vector<double> &B,
                                std::size_t max_iter, double tol,
                                std::size_t *iters_used = nullptr);
    void shutdown();

private:
    double read_max_rr() const;
    void   read_pair_array(std::vector<double> &out) const;
    void   write_alpha(const std::vector<double> &alpha_d,
                         bool negate);
    void   write_alpha_pair();   // writes both u_alpha and u_alpha_neg
    void   write_beta(const std::vector<double> &beta_d);
    void   write_uniform_n_k(Buffer &u);

    VulkanCompute *vk = nullptr;

    ComputeKernel k_spmv;
    ComputeKernel k_dot;
    ComputeKernel k_axpy;
    ComputeKernel k_jacobi;
    ComputeKernel k_saxpby;

    VkDescriptorPool pool = VK_NULL_HANDLE;

    Buffer A_row_ptr, A_col_idx, A_values, A_diag;

    Buffer b_buf, x_buf, r_buf, z_buf, p_buf, Ap_buf;
    Buffer dot_pair;          // 2 * k floats: k df32 (hi, lo) pairs

    Buffer u_n_k;             // {uint n; uint k;}
    Buffer alpha_buf;         // length-k storage (per-column α)
    Buffer alpha_neg_buf;     // length-k storage (per-column −α)
    Buffer beta_buf;          // length-k storage (per-column β)
    Buffer ones_buf;          // length-k storage of 1.0f, for saxpby α slot

    VkDescriptorSet ds_spmv      = VK_NULL_HANDLE;
    VkDescriptorSet ds_dot_pAp   = VK_NULL_HANDLE;
    VkDescriptorSet ds_dot_rr    = VK_NULL_HANDLE;
    VkDescriptorSet ds_dot_rz    = VK_NULL_HANDLE;
    VkDescriptorSet ds_axpy_x    = VK_NULL_HANDLE;
    VkDescriptorSet ds_axpy_r    = VK_NULL_HANDLE;
    VkDescriptorSet ds_jacobi_b  = VK_NULL_HANDLE;
    VkDescriptorSet ds_jacobi_r  = VK_NULL_HANDLE;
    VkDescriptorSet ds_saxpby    = VK_NULL_HANDLE;

    CommandBatch batch_spmv_pAp;
    CommandBatch batch_axpy_rr;
    CommandBatch batch_jacobi_rz;
    CommandBatch batch_init_cold;

    std::uint32_t n_rows = 0;
    std::uint32_t k_cols = 0;
};

inline void GpuCgMultiSolver::init(VulkanCompute &vk_in,
                                       const std::vector<std::uint32_t> &spmv_multi_spv,
                                       const std::vector<std::uint32_t> &dot_multi_spv,
                                       const std::vector<std::uint32_t> &axpy_multi_spv,
                                       const std::vector<std::uint32_t> &jacobi_multi_spv,
                                       const std::vector<std::uint32_t> &saxpby_multi_spv) {
    vk = &vk_in;
    k_spmv  .init(vk->device, spmv_multi_spv,   5);
    k_dot   .init(vk->device, dot_multi_spv,    3);
    k_axpy  .init(vk->device, axpy_multi_spv,   3);
    k_jacobi.init(vk->device, jacobi_multi_spv, 3);
    k_saxpby.init(vk->device, saxpby_multi_spv, 5);

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 16;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 80;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 16;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = sizes;
    VK_OR_DIE(vkCreateDescriptorPool(vk->device, &dpci, nullptr, &pool));

    batch_spmv_pAp .init(*vk);
    batch_axpy_rr  .init(*vk);
    batch_jacobi_rz.init(*vk);
    batch_init_cold.init(*vk);
}

inline void GpuCgMultiSolver::shutdown() {
    if (!vk) return;
    auto kill = [this](Buffer &b) { destroy_buffer(vk->device, b); };
    kill(A_row_ptr); kill(A_col_idx); kill(A_values); kill(A_diag);
    kill(b_buf); kill(x_buf); kill(r_buf); kill(z_buf);
    kill(p_buf); kill(Ap_buf); kill(dot_pair);
    kill(u_n_k); kill(alpha_buf); kill(alpha_neg_buf); kill(beta_buf); kill(ones_buf);
    batch_spmv_pAp .shutdown();
    batch_axpy_rr  .shutdown();
    batch_jacobi_rz.shutdown();
    batch_init_cold.shutdown();
    if (pool) vkDestroyDescriptorPool(vk->device, pool, nullptr);
    pool = VK_NULL_HANDLE;
    k_spmv  .shutdown(vk->device);
    k_dot   .shutdown(vk->device);
    k_axpy  .shutdown(vk->device);
    k_jacobi.shutdown(vk->device);
    k_saxpby.shutdown(vk->device);
    vk = nullptr;
}

inline void GpuCgMultiSolver::prepare(const sp::SparseMatrixCSR &A, std::uint32_t k) {
    n_rows = static_cast<std::uint32_t>(A.rows);
    k_cols = k;
    const std::size_t nnz = A.values.size();

    auto reset = [this](Buffer &b) { if (b.handle) destroy_buffer(vk->device, b); };
    reset(A_row_ptr); reset(A_col_idx); reset(A_values); reset(A_diag);
    reset(b_buf); reset(x_buf); reset(r_buf); reset(z_buf);
    reset(p_buf); reset(Ap_buf); reset(dot_pair);
    reset(u_n_k); reset(alpha_buf); reset(alpha_neg_buf); reset(beta_buf); reset(ones_buf);
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

    const std::size_t nk_bytes = n_rows * k_cols * sizeof(float);
    b_buf  = make_buffer(*vk, nk_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    x_buf  = make_buffer(*vk, nk_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    r_buf  = make_buffer(*vk, nk_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    z_buf  = make_buffer(*vk, nk_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    p_buf  = make_buffer(*vk, nk_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Ap_buf = make_buffer(*vk, nk_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    dot_pair = make_buffer(*vk, 2 * k_cols * sizeof(float),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    u_n_k         = make_buffer(*vk, 8, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    alpha_buf     = make_buffer(*vk, k_cols * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    alpha_neg_buf = make_buffer(*vk, k_cols * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    beta_buf      = make_buffer(*vk, k_cols * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ones_buf      = make_buffer(*vk, k_cols * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    write_uniform_n_k(u_n_k);
    {
        std::vector<float> ones(k_cols, 1.0f);
        std::memcpy(ones_buf.mapped, ones.data(), k_cols * sizeof(float));
    }
    std::memset(alpha_buf    .mapped, 0, k_cols * sizeof(float));
    std::memset(alpha_neg_buf.mapped, 0, k_cols * sizeof(float));
    std::memset(beta_buf     .mapped, 0, k_cols * sizeof(float));

    // Descriptor sets — one per logical use.
    ds_spmv = alloc_and_bind(vk->device, pool, k_spmv.dsl, u_n_k,
                              { A_row_ptr, A_col_idx, A_values, p_buf, Ap_buf });
    ds_dot_pAp = alloc_and_bind(vk->device, pool, k_dot.dsl, u_n_k,
                                  { p_buf, Ap_buf, dot_pair });
    ds_dot_rr  = alloc_and_bind(vk->device, pool, k_dot.dsl, u_n_k,
                                  { r_buf, r_buf, dot_pair });
    ds_dot_rz  = alloc_and_bind(vk->device, pool, k_dot.dsl, u_n_k,
                                  { r_buf, z_buf, dot_pair });
    // axpy_multi binding order: Alpha, X, Y. ds_axpy_x: alpha, p, x.
    ds_axpy_x = alloc_and_bind(vk->device, pool, k_axpy.dsl, u_n_k,
                                  { alpha_buf,     p_buf,  x_buf });
    ds_axpy_r = alloc_and_bind(vk->device, pool, k_axpy.dsl, u_n_k,
                                  { alpha_neg_buf, Ap_buf, r_buf });
    // jacobi_multi binding order: D, B, Y.
    ds_jacobi_b = alloc_and_bind(vk->device, pool, k_jacobi.dsl, u_n_k,
                                   { A_diag, b_buf, z_buf });
    ds_jacobi_r = alloc_and_bind(vk->device, pool, k_jacobi.dsl, u_n_k,
                                   { A_diag, r_buf, z_buf });
    // saxpby_multi binding order: Alpha, Beta, X, Y, Out.
    // p ← 1·z + β·p  →  Alpha=ones (length-k 1.0), Beta=β, X=z, Y=p, Out=p.
    ds_saxpby = alloc_and_bind(vk->device, pool, k_saxpby.dsl, u_n_k,
                                  { ones_buf, beta_buf, z_buf, p_buf, p_buf });

    // ---- Pre-record per-iter command buffers ----
    const std::uint32_t gx_rows = (n_rows + 255) / 256;
    const std::uint32_t gx_nk   = (n_rows * k_cols + 255) / 256;

    batch_spmv_pAp.begin();
    batch_spmv_pAp.dispatch(k_saxpby.pipeline, k_saxpby.layout, ds_saxpby, gx_nk);
    batch_spmv_pAp.barrier();
    batch_spmv_pAp.dispatch(k_spmv  .pipeline, k_spmv  .layout, ds_spmv,    gx_rows);
    batch_spmv_pAp.barrier();
    batch_spmv_pAp.dispatch(k_dot   .pipeline, k_dot   .layout, ds_dot_pAp, 1);
    batch_spmv_pAp.end();

    batch_axpy_rr.begin();
    batch_axpy_rr.dispatch(k_axpy.pipeline, k_axpy.layout, ds_axpy_x, gx_nk);
    batch_axpy_rr.dispatch(k_axpy.pipeline, k_axpy.layout, ds_axpy_r, gx_nk);
    batch_axpy_rr.barrier();
    batch_axpy_rr.dispatch(k_dot .pipeline, k_dot .layout, ds_dot_rr, 1);
    batch_axpy_rr.end();

    batch_jacobi_rz.begin();
    batch_jacobi_rz.dispatch(k_jacobi.pipeline, k_jacobi.layout, ds_jacobi_r, gx_nk);
    batch_jacobi_rz.barrier();
    batch_jacobi_rz.dispatch(k_dot   .pipeline, k_dot   .layout, ds_dot_rz,    1);
    batch_jacobi_rz.end();

    batch_init_cold.begin();
    batch_init_cold.dispatch(k_jacobi.pipeline, k_jacobi.layout, ds_jacobi_b, gx_nk);
    batch_init_cold.barrier();
    batch_init_cold.dispatch(k_dot   .pipeline, k_dot   .layout, ds_dot_rz,    1);
    batch_init_cold.end();
}

inline void GpuCgMultiSolver::write_uniform_n_k(Buffer &u) {
    std::memcpy(static_cast<char *>(u.mapped) + 0, &n_rows, 4);
    std::memcpy(static_cast<char *>(u.mapped) + 4, &k_cols, 4);
}

inline double GpuCgMultiSolver::read_max_rr() const {
    std::vector<double> rr(k_cols);
    read_pair_array(rr);
    double m = 0.0;
    for (double v : rr) if (v > m) m = v;
    return m;
}

inline void GpuCgMultiSolver::read_pair_array(std::vector<double> &out) const {
    std::vector<float> pair(2 * k_cols);
    std::memcpy(pair.data(), dot_pair.mapped, 2 * k_cols * sizeof(float));
    for (std::uint32_t c = 0; c < k_cols; ++c) {
        out[c] = static_cast<double>(pair[2 * c + 0]) +
                 static_cast<double>(pair[2 * c + 1]);
    }
}

inline void GpuCgMultiSolver::write_alpha(const std::vector<double> &alpha_d,
                                              bool negate) {
    std::vector<float> a32(k_cols);
    for (std::uint32_t c = 0; c < k_cols; ++c) {
        a32[c] = static_cast<float>(negate ? -alpha_d[c] : alpha_d[c]);
    }
    Buffer &dst = negate ? alpha_neg_buf : alpha_buf;
    std::memcpy(dst.mapped, a32.data(), k_cols * sizeof(float));
}

inline void GpuCgMultiSolver::write_beta(const std::vector<double> &beta_d) {
    std::vector<float> b32(k_cols);
    for (std::uint32_t c = 0; c < k_cols; ++c) b32[c] = static_cast<float>(beta_d[c]);
    std::memcpy(beta_buf.mapped, b32.data(), k_cols * sizeof(float));
}

inline std::vector<double> GpuCgMultiSolver::solve(
        const std::vector<double> &B, std::size_t max_iter, double tol,
        std::size_t *iters_used) {
    // Upload B (fp32) and zero X.
    {
        std::vector<float> b32(n_rows * k_cols);
        for (std::size_t i = 0; i < b32.size(); ++i) b32[i] = static_cast<float>(B[i]);
        std::memcpy(b_buf.mapped, b32.data(), b32.size() * sizeof(float));
    }
    std::memset(x_buf.mapped, 0, n_rows * k_cols * sizeof(float));

    // r ← B (since x₀ = 0).
    std::memcpy(r_buf.mapped, b_buf.mapped, n_rows * k_cols * sizeof(float));

    // Cold init: z = M⁻¹·b ; rz_old[c] = r·z per column.
    batch_init_cold.submit_wait();
    std::vector<double> rz_old(k_cols);
    read_pair_array(rz_old);

    // First iter's saxpby acts as p ← 1·z + 0·p = z. Set β = 0.
    {
        std::vector<double> beta_zero(k_cols, 0.0);
        write_beta(beta_zero);
    }

    const double tol_sq = tol * tol;
    std::vector<double> pAp(k_cols);
    std::vector<double> rr (k_cols);
    std::vector<double> rz_new(k_cols);
    std::vector<double> alpha (k_cols);
    std::vector<double> beta  (k_cols);

    std::size_t iter = 0;
    for (; iter < max_iter; ++iter) {
        batch_spmv_pAp.submit_wait();
        read_pair_array(pAp);

        bool any_active = false;
        for (std::uint32_t c = 0; c < k_cols; ++c) {
            alpha[c] = (pAp[c] != 0.0) ? (rz_old[c] / pAp[c]) : 0.0;
            if (pAp[c] != 0.0) any_active = true;
        }
        if (!any_active) break;
        write_alpha(alpha, false);
        write_alpha(alpha, true);

        batch_axpy_rr.submit_wait();
        read_pair_array(rr);
        double max_rr = 0.0;
        for (double v : rr) if (v > max_rr) max_rr = v;
        if (max_rr < tol_sq) {
            ++iter;
            break;
        }

        batch_jacobi_rz.submit_wait();
        read_pair_array(rz_new);
        for (std::uint32_t c = 0; c < k_cols; ++c) {
            beta[c] = (rz_old[c] != 0.0) ? (rz_new[c] / rz_old[c]) : 0.0;
        }
        write_beta(beta);
        rz_old = rz_new;
    }
    if (iters_used) *iters_used = iter;

    std::vector<double> X(n_rows * k_cols);
    {
        std::vector<float> x32(n_rows * k_cols);
        std::memcpy(x32.data(), x_buf.mapped, x32.size() * sizeof(float));
        for (std::size_t i = 0; i < x32.size(); ++i) X[i] = static_cast<double>(x32[i]);
    }
    return X;
}

} // namespace curvenet_gpu_test

#endif
