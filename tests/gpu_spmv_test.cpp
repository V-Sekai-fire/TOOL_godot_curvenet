// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Standalone Vulkan-compute spmv test. Confirms the GPU path the
// deformer's todos/08 GPU CG milestone will use produces the same
// answer as the CPU sparse::spmv on a few SPD systems, without
// dragging in Godot or RenderingDevice.
//
// On macOS this runs on MoltenVK via the homebrew vulkan-loader.
// Build + run with `make -C tests gpu`. The Makefile rule sets
// VK_ICD_FILENAMES so the loader can find MoltenVK.
//
// Same SPIR-V will eventually be dispatched through Godot's
// RenderingDevice on Steam Deck and Quest 3 (both Vulkan), so this
// test gates the algorithm before the Godot wiring lands.

#include "curvenet/sparse_linalg.h"
#include "gpu_compute_helpers.h"

#include <algorithm>

namespace sp = curvenet::sparse;
using namespace curvenet_gpu_test;

namespace {

struct SpmvPipeline {
    VkDescriptorSetLayout dsl       = VK_NULL_HANDLE;
    VkPipelineLayout      layout    = VK_NULL_HANDLE;
    VkShaderModule        shader    = VK_NULL_HANDLE;
    VkPipeline            pipeline  = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set  = VK_NULL_HANDLE;

    void init(const VulkanCompute &vk, const std::vector<std::uint32_t> &spv) {
        VkDescriptorSetLayoutBinding bindings[6]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        for (int i = 1; i < 6; ++i) {
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 6;
        dlci.pBindings    = bindings;
        VK_OR_DIE(vkCreateDescriptorSetLayout(vk.device, &dlci, nullptr, &dsl));

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &dsl;
        VK_OR_DIE(vkCreatePipelineLayout(vk.device, &plci, nullptr, &layout));

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(std::uint32_t);
        smci.pCode    = spv.data();
        VK_OR_DIE(vkCreateShaderModule(vk.device, &smci, nullptr, &shader));

        VkComputePipelineCreateInfo cpci{};
        cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader;
        cpci.stage.pName  = "main";
        cpci.layout       = layout;
        VK_OR_DIE(vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &cpci,
                                            nullptr, &pipeline));

        VkDescriptorPoolSize pool_sizes[2]{};
        pool_sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[0].descriptorCount = 1;
        pool_sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[1].descriptorCount = 5;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = pool_sizes;
        VK_OR_DIE(vkCreateDescriptorPool(vk.device, &dpci, nullptr, &desc_pool));

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &dsl;
        VK_OR_DIE(vkAllocateDescriptorSets(vk.device, &dsai, &desc_set));
    }

    void bind_buffers(VkDevice device,
                       const Buffer &params, const Buffer &row_ptr,
                       const Buffer &col_idx, const Buffer &values,
                       const Buffer &x, const Buffer &y) {
        VkDescriptorBufferInfo infos[6]{};
        infos[0] = { params.handle,  0, VK_WHOLE_SIZE };
        infos[1] = { row_ptr.handle, 0, VK_WHOLE_SIZE };
        infos[2] = { col_idx.handle, 0, VK_WHOLE_SIZE };
        infos[3] = { values.handle,  0, VK_WHOLE_SIZE };
        infos[4] = { x.handle,       0, VK_WHOLE_SIZE };
        infos[5] = { y.handle,       0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[6]{};
        for (int i = 0; i < 6; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = desc_set;
            writes[i].dstBinding      = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = (i == 0)
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &infos[i];
        }
        vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    }

    void shutdown(VkDevice device) {
        if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, nullptr);
        if (pipeline)  vkDestroyPipeline(device, pipeline, nullptr);
        if (shader)    vkDestroyShaderModule(device, shader, nullptr);
        if (layout)    vkDestroyPipelineLayout(device, layout, nullptr);
        if (dsl)       vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    }
};

std::vector<double> gpu_spmv(VulkanCompute &vk, SpmvPipeline &pipe,
                                const sp::SparseMatrixCSR &A,
                                const std::vector<double> &x) {
    const std::uint32_t rows = static_cast<std::uint32_t>(A.rows);
    const std::size_t   nnz  = A.values.size();

    Buffer params  = make_buffer(vk, sizeof(std::uint32_t),
                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    Buffer row_ptr = make_buffer(vk, A.row_ptr.size() * sizeof(int),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer col_idx = make_buffer(vk, A.col_idx.size() * sizeof(int),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer values  = make_buffer(vk, nnz * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer x_buf   = make_buffer(vk, x.size() * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer y_buf   = make_buffer(vk, rows * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::memcpy(params.mapped,  &rows,            sizeof(std::uint32_t));
    std::memcpy(row_ptr.mapped, A.row_ptr.data(), A.row_ptr.size() * sizeof(int));
    std::memcpy(col_idx.mapped, A.col_idx.data(), A.col_idx.size() * sizeof(int));
    {
        std::vector<float> v32(nnz);
        for (std::size_t i = 0; i < nnz; ++i) v32[i] = static_cast<float>(A.values[i]);
        std::memcpy(values.mapped, v32.data(), v32.size() * sizeof(float));
    }
    {
        std::vector<float> x32(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) x32[i] = static_cast<float>(x[i]);
        std::memcpy(x_buf.mapped, x32.data(), x32.size() * sizeof(float));
    }
    std::memset(y_buf.mapped, 0, rows * sizeof(float));

    pipe.bind_buffers(vk.device, params, row_ptr, col_idx, values, x_buf, y_buf);
    const std::uint32_t groups = (rows + 63) / 64;
    run_compute_once(vk, pipe.pipeline, pipe.layout, pipe.desc_set, groups);

    std::vector<double> y_out(rows, 0.0);
    {
        std::vector<float> y32(rows);
        std::memcpy(y32.data(), y_buf.mapped, rows * sizeof(float));
        for (std::uint32_t i = 0; i < rows; ++i) y_out[i] = static_cast<double>(y32[i]);
    }

    destroy_buffer(vk.device, y_buf);
    destroy_buffer(vk.device, x_buf);
    destroy_buffer(vk.device, values);
    destroy_buffer(vk.device, col_idx);
    destroy_buffer(vk.device, row_ptr);
    destroy_buffer(vk.device, params);
    return y_out;
}

sp::SparseMatrixCSR laplacian4() {
    sp::SparseMatrixCSR A;
    A.rows = 4;
    A.cols = 4;
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

sp::SparseMatrixCSR diag5() {
    sp::SparseMatrixCSR A;
    A.rows = 5;
    A.cols = 5;
    A.row_ptr = { 0, 1, 2, 3, 4, 5 };
    A.col_idx = { 0, 1, 2, 3, 4 };
    A.values  = { 2.0, 3.0, 4.0, 5.0, 6.0 };
    return A;
}

sp::SparseMatrixCSR grid_laplacian(int n) {
    const int nv = n * n;
    std::vector<std::vector<std::pair<int, double>>> rows(nv);
    auto idx = [n](int i, int j) { return j * n + i; };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int v = idx(i, j);
            double diag = 0.0;
            const int neigh[4][2] = { {i+1,j}, {i-1,j}, {i,j+1}, {i,j-1} };
            for (auto &nb : neigh) {
                if (nb[0] < 0 || nb[0] >= n || nb[1] < 0 || nb[1] >= n) continue;
                rows[v].push_back({ idx(nb[0], nb[1]), -1.0 });
                diag += 1.0;
            }
            rows[v].push_back({ v, diag });
        }
    }
    sp::SparseMatrixCSR A;
    A.rows = nv;
    A.cols = nv;
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

int run_case(VulkanCompute &vk, SpmvPipeline &pipe,
              const char *name,
              const sp::SparseMatrixCSR &A,
              const std::vector<double> &x) {
    const std::vector<double> y_cpu = sp::spmv(A, x);
    const std::vector<double> y_gpu = gpu_spmv(vk, pipe, A, x);
    const bool ok = vec_close(y_cpu, y_gpu, 1e-5, 1e-5);
    std::printf("  %-30s  %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        std::printf("    cpu:");
        for (std::size_t i = 0; i < y_cpu.size() && i < 8; ++i) std::printf(" %.6f", y_cpu[i]);
        std::printf("\n    gpu:");
        for (std::size_t i = 0; i < y_gpu.size() && i < 8; ++i) std::printf(" %.6f", y_gpu[i]);
        std::printf("\n");
    }
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    const char *spv_path = (argc > 1) ? argv[1] : "bin/spmv.spv";
    std::printf("Loading SPIR-V from %s\n", spv_path);
    const std::vector<std::uint32_t> spv = load_spv(spv_path);

    VulkanCompute vk;
    vk.init();

    SpmvPipeline pipe;
    pipe.init(vk, spv);

    int failures = 0;

    failures += run_case(vk, pipe, "laplacian4 * (1,2,3,4)",
                          laplacian4(), { 1.0, 2.0, 3.0, 4.0 });

    failures += run_case(vk, pipe, "diag5 * ones",
                          diag5(), { 1.0, 1.0, 1.0, 1.0, 1.0 });

    {
        const auto A = grid_laplacian(5);
        std::vector<double> x(25, 0.0);
        x[12] = 1.0;
        failures += run_case(vk, pipe, "grid_laplacian(5) * e12", A, x);
    }

    {
        const auto A = grid_laplacian(7);
        std::vector<double> x(49);
        for (int i = 0; i < 49; ++i) x[i] = 0.1 * i;
        failures += run_case(vk, pipe, "grid_laplacian(7) * ramp", A, x);
    }

    {
        const auto A = grid_laplacian(10);
        std::vector<double> x(100);
        for (int i = 0; i < 100; ++i) x[i] = (i % 2 == 0) ? 1.0 : -1.0;
        failures += run_case(vk, pipe, "grid_laplacian(10) * +/-1", A, x);
    }

    pipe.shutdown(vk.device);
    vk.shutdown();

    std::printf("\n%s: %d failing case(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
