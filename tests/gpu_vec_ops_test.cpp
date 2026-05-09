// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Standalone Vulkan-compute tests for the three fp32 element-wise
// vector kernels that complete the GPU CG kernel set:
//
//   axpy.comp     y[i] += alpha * x[i]
//   jacobi.comp   y[i] = b[i] / d[i]   (zero diagonal -> 0)
//   saxpby.comp   out[i] = alpha * x[i] + beta * y[i]
//
// Each kernel is checked against the corresponding CPU reference in
// curvenet::sparse (axpy_inplace, apply_jacobi, saxpby) on a few
// hand-rolled and one larger pseudo-random case. Tolerances are tight
// — these kernels are pure fp32, no accumulation across elements, so
// the GPU result is bitwise-equal to the CPU reference up to FMA
// fusion order. We allow ~2 ulp absolute slack to cover the FMA case.
//
// Run with `make -C tests gpu`.

#include "curvenet/sparse_linalg.h"
#include "gpu_compute_helpers.h"

#include <cstdint>
#include <random>
#include <vector>

namespace sp = curvenet::sparse;
using namespace curvenet_gpu_test;

namespace {

VkPipeline       make_pipeline(VkDevice device, VkPipelineLayout layout,
                                 VkShaderModule shader) {
    VkComputePipelineCreateInfo cpci{};
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName  = "main";
    cpci.layout       = layout;
    VkPipeline p = VK_NULL_HANDLE;
    VK_OR_DIE(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
    return p;
}

VkShaderModule make_shader(VkDevice device, const std::vector<std::uint32_t> &spv) {
    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size() * sizeof(std::uint32_t);
    smci.pCode    = spv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    VK_OR_DIE(vkCreateShaderModule(device, &smci, nullptr, &m));
    return m;
}

VkDescriptorSetLayout make_dsl(VkDevice device, std::uint32_t storage_count) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(1 + storage_count);
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    for (std::uint32_t i = 0; i < storage_count; ++i) {
        bindings[1 + i].binding         = 1 + i;
        bindings[1 + i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1 + i].descriptorCount = 1;
        bindings[1 + i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = static_cast<std::uint32_t>(bindings.size());
    dlci.pBindings    = bindings.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VK_OR_DIE(vkCreateDescriptorSetLayout(device, &dlci, nullptr, &dsl));
    return dsl;
}

VkPipelineLayout make_layout(VkDevice device, VkDescriptorSetLayout dsl) {
    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &dsl;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VK_OR_DIE(vkCreatePipelineLayout(device, &plci, nullptr, &pl));
    return pl;
}

VkDescriptorPool make_pool(VkDevice device, std::uint32_t storage_count) {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 1;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = storage_count;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_OR_DIE(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));
    return pool;
}

VkDescriptorSet alloc_set(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout dsl) {
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_OR_DIE(vkAllocateDescriptorSets(device, &dsai, &set));
    return set;
}

void bind_buffers(VkDevice device, VkDescriptorSet set,
                   const Buffer &uniform, const std::vector<Buffer> &storage) {
    std::vector<VkDescriptorBufferInfo> infos(1 + storage.size());
    infos[0] = { uniform.handle, 0, VK_WHOLE_SIZE };
    for (std::size_t i = 0; i < storage.size(); ++i) {
        infos[1 + i] = { storage[i].handle, 0, VK_WHOLE_SIZE };
    }
    std::vector<VkWriteDescriptorSet> writes(infos.size());
    for (std::size_t i = 0; i < writes.size(); ++i) {
        writes[i] = {};
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = static_cast<std::uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = (i == 0)
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                            writes.data(), 0, nullptr);
}

std::vector<float> to_fp32(const std::vector<double> &v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
    return out;
}

std::vector<double> to_fp64(const float *p, std::size_t n) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
    return out;
}

bool close(const std::vector<double> &gpu, const std::vector<double> &cpu,
            double abs_tol, double rel_tol, std::size_t print_max = 6) {
    if (gpu.size() != cpu.size()) return false;
    double worst = 0.0;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const double d = std::fabs(gpu[i] - cpu[i]);
        const double m = std::fabs(gpu[i]) + std::fabs(cpu[i]);
        const double r = (m > 0.0) ? (d / m) : d;
        if (r > worst) { worst = r; worst_i = i; }
        if (d > abs_tol + rel_tol * m) {
            std::printf("    mismatch at i=%zu: gpu=%.10g cpu=%.10g abs_diff=%.3e rel=%.3e\n",
                          i, gpu[i], cpu[i], d, r);
            (void)print_max;
            return false;
        }
    }
    std::printf("    worst rel %.3e at i=%zu\n", worst, worst_i);
    return true;
}

// ============================================================
// axpy
// ============================================================

int test_axpy(VulkanCompute &vk) {
    const std::vector<std::uint32_t> spv = load_spv("bin/axpy.spv");
    VkShaderModule        sh   = make_shader(vk.device, spv);
    VkDescriptorSetLayout dsl  = make_dsl(vk.device, 2);
    VkPipelineLayout      pl   = make_layout(vk.device, dsl);
    VkPipeline            pipe = make_pipeline(vk.device, pl, sh);
    VkDescriptorPool      pool = make_pool(vk.device, 2);
    VkDescriptorSet       set  = alloc_set(vk.device, pool, dsl);

    int failures = 0;
    auto run = [&](const char *name, double alpha,
                    const std::vector<double> &x, const std::vector<double> &y0) {
        const std::uint32_t n = static_cast<std::uint32_t>(x.size());
        struct UB { std::uint32_t n; float alpha; } ub{ n, static_cast<float>(alpha) };

        Buffer ubuf  = make_buffer(vk, sizeof(UB), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        Buffer x_buf = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer y_buf = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(ubuf.mapped, &ub, sizeof(UB));
        const auto x32 = to_fp32(x);  std::memcpy(x_buf.mapped, x32.data(), n * sizeof(float));
        const auto y32 = to_fp32(y0); std::memcpy(y_buf.mapped, y32.data(), n * sizeof(float));

        bind_buffers(vk.device, set, ubuf, { x_buf, y_buf });
        run_compute_once(vk, pipe, pl, set, (n + 255) / 256);

        const auto y_gpu = to_fp64(static_cast<float *>(y_buf.mapped), n);
        // CPU reference: same fp32-rounded inputs, axpy_inplace mutates a copy.
        std::vector<double> y_cpu_d(y0);
        // Match GPU's fp32 inputs exactly: round inputs to fp32 first.
        for (std::size_t i = 0; i < n; ++i) y_cpu_d[i] = static_cast<double>(y32[i]);
        std::vector<double> x_cpu_d(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) x_cpu_d[i] = static_cast<double>(x32[i]);
        const float alpha_f = static_cast<float>(alpha);
        for (std::size_t i = 0; i < n; ++i) {
            // fma(alpha, x[i], y[i]) in fp32 to mirror the shader.
            const float v = std::fma(alpha_f, x32[i], y32[i]);
            y_cpu_d[i] = static_cast<double>(v);
        }
        const bool ok = close(y_gpu, y_cpu_d, 0.0, 1e-6);
        std::printf("  axpy %-22s n=%-7u alpha=%7.4g  %s\n",
                      name, n, alpha, ok ? "OK" : "FAIL");
        if (!ok) ++failures;

        destroy_buffer(vk.device, y_buf);
        destroy_buffer(vk.device, x_buf);
        destroy_buffer(vk.device, ubuf);
    };

    run("trivial", 2.0, { 1.0, 2.0, 3.0, 4.0 }, { 0.0, 0.0, 0.0, 0.0 });
    run("ones+ones", 1.0, std::vector<double>(1024, 1.0), std::vector<double>(1024, 1.0));
    {
        std::mt19937_64 rng(0xdeadbeefULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> x(50000), y(50000);
        for (auto &v : x) v = d(rng);
        for (auto &v : y) v = d(rng);
        run("random 50k", 0.7, x, y);
    }

    vkDestroyDescriptorPool(vk.device, pool, nullptr);
    vkDestroyPipeline(vk.device, pipe, nullptr);
    vkDestroyShaderModule(vk.device, sh, nullptr);
    vkDestroyPipelineLayout(vk.device, pl, nullptr);
    vkDestroyDescriptorSetLayout(vk.device, dsl, nullptr);
    return failures;
}

// ============================================================
// jacobi
// ============================================================

int test_jacobi(VulkanCompute &vk) {
    const std::vector<std::uint32_t> spv = load_spv("bin/jacobi.spv");
    VkShaderModule        sh   = make_shader(vk.device, spv);
    VkDescriptorSetLayout dsl  = make_dsl(vk.device, 3);
    VkPipelineLayout      pl   = make_layout(vk.device, dsl);
    VkPipeline            pipe = make_pipeline(vk.device, pl, sh);
    VkDescriptorPool      pool = make_pool(vk.device, 3);
    VkDescriptorSet       set  = alloc_set(vk.device, pool, dsl);

    int failures = 0;
    auto run = [&](const char *name,
                    const std::vector<double> &d, const std::vector<double> &b) {
        const std::uint32_t n = static_cast<std::uint32_t>(d.size());

        Buffer ubuf  = make_buffer(vk, sizeof(std::uint32_t),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        Buffer d_buf = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer b_buf = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer y_buf = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(ubuf.mapped, &n, sizeof(std::uint32_t));
        const auto d32 = to_fp32(d); std::memcpy(d_buf.mapped, d32.data(), n * sizeof(float));
        const auto b32 = to_fp32(b); std::memcpy(b_buf.mapped, b32.data(), n * sizeof(float));
        std::memset(y_buf.mapped, 0, n * sizeof(float));

        bind_buffers(vk.device, set, ubuf, { d_buf, b_buf, y_buf });
        run_compute_once(vk, pipe, pl, set, (n + 255) / 256);

        const auto y_gpu = to_fp64(static_cast<float *>(y_buf.mapped), n);
        std::vector<double> y_cpu(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            y_cpu[i] = (d32[i] == 0.0f) ? 0.0
                                          : static_cast<double>(b32[i] / d32[i]);
        }
        const bool ok = close(y_gpu, y_cpu, 0.0, 1e-6);
        std::printf("  jacobi %-20s n=%-7u  %s\n", name, n, ok ? "OK" : "FAIL");
        if (!ok) ++failures;

        destroy_buffer(vk.device, y_buf);
        destroy_buffer(vk.device, b_buf);
        destroy_buffer(vk.device, d_buf);
        destroy_buffer(vk.device, ubuf);
    };

    run("trivial", { 2.0, 3.0, 4.0, 5.0 }, { 4.0, 9.0, 16.0, 25.0 });
    run("zeros pass-through", { 1.0, 0.0, 1.0, 0.0 }, { 7.0, 99.0, 8.0, -3.0 });
    {
        std::mt19937_64 rng(0xfeedfaceULL);
        std::uniform_real_distribution<double> dd(0.5, 2.0);
        std::uniform_real_distribution<double> bd(-1.0, 1.0);
        std::vector<double> d(50000), b(50000);
        for (auto &v : d) v = dd(rng);
        for (auto &v : b) v = bd(rng);
        run("random 50k", d, b);
    }

    vkDestroyDescriptorPool(vk.device, pool, nullptr);
    vkDestroyPipeline(vk.device, pipe, nullptr);
    vkDestroyShaderModule(vk.device, sh, nullptr);
    vkDestroyPipelineLayout(vk.device, pl, nullptr);
    vkDestroyDescriptorSetLayout(vk.device, dsl, nullptr);
    return failures;
}

// ============================================================
// saxpby
// ============================================================

int test_saxpby(VulkanCompute &vk) {
    const std::vector<std::uint32_t> spv = load_spv("bin/saxpby.spv");
    VkShaderModule        sh   = make_shader(vk.device, spv);
    VkDescriptorSetLayout dsl  = make_dsl(vk.device, 3);
    VkPipelineLayout      pl   = make_layout(vk.device, dsl);
    VkPipeline            pipe = make_pipeline(vk.device, pl, sh);
    VkDescriptorPool      pool = make_pool(vk.device, 3);
    VkDescriptorSet       set  = alloc_set(vk.device, pool, dsl);

    int failures = 0;
    auto run = [&](const char *name, double alpha, double beta,
                    const std::vector<double> &x, const std::vector<double> &y) {
        const std::uint32_t n = static_cast<std::uint32_t>(x.size());
        struct UB { std::uint32_t n; float alpha; float beta; }
            ub{ n, static_cast<float>(alpha), static_cast<float>(beta) };

        Buffer ubuf   = make_buffer(vk, sizeof(UB), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        Buffer x_buf  = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer y_buf  = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer dst_buf = make_buffer(vk, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(ubuf.mapped, &ub, sizeof(UB));
        const auto x32 = to_fp32(x); std::memcpy(x_buf.mapped, x32.data(), n * sizeof(float));
        const auto y32 = to_fp32(y); std::memcpy(y_buf.mapped, y32.data(), n * sizeof(float));
        std::memset(dst_buf.mapped, 0, n * sizeof(float));

        bind_buffers(vk.device, set, ubuf, { x_buf, y_buf, dst_buf });
        run_compute_once(vk, pipe, pl, set, (n + 255) / 256);

        const auto y_gpu = to_fp64(static_cast<float *>(dst_buf.mapped), n);
        const float alpha_f = static_cast<float>(alpha);
        const float beta_f  = static_cast<float>(beta);
        std::vector<double> y_cpu(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const float v = std::fma(alpha_f, x32[i], beta_f * y32[i]);
            y_cpu[i] = static_cast<double>(v);
        }
        const bool ok = close(y_gpu, y_cpu, 0.0, 1e-6);
        std::printf("  saxpby %-19s n=%-7u alpha=%5.2g beta=%5.2g  %s\n",
                      name, n, alpha, beta, ok ? "OK" : "FAIL");
        if (!ok) ++failures;

        destroy_buffer(vk.device, dst_buf);
        destroy_buffer(vk.device, y_buf);
        destroy_buffer(vk.device, x_buf);
        destroy_buffer(vk.device, ubuf);
    };

    run("trivial", 2.0, 3.0, { 1.0, 2.0, 3.0, 4.0 }, { 10.0, 20.0, 30.0, 40.0 });
    run("alpha=1 beta=0", 1.0, 0.0, { 1.0, 2.0, 3.0 }, { 9.0, 9.0, 9.0 });
    {
        std::mt19937_64 rng(0xc0ffee01ULL);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        std::vector<double> x(50000), y(50000);
        for (auto &v : x) v = d(rng);
        for (auto &v : y) v = d(rng);
        run("random 50k", 0.3, -0.7, x, y);
    }

    vkDestroyDescriptorPool(vk.device, pool, nullptr);
    vkDestroyPipeline(vk.device, pipe, nullptr);
    vkDestroyShaderModule(vk.device, sh, nullptr);
    vkDestroyPipelineLayout(vk.device, pl, nullptr);
    vkDestroyDescriptorSetLayout(vk.device, dsl, nullptr);
    return failures;
}

} // namespace

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    VulkanCompute vk;
    vk.init();

    int failures = 0;
    failures += test_axpy(vk);
    failures += test_jacobi(vk);
    failures += test_saxpby(vk);

    vk.shutdown();
    std::printf("\n%s: %d failing case(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
