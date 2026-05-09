// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Standalone Vulkan-compute test for the df32 dot-product reduction.
// The kernel reads two fp32 vectors and writes the result as a df32
// pair (hi, lo); the host folds (double)hi + (double)lo back to fp64
// for comparison against a reference fp64 dot.
//
// The whole point of df32 is to keep fp64-quality dot scalars under
// the GPU CG iteration loop without paying for fp64 throughput. Each
// case is constructed to be a precision trap that a naive fp32 sum
// would fail; we assert that the df32 output matches the fp64
// reference within ~1e-12 *relative*, well below the n·ε ≈ 6e-3
// floor of a 100k-element fp32 sum.
//
// Build + run with `make -C tests gpu`. The Makefile uses MoltenVK on
// macOS; on Steam Deck / Linux the Vulkan loader is in the default
// search path and the same SPIR-V dispatches unchanged.

#include "gpu_compute_helpers.h"

#include <cmath>
#include <random>
#include <vector>

using namespace curvenet_gpu_test;

namespace {

struct DotPipeline {
    VkDescriptorSetLayout dsl       = VK_NULL_HANDLE;
    VkPipelineLayout      layout    = VK_NULL_HANDLE;
    VkShaderModule        shader    = VK_NULL_HANDLE;
    VkPipeline            pipeline  = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set  = VK_NULL_HANDLE;

    void init(const VulkanCompute &vk, const std::vector<std::uint32_t> &spv) {
        // 1 uniform + 3 storage (a, b, out).
        VkDescriptorSetLayoutBinding bindings[4]{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        for (int i = 1; i < 4; ++i) {
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 4;
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
        pool_sizes[1].descriptorCount = 3;
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

    void bind_buffers(VkDevice device, const Buffer &params,
                       const Buffer &a, const Buffer &b, const Buffer &dst) {
        VkDescriptorBufferInfo infos[4]{};
        infos[0] = { params.handle, 0, VK_WHOLE_SIZE };
        infos[1] = { a.handle,      0, VK_WHOLE_SIZE };
        infos[2] = { b.handle,      0, VK_WHOLE_SIZE };
        infos[3] = { dst.handle,    0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[4]{};
        for (int i = 0; i < 4; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = desc_set;
            writes[i].dstBinding      = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = (i == 0)
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &infos[i];
        }
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }

    void shutdown(VkDevice device) {
        if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, nullptr);
        if (pipeline)  vkDestroyPipeline(device, pipeline, nullptr);
        if (shader)    vkDestroyShaderModule(device, shader, nullptr);
        if (layout)    vkDestroyPipelineLayout(device, layout, nullptr);
        if (dsl)       vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    }
};

// Run dot_reduce on the GPU and return (hi + lo) folded to fp64.
double gpu_dot_df(VulkanCompute &vk, DotPipeline &pipe,
                    const std::vector<float> &a, const std::vector<float> &b) {
    const std::uint32_t n = static_cast<std::uint32_t>(a.size());

    Buffer params  = make_buffer(vk, sizeof(std::uint32_t),
                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    Buffer a_buf   = make_buffer(vk, n * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer b_buf   = make_buffer(vk, n * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer dst_buf = make_buffer(vk, 2 * sizeof(float),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::memcpy(params.mapped, &n,        sizeof(std::uint32_t));
    std::memcpy(a_buf.mapped,  a.data(),  n * sizeof(float));
    std::memcpy(b_buf.mapped,  b.data(),  n * sizeof(float));
    std::memset(dst_buf.mapped, 0, 2 * sizeof(float));

    pipe.bind_buffers(vk.device, params, a_buf, b_buf, dst_buf);
    // Single workgroup; each thread strides over the whole input.
    run_compute_once(vk, pipe.pipeline, pipe.layout, pipe.desc_set, 1);

    float pair[2];
    std::memcpy(pair, dst_buf.mapped, 2 * sizeof(float));

    destroy_buffer(vk.device, dst_buf);
    destroy_buffer(vk.device, b_buf);
    destroy_buffer(vk.device, a_buf);
    destroy_buffer(vk.device, params);

    return static_cast<double>(pair[0]) + static_cast<double>(pair[1]);
}

// Reference fp64 dot — the gold standard we compare against.
double cpu_dot_fp64(const std::vector<float> &a, const std::vector<float> &b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return s;
}

// Naive fp32 dot — included to demonstrate the precision floor df32
// is meant to break through.
float cpu_dot_fp32(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

struct Case {
    const char            *name;
    std::vector<float>     a;
    std::vector<float>     b;
    double                 expected;
};

bool close_relative(double a, double b, double tol) {
    const double m = (std::fabs(a) + std::fabs(b)) * 0.5 + 1e-30;
    return std::fabs(a - b) <= tol * m + tol;
}

int run_case(VulkanCompute &vk, DotPipeline &pipe, const Case &c) {
    const double df  = gpu_dot_df  (vk, pipe, c.a, c.b);
    const float  f32 = cpu_dot_fp32(    c.a, c.b);
    const double f64 = cpu_dot_fp64(    c.a, c.b);

    // The honest reference for "what should the GPU produce" is the
    // fp64 dot of the *fp32-rounded* inputs. df32 should match that
    // within ~1e-12 relative; fp32 only manages ~n·ε ≈ 1e-3 at 100k.
    const double err_df  = std::fabs(df  - f64);
    const double err_f32 = std::fabs(static_cast<double>(f32) - f64);
    const double mag     = std::fabs(f64) + 1e-30;
    const double rel_df  = err_df  / mag;
    const double rel_f32 = err_f32 / mag;

    const bool ok = (rel_df <= 1e-10);

    std::printf("  %-30s  fp32 rel=%9.2e  df32 rel=%9.2e  ratio=%6.0fx  %s\n",
                  c.name, rel_f32, rel_df,
                  rel_df > 0 ? rel_f32 / rel_df : 0.0,
                  ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// Long run of ones — sums to exactly N if precision holds.
Case ones(std::size_t n) {
    Case c;
    c.name = "ones(N)";
    c.a.assign(n, 1.0f);
    c.b.assign(n, 1.0f);
    c.expected = static_cast<double>(n);
    return c;
}

// Catastrophic-cancellation classic: huge positive + huge negative +
// many small positives. The tail should survive but in fp32 it's lost
// when the two huge terms cancel late in the sum.
Case kahan_classic() {
    Case c;
    c.name = "kahan classic (1e7,-1e7,1*1024)";
    c.a.reserve(1026);
    c.b.reserve(1026);
    c.a.push_back(1.0e7f);
    c.b.push_back(1.0f);
    c.a.push_back(-1.0e7f);
    c.b.push_back(1.0f);
    for (int i = 0; i < 1024; ++i) {
        c.a.push_back(1.0f);
        c.b.push_back(1.0f);
    }
    c.expected = 1024.0;
    return c;
}

// Two sinusoids — randomish signs and magnitudes; checks that there's
// no systematic bias.
Case sinusoid(std::size_t n) {
    Case c;
    c.name = "sinusoid(N)";
    c.a.resize(n);
    c.b.resize(n);
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double t  = (static_cast<double>(i) + 1.0) * 0.001;
        const double ai = std::sin(t);
        const double bi = std::cos(t * 1.1);
        c.a[i] = static_cast<float>(ai);
        c.b[i] = static_cast<float>(bi);
        s += ai * bi;
    }
    c.expected = s;
    return c;
}

// Random fp32 vectors of length n, uniform in [-1, 1].
Case random_uniform(std::size_t n, std::uint64_t seed) {
    Case c;
    c.name = "random uniform(N)";
    c.a.resize(n);
    c.b.resize(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        c.a[i] = dist(rng);
        c.b[i] = dist(rng);
        s += static_cast<double>(c.a[i]) * static_cast<double>(c.b[i]);
    }
    c.expected = s;
    return c;
}

} // namespace

int main(int argc, char **argv) {
    const char *spv_path = (argc > 1) ? argv[1] : "bin/dot_reduce.spv";
    std::printf("Loading SPIR-V from %s\n", spv_path);
    const std::vector<std::uint32_t> spv = load_spv(spv_path);

    VulkanCompute vk;
    vk.init();

    DotPipeline pipe;
    pipe.init(vk, spv);

    int failures = 0;

    failures += run_case(vk, pipe, ones(1024));
    failures += run_case(vk, pipe, ones(100000));
    failures += run_case(vk, pipe, kahan_classic());
    failures += run_case(vk, pipe, sinusoid(50000));
    failures += run_case(vk, pipe, random_uniform(50000, 0x12345678abcdULL));
    failures += run_case(vk, pipe, random_uniform(100000, 0xdeadbeefcafeULL));

    pipe.shutdown(vk.device);
    vk.shutdown();

    std::printf("\n%s: %d failing case(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
