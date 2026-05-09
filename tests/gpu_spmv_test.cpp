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

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace sp = curvenet::sparse;

namespace {

#define VK_OR_DIE(expr)                                                          \
    do {                                                                          \
        VkResult _vk_r = (expr);                                                  \
        if (_vk_r != VK_SUCCESS) {                                                \
            std::fprintf(stderr, "vulkan call failed: %s -> %d\n", #expr, _vk_r); \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

std::vector<std::uint32_t> load_spv(const char *path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    const std::streamsize bytes = f.tellg();
    if (bytes <= 0 || (bytes % 4) != 0) {
        std::fprintf(stderr, "bad SPIR-V size %lld\n", static_cast<long long>(bytes));
        std::exit(1);
    }
    std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(code.data()), bytes);
    return code;
}

struct VulkanCompute {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physical       = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          queue          = VK_NULL_HANDLE;
    std::uint32_t    queue_family   = 0;
    VkCommandPool    cmd_pool       = VK_NULL_HANDLE;

    void init() {
        VkApplicationInfo app{};
        app.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "curvenet-gpu-spmv-test";
        app.apiVersion       = VK_API_VERSION_1_2;

        // MoltenVK on macOS reports VK_KHR_portability_enumeration; the
        // loader requires us to opt in or instance creation fails.
        const char *instance_exts[] = {
            "VK_KHR_portability_enumeration"
        };
        VkInstanceCreateInfo ici{};
        ici.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.flags                   = 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        ici.pApplicationInfo        = &app;
        ici.enabledExtensionCount   = 1;
        ici.ppEnabledExtensionNames = instance_exts;
        VK_OR_DIE(vkCreateInstance(&ici, nullptr, &instance));

        std::uint32_t pd_count = 0;
        VK_OR_DIE(vkEnumeratePhysicalDevices(instance, &pd_count, nullptr));
        if (pd_count == 0) {
            std::fprintf(stderr, "no Vulkan physical devices\n");
            std::exit(1);
        }
        std::vector<VkPhysicalDevice> pds(pd_count);
        VK_OR_DIE(vkEnumeratePhysicalDevices(instance, &pd_count, pds.data()));
        physical = pds[0];

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical, &props);
        std::printf("GPU: %s (Vulkan %u.%u.%u)\n", props.deviceName,
                     VK_VERSION_MAJOR(props.apiVersion),
                     VK_VERSION_MINOR(props.apiVersion),
                     VK_VERSION_PATCH(props.apiVersion));

        std::uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &qf_count, qfs.data());
        queue_family = qf_count;
        for (std::uint32_t i = 0; i < qf_count; ++i) {
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == qf_count) {
            std::fprintf(stderr, "no compute queue family\n");
            std::exit(1);
        }

        const float qprio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queue_family;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &qprio;

        const char *device_exts[] = { "VK_KHR_portability_subset" };
        VkDeviceCreateInfo dci{};
        dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount    = 1;
        dci.pQueueCreateInfos       = &qci;
        // portability_subset is only required if the physical device
        // advertises it; MoltenVK does, native Vulkan drivers don't.
        std::uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, exts.data());
        bool need_portability = false;
        for (const auto &e : exts) {
            if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) {
                need_portability = true;
                break;
            }
        }
        if (need_portability) {
            dci.enabledExtensionCount   = 1;
            dci.ppEnabledExtensionNames = device_exts;
        }
        VK_OR_DIE(vkCreateDevice(physical, &dci, nullptr, &device));
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        VkCommandPoolCreateInfo cpi{};
        cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = queue_family;
        cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_OR_DIE(vkCreateCommandPool(device, &cpi, nullptr, &cmd_pool));
    }

    void shutdown() {
        if (cmd_pool) vkDestroyCommandPool(device, cmd_pool, nullptr);
        if (device)   vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

struct Buffer {
    VkBuffer       handle = VK_NULL_HANDLE;
    VkDeviceMemory mem    = VK_NULL_HANDLE;
    VkDeviceSize   bytes  = 0;
    void          *mapped = nullptr;
};

std::uint32_t pick_memory_type(VkPhysicalDevice pd,
                                  std::uint32_t type_bits,
                                  VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    std::fprintf(stderr, "no compatible memory type\n");
    std::exit(1);
}

Buffer make_buffer(const VulkanCompute &vk, VkDeviceSize bytes,
                    VkBufferUsageFlags usage) {
    Buffer b;
    b.bytes = bytes;
    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = bytes;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_OR_DIE(vkCreateBuffer(vk.device, &bci, nullptr, &b.handle));

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(vk.device, b.handle, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = pick_memory_type(vk.physical, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_OR_DIE(vkAllocateMemory(vk.device, &mai, nullptr, &b.mem));
    VK_OR_DIE(vkBindBufferMemory(vk.device, b.handle, b.mem, 0));
    VK_OR_DIE(vkMapMemory(vk.device, b.mem, 0, bytes, 0, &b.mapped));
    return b;
}

void destroy_buffer(VkDevice device, Buffer &b) {
    if (b.mapped)  vkUnmapMemory(device, b.mem);
    if (b.handle)  vkDestroyBuffer(device, b.handle, nullptr);
    if (b.mem)     vkFreeMemory(device, b.mem, nullptr);
    b = Buffer{};
}

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
        cpci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module       = shader;
        cpci.stage.pName        = "main";
        cpci.layout             = layout;
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

// Run spmv on the GPU and return y as a host-side vector<double>.
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

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool        = vk.cmd_pool;
    cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_OR_DIE(vkAllocateCommandBuffers(vk.device, &cbi, &cmd));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_OR_DIE(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1,
                              &pipe.desc_set, 0, nullptr);
    const std::uint32_t groups = (rows + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);
    VK_OR_DIE(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_OR_DIE(vkCreateFence(vk.device, &fci, nullptr, &fence));

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    VK_OR_DIE(vkQueueSubmit(vk.queue, 1, &si, fence));
    VK_OR_DIE(vkWaitForFences(vk.device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(vk.device, fence, nullptr);
    vkFreeCommandBuffers(vk.device, vk.cmd_pool, 1, &cmd);

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

bool vec_close(const std::vector<double> &a, const std::vector<double> &b,
                double abs_tol, double rel_tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        const double m = (a[i] < 0 ? -a[i] : a[i]) + (b[i] < 0 ? -b[i] : b[i]);
        const double d_abs = (d < 0 ? -d : d);
        if (d_abs > abs_tol + rel_tol * m) return false;
    }
    return true;
}

// 4×4 1D Laplacian (same instance as the SparseLinAlg.lean proof).
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

// Diagonal `diag(2, 3, 4, 5, 6)`.
sp::SparseMatrixCSR diag5() {
    sp::SparseMatrixCSR A;
    A.rows = 5;
    A.cols = 5;
    A.row_ptr = { 0, 1, 2, 3, 4, 5 };
    A.col_idx = { 0, 1, 2, 3, 4 };
    A.values  = { 2.0, 3.0, 4.0, 5.0, 6.0 };
    return A;
}

// 7-point-stencil-style 25×25 matrix from a 5×5 grid 2D Laplacian. Keeps
// the test interesting (non-trivial sparsity, more than one workgroup).
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
        // Sort by column for stable row layout.
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
    // fp32 GPU vs fp64 CPU: ~1e-5 relative is the right tolerance.
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

    // Case 1: 4×4 1D Laplacian × (1, 2, 3, 4) — same as a Lean check.
    failures += run_case(vk, pipe, "laplacian4 * (1,2,3,4)",
                          laplacian4(), { 1.0, 2.0, 3.0, 4.0 });

    // Case 2: diag(2,3,4,5,6) × (1,1,1,1,1).
    failures += run_case(vk, pipe, "diag5 * ones",
                          diag5(), { 1.0, 1.0, 1.0, 1.0, 1.0 });

    // Case 3: 5×5 2D grid Laplacian × indicator vector.
    {
        const auto A = grid_laplacian(5);
        std::vector<double> x(25, 0.0);
        x[12] = 1.0; // centre node
        failures += run_case(vk, pipe, "grid_laplacian(5) * e12", A, x);
    }

    // Case 4: 7×7 2D grid Laplacian (49 rows, exercises >1 workgroup).
    {
        const auto A = grid_laplacian(7);
        std::vector<double> x(49);
        for (int i = 0; i < 49; ++i) x[i] = 0.1 * i;
        failures += run_case(vk, pipe, "grid_laplacian(7) * ramp", A, x);
    }

    // Case 5: 10×10 2D grid Laplacian (100 rows, exactly fills 2 workgroups).
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
