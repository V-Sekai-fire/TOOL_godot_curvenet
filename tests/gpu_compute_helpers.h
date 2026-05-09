// Copyright (c) 2026 K. S. Ernest (iFire) Lee.
// SPDX-License-Identifier: MIT
//
// Minimal Vulkan-compute scaffolding shared between the standalone
// shader tests (gpu_spmv_test.cpp, gpu_dot_test.cpp). Header-only;
// each test is its own translation unit and includes this once.
//
// What's here:
//   * VulkanCompute   instance / device / queue / command pool
//   * Buffer          host-mapped storage / uniform buffer wrapper
//   * load_spv        read a .spv file into a vector<uint32_t>
//   * vec_close       fp tolerance comparator
//   * VK_OR_DIE       error macro that prints + exit(1) on non-VK_SUCCESS
//
// Not here: pipelines, descriptor layouts. Those are kernel-specific
// and live in the test's own translation unit.

#ifndef CURVENET_GPU_COMPUTE_HELPERS_H
#define CURVENET_GPU_COMPUTE_HELPERS_H

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace curvenet_gpu_test {

#define VK_OR_DIE(expr)                                                          \
    do {                                                                          \
        VkResult _vk_r = (expr);                                                  \
        if (_vk_r != VK_SUCCESS) {                                                \
            std::fprintf(stderr, "vulkan call failed: %s -> %d\n", #expr, _vk_r); \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

inline std::vector<std::uint32_t> load_spv(const char *path) {
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
    VkInstance       instance     = VK_NULL_HANDLE;
    VkPhysicalDevice physical     = VK_NULL_HANDLE;
    VkDevice         device       = VK_NULL_HANDLE;
    VkQueue          queue        = VK_NULL_HANDLE;
    std::uint32_t    queue_family = 0;
    VkCommandPool    cmd_pool     = VK_NULL_HANDLE;

    void init() {
        VkApplicationInfo app{};
        app.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "curvenet-gpu-test";
        app.apiVersion       = VK_API_VERSION_1_2;

        const char *instance_exts[] = { "VK_KHR_portability_enumeration" };
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

        VkDeviceCreateInfo dci{};
        dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &qci;

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
        const char *device_exts[] = { "VK_KHR_portability_subset" };
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

inline std::uint32_t pick_memory_type(VkPhysicalDevice pd,
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

inline Buffer make_buffer(const VulkanCompute &vk, VkDeviceSize bytes,
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

inline void destroy_buffer(VkDevice device, Buffer &b) {
    if (b.mapped) vkUnmapMemory(device, b.mem);
    if (b.handle) vkDestroyBuffer(device, b.handle, nullptr);
    if (b.mem)    vkFreeMemory(device, b.mem, nullptr);
    b = Buffer{};
}

inline bool vec_close(const std::vector<double> &a, const std::vector<double> &b,
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

// Shorthand pipeline + descriptor set wiring for the simple "1 uniform
// + N storage buffers" kernels used throughout the GPU CG path.
struct ComputeKernel {
    VkDescriptorSetLayout dsl       = VK_NULL_HANDLE;
    VkPipelineLayout      layout    = VK_NULL_HANDLE;
    VkShaderModule        shader    = VK_NULL_HANDLE;
    VkPipeline            pipeline  = VK_NULL_HANDLE;
    std::uint32_t         storage_count = 0;

    void init(VkDevice device, const std::vector<std::uint32_t> &spv,
               std::uint32_t n_storage) {
        storage_count = n_storage;
        std::vector<VkDescriptorSetLayoutBinding> bindings(1 + n_storage);
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        for (std::uint32_t i = 0; i < n_storage; ++i) {
            bindings[1 + i].binding         = 1 + i;
            bindings[1 + i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1 + i].descriptorCount = 1;
            bindings[1 + i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<std::uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        VK_OR_DIE(vkCreateDescriptorSetLayout(device, &dlci, nullptr, &dsl));

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &dsl;
        VK_OR_DIE(vkCreatePipelineLayout(device, &plci, nullptr, &layout));

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(std::uint32_t);
        smci.pCode    = spv.data();
        VK_OR_DIE(vkCreateShaderModule(device, &smci, nullptr, &shader));

        VkComputePipelineCreateInfo cpci{};
        cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader;
        cpci.stage.pName  = "main";
        cpci.layout       = layout;
        VK_OR_DIE(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci,
                                            nullptr, &pipeline));
    }

    void shutdown(VkDevice device) {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader)   vkDestroyShaderModule(device, shader, nullptr);
        if (layout)   vkDestroyPipelineLayout(device, layout, nullptr);
        if (dsl)      vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        *this = ComputeKernel{};
    }
};

// Allocate a descriptor set and bind {1 uniform} + {N storage} buffers
// to it. The buffers must outlive the descriptor set.
inline VkDescriptorSet alloc_and_bind(VkDevice device,
                                        VkDescriptorPool pool,
                                        VkDescriptorSetLayout dsl,
                                        const Buffer &uniform,
                                        const std::vector<Buffer> &storage) {
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_OR_DIE(vkAllocateDescriptorSets(device, &dsai, &set));

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
    return set;
}

// Record N compute dispatches into a single command buffer with
// SHADER_WRITE -> SHADER_READ pipeline barriers between them, then
// submit once. The CB and fence are allocated once in `init()` and
// recycled across calls — important on MoltenVK where each
// vkAllocateCommandBuffers + vkCreateFence pair adds ~150 us.
struct CommandBatch {
    VkCommandBuffer cmd   = VK_NULL_HANDLE;
    VkFence         fence = VK_NULL_HANDLE;
    const VulkanCompute *vk = nullptr;

    void init(const VulkanCompute &vk_in) {
        vk = &vk_in;
        VkCommandBufferAllocateInfo cbi{};
        cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool        = vk->cmd_pool;
        cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        VK_OR_DIE(vkAllocateCommandBuffers(vk->device, &cbi, &cmd));

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_OR_DIE(vkCreateFence(vk->device, &fci, nullptr, &fence));
    }

    void shutdown() {
        if (!vk) return;
        if (fence) vkDestroyFence(vk->device, fence, nullptr);
        if (cmd)   vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &cmd);
        vk = nullptr;
        cmd = VK_NULL_HANDLE;
        fence = VK_NULL_HANDLE;
    }

    // Reset the CB to recording state so the same batch object can
    // be re-recorded. No ONE_TIME flag — we want to allow re-submitting
    // the recorded CB multiple times via submit_wait().
    void begin() {
        VK_OR_DIE(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = 0;
        VK_OR_DIE(vkBeginCommandBuffer(cmd, &bi));
    }

    void end() {
        VK_OR_DIE(vkEndCommandBuffer(cmd));
    }

    // Submit the previously-recorded CB and wait for completion.
    // Resets the fence so the same batch can be submitted again.
    void submit_wait() {
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        VK_OR_DIE(vkQueueSubmit(vk->queue, 1, &si, fence));
        VK_OR_DIE(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_OR_DIE(vkResetFences(vk->device, 1, &fence));
    }

    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                    VkDescriptorSet desc_set, std::uint32_t groups_x) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                                  &desc_set, 0, nullptr);
        vkCmdDispatch(cmd, groups_x, 1, 1);
    }

    // Heavy-handed compute->compute barrier; sufficient for our flow
    // since iter ordering is sequential and we don't need overlap
    // between dispatches that hit different buffers.
    void barrier() {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void end_and_submit_wait() {
        end();
        submit_wait();
    }
};

inline void run_compute_once(const VulkanCompute &vk, VkPipeline pipeline,
                                VkPipelineLayout layout, VkDescriptorSet desc_set,
                                std::uint32_t groups_x) {
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                              &desc_set, 0, nullptr);
    vkCmdDispatch(cmd, groups_x, 1, 1);
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
}

} // namespace curvenet_gpu_test

#endif
