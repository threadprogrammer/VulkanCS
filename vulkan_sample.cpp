#include "vulkan_compute.h"


std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}


void VulkanComputeInitAndRun(std::string spvFile, uint32_t gDevice, float qPriority, uint32_t cBufferCount) {

    // Initialize Vulkan
    VkInstance instance;
    VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    vkCreateInstance(&createInfo, nullptr, &instance);

    // Pick the first available physical device
    VkPhysicalDevice physicalDevice;
    vkEnumeratePhysicalDevices(instance, &gDevice, &physicalDevice);

    // Create a logical device with a compute queue
    VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueCreateInfo.queueFamilyIndex = 0; // Assume first family supports compute
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &qPriority;

    VkDevice device;
    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);

    VkQueue computeQueue;
    vkGetDeviceQueue(device, 0, 0, &computeQueue);

    // Read SPIR-V shader
    auto shaderCode = readFile(spvFile);

    // Create shader module
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleCreateInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    shaderModuleCreateInfo.codeSize = shaderCode.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
    vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = "main";

    VkPipelineLayout pipelineLayout;
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

    VkPipeline computePipeline;
    VkComputePipelineCreateInfo pipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &computePipeline);

    // Create command pool and buffer
    VkCommandPool commandPool;
    VkCommandPoolCreateInfo poolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolCreateInfo.queueFamilyIndex = 0; // Assume first family supports compute
    vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool);

    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // Record command buffer
    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdDispatch(commandBuffer, 1, 1, 1);
    vkEndCommandBuffer(commandBuffer);

    // Submit command buffer and wait
    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = cBufferCount;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

int main() {

    VulkanComputeInitAndRun("compute_shader.spv", 1, 1.0f, 1);
    // VulkanComputeInitAndRun("compute_shader.spv");

    return 0;
}