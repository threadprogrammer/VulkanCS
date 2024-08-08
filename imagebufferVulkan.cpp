#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <vulkan/vulkan.h>
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

const char* shaderCode = R"(
#version 450

layout(binding = 0) uniform sampler2D inputImage;
layout(binding = 1) writeonly uniform image2D outputImage;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec4 color = texelFetch(inputImage, pos, 0);
    imageStore(outputImage, pos, color);
}
)";

// Helper function to load shader from string
VkShaderModule createShaderModule(VkDevice device, const char* code, size_t codeSize) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    std::vector<uint32_t> codeBuffer((uint32_t*)code, (uint32_t*)(code + codeSize));
    createInfo.pCode = codeBuffer.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module");
    }
    return shaderModule;
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}


// Main function
int main() {
    VkImage inputImage;
    VkImageView inputImageView;
    VkSampler inputSampler;
    // Initialize Vulkan instance
    VkInstance instance;
    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    vkCreateInstance(&instanceCreateInfo, nullptr, &instance);

    // Select a physical device and create a logical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    VkPhysicalDevice physicalDevice = physicalDevices[0];
    VkDevice device;
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);

    // Load image with stb_image
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load("input.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    

    VkBuffer inputBuffer, outputBuffer;
    VkDeviceMemory inputMemory, outputMemory;
    VkDeviceSize bufferSize = imageSize;

    // Create the input buffer
    VkBufferCreateInfo bufferInfoC{};
    bufferInfoC.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfoC.size = bufferSize;
    bufferInfoC.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfoC.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfoC, nullptr, &inputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create input buffer!");
    }

    // Get memory requirements for the input buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, inputBuffer, &memRequirements);

    // Allocate memory for the input buffer
    VkMemoryAllocateInfo allocInfoM{};
    allocInfoM.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoM.allocationSize = memRequirements.size;
    allocInfoM.memoryTypeIndex = 0;

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            allocInfoM.memoryTypeIndex = i;
            break;
        }
    }

    if (vkAllocateMemory(device, &allocInfoM, nullptr, &inputMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate input buffer memory!");
    }

    // Bind memory to the input buffer
    vkBindBufferMemory(device, inputBuffer, inputMemory, 0);

    // Create the output buffer
    bufferInfoC.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (vkCreateBuffer(device, &bufferInfoC, nullptr, &outputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create output buffer!");
    }

    // Get memory requirements for the output buffer
    vkGetBufferMemoryRequirements(device, outputBuffer, &memRequirements);

    // Allocate memory for the output buffer
    allocInfoM.allocationSize = memRequirements.size;
    allocInfoM.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            allocInfoM.memoryTypeIndex = i;
            break;
        }
    }

    if (vkAllocateMemory(device, &allocInfoM, nullptr, &outputMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate output buffer memory!");
    }

    // Bind memory to the output buffer
    vkBindBufferMemory(device, outputBuffer, outputMemory, 0);

    // Load shader module
    VkShaderModule shaderModule = createShaderModule(device, shaderCode, strlen(shaderCode));
    // Create input image, input image view, and sampler

// Input Image (assuming the image data has already been loaded using stb_image)

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = texWidth;
    imageInfo.extent.height = texHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &inputImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create input image!");
    }

    vkGetImageMemoryRequirements(device, inputImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &inputMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate input image memory!");
    }

    vkBindImageMemory(device, inputImage, inputMemory, 0);

    // Transition input image layout to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and copy data
    // (This would typically involve command buffer operations not shown here for brevity)

    // Create Image View for Input Image
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = inputImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &inputImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create input image view!");
    }

    // Create a sampler for the input image
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &inputSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create input sampler!");
    }

    // Create output image and output image view

    VkImage outputImage;
    VkImageView outputImageView;

    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imageInfo, nullptr, &outputImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create output image!");
    }

    vkGetImageMemoryRequirements(device, outputImage, &memRequirements);

    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &outputMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate output image memory!");
    }

    vkBindImageMemory(device, outputImage, outputMemory, 0);

    // Transition output image layout to VK_IMAGE_LAYOUT_GENERAL
    // (This would typically involve command buffer operations not shown here for brevity)

    // Create Image View for Output Image
    viewInfo.image = outputImage;
    if (vkCreateImageView(device, &viewInfo, nullptr, &outputImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create output image view!");
    }


    // Create descriptor sets, pipeline, etc.
    // Allocate and bind memory for buffers
    // Create descriptor sets, pipeline, etc.

// Descriptor Set Layout
    VkDescriptorSetLayoutBinding inputBinding{};
    inputBinding.binding = 0;
    inputBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    inputBinding.descriptorCount = 1;
    inputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    inputBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding outputBinding{};
    outputBinding.binding = 1;
    outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputBinding.descriptorCount = 1;
    outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    outputBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding bindings[] = { inputBinding, outputBinding };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    // Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    // Shader Stage
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    // Compute Pipeline
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline computePipeline;
    auto res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
    std::cout << res << std::endl;
    

    // Descriptor Pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolSize poolSizeOutput{};
    poolSizeOutput.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizeOutput.descriptorCount = 1;

    VkDescriptorPoolSize poolSizes[] = { poolSize, poolSizeOutput };

    VkDescriptorPoolCreateInfo poolInfoD{};
    poolInfoD.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfoD.poolSizeCount = 2;
    poolInfoD.pPoolSizes = poolSizes;
    poolInfoD.maxSets = 1;

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfoD, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    // Descriptor Sets
    VkDescriptorSetAllocateInfo allocInfoD{};
    allocInfoD.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfoD.descriptorPool = descriptorPool;
    allocInfoD.descriptorSetCount = 1;
    allocInfoD.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(device, &allocInfoD, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor set!");
    }

    // Update descriptor sets
    VkDescriptorImageInfo inputImageInfo{};
    inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    inputImageInfo.imageView = inputImageView;
    inputImageInfo.sampler = inputSampler;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &inputImageInfo;

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputImageInfo.imageView = outputImageView;

    VkWriteDescriptorSet descriptorWriteOutput{};
    descriptorWriteOutput.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWriteOutput.dstSet = descriptorSet;
    descriptorWriteOutput.dstBinding = 1;
    descriptorWriteOutput.dstArrayElement = 0;
    descriptorWriteOutput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWriteOutput.descriptorCount = 1;
    descriptorWriteOutput.pImageInfo = &outputImageInfo;

    VkWriteDescriptorSet descriptorWrites[] = { descriptorWrite, descriptorWriteOutput };
    vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);


    // Allocate and bind memory for buffers

    // Input Buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &inputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create input buffer!");
    }

    vkGetBufferMemoryRequirements(device, inputBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfoMem{};
    allocInfoMem.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoMem.allocationSize = memRequirements.size;
    allocInfoMem.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);

    if (vkAllocateMemory(device, &allocInfoMem, nullptr, &inputMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate input buffer memory!");
    }

    vkBindBufferMemory(device, inputBuffer, inputMemory, 0);

    // Output Buffer
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &outputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create output buffer!");
    }

    vkGetBufferMemoryRequirements(device, outputBuffer, &memRequirements);

    allocInfoMem.allocationSize = memRequirements.size;
    allocInfoMem.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);

    if (vkAllocateMemory(device, &allocInfoMem, nullptr, &outputMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate output buffer memory!");
    }

    vkBindBufferMemory(device, outputBuffer, outputMemory, 0);


    // Copy the loaded image to Vulkan buffer (inputBuffer)
    void* data;
    vkMapMemory(device, inputMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, inputMemory);

    // Run the compute shader pipeline
    // Setup code for dispatching the compute shader...
    // Run the compute shader pipeline
// Setup code for dispatching the compute shader...

// Create a command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;  // Assuming the queue family index for the compute queue is 0
    poolInfo.flags = 0;

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    // Allocate a command buffer
    VkCommandBufferAllocateInfo allocInfoC{};
    allocInfoC.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfoC.commandPool = commandPool;
    allocInfoC.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfoC.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(device, &allocInfoC, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffer!");
    }

    // Begin recording commands in the command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    // Bind the compute pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

    // Bind the descriptor set
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Dispatch the compute shader
    uint32_t groupCountX = (texWidth + 7) / 8;  // Assuming a workgroup size of 8x8
    uint32_t groupCountY = (texHeight + 7) / 8;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    // End recording commands
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    // Submit the command buffer to the queue
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkQueue computeQueue;
    vkGetDeviceQueue(device, 0, 0, &computeQueue);  // Assuming queue family index 0 and queue index 0

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit compute command buffer!");
    }

    // Wait for the queue to finish
    vkQueueWaitIdle(computeQueue);

    // Cleanup command buffer and pool
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, nullptr);


    // Copy the output buffer to host-visible memory and save the image with stb_image_write
    vkMapMemory(device, outputMemory, 0, imageSize, 0, &data);
    stbi_write_png("output.png", texWidth, texHeight, STBI_rgb_alpha, data, texWidth * 4);
    vkUnmapMemory(device, outputMemory);

    // Cleanup Vulkan resources
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyBuffer(device, inputBuffer, nullptr);
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, inputMemory, nullptr);
    vkFreeMemory(device, outputMemory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    // Free the loaded image
    stbi_image_free(pixels);

    return 0;
}

