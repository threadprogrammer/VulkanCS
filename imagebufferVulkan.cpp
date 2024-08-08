#define _CRT_SECURE_NO_WARNINGS

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Function to read shader code from file
std::vector<uint32_t> readShaderCode(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Failed to open file");

    std::ifstream::pos_type fileSize = file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

bool loadImage(const std::string& filename, int& width, int& height, int& channels, unsigned char*& data) {
    data = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha); // Force 4 channels (RGBA)
    if (!data) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
        return false;
    }
    return true;
}



// Initialize Vulkan instance, device, and other necessary components
void initVulkan(VkInstance& instance, VkDevice& device, VkQueue& computeQueue, VkCommandPool& commandPool, VkPipelineLayout& pipelineLayout, VkPipeline& computePipeline, VkDescriptorSetLayout& descriptorSetLayout) {
    // Create Vulkan instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Compute Shader";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    // Choose physical device
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    uint32_t deviceCount = 0;

    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    physicalDevice = physicalDevices[0];  // Simplification: choose the first device

    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

    // Create logical device
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0;  // Simplification: choose the first queue family
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(device, 0, 0, &computeQueue);

    // Create command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = 0;  // Simplification: choose the first queue family

    if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    // Create descriptor set layout
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.bindingCount = 1;
    layoutCreateInfo.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Load and create compute shader module
    auto shaderCode = readShaderCode("image_shader.spv");
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
    shaderModuleCreateInfo.pCode = shaderCode.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    vkDestroyShaderModule(device, shaderModule, nullptr);
}

// Create input and output images
void createImages(VkDevice device, VkImage& inputImage, VkImage& outputImage, VkDeviceMemory& inputImageMemory, VkDeviceMemory& outputImageMemory, VkImageView& inputImageView, VkImageView& outputImageView, VkFormat format, VkExtent2D extent) {
    // Create input image
    VkImageCreateInfo imageCreateInfo = {};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.extent.width = extent.width;
    imageCreateInfo.extent.height = extent.height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageCreateInfo, nullptr, &inputImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create input image");
    }

    if (vkCreateImage(device, &imageCreateInfo, nullptr, &outputImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create output image");
    }

    // Allocate memory for input image
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, inputImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    // Find suitable memory type (omitted for brevity; use a memory type index for VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    if (vkAllocateMemory(device, &allocInfo, nullptr, &inputImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate input image memory");
    }

    if (vkAllocateMemory(device, &allocInfo, nullptr, &outputImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate output image memory");
    }

    // Bind memory to images
    vkBindImageMemory(device, inputImage, inputImageMemory, 0);
    vkBindImageMemory(device, outputImage, outputImageMemory, 0);

    //int width, height, channels;
    //width = 864;
    //width = 409;
    //channels = 4;
    //unsigned char* imageData = nullptr;
    //if (!loadImage("input.png", width, height, channels, imageData)) {
    //    throw std::runtime_error("Failed to open image");
    //}
    //// Copy data to input buffer
    //void* data;
    //vkMapMemory(device, inputImageMemory, 0, sizeof(imageData), 0, &data);
    //memcpy(data, imageData.data(), sizeof(imageData));
    //vkUnmapMemory(device, inputImageMemory);

    // Create image views
    VkImageViewCreateInfo imageViewCreateInfo = {};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = inputImage;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &imageViewCreateInfo, nullptr, &inputImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create input image view");
    }

    imageViewCreateInfo.image = outputImage;

    if (vkCreateImageView(device, &imageViewCreateInfo, nullptr, &outputImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create output image view");
    }
}

void createDescriptorSets(VkDevice device, VkDescriptorPool& descriptorPool, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet& descriptorSet, VkImageView inputImageView, VkImageView outputImageView) {
    // Create descriptor pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 2;  // One for input and one for output

    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;
    poolCreateInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    // Update descriptor set
    VkDescriptorImageInfo inputImageInfo = {};
    inputImageInfo.imageView = inputImageView;
    inputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputImageInfo = {};
    outputImageInfo.imageView = outputImageView;
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet descriptorWrite[2] = {};

    descriptorWrite[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite[0].dstSet = descriptorSet;
    descriptorWrite[0].dstBinding = 0;
    descriptorWrite[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrite[0].descriptorCount = 1;
    descriptorWrite[0].pImageInfo = &inputImageInfo;

    descriptorWrite[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite[1].dstSet = descriptorSet;
    descriptorWrite[1].dstBinding = 1;
    descriptorWrite[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrite[1].descriptorCount = 1;
    descriptorWrite[1].pImageInfo = &outputImageInfo;

    vkUpdateDescriptorSets(device, 2, descriptorWrite, 0, nullptr);
}


// Record command buffer
void recordCommandBuffer(VkDevice device, VkCommandPool commandPool, VkPipeline computePipeline, VkPipelineLayout pipelineLayout, VkDescriptorSet descriptorSet, VkCommandBuffer& commandBuffer, VkExtent2D extent) {
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }

    // Begin command buffer recording
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer recording");
    }

    // Bind compute pipeline and descriptor set
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Dispatch compute shader
    vkCmdDispatch(commandBuffer, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    // End command buffer recording
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer recording");
    }
}

// Submit command buffer and synchronize
void submitCommandBuffer(VkDevice device, VkQueue computeQueue, VkCommandBuffer commandBuffer) {
    // Create a fence for synchronization
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence");
    }

    // Submit command buffer to queue
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }

    // Wait for the fence to signal
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // HERE
    /*vkMapMemory(device, outputBufferMemory, 0, sizeof(outputData), 0, &data);
    memcpy(outputData.data(), data, sizeof(outputData));
    vkUnmapMemory(device, outputBufferMemory);
    saveImage("output.png", width, height, outputData.data());

    stbi_image_free(imageData);*/


    vkDestroyFence(device, fence, nullptr);
}

void saveImage(const std::string& filename, int width, int height, const unsigned char* data) {
    stbi_write_png(filename.c_str(), width, height, 4, data, width * 4); // 4 bytes per pixel (RGBA)
}
// Main function
int main() {
    

    VkInstance instance;
    VkDevice device;
    VkQueue computeQueue;
    VkCommandPool commandPool;
    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline;
    VkDescriptorSetLayout descriptorSetLayout;

    // Initialize Vulkan components
    initVulkan(instance, device, computeQueue, commandPool, pipelineLayout, computePipeline, descriptorSetLayout);

    // Image parameters
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent2D extent = { 864, 409 };  // Example dimensions

    VkImage inputImage, outputImage;
    VkDeviceMemory inputImageMemory, outputImageMemory;
    VkImageView inputImageView, outputImageView;

    // Create input and output images
    createImages(device, inputImage, outputImage, inputImageMemory, outputImageMemory, inputImageView, outputImageView, format, extent);

    VkDescriptorSet descriptorSet;
    VkDescriptorPool descriptorPool;

    // Create descriptor sets
    createDescriptorSets(device, descriptorPool, descriptorSetLayout, descriptorSet, inputImageView, outputImageView);

    VkCommandBuffer commandBuffer;

    // Record command buffer
    recordCommandBuffer(device, commandPool, computePipeline, pipelineLayout, descriptorSet, commandBuffer, extent);

    // Submit command buffer and synchronize
    submitCommandBuffer(device, computeQueue, commandBuffer);
    
    
    // Clean up Vulkan resources
    vkDestroyImageView(device, outputImageView, nullptr);
    vkDestroyImageView(device, inputImageView, nullptr);
    vkDestroyImage(device, outputImage, nullptr);
    vkDestroyImage(device, inputImage, nullptr);
    vkFreeMemory(device, outputImageMemory, nullptr);
    vkFreeMemory(device, inputImageMemory, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}