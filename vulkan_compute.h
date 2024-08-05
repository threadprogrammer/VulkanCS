#pragma once
// vulkan_compute.h

#ifndef VULKAN_COMPUTE_H
#define VULKAN_COMPUTE_H

#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

// Function to initialize Vulkan, load the compute shader, and run it
void VulkanComputeInitAndRun(std::string spvFile, uint32_t device = 1, float qPriority = 1.0f, uint32_t cBufferCount = 1);

// Function to read a binary file (e.g., SPIR-V shader file)
std::vector<char> readFile(const std::string& filename);

#endif // VULKAN_COMPUTE_H