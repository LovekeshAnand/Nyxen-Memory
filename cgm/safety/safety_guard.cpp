#include <windows.h>
#include <iostream>
#include <cmath>

struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

extern "C" {
    // Queries Windows system memory.
    // Returns 0 on success, non-zero on failure.
    __declspec(dllexport) int get_system_ram(unsigned long long* free_bytes, unsigned long long* total_bytes) {
        if (!free_bytes || !total_bytes) return -1;
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo)) {
            *free_bytes = memInfo.ullAvailPhys;
            *total_bytes = memInfo.ullTotalPhys;
            return 0;
        }
        return -2;
    }

    // Queries NVIDIA GPU memory using NVML dynamically.
    // Returns 0 on success, non-zero on error.
    __declspec(dllexport) int get_gpu_vram(unsigned long long* free_bytes, unsigned long long* total_bytes) {
        if (!free_bytes || !total_bytes) return -1;

        // Try to load nvml.dll from system directories
        HMODULE nvml_lib = LoadLibraryA("nvml.dll");
        if (!nvml_lib) {
            // Try standard NVIDIA install path
            nvml_lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        }
        if (!nvml_lib) {
            // Try System32 explicitly
            nvml_lib = LoadLibraryA("C:\\Windows\\System32\\nvml.dll");
        }
        if (!nvml_lib) {
            return -2; // NVML library not found
        }

        typedef int (*nvmlInit_t)();
        typedef int (*nvmlShutdown_t)();
        typedef int (*nvmlDeviceGetHandleByIndex_t)(unsigned int, void**);
        typedef int (*nvmlDeviceGetMemoryInfo_t)(void*, nvmlMemory_t*);

        nvmlInit_t nvmlInit = (nvmlInit_t)GetProcAddress(nvml_lib, "nvmlInit");
        nvmlShutdown_t nvmlShutdown = (nvmlShutdown_t)GetProcAddress(nvml_lib, "nvmlShutdown");
        nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)GetProcAddress(nvml_lib, "nvmlDeviceGetHandleByIndex");
        nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo = (nvmlDeviceGetMemoryInfo_t)GetProcAddress(nvml_lib, "nvmlDeviceGetMemoryInfo");

        if (!nvmlInit || !nvmlShutdown || !nvmlDeviceGetHandleByIndex || !nvmlDeviceGetMemoryInfo) {
            FreeLibrary(nvml_lib);
            return -3; // NVML symbol loading failed
        }

        if (nvmlInit() != 0) {
            FreeLibrary(nvml_lib);
            return -4; // NVML init failed
        }

        void* device_handle = nullptr;
        if (nvmlDeviceGetHandleByIndex(0, &device_handle) != 0) {
            nvmlShutdown();
            FreeLibrary(nvml_lib);
            return -5; // NVML get device handle failed
        }

        nvmlMemory_t mem_info;
        if (nvmlDeviceGetMemoryInfo(device_handle, &mem_info) != 0) {
            nvmlShutdown();
            FreeLibrary(nvml_lib);
            return -6; // NVML get memory info failed
        }

        *free_bytes = mem_info.free;
        *total_bytes = mem_info.total;

        nvmlShutdown();
        FreeLibrary(nvml_lib);
        return 0; // Success
    }

    // High performance cosine similarity using compiler auto-vectorization
    __declspec(dllexport) float compute_cosine_similarity(const float* a, const float* b, int dim) {
        if (!a || !b || dim <= 0) return 0.0f;
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        // Loop is structured to enable SSE/AVX vectorization when compiled with -O3
        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        if (norm_a <= 0.0f || norm_b <= 0.0f) {
            return 0.0f;
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
}
