#pragma once

#include <cuda_runtime.h>
#include <iostream>

#ifdef BENCHMARK
#define cudaCheck(val) (val)
#define cudaCheckLastError() ((void)0)
#else
#define cudaCheck(val) cuda_check((val), #val, __FILE__, __LINE__)
#define cudaCheckLastError() cudaCheck(cudaGetLastError())
#endif

// https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda
inline void cuda_check(cudaError_t result, const char* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func
                  << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}