#pragma once

#ifdef __CUDACC__
#define __kernel__ __global__
#define __cpu__ __host__
#define __gpu__ __device__
#define __cpu_gpu__ __host__ __device__
#else
#define __kernel__
#define __cpu__
#define __gpu__
#define __cpu_gpu__
#endif

// https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func
                  << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}