#pragma once

#include <cuda_runtime.h>

#include "common.h"
#include "cuda_error.h"

namespace cuwfrt
{

struct CudaResource1D
{
    virtual void Init(int32 capacity) = 0;
    virtual void Free() = 0;
    virtual void Resize(int32 capacity) = 0;
};

struct CudaResource2D
{
    virtual void Init(Point2i resolution) = 0;
    virtual void Free() = 0;
    virtual void Resize(Point2i resolution) = 0;
};

template <typename T>
struct Buffer : CudaResource1D
{
    void Init(int32 capacity)
    {
        cudaCheck(cudaMalloc<T>(&buffer, capacity * sizeof(T)));
    }

    void Free()
    {
        cudaCheck(cudaFree(buffer));
    }

    void Resize(int32 capacity)
    {
        Free();
        Init(capacity);
    }

    T* operator&()
    {
        return buffer;
    }

    T* buffer;
};

} // namespace cuwfrt
