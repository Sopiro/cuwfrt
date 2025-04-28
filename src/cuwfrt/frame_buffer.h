#pragma once

#include "alzartak/common.h"
#include "cuda_buffer.h"

namespace cuwfrt
{

struct FrameBuffer : CudaResource2D
{
    // Initialize PBO & CUDA interop capability
    void Init(Point2i res);
    void Free();
    void Resize(Point2i res);

    Vec4* operator&()
    {
        return buffer;
    }

    GLuint pbo, texture;
    cudaGraphicsResource* cuda_pbo;
    Vec4* buffer;
};

} // namespace cuwfrt
