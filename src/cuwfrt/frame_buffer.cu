#include "frame_buffer.h"

#include <cuda_gl_interop.h>

#include "cuda_buffer.h"
#include "cuda_error.h"

namespace cuwfrt
{

void FrameBuffer::Init(Point2i res)
{
    WakAssert(sizeof(float4) == sizeof(Vec4f));
    WakAssert(sizeof(float4::x) == sizeof(Vec4f::x));

    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, res.x * res.y * sizeof(Vec4f), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register with CUDA
    cudaCheck(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // Create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, res.x, res.y, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    size_t size;
    cudaCheck(cudaGraphicsMapResources(1, &cuda_pbo));
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&buffer, &size, cuda_pbo));

    WakAssert(size == (res.x * res.y * sizeof(float4)));
}

void FrameBuffer::Free()
{
    cudaCheck(cudaGraphicsUnmapResources(1, &cuda_pbo));
    cudaCheck(cudaGraphicsUnregisterResource(cuda_pbo));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
}

void FrameBuffer::Resize(Point2i res)
{
    Free();
    Init(res);
}

} // namespace cuwfrt
