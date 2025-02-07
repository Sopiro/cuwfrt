#include "cuda_error.cuh"
#include "kernels.cuh"
#include "raytracer.cuh"
#include "scene.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace cuwfrt
{

RayTracer::RayTracer(Window* window, Scene* scene, Camera* camera, Options* options)
    : window{ window }
    , scene{ scene }
    , camera{ camera }
    , options{ options }
{
    res = window->GetWindowSize();

    window->SetFramebufferSizeChangeCallback([&](int32 width, int32 height) -> void {
        glViewport(0, 0, width, height);
        res.Set(width, height);

        // Recreate framebuffer
        DeleteFrameBuffer();
        CreateFrameBuffer();
    });

    CreateFrameBuffer();
    InitGPUResources();
}

RayTracer::~RayTracer()
{
    FreeGPUResources();
    DeleteFrameBuffer();
}

void RayTracer::CreateFrameBuffer()
{
    WakAssert(sizeof(float4) == sizeof(Vec4f));

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
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&d_frame_buffer, &size, cuda_pbo));
    WakAssert(size == (res.x * res.y * sizeof(float4)));

    cudaCheck(cudaMalloc(&d_sample_buffer, size));
}

void RayTracer::DeleteFrameBuffer()
{
    cudaCheck(cudaFree(d_sample_buffer));

    cudaCheck(cudaGraphicsUnmapResources(1, &cuda_pbo));
    cudaCheck(cudaGraphicsUnregisterResource(cuda_pbo));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
}

void RayTracer::InitGPUResources()
{
    std::cout << "Init GPU resources" << std::endl;
    gpu_scene.Init(scene);
}

void RayTracer::FreeGPUResources()
{
    std::cout << "Free GPU resources" << std::endl;
    gpu_scene.Free();
}

void RayTracer::RayTrace(int32 t)
{
    time = t;

    RenderGPU();
}

void RayTracer::DrawFrame()
{
    UpdateTexture();
    RenderQuad();
}

// Render to the PBO using CUDA
void RayTracer::RenderGPU()
{
    const dim3 threads(8, 8);
    const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);

    PathTrace<<<blocks, threads>>>(d_sample_buffer, d_frame_buffer, res, gpu_scene.data, *camera, *options, time);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

// Copy PBO data to texture
void RayTracer::UpdateTexture()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, res.x, res.y, GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// OpenGL Rendering: Use PBO texture on a fullscreen quad
void RayTracer::RenderQuad()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    qr.Draw();
}

} // namespace cuwfrt
