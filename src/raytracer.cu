#include "raytracer.h"
#include "scene.h"

#include "alzartak/window.h"
#include "kernel/kernels.cuh"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace cuwfrt
{

RayTracer::RayTracer(Scene* scene, Camera* camera)
    : scene{ scene }
    , camera{ camera }
{
    res = Window::Get()->GetWindowSize();

    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, res.x * res.y * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
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

    // Init GPU resources
    InitGPUResources();
}

RayTracer::~RayTracer()
{
    FreeGPUResources();

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

void RayTracer::Update()
{
    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos({ 4, 4 }, ImGuiCond_Once, { 0.0f, 0.0f });
    if (ImGui::Begin("alzartak", NULL))
    {
        ImGui::Text("%dfps", int32(io.Framerate));
    }
    ImGui::End();

    RenderGPU();
    UpdateTexture();

    RenderQuad();
}

// Render to the PBO using CUDA
void RayTracer::RenderGPU()
{
    float4* device_ptr;
    size_t size;

    cudaCheck(cudaGraphicsMapResources(1, &cuda_pbo));
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &size, cuda_pbo));

    const dim3 threads(8, 8);
    const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);

    Render<<<blocks, threads>>>(device_ptr, res, *camera);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaGraphicsUnmapResources(1, &cuda_pbo));
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
