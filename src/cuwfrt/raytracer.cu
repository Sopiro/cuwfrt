#include "cuda_error.cuh"
#include "raytracer.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cuwfrt/kernel/kernel_wavefront.cuh"

namespace cuwfrt
{

RayTracer::RayTracer(Window* window, const Scene* scene, const Camera* camera, const Options* options)
    : window{ window }
    , scene{ scene }
    , camera{ camera }
    , options{ options }
{
    res = window->GetWindowSize();

    window->SetFramebufferSizeChangeCallback([&](int32 width, int32 height) -> void { Resize(width, height); });

    CreateFrameBuffer();
    InitGPUResources();
}

RayTracer::~RayTracer()
{
    FreeGPUResources();
    DeleteFrameBuffer();
}

// Initialize PBO & CUDA interop capability
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
    gpu_res.Init(scene);
    wf.Init(res);
}

void RayTracer::FreeGPUResources()
{
    std::cout << "Free GPU resources" << std::endl;
    gpu_res.Free();
    wf.Free();
}

void RayTracer::Resize(int32 width, int32 height)
{
    if (width <= 0 || height <= 0 || (res.x == width && res.y == height))
    {
        return;
    }

    res.Set(width, height);
    glViewport(0, 0, width, height);

    cudaCheck(cudaDeviceSynchronize());

    // Recreate framebuffer
    DeleteFrameBuffer();
    CreateFrameBuffer();

    wf.Resize(res);
}

void RayTracer::RayTrace(Kernel* kernel, int32 t)
{
    time = t;

    if (kernel)
    {
        // Render to the PBO using CUDA
        const dim3 threads(8, 8);
        const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);

        kernel<<<blocks, threads>>>(d_sample_buffer, d_frame_buffer, res, gpu_res.scene, *camera, *options, time);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
    }
}

void RayTracer::RayTraceWavefront(int32 t)
{
    time = t;

    int32 num_active_rays = wf.ray_capacity;
    int32 num_next_rays = 0;
    int32 num_closest_rays = 0;
    int32 num_miss_rays = 0;
    int32 num_shadow_rays = 0;

    // Generate Primary Rays
    {
        const dim3 threads(16, 16);
        const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);
        GeneratePrimaryRays<<<blocks, threads>>>(d_sample_buffer, wf.rays_active, res, *camera, time);
        cudaCheck(cudaGetLastError());
    }

    int32 bounce = 0;
    while (true)
    {
        ResetCounts<<<1, 1>>>(wf.next_ray_count, wf.closest_ray_count, wf.miss_ray_count, wf.shadow_ray_count);
        cudaCheck(cudaGetLastError());

        // Extend rays
        {
            int32 threads = 128;
            int32 blocks = (num_active_rays + threads - 1) / threads;
            Extend<<<blocks, threads>>>(
                wf.rays_active, num_active_rays, wf.rays_closest, wf.closest_ray_count, wf.miss_rays, wf.miss_ray_count,
                gpu_res.scene
            );
            cudaCheck(cudaGetLastError());
        }

        // Get counts of newly generated rays (closest hit and miss)
        cudaCheck(cudaMemcpy(&num_closest_rays, wf.closest_ray_count, sizeof(int32), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(&num_miss_rays, wf.miss_ray_count, sizeof(int32), cudaMemcpyDeviceToHost));

        // Handle misses
        if (num_miss_rays > 0)
        {
            int32 threads = 128;
            int32 blocks = (num_miss_rays + threads - 1) / threads;
            Miss<<<blocks, threads>>>(wf.miss_rays, num_miss_rays, d_sample_buffer, *options);
            cudaCheck(cudaGetLastError());
        }

        // Shade surfaces
        if (num_closest_rays > 0)
        {
            int32 threads = 128;
            int32 blocks = (num_closest_rays + threads - 1) / threads;
            Shade<<<blocks, threads>>>(
                wf.rays_closest, num_closest_rays, wf.rays_next, wf.next_ray_count, wf.shadow_rays, wf.shadow_ray_count,
                d_sample_buffer, gpu_res.scene, *options, bounce, time
            );
            cudaCheck(cudaGetLastError());
        }

        // Get counts of newly generated rays (next bounce and shadow)
        cudaCheck(cudaMemcpy(&num_next_rays, wf.next_ray_count, sizeof(int32), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(&num_shadow_rays, wf.shadow_ray_count, sizeof(int32), cudaMemcpyDeviceToHost));

        if (bounce++ >= options->max_bounces)
        {
            break;
        }

        // Prepare for next bounce
        std::swap(wf.rays_active, wf.rays_next);
        num_active_rays = num_next_rays;

        if (num_active_rays <= 0)
        {
            break;
        }
    }

    // Finalize samples
    {
        const dim3 threads(16, 16);
        const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);
        Finalize<<<blocks, threads>>>(d_sample_buffer, d_frame_buffer, res, time);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
    }
}

void RayTracer::DrawFrame()
{
    UpdateTexture();
    RenderQuad();
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
