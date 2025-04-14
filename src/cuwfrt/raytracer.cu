#include "cuda_error.cuh"
#include "raytracer.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cuwfrt/kernel/kernel_albedo.cuh"
#include "cuwfrt/kernel/kernel_ao.cuh"
#include "cuwfrt/kernel/kernel_debug.cuh"
#include "cuwfrt/kernel/kernel_pt_naive.cuh"
#include "cuwfrt/kernel/kernel_pt_nee.cuh"
#include "cuwfrt/kernel/kernel_wavefront.cuh"

namespace cuwfrt
{

const int32 RayTracer::num_kernels = 7;
const char* RayTracer::kernel_name[] = { "Gradient", "Normal", "AO", "Albedo", "Pathtrace Naive", "Pathtrace NEE", "Wavefront" };

static Kernel* kernels[RayTracer::num_kernels] = { RenderGradient, RenderNormal,   RaytraceAO,
                                                   RaytraceAlbedo, PathTraceNaive, PathTraceNEE };

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

    for (size_t i = 0; i < streams.size(); ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    for (size_t i = 0; i < ray_queue_streams.size(); ++i)
    {
        cudaStreamCreate(&ray_queue_streams[i]);
    }
}

void RayTracer::FreeGPUResources()
{
    std::cout << "Free GPU resources" << std::endl;
    gpu_res.Free();
    wf.Free();

    for (size_t i = 0; i < streams.size(); ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    for (size_t i = 0; i < ray_queue_streams.size(); ++i)
    {
        cudaStreamDestroy(ray_queue_streams[i]);
    }
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

void RayTracer::RayTrace(int32 kernel_index, int32 t)
{
    time = t;
    kernel_index = (kernel_index + num_kernels) % num_kernels;

    // Render to the PBO using CUDA
    const dim3 threads(8, 8);
    const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);

    kernels[kernel_index]<<<blocks, threads>>>(d_sample_buffer, d_frame_buffer, res, gpu_res.scene, *camera, *options, time);

    cudaCheckLastError();
    cudaCheck(cudaDeviceSynchronize());
}

void RayTracer::RayTraceWavefront(int32 t)
{
    time = t;

    int32 num_active_rays = wf.ray_capacity;
    int32 num_next_rays = 0;
    int32 num_closest_rays[Materials::count] = { 0 };
    int32 num_miss_rays = 0;
    int32 num_shadow_rays = 0;

    // Generate Primary Rays
    {
        const dim3 threads(16, 16);
        const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);
        GeneratePrimaryRays<<<blocks, threads>>>(d_sample_buffer, wf.rays_active, res, *camera, time);
        cudaCheckLastError();
    }

    int32 bounce = 0;
    while (true)
    {
        ResetCounts<<<1, 1>>>(wf.next_ray_count, wf.rays_closest, wf.miss_ray_count, wf.shadow_ray_count);
        cudaCheckLastError();

        // Extend rays
        {
            const int32 threads = 128;
            int32 blocks = (num_active_rays + threads - 1) / threads;
            Extend<<<blocks, threads>>>(
                wf.rays_active, num_active_rays, wf.rays_closest, wf.miss_rays, wf.miss_ray_count, gpu_res.scene
            );
            cudaCheckLastError();
        }

        // Get counts of newly generated rays (closest hit and miss)
        cudaCheck(cudaMemcpy(&num_miss_rays, wf.miss_ray_count, sizeof(int32), cudaMemcpyDeviceToHost));
        for (int32 i = 0; i < Materials::count; ++i)
        {
            cudaCheck(cudaMemcpy(&num_closest_rays[i], wf.rays_closest.ray_counts[i], sizeof(int32), cudaMemcpyDeviceToHost));
        }

        // Handle misses
        if (num_miss_rays > 0)
        {
            const int32 threads = 128;
            int32 blocks = (num_miss_rays + threads - 1) / threads;
            Miss<<<blocks, threads, 0, streams[0]>>>(wf.miss_rays, num_miss_rays, d_sample_buffer, *options);
            cudaCheckLastError();
        }

        // Intersects closest
        for (int32 i = 0; i < Materials::count; ++i)
        {
            if (num_closest_rays[i] > 0)
            {
                const int32 threads = 128;
                int32 blocks = (num_closest_rays[i] + threads - 1) / threads;

                DynamicDispatcher<Materials>(i).Dispatch([&](auto* m) {
                    using MaterialType = std::remove_pointer_t<decltype(m)>;
                    Closest<MaterialType><<<blocks, threads, 0, ray_queue_streams[i]>>>(
                        wf.rays_closest.rays[i], num_closest_rays[i], wf.rays_next, wf.next_ray_count, wf.shadow_rays,
                        wf.shadow_ray_count, d_sample_buffer, gpu_res.scene, *options, bounce, time
                    );
                    cudaCheckLastError();
                });
            }
        }

        if (bounce++ >= options->max_bounces)
        {
            break;
        }

        for (int32 i = 0; i < Materials::count; ++i)
        {
            cudaCheck(cudaStreamSynchronize(ray_queue_streams[i]));
        }

        // Get counts of newly generated rays (shadow and next bounce)
        cudaCheck(cudaMemcpyAsync(&num_shadow_rays, wf.shadow_ray_count, sizeof(int32), cudaMemcpyDeviceToHost, streams[1]));

        // Test shadow ray and incorporate direct light
        if (num_shadow_rays > 0)
        {
            const int32 threads = 128;
            int32 blocks = (num_shadow_rays + threads - 1) / threads;
            Connect<<<blocks, threads, 0, streams[1]>>>(wf.shadow_rays, num_shadow_rays, d_sample_buffer, gpu_res.scene);
            cudaCheckLastError();
        }

        cudaCheck(cudaMemcpyAsync(&num_next_rays, wf.next_ray_count, sizeof(int32), cudaMemcpyDeviceToHost));

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
        cudaCheckLastError();
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
