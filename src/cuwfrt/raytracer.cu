#include "cuda_error.h"
#include "raytracer.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cuwfrt/kernel/kernel_accumulate.cuh"
#include "cuwfrt/kernel/kernel_albedo.cuh"
#include "cuwfrt/kernel/kernel_ao.cuh"
#include "cuwfrt/kernel/kernel_debug.cuh"
#include "cuwfrt/kernel/kernel_denoise.cuh"
#include "cuwfrt/kernel/kernel_pt_naive.cuh"
#include "cuwfrt/kernel/kernel_pt_nee.cuh"
#include "cuwfrt/kernel/kernel_wavefront.cuh"

namespace cuwfrt
{

const int32 RayTracer::num_kernels = 6;
const char* RayTracer::kernel_names[] = { "Normal", "AO", "Albedo", "Pathtrace Naive", "Pathtrace NEE", "Wavefront" };

static Kernel* kernels[RayTracer::num_kernels - 1] = { RenderNormal, RaytraceAO, RaytraceAlbedo, PathTraceNaive, PathTraceNEE };

RayTracer::RayTracer(Window* window, const Scene* scene, const Camera* camera, const Options* options)
    : window{ window }
    , scene{ scene }
    , camera{ camera }
    , options{ options }
    , frame_index{ 0 }
    , spp{ 0 }
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
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&frame_buffer, &size, cuda_pbo));

    WakAssert(size == (res.x * res.y * sizeof(float4)));
}

void RayTracer::DeleteFrameBuffer()
{
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

    const int32 capacity = res.x * res.y;
    for (int32 i = 0; i < 2; ++i)
    {
        cudaCheck(cudaMalloc(&sample_buffer[i], capacity * sizeof(float4)));
        g_buffer[i].Init(capacity);
        h_buffer[i].Init(capacity);
    }
    cudaCheck(cudaMalloc(&accumulation_buffer, capacity * sizeof(float4)));

    for (size_t i = 0; i < streams.size(); ++i)
    {
        cudaCheck(cudaStreamCreate(&streams[i]));
    }

    for (size_t i = 0; i < ray_queue_streams.size(); ++i)
    {
        cudaCheck(cudaStreamCreate(&ray_queue_streams[i]));
    }
}

void RayTracer::FreeGPUResources()
{
    std::cout << "Free GPU resources" << std::endl;
    gpu_res.Free();
    wf.Free();

    for (int32 i = 0; i < 2; ++i)
    {
        cudaCheck(cudaFree(sample_buffer[i]));
        g_buffer[i].Free();
        h_buffer[i].Free();
    }
    cudaCheck(cudaFree(accumulation_buffer));

    for (size_t i = 0; i < streams.size(); ++i)
    {
        cudaCheck(cudaStreamDestroy(streams[i]));
    }

    for (size_t i = 0; i < ray_queue_streams.size(); ++i)
    {
        cudaCheck(cudaStreamDestroy(ray_queue_streams[i]));
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

    const int32 capacity = res.x * res.y;
    for (int32 i = 0; i < 2; ++i)
    {
        cudaCheck(cudaFree(sample_buffer[i]));
        cudaCheck(cudaMalloc(&sample_buffer[i], capacity * sizeof(float4)));
        g_buffer[i].Resize(capacity);
        h_buffer[i].Resize(capacity);
    }
    cudaCheck(cudaFree(accumulation_buffer));
    cudaCheck(cudaMalloc(&accumulation_buffer, capacity * sizeof(float4)));
}

void RayTracer::Clear()
{
    g_camera[frame_index] = *camera;
    spp = 0;
}

void RayTracer::RayTrace(int32 kernel_index)
{
    kernel_index = Clamp(kernel_index, 0, num_kernels - 1);
    frame_index = 1 - frame_index;

    const dim3 threads(16, 16);
    const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);

    kernels[kernel_index]<<<blocks, threads>>>(sample_buffer[frame_index], res, gpu_res.scene, *camera, *options, spp);
    cudaCheckLastError();

    Accumulate<<<blocks, threads>>>(sample_buffer[frame_index], accumulation_buffer, frame_buffer, res, spp++);
    cudaCheckLastError();

    cudaCheck(cudaDeviceSynchronize());
}

void RayTracer::RayTraceWavefront()
{
    frame_index = 1 - frame_index;

    // Save camera data for motion vector calculation
    g_camera[frame_index] = *camera;

    int32 num_active_rays = wf.ray_capacity;
    int32 num_next_rays = 0;
    int32 num_closest_rays[wf.closest_queue_count] = { 0 };
    int32 num_miss_rays = 0;
    int32 num_shadow_rays = 0;

    // Generate Primary Rays
    {
        const dim3 threads(16, 16);
        const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);
        GeneratePrimaryRays<<<blocks, threads>>>(
            wf.active.rays, sample_buffer[frame_index], res, g_camera[frame_index], g_buffer[frame_index], spp++
        );
        cudaCheckLastError();
    }

    int32 bounce = 0;
    while (true)
    {
        ResetCounts<<<1, 1>>>(wf.next.count, wf.closest, wf.miss.count, wf.shadow.count);
        cudaCheckLastError();

        // Trace rays
        {
            const int32 threads = 128;
            int32 blocks = (num_active_rays + threads - 1) / threads;
            TraceRay<<<blocks, threads>>>(wf.active.rays, num_active_rays, wf.closest, wf.miss, gpu_res.scene);
            cudaCheckLastError();
        }

        // Get counts of newly generated rays (closest hit and miss)
        cudaCheck(cudaMemcpy(&num_miss_rays, wf.miss.count, sizeof(int32), cudaMemcpyDeviceToHost));
        for (int32 i = 0; i < wf.closest_queue_count; ++i)
        {
            cudaCheck(cudaMemcpy(&num_closest_rays[i], wf.closest.counts[i], sizeof(int32), cudaMemcpyDeviceToHost));
        }

        // Handle misses
        if (options->render_sky && num_miss_rays > 0)
        {
            const int32 threads = 128;
            int32 blocks = (num_miss_rays + threads - 1) / threads;
            Miss<<<blocks, threads, 0, streams[0]>>>(wf.miss.rays, num_miss_rays, sample_buffer[frame_index]);
            cudaCheckLastError();
        }

        // Intersects closest
        for (int32 i = 0; i < wf.closest_queue_count; ++i)
        {
            if (num_closest_rays[i] > 0)
            {
                const int32 threads = 128;
                int32 blocks = (num_closest_rays[i] + threads - 1) / threads;

                DynamicDispatcher<Materials>(i).Dispatch([&](auto* m) {
                    using MaterialType = std::remove_pointer_t<decltype(m)>;
                    Closest<MaterialType><<<blocks, threads, 0, ray_queue_streams[i]>>>(
                        wf.closest.rays[i], num_closest_rays[i], wf.next, wf.shadow, sample_buffer[frame_index], gpu_res.scene,
                        g_buffer[frame_index], bounce
                    );
                    cudaCheckLastError();
                });
            }
        }

        if (bounce++ >= options->max_bounces)
        {
            break;
        }

        for (int32 i = 0; i < wf.closest_queue_count; ++i)
        {
            cudaCheck(cudaStreamSynchronize(ray_queue_streams[i]));
        }

        // Get counts of newly generated rays (shadow and next bounce)
        cudaCheck(cudaMemcpyAsync(&num_shadow_rays, wf.shadow.count, sizeof(int32), cudaMemcpyDeviceToHost, streams[1]));

        // Test shadow ray and incorporate direct light
        if (num_shadow_rays > 0)
        {
            const int32 threads = 128;
            int32 blocks = (num_shadow_rays + threads - 1) / threads;
            TraceShadowRay<<<blocks, threads, 0, streams[1]>>>(
                wf.shadow.rays, num_shadow_rays, sample_buffer[frame_index], gpu_res.scene
            );
            cudaCheckLastError();
        }

        cudaCheck(cudaMemcpyAsync(&num_next_rays, wf.next.count, sizeof(int32), cudaMemcpyDeviceToHost));

        // Prepare for next bounce
        std::swap(wf.active, wf.next);
        num_active_rays = num_next_rays;

        if (num_active_rays <= 0)
        {
            break;
        }
    }

    cudaCheck(cudaDeviceSynchronize());
}

void RayTracer::Denoise()
{
    const dim3 threads(16, 16);
    const dim3 blocks((res.x + threads.x - 1) / threads.x, (res.y + threads.y - 1) / threads.y);

    PrepareDenoise<<<blocks, threads>>>(sample_buffer[frame_index], g_buffer[frame_index], h_buffer[frame_index], res);
    cudaCheckLastError();

    FilterTemporal<<<blocks, threads>>>(
        sample_buffer[frame_index], res, g_buffer[1 - frame_index], g_buffer[frame_index], g_camera[1 - frame_index],
        h_buffer[1 - frame_index], h_buffer[frame_index]
    );
    cudaCheckLastError();

    EstimateVariance<<<blocks, threads>>>(g_buffer[frame_index], h_buffer[frame_index], h_buffer[1 - frame_index], res);
    cudaCheckLastError();

    FilterVariance<<<blocks, threads>>>(h_buffer[1 - frame_index], h_buffer[frame_index], res);
    cudaCheckLastError();

    const int32 atrous_iterations = 5;

    int32 current_index = frame_index;
    for (int32 i = 0; i < atrous_iterations; ++i)
    {
        int32 step = 1 << i;
        FilterSpatial<<<blocks, threads>>>(
            sample_buffer[current_index], sample_buffer[1 - current_index], res, step, g_buffer[frame_index],
            h_buffer[current_index], h_buffer[1 - current_index]
        );
        cudaCheckLastError();

        current_index = 1 - current_index;
    }

    FinalizeDenoise<<<blocks, threads>>>(frame_buffer, sample_buffer[1 - current_index], res, g_buffer[frame_index]);
    cudaCheckLastError();

    cudaCheck(cudaDeviceSynchronize());
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
