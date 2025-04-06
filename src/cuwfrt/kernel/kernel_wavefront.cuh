#pragma once

#include "cuwfrt/raytracer.cuh"
#include "device_launch_parameters.h"
#include "kernel_intersect.cuh"
#include "kernel_material.cuh"

namespace cuwfrt
{

__KERNEL__ void ResetCounts(int32* next_ray_count, int32* shadow_ray_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *next_ray_count = 0;
        *shadow_ray_count = 0;
    }
}

// Generate primary rays for each pixel
__KERNEL__ void GeneratePrimaryRays(WavefrontRay* __restrict__ active_rays, Point2i res, Camera camera, int32 time)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    // Initialize wavefront path states
    WavefrontRay& wf_ray = active_rays[index];

    RNG rng(Hash(x, y, time));

    // Generate primary ray
    camera.SampleRay(&wf_ray.ray, { x, y }, { rng.NextFloat(), rng.NextFloat() }, { rng.NextFloat(), rng.NextFloat() });
    wf_ray.isect = { 0 };

    wf_ray.rng = rng;
    wf_ray.beta = Vec3(1);
    wf_ray.bounce = 0;

    wf_ray.last_pdf = 1.0f;
    wf_ray.is_specular = false;

    wf_ray.pixel_index = index;
}

// Trace rays and find closest intersection
__KERNEL__ void Extend(WavefrontRay* __restrict__ active_rays, int32 active_ray_count, GPUScene scene)
{
    int32 index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= active_ray_count) return;

    WavefrontRay& wf_ray = active_rays[index];
    bool found_intersection = Intersect(&wf_ray.isect, &scene, wf_ray.ray, Ray::epsilon, infinity);
    if (!found_intersection)
    {
        wf_ray.isect.t = -1;
    }
}

// Shade hit points, generate next bounce rays and shadow rays
__KERNEL__ void Shade(
    WavefrontRay* __restrict__ active_rays,
    int32 active_ray_count,
    WavefrontRay* __restrict__ next_rays,
    int32* next_ray_count,
    WavefrontShadowRay* __restrict__ shadow_rays,
    int32* shadow_ray_count,
    Vec4* __restrict__ sample_buffer,
    GPUScene scene,
    Options options,
    int32 time
)
{
    int32 index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= active_ray_count) return;

    WavefrontRay& wf_ray = active_rays[index];
    int32 pixel_index = wf_ray.pixel_index;

    if (wf_ray.isect.t > 0)
    {
        sample_buffer[pixel_index] *= time;
        sample_buffer[pixel_index] += Vec4((wf_ray.isect.normal + 1) * 0.5, 1);
        sample_buffer[pixel_index] /= time + 1.0f;
    }
    else
    {
        sample_buffer[pixel_index] = Vec4(0, 0, 0, 1);
    }
}

// Process shadow rays, add contribution if unoccluded
__KERNEL__ void Connect(
    WavefrontShadowRay* __restrict__ shadow_rays, int32 shadow_ray_count, Vec4* __restrict__ sample_buffer, GPUScene scene
)
{
    int32 index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= shadow_ray_count) return;

    WavefrontShadowRay& wf_shadow_ray = shadow_rays[index];
    // todo
}

// Finalize frame: average samples and apply gamma correction
__KERNEL__ void Finalize(const Vec4* __restrict__ sample_buffer, Vec4* __restrict__ frame_buffer, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int32 index = y * res.x + x;

    frame_buffer[index].x = std::pow(sample_buffer[index].x, 1 / 2.2f);
    frame_buffer[index].y = std::pow(sample_buffer[index].y, 1 / 2.2f);
    frame_buffer[index].z = std::pow(sample_buffer[index].z, 1 / 2.2f);
    frame_buffer[index].w = 1.0f;
}

} // namespace cuwfrt