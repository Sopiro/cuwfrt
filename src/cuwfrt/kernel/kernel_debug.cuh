#pragma once

#include <device_launch_parameters.h>

#include "cuwfrt/raytracer.h"
#include "kernel_intersect.cuh"
#include "kernel_material.cuh"

namespace cuwfrt
{

__KERNEL__ void RenderGradient(
    Vec4* __restrict__ sample_buffer,
    Vec4* __restrict__ frame_buffer,
    Point2i res,
    GPUScene scene,
    Camera camera,
    Options options,
    int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    frame_buffer[index] = Vec4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

__KERNEL__ void RenderNormal(
    Vec4* __restrict__ sample_buffer,
    Vec4* __restrict__ frame_buffer,
    Point2i res,
    GPUScene scene,
    Camera camera,
    Options options,
    int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    RNG rng(Hash(x, y, time));

    // Generate primary ray
    Ray ray;
    Point2 u0{ rng.NextFloat(), rng.NextFloat() };
    Point2 u1{ rng.NextFloat(), rng.NextFloat() };
    camera.SampleRay(&ray, Point2i(x, y), u0, u1);

    Intersection isect;
    bool found_intersection = Intersect(&isect, &scene, ray, Ray::epsilon, infinity);

    if (found_intersection)
    {
        sample_buffer[index] *= time;
        sample_buffer[index] += Vec4((isect.shading_normal + 1) * 0.5, 1);
        sample_buffer[index] /= time + 1.0f;
    }
    else
    {
        sample_buffer[index] = Vec4(0, 0, 0, 1);
    }

    frame_buffer[index].x = sample_buffer[index].x;
    frame_buffer[index].y = sample_buffer[index].y;
    frame_buffer[index].z = sample_buffer[index].z;
    frame_buffer[index].w = 1.0f;
}

__KERNEL__ void ClearBuffer(Vec4* __restrict__ buffer, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;
    buffer[index] = Vec4(0);
}

} // namespace cuwfrt
