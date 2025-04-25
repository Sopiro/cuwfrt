#pragma once

#include <device_launch_parameters.h>

#include "cuwfrt/raytracer.h"
#include "device_intersect.cuh"
#include "device_material.cuh"

namespace cuwfrt
{

__KERNEL__ void RenderGradient(
    Vec4* __restrict__ sample_buffer, Point2i res, GPUScene scene, Camera camera, Options options, int32 seed
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    sample_buffer[index] = Vec4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

__KERNEL__ void RenderNormal(
    Vec4* __restrict__ sample_buffer, Point2i res, GPUScene scene, Camera camera, Options options, int32 seed
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    RNG rng(Hash(x, y, seed));

    // Generate primary ray
    Ray ray;
    Point2 u0{ rng.NextFloat(), rng.NextFloat() };
    Point2 u1{ rng.NextFloat(), rng.NextFloat() };
    camera.SampleRay(&ray, Point2i(x, y), u0, u1);

    Intersection isect;
    bool found_intersection = Intersect(&isect, &scene, ray, Ray::epsilon, infinity);

    if (found_intersection)
    {
        sample_buffer[index] = Vec4((isect.shading_normal + 1) * 0.5, 1);
    }
    else
    {
        sample_buffer[index] = Vec4(0, 0, 0, 1);
    }
}

} // namespace cuwfrt
