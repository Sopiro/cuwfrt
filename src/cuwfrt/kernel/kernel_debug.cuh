#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"

#include "cuwfrt/cuda_api.h"
#include "cuwfrt/shading/frame.h"
#include "cuwfrt/shading/sampling.h"

#include "cuwfrt/camera/camera.h"
#include "cuwfrt/scene/gpu_scene.cuh"

#include "kernel_intersect.cuh"
#include "kernel_material.cuh"

namespace cuwfrt
{

__KERNEL__ void RenderGradient(
    Vec4* sample_buffer, Vec4* frame_buffer, Point2i res, GPUScene scene, Camera camera, Options options, int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    frame_buffer[index] = Vec4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

__KERNEL__ void RenderNormal(
    Vec4* sample_buffer, Vec4* frame_buffer, Point2i res, GPUScene scene, Camera camera, Options options, int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    RNG rng(Hash(x, y, time));

    // Generate primary ray
    Ray ray;
    Point2 u0{ rng.NextFloat(), rng.NextFloat() };
    Point2 u1{ rng.NextFloat(), rng.NextFloat() };
    camera.SampleRay(&ray, Point2i(x, y), u0, u1);

    Intersection isect;
    bool found_intersection = Intersect(&isect, &scene, ray, Ray::epsilon, infinity);

    int32 index = y * res.x + x;
    if (found_intersection)
    {
        frame_buffer[index] = Vec4((isect.normal + 1) * 0.5, 1);
    }
    else
    {
        frame_buffer[index] = Vec4(0, 0, 0, 1);
    }
}

} // namespace cuwfrt
