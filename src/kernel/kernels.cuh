#pragma once

#include "../camera.h"
#include "api.cuh"
#include "wak/hash.h"
#include "wak/random.h"

using namespace cuwfrt;
using namespace wak;

__kernel__ void RenderGradient(float4* pixels, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    pixels[index] = make_float4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

__kernel__ void Render(float4* pixels, Point2i res, Camera camera)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    RNG rng(Hash(x, y));

    Ray ray;
    camera.SampleRay(&ray, Point2i(x, y), { rng.NextFloat(), rng.NextFloat() }, { rng.NextFloat(), rng.NextFloat() });

    pixels[y * res.x + x] = make_float4(ray.d.x, ray.d.y, ray.d.z, 1.0f);
}
