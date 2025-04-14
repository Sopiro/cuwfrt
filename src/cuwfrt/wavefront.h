#pragma once

#include "common.h"
#include "cuda_error.cuh"
#include "cuwfrt/geometry/intersection.h"

namespace cuwfrt
{

struct WavefrontRay
{
    RNG rng;

    Ray ray;
    Intersection isect;

    Vec3 beta;

    Float last_bsdf_pdf;
    bool is_specular;

    int32 pixel_index;
};

struct WavefrontMissRay
{
    Vec3 d;
    Vec3 beta;
    int32 pixel_index;
};

struct WavefrontShadowRay
{
    Ray ray;
    Float visibility;

    Vec3 Li;
    int32 pixel_index;
};

template <typename T>
struct RayQueue
{
    T* rays;
    int32* count;

    void Init(int32 capacity)
    {
        cudaCheck(cudaMalloc(&rays, capacity * sizeof(T)));
        cudaCheck(cudaMalloc(&count, sizeof(int32)));
    }

    void Free()
    {
        cudaCheck(cudaFree(rays));
        cudaCheck(cudaFree(count));
    }

    void Resize(int32 capacity)
    {
        cudaCheck(cudaFree(rays));
        cudaCheck(cudaMalloc(&rays, capacity * sizeof(T)));
    }
};

template <typename T, int32 size>
struct RayQueues
{
    T* rays[size];
    int32* counts[size];

    void Init(int32 capacity)
    {
        for (int32 i = 0; i < size; ++i)
        {
            cudaCheck(cudaMalloc(&rays[i], capacity * sizeof(T)));
            cudaCheck(cudaMalloc(&counts[i], sizeof(int32)));
        }
    }

    void Free()
    {
        for (int32 i = 0; i < size; ++i)
        {
            cudaCheck(cudaFree(rays[i]));
            cudaCheck(cudaFree(counts[i]));
        }
    }

    void Resize(int32 capacity)
    {
        for (int32 i = 0; i < size; ++i)
        {
            cudaCheck(cudaFree(rays[i]));
        }

        for (int32 i = 0; i < size; ++i)
        {
            cudaCheck(cudaMalloc(&rays[i], capacity * sizeof(T)));
        }
    }
};

struct WavefrontResources
{
    int32 ray_capacity;

    RayQueue<WavefrontRay> active;
    RayQueue<WavefrontRay> next;
    RayQueues<WavefrontRay, Materials::count> closest;

    RayQueue<WavefrontMissRay> miss;
    RayQueue<WavefrontShadowRay> shadow;

    void Init(Point2i res)
    {
        ray_capacity = res.x * res.y;

        active.Init(ray_capacity);
        next.Init(ray_capacity);
        closest.Init(ray_capacity);
        miss.Init(ray_capacity);
        shadow.Init(ray_capacity);
    }

    void Free()
    {
        active.Free();
        next.Free();
        closest.Free();
        miss.Free();
        shadow.Free();
    }

    void Resize(Point2i res)
    {
        ray_capacity = res.x * res.y;

        active.Resize(ray_capacity);
        next.Resize(ray_capacity);
        closest.Resize(ray_capacity);
        miss.Resize(ray_capacity);
        shadow.Resize(ray_capacity);
    }
};

} // namespace cuwfrt
