#pragma once

#include "common.h"
#include "cuda_buffer.h"
#include "cuda_error.h"
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

struct WavefrontPathStates : CudaResource1D
{
    // SoA-ed WavefrontRay struct
    RNG* rngs = nullptr;
    Ray* rays = nullptr;
    Intersection* isects = nullptr;
    Vec3* betas = nullptr;
    Float* last_bsdf_pdfs = nullptr;
    uint8* specular_bounces = nullptr;

    void Init(int32 capacity)
    {
        cudaCheck(cudaMalloc(&rngs, capacity * sizeof(RNG)));
        cudaCheck(cudaMalloc(&rays, capacity * sizeof(Ray)));
        cudaCheck(cudaMalloc(&isects, capacity * sizeof(Intersection)));
        cudaCheck(cudaMalloc(&betas, capacity * sizeof(Vec3)));
        cudaCheck(cudaMalloc(&last_bsdf_pdfs, capacity * sizeof(Float)));
        cudaCheck(cudaMalloc(&specular_bounces, capacity * sizeof(uint8)));
    }

    void Free()
    {
        cudaCheck(cudaFree(rngs));
        cudaCheck(cudaFree(rays));
        cudaCheck(cudaFree(isects));
        cudaCheck(cudaFree(betas));
        cudaCheck(cudaFree(last_bsdf_pdfs));
        cudaCheck(cudaFree(specular_bounces));
    }

    void Resize(int32 capacity)
    {
        Free();
        Init(capacity);
    }
};

struct WavefrontShadowRay
{
    Ray ray;
    Float visibility;

    Vec3 Li;
    int32 pixel_index;
};

template <typename T>
struct RayQueue : CudaResource1D
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
struct RayQueues : CudaResource1D
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

struct WavefrontResources : CudaResource2D
{
    static constexpr inline int32 closest_queue_count = Materials::count;

    int32 ray_capacity;

    WavefrontPathStates path_states;
    RayQueue<int32> active;
    RayQueue<int32> next;
    RayQueues<int32, closest_queue_count> closest;

    RayQueue<int32> miss;
    RayQueue<WavefrontShadowRay> shadow;

    void Init(Point2i res)
    {
        ray_capacity = res.x * res.y;

        path_states.Init(ray_capacity);
        active.Init(ray_capacity);
        next.Init(ray_capacity);
        closest.Init(ray_capacity);
        miss.Init(ray_capacity);
        shadow.Init(ray_capacity);
    }

    void Free()
    {
        path_states.Free();
        active.Free();
        next.Free();
        closest.Free();
        miss.Free();
        shadow.Free();
    }

    void Resize(Point2i res)
    {
        ray_capacity = res.x * res.y;

        path_states.Resize(ray_capacity);
        active.Resize(ray_capacity);
        next.Resize(ray_capacity);
        closest.Resize(ray_capacity);
        miss.Resize(ray_capacity);
        shadow.Resize(ray_capacity);
    }
};

} // namespace cuwfrt
