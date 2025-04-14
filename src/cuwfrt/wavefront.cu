#include "wavefront.h"

#include "cuwfrt/cuda_error.cuh"

namespace cuwfrt
{

void WavefrontResources::Init(Point2i res)
{
    ray_capacity = res.x * res.y;

    size_t ray_buffer_size = sizeof(WavefrontRay) * ray_capacity;
    size_t miss_ray_buffer_size = sizeof(WavefrontMissRay) * ray_capacity;
    size_t shadow_ray_buffer_size = sizeof(WavefrontShadowRay) * ray_capacity;

    cudaCheck(cudaMalloc(&rays_active, ray_buffer_size));
    cudaCheck(cudaMalloc(&rays_next, ray_buffer_size));

    for (size_t i = 0; i < Materials::count; ++i)
    {
        cudaCheck(cudaMalloc(&rays_closest.rays[i], ray_buffer_size));
        cudaCheck(cudaMalloc(&rays_closest.ray_counts[i], sizeof(int32)));
    }

    cudaCheck(cudaMalloc(&miss_rays, miss_ray_buffer_size));
    cudaCheck(cudaMalloc(&shadow_rays, shadow_ray_buffer_size));

    cudaCheck(cudaMalloc(&active_ray_count, sizeof(int32)));
    cudaCheck(cudaMalloc(&next_ray_count, sizeof(int32)));

    cudaCheck(cudaMalloc(&miss_ray_count, sizeof(int32)));
    cudaCheck(cudaMalloc(&shadow_ray_count, sizeof(int32)));
}

void WavefrontResources::Free()
{
    cudaCheck(cudaFree(rays_active));
    cudaCheck(cudaFree(rays_next));

    for (size_t i = 0; i < Materials::count; ++i)
    {
        cudaCheck(cudaFree(rays_closest.rays[i]));
        cudaCheck(cudaFree(rays_closest.ray_counts[i]));
    }

    cudaCheck(cudaFree(miss_rays));
    cudaCheck(cudaFree(shadow_rays));

    cudaCheck(cudaFree(active_ray_count));
    cudaCheck(cudaFree(next_ray_count));

    cudaCheck(cudaFree(miss_ray_count));
    cudaCheck(cudaFree(shadow_ray_count));
}

void WavefrontResources::Resize(Point2i res)
{
    ray_capacity = res.x * res.y;
    cudaCheck(cudaFree(rays_active));
    cudaCheck(cudaFree(rays_next));

    for (size_t i = 0; i < Materials::count; ++i)
    {
        cudaCheck(cudaFree(rays_closest.rays[i]));
    }

    cudaCheck(cudaFree(miss_rays));
    cudaCheck(cudaFree(shadow_rays));

    size_t ray_buffer_size = sizeof(WavefrontRay) * ray_capacity;
    size_t miss_ray_buffer_size = sizeof(WavefrontMissRay) * ray_capacity;
    size_t shadow_ray_buffer_size = sizeof(WavefrontShadowRay) * ray_capacity;

    cudaCheck(cudaMalloc(&rays_active, ray_buffer_size));
    cudaCheck(cudaMalloc(&rays_next, ray_buffer_size));

    for (size_t i = 0; i < Materials::count; ++i)
    {
        cudaCheck(cudaMalloc(&rays_closest.rays[i], ray_buffer_size));
    }

    cudaCheck(cudaMalloc(&miss_rays, miss_ray_buffer_size));
    cudaCheck(cudaMalloc(&shadow_rays, shadow_ray_buffer_size));
}

} // namespace cuwfrt
