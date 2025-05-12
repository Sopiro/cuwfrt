#pragma once

#include "common.h"
#include "cuda_error.h"

namespace cuwfrt
{

struct HistoryBuffer
{
    Vec4* color;   // 1th filtered illumination
    Vec4* moments; // x: l, y: l^2, z: variance, w: history length

    void Init(int32 capacity)
    {
        cudaCheck(cudaMalloc(&color, capacity * sizeof(decltype(*color))));
        cudaCheck(cudaMalloc(&moments, capacity * sizeof(decltype(*moments))));
    }

    void Free()
    {
        cudaCheck(cudaFree(color));
        cudaCheck(cudaFree(moments));
    }

    void Resize(int32 capacity)
    {
        cudaCheck(cudaFree(color));
        cudaCheck(cudaFree(moments));
        cudaCheck(cudaMalloc(&color, capacity * sizeof(decltype(*color))));
        cudaCheck(cudaMalloc(&moments, capacity * sizeof(decltype(*moments))));
    }
};

struct GBuffer
{
    Vec4* position; // xyz: world space position        w: camera space linear depth
    Vec4* normal;   // xyz: world space shading normal  w: primitive index
    Vec4* albedo;   // directional-hemispherical reflectance
    Vec2* dzdp;     // screen space depth derivates (dzdx, dzdy)
    Vec2i* motion;  // screen space motion vector (pixel index in previous frame)

    void Init(int32 capacity)
    {
        cudaCheck(cudaMalloc(&position, capacity * sizeof(decltype(*position))));
        cudaCheck(cudaMalloc(&normal, capacity * sizeof(decltype(*normal))));
        cudaCheck(cudaMalloc(&albedo, capacity * sizeof(decltype(*albedo))));
        cudaCheck(cudaMalloc(&dzdp, capacity * sizeof(decltype(*dzdp))));
        cudaCheck(cudaMalloc(&motion, capacity * sizeof(decltype(*motion))));
    }

    void Free()
    {
        cudaCheck(cudaFree(position));
        cudaCheck(cudaFree(normal));
        cudaCheck(cudaFree(albedo));
        cudaCheck(cudaFree(dzdp));
        cudaCheck(cudaFree(motion));
    }

    void Resize(int32 capacity)
    {
        cudaCheck(cudaFree(position));
        cudaCheck(cudaFree(normal));
        cudaCheck(cudaFree(albedo));
        cudaCheck(cudaFree(dzdp));
        cudaCheck(cudaFree(motion));
        cudaCheck(cudaMalloc(&position, capacity * sizeof(decltype(*position))));
        cudaCheck(cudaMalloc(&normal, capacity * sizeof(decltype(*normal))));
        cudaCheck(cudaMalloc(&albedo, capacity * sizeof(decltype(*albedo))));
        cudaCheck(cudaMalloc(&dzdp, capacity * sizeof(decltype(*dzdp))));
        cudaCheck(cudaMalloc(&motion, capacity * sizeof(decltype(*motion))));
    }
};

} // namespace cuwfrt
