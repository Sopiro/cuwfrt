#pragma once

#include "common.h"
#include "cuda_error.h"

namespace cuwfrt
{

struct HistoryBuffer
{
    Vec4* color;
    Vec2* moments;

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
    Vec4* position; // w: camera space linear depth
    Vec4* normal;
    Vec4* albedo;

    void Init(int32 capacity)
    {
        cudaCheck(cudaMalloc(&position, capacity * sizeof(decltype(*position))));
        cudaCheck(cudaMalloc(&normal, capacity * sizeof(decltype(*normal))));
        cudaCheck(cudaMalloc(&albedo, capacity * sizeof(decltype(*albedo))));
    }

    void Free()
    {
        cudaCheck(cudaFree(position));
        cudaCheck(cudaFree(normal));
        cudaCheck(cudaFree(albedo));
    }

    void Resize(int32 capacity)
    {
        cudaCheck(cudaFree(position));
        cudaCheck(cudaFree(normal));
        cudaCheck(cudaFree(albedo));
        cudaCheck(cudaMalloc(&position, capacity * sizeof(decltype(*position))));
        cudaCheck(cudaMalloc(&normal, capacity * sizeof(decltype(*normal))));
        cudaCheck(cudaMalloc(&albedo, capacity * sizeof(decltype(*albedo))));
    }
};

} // namespace cuwfrt
