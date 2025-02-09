#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"

namespace cuwfrt
{

__KERNEL__ void RenderGradient(Vec4* pixels, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    pixels[index] = Vec4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

} // namespace cuwfrt
