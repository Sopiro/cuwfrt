#pragma once

#include "api.cuh"

__kernel__ void RenderGradient(Point3* pixels, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y)
    {
        return;
    }

    int index = y * res.x + x;
    pixels[index] = Point3(x / (float)res.x, y / (float)res.y, 128 / 255.0f); // Simple gradient
}
