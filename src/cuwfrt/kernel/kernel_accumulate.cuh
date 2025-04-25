#pragma once

#include <device_launch_parameters.h>

#include "cuwfrt/raytracer.h"

namespace cuwfrt
{

__KERNEL__ void Accumulate(
    Vec4* __restrict__ sample_buffer,
    Vec4* __restrict__ accumulation_buffer,
    Vec4* __restrict__ frame_buffer,
    Point2i res,
    int32 spp
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    accumulation_buffer[index] *= spp;
    accumulation_buffer[index] += sample_buffer[index];
    accumulation_buffer[index] /= spp + 1;

    frame_buffer[index] = ToSRGB(accumulation_buffer[index]);
}

} // namespace cuwfrt
