#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuwfrt/cuda_api.h"
#include "cuwfrt/raytracer.h"

namespace cuwfrt
{

inline __GPU__ bool ValidateReprojection(
    const GBuffer& prev_g_buffer, const GBuffer& g_buffer, Point2i res, int32 index, Point2i p0, int32 index0
)
{
    // Test screen bounds
    if (p0.x < 0 || p0.x >= res.x || p0.y < 0 || p0.y >= res.y)
    {
        return false;
    }

    // Test intersection
    if (prev_g_buffer.position[index0].w <= 0 || g_buffer.position[index].w <= 0)
    {
        return false;
    }

    // Test normal differences
    if (Dot(GetVec3(prev_g_buffer.normal[index0]), GetVec3(g_buffer.normal[index])) < 0.9f)
    {
        return false;
    }

    // Test position differences
    if (Dist2(GetVec3(prev_g_buffer.position[index0]), GetVec3(g_buffer.position[index])) > 0.001f)
    {
        return false;
    }

    return true;
}

__KERNEL__ void PrepareDenoise(
    Vec4* frame_buffer,
    Vec4* prev_sample_buffer,
    Vec4* sample_buffer,
    Point2i res,
    GBuffer prev_g_buffer,
    GBuffer g_buffer,
    Camera prev_camera
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    // Approximate depth derivatives
    Float dzdx, dzdy;
    if (x > 0 && x < res.x - 1)
    {
        Float z0 = g_buffer.position[y * res.x + (x - 1)].w;
        Float z1 = g_buffer.position[y * res.x + (x + 1)].w;
        dzdx = 0.5f * (z1 - z0);
    }
    else if (x < res.x - 1)
    {
        Float z = g_buffer.position[index].w;
        Float z1 = g_buffer.position[y * res.x + (x + 1)].w;
        dzdx = z1 - z;
    }

    if (y > 0 && y < res.y - 1)
    {
        Float z1 = g_buffer.position[(y - 1) * res.x + x].w;
        Float z0 = g_buffer.position[(y + 1) * res.x + x].w;
        dzdy = 0.5f * (z0 - z1);
    }
    else if (y < res.y - 1)
    {
        Float z = g_buffer.position[index].w;
        Float z0 = g_buffer.position[(y + 1) * res.x + x].w;
        dzdy = z0 - z;
    }

    // Find previous pixel position
    const Point2i p0 = prev_camera.GetRasterPos(GetVec3(g_buffer.position[index]));
    const int32 index0 = p0.x + p0.y * res.x;

    // Demomuldate albedo. We are going to filter the illumination only
    // sample_buffer[index] /= g_buffer.albedo[index];

    Float alpha = 0.2f;

    // Temporal reprojection
    bool reprojectable = ValidateReprojection(prev_g_buffer, g_buffer, res, index, p0, index0);
    if (reprojectable)
    {
        sample_buffer[index] = Lerp(prev_sample_buffer[index0], sample_buffer[index], alpha);
    }

    frame_buffer[index] = ToSRGB(sample_buffer[index]);
}

} // namespace cuwfrt
