#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuwfrt/cuda_api.h"
#include "cuwfrt/raytracer.h"

namespace cuwfrt
{

inline __GPU__ Float Luminance(Vec4 color)
{
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

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

__KERNEL__ void FilterTemporal(
    Vec4* sample_buffer,
    Point2i res,
    GBuffer prev_g_buffer,
    GBuffer g_buffer,
    Camera prev_camera,
    HistoryBuffer prev_h_buffer,
    HistoryBuffer h_buffer
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    // Find previous pixel position
    Point2i p0 = prev_camera.GetRasterPos(GetVec3(g_buffer.position[index]));
    int32 index0 = p0.x + p0.y * res.x;

    // Demomuldate albedo. We are going to filter the illumination only
    sample_buffer[index] /= g_buffer.albedo[index];

    int32 history = 0;
    Float alpha = 1;
    Vec2 moments0(0);
    Vec4 color0(0);

    // Temporal reprojection
    bool reprojectable = ValidateReprojection(prev_g_buffer, g_buffer, res, index, p0, index0);
    if (reprojectable)
    {
        alpha = 0.2f;
        moments0 = GetVec2(prev_h_buffer.moments[index0]);
        color0 = prev_h_buffer.color[index0];
        history = prev_h_buffer.moments[index0].w + 1;
    }

    // Estimate variance with exponentially averaged the luminance moments
    Float l = Luminance(sample_buffer[index]);
    Float l2 = l * l;

    Vec2 moments = Lerp(moments0, Vec2(l, l2), alpha);
    Float variance = fmax(0.0f, moments.y - moments.x * moments.x);

    // Temporal filter the illumination 4.1
    sample_buffer[index] = Lerp(color0, sample_buffer[index], alpha);

    h_buffer.moments[index] = Vec4(moments.x, moments.y, variance, history);
}

__KERNEL__ void EstimateVariance(
    Vec4* frame_buffer,
    Vec4* prev_sample_buffer,
    Vec4* sample_buffer,
    Point2i res,
    GBuffer prev_g_buffer,
    GBuffer g_buffer,
    Camera prev_camera,
    HistoryBuffer prev_h_buffer,
    HistoryBuffer h_buffer
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    int32 history = int32(h_buffer.color[index].w);
    frame_buffer[index] = ToSRGB(Vec4(history / 64.0f, history / 64.0f, history / 64.0f, 1));

    if (history > 3)
    {
        // Just go with temporally estimated variance
        return;
    }

    // Too short history length
    // We estimate variance spatially 4.2
}

} // namespace cuwfrt
