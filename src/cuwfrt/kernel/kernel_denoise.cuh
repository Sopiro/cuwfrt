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

inline __GPU__ Float EdgeStoppingWeight(Point2i p, Point2i q, Float z_p, Float z_q, Vec2 dzdp, Vec3 n_p, Vec3 n_q)
{
    constexpr Float sigma_z = 1;
    constexpr Float sigma_n = 128;

    Float w_z = abs(z_p - z_q) / (sigma_z * abs(Dot(dzdp, Vec2(p - q))) + 1e-9);

    Float w_n = powf(fmax(0.0f, Dot(n_p, n_q)), sigma_n);

    return w_n * exp(-w_z);
}

inline __GPU__ Float
EdgeStoppingWeight(Point2i p, Point2i q, Float z_p, Float z_q, Vec2 dzdp, Vec3 n_p, Vec3 n_q, Float l_p, Float l_q, Float var_p)
{
    constexpr Float sigma_z = 1;
    constexpr Float sigma_n = 128;
    constexpr Float sigma_l = 4;

    Float w_z = abs(z_p - z_q) / (sigma_z * abs(Dot(dzdp, Vec2(p - q))) + 1e-9);

    Float w_n = powf(fmax(0.0f, Dot(n_p, n_q)), sigma_n);

    Float w_l = abs(l_p - l_q) / (sigma_l * sqrt(var_p) + 1e-9);

    return w_n * exp(-(w_z + w_l));
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
        moments0 = GetVec2(prev_h_buffer.moments[index0]);
        color0 = prev_h_buffer.color[index0];
        history = prev_h_buffer.moments[index0].w + 1;
        alpha = 1.0f / history;
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

    int32 history = int32(h_buffer.moments[index].w);

    if (history > 3)
    {
        frame_buffer[index] = ToSRGB(Vec4(h_buffer.moments[index].z, h_buffer.moments[index].z, h_buffer.moments[index].z, 1));
        // Just go with temporally estimated variance
        return;
    }

    // Too short history length
    // We estimate variance spatially 4.2

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

    Point2i p(x, y);
    Float z_p = g_buffer.position[index].w;
    Vec2 dzdp(dzdx, dzdy);
    Vec3 n_p = GetVec3(g_buffer.normal[index]);

    Float sum_weights = 0;
    Vec2 sum_moments(0);

    const int32 r = 3;
    for (int32 j = -r; j <= r; ++j)
    {
        for (int32 i = -r; i <= r; ++i)
        {
            Point2i q(x + i, y + j);
            if (q.x < 0 || q.x >= res.x || q.y < 0 || q.y >= res.y)
            {
                continue;
            }

            int32 index_q = q.x + q.y * res.x;

            Float z_q = g_buffer.position[index_q].w;
            Vec3 n_q = GetVec3(g_buffer.normal[index_q]);

            Float w = EdgeStoppingWeight(p, q, z_p, z_q, dzdp, n_p, n_q);

            sum_weights += w;
            sum_moments += w * GetVec2(h_buffer.moments[index_q]);
        }
    }

    sum_moments /= sum_weights;

    // Spatially estimated variance
    Float variance = fmax(0.0f, sum_moments.y - sum_moments.x * sum_moments.x);

    h_buffer.moments[index].z = variance;

    frame_buffer[index] = ToSRGB(Vec4(variance, variance, variance, 1));
}

} // namespace cuwfrt
