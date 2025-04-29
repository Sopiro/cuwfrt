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
    constexpr Float sigma_l = 2; // 4 in paper

    Float w_z = abs(z_p - z_q) / (sigma_z * abs(Dot(dzdp, Vec2(p - q))) + 1e-9);

    Float w_n = powf(fmax(0.0f, Dot(n_p, n_q)), sigma_n);

    Float w_l = abs(l_p - l_q) / (sigma_l * sqrt(var_p) + 1e-9);

    return w_n * exp(-(w_z + w_l));
}

__KERNEL__ void PrepareDenoise(Vec4* in_buffer, Vec4* out_buffer, GBuffer g_buffer, HistoryBuffer h_buffer, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    // Approximate depth derivatives
    Float dzdx = 0;
    Float dzdy = 0;

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

    // Demomuldate albedo. We are going to filter the illumination only
    if (g_buffer.albedo[index] != Vec4(0))
    {
        out_buffer[index] = in_buffer[index] / g_buffer.albedo[index];

        g_buffer.albedo[index].w = 1;
    }
    else
    {
        g_buffer.albedo[index] = in_buffer[index];
        g_buffer.albedo[index].w = 0;

        out_buffer[index] = Vec4(0);
    }

    g_buffer.dzdp[index] = Vec2(dzdx, dzdy);
}

__KERNEL__ void FilterTemporal(
    Vec4* buffer,
    Point2i res,
    GBuffer prev_g_buffer,
    GBuffer g_buffer,
    HistoryBuffer prev_h_buffer,
    HistoryBuffer h_buffer,
    Camera prev_camera,
    bool consistent
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    // Find previous pixel position
    Point2i p0 = prev_camera.GetRasterPos(GetVec3(g_buffer.position[index]));
    int32 index0 = p0.x + p0.y * res.x;

    int32 history = 1;
    Float alpha = 1;
    Vec2 moments0(0);
    Vec4 color0(0);

    // Temporal reprojection
    if (consistent && ValidateReprojection(prev_g_buffer, g_buffer, res, index, p0, index0))
    {
        moments0 = GetVec2(prev_h_buffer.moments[index0]);
        color0 = prev_h_buffer.color[index0];
        history = prev_h_buffer.moments[index0].w + 1;
        alpha = 0.2f;
    }

    // Estimate variance with exponentially averaged the luminance moments
    Float l = Luminance(buffer[index]);
    Float l2 = l * l;

    Vec2 moments = Lerp(moments0, Vec2(l, l2), alpha);
    Float variance = fmax(0.0f, moments.y - moments.x * moments.x);

    // Temporal filter the illumination 4.1
    buffer[index] = Lerp(color0, buffer[index], alpha);

    h_buffer.moments[index] = Vec4(moments.x, moments.y, variance, history);
}

__KERNEL__ void EstimateVariance(GBuffer g_buffer, HistoryBuffer h_buffer, HistoryBuffer out_h_buffer, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;
    const int32 index = y * res.x + x;

    out_h_buffer.moments[index] = h_buffer.moments[index];

    int32 history = int32(h_buffer.moments[index].w);
    if (history > 3)
    {
        // Just go with temporally estimated variance
        return;
    }

    // Too short history length
    // We estimate variance spatially 4.2

    Point2i p(x, y);
    Float z_p = g_buffer.position[index].w;
    Vec2 dzdp = g_buffer.dzdp[index];
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

    out_h_buffer.moments[index].z = variance;
}

__KERNEL__ void FilterVariance(HistoryBuffer h_buffer, HistoryBuffer out_h_buffer, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;
    const int32 index = y * res.x + x;

    //  Eq. 5 Filter variance using 3x3 gaussian kernel

    // clang-format off
    constexpr int32 s = 3;
    constexpr Float gaussian3x3[] =
    {
        1 / 16.0f, 1 / 8.0f, 1 / 16.0f,
        1 / 8.0f, 1 / 4.0f, 1 / 8.0f,
        1 / 16.0f, 1 / 8.0f, 1 / 16.0f,
    };
    // clang-format on

    Float variance = 0;

    constexpr int32 r = 1;
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
            int32 kernel_index = (j + r) * s + (i + r);
            variance += gaussian3x3[kernel_index] * h_buffer.moments[index_q].z;
        }
    }

    out_h_buffer.moments[index].z = variance;
}

__KERNEL__ void FilterSpatial(
    Vec4* in_sample_buffer,
    Vec4* out_sample_buffer,
    Point2i res,
    int32 step,
    GBuffer g_buffer,
    HistoryBuffer in_h_buffer,
    HistoryBuffer out_h_buffer
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;
    const int32 index = y * res.x + x;

    if (g_buffer.position[index].w <= 0)
    {
        return;
    }

    Point2i p(x, y);
    Float z_p = g_buffer.position[index].w;
    Vec2 dzdp = g_buffer.dzdp[index];
    Vec3 n_p = GetVec3(g_buffer.normal[index]);
    Float l_p = Luminance(in_sample_buffer[index]);

    Float var_p = in_h_buffer.moments[index].z;

    // clang-format off
    constexpr int32 s = 5;
    constexpr Float h[s * s] =
    {
        0.00390625f, 0.015625f,   0.0234375f,  0.015625f,   0.00390625f,
        0.015625f,   0.0625f,     0.09375f,    0.0625f,     0.015625f,
        0.0234375f,  0.09375f,    0.140625f,   0.09375f,    0.0234375f,
        0.015625f,   0.0625f,     0.09375f,    0.0625f,     0.015625f,
        0.00390625f, 0.015625f,   0.0234375f,  0.015625f,   0.00390625f
    };
    // clang-format on

    Float sum_weights = 0;
    Vec4 sum_l(0);

    Float sum_weights2 = 0;
    Float sum_var(0);

    // a-trous wavelet filter
    constexpr int32 r = 2;
    for (int32 j = -r; j <= r; ++j)
    {
        for (int32 i = -r; i <= r; ++i)
        {
            Point2i q(x + i * step, y + j * step);
            if (q.x < 0 || q.x >= res.x || q.y < 0 || q.y >= res.y)
            {
                continue;
            }

            int32 index_q = q.x + q.y * res.x;

            Float z_q = g_buffer.position[index_q].w;
            Vec3 n_q = GetVec3(g_buffer.normal[index_q]);
            Float l_q = Luminance(in_sample_buffer[index_q]);

            int32 kernel_index = (j + r) * s + (i + r);
            Float w = h[kernel_index] * EdgeStoppingWeight(p, q, z_p, z_q, dzdp, n_p, n_q, l_p, l_q, var_p);

            sum_weights += w;
            sum_l += w * in_sample_buffer[index_q];

            sum_weights2 += w * w;
            sum_var += w * w * in_h_buffer.moments[index_q].z;
        }
    }

    sum_l /= sum_weights;
    sum_var /= sum_weights2;

    out_sample_buffer[index] = sum_l;
    out_h_buffer.moments[index].z = sum_var;

    if (step == 1)
    {
        in_h_buffer.color[index] = sum_l;
    }
}

__KERNEL__ void FinalizeDenoise(Vec4* frame_buffer, Vec4* filtered_buffer, Point2i res, GBuffer g_buffer)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;
    const int32 index = y * res.x + x;

    // Remodulate albedo
    Vec4 albedo = g_buffer.albedo[index];
    if (albedo.w)
    {
        frame_buffer[index] = ToSRGB(filtered_buffer[index] * albedo);
    }
    else
    {
        frame_buffer[index] = ToSRGB(albedo);

        g_buffer.albedo[index] = Vec4(0);
    }
}

} // namespace cuwfrt
