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
    constexpr Float sigma_l = 4; // 4 in paper

    Float w_z = abs(z_p - z_q) / (sigma_z * abs(Dot(dzdp, Vec2(p - q))) + 1e-9);

    Float w_n = powf(fmax(0.0f, Dot(n_p, n_q)), sigma_n);

    Float w_l = abs(l_p - l_q) / (sigma_l * sqrt(var_p) + 1e-9);

    return w_n * exp(-(w_z + w_l));
}

__KERNEL__ void PrepareDenoise(
    Vec4* in_buffer, Vec4* out_buffer, GBuffer g_buffer, HistoryBuffer h_buffer, Camera prev_camera, Point2i res
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    Float z = g_buffer.position[index].w;

    // Approximate depth derivatives
    Float dzdx, dzdy;

    if (x == 0) // Left boundary
    {
        // Forward differences
        Float z1 = g_buffer.position[y * res.x + (x + 1)].w;
        dzdx = z1 - z;
    }
    else if (x == res.x - 1) // Right boundary
    {
        // Backward differences
        Float z0 = g_buffer.position[y * res.x + (x - 1)].w;
        dzdx = z - z0;
    }
    else // Totally inside
    {
        Float z0 = g_buffer.position[y * res.x + (x - 1)].w;
        Float z1 = g_buffer.position[y * res.x + (x + 1)].w;
        dzdx = 0.5f * (z1 - z0);
    }

    // Same goes for y axis
    if (y == 0)
    {
        Float z1 = g_buffer.position[(y + 1) * res.x + x].w;
        dzdy = z1 - z;
    }
    else if (y == res.y - 1)
    {
        Float z0 = g_buffer.position[(y - 1) * res.x + x].w;
        dzdy = z - z0;
    }
    else
    {
        Float z0 = g_buffer.position[(y - 1) * res.x + x].w;
        Float z1 = g_buffer.position[(y + 1) * res.x + x].w;
        dzdy = 0.5f * (z1 - z0);
    }

    // Demomuldate albedo. We are going to filter the illumination only
    if (g_buffer.albedo[index].w > 0)
    {
        out_buffer[index] = in_buffer[index] / g_buffer.albedo[index];
    }
    else
    {
        g_buffer.albedo[index] = in_buffer[index];
        g_buffer.albedo[index].w = 0;

        out_buffer[index] = Vec4(0);
    }

    g_buffer.dzdp[index] = Vec2(dzdx, dzdy);

    g_buffer.motion[index] = prev_camera.GetRasterPos(GetVec3(g_buffer.position[index]));
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
    Point2i p0 = g_buffer.motion[index];
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
    HistoryBuffer out_h_buffer,
    int32 spp
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

    // Scale variance by sqrt(spp) to steer the edge stopping strength
    Float var_p = sqrtf(spp) * in_h_buffer.moments[index].z;

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
            sum_var += w * w * in_h_buffer.moments[index_q].z;
        }
    }

    sum_l /= sum_weights;
    sum_var /= sum_weights * sum_weights;

    out_sample_buffer[index] = sum_l;
    out_h_buffer.moments[index].z = sum_var;

    if (step == 1)
    {
        in_h_buffer.color[index] = sum_l;
    }
}

__KERNEL__ void FinalizeDenoise(Vec4* filtered_buffer, Point2i res, GBuffer g_buffer)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;
    const int32 index = y * res.x + x;

    // Remodulate albedo
    Vec4 albedo = g_buffer.albedo[index];
    if (albedo.w > 0)
    {
        filtered_buffer[index] = filtered_buffer[index] * albedo;
    }
    else
    {
        filtered_buffer[index] = albedo;
        g_buffer.albedo[index] = Vec4(0);
    }
}

// https://advances.realtimerendering.com/s2014/epic/TemporalAA.pptx
__KERNEL__ void TemporalAntiAliasing(
    Vec4* prev_frame_buffer,
    Vec4* out_frame_buffer,
    Vec4* filtered_buffer,
    Point2i res,
    GBuffer prev_g_buffer,
    GBuffer g_buffer,
    bool consistent
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;
    const int32 index = y * res.x + x;

    Float alpha = 1.0f;
    Vec4 color0;
    Vec4 color1 = ToSRGB(filtered_buffer[index]);

    Point2i p0 = g_buffer.motion[index];
    int32 index0 = p0.x + p0.y * res.x;

    if (consistent && p0.x >= 0 && p0.x < res.x && p0.y >= 0 && p0.y < res.y && g_buffer.position[index].w > 0)
    {
        color0 = RGBtoYUV(ToLinearRGB(prev_frame_buffer[index0]));

        Float length = sqrtf(Dist2(p0, Point2i(x, y)));
        const Float sigma = 2.0f;
        Float w = exp(-length * sigma);
        alpha = Lerp(1.0f, 0.1f, w);

        Vec2 uv(Float(x) / res.x, Float(y) / res.y);
        Vec2 d(1.0f / res.x, 1.0f / res.y);

        // Clamp color in Y'UV space
        Vec4 c1 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(-d.x, d.y), TexCoordFilter::clamp));
        Vec4 c2 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(0, d.y), TexCoordFilter::clamp));
        Vec4 c3 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(d.x, d.y), TexCoordFilter::clamp));
        Vec4 c4 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(-d.x, 0), TexCoordFilter::clamp));
        Vec4 c5 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(d.x, 0), TexCoordFilter::clamp));
        Vec4 c6 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(-d.x, -d.y), TexCoordFilter::clamp));
        Vec4 c7 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(0, -d.y), TexCoordFilter::clamp));
        Vec4 c8 = RGBtoYUV(SampleTexture(filtered_buffer, res, uv + Vec2(d.x, -d.y), TexCoordFilter::clamp));

        Vec4 min;
        min.x = Min(c1.x, c2.x, c3.x, c4.x, c5.x, c6.x, c7.x, c8.x);
        min.y = Min(c1.y, c2.y, c3.y, c4.y, c5.y, c6.y, c7.y, c8.y);
        min.z = Min(c1.z, c2.z, c3.z, c4.z, c5.z, c6.z, c7.z, c8.z);
        min.w = 1.0f;

        Vec4 max;
        max.x = Max(c1.x, c2.x, c3.x, c4.x, c5.x, c6.x, c7.x, c8.x);
        max.y = Max(c1.y, c2.y, c3.y, c4.y, c5.y, c6.y, c7.y, c8.y);
        max.z = Max(c1.z, c2.z, c3.z, c4.z, c5.z, c6.z, c7.z, c8.z);
        max.w = 1.0f;

        color0 = Max(Min(color0, max), min);
        color0 = ToSRGB(YUVtoRGB(color0));
    }

    out_frame_buffer[index] = Lerp(color0, color1, alpha);
}

} // namespace cuwfrt
