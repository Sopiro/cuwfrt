#pragma once

#include <cuda_runtime.h>

#include "cuwfrt/scene/gpu_scene.h"
#include "device_utils.cuh"

namespace cuwfrt
{

inline __GPU__ Vec3 SampleTexture(const GPUScene* scene, TextureIndex ti, Point2 uv)
{
    float4 tex_color = tex2D<float4>(scene->tex_objs[ti], uv.x, 1 - uv.y);
    return Vec3(tex_color.x, tex_color.y, tex_color.z);
}

enum TexCoordFilter
{
    repeat,
    clamp,
};

inline __GPU__ void FilterTexCoord(int32* u, int32* v, int32 width, int32 height, TexCoordFilter texcoord_filter)
{
    switch (texcoord_filter)
    {
    case TexCoordFilter::repeat:
    {
        *u = *u >= 0 ? *u % width : width - (-(*u) % width) - 1;
        *v = *v >= 0 ? *v % height : height - (-(*v) % height) - 1;
    }
    break;
    case TexCoordFilter::clamp:
    {
        *u = Clamp(*u, 0, width - 1);
        *v = Clamp(*v, 0, height - 1);
    }
    break;
    default:
        WakAssert(false);
        break;
    }
}

template <typename T>
inline __GPU__ T SampleTexture(const T* image, Point2i res, Point2 uv, TexCoordFilter texcoord_filter = repeat)
{
#if 0
    // Nearest sampling
    Float w = uv.x * res.x + 0.5f;
    Float h = uv.y * res.y + 0.5f;

    int32 i = int32(w);
    int32 j = int32(h);

    FilterTexCoord(&i, &j);

    return image[i + j * res.x];
#else
    // Bilinear sampling
    Float w = uv.x * res.x + 0.5f;
    Float h = uv.y * res.y + 0.5f;

    int32 i0 = int32(w), i1 = int32(w) + 1;
    int32 j0 = int32(h), j1 = int32(h) + 1;

    FilterTexCoord(&i0, &j0, res.x, res.y, texcoord_filter);
    FilterTexCoord(&i1, &j1, res.x, res.y, texcoord_filter);

    Float fu = w - std::floor(w);
    Float fv = h - std::floor(h);

    T v00 = image[i0 + j0 * res.x], v10 = image[i1 + j0 * res.x];
    T v01 = image[i0 + j1 * res.x], v11 = image[i1 + j1 * res.x];

    // clang-format off
return (1-fu) * (1-fv) * v00 + (1-fu) * (fv) * v01 +
       (  fu) * (1-fv) * v10 + (  fu) * (fv) * v11;
    // clang-format on
#endif
}

} // namespace cuwfrt
