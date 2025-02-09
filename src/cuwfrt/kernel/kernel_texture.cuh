#pragma once

#include "cuwfrt/scene/gpu_scene.cuh"

#include "kernel_utils.cuh"

namespace cuwfrt
{

inline __GPU__ Vec3 SampleTexture(const GPUScene::Data* scene, TextureIndex ti, Point2 uv)
{
    float4 tex_color = tex2D<float4>(scene->tex_objs[ti], uv.x, 1 - uv.y);
    return Vec3(tex_color.x, tex_color.y, tex_color.z);
}

} // namespace cuwfrt
