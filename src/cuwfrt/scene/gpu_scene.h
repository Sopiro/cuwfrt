#pragma once

#include <cuda_runtime.h>

#include "cuwfrt/accel/bvh.h"
#include "cuwfrt/material/material.h"
#include "cuwfrt/texture/texture.h"

namespace cuwfrt
{

class Scene;

struct GPUScene
{
    // Textures
    cudaTextureObject_t* __restrict__ tex_objs;

    // Materials
    uint8* __restrict__ materials;
    int32* __restrict__ offsets;

    // Scene primitives
    Point3* __restrict__ positions;
    Vec3* __restrict__ normals;
    Vec3* __restrict__ tangents;
    Vec2* __restrict__ texcoords;
    Vec3i* __restrict__ indices;
    MaterialIndex* __restrict__ material_indices;

    // Area lights
    PrimitiveIndex* __restrict__ light_indices;
    int32 light_count;

    // BVH
    PrimitiveIndex* __restrict__ bvh_primitives;
    LinearBVHNode* __restrict__ bvh_nodes;
};

struct GPUResources
{
    GPUScene scene;
    std::vector<Texture> textures;

    void Init(const Scene* scene);
    void Free();
};

} // namespace cuwfrt
