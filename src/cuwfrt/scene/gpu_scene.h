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
    cudaTextureObject_t* tex_objs;

    // Materials
    uint8* materials;
    int32* offsets;

    // Scene primitives
    Point3* positions;
    Vec3* normals;
    Vec3* tangents;
    Vec2* texcoords;
    Vec3i* indices;
    MaterialIndex* material_indices;

    // Area lights
    PrimitiveIndex* light_indices;
    int32 light_count;

    // BVH
    PrimitiveIndex* bvh_primitives;
    LinearBVHNode* bvh_nodes;
};

struct GPUResources
{
    GPUScene scene;
    std::vector<Texture> textures;

    void Init(const Scene* scene);
    void Free();
};

} // namespace cuwfrt
