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
    cudaTextureObject_t* tex_objs = nullptr;

    // Materials
    uint8* materials = nullptr;
    int32* offsets = nullptr;

    // Scene primitives
    Point3* positions = nullptr;
    Vec3* normals = nullptr;
    Vec3* tangents = nullptr;
    Vec2* texcoords = nullptr;
    Vec3i* indices = nullptr;
    MaterialIndex* material_indices = nullptr;

    // Area lights
    PrimitiveIndex* light_indices = nullptr;
    int32 light_count = 0;

    // BVH
    PrimitiveIndex* bvh_primitives = nullptr;
    LinearBVHNode* bvh_nodes = nullptr;
};

struct GPUData
{
    GPUScene scene;
    std::vector<Texture> textures;

    void Init(const Scene* scene);
    void Free();
};

} // namespace cuwfrt
