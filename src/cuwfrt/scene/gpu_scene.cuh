#pragma once

#include "cuwfrt/accel/bvh.cuh"
#include "cuwfrt/material/material.h"
#include "cuwfrt/texture/texture.cuh"

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

    // BVH
    PrimitiveIndex* bvh_primitives;
    LinearBVHNode* bvh_nodes;
};

struct GPUData
{
    GPUScene scene;
    std::vector<Texture> textures;

    void Init(const Scene* scene);
    void Free();
};

} // namespace cuwfrt
