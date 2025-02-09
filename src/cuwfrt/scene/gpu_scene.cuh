#pragma once

#include "cuwfrt/accel/bvh.cuh"
#include "cuwfrt/material/material.h"
#include "cuwfrt/texture/texture.cuh"

namespace cuwfrt
{

class Scene;

struct GPUScene
{
    struct Data
    {
        // Textures
        cudaTextureObject_t* tex_objs = nullptr;

        // Materials
        uint8* materials = nullptr;
        int32* offsets = nullptr;

        // Scene primitives
        Vec3* positions = nullptr;
        Vec3* normals = nullptr;
        Vec3* tangents = nullptr;
        Vec2* texcoords = nullptr;
        MaterialIndex* material_indices = nullptr;
        Vec3i* indices = nullptr;
        int32* light_indices = nullptr;

        // BVH
        PrimitiveIndex* bvh_primitives;
        LinearBVHNode* bvh_nodes;
    } data;

    std::vector<Texture> textures;

    void Init(const Scene* scene);
    void Free();
};

} // namespace cuwfrt
