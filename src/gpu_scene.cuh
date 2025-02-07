#pragma once

#include "bvh.cuh"
#include "indices.h"
#include "material.h"
#include "texture.cuh"

namespace cuwfrt
{

class Scene;

struct GPUScene
{
    struct Data
    {
        cudaTextureObject_t* tex_objs = nullptr;
        Material* materials = nullptr;

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
