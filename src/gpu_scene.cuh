#pragma once

#include "indices.h"
#include "material.h"

namespace cuwfrt
{

class Scene;

struct GPUScene
{
    Material* materials = nullptr;

    Vec3* positions = nullptr;
    Vec3* normals = nullptr;
    Vec3* tangents = nullptr;
    Vec2* texcoords = nullptr;
    MaterialIndex* material_indices = nullptr;
    Vec3i* indices = nullptr;
    int32* light_indices = nullptr;

    void Init(Scene* scene);
    void Free();
};

} // namespace cuwfrt
