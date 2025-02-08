#pragma once

#include "kernel_utils.cuh"

#include "gpu_scene.cuh"
#include "materials.h"

namespace cuwfrt
{

inline __GPU__ Material* GetMaterial(const GPUScene::Data* scene, MaterialIndex mi)
{
    return GetPolymorphicObject<Material, DiffuseMaterial, DiffuseLightMaterial>(scene->materials, scene->offsets, mi);
}

} // namespace cuwfrt
