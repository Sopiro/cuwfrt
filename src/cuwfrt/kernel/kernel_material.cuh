#pragma once

#include "cuwfrt/material/materials.h"
#include "cuwfrt/scene/gpu_scene.cuh"

#include "kernel_utils.cuh"

namespace cuwfrt
{

inline __GPU__ Material* GetMaterial(const GPUScene::Data* scene, MaterialIndex mi)
{
    return GetPolymorphicObject<Material, DiffuseLightMaterial, DiffuseMaterial, MirrorMaterial>(
        scene->materials, scene->offsets, mi
    );
}

} // namespace cuwfrt
