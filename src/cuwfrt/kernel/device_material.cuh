#pragma once

#include "cuwfrt/material/materials.h"
#include "cuwfrt/scene/gpu_scene.h"

#include "device_utils.cuh"

namespace cuwfrt
{

inline __GPU__ Material* GetMaterial(const GPUScene* scene, PrimitiveIndex prim)
{
    MaterialIndex mi = scene->material_indices[prim];
    return GetPolymorphicObject<
        Material, DiffuseLightMaterial, DiffuseMaterial, MirrorMaterial, DielectricMaterial, MetallicRoughnessMaterial>(
        scene->materials, scene->offsets, mi
    );
}

} // namespace cuwfrt
