#pragma once

#include "cuwfrt/scene/scene.cuh"
#include <filesystem>

namespace cuwfrt
{

void SetLoaderFlipNormal(bool flip_normal);
void SetLoaderFlipTexcoord(bool flip_texcoord);
void SetLoaderUseForceFallbackMaterial(bool force_use_fallback_material);
void SetLoaderFallbackMaterial(MaterialIndex material_index);

void LoadModel(Scene& scene, std::filesystem::path filename, const Transform& transform = identity);
void LoadGLTF(Scene& scene, std::filesystem::path filename, const Transform& transform = identity);
void LoadOBJ(Scene& scene, std::filesystem::path filename, const Transform& transform = identity);

} // namespace cuwfrt
