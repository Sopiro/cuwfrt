#pragma once

#include "cuwfrt/scene/scene.cuh"
#include <filesystem>

namespace cuwfrt
{

void SetFallbackMaterial(MaterialIndex material_index);
void LoadGLTF(Scene& scene, std::filesystem::path filename, const Transform& transform = identity);

} // namespace cuwfrt
