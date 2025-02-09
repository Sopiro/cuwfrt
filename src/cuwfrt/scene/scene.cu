#pragma once

#include "cuwfrt/mesh.h"
#include "cuwfrt/triangle.h"

#include "scene.cuh"

namespace cuwfrt
{

TextureIndex Scene::AddTexture(TextureDesc tex)
{
    for (int32 i = 0; i < textures.size(); ++i)
    {
        if (textures[i] == tex)
        {
            return TextureIndex(i);
        }
    }

    TextureIndex ti = int32(textures.size());
    textures.push_back(std::move(tex));
    return ti;
}

void Scene::AddMesh(const Mesh& mesh, MaterialIndex mi)
{
    const int32 offset = int32(positions.size());

    positions.insert(positions.end(), mesh.positions.begin(), mesh.positions.end());
    normals.insert(normals.end(), mesh.normals.begin(), mesh.normals.end());
    tangents.insert(tangents.end(), mesh.tangents.begin(), mesh.tangents.end());
    texcoords.insert(texcoords.end(), mesh.texcoords.begin(), mesh.texcoords.end());

    indices.reserve(indices.size() + mesh.triangle_count);
    material_indices.reserve(material_indices.size() + mesh.triangle_count);
    aabbs.reserve(aabbs.size() + mesh.triangle_count);

    for (size_t i = 0; i < mesh.indices.size(); i += 3)
    {
        int32 i0 = offset + mesh.indices[i + 0];
        int32 i1 = offset + mesh.indices[i + 1];
        int32 i2 = offset + mesh.indices[i + 2];

        indices.emplace_back(i0, i1, i2);
        material_indices.push_back(mi);

        if (materials[mi]->Is<DiffuseLightMaterial>())
        {
            light_indices.push_back(int32(indices.size() - 1));
        }

        aabbs.push_back(TriangleAABB(positions[i0], positions[i1], positions[i2]));
    }
}

void Scene::Clear()
{
    for (auto v : materials.get_vectors())
    {
        v.resize(0);
        v.shrink_to_fit();
    }

    positions.resize(0);
    positions.shrink_to_fit();
    normals.resize(0);
    normals.shrink_to_fit();
    tangents.resize(0);
    tangents.shrink_to_fit();
    texcoords.resize(0);
    texcoords.shrink_to_fit();
    material_indices.resize(0);
    material_indices.shrink_to_fit();
    indices.resize(0);
    indices.shrink_to_fit();
    light_indices.resize(0);
    light_indices.shrink_to_fit();

    aabbs.resize(0);
    aabbs.shrink_to_fit();
}

} // namespace cuwfrt
