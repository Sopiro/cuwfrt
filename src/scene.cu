#pragma once

#include "scene.h"
#include "mesh.h"
#include "triangle.cuh"

namespace cuwfrt
{

MaterialIndex Scene::AddMaterial(Material mat)
{
    MaterialIndex mi = int32(materials.size());
    materials.push_back(std::move(mat));
    return mi;
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

        if (materials[mi].is_light)
        {
            light_indices.push_back(int32(indices.size() - 1));
        }

        aabbs.push_back(TriangleAABB(positions[i0], positions[i1], positions[i2]));
    }
}

} // namespace cuwfrt
