#pragma once

#include "indices.h"
#include "material.h"
#include "mesh.h"

namespace cuwfrt
{

class Scene
{
public:
    Scene() = default;

    MaterialIndex AddMaterial(Material mat);
    void AddMesh(const Mesh& mat, MaterialIndex mi);

    void Clear();

    std::vector<Material> materials;

    // Triangles
    std::vector<Point3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec3> tangents;
    std::vector<Point2> texcoords;
    std::vector<MaterialIndex> material_indices;
    std::vector<Point3i> indices;
    std::vector<int32> light_indices;

    std::vector<AABB> aabbs;
};

} // namespace cuwfrt
