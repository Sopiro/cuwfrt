#pragma once

#include "materials.h"
#include "mesh.h"
#include "polymorphic_vector.h"
#include "texture_desc.h"

namespace cuwfrt
{

class Scene
{
public:
    Scene() = default;

    TextureIndex AddTexture(TextureDesc tex);

    template <typename T, typename... Args>
    MaterialIndex AddMaterial(Args&&... args);

    void AddMesh(const Mesh& mat, MaterialIndex mi);

    void Clear();

    std::vector<TextureDesc> textures;
    PolymorphicVector<Material> materials;

    // Triangles
    std::vector<Point3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec3> tangents;
    std::vector<Point2> texcoords;

    std::vector<MaterialIndex> material_indices;
    std::vector<AABB> aabbs;

    std::vector<Point3i> indices;
    std::vector<int32> light_indices;
};

template <typename T, typename... Args>
inline MaterialIndex Scene::AddMaterial(Args&&... args)
{
    return materials.emplace_back<T>(std::forward<Args>(args)...);
}

} // namespace cuwfrt
