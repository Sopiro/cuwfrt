#pragma once

#include "common.h"

namespace cuwfrt
{

struct Vertex
{
    Point3 position;
    Vec3 normal;
    Vec3 tangent;
    Point2 texcoord;
};

// Represents triangle mesh
class Mesh
{
public:
    Mesh(
        std::vector<Point3> positions,
        std::vector<Vec3> normals,
        std::vector<Vec3> tangents,
        std::vector<Point2> texcoords,
        std::vector<int32> indices,
        const Mat4& transform
    );
    Mesh(const std::vector<Vertex>& vertices, std::vector<int32> indices, const Mat4& transform);

    int32 triangle_count;
    std::vector<Point3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec3> tangents;
    std::vector<Point2> texcoords;
    std::vector<int32> indices;
};

} // namespace cuwfrt
