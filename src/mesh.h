#pragma once

#include "common.h"

namespace cuwfrt
{

struct Mesh
{
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> texcoords;
    std::vector<int32> indices;
    int32 materialIndex;
};

} // namespace cuwfrt
