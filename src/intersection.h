#pragma once

#include "common.h"
#include "indices.h"

namespace cuwfrt
{

struct Intersection
{
    Float t;
    Point3 point;

    Vec3 normal; // Geometric normal
    Point3 uvw;

    bool front_face;

    PrimitiveIndex index;
};

} // namespace cuwfrt
