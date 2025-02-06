#pragma once

#include "common.h"

namespace cuwfrt
{

struct Intersection
{
    Float t;
    Point3 point;
    Vec3 normal; // Geometric normal
    Point3 uvw;
};

} // namespace cuwfrt
