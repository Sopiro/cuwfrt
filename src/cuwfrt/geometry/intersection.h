#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/scene/gpu_scene.cuh"

#include "primitive.h"

namespace cuwfrt
{

struct Intersection
{
    Float t;
    Point3 point;

    Vec3 normal; // Geometric normal
    Point3 uvw;

    bool front_face;

    PrimitiveIndex prim;
};

} // namespace cuwfrt
