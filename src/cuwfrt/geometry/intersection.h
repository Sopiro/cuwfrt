#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/scene/gpu_scene.h"

#include "primitive.h"

namespace cuwfrt
{

struct Intersection
{
    Float t;
    Point3 point;
    Point2 uv;

    Vec3 normal; // Geometric normal
    Vec3 shading_normal;

    bool front_face;

    PrimitiveIndex prim;
};

} // namespace cuwfrt
