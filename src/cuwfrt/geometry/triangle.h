#pragma once

#include "cuwfrt/common.h"

namespace cuwfrt
{

inline static AABB TriangleAABB(const Point3& p0, const Point3& p1, const Point3& p2)
{
    const Vec3 aabb_epsilon{ epsilon * 10 };

    Point3 min = Min(p0, Min(p1, p2));
    Point3 max = Max(p0, Max(p1, p2));

    return { min - aabb_epsilon, max + aabb_epsilon };
}

} // namespace cuwfrt
