#pragma once

#include "common.h"
#include "kernel/api.cuh"
#include "wak/ray.h"

namespace cuwfrt
{

__cpu_gpu__ bool TriangleIntersect(
    Float* t_hit, const Point3& p0, const Point3& p1, const Point3& p2, const Ray& ray, Float t_min, Float t_max
)
{
    Vec3 e1 = p1 - p0;
    Vec3 e2 = p2 - p0;

    Vec3 d = ray.d;
    Float l = d.Normalize();
    Vec3 pvec = Cross(d, e2);

    Float det = Dot(e1, pvec);
    // bool backface = det < epsilon;

    // Ray and triangle are parallel
    if (std::fabs(det) < epsilon)
    {
        return false;
    }

    Float invDet = 1 / det;

    Vec3 tvec = ray.o - p0;
    Float u = Dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1)
    {
        return false;
    }

    Vec3 qvec = Cross(tvec, e1);
    Float v = Dot(d, qvec) * invDet;
    if (v < 0 || u + v > 1)
    {
        return false;
    }

    Float t = Dot(e2, qvec) * invDet / l;
    if (t < t_min || t > t_max)
    {
        return false;
    }

    WakAssert(t_hit);

    *t_hit = t;
    return true;
}

} // namespace cuwfrt
