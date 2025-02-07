#pragma once

#include "api.cuh"
#include "common.h"
#include "intersection.h"
#include "wak/ray.h"

namespace cuwfrt
{

__cpu_gpu__ bool TriangleIntersect(
    Intersection* isect, const Point3& p0, const Point3& p1, const Point3& p2, const Ray& ray, Float t_min, Float t_max
)
{
    Vec3 e1 = p1 - p0;
    Vec3 e2 = p2 - p0;

    Float l = Length(ray.d);
    Vec3 d = ray.d / l;
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

    // Found intersection
    Float w = 1 - u - v;

    isect->t = t;
    isect->point = ray.At(t);
    isect->uvw = Point3(u, v, w);
    Vec3 face_normal = Normalize(Cross(e1, e2));

    bool front_face = Dot(ray.d, face_normal) < 0;
    Float sign = front_face ? 1.0f : -1.0f;

    isect->front_face = front_face;
    isect->normal = sign * face_normal;

    return true;
}

} // namespace cuwfrt
