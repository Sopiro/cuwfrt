#pragma once

#include "intersection.h"

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"
#include "cuwfrt/scene/gpu_scene.cuh"

namespace cuwfrt
{

inline __CPU_GPU__ bool TriangleIntersect(
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

inline __CPU_GPU__ bool TriangleIntersectAny(
    const Point3& p0, const Point3& p1, const Point3& p2, const Ray& ray, Float t_min, Float t_max
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

    return true;
}

inline __CPU_GPU__ static AABB TriangleAABB(const Point3& p0, const Point3& p1, const Point3& p2)
{
    const Vec3 aabb_epsilon{ epsilon * 10 };

    Point3 min = Min(p0, Min(p1, p2));
    Point3 max = Max(p0, Max(p1, p2));

    return { min - aabb_epsilon, max + aabb_epsilon };
}

inline __CPU_GPU__ Point2 GetTexcoord(const GPUScene::Data* scene, PrimitiveIndex prim, const Vec3& uvw)
{
    Vec3i index = scene->indices[prim];
    Point2 tc0 = scene->texcoords[index[0]];
    Point2 tc1 = scene->texcoords[index[1]];
    Point2 tc2 = scene->texcoords[index[2]];
    return uvw.z * tc0 + uvw.x * tc1 + uvw.y * tc2;
}

inline __CPU_GPU__ Vec3 GetNormal(const GPUScene::Data* scene, PrimitiveIndex prim, const Vec3& uvw)
{
    Vec3i index = scene->indices[prim];
    Vec3 n0 = scene->normals[index[0]];
    Vec3 n1 = scene->normals[index[1]];
    Vec3 n2 = scene->normals[index[2]];
    return Normalize(uvw.z * n0 + uvw.x * n1 + uvw.y * n2);
}

inline __CPU_GPU__ Vec3 GetTangent(const GPUScene::Data* scene, PrimitiveIndex prim, const Vec3& uvw)
{
    Vec3i index = scene->indices[prim];
    Vec3 t0 = scene->tangents[index[0]];
    Vec3 t1 = scene->tangents[index[1]];
    Vec3 t2 = scene->tangents[index[2]];
    return Normalize(uvw.z * t0 + uvw.x * t1 + uvw.y * t2);
}

} // namespace cuwfrt
