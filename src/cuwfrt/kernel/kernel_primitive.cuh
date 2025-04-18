#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/geometry/intersection.h"
#include "cuwfrt/scene/gpu_scene.h"

namespace cuwfrt
{

struct PrimitiveSample
{
    Point3 point;
    Vec3 normal;
    Float pdf;
};

namespace triangle
{

inline __GPU__ bool Intersect(
    Intersection* isect, const GPUScene* scene, PrimitiveIndex prim, const Ray& ray, Float t_min, Float t_max
)
{
    Vec3i index = scene->indices[prim];
    Vec3 p0 = scene->positions[index[0]];
    Vec3 p1 = scene->positions[index[1]];
    Vec3 p2 = scene->positions[index[2]];

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

    // Found intersection
    Float w = 1 - u - v;

    isect->t = t;
    isect->point = ray.At(t);
    isect->uv = Point2(u, v);
    Vec3 face_normal = Normalize(Cross(e1, e2));

    bool front_face = Dot(ray.d, face_normal) < 0;
    Float sign = front_face ? 1.0f : -1.0f;

    isect->front_face = front_face;
    isect->normal = sign * face_normal;

    Vec3 n0 = scene->normals[index[0]];
    Vec3 n1 = scene->normals[index[1]];
    Vec3 n2 = scene->normals[index[2]];
    isect->shading_normal = sign * Normalize(w * n0 + u * n1 + v * n2);

    return true;
}

inline __GPU__ bool IntersectAny(const GPUScene* scene, PrimitiveIndex prim, const Ray& ray, Float t_min, Float t_max)
{
    Vec3i index = scene->indices[prim];
    Vec3 p0 = scene->positions[index[0]];
    Vec3 p1 = scene->positions[index[1]];
    Vec3 p2 = scene->positions[index[2]];

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

inline __GPU__ Point2 GetTexcoord(const GPUScene* scene, const Intersection& isect)
{
    Vec3i index = scene->indices[isect.prim];
    Point2 tc0 = scene->texcoords[index[0]];
    Point2 tc1 = scene->texcoords[index[1]];
    Point2 tc2 = scene->texcoords[index[2]];

    Float w = 1 - isect.uv[0] - isect.uv[1];
    return w * tc0 + isect.uv.x * tc1 + isect.uv.y * tc2;
}

inline __GPU__ Vec3 GetNormal(const GPUScene* scene, const Intersection& isect)
{
    Vec3i index = scene->indices[isect.prim];
    Vec3 n0 = scene->normals[index[0]];
    Vec3 n1 = scene->normals[index[1]];
    Vec3 n2 = scene->normals[index[2]];

    Float w = 1 - isect.uv[0] - isect.uv[1];
    return Normalize(w * n0 + isect.uv.x * n1 + isect.uv.y * n2);
}

inline __GPU__ Vec3 GetTangent(const GPUScene* scene, const Intersection& isect)
{
    Vec3i index = scene->indices[isect.prim];
    Vec3 t0 = scene->tangents[index[0]];
    Vec3 t1 = scene->tangents[index[1]];
    Vec3 t2 = scene->tangents[index[2]];

    Float w = 1 - isect.uv[0] - isect.uv[1];
    return Normalize(w * t0 + isect.uv.x * t1 + isect.uv.y * t2);
}

inline __GPU__ PrimitiveSample Sample(const GPUScene* scene, PrimitiveIndex prim, Point2 u0)
{
    Vec3i index = scene->indices[prim];
    Vec3 p0 = scene->positions[index[0]];
    Vec3 p1 = scene->positions[index[1]];
    Vec3 p2 = scene->positions[index[2]];

    Vec3 e1 = p1 - p0;
    Vec3 e2 = p2 - p0;

    Float u = u0[0];
    Float v = u0[1];

    if (u + v > 1)
    {
        u = 1 - u;
        v = 1 - v;
    }

    PrimitiveSample sample;
    sample.normal = Cross(e1, e2);
    sample.point = p0 + e1 * u + e2 * v;

    Float area = sample.normal.Normalize() * 0.5f;
    sample.pdf = 1 / area;

    return sample;
}

inline __GPU__ PrimitiveSample Sample(const GPUScene* scene, PrimitiveIndex prim, const Point3& ref, Point2 u)
{
    PrimitiveSample sample = triangle::Sample(scene, prim, u);

    Vec3 d = sample.point - ref;
    Float distance_squared = Dot(d, d);

    Float cosine = Dot(d, sample.normal) / std::sqrt(distance_squared);
    if (cosine < 0)
    {
        cosine = -cosine;
    }

    sample.pdf *= distance_squared / cosine; // Convert to solid angle measure

    return sample;
}

inline __GPU__ Float PDF(const GPUScene* scene, PrimitiveIndex prim, const Intersection& hit_is, const Ray& hit_ray)
{
    Float distance_squared = hit_is.t * hit_is.t * Length2(hit_ray.d);
    Float cosine = std::fabs(Dot(hit_ray.d, hit_is.normal) / Length(hit_ray.d));

    Vec3i index = scene->indices[prim];
    Vec3 p0 = scene->positions[index[0]];
    Vec3 p1 = scene->positions[index[1]];
    Vec3 p2 = scene->positions[index[2]];

    Vec3 e1 = p1 - p0;
    Vec3 e2 = p2 - p0;

    Float area = 0.5f * Length(Cross(e1, e2));

    return distance_squared / (cosine * area);
}

inline __GPU__ Float EvaluatePDF(const GPUScene* scene, PrimitiveIndex prim, const Ray& ray)
{
    Intersection isect;
    if (!Intersect(&isect, scene, prim, ray, Ray::epsilon, infinity))
    {
        return 0;
    }

    return PDF(scene, prim, isect, ray);
}

} // namespace triangle

} // namespace cuwfrt
