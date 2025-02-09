#pragma once

#include "wak/hash.h"
#include "wak/random.h"

#include "cuda_api.h"
#include "frame.h"
#include "gpu_scene.cuh"
#include "kernel_material.cuh"
#include "raytracer.cuh"
#include "sampling.h"
#include "triangle.h"

#include "camera.h"

using namespace cuwfrt;
using namespace wak;

inline __GPU__ Vec3 SkyColor(Vec3 d)
{
    Float a = 0.5 * (d.y + 1.0);
    return Lerp(Vec3(0.5, 0.7, 1.0), Vec3(1.0, 1.0, 1.0), a);
}

__GPU__ bool Intersect(Intersection* closest, const GPUScene::Data* scene, Ray r, Float t_min, Float t_max)
{
    bool hit_closest = false;

    const Vec3 inv_dir(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
    const int32 is_dir_neg[3] = { int32(inv_dir.x < 0), int32(inv_dir.y < 0), int32(inv_dir.z < 0) };

    int32 stack[64];
    int32 stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0)
    {
        int32 index = stack[--stack_ptr];

        LinearBVHNode& node = scene->bvh_nodes[index];

        if (node.aabb.TestRay(r.o, t_min, t_max, inv_dir, is_dir_neg))
        {
            if (node.primitive_count > 0)
            {
                // Leaf node
                for (int32 i = 0; i < node.primitive_count; ++i)
                {
                    PrimitiveIndex primitive = scene->bvh_primitives[node.primitives_offset + i];
                    Vec3i index = scene->indices[primitive];
                    Vec3 p0 = scene->positions[index[0]];
                    Vec3 p1 = scene->positions[index[1]];
                    Vec3 p2 = scene->positions[index[2]];

                    Intersection isect(scene);
                    bool hit = TriangleIntersect(&isect, p0, p1, p2, r, t_min, t_max);
                    if (hit)
                    {
                        WakAssert(isect.t <= t_max);
                        hit_closest = true;
                        isect.prim = primitive;

                        t_max = isect.t;
                        *closest = isect;
                    }
                }
            }
            else
            {
                // Internal node

                // Ordered traversal
                // Put far child on stack first
                int32 child1 = index + 1;
                int32 child2 = node.child2_offset;

                if (is_dir_neg[scene->bvh_nodes[index].axis])
                {
                    stack[stack_ptr++] = child1;
                    stack[stack_ptr++] = child2;
                }
                else
                {
                    stack[stack_ptr++] = child2;
                    stack[stack_ptr++] = child1;
                }
            }
        }
    }

    return hit_closest;
}

__GPU__ bool IntersectAny(const GPUScene::Data* scene, Ray r, Float t_min, Float t_max)
{
    const Vec3 inv_dir(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
    const int32 is_dir_neg[3] = { int32(inv_dir.x < 0), int32(inv_dir.y < 0), int32(inv_dir.z < 0) };

    int32 stack[64];
    int32 stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0)
    {
        int32 index = stack[--stack_ptr];

        LinearBVHNode& node = scene->bvh_nodes[index];

        if (node.aabb.TestRay(r.o, t_min, t_max, inv_dir, is_dir_neg))
        {
            if (node.primitive_count > 0)
            {
                // Leaf node
                for (int32 i = 0; i < node.primitive_count; ++i)
                {
                    PrimitiveIndex primitive = scene->bvh_primitives[node.primitives_offset + i];
                    Vec3i index = scene->indices[primitive];
                    Vec3 p0 = scene->positions[index[0]];
                    Vec3 p1 = scene->positions[index[1]];
                    Vec3 p2 = scene->positions[index[2]];

                    Intersection isect(scene);
                    bool hit = TriangleIntersectAny(p0, p1, p2, r, t_min, t_max);
                    if (hit)
                    {
                        return true;
                    }
                }
            }
            else
            {
                // Internal node

                // Ordered traversal
                // Put far child on stack first
                int32 child1 = index + 1;
                int32 child2 = node.child2_offset;

                if (is_dir_neg[scene->bvh_nodes[index].axis])
                {
                    stack[stack_ptr++] = child1;
                    stack[stack_ptr++] = child2;
                }
                else
                {
                    stack[stack_ptr++] = child2;
                    stack[stack_ptr++] = child1;
                }
            }
        }
    }

    return false;
}

__KERNEL__ void RenderGradient(Vec4* pixels, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    pixels[index] = Vec4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

__KERNEL__ void PathTrace(
    Vec4* sample_buffer, Vec4* frame_buffer, Point2i res, GPUScene::Data scene, Camera camera, Options options, int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    RNG rng(Hash(x, y, time));

    Ray ray;
    camera.SampleRay(&ray, Point2i(x, y), { rng.NextFloat(), rng.NextFloat() }, { rng.NextFloat(), rng.NextFloat() });

    Vec3 throughput(1);
    Vec3 L(0);

    int32 bounce = 0;
    while (true)
    {
        Intersection isect(&scene);
        bool found_intersection = Intersect(&isect, &scene, ray, Ray::epsilon, infinity);
        if (!found_intersection)
        {
            L += throughput * SkyColor(ray.d);
            break;
        }

        Vec3 wo = Normalize(-ray.d);

        MaterialIndex mi = scene.material_indices[isect.prim];
        Material* m = GetMaterial(&scene, mi);

        if (Vec3 Le = m->Le(isect, wo); Le != Vec3(0))
        {
            L += throughput * Le;
            break;
        }

        if (bounce++ >= options.max_bounces)
        {
            break;
        }

        SurfaceScattering ss;
        Point2 u{ rng.NextFloat(), rng.NextFloat() };
        if (!m->Scatter(&ss, isect, wo, u))
        {
            break;
        }

        throughput *= ss.atten;

        if (bounce > 1)
        {
            Float rr = fmin(1.0f, fmax(throughput.x, fmax(throughput.y, throughput.z)));
            if (rng.NextFloat() < rr)
            {
                throughput /= rr;
            }
            else
            {
                break;
            }
        }

        ray.o = isect.point;
        ray.d = ss.wi;

        ++bounce;
    }

    int32 index = y * res.x + x;
    sample_buffer[index] *= time;
    sample_buffer[index] += Vec4(L, 0);
    sample_buffer[index] /= time + 1.0f;

    frame_buffer[index].x = std::pow(sample_buffer[index].x, 1 / 2.2f);
    frame_buffer[index].y = std::pow(sample_buffer[index].y, 1 / 2.2f);
    frame_buffer[index].z = std::pow(sample_buffer[index].z, 1 / 2.2f);
    frame_buffer[index].w = 1.0f;
}
