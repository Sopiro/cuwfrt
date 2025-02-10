#pragma once

#include "kernel_primitive.cuh"

namespace cuwfrt
{

inline __GPU__ Vec3 SkyColor(Vec3 d)
{
    Float a = 0.5 * (d.y + 1.0);
    return Lerp(Vec3(0.5, 0.7, 1.0), Vec3(1.0, 1.0, 1.0), a);
}

__GPU__ bool Intersect(Intersection* closest, const GPUScene* scene, Ray r, Float t_min, Float t_max)
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
                    PrimitiveIndex prim = scene->bvh_primitives[node.primitives_offset + i];
                    Intersection isect;
                    bool hit = triangle::Intersect(&isect, scene, prim, r, t_min, t_max);
                    if (hit)
                    {
                        WakAssert(isect.t <= t_max);
                        hit_closest = true;
                        isect.prim = prim;

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

__GPU__ bool IntersectAny(const GPUScene* scene, Ray r, Float t_min, Float t_max)
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
                    PrimitiveIndex prim = scene->bvh_primitives[node.primitives_offset + i];
                    bool hit = triangle::IntersectAny(scene, prim, r, t_min, t_max);
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

} // namespace cuwfrt
