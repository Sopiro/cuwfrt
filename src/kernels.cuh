#pragma once

#include "wak/hash.h"
#include "wak/random.h"

#include "api.cuh"
#include "gpu_scene.cuh"
#include "triangle.cuh"

#include "camera.h"

using namespace cuwfrt;
using namespace wak;

__kernel__ void RenderGradient(float4* pixels, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    pixels[index] = make_float4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

__kernel__ void Render(float4* pixels, Point2i res, GPUScene scene, Camera camera, int32 tri_count)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    RNG rng(Hash(x, y));

    Ray ray;
    camera.SampleRay(&ray, Point2i(x, y), { rng.NextFloat(), rng.NextFloat() }, { rng.NextFloat(), rng.NextFloat() });

    bool found_intersection = false;
    MaterialIndex mi;
    Intersection closest{ .t = infinity };

    for (int32 i = 0; i < tri_count; ++i)
    {
        Vec3i index = scene.indices[i];
        Vec3 p0 = scene.positions[index[0]];
        Vec3 p1 = scene.positions[index[1]];
        Vec3 p2 = scene.positions[index[2]];

        Intersection isect;
        if (TriangleIntersect(&isect, p0, p1, p2, ray, Ray::epsilon, infinity))
        {
            found_intersection = true;
            if (isect.t < closest.t)
            {
                mi = scene.material_indices[i];
                closest = isect;
            }
        }
    }

    if (found_intersection)
    {
        Vec3 albedo = scene.materials[mi].reflectance;
        pixels[y * res.x + x] = float4(albedo.x, albedo.y, albedo.z, 1.0f);
        // pixels[y * res.x + x] = float4(closest.normal.x / 2 + 0.5, closest.normal.y / 2 + 0.5, closest.normal.z / 2 + 0.5, 1);
    }
    else
    {
        pixels[y * res.x + x] = float4(0, 0, 0, 1);
    }
}
