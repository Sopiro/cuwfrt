#pragma once

#include "wak/hash.h"
#include "wak/random.h"

#include "api.cuh"
#include "frame.h"
#include "gpu_scene.cuh"
#include "sampling.h"
#include "triangle.cuh"

#include "camera.h"

using namespace cuwfrt;
using namespace wak;

__gpu__ bool Intersect(Intersection* closest, const GPUScene& scene, Ray ray, int32 tri_count)
{
    bool found_intersection = false;
    closest->t = infinity;

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
            if (isect.t < closest->t)
            {
                isect.index = i;
                *closest = isect;
            }
        }
    }

    return found_intersection;
}

__kernel__ void RenderGradient(Vec4* pixels, Point2i res)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int index = y * res.x + x;
    pixels[index] = Vec4(x / (float)res.x, y / (float)res.y, 128 / 255.0f, 1.0f); // Simple gradient
}

__kernel__ void Render(
    Vec4* sample_buffer, Vec4* frame_buffer, Point2i res, GPUScene scene, Camera camera, int32 tri_count, int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    RNG rng(Hash(x, y, time));

    Ray ray;
    camera.SampleRay(&ray, Point2i(x, y), { rng.NextFloat(), rng.NextFloat() }, { rng.NextFloat(), rng.NextFloat() });

    Vec3 throughput(1);
    Vec3 r(0);

    int32 bounce = 0;
    while (bounce < 10)
    {
        Intersection isect;
        bool found_intersection = Intersect(&isect, scene, ray, tri_count);

        if (found_intersection)
        {
            MaterialIndex mi = scene.material_indices[isect.index];
            Material& m = scene.materials[mi];
            if (m.is_light)
            {
                r += throughput * m.reflectance;
                break;
            }
            else
            {
                throughput *= m.reflectance;
            }
        }
        else
        {
            break;
        }

        if (bounce > 2)
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

        Frame f(isect.normal);
        Vec3 wi = SampleCosineHemisphere({ rng.NextFloat(), rng.NextFloat() });
        wi = f.FromLocal(wi);

        ray.o = isect.point;
        ray.d = wi;

        ++bounce;
    }

    int32 index = y * res.x + x;
    sample_buffer[index] *= time;
    sample_buffer[index] += Vec4(r, 0);
    sample_buffer[index] /= time + 1.0f;

    frame_buffer[index].x = std::pow(sample_buffer[index].x, 1 / 2.2f);
    frame_buffer[index].y = std::pow(sample_buffer[index].y, 1 / 2.2f);
    frame_buffer[index].z = std::pow(sample_buffer[index].z, 1 / 2.2f);
    frame_buffer[index].w = 1.0f;
}
