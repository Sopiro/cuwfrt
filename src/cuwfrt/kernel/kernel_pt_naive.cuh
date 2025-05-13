#pragma once

#include <device_launch_parameters.h>

#include "cuwfrt/raytracer.h"
#include "device_intersect.cuh"
#include "device_material.cuh"

namespace cuwfrt
{

__KERNEL__ void PathTraceNaive(
    Vec4* __restrict__ sample_buffer, Point2i res, GPUScene scene, Camera camera, GBuffer g_buffer, Options options, int32 seed
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    g_buffer.position[index] = Vec4(0);
    g_buffer.normal[index] = Vec4(0);
    g_buffer.albedo[index] = Vec4(0);

    RNG rng(Hash(x, y, seed));

    // Generate primary ray
    Ray ray;
    Point2 u0{ rng.NextFloat(), rng.NextFloat() };
    Point2 u1{ rng.NextFloat(), rng.NextFloat() };
    camera.SampleRay(&ray, Point2i(x, y), u0, u1);

    Vec3 L(0), beta(1);

    int32 bounce = 0;
    while (true)
    {
        Intersection isect;
        bool found_intersection = Intersect(&isect, &scene, ray, Ray::epsilon, infinity);
        if (!found_intersection)
        {
            if (options.render_sky)
            {
                L += beta * SkyColor(ray.d);
            }
            break;
        }

        Vec3 wo = Normalize(-ray.d);

        Material* mat = GetMaterial(&scene, isect.prim);

        if (bounce == 0)
        {
            g_buffer.position[index] = Vec4(isect.point, isect.t);
            g_buffer.normal[index] = Vec4(isect.shading_normal, isect.prim);
            g_buffer.albedo[index] = Vec4(mat->Albedo(&scene, isect, wo), 0);
        }

        if (Vec3 Le = mat->Le(&scene, isect, wo); Le != Vec3(0))
        {
            L += beta * Le;
            break;
        }

        if (bounce++ >= options.max_bounces)
        {
            break;
        }

        Scattering ss;
        if (!mat->SampleBSDF(&ss, &scene, isect, wo, rng.NextFloat(), { rng.NextFloat(), rng.NextFloat() }))
        {
            break;
        }

        beta *= ss.s * AbsDot(isect.shading_normal, ss.wi) / ss.pdf;

        if (bounce > 1)
        {
            Float rr = fmin(1.0f, fmax(beta.x, fmax(beta.y, beta.z)));
            if (rng.NextFloat() < rr)
            {
                beta /= rr;
            }
            else
            {
                break;
            }
        }

        ray.o = isect.point;
        ray.d = ss.wi;
    }

    sample_buffer[index] = Vec4(L, 0);
}

} // namespace cuwfrt
