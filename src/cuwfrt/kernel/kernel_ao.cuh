#pragma once

#include <device_launch_parameters.h>

#include "cuwfrt/raytracer.h"

#include "kernel_intersect.cuh"
#include "kernel_material.cuh"

namespace cuwfrt
{

__KERNEL__ void RaytraceAO(
    Vec4* __restrict__ sample_buffer,
    Vec4* __restrict__ frame_buffer,
    Point2i res,
    GPUScene scene,
    Camera camera,
    Options options,
    int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    RNG rng(Hash(x, y, time));

    // Generate primary ray
    Ray ray;
    Point2 u0{ rng.NextFloat(), rng.NextFloat() };
    Point2 u1{ rng.NextFloat(), rng.NextFloat() };
    camera.SampleRay(&ray, Point2i(x, y), u0, u1);

    Intersection isect;
    bool found_intersection = Intersect(&isect, &scene, ray, Ray::epsilon, infinity);
    int32 index = y * res.x + x;

    if (!found_intersection)
    {
        frame_buffer[index] = Vec4(0, 0, 0, 1);
        return;
    }

    Vec3 wi_local = SampleCosineHemisphere({ rng.NextFloat(), rng.NextFloat() });
    // Float pdf = CosineHemispherePDF(wi_local.z);

    if (wi_local.z <= 0)
    {
        wi_local.z = -wi_local.z;
    }

    Frame frame(isect.shading_normal);
    Vec3 wi = frame.FromLocal(wi_local);

    Vec4 occlusion;

    Ray ao_ray(isect.point, wi);
    if (IntersectAny(&scene, ao_ray, Ray::epsilon, 0.1f))
    {
        occlusion = Vec4(0, 0, 0, 1);
    }
    else
    {
        occlusion = Vec4(1, 1, 1, 1);
    }

    sample_buffer[index] *= time;
    sample_buffer[index] += occlusion;
    sample_buffer[index] /= time + 1.0f;

    frame_buffer[index] = ToSRGB(sample_buffer[index]);
}

} // namespace cuwfrt
