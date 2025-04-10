#pragma once

#include <device_launch_parameters.h>

#include "cuwfrt/raytracer.h"
#include "kernel_intersect.cuh"
#include "kernel_material.cuh"
#include "kernel_utils.cuh"

namespace cuwfrt
{

__KERNEL__ void ResetCounts(int32* next_ray_count, int32* closest_ray_count, int32* miss_ray_count, int32* shadow_ray_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *next_ray_count = 0;
        *closest_ray_count = 0;
        *miss_ray_count = 0;
        *shadow_ray_count = 0;
    }
}

// Generate primary rays for each pixel
__KERNEL__ void GeneratePrimaryRays(
    Vec4* __restrict__ sample_buffer, WavefrontRay* __restrict__ active_rays, Point2i res, Camera camera, int32 time
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    const int32 index = y * res.x + x;

    // Initialize wavefront path states
    WavefrontRay& wf_ray = active_rays[index];

    RNG rng(Hash(x, y, time));

    // Generate primary ray
    camera.SampleRay(&wf_ray.ray, { x, y }, { rng.NextFloat(), rng.NextFloat() }, { rng.NextFloat(), rng.NextFloat() });
    wf_ray.isect = { 0 };

    wf_ray.rng = rng;
    wf_ray.beta = Vec3(1);

    wf_ray.last_bsdf_pdf = 0.0f;
    wf_ray.is_specular = false;

    wf_ray.pixel_index = index;

    sample_buffer[index] *= time;
}

// Trace rays and find closest intersection
__KERNEL__ void Extend(
    WavefrontRay* __restrict__ active_rays,
    int32 active_ray_count,
    WavefrontRay* __restrict__ closest_rays,
    int32* closest_ray_count,
    WavefrontMissRay* __restrict__ miss_rays,
    int32* miss_ray_count,
    GPUScene scene
)
{
    int32 index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= active_ray_count) return;

    WavefrontRay& wf_ray = active_rays[index];

    bool found_intersection = Intersect(&wf_ray.isect, &scene, wf_ray.ray, Ray::epsilon, infinity);
    if (found_intersection)
    {
        int32 next_index = atomicAdd(closest_ray_count, 1);
        closest_rays[next_index] = active_rays[index];
    }
    else
    {
        int32 next_index = atomicAdd(miss_ray_count, 1);
        miss_rays[next_index].pixel_index = wf_ray.pixel_index;
        miss_rays[next_index].d = wf_ray.ray.d;
        miss_rays[next_index].beta = wf_ray.beta;
    }
}

__KERNEL__ void Miss(
    WavefrontMissRay* __restrict__ miss_rays, int32 miss_ray_count, Vec4* __restrict__ sample_buffer, Options options
)
{
    int32 index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= miss_ray_count) return;

    WavefrontMissRay& miss_ray = miss_rays[index];

    if (options.render_sky)
    {
        Vec4& L = sample_buffer[miss_ray.pixel_index];
        AtomicAdd(&L, miss_ray.beta * SkyColor(miss_ray.d));
    }
}

// Shade hit points, generate next bounce rays and shadow rays
__KERNEL__ void Shade(
    WavefrontRay* __restrict__ closest_rays,
    int32 closest_ray_count,
    WavefrontRay* __restrict__ next_rays,
    int32* next_ray_count,
    WavefrontShadowRay* __restrict__ shadow_rays,
    int32* shadow_ray_count,
    Vec4* __restrict__ sample_buffer,
    GPUScene scene,
    Options options,
    int32 bounce,
    int32 time
)
{
    int32 index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= closest_ray_count) return;

    WavefrontRay& wf_ray = closest_rays[index];
    int32 pixel_index = wf_ray.pixel_index;

    Ray& ray = wf_ray.ray;
    Intersection& isect = wf_ray.isect;
    Vec4& L = sample_buffer[pixel_index];
    Vec3& beta = wf_ray.beta;
    RNG& rng = wf_ray.rng;
    bool& specular_bounce = wf_ray.is_specular;
    Float& prev_bsdf_pdf = wf_ray.last_bsdf_pdf;

    Vec3 wo = Normalize(-ray.d);

    Material* mat = GetMaterial(&scene, isect.prim);

    if (Vec3 Le = mat->Le(isect, wo); Le != Vec3(0))
    {
        bool is_area_light = mat->Is<DiffuseLightMaterial>();
        if (bounce == 0 || specular_bounce || !is_area_light)
        {
            AtomicAdd(&L, beta * Le);
        }
        else if (is_area_light)
        {
            // Evaluate BSDF sample with MIS for area light
            Float light_sample_pmf = 1.0f / scene.light_count;
            Float light_pdf = triangle::PDF(&scene, isect.prim, isect, ray) * light_sample_pmf;
            Float mis_weight = PowerHeuristic(1, prev_bsdf_pdf, 1, light_pdf);

            AtomicAdd(&L, beta * mis_weight * Le);
        }
    }

    // Sample direct light
    Float u0 = rng.NextFloat();
    Point2 u12 = { rng.NextFloat(), rng.NextFloat() };
    int32 light_index = std::min(int32(u0 * scene.light_count), scene.light_count - 1);
    if (light_index >= 0)
    {
        PrimitiveIndex light = scene.light_indices[light_index];
        Material* light_mat = GetMaterial(&scene, light);
        Float light_sample_pmf = 1.0f / scene.light_count;

        PrimitiveSample primitive_sample = triangle::Sample(&scene, light, isect.point, u12);

        Vec3 wi = primitive_sample.point - isect.point;
        Float visibility = wi.Normalize() - Ray::epsilon;
        Vec3 Li = light_mat->Le(Intersection{ .front_face = Dot(primitive_sample.normal, wi) < 0 }, wo);

        Float bsdf_pdf = mat->PDF(&scene, isect, wo, wi);
        if (Li != Vec3(0) && bsdf_pdf != 0)
        {
            Float light_pdf = primitive_sample.pdf * light_sample_pmf;
            Vec3 f_cos = mat->BSDF(&scene, isect, wo, wi) * AbsDot(isect.shading_normal, wi);

            Float mis_weight = PowerHeuristic(1, light_pdf, 1, bsdf_pdf);

            int32 shadow_ray_index = atomicAdd(shadow_ray_count, 1);
            shadow_rays[shadow_ray_index].ray = Ray(isect.point, wi);
            shadow_rays[shadow_ray_index].visibility = visibility;
            shadow_rays[shadow_ray_index].Li = beta * mis_weight * Li * f_cos / light_pdf;
            shadow_rays[shadow_ray_index].pixel_index = pixel_index;
        }
    }

    Scattering ss;
    if (!mat->SampleBSDF(&ss, &scene, isect, wo, rng.NextFloat(), { rng.NextFloat(), rng.NextFloat() }))
    {
        return;
    }
    specular_bounce = ss.is_specular;

    // Save bsdf pdf for MIS
    prev_bsdf_pdf = ss.pdf;
    beta *= ss.s * AbsDot(isect.shading_normal, ss.wi) / ss.pdf;
    ray = Ray(isect.point + ss.wi * Ray::epsilon, ss.wi);

    if (bounce > 1)
    {
        Float rr = fmin(1.0f, fmax(beta.x, fmax(beta.y, beta.z)));
        if (rng.NextFloat() < rr)
        {
            beta /= rr;
        }
        else
        {
            return;
        }
    }

    ray.o = isect.point;
    ray.d = ss.wi;

    int32 new_index = atomicAdd(next_ray_count, 1);
    next_rays[new_index] = wf_ray;
}

// Process shadow rays, add contribution if unoccluded
__KERNEL__ void Connect(
    WavefrontShadowRay* __restrict__ shadow_rays, int32 shadow_ray_count, Vec4* __restrict__ sample_buffer, GPUScene scene
)
{
    int32 index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= shadow_ray_count) return;

    WavefrontShadowRay& wf_shadow_ray = shadow_rays[index];

    if (!IntersectAny(&scene, wf_shadow_ray.ray, Ray::epsilon, wf_shadow_ray.visibility))
    {
        Vec4& L = sample_buffer[wf_shadow_ray.pixel_index];
        AtomicAdd(&L, wf_shadow_ray.Li);
    }
}

// Finalize frame: average samples and apply gamma correction
__KERNEL__ void Finalize(Vec4* __restrict__ sample_buffer, Vec4* __restrict__ frame_buffer, Point2i res, int32 time)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    int32 index = y * res.x + x;

    sample_buffer[index] /= time + 1.0f;

    frame_buffer[index].x = std::pow(sample_buffer[index].x, 1 / 2.2f);
    frame_buffer[index].y = std::pow(sample_buffer[index].y, 1 / 2.2f);
    frame_buffer[index].z = std::pow(sample_buffer[index].z, 1 / 2.2f);
    frame_buffer[index].w = 1.0f;
}

} // namespace cuwfrt