#pragma once

#include <device_launch_parameters.h>

#include "cuwfrt/raytracer.h"
#include "device_intersect.cuh"
#include "device_material.cuh"

namespace cuwfrt
{

inline __GPU__ Vec3 SampleDirectLight(
    const GPUScene* scene, const Vec3& wo, const Intersection& isect, const Material* mat, Float u0, Point2 u12, const Vec3& beta
)
{
    int32 light_index = std::min(int32(u0 * scene->light_count), scene->light_count - 1);
    if (light_index < 0)
    {
        return Vec3(0);
    }

    PrimitiveIndex light = scene->light_indices[light_index];
    Material* light_mat = GetMaterial(scene, light);
    Float light_sample_pmf = 1.0f / scene->light_count;

    PrimitiveSample primitive_sample = triangle::Sample(scene, light, isect.point, u12);

    Vec3 wi = primitive_sample.point - isect.point;
    Float visibility = wi.Normalize() - Ray::epsilon;
    Vec3 Li = light_mat->Le(scene, Intersection{ .front_face = Dot(primitive_sample.normal, wi) < 0 }, wo);

    Float bsdf_pdf = mat->PDF(scene, isect, wo, wi);
    if (Li == Vec3(0) || bsdf_pdf == 0)
    {
        return Vec3(0);
    }

    Ray shadow_ray(isect.point, wi);
    if (IntersectAny(scene, shadow_ray, Ray::epsilon, visibility))
    {
        return Vec3(0);
    }

    Float light_pdf = primitive_sample.pdf * light_sample_pmf;
    Vec3 f_cos = mat->BSDF(scene, isect, wo, wi) * AbsDot(isect.shading_normal, wi);

    Float mis_weight = PowerHeuristic(1, light_pdf, 1, bsdf_pdf);
    return beta * mis_weight * Li * f_cos / light_pdf;
}

__KERNEL__ void PathTraceNEE(
    Vec4* __restrict__ sample_buffer, Point2i res, GPUScene scene, Camera camera, Options options, int32 seed
)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res.x || y >= res.y) return;

    RNG rng(Hash(x, y, seed));

    // Generate primary ray
    Ray ray;
    Point2 u0{ rng.NextFloat(), rng.NextFloat() };
    Point2 u1{ rng.NextFloat(), rng.NextFloat() };
    camera.SampleRay(&ray, Point2i(x, y), u0, u1);

    Vec3 L(0), beta(1);
    bool specular_bounce = false;
    int32 bounce = 0;
    Float prev_bsdf_pdf = 0;

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

        if (Vec3 Le = mat->Le(&scene, isect, wo); Le != Vec3(0))
        {
            bool is_area_light = mat->Is<DiffuseLightMaterial>();
            if (bounce == 0 || specular_bounce || !is_area_light)
            {
                L += beta * Le;
            }
            else if (is_area_light)
            {
                // Evaluate BSDF sample with MIS for area light
                Float light_sample_pmf = 1.0f / scene.light_count;
                Float light_pdf = triangle::PDF(&scene, isect.prim, isect, ray) * light_sample_pmf;
                Float mis_weight = PowerHeuristic(1, prev_bsdf_pdf, 1, light_pdf);

                L += beta * mis_weight * Le;
            }
        }

        if (bounce++ >= options.max_bounces)
        {
            break;
        }

        L += SampleDirectLight(&scene, wo, isect, mat, rng.NextFloat(), { rng.NextFloat(), rng.NextFloat() }, beta);

        Scattering ss;
        if (!mat->SampleBSDF(&ss, &scene, isect, wo, rng.NextFloat(), { rng.NextFloat(), rng.NextFloat() }))
        {
            break;
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
                break;
            }
        }
    }

    int32 index = y * res.x + x;
    sample_buffer[index] = Vec4(L, 0);
}

} // namespace cuwfrt
