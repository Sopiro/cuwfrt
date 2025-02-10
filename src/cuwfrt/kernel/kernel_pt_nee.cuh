#pragma once

#include "wak/hash.h"
#include "wak/random.h"

#include "cuwfrt/cuda_api.h"
#include "cuwfrt/shading/frame.h"
#include "cuwfrt/shading/sampling.h"

#include "cuwfrt/camera/camera.h"
#include "cuwfrt/scene/gpu_scene.cuh"

#include "kernel_intersect.cuh"
#include "kernel_material.cuh"
#include "kernel_primitive.cuh"

namespace cuwfrt
{

inline __GPU__ Vec3 SampleDirectLight(
    const GPUScene* scene, const Vec3& wo, const Intersection& isect, const Material* mat, Float u0, Point2 u12, const Vec3& beta
)
{
    int32 light_index = std::min(int32(u0 * scene->light_count), scene->light_count - 1);
    PrimitiveIndex light = scene->light_indices[light_index];
    Material* light_mat = GetMaterial(scene, scene->material_indices[light]);
    Float light_sample_pmf = 1.0f / scene->light_count;

    PrimitiveSample primitive_sample = triangle::Sample(scene, light, isect.point, u12);

    Vec3 wi = primitive_sample.point - isect.point;
    Float visibility = wi.Normalize() - Ray::epsilon;
    Vec3 Li = light_mat->Le(Intersection{ .front_face = Dot(primitive_sample.normal, wi) < 0 }, wo);

    Float bsdf_pdf = mat->PDF(isect, wo, wi);
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
    Vec3 f_cos = mat->BSDF(scene, isect, wo, wi) * AbsDot(isect.normal, wi);

    Float mis_weight = PowerHeuristic(1, light_pdf, 1, bsdf_pdf);
    return beta * mis_weight * Li * f_cos / light_pdf;
}

__KERNEL__ void PathTraceNEE(
    Vec4* sample_buffer, Vec4* frame_buffer, Point2i res, GPUScene scene, Camera camera, Options options, int32 time
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
            L += beta * SkyColor(ray.d);
            break;
        }

        Vec3 wo = Normalize(-ray.d);

        MaterialIndex mi = scene.material_indices[isect.prim];
        Material* mat = GetMaterial(&scene, mi);

        if (Vec3 Le = mat->Le(isect, wo); Le != Vec3(0))
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
        Point2 u{ rng.NextFloat(), rng.NextFloat() };
        if (!mat->SampleBSDF(&ss, &scene, isect, wo, u))
        {
            break;
        }
        specular_bounce = ss.is_specular;

        // Save bsdf pdf for MIS
        prev_bsdf_pdf = ss.pdf;
        beta *= ss.s * AbsDot(isect.normal, ss.wi) / ss.pdf;
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

        bounce++;
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

} // namespace cuwfrt
