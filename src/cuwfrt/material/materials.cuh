#pragma once

#include "cuwfrt/material/material.h"

#include "cuwfrt/geometry/intersection.h"
#include "cuwfrt/shading/frame.h"
#include "cuwfrt/shading/sampling.h"
#include "cuwfrt/shading/scattering.h"

#include "cuwfrt/kernel/kernel_primitive.cuh"
#include "cuwfrt/kernel/kernel_texture.cuh"

#include <cuda_fp16.h>

namespace cuwfrt
{

class alignas(16) DiffuseLightMaterial : public Material
{
public:
    DiffuseLightMaterial(Vec3 emission)
        : Material(Material::TypeIndexOf<DiffuseLightMaterial>())
        , emission{ emission }
    {
    }

    __GPU__ Vec3 Le(const Intersection& isect, const Vec3& wo) const
    {
        if (isect.front_face)
        {
            return emission;
        }
        else
        {
            return Vec3(0);
        }
    }

    __GPU__ bool SampleBSDF(Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        return false;
    }

    __GPU__ Float PDF(const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return 0;
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return Vec3(0);
    }

    Vec3 emission;
};

class alignas(16) DiffuseMaterial : public Material
{
public:
    DiffuseMaterial(Vec3 reflectance)
        : Material(Material::TypeIndexOf<DiffuseMaterial>())
        , r{ reflectance }
    {
    }

    DiffuseMaterial(TextureIndex texture)
        : Material(Material::TypeIndexOf<DiffuseMaterial>())
        , r{ -1, -1, Float(texture) }
    {
    }

    __GPU__ Vec3 Le(const Intersection& isect, const Vec3& wo) const
    {
        return Vec3(0);
    }

    __GPU__ bool SampleBSDF(Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        Frame f(isect.normal);
        Vec3 wi = SampleCosineHemisphere(u);
        ss->pdf = CosineHemispherePDF(wi.z);
        ss->wi = f.FromLocal(wi);
        ss->is_specular = false;
        ss->s = BSDF(scene, isect);

        return true;
    }

    __GPU__ Float PDF(const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        Frame f(isect.normal);
        Vec3 wi_local = f.ToLocal(wi);
        Vec3 wo_local = f.ToLocal(wo);
        if (!SameHemisphere(wi_local, wo_local))
        {
            return 0;
        }

        return CosineHemispherePDF(AbsCosTheta(wi_local));
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        Frame f(isect.normal);
        Vec3 wi_local = f.ToLocal(wi);
        Vec3 wo_local = f.ToLocal(wo);
        if (!SameHemisphere(wi_local, wo_local))
        {
            return Vec3(0);
        }

        return BSDF(scene, isect);
    }

    Vec3 r;

private:
    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect) const
    {
        if (r.x < 0)
        {
            Point2 uv = triangle::GetTexcoord(scene, isect);
            Vec3 tex = SampleTexture(scene, TextureIndex(r.z), uv);
            return tex * inv_pi;
        }
        else
        {
            return r * inv_pi;
        }
    }
};

class alignas(16) MirrorMaterial : public Material
{
public:
    MirrorMaterial(Vec3 reflectance)
        : Material(Material::TypeIndexOf<MirrorMaterial>())
        , reflectance{ reflectance }
    {
    }

    __GPU__ Vec3 Le(const Intersection& isect, const Vec3& wo) const
    {
        return Vec3(0);
    }

    __GPU__ bool SampleBSDF(Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        ss->s = reflectance;
        ss->wi = Reflect(wo, isect.normal);
        ss->pdf = 1;
        ss->is_specular = true;

        return true;
    }

    __GPU__ Float PDF(const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return 0;
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return Vec3(0);
    }

    Vec3 reflectance;
};

class alignas(16) DielectricMaterial : public Material
{
public:
    DielectricMaterial(Float ior, Vec3 reflectance = Vec3(1))
        : Material(Material::TypeIndexOf<DielectricMaterial>())
        , eta{ ior }
    {
        r[0] = __float2half(reflectance[0]);
        r[1] = __float2half(reflectance[1]);
        r[2] = __float2half(reflectance[2]);
    }

    __GPU__ Vec3 Le(const Intersection& isect, const Vec3& wo) const
    {
        return Vec3(0);
    }

    __GPU__ bool SampleBSDF(Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        Frame f(isect.normal);
        Vec3 wo_local = f.ToLocal(wo);

        // Sample perfect specular dielectric BSDF
        Float R = FresnelSchlick(CosTheta(wo_local), eta);
        Float T = 1 - R;

        // Compute sampling probabilities for reflection and transmission
        Float pr = R;
        Float pt = T;

        ss->s = Vec3(__half2float(r[0]), __half2float(r[1]), __half2float(r[2]));

        if (u[0] < pr / (pr + pt))
        {
            // Sample perfect specular dielectric BRDF
            Vec3 wi(-wo_local.x, -wo_local.y, wo_local.z);

            Vec3 fr(R / AbsCosTheta(wi));
            ss->s *= fr;
            ss->is_specular = true;
            ss->wi = f.FromLocal(wi);
            ss->pdf = pr / (pr + pt);
        }
        else
        {
            // Sample perfect specular dielectric BTDF
            // Compute ray direction for specular transmission
            Vec3 wi;
            Float eta_p;
            if (!Refract(&wi, wo_local, z_axis, eta, &eta_p))
            {
                return false;
            }

            Vec3 ft(T / AbsCosTheta(wi));

            ss->s *= ft;
            ss->is_specular = true;
            ss->wi = f.FromLocal(wi);
            ss->pdf = pt / (pr + pt);
        }

        return true;
    }

    __GPU__ Float PDF(const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return 0;
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return Vec3(0);
    }

    Float eta;
    half r[3];
};

inline __GPU__ Vec3 Material::Le(const Intersection& isect, const Vec3& wo) const
{
    return Dispatch([&](auto mat) { return mat->Le(isect, wo); });
}

inline __GPU__ bool Material::SampleBSDF(
    Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Point2 u
) const
{
    return Dispatch([&](auto mat) { return mat->SampleBSDF(ss, scene, isect, wo, u); });
}

inline __GPU__ Float Material::PDF(const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    return Dispatch([&](auto mat) { return mat->PDF(isect, wo, wi); });
}

inline __GPU__ Vec3 Material::BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    return Dispatch([&](auto mat) { return mat->BSDF(scene, isect, wo, wi); });
}

} // namespace cuwfrt
