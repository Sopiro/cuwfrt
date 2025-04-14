#pragma once

#include "cuwfrt/material/material.h"

#include "cuwfrt/geometry/intersection.h"
#include "cuwfrt/shading/microfacet.h"
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

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
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

    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const
    {
        return false;
    }

    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return 0;
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return Vec3(0);
    }

    __GPU__ Vec3 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        return Vec3(1);
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

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        return Vec3(0);
    }

    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const
    {
        Frame f(isect.shading_normal);
        Vec3 wi = SampleCosineHemisphere(u12);
        ss->pdf = CosineHemispherePDF(wi.z);
        ss->wi = f.FromLocal(wi);
        ss->is_specular = false;
        ss->s = Lambertian(scene, isect);

        return true;
    }

    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        Frame f(isect.shading_normal);
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
        Frame f(isect.shading_normal);
        Vec3 wi_local = f.ToLocal(wi);
        Vec3 wo_local = f.ToLocal(wo);
        if (!SameHemisphere(wi_local, wo_local))
        {
            return Vec3(0);
        }

        return Lambertian(scene, isect);
    }

    __GPU__ Vec3 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        if (r.x < 0)
        {
            Point2 uv = triangle::GetTexcoord(scene, isect);
            Vec3 tex = SampleTexture(scene, TextureIndex(r.z), uv);
            return tex;
        }
        else
        {
            return r;
        }
    }

    Vec3 r;

private:
    __GPU__ Vec3 Lambertian(const GPUScene* scene, const Intersection& isect) const
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

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        return Vec3(0);
    }

    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const
    {
        ss->s = reflectance;
        ss->wi = Reflect(wo, isect.shading_normal);
        ss->pdf = 1;
        ss->is_specular = true;

        return true;
    }

    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return 0;
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return Vec3(0);
    }

    __GPU__ Vec3 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        return reflectance;
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

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        return Vec3(0);
    }

    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const
    {
        Frame f(isect.shading_normal);
        Vec3 wo_local = f.ToLocal(wo);

        // Sample perfect specular dielectric BSDF
        Float R = FresnelSchlick(CosTheta(wo_local), eta);
        Float T = 1 - R;

        // Compute sampling probabilities for reflection and transmission
        Float pr = R;
        Float pt = T;

        ss->s = Vec3(__half2float(r[0]), __half2float(r[1]), __half2float(r[2]));

        if (u0 < pr / (pr + pt))
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

    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return 0;
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        return Vec3(0);
    }

    __GPU__ Vec3 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        return rho(this, scene, isect, wo);
    }

    Float eta;
    half r[3];
};

class alignas(16) PBRMaterial : public Material
{
public:
    PBRMaterial(TextureIndex basecolor, TextureIndex matallic, TextureIndex roughness, TextureIndex emissive = -1)
        : Material(Material::TypeIndexOf<PBRMaterial>())
        , tex_basecolor{ basecolor }
        , tex_metallic{ matallic }
        , tex_roughness{ roughness }
        , tex_emissive{ emissive }
    {
    }

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        if (tex_emissive > 0)
        {
            Point2 uv = triangle::GetTexcoord(scene, isect);
            return SampleTexture(scene, tex_emissive, uv);
        }
        else
        {
            return Vec3(0);
        }
    }

    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const
    {
        Frame f(isect.shading_normal);

        Vec3 wo_local = f.ToLocal(wo);
        if (wo_local.z == 0)
        {
            return false;
        }

        Point2 uv = triangle::GetTexcoord(scene, isect);
        Vec3 basecolor = SampleTexture(scene, tex_basecolor, uv);
        Float metallic = SampleTexture(scene, tex_metallic, uv).z;
        Float roughness = SampleTexture(scene, tex_roughness, uv).y;
        Float alpha = mf::RoughnessToAlpha(roughness);

        constexpr Vec3 coefficient(0.2126f, 0.7152f, 0.0722f);

        Vec3 f0 = mf::F0(basecolor, metallic);
        Vec3 F = mf::F_Schlick(f0, Dot(wo, isect.shading_normal));
        Float diff_weight = (1 - metallic);
        Float spec_weight = Dot(coefficient, F);
        // Float spec_weight = std::fmax(F.x, std::fmax(F.y, F.z));
        Float t = Clamp(spec_weight / (diff_weight + spec_weight), 0.15f, 0.9f);

        Vec3 wm, wi;
        if (u0 < t)
        {
            // Sample glossy
            wm = mf::Sample_Wm(wo_local, alpha, alpha, u12);
            wi = Reflect(wo_local, wm);

            if (!SameHemisphere(wo_local, wi))
            {
                return false;
            }
        }
        else
        {
            // Sample diffuse
            wi = SampleCosineHemisphere(u12);
            wm = Normalize(wi + wo_local);
        }

        Float cos_theta_o = AbsCosTheta(wo_local);
        Float cos_theta_i = AbsCosTheta(wi);
        if (cos_theta_i == 0 || cos_theta_o == 0)
        {
            return false;
        }

        Vec3 f_s = F * mf::D(wm, alpha, alpha) * mf::G(wo_local, wi, alpha, alpha) / (4 * cos_theta_i * cos_theta_o);
        Vec3 f_d = (Vec3(1) - F) * (1 - metallic) * (basecolor * inv_pi);

        Float p_s = mf::D(wo_local, wm, alpha, alpha) / (4 * AbsDot(wo_local, wm));
        Float p_d = cos_theta_i * inv_pi;

        ss->is_specular = false;
        ss->s = f_s + f_d;
        ss->wi = f.FromLocal(wi);
        ss->pdf = t * p_s + (1 - t) * p_d;
        return true;
    }

    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        Frame f(isect.shading_normal);
        Vec3 wi_local = f.ToLocal(wi);
        Vec3 wo_local = f.ToLocal(wo);
        if (!SameHemisphere(wi_local, wo_local))
        {
            return 0;
        }

        Vec3 wm = wo_local + wi_local;
        if (Length2(wm) == 0)
        {
            return 0;
        }
        wm.Normalize();

        if (Dot(wm, Vec3(0, 0, 1)) < 0)
        {
            wm.Negate();
        }

        Point2 uv = triangle::GetTexcoord(scene, isect);
        Vec3 basecolor = SampleTexture(scene, tex_basecolor, uv);
        Float metallic = SampleTexture(scene, tex_metallic, uv).z;
        Float roughness = SampleTexture(scene, tex_roughness, uv).y;
        Float alpha = mf::RoughnessToAlpha(roughness);

        constexpr Vec3 coefficient(0.2126f, 0.7152f, 0.0722f);

        Vec3 f0 = mf::F0(basecolor, metallic);
        Vec3 F = mf::F_Schlick(f0, Dot(wo, isect.shading_normal));
        Float diff_weight = (1 - metallic);
        Float spec_weight = Dot(coefficient, F);
        // Float spec_weight = std::fmax(F.x, std::fmax(F.y, F.z));
        Float t = Clamp(spec_weight / (diff_weight + spec_weight), 0.15f, 0.9f);

        Float p_s = mf::PDF(wo_local, wm, alpha, alpha) / (4 * AbsDot(wo_local, wm));
        Float p_d = AbsCosTheta(wi_local) * inv_pi;

        return t * p_s + (1 - t) * p_d;
    }

    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
    {
        Frame f(isect.shading_normal);
        Vec3 wi_local = f.ToLocal(wi);
        Vec3 wo_local = f.ToLocal(wo);
        if (!SameHemisphere(wi_local, wo_local))
        {
            return Vec3(0);
        }

        Float cos_theta_o = AbsCosTheta(wo_local);
        Float cos_theta_i = AbsCosTheta(wi_local);
        if (cos_theta_i == 0 || cos_theta_o == 0)
        {
            return Vec3(0);
        }

        Vec3 wm = wo_local + wi_local;
        if (Length2(wm) == 0)
        {
            return Vec3(0);
        }
        wm.Normalize();

        Point2 uv = triangle::GetTexcoord(scene, isect);
        Vec3 basecolor = SampleTexture(scene, tex_basecolor, uv);
        Float metallic = SampleTexture(scene, tex_metallic, uv).z;
        Float roughness = SampleTexture(scene, tex_roughness, uv).y;
        Float alpha = mf::RoughnessToAlpha(roughness);

        Vec3 f0 = mf::F0(basecolor, metallic);
        Vec3 F = mf::F_Schlick(f0, Dot(wi_local, wm));

        Vec3 f_s = F * mf::D(wm, alpha, alpha) * mf::G(wo_local, wi_local, alpha, alpha) / (4 * cos_theta_i * cos_theta_o);
        Vec3 f_d = (Vec3(1) - F) * (1 - metallic) * (basecolor * inv_pi);

        return f_d + f_s;
    }

    __GPU__ Vec3 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
    {
        return rho(this, scene, isect, wo);
    }

    TextureIndex tex_basecolor, tex_metallic, tex_roughness, tex_emissive;
    TextureIndex padding[2];
};

inline __GPU__ Vec3 Material::Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return Dispatch([&](auto mat) { return mat->Le(scene, isect, wo); });
}

inline __GPU__ bool Material::SampleBSDF(
    Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
) const
{
    return Dispatch([&](auto mat) { return mat->SampleBSDF(ss, scene, isect, wo, u0, u12); });
}

inline __GPU__ Float Material::PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    return Dispatch([&](auto mat) { return mat->PDF(scene, isect, wo, wi); });
}

inline __GPU__ Vec3 Material::BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const
{
    return Dispatch([&](auto mat) { return mat->BSDF(scene, isect, wo, wi); });
}

inline __GPU__ Vec3 Material::Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return Dispatch([&](auto mat) { return mat->Albedo(scene, isect, wo); });
}

} // namespace cuwfrt
