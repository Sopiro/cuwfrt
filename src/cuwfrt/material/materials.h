#pragma once

#include "cuwfrt/material/material.h"

#include "cuwfrt/geometry/intersection.h"
#include "cuwfrt/shading/microfacet.h"
#include "cuwfrt/shading/scattering.h"

#if __CUDACC__
#include "cuwfrt/kernel/device_primitive.cuh"
#include "cuwfrt/kernel/device_texture.cuh"
#endif

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

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;
    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const;
    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec4 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;

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

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;
    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const;
    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec4 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;

    Vec3 r;

private:
    __GPU__ Vec3 Lambertian(const GPUScene* scene, const Intersection& isect) const;
};

class alignas(16) MirrorMaterial : public Material
{
public:
    MirrorMaterial(Vec3 reflectance)
        : Material(Material::TypeIndexOf<MirrorMaterial>())
        , reflectance{ reflectance }
    {
    }

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;
    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const;
    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec4 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;

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

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;
    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const;
    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec4 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;

    Float eta;
    half r[3];
};

class alignas(16) MetallicRoughnessMaterial : public Material
{
public:
    MetallicRoughnessMaterial(TextureIndex basecolor, TextureIndex matallic, TextureIndex roughness, TextureIndex emissive = -1)
        : Material(Material::TypeIndexOf<MetallicRoughnessMaterial>())
        , tex_basecolor{ basecolor }
        , tex_metallic{ matallic }
        , tex_roughness{ roughness }
        , tex_emissive{ emissive }
    {
    }

    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;
    __GPU__ bool SampleBSDF(
        Scattering* ss, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const;
    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec4 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;

    TextureIndex tex_basecolor, tex_metallic, tex_roughness, tex_emissive;
    TextureIndex padding[2];
};

#ifdef __CUDACC__
#include "dielectric_material.cuh"
#include "diffuse_light_material.cuh"
#include "diffuse_material.cuh"
#include "metallic_roughness_material.cuh"
#include "mirror_material.cuh"
#endif

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

inline __GPU__ Vec4 Material::Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const
{
    return Dispatch([&](auto mat) { return mat->Albedo(scene, isect, wo); });
}

} // namespace cuwfrt
