#pragma once

#include "cuwfrt/geometry/intersection.h"
#include "cuwfrt/kernel/kernel_primitive.cuh"
#include "cuwfrt/kernel/kernel_texture.cuh"
#include "cuwfrt/material/material.h"
#include "cuwfrt/shading/frame.h"
#include "cuwfrt/shading/sampling.h"
#include "cuwfrt/texture/texture.cuh"

namespace cuwfrt
{

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
        return Vec3(0, 0, 0);
    }

    __GPU__ bool Scatter(SurfaceScattering* ss, const GPUData* scene, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        if (r.x < 0)
        {
            Point2 uv = GetTexcoord(scene, isect);
            Vec3 tex = SampleTexture(scene, TextureIndex(r.z), uv);
            ss->atten = tex * inv_pi;
        }
        else
        {
            ss->atten = r * inv_pi;
        }

        Frame f(isect.normal);
        ss->wi = f.FromLocal(SampleCosineHemisphere(u));

        return true;
    }

    Vec3 r;
};

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
        WakNotUsed(isect);
        WakNotUsed(wo);
        return emission;
    }

    __GPU__ bool Scatter(SurfaceScattering* ss, const GPUData* scene, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        WakNotUsed(ss);
        WakNotUsed(scene);
        WakNotUsed(isect);
        WakNotUsed(wo);
        WakNotUsed(u);
        return false;
    }

    Vec3 emission;
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
        WakNotUsed(isect);
        WakNotUsed(wo);
        return Vec3(0, 0, 0);
    }

    __GPU__ bool Scatter(SurfaceScattering* ss, const GPUData* scene, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        ss->atten = reflectance;
        ss->wi = Reflect(wo, isect.normal);
        return true;
    }

    Vec3 reflectance;
};

inline __GPU__ Vec3 Material::Le(const Intersection& isect, const Vec3& wo) const
{
    return Dispatch([&](auto mat) { return mat->Le(isect, wo); });
}

inline __GPU__ bool Material::Scatter(
    SurfaceScattering* ss, const GPUData* scene, const Intersection& isect, const Vec3& wo, Point2 u
) const
{
    return Dispatch([&](auto mat) { return mat->Scatter(ss, scene, isect, wo, u); });
}

} // namespace cuwfrt
