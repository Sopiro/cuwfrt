#pragma once

#include "frame.h"
#include "intersection.h"
#include "kernel_texture.cuh"
#include "material.h"
#include "sampling.h"
#include "texture.cuh"
#include "triangle.h"

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

    __GPU__ bool Scatter(SurfaceScattering* ss, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        if (r.x < 0)
        {
            Point2 uv = GetTexcoord(isect.scene, isect.prim, isect.uvw);
            Vec3 tex = SampleTexture(isect.scene, TextureIndex(r.z), uv);
            ss->atten = tex;
        }
        else
        {
            ss->atten = r;
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

    __GPU__ bool Scatter(SurfaceScattering* ss, const Intersection& isect, const Vec3& wo, Point2 u) const
    {
        WakNotUsed(ss);
        WakNotUsed(isect);
        WakNotUsed(wo);
        WakNotUsed(u);
        return false;
    }

    Vec3 emission;
};

inline __GPU__ Vec3 Material::Le(const Intersection& isect, const Vec3& wo) const
{
    return Dispatch([&](auto mat) { return mat->Le(isect, wo); });
}

inline __GPU__ bool Material::Scatter(SurfaceScattering* ss, const Intersection& isect, const Vec3& wo, Point2 u) const
{
    return Dispatch([&](auto mat) { return mat->Scatter(ss, isect, wo, u); });
}

} // namespace cuwfrt
