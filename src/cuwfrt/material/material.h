#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"
#include "cuwfrt/util/polymorphic_vector.h"

namespace cuwfrt
{

struct Intersection;
struct GPUScene;

struct Scattering
{
    Vec3 s;

    Vec3 wi;
    Float pdf;

    bool is_specular;
};

using Materials = TypePack<
    class DiffuseLightMaterial,
    class DiffuseMaterial,
    class MirrorMaterial,
    class DielectricMaterial,
    class PBRMaterial>;

class Material : public DynamicDispatcher<Materials>
{
public:
    using Types = Materials;

protected:
    Material(int32 type_index)
        : DynamicDispatcher{ type_index }
    {
    }

public:
    __GPU__ Vec3 Le(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;
    __GPU__ bool SampleBSDF(
        Scattering* s, const GPUScene* scene, const Intersection& isect, const Vec3& wo, Float u0, Point2 u12
    ) const;
    __GPU__ Float PDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec3 BSDF(const GPUScene* scene, const Intersection& isect, const Vec3& wo, const Vec3& wi) const;
    __GPU__ Vec4 Albedo(const GPUScene* scene, const Intersection& isect, const Vec3& wo) const;
};

using MaterialIndex = PolymorphicVector<Material>::Index;

} // namespace cuwfrt
