#pragma once

#include "cuwfrt/common.h"
#include "cuwfrt/util/polymorphic_vector.h"

namespace cuwfrt
{

struct Intersection;

struct SurfaceScattering
{
    Vec3 atten;
    Vec3 wi;
};

using Materials = TypePack<class DiffuseLightMaterial, class DiffuseMaterial, class MirrorMaterial>;

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
    __GPU__ Vec3 Le(const Intersection& isect, const Vec3& wo) const;
    __GPU__ bool Scatter(SurfaceScattering* ss, const Intersection& isect, const Vec3& wo, Point2 u) const;
};

using MaterialIndex = PolymorphicVector<Material>::Index;

} // namespace cuwfrt
