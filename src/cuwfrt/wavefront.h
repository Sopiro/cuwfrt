#pragma once

#include "common.h"
#include "cuwfrt/geometry/intersection.h"

namespace cuwfrt
{

struct WavefrontRay
{
    RNG rng;

    Ray ray;
    Intersection isect;

    Vec3 beta;
    int32 bounce;

    Float last_pdf;
    bool is_specular;

    int32 pixel_index;
};

struct WavefrontShadowRay
{
    Ray ray;

    Vec3 L;
    Float visibility;

    int32 pixel_index;
};

struct WavefrontResources
{
    int32 ray_capacity;
    WavefrontRay* rays_active;
    WavefrontRay* rays_next;
    WavefrontShadowRay* shadow_rays;

    int32* active_ray_count;
    int32* next_ray_count;
    int32* shadow_ray_count;

    void Init(Point2i res);
    void Free();
    void Resize(Point2i res);
};

} // namespace cuwfrt
