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

} // namespace cuwfrt
