#pragma once

#include "common.h"

namespace cuwfrt
{

struct WavefrontRay
{
    Vec3 origin;
    Vec3 direction;

    Vec3 beta;
    int32 bounce;

    Float hit_t;
    int32 hit_idx;

    Float last_pdf;
    bool is_specular;

    int32 pixel_index;
};

struct WavefrontShadowRay
{
    Vec3 origin;
    Vec3 direction;

    Vec3 L;
    Float visibility;

    int32 pixel_index;
};

} // namespace cuwfrt
