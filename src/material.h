#pragma once

#include "common.h"
#include "indices.h"

namespace cuwfrt
{

struct Material
{
    bool is_light = false;
    Vec3 reflectance = Vec3::zero;
    TextureIndex texture = -1;
};

} // namespace cuwfrt
