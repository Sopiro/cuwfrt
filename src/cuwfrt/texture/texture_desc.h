#pragma once

#include "cuwfrt/common.h"

namespace cuwfrt
{

struct TextureDesc
{
    std::string filename;
    bool non_color;

    bool is_constant;
    Vec3 color;

    bool operator==(const TextureDesc& o) const = default;
    bool operator!=(const TextureDesc& o) const = default;
};

using TextureIndex = int32;

} // namespace cuwfrt
