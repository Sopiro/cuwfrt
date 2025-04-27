#pragma once

#include "cuwfrt/common.h"

namespace cuwfrt
{

enum TextureType
{
    constant_texture = 0,
    image_texture = 1
};

struct TextureDesc
{
    TextureType type = constant_texture;

    Vec3 color = Vec3(1, 0, 1);

    std::string filename;
    bool non_color;

    bool operator==(const TextureDesc& o) const = default;
    bool operator!=(const TextureDesc& o) const = default;
};

using TextureIndex = int16;

} // namespace cuwfrt
