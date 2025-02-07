#pragma once

#include <string>

namespace cuwfrt
{

struct TextureDesc
{
    std::string filename;
    bool non_color;

    bool operator==(const TextureDesc& o) const = default;
    bool operator!=(const TextureDesc& o) const = default;
};

} // namespace cuwfrt
