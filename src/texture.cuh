#pragma once

#include "texture_desc.h"

namespace cuwfrt
{

struct Texture
{
    Texture(const TextureDesc& td);
    ~Texture();

    cudaTextureObject_t tex_obj;

private:
    cudaArray* cu_array;
};

} // namespace cuwfrt
