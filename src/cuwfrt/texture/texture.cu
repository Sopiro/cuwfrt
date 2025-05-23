#include "texture.h"

#include "alzartak/image.h"
#include "cuwfrt/scene/gpu_scene.h"

namespace cuwfrt
{

using namespace alzartak;

Texture::Texture(const TextureDesc& td)
{
    Image4 image;
    switch (td.type)
    {
    case constant_texture:
    {
        image = Image4(1, 1);
        image[0] = Vec4(td.color, 1);
    }
    break;
    case image_texture:
    {
        image = alzartak::ReadImage4(td.filename, td.non_color);
        if (!image)
        {
            std::cout << "Faild to read texture: " << td.filename << std::endl;
            image = Image4(1, 1);
            image[0] = Vec4(1, 0, 1, 1);
        }
    }
    break;
    default:
        WakAssert(false);
        break;
    }

    // Create CUDA array and texture objext
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&cu_array, &channel_desc, image.width, image.height);
    cudaMemcpy2DToArray(
        cu_array, 0, 0, image.data.get(), image.width * sizeof(float4), image.width * sizeof(float4), image.height,
        cudaMemcpyHostToDevice
    );

    // Set image resouce
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    // Set texture parameters
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap; // wrap mode
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;    // filter mode
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
}

Texture::~Texture()
{
    cudaFreeArray(cu_array);
    cudaDestroyTextureObject(tex_obj);
}

} // namespace cuwfrt
