#pragma once

#include <cuda_runtime.h>

#include "cuwfrt/common.h"
#include "cuwfrt/cuda_api.h"
#include "cuwfrt/util/polymorphic_vector.h"

namespace cuwfrt
{

template <typename Base, typename... Types>
inline __GPU__ Base* GetPolymorphicObject(
    const uint8* vectors, const int32* offsets, typename PolymorphicVector<Base, TypePack<Types...>>::Index index
)
{
    using TypePack = TypePack<Types...>;
    using Handler = Base* (*)(const uint8*, const int32*, int32);

    constexpr static Handler handlers[] = { [](const uint8* vectors, const int32* offsets, int32 element_index) -> Base* {
        return (Types*)(vectors + offsets[detail::IndexOf<Types, TypePack>::value]) + element_index;
    }... };

    return handlers[index.type_index](vectors, offsets, index.element_index);
}

inline __GPU__ void AtomicAdd(Vec4* a, Vec3 b)
{
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
    atomicAdd(&a->z, b.z);
}

inline __GPU__ Vec4 ToLinearRGB(Vec4 srgb)
{
    return Vec4(std::pow(srgb.x, 2.2f), std::pow(srgb.y, 2.2f), std::pow(srgb.z, 2.2f), 1.0f);
}

inline __GPU__ Vec4 ToSRGB(Vec4 rgb)
{
    return Vec4(std::pow(rgb.x, 1 / 2.2f), std::pow(rgb.y, 1 / 2.2f), std::pow(rgb.z, 1 / 2.2f), 1.0f);
}

// https://en.wikipedia.org/wiki/Y%E2%80%B2UV#Conversion_to/from_RGB
inline __GPU__ Vec4 RGBtoYUV(Vec4 rgb)
{
    constexpr Mat4 m(
        Vec4{ 0.299f, -0.14713f, 0.615f, 0.0f }, Vec4{ 0.587f, -0.28886f, -0.51499f, 0.0f },
        Vec4{ 0.114f, 0.436f, -0.10001f, 0.0f }, Vec4{ 0.0f, 0.0f, 0.0f, 0.0f }
    );

    return Mul(m, rgb);
}

inline __GPU__ Vec4 YUVtoRGB(Vec4 yuv)
{
    constexpr Mat4 m(
        Vec4{ 1.0, 1.0f, 1.0f, 0.0f }, Vec4{ 0.0f, -0.39465f, 2.03211f, 0.0f }, Vec4{ 1.13983f, -0.58060f, 0.0f, 0.0f },
        Vec4{ 0.0f, 0.0f, 0.0f, 0.0f }
    );

    return Mul(m, yuv);
}

inline __GPU__ Vec2 GetVec2(Vec4 v)
{
    return Vec2(v.x, v.y);
}

inline __GPU__ Vec3 GetVec3(Vec4 v)
{
    return Vec3(v.x, v.y, v.z);
}

template <typename T>
inline __GPU__ T Min(T a)
{
    return a;
}

template <typename T, typename... Args>
inline __GPU__ T Min(T a, Args... args)
{
    T b = Min(args...);
    return (a < b) ? a : b;
}

template <typename T>
inline __GPU__ T Max(T a)
{
    return a;
}

template <typename T, typename... Args>
inline __GPU__ T Max(T a, Args... args)
{
    T b = Max(args...);
    return (a > b) ? a : b;
}

} // namespace cuwfrt
