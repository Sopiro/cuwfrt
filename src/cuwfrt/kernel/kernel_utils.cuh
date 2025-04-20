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

inline __GPU__ Vec4 ToSRGB(Vec4 color)
{
    return Vec4(std::pow(color.x, 1 / 2.2f), std::pow(color.y, 1 / 2.2f), std::pow(color.z, 1 / 2.2f), 1.0f);
}

inline __GPU__ Vec3 GetVec3(Vec4 v)
{
    return Vec3(v.x, v.y, v.z);
}

} // namespace cuwfrt
