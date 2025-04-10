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

inline __GPU__ void AtomicAdd(Vec4* a, const Vec3& b)
{
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
    atomicAdd(&a->z, b.z);
}

} // namespace cuwfrt
