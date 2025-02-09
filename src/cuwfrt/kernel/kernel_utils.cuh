#pragma once

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
    using Handler = Base* (*)(const uint8_t*, const int32*, int32_t);

    constexpr static Handler handlers[] = { [](const uint8* vectors, const int32* offsets, int32_t element_index) -> Base* {
        return (Types*)(vectors + offsets[detail::IndexOf<Types, TypePack>::value]) + element_index;
    }... };

    return handlers[index.type_index](vectors, offsets, index.element_index);
}

} // namespace cuwfrt
