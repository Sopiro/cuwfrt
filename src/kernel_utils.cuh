#pragma once

#include "cuda_api.h"
#include "polymorphic_vector.h"

namespace cuwfrt
{

template <typename Base, typename... Types>
inline __GPU__ Base* GetPolymorphicObject(
    const std::array<uint8_t*, TypePack<Types...>::count>& vectors,
    typename PolymorphicVector<Base, TypePack<Types...>>::Index index
)
{
    using TypePack = TypePack<Types...>;
    using Handler = Base* (*)(uint8_t*, int32_t);

    constexpr static Handler handlers[] = { [](uint8_t* v, int32_t element_index) -> Base* {
        return reinterpret_cast<Types*>(v) + element_index;
    }... };

    return handlers[index.type_index](vectors[index.type_index], index.element_index);
}

} // namespace cuwfrt
