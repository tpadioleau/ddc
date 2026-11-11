// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Type traits and concepts identifying chunk types (Chunk, ChunkSpan).
#pragma once

#include <type_traits>
#include <utility>

namespace ddc {

//! @brief Customization point opting a type in as a borrowed chunk.
//!
//! @details Defaults to `false`. ChunkSpan specializes it to `true`: a ChunkSpan does not own its
//! data, so it remains valid after the ChunkSpan itself is destroyed.
//!
//! @tparam T The type to query.
//! @ingroup ddc_containers
template <class T>
inline constexpr bool enable_borrowed_chunk = false;

//! @brief Customization point opting a type in as a chunk.
//!
//! @details Defaults to `false`. Chunk and ChunkSpan specialize it to `true`.
//!
//! @tparam T The type to query.
//! @ingroup ddc_containers
template <class T>
inline constexpr bool enable_chunk = false;

//! @brief Tells whether a type is a chunk (Chunk or ChunkSpan).
//!
//! @details `const` and reference qualifiers of `T` are ignored.
//!
//! @tparam T The type to query.
//! @ingroup ddc_containers
template <class T>
inline constexpr bool is_chunk_v = enable_chunk<std::remove_const_t<std::remove_reference_t<T>>>;

//! @brief Tells whether a type is a chunk that can be safely used from a temporary.
//!
//! @details `true` if `T` is a chunk and either `T` is an lvalue reference or
//! enable_borrowed_chunk is `true` for the unqualified type, similarly to
//! `std::ranges::borrowed_range`.
//!
//! @tparam T The type to query.
//! @ingroup ddc_containers
template <class T>
inline constexpr bool is_borrowed_chunk_v
        = is_chunk_v<T>
          && (std::is_lvalue_reference_v<T>
              || enable_borrowed_chunk<std::remove_cv_t<std::remove_reference_t<T>>>);

namespace concepts {

//! @brief Satisfied by the types `T` for which is_borrowed_chunk_v is `true`.
//!
//! @tparam T The type to constrain.
//! @ingroup ddc_containers
template <class T>
concept borrowed_chunk = is_borrowed_chunk_v<T>;

}

//! @brief Gives access to the value, pointer and reference types of a chunk.
//!
//! @details `T` must be a chunk type (checked by a `static_assert`). The types are deduced from
//! the return type of `T::data_handle()`.
//!
//! @tparam T The chunk type.
//! @ingroup ddc_containers
template <class T>
struct chunk_traits
{
    static_assert(is_chunk_v<T>);
    using value_type
            = std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<T>().data_handle())>>;
    using pointer_type = decltype(std::declval<T>().data_handle());
    using reference_type = decltype(*std::declval<T>().data_handle());
};

//! @brief Type of the values stored in the chunk `T`, without `const` qualification.
//!
//! @tparam T The chunk type.
//! @ingroup ddc_containers
template <class T>
using chunk_value_t = chunk_traits<T>::value_type;

//! @brief Type of the pointer returned by `data_handle()` for the chunk `T`.
//!
//! @tparam T The chunk type.
//! @ingroup ddc_containers
template <class T>
using chunk_pointer_t = chunk_traits<T>::pointer_type;

//! @brief Type obtained by dereferencing the pointer of the chunk `T`.
//!
//! @tparam T The chunk type.
//! @ingroup ddc_containers
template <class T>
using chunk_reference_t = chunk_traits<T>::reference_type;

//! @brief Tells whether the elements of a chunk can be modified.
//!
//! @details `true` if the pointer type of the chunk `T` does not point to a `const` type.
//!
//! @tparam T The chunk type.
//! @ingroup ddc_containers
template <class T>
inline constexpr bool is_writable_chunk_v
        = !std::is_const_v<std::remove_pointer_t<chunk_pointer_t<T>>>;

} // namespace ddc
