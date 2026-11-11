// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Parallel copy between chunks having the same domain.

#pragma once

#include <cassert>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "chunk_traits.hpp"

namespace ddc {

/** Copy the content of a borrowed chunk into another
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in]  src the borrowed chunk from which to copy
 * @return dst as a ChunkSpan
 *
 * The copy is performed by `Kokkos::deep_copy` on the allocation views, without an execution
 * space instance.
 *
 * @pre The domains of `dst` and `src` must have the same type (`static_assert`) and be equal
 *      (checked by `assert`).
 * @pre The element type of `src` must be assignable to the one of `dst` (`static_assert`).
 *
 * @ingroup ddc_algorithms
*/
template <concepts::borrowed_chunk ChunkDst, concepts::borrowed_chunk ChunkSrc>
auto parallel_deepcopy(ChunkDst&& dst, ChunkSrc&& src)
{
    static_assert(
            std::is_assignable_v<chunk_reference_t<ChunkDst>, chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    static_assert(std::is_same_v<decltype(dst.domain()), decltype(src.domain())>);
    assert(dst.domain() == src.domain());
    Kokkos::deep_copy(dst.allocation_kokkos_view(), src.allocation_kokkos_view());
    return dst.span_view();
}

/** Copy the content of a borrowed chunk into another
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in]  src the borrowed chunk from which to copy
 * @return dst as a ChunkSpan
 *
 * The copy is performed by `Kokkos::deep_copy` on the allocation views, using `execution_space`.
 *
 * @pre The domains of `dst` and `src` must have the same type (`static_assert`) and be equal
 *      (checked by `assert`).
 * @pre The element type of `src` must be assignable to the one of `dst` (`static_assert`).
 *
 * @ingroup ddc_algorithms
*/
template <class ExecSpace, concepts::borrowed_chunk ChunkDst, concepts::borrowed_chunk ChunkSrc>
auto parallel_deepcopy(ExecSpace const& execution_space, ChunkDst&& dst, ChunkSrc&& src)
{
    static_assert(
            std::is_assignable_v<chunk_reference_t<ChunkDst>, chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    static_assert(
            std::is_same_v<decltype(dst.domain()), decltype(src.domain())>,
            "ddc::parallel_deepcopy only supports domains whose dimensions are of the same order");
    assert(dst.domain() == src.domain());
    Kokkos::deep_copy(execution_space, dst.allocation_kokkos_view(), src.allocation_kokkos_view());
    return dst.span_view();
}

} // namespace ddc
