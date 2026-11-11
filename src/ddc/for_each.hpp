// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Serial iteration over a domain, on the host or inside a Kokkos function.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Macros.hpp>

#include "discrete_vector.hpp"

namespace ddc {

namespace detail {

template <class Support, std::size_t N, class Functor, class... Is>
void host_for_each_serial(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& size,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(domain(typename Support::discrete_vector_type(is...)));
    } else {
        for (DiscreteVectorElement ii = 0; ii < size[I]; ++ii) {
            host_for_each_serial(domain, size, f, is..., ii);
        }
    }
}

template <class Support, std::size_t N, class Functor, class... Is>
KOKKOS_FUNCTION void device_for_each_serial(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& size,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(domain(typename Support::discrete_vector_type(is...)));
    } else {
        for (DiscreteVectorElement ii = 0; ii < size[I]; ++ii) {
            device_for_each_serial(domain, size, f, is..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nD domain in serial
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 *
 * The elements are visited in lexicographic order: the last dimension of the domain varies the
 * fastest. The functor is called with the discrete element of the domain at each position.
 *
 * @ingroup ddc_algorithms
 */
template <class Support, class Functor>
void host_for_each(Support const& domain, Functor&& f) noexcept
{
    detail::host_for_each_serial(domain, detail::array(domain.extents()), std::forward<Functor>(f));
}

/** iterates over a nD domain in serial
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 *
 * Same as ddc::host_for_each, but it is a `KOKKOS_FUNCTION`: it can be called from code running
 * on a device (the functor must then be callable there too).
 *
 * @ingroup ddc_algorithms
 */
template <class Support, class Functor>
KOKKOS_FUNCTION void device_for_each(Support const& domain, Functor&& f) noexcept
{
    detail::device_for_each_serial(
            domain,
            detail::array(domain.extents()),
            std::forward<Functor>(f));
}

} // namespace ddc
