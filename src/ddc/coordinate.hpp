// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

//! @file coordinate.hpp
//! @brief Defines `ddc::Coordinate`, the tagged coordinate type of the continuous space, and the
//! `ddc::coordinate` function mapping discrete elements to coordinates.

#include <utility>

#include <Kokkos_Macros.hpp>

#include "detail/tagged_vector.hpp"

#include "discrete_dimension.hpp"
#include "discrete_element.hpp"
#include "real_type.hpp"

namespace ddc {

//! @addtogroup ddc_samplings
//! @{

/** A CoordinateElement the type of the scalar used to represent elements of coordinates in the
 * continuous space.
 */
using CoordinateElement = Real;

/** A Coordinate represents a coordinate in the continuous space
 *
 * It is tagged by its dimensions.
 */

template <class... CDims>
using Coordinate = detail::TaggedVector<CoordinateElement, CDims...>;

//! @brief Gives the coordinate of a multi-dimensional discrete element, one component per
//! discrete dimension.
//!
//! Each component is obtained by calling `coordinate` on the corresponding one-dimensional discrete
//! element, so each discrete space involved must have been initialised.
//! Only available for two or more discrete dimensions.
//! @param c the discrete element
//! @return the coordinate tagged by the continuous dimensions `DDims::continuous_dimension_type...`
//! @note Callable from device code (`KOKKOS_FUNCTION`).
template <concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION Coordinate<typename DDims::continuous_dimension_type...> coordinate(
        DiscreteElement<DDims...> const& c)
    requires(sizeof...(DDims) > 1)
{
    return Coordinate<typename DDims::continuous_dimension_type...>(
            coordinate(DiscreteElement<DDims>(c))...);
}

//! @brief Gives the type of the coordinates of a discrete element type, through the member `type`.
//!
//! The type is the result type of `coordinate` applied to a `T`.
//! @tparam T a discrete element type
// Gives access to the type of the coordinates of a discrete element
// Example usage : "using Coords = coordinate_of_t<DElem>;"
template <concepts::discrete_element T>
struct coordinate_of
{
    using type = decltype(coordinate(std::declval<T>()));
};

/// Helper type of \ref ddc::coordinate_of
/// @tparam T a discrete element type
template <concepts::discrete_element T>
using coordinate_of_t = coordinate_of<T>::type;

//! @}

} // namespace ddc
