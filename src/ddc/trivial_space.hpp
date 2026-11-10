// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

//! @file trivial_space.hpp
//! @brief Provides helpers to define discrete dimensions that carry no attribute (no stored data),
//! only an index range.

#include "discrete_dimension.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"

namespace ddc {

//! @addtogroup ddc_samplings
//! @{

/** Construct a bounded dimension without attributes.
 *
 * @param n number of elements
 * @return a DiscreteDomain of size `n`
 */
template <concepts::discrete_dimension DDim>
constexpr DiscreteDomain<DDim> init_trivial_bounded_space(DiscreteVector<DDim> const n) noexcept
{
    return DiscreteDomain<DDim>(create_reference_discrete_element<DDim>(), n);
}

/** Construct a half bounded dimension without attributes.
 *
 * @tparam DDim the discrete dimension
 * @return the first DiscreteElement of the dimension
 */
template <concepts::discrete_dimension DDim>
constexpr DiscreteElement<DDim> init_trivial_half_bounded_space() noexcept
{
    return create_reference_discrete_element<DDim>();
}

//! @}

} // namespace ddc
