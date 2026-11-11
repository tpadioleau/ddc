// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides the discrete dimension used to store derivatives of a function.

#pragma once

#include <ddc/ddc.hpp>

namespace ddc {

/**
 * @brief A templated struct representing a discrete dimension storing
 * the derivatives of a function along a continuous dimension CDim.
 *
 * @tparam CDim The continuous dimension along which the derivatives are taken.
 *
 * @ingroup ddc_splines
 */
template <class CDim>
struct Deriv : ddc::DiscreteDimension
{
};

} // namespace ddc
