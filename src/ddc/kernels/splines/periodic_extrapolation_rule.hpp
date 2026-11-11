// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides the placeholder extrapolation rule used to request periodic boundaries.

#pragma once

#include <ddc/ddc.hpp>

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

namespace ddc {

/**
 * @brief A functor to represent periodic extrapolation in a 1D spline evaluator.
 *
 * This rule is handled by the spline evaluator itself and therefore this
 * functor should never be invoked.
 *
 * @tparam DimI The continuous dimension along which the evaluation is periodic.
 *
 * @ingroup ddc_splines
 */
template <class DimI>
struct PeriodicExtrapolationRule
{
    /**
     * @brief This function should never be called.
     *
     * @return Undefined.
     */
    template <class CoordType, class ChunkSpan>
    KOKKOS_FUNCTION Real operator()(CoordType, ChunkSpan) const
    {
        KOKKOS_ASSERT("PeriodicExtrapolationRule::operator() should never be called")

        return 0.;
    }
};

} // namespace ddc
