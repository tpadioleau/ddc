// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

//! @file discrete_dimension.hpp
//! @brief Defines the `ddc::DiscreteDimension` base class and the `ddc::concepts::discrete_dimension`
//! concept that identify discrete dimensions.

#include <ddc/config.hpp>

#if !DDC_BUILD_DEPRECATED_CODE()
#    include <type_traits>
#endif

namespace ddc {

//! @addtogroup ddc_samplings
//! @{

//! @brief Base class of discrete dimensions.
//!
//! A discrete dimension is a type deriving from this class, directly or through a sampling such as
//! `ddc::UniformPointSampling`. It is used as a tag by discrete elements, discrete vectors, discrete
//! domains and chunks.
struct DiscreteDimension
{
};

namespace concepts {

#if DDC_BUILD_DEPRECATED_CODE()
//! @brief Satisfied by every type when `DDC_BUILD_DEPRECATED_CODE()` is enabled (deprecated
//! compatibility mode), otherwise see the other definition.
//! @tparam T the type to check
template <class T>
concept discrete_dimension = true;
#else
//! @brief Satisfied by types that derive from `ddc::DiscreteDimension`.
//! @tparam T the type to check
template <class T>
concept discrete_dimension = std::is_base_of_v<DiscreteDimension, T>;
#endif

} // namespace concepts

//! @}

} // namespace ddc
