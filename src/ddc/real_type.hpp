// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Defines ddc::Real, the floating-point type selected at build time.

#pragma once

#include <ddc/config.hpp>

namespace ddc {

#if DDC_BUILD_DOUBLE_PRECISION()

//! @brief Default floating-point type of DDC, `double` when DDC is built with double precision.
//!
//! @ingroup ddc_core_utils
using Real = double;

#else

//! @brief Default floating-point type of DDC, `float` when DDC is not built with double
//! precision.
//!
//! @ingroup ddc_core_utils
using Real = float;

#endif

} // namespace ddc
