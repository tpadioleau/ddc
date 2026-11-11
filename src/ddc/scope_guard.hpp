// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ddc::ScopeGuard, an RAII helper managing the DDC runtime lifetime.

#pragma once

namespace ddc {

//! @brief RAII object managing the global store of discrete spaces during its lifetime.
//!
//! @details The constructor creates the global store in which discrete spaces are registered
//! (see ddc::init_discrete_space). The destructor calls the registered finalization functions of
//! the discrete spaces and then destroys the store. A ScopeGuard must therefore be alive while
//! discrete spaces are used.
//!
//! @ingroup ddc_core_utils
class ScopeGuard
{
public:
    //! @brief Creates the global store of discrete spaces.
    ScopeGuard();

    //! @brief Creates the global store of discrete spaces.
    //!
    //! @param[in] argc Number of command-line arguments (unused).
    //! @param[in,out] argv Command-line arguments (unused).
    ScopeGuard(int argc, char**& argv);

    //! @brief Copy constructor (deleted).
    ScopeGuard(ScopeGuard const& x) = delete;

    //! @brief Move constructor (deleted).
    ScopeGuard(ScopeGuard&& x) noexcept = delete;

    //! @brief Finalizes the registered discrete spaces and destroys the global store.
    ~ScopeGuard() noexcept;

    //! @brief Copy assignment operator (deleted).
    ScopeGuard& operator=(ScopeGuard const& x) = delete;

    //! @brief Move assignment operator (deleted).
    ScopeGuard& operator=(ScopeGuard&& x) noexcept = delete;
};

} // namespace ddc
