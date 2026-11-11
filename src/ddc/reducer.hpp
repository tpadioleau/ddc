// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides the binary reduction operators (sum, prod, min, max, ...) used by
//! reduction algorithms.

#pragma once

#include <utility>

#include <Kokkos_Macros.hpp>

namespace ddc::reducer {

//! @brief Reducer computing the sum (`lhs + rhs`).
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct sum
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs + rhs;
    }
};

//! @brief Reducer computing the product (`lhs * rhs`).
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct prod
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs * rhs;
    }
};

//! @brief Reducer computing the logical AND (`lhs && rhs`).
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct land
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs && rhs;
    }
};

//! @brief Reducer computing the logical OR (`lhs || rhs`).
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct lor
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs || rhs;
    }
};

//! @brief Reducer computing the bitwise AND (`lhs & rhs`).
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct band
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs & rhs;
    }
};

//! @brief Reducer computing the bitwise OR (`lhs | rhs`).
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct bor
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs | rhs;
    }
};

//! @brief Reducer computing the bitwise XOR (`lhs ^ rhs`).
//!
//!
//! @note The conversion of this reducer to a Kokkos reducer in parallel reductions is not
//! implemented (static_assert in `ddc::detail::DdcToKokkosReducer`).
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct bxor
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs ^ rhs;
    }
};

//! @brief Reducer keeping the smaller of two values.
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct min
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs < rhs ? lhs : rhs;
    }
};

//! @brief Reducer keeping the larger of two values.
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct max
{
    //! @brief Type of the reduced values: `T`.
    using value_type = T;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return lhs > rhs ? lhs : rhs;
    }
};

//! @brief Reducer computing, element-wise, the minimum of the first members and the maximum of
//! the second members of two pairs.
//!
//! @tparam T The type of the values to reduce.
//!
//! @ingroup ddc_algorithms
template <class T>
struct minmax
{
    //! @brief Type of the reduced values: `std::pair<T, T>` holding the (minimum, maximum) pair.
    using value_type = std::pair<T, T>;

    //! @brief Combines two values.
    //!
    //! @param[in] lhs Left operand.
    //! @param[in] rhs Right operand.
    //! @return The combination of `lhs` and `rhs`.
    KOKKOS_FUNCTION constexpr value_type operator()(value_type const& lhs, value_type const& rhs)
            const noexcept
    {
        return value_type(
                lhs.first < rhs.first ? lhs.first : rhs.first,
                lhs.second > rhs.second ? lhs.second : rhs.second);
    }
};

} // namespace ddc::reducer
