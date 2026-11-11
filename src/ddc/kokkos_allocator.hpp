// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides KokkosAllocator and the DeviceAllocator and HostAllocator aliases.
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace ddc {

//! @brief Allocator allocating memory in a Kokkos memory space.
//!
//! @details Memory is obtained with `Kokkos::kokkos_malloc` and released with
//! `Kokkos::kokkos_free`. A `static_assert` rejects types whose alignment exceeds that supported
//! by Kokkos. The allocator is stateless: two KokkosAllocator compare equal if and only if they
//! have the same type.
//!
//! @tparam T The type of the allocated elements.
//! @tparam MemorySpace The Kokkos memory space in which the memory is allocated.
//! @ingroup ddc_containers
template <class T, class MemorySpace>
class KokkosAllocator
{
    // Kokkos natively supports alignment for any scalar type and `Kokkos::complex<T>`
    static_assert(
            alignof(T)
                    <= std::max(alignof(std::max_align_t), alignof(Kokkos::complex<long double>)),
            "Alignment not supported");

public:
    //! @brief Type of the allocated elements.
    using value_type = T;

    //! @brief Kokkos memory space in which the memory is allocated.
    using memory_space = MemorySpace;

    //! @brief Rebinds the allocator to another element type, in the same memory space.
    //!
    //! @tparam U The new element type.
    template <class U>
    struct rebind
    {
        using other = KokkosAllocator<U, MemorySpace>;
    };

    constexpr KokkosAllocator() = default;

    //! @brief Copy constructor.
    //!
    //! @param[in] x The allocator to copy.
    constexpr KokkosAllocator(KokkosAllocator const& x) = default;

    //! @brief Move constructor.
    //!
    //! @param[in] x The allocator to move.
    constexpr KokkosAllocator(KokkosAllocator&& x) noexcept = default;

    //! @brief Converting constructor from an allocator of another element type.
    //!
    //! @details Allocators are stateless, so nothing is copied.
    //!
    //! @tparam U The element type of the source allocator.
    template <class U>
    constexpr explicit KokkosAllocator(KokkosAllocator<U, MemorySpace> const&) noexcept
    {
    }

    ~KokkosAllocator() = default;

    //! @brief Copy assignment operator.
    //!
    //! @param[in] x The allocator to copy.
    //! @return `*this`.
    constexpr KokkosAllocator& operator=(KokkosAllocator const& x) = default;

    //! @brief Move assignment operator.
    //!
    //! @param[in] x The allocator to move.
    //! @return `*this`.
    constexpr KokkosAllocator& operator=(KokkosAllocator&& x) noexcept = default;

    //! @brief Converting assignment from an allocator of another element type.
    //!
    //! @details Allocators are stateless, so nothing is copied.
    //!
    //! @tparam U The element type of the source allocator.
    //! @return Not specified by the source: the function body has no `return` statement.
    template <class U>
    constexpr KokkosAllocator& operator=(KokkosAllocator<U, MemorySpace> const&) noexcept
    {
    }

    //! @brief Allocates an unlabeled, uninitialized array of `n` elements.
    //!
    //! @param[in] n Number of elements.
    //! @return Pointer to the allocated memory, obtained with `Kokkos::kokkos_malloc`.
    [[nodiscard]] T* allocate(std::size_t n) const
    {
        return static_cast<T*>(Kokkos::kokkos_malloc<MemorySpace>(sizeof(T) * n));
    }

    //! @brief Allocates a labeled, uninitialized array of `n` elements.
    //!
    //! @param[in] label Label given to the allocation, e.g. for Kokkos profiling tools.
    //! @param[in] n Number of elements.
    //! @return Pointer to the allocated memory, obtained with `Kokkos::kokkos_malloc`.
    [[nodiscard]] T* allocate(std::string const& label, std::size_t n) const
    {
        return static_cast<T*>(Kokkos::kokkos_malloc<MemorySpace>(label, sizeof(T) * n));
    }

    //! @brief Releases memory obtained from allocate().
    //!
    //! @param[in] p Pointer returned by allocate().
    //! @details The number of elements is ignored, the memory is released with
    //! `Kokkos::kokkos_free`.
    void deallocate(T* p, std::size_t) const
    {
        Kokkos::kokkos_free(p);
    }
};

//! @brief Compares two KokkosAllocator.
//!
//! @return `true` if both allocators have exactly the same type (same element type and same
//! memory space), `false` otherwise.
//! @ingroup ddc_containers
template <class T, class MST, class U, class MSU>
constexpr bool operator==(KokkosAllocator<T, MST> const&, KokkosAllocator<U, MSU> const&) noexcept
{
    return std::is_same_v<KokkosAllocator<T, MST>, KokkosAllocator<U, MSU>>;
}

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
// In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
//! @brief Inequality comparison of two KokkosAllocator.
//!
//! @return The negation of the equality comparison.
//! @ingroup ddc_containers
template <class T, class MST, class U, class MSU>
constexpr bool operator!=(KokkosAllocator<T, MST> const&, KokkosAllocator<U, MSU> const&) noexcept
{
    return !std::is_same_v<KokkosAllocator<T, MST>, KokkosAllocator<U, MSU>>;
}
#endif

//! @brief KokkosAllocator allocating in the memory space of the default execution space.
//!
//! @tparam T The type of the allocated elements.
//! @ingroup ddc_containers
template <class T>
using DeviceAllocator = KokkosAllocator<T, Kokkos::DefaultExecutionSpace::memory_space>;

//! @brief KokkosAllocator allocating in `Kokkos::HostSpace`.
//!
//! @tparam T The type of the allocated elements.
//! @ingroup ddc_containers
template <class T>
using HostAllocator = KokkosAllocator<T, Kokkos::HostSpace>;

} // namespace ddc
