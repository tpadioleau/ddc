// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides AlignedAllocator, an allocator returning over-aligned host memory.
#pragma once

#include <cstddef>
#include <new>
#include <type_traits>

namespace ddc {

//! @brief Allocator returning host memory aligned on `N` bytes.
//!
//! @details Memory is obtained with an aligned `new[]` expression and released with the
//! aligned `operator delete[]`. The allocator is stateless: two AlignedAllocator compare equal
//! if and only if they have the same type (same element type and same alignment).
//!
//! @tparam T The type of the allocated elements.
//! @tparam N The alignment of the allocated memory, in bytes.
//! @ingroup ddc_containers
template <class T, std::size_t N>
class AlignedAllocator
{
public:
    //! @brief Type of the allocated elements.
    using value_type = T;

    //! @brief Rebinds the allocator to another element type, with the same alignment.
    //!
    //! @tparam U The new element type.
    template <class U>
    struct rebind
    {
        using other = AlignedAllocator<U, N>;
    };

    //! @brief Default constructor.
    constexpr AlignedAllocator() = default;

    //! @brief Copy constructor.
    //!
    //! @param[in] x The allocator to copy.
    constexpr AlignedAllocator(AlignedAllocator const& x) = default;

    //! @brief Move constructor.
    //!
    //! @param[in] x The allocator to move.
    constexpr AlignedAllocator(AlignedAllocator&& x) noexcept = default;

    //! @brief Converting constructor from an allocator of another element type.
    //!
    //! @details Allocators are stateless, so nothing is copied.
    //!
    //! @tparam U The element type of the source allocator.
    template <class U>
    constexpr explicit AlignedAllocator(AlignedAllocator<U, N> const&) noexcept
    {
    }

    //! @brief Destructor.
    ~AlignedAllocator() = default;

    //! @brief Copy assignment operator.
    //!
    //! @param[in] x The allocator to copy.
    //! @return `*this`.
    constexpr AlignedAllocator& operator=(AlignedAllocator const& x) = default;

    //! @brief Move assignment operator.
    //!
    //! @param[in] x The allocator to move.
    //! @return `*this`.
    constexpr AlignedAllocator& operator=(AlignedAllocator&& x) noexcept = default;

    //! @brief Converting assignment from an allocator of another element type.
    //!
    //! @details Allocators are stateless, so nothing is copied.
    //!
    //! @tparam U The element type of the source allocator.
    //! @return Not specified by the source: the function body has no `return` statement.
    template <class U>
    constexpr AlignedAllocator& operator=(AlignedAllocator<U, N> const&) noexcept
    {
    }

    //! @brief Allocates an array of `n` elements aligned on `N` bytes.
    //!
    //! @param[in] n Number of elements.
    //! @return Pointer to the allocated memory.
    [[nodiscard]] T* allocate(std::size_t n) const
    {
        return new (std::align_val_t(N)) value_type[n];
    }

    //! @brief Releases memory obtained from allocate().
    //!
    //! @param[in] p Pointer returned by allocate().
    //!
    //! @details The number of elements is ignored.
    void deallocate(T* p, std::size_t) const
    {
        operator delete[](p, std::align_val_t(N));
    }
};

//! @brief Compares two AlignedAllocator.
//!
//! @return `true` if both allocators have exactly the same type (same element type and same
//! alignment), `false` otherwise.
//! @ingroup ddc_containers
template <class T, std::size_t NT, class U, std::size_t NU>
constexpr bool operator==(AlignedAllocator<T, NT> const&, AlignedAllocator<U, NU> const&) noexcept
{
    return std::is_same_v<AlignedAllocator<T, NT>, AlignedAllocator<U, NU>>;
}

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
// In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
//! @brief Inequality comparison of two AlignedAllocator.
//!
//! @return The negation of the equality comparison.
//! @ingroup ddc_containers
template <class T, std::size_t NT, class U, std::size_t NU>
constexpr bool operator!=(AlignedAllocator<T, NT> const&, AlignedAllocator<U, NU> const&) noexcept
{
    return !std::is_same_v<AlignedAllocator<T, NT>, AlignedAllocator<U, NU>>;
}
#endif

} // namespace ddc
