// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ddc::DiscreteDomain, a contiguous multi-dimensional range of DiscreteElement.
#pragma once

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

#include "discrete_dimension.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "type_seq.hpp"

namespace ddc {

template <concepts::discrete_dimension DDim>
struct DiscreteDomainIterator;

template <concepts::discrete_dimension... DDims>
class DiscreteDomain;

//! @brief Type trait that is `std::true_type` if `T` is a ddc::DiscreteDomain.
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
struct is_discrete_domain : std::false_type
{
};

//! @brief Specialization of ddc::is_discrete_domain for ddc::DiscreteDomain.
//! @tparam Tags The discrete dimensions of the domain.
//! @ingroup ddc_domains
template <class... Tags>
struct is_discrete_domain<DiscreteDomain<Tags...>> : std::true_type
{
};

//! @brief Helper variable template for ddc::is_discrete_domain.
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
inline constexpr bool is_discrete_domain_v = is_discrete_domain<T>::value;

namespace concepts {

//! @brief Satisfied by the types that are a ddc::DiscreteDomain.
//! @tparam T The type to check.
//! @ingroup ddc_domains
template <class T>
concept discrete_domain = is_discrete_domain_v<T>;

} // namespace concepts

namespace detail {

template <class... Tags>
struct ToTypeSeq<DiscreteDomain<Tags...>>
{
    using type = ddc::TypeSeq<Tags...>;
};

template <class... DDims, class... ODDims>
struct Rebind<DiscreteDomain<DDims...>, ddc::TypeSeq<ODDims...>>
{
    using type = DiscreteDomain<ODDims...>;
};

} // namespace detail

//! @brief A rectangular range of contiguous DiscreteElement along one or more discrete dimensions.
//!
//! @details The domain is stored as its first element (inclusive lower bound) and an end element
//! (exclusive upper bound). Along each dimension, it contains the elements whose identifier `i`
//! satisfies `begin <= i < end`. It is a lightweight value type that does not own any data.
//!
//! The specialization `DiscreteDomain<>` is a zero-dimensional domain: it contains exactly one
//! element, and is never empty.
//!
//! @tparam DDims The discrete dimensions of the domain, they must be unique.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDims>
class DiscreteDomain
{
    template <concepts::discrete_dimension...>
    friend class DiscreteDomain;

    static_assert(
            type_seq_is_unique_v<TypeSeq<DDims...>>,
            "The dimensions of a DiscreteDomain must be unique");

public:
    //! @brief Type of the elements of the domain.
    using discrete_element_type = DiscreteElement<DDims...>;

    //! @brief Type of the vectors of offsets between elements of the domain.
    using discrete_vector_type = DiscreteVector<DDims...>;

private:
    DiscreteElement<DDims...> m_element_begin;

    DiscreteElement<DDims...> m_element_end;

public:
    //! @brief Returns the number of dimensions of the domain.
    //! @return The number of discrete dimensions `sizeof...(DDims)`.
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return sizeof...(DDims);
    }

    //! @brief Default constructor.
    //! @note The bounds are not initialized by default-initialization.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain() = default;

    /// Construct a DiscreteDomain by copies and merge of domains
    //! @details Along each dimension of `DDims`, the bounds are taken from the argument that has
    //! it. The
    //! arguments may have further dimensions, which are ignored. Each dimension of `DDims` must be
    //! provided by exactly one argument (checked by `static_assert`).
    //!
    //! @param[in] domains The domains providing the bounds along the dimensions of `DDims`.
    template <concepts::discrete_domain... DDoms>
    KOKKOS_FUNCTION constexpr explicit DiscreteDomain(DDoms const&... domains)
        : m_element_begin(domains.front()...)
        , m_element_end((domains.front() + domains.extents())...)
    {
    }

    /** Construct a DiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param size the number of points in each direction
     */
    KOKKOS_FUNCTION constexpr DiscreteDomain(
            discrete_element_type const& element_begin,
            discrete_vector_type const& size)
        : m_element_begin(element_begin)
        , m_element_end(element_begin + size)
    {
    }

    //! @brief Copy constructor.
    //! @param[in] x The domain to copy.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain const& x) = default;

    //! @brief Move constructor.
    //! @param[in] x The domain to move from.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain&& x) = default;

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~DiscreteDomain() = default;

    //! @brief Copy assignment operator.
    //! @param[in] x The domain to copy.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain const& x) = default;

    //! @brief Move assignment operator.
    //! @param[in] x The domain to move from.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain&& x) = default;

    //! @brief Compares two domains for equality.
    //! @param[in] other The domain to compare with. It must have all the dimensions of this domain.
    //! @return `true` if both domains are empty, or if their first and end elements are equal.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr bool operator==(DiscreteDomain<ODDims...> const& other) const
    {
        if (empty() && other.empty()) {
            return true;
        }
        return m_element_begin == other.m_element_begin && m_element_end == other.m_element_end;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Compares two domains for inequality.
    //! @param[in] other The domain to compare with.
    //! @return `!(*this == other)`.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr bool operator!=(DiscreteDomain<ODDims...> const& other) const
    {
        return !(*this == other);
    }
#endif

    //! @brief Returns the total number of elements of the domain.
    //! @return The product of the number of elements along each dimension.
    KOKKOS_FUNCTION constexpr std::size_t size() const
    {
        return (1UL * ... * extent<DDims>().value());
    }

    //! @brief Returns the number of elements along each dimension.
    //! @return A vector whose components are the end element minus the first element.
    KOKKOS_FUNCTION constexpr discrete_vector_type extents() const noexcept
    {
        return m_element_end - m_element_begin;
    }

    //! @brief Returns the number of elements along one dimension.
    //! @tparam QueryDDim The discrete dimension to query, it must be in `DDims`.
    //! @return The extent of the domain along `QueryDDim`.
    template <concepts::discrete_dimension QueryDDim>
    KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDim> extent() const noexcept
    {
        return DiscreteElement<QueryDDim>(m_element_end)
               - DiscreteElement<QueryDDim>(m_element_begin);
    }

    //! @brief Returns the first element of the domain.
    //! @return The inclusive lower bound of the domain.
    KOKKOS_FUNCTION constexpr discrete_element_type front() const noexcept
    {
        return m_element_begin;
    }

    //! @brief Returns the last element of the domain.
    //! @return The element whose identifiers are those of the end element minus one.
    //! @note For an empty domain, the result is not an element of the domain.
    KOKKOS_FUNCTION constexpr discrete_element_type back() const noexcept
    {
        return discrete_element_type((DiscreteElement<DDims>(m_element_end) - 1)...);
    }

    //! @brief Takes the first elements of the domain along each dimension.
    //! @param[in] n The number of elements to keep along each dimension. It is not checked against
    //! the
    //! extents of the domain.
    //! @return The sub-domain starting at the first element and whose extents are `n`.
    KOKKOS_FUNCTION constexpr DiscreteDomain take_first(discrete_vector_type n) const
    {
        return DiscreteDomain(front(), n);
    }

    //! @brief Takes the last elements of the domain along each dimension.
    //! @param[in] n The number of elements to keep along each dimension. It is not checked against
    //! the
    //! extents of the domain.
    //! @return The sub-domain that ends where this domain ends and whose extents are `n`.
    KOKKOS_FUNCTION constexpr DiscreteDomain take_last(discrete_vector_type n) const
    {
        return DiscreteDomain(front() + (extents() - n), n);
    }

    //! @brief Removes the first elements of the domain along each dimension.
    //! @param[in] n The number of elements to remove along each dimension. It is not checked
    //! against
    //! the extents of the domain.
    //! @return The sub-domain that starts `n` elements after the first element and ends where this
    //! domain ends.
    KOKKOS_FUNCTION constexpr DiscreteDomain remove_first(discrete_vector_type n) const
    {
        return DiscreteDomain(front() + n, extents() - n);
    }

    //! @brief Removes the last elements of the domain along each dimension.
    //! @param[in] n The number of elements to remove along each dimension. It is not checked
    //! against
    //! the extents of the domain.
    //! @return The sub-domain that starts at the first element and whose extents are reduced by
    //! `n`.
    KOKKOS_FUNCTION constexpr DiscreteDomain remove_last(discrete_vector_type n) const
    {
        return DiscreteDomain(front(), extents() - n);
    }

    //! @brief Removes elements at both ends of the domain along each dimension.
    //! @param[in] n1 The number of elements to remove at the beginning along each dimension.
    //! @param[in] n2 The number of elements to remove at the end along each dimension.
    //! @return The sub-domain that starts `n1` elements after the first element and whose extents
    //! are
    //! reduced by `n1 + n2`. The numbers are not checked against the extents of the domain.
    KOKKOS_FUNCTION constexpr DiscreteDomain remove(
            discrete_vector_type n1,
            discrete_vector_type n2) const
    {
        return DiscreteDomain(front() + n1, extents() - n1 - n2);
    }

    //! @brief Returns the element located at a given offset from the first element.
    //! @param[in] dvect The offset from the first element. It is not checked against the extents.
    //! @return The first element of the domain shifted by `dvect`.
    KOKKOS_FUNCTION constexpr DiscreteElement<DDims...> operator()(
            DiscreteVector<DDims...> const& dvect) const noexcept
    {
        return m_element_begin + dvect;
    }

    //! @brief Restricts this domain along the dimensions of another domain.
    //!
    //! @details Along the dimensions of `odomain`, the result takes the bounds of `odomain`. Along
    //! the other dimensions of this domain, the bounds are unchanged.
    //!
    //! @param[in] odomain The domain providing the new bounds. Its dimensions must be among those
    //! of
    //! this domain.
    //! @return A domain with the same dimensions as this one.
    //! @pre Along each dimension of `odomain`, the bounds of `odomain` must be included in those of
    //! this domain (`KOKKOS_ASSERT`).
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr auto restrict_with(DiscreteDomain<ODDims...> const& odomain) const
    {
        KOKKOS_ASSERT(
                ((DiscreteElement<ODDims>(m_element_begin)
                  <= DiscreteElement<ODDims>(odomain.m_element_begin))
                 && ...))
        KOKKOS_ASSERT(
                ((DiscreteElement<ODDims>(m_element_end)
                  >= DiscreteElement<ODDims>(odomain.m_element_end))
                 && ...))
        DiscreteVector<DDims...> const myextents = extents();
        DiscreteVector<ODDims...> const oextents = odomain.extents();
        return DiscreteDomain(
                DiscreteElement<DDims...>((select_or<DDims>(
                        odomain.m_element_begin,
                        DiscreteElement<DDims>(m_element_begin)))...),
                DiscreteVector<DDims...>(
                        (select_or<DDims>(oextents, DiscreteVector<DDims>(myextents)))...));
    }

    //! @brief Checks whether an element belongs to the domain.
    //! @param[in] delems One or several DiscreteElements which, together, provide one identifier
    //! for
    //! each dimension of the domain (`static_assert` on the total number of dimensions).
    //! @return `true` if, along each dimension, the identifier is at least that of the first
    //! element and
    //! lower than that of the end element.
    template <concepts::discrete_element... DElems>
    KOKKOS_FUNCTION bool contains(DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        DiscreteElement<DDims...> const delem(delems...);
        for (std::size_t i = 0; i < rank(); ++i) {
            if ((detail::array(delem)[i] < detail::array(m_element_begin)[i])
                || (detail::array(delem)[i] >= detail::array(m_element_end)[i])) {
                return false;
            }
        }
        return true;
    }

    //! @brief Computes the offset of an element from the first element of the domain.
    //! @param[in] delems One or several DiscreteElements which, together, provide one identifier
    //! for
    //! each dimension of the domain (`static_assert` on the total number of dimensions).
    //! @return The vector going from the first element of the domain to the given element. The
    //! element
    //! is not checked to be in the domain.
    template <concepts::discrete_element... DElems>
    KOKKOS_FUNCTION DiscreteVector<DDims...> distance_from_front(
            DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        return DiscreteVector<DDims...>(
                (DiscreteElement<DDims>(take<DDims>(delems...))
                 - DiscreteElement<DDims>(m_element_begin))...);
    }

    //! @brief Checks whether the domain has no element.
    //! @return `true` if the domain has no element.
    KOKKOS_FUNCTION constexpr bool empty() const noexcept
    {
        return size() == 0;
    }

    //! @brief Converts the domain to a boolean.
    //! @return `true` if the domain is not empty.
    KOKKOS_FUNCTION constexpr explicit operator bool()
    {
        return !empty();
    }

    //! @brief Returns an iterator to the first element of a one-dimensional domain.
    //! @return A ddc::DiscreteDomainIterator pointing to the first element.
    KOKKOS_FUNCTION auto begin() const
        requires(sizeof...(DDims) == 1)
    {
        return DiscreteDomainIterator<std::tuple_element_t<0, std::tuple<DDims...>>>(front());
    }

    //! @brief Returns an iterator past the last element of a one-dimensional domain.
    //! @return A ddc::DiscreteDomainIterator pointing to the end element.
    KOKKOS_FUNCTION auto end() const
        requires(sizeof...(DDims) == 1)
    {
        return DiscreteDomainIterator<std::tuple_element_t<0, std::tuple<DDims...>>>(m_element_end);
    }

    //! @brief Returns an iterator to the first element of a one-dimensional domain.
    //! @return A ddc::DiscreteDomainIterator pointing to the first element.
    KOKKOS_FUNCTION auto cbegin() const
        requires(sizeof...(DDims) == 1)
    {
        return DiscreteDomainIterator<std::tuple_element_t<0, std::tuple<DDims...>>>(front());
    }

    //! @brief Returns an iterator past the last element of a one-dimensional domain.
    //! @return A ddc::DiscreteDomainIterator pointing to the end element.
    KOKKOS_FUNCTION auto cend() const
        requires(sizeof...(DDims) == 1)
    {
        return DiscreteDomainIterator<std::tuple_element_t<0, std::tuple<DDims...>>>(m_element_end);
    }

    //! @brief Accesses an element of a one-dimensional domain by its position.
    //! @param[in] n The position of the element, counted from the first element.
    //! @return The DiscreteElement located `n` elements after the first one. The position is not
    //! checked.
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n)
        requires(sizeof...(DDims) == 1)
    {
        return begin()[n];
    }

    //! @brief Accesses an element of a one-dimensional domain by its position.
    //! @param[in] n The position of the element, counted from the first element.
    //! @return The DiscreteElement located `n` elements after the first one. The position is not
    //! checked.
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n) const
        requires(sizeof...(DDims) == 1)
    {
        return begin()[n];
    }
};

//! @brief Specialization of ddc::DiscreteDomain for zero dimensions.
//!
//! @details This domain contains exactly one (empty) element: its size is 1 and it is never
//! empty. All the operations that select, restrict or remove elements return the domain itself.
//!
//! @ingroup ddc_domains
template <>
class DiscreteDomain<>
{
    template <concepts::discrete_dimension...>
    friend class DiscreteDomain;

public:
    //! @brief Type of the elements of the domain.
    using discrete_element_type = DiscreteElement<>;

    //! @brief Type of the vectors of offsets between elements of the domain.
    using discrete_vector_type = DiscreteVector<>;

    //! @brief Returns the number of dimensions of the domain.
    //! @return Always 0.
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return 0;
    }

    //! @brief Default constructor.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteDomain() = default;

    /// Construct a DiscreteDomain by copies and merge of domains
    //! @details The arguments are ignored: all the domains of zero dimensions are equal.
    //! @param ... Discrete domains, ignored.
    template <class... DDoms>
    KOKKOS_FUNCTION constexpr explicit DiscreteDomain(DDoms const&... /*domains*/)
        requires(is_discrete_domain_v<DDoms> && ...)
    {
    }

    /** Construct a DiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param size the number of points in each direction
     */
    KOKKOS_FUNCTION constexpr DiscreteDomain(
            [[maybe_unused]] discrete_element_type const& element_begin,
            [[maybe_unused]] discrete_vector_type const& size)
    {
        // Parameters `element_begin` and `size` are always unused, but needed for Doxygen
    }

    //! @brief Copy constructor.
    //! @param[in] x The domain to copy.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain const& x) = default;

    //! @brief Move constructor.
    //! @param[in] x The domain to move from.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain(DiscreteDomain&& x) = default;

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~DiscreteDomain() = default;

    //! @brief Copy assignment operator.
    //! @param[in] x The domain to copy.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain const& x) = default;

    //! @brief Move assignment operator.
    //! @param[in] x The domain to move from.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomain& operator=(DiscreteDomain&& x) = default;

    //! @brief Compares two zero-dimensional domains for equality.
    //! @return Always `true`.
    KOKKOS_FUNCTION constexpr bool operator==(DiscreteDomain const& /*other*/) const
    {
        return true;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Compares two zero-dimensional domains for inequality.
    //! @param[in] other The domain to compare with.
    //! @return Always `false`.
    KOKKOS_FUNCTION constexpr bool operator!=(DiscreteDomain const& other) const
    {
        return !(*this == other);
    }
#endif

    //! @brief Returns the total number of elements of the domain.
    //! @return Always 1.
    static KOKKOS_FUNCTION constexpr std::size_t size()
    {
        return 1;
    }

    //! @brief Returns the (empty) vector of extents.
    //! @return An empty DiscreteVector.
    static KOKKOS_FUNCTION constexpr discrete_vector_type extents() noexcept
    {
        return {};
    }

    //! @brief Returns the first element of the domain.
    //! @return An empty DiscreteElement.
    static KOKKOS_FUNCTION constexpr discrete_element_type front() noexcept
    {
        return {};
    }

    //! @brief Returns the last element of the domain.
    //! @return An empty DiscreteElement.
    static KOKKOS_FUNCTION constexpr discrete_element_type back() noexcept
    {
        return {};
    }

    //! @brief Takes the first elements of the domain, which has no effect on a zero-dimensional
    //! domain.
    //! @return This domain.
    KOKKOS_FUNCTION constexpr DiscreteDomain take_first(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Takes the last elements of the domain, which has no effect on a zero-dimensional
    //! domain.
    //! @return This domain.
    KOKKOS_FUNCTION constexpr DiscreteDomain take_last(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes the first elements of the domain, which has no effect on a zero-dimensional
    //! domain.
    //! @return This domain.
    KOKKOS_FUNCTION constexpr DiscreteDomain remove_first(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes the last elements of the domain, which has no effect on a zero-dimensional
    //! domain.
    //! @return This domain.
    KOKKOS_FUNCTION constexpr DiscreteDomain remove_last(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes elements at both ends of the domain, which has no effect on a
    //! zero-dimensional domain.
    //! @return This domain.
    KOKKOS_FUNCTION constexpr DiscreteDomain remove(
            discrete_vector_type /*n1*/,
            discrete_vector_type /*n2*/) const
    {
        return *this;
    }

    //! @brief Returns the element located at a given offset from the first element.
    //! @return An empty DiscreteElement.
    KOKKOS_FUNCTION constexpr DiscreteElement<> operator()(
            DiscreteVector<> const& /*dvect*/) const noexcept
    {
        return {};
    }

    //! @brief Restricts this domain along the dimensions of another domain, which has no effect
    //! here.
    //! @return This domain.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr DiscreteDomain restrict_with(
            DiscreteDomain<ODDims...> const& /*odomain*/) const
    {
        return *this;
    }

    //! @brief Checks whether the (empty) element belongs to the domain.
    //! @return Always `true`.
    static KOKKOS_FUNCTION bool contains() noexcept
    {
        return true;
    }

    //! @brief Checks whether an element belongs to the domain.
    //! @return Always `true`.
    static KOKKOS_FUNCTION bool contains(DiscreteElement<>) noexcept
    {
        return true;
    }

    //! @brief Computes the offset from the first element of the domain.
    //! @return An empty DiscreteVector.
    static KOKKOS_FUNCTION DiscreteVector<> distance_from_front() noexcept
    {
        return {};
    }

    //! @brief Computes the offset of an element from the first element of the domain.
    //! @return An empty DiscreteVector.
    static KOKKOS_FUNCTION DiscreteVector<> distance_from_front(DiscreteElement<>) noexcept
    {
        return {};
    }

    //! @brief Checks whether the domain has no element.
    //! @return Always `false`.
    static KOKKOS_FUNCTION constexpr bool empty() noexcept
    {
        return false;
    }

    //! @brief Converts the domain to a boolean.
    //! @return Always `true`.
    KOKKOS_FUNCTION constexpr explicit operator bool()
    {
        return true;
    }
};

//! @brief Extracts the sub-domain along the dimensions `QueryDDims`.
//! @tparam QueryDDims The discrete dimensions to extract, they must all be dimensions of `domain`.
//! @param[in] domain The domain to extract from.
//! @return A DiscreteDomain along `QueryDDims` with the same bounds as `domain` along them.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteDomain<QueryDDims...> select(
        DiscreteDomain<DDims...> const& domain)
{
    return DiscreteDomain<QueryDDims...>(domain);
}

namespace detail {

template <ddc::concepts::type_seq T>
struct ConvertTypeSeqToDiscreteDomain
{
};

template <ddc::concepts::discrete_dimension... DDims>
struct ConvertTypeSeqToDiscreteDomain<ddc::TypeSeq<DDims...>>
{
    using type = DiscreteDomain<DDims...>;
};

template <ddc::concepts::type_seq T>
using convert_type_seq_to_discrete_domain_t = ConvertTypeSeqToDiscreteDomain<T>::type;

} // namespace detail

// Computes the subtraction DDom_a - DDom_b in the sense of linear spaces(retained dimensions are those in DDom_a which are not in DDom_b)
//! @brief Removes from a domain the dimensions of another domain.
//!
//! @details The second argument is only used to deduce the dimensions to remove.
//!
//! @param[in] DDom_a The domain on which to remove dimensions.
//! @return A domain with the dimensions of `DDom_a` that are not in the second argument, with the
//! same bounds.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDimsA, concepts::discrete_dimension... DDimsB>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        DiscreteDomain<DDimsA...> const& DDom_a,
        DiscreteDomain<DDimsB...> const& /*DDom_b*/) noexcept
{
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_discrete_domain_t<type_seq_r>(DDom_a);
}

//! Remove the dimensions DDimsB from DDom_a
//! @param[in] DDom_a The discrete domain on which to remove dimensions
//! @return The discrete domain without DDimsB dimensions
//! @tparam DDimsB The discrete dimensions to remove.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDimsB, concepts::discrete_dimension... DDimsA>
KOKKOS_FUNCTION constexpr auto remove_dims_of(DiscreteDomain<DDimsA...> const& DDom_a) noexcept
{
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_discrete_domain_t<type_seq_r>(DDom_a);
}

//! Type alias for the discrete domain obtained by removing dimensions DDims from DDom.
//! @tparam DDom The discrete domain type from which to remove dimensions.
//! @tparam DDims The discrete dimensions to remove.
//! @return The discrete domain type without the DDims dimensions.
//! @ingroup ddc_domains
template <typename DDom, concepts::discrete_dimension... DDims>
using remove_dims_of_t = decltype(remove_dims_of<DDims...>(std::declval<DDom>()));

namespace detail {

// Checks if dimension of DDom_a is DDim1. If not, returns restriction to DDim2 of DDom_b. May not be useful in its own, it helps for replace_dim_of
template <
        ddc::concepts::discrete_dimension DDim1,
        ddc::concepts::discrete_dimension DDim2,
        ddc::concepts::discrete_dimension DDimA,
        ddc::concepts::discrete_dimension... DDimsB>
KOKKOS_FUNCTION constexpr std::conditional_t<
        std::is_same_v<DDimA, DDim1>,
        ddc::DiscreteDomain<DDim2>,
        ddc::DiscreteDomain<DDimA>>
replace_dim_of_1d(
        [[maybe_unused]] DiscreteDomain<DDimA> const& DDom_a,
        [[maybe_unused]] DiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    if constexpr (std::is_same_v<DDimA, DDim1>) {
        return ddc::DiscreteDomain<DDim2>(DDom_b);
    } else {
        return DDom_a;
    }
}

} // namespace detail

// Replace in DDom_a the dimension Dim1 by the dimension Dim2 of DDom_b
//! @brief Replaces a dimension of a domain by a dimension of another domain.
//! @tparam DDim1 The dimension of `DDom_a` to replace.
//! @tparam DDim2 The dimension of `DDom_b` that takes its place.
//! @param[in] DDom_a The domain in which the dimension is replaced.
//! @param[in] DDom_b The domain providing the bounds along `DDim2`. It must contain `DDim2`.
//! @return The domain with the dimensions of `DDom_a`, where `DDim1` is replaced by `DDim2` and has
//! the bounds of `DDom_b` along `DDim2`.
//! @ingroup ddc_domains
template <
        concepts::discrete_dimension DDim1,
        concepts::discrete_dimension DDim2,
        concepts::discrete_dimension... DDimsA,
        concepts::discrete_dimension... DDimsB>
KOKKOS_FUNCTION constexpr auto replace_dim_of(
        DiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] DiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    // TODO : static_asserts
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDim1>;
    using TagSeqC = TypeSeq<DDim2>;

    using type_seq_r = ddc::type_seq_replace_t<TagSeqA, TagSeqB, TagSeqC>;
    return ddc::detail::convert_type_seq_to_discrete_domain_t<type_seq_r>(
            detail::replace_dim_of_1d<
                    DDim1,
                    DDim2,
                    DDimsA,
                    DDimsB...>(ddc::DiscreteDomain<DDimsA>(DDom_a), DDom_b)...);
}

// Replace dimensions from a domain type
//! @brief Type of the domain obtained by replacing a dimension in a domain type.
//! @tparam DDom The discrete domain type in which to replace a dimension.
//! @tparam DDim1 The dimension to replace.
//! @tparam DDim2 The dimension that takes its place.
//! @ingroup ddc_domains
template <typename DDom, concepts::discrete_dimension DDim1, concepts::discrete_dimension DDim2>
using replace_dim_of_t = decltype(replace_dim_of<DDim1, DDim2>(
        std::declval<DDom>(),
        std::declval<detail::rebind_t<DDom, TypeSeq<DDim2>>>()));


//! @brief Returns the number of elements of a domain along the dimensions `QueryDDims`.
//! @tparam QueryDDims The discrete dimensions to query, they must all be dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return A DiscreteVector along `QueryDDims` holding the extents.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDims...> extents(
        DiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteVector<QueryDDims...>(DiscreteDomain<QueryDDims>(domain).size()...);
}

//! @brief Returns the first element of a domain along the dimensions `QueryDDims`.
//! @tparam QueryDDims The discrete dimensions to query, they must all be dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return A DiscreteElement along `QueryDDims` holding the first element of `domain`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> front(
        DiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(DiscreteDomain<QueryDDims>(domain).front()...);
}

//! @brief Returns the last element of a domain along the dimensions `QueryDDims`.
//! @tparam QueryDDims The discrete dimensions to query, they must all be dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return A DiscreteElement along `QueryDDims` holding the last element of `domain`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> back(
        DiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(DiscreteDomain<QueryDDims>(domain).back()...);
}

//! @brief Random access iterator over the elements of a one-dimensional DiscreteDomain.
//!
//! @details Dereferencing the iterator returns the DiscreteElement it points to, by value. The
//! iterator does not know the bounds of the domain it comes from.
//!
//! @tparam DDim The discrete dimension of the iterated elements.
//! @ingroup ddc_domains
template <concepts::discrete_dimension DDim>
struct DiscreteDomainIterator
{
private:
    DiscreteElement<DDim> m_value = DiscreteElement<DDim>();

public:
    //! @brief Iterator category.
    using iterator_category = std::random_access_iterator_tag;

    //! @brief Type of the elements the iterator points to.
    using value_type = DiscreteElement<DDim>;

    //! @brief Signed type for the distance between iterators.
    using difference_type = std::ptrdiff_t;

    //! @brief Default constructor, pointing to the element with identifier 0.
    KOKKOS_DEFAULTED_FUNCTION DiscreteDomainIterator() = default;

    //! @brief Constructs an iterator pointing to a given element.
    //! @param[in] value The element the iterator points to.
    KOKKOS_FUNCTION constexpr explicit DiscreteDomainIterator(DiscreteElement<DDim> value)
        : m_value(value)
    {
    }

    //! @brief Dereferences the iterator.
    //! @return The element the iterator points to.
    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator*() const noexcept
    {
        return m_value;
    }

    //! @brief Pre-increment: moves to the next element.
    //! @return A reference to this iterator, after the increment.
    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator++()
    {
        ++m_value;
        return *this;
    }

    //! @brief Post-increment: moves to the next element.
    //! @return A copy of this iterator, before the increment.
    KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    //! @brief Pre-decrement: moves to the previous element.
    //! @return A reference to this iterator, after the decrement.
    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator--()
    {
        --m_value;
        return *this;
    }

    //! @brief Post-decrement: moves to the previous element.
    //! @return A copy of this iterator, before the decrement.
    KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator--(int)
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    //! @brief Advances the iterator by `n` elements, backwards if `n` is negative.
    //! @param[in] n The number of elements to move by.
    //! @return A reference to this iterator.
    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator+=(difference_type n)
    {
        if (n >= static_cast<difference_type>(0)) {
            m_value += static_cast<DiscreteElementType>(n);
        } else {
            m_value -= static_cast<DiscreteElementType>(-n);
        }
        return *this;
    }

    //! @brief Moves the iterator back by `n` elements, forwards if `n` is negative.
    //! @param[in] n The number of elements to move back by.
    //! @return A reference to this iterator.
    KOKKOS_FUNCTION constexpr DiscreteDomainIterator& operator-=(difference_type n)
    {
        if (n >= static_cast<difference_type>(0)) {
            m_value -= static_cast<DiscreteElementType>(n);
        } else {
            m_value += static_cast<DiscreteElementType>(-n);
        }
        return *this;
    }

    //! @brief Returns the element located `n` elements after the one pointed to.
    //! @param[in] n The offset from the current element.
    //! @return The element at the given offset.
    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator[](difference_type n) const
    {
        return m_value + n;
    }

    //! @brief Compares two iterators for equality.
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if both iterators point to the same element.
    friend KOKKOS_FUNCTION constexpr bool operator==(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return xx.m_value == yy.m_value;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Compares two iterators for inequality.
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if the iterators point to different elements.
    friend KOKKOS_FUNCTION constexpr bool operator!=(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return xx.m_value != yy.m_value;
    }
#endif

    //! @brief Compares the positions of two iterators.
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if `xx` points strictly before `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator<(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return xx.m_value < yy.m_value;
    }

    //! @brief Compares the positions of two iterators.
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if `xx` points strictly after `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator>(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return yy < xx;
    }

    //! @brief Compares the positions of two iterators.
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if `xx` points before or at the same element as `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator<=(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return !(yy < xx);
    }

    //! @brief Compares the positions of two iterators.
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if `xx` points after or at the same element as `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator>=(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return !(xx < yy);
    }

    //! @brief Returns an iterator advanced by `n` elements.
    //! @param[in] i The iterator.
    //! @param[in] n The number of elements to move by.
    //! @return The advanced iterator.
    friend KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator+(
            DiscreteDomainIterator i,
            difference_type n)
    {
        return i += n;
    }

    //! @brief Returns an iterator advanced by `n` elements.
    //! @param[in] n The number of elements to move by.
    //! @param[in] i The iterator.
    //! @return The advanced iterator.
    friend KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator+(
            difference_type n,
            DiscreteDomainIterator i)
    {
        return i += n;
    }

    //! @brief Returns an iterator moved back by `n` elements.
    //! @param[in] i The iterator.
    //! @param[in] n The number of elements to move back by.
    //! @return The moved iterator.
    friend KOKKOS_FUNCTION constexpr DiscreteDomainIterator operator-(
            DiscreteDomainIterator i,
            difference_type n)
    {
        return i -= n;
    }

    //! @brief Computes the signed distance between two iterators.
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return The number of elements to go from `yy` to `xx`, negative if `xx` is before `yy`.
    friend KOKKOS_FUNCTION constexpr difference_type operator-(
            DiscreteDomainIterator const& xx,
            DiscreteDomainIterator const& yy)
    {
        return (yy.m_value > xx.m_value) ? (-static_cast<difference_type>(yy.m_value - xx.m_value))
                                         : (xx.m_value - yy.m_value);
    }
};

} // namespace ddc
