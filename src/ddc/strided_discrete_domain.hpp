// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides StridedDiscreteDomain, a multidimensional domain of regularly spaced discrete
//! elements, and its related free functions and iterator.
#pragma once

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

#include "discrete_dimension.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "type_seq.hpp"

namespace ddc {

template <concepts::discrete_dimension DDim>
struct StridedDiscreteDomainIterator;

template <concepts::discrete_dimension... DDims>
class StridedDiscreteDomain;

//! @brief Type trait that is true if `T` is a StridedDiscreteDomain.
//!
//! @details This primary template is `std::false_type`; it is specialized for
//! StridedDiscreteDomain.
//!
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
struct is_strided_discrete_domain : std::false_type
{
};

//! @brief Specialization of is_strided_discrete_domain for StridedDiscreteDomain (true).
//!
//! @tparam Tags The discrete dimensions of the StridedDiscreteDomain.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags>
struct is_strided_discrete_domain<StridedDiscreteDomain<Tags...>> : std::true_type
{
};

//! @brief Shorthand for `is_strided_discrete_domain<T>::value`.
//!
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
inline constexpr bool is_strided_discrete_domain_v = is_strided_discrete_domain<T>::value;

namespace concepts {

//! @brief Concept satisfied by the types that are a StridedDiscreteDomain.
//!
//! @tparam T The type to constrain.
//! @ingroup ddc_domains
template <class T>
concept strided_discrete_domain = is_strided_discrete_domain_v<T>;

}

namespace detail {

template <ddc::concepts::discrete_dimension... Tags>
struct ToTypeSeq<StridedDiscreteDomain<Tags...>>
{
    using type = ddc::TypeSeq<Tags...>;
};

template <ddc::concepts::discrete_dimension... DDims, ddc::concepts::discrete_dimension... ODDims>
struct Rebind<StridedDiscreteDomain<DDims...>, ddc::TypeSeq<ODDims...>>
{
    using type = StridedDiscreteDomain<ODDims...>;
};

} // namespace detail

//! @brief Computes the element-wise product of two DiscreteVector.
//!
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return A DiscreteVector whose component along each dimension is the product of the
//! components of `lhs` and `rhs` along this dimension.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... ODDims>
KOKKOS_FUNCTION DiscreteVector<ODDims...> prod(
        DiscreteVector<ODDims...> const& lhs,
        DiscreteVector<ODDims...> const& rhs) noexcept
{
    return DiscreteVector<ODDims...>((get<ODDims>(lhs) * get<ODDims>(rhs))...);
}

//! @brief A multidimensional domain of discrete elements regularly spaced along each dimension.
//!
//! @details The domain is described by its first element, the number of elements (extent) along
//! each dimension and the step (stride) between two consecutive elements along each dimension.
//! Along dimension `d`, the elements are `front + i * stride`, with `i` in `[0, extent)`.
//! The dimensions must be unique (checked by a `static_assert`).
//!
//! The domain stores the front element, the extents and the strides by value: it does not own
//! any other resource.
//!
//! @tparam DDims The discrete dimensions of the domain.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDims>
class StridedDiscreteDomain
{
    template <concepts::discrete_dimension...>
    friend class StridedDiscreteDomain;

    static_assert(
            type_seq_is_unique_v<TypeSeq<DDims...>>,
            "The dimensions of a StridedDiscreteDomain must be unique");

public:
    //! @brief Type of the elements of the domain.
    using discrete_element_type = DiscreteElement<DDims...>;

    //! @brief Type of the vectors holding one value (extent, stride, offset) per dimension.
    using discrete_vector_type = DiscreteVector<DDims...>;

private:
    DiscreteElement<DDims...> m_element_begin;

    DiscreteVector<DDims...> m_extents;

    DiscreteVector<DDims...> m_strides;

public:
    //! @brief Gets the number of dimensions of the domain.
    //!
    //! @return The number of discrete dimensions `sizeof...(DDims)`.
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return sizeof...(DDims);
    }

    //! @brief Default constructor.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain() = default;

    /// Construct a StridedDiscreteDomain by copies and merge of domains
    ///
    /// @param[in] domains The domains whose dimensions are merged; the front element, extents and
    /// strides of the result are the concatenation of theirs.
    template <concepts::strided_discrete_domain... DDoms>
    KOKKOS_FUNCTION constexpr explicit StridedDiscreteDomain(DDoms const&... domains)
        : m_element_begin(domains.front()...)
        , m_extents(domains.extents()...)
        , m_strides(domains.strides()...)
    {
    }

    /** Construct a StridedDiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param extents the number of points in each direction
     * @param strides the step between two elements
     */
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain(
            discrete_element_type const& element_begin,
            discrete_vector_type const& extents,
            discrete_vector_type const& strides)
        : m_element_begin(element_begin)
        , m_extents(extents)
        , m_strides(strides)
    {
    }

    //! @brief Copy constructor.
    //!
    //! @param[in] x The domain to copy.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain const& x) = default;

    //! @brief Move constructor.
    //!
    //! @param[in] x The domain to move from.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain&& x) = default;

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~StridedDiscreteDomain() = default;

    //! @brief Copy assignment operator.
    //!
    //! @param[in] x The domain to copy.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain const& x)
            = default;

    //! @brief Move assignment operator.
    //!
    //! @param[in] x The domain to move from.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain&& x) = default;

    //! @brief Checks whether two domains are equal.
    //!
    //! @details Two empty domains are always equal. Otherwise, the domains are equal if their front
    //! elements, their extents and their strides are all equal.
    //!
    //! @param[in] other The domain to compare with.
    //! @return `true` if the domains are equal, `false` otherwise.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr bool operator==(StridedDiscreteDomain<ODDims...> const& other) const
    {
        if (empty() && other.empty()) {
            return true;
        }
        return m_element_begin == other.m_element_begin && m_extents == other.m_extents
               && m_strides == other.m_strides;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Checks whether two domains are different.
    //!
    //! @param[in] other The domain to compare with.
    //! @return `!(*this == other)`.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr bool operator!=(StridedDiscreteDomain<ODDims...> const& other) const
    {
        return !(*this == other);
    }
#endif

    //! @brief Gets the number of elements of the domain.
    //!
    //! @return The product of the extents along all dimensions.
    KOKKOS_FUNCTION constexpr std::size_t size() const
    {
        return (1UL * ... * get<DDims>(m_extents));
    }

    //! @brief Gets the number of elements along each dimension.
    //!
    //! @return The extents of the domain.
    KOKKOS_FUNCTION constexpr discrete_vector_type extents() const noexcept
    {
        return m_extents;
    }

    //! @brief Gets the step between two consecutive elements along each dimension.
    //!
    //! @return The strides of the domain.
    KOKKOS_FUNCTION constexpr discrete_vector_type strides() const noexcept
    {
        return m_strides;
    }

    //! @brief Gets the number of elements along one dimension.
    //!
    //! @tparam QueryDDim The dimension to query, one of `DDims`.
    //! @return The extent of the domain along `QueryDDim`.
    template <concepts::discrete_dimension QueryDDim>
    KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDim> extent() const noexcept
    {
        return DiscreteVector<QueryDDim>(m_extents);
    }

    //! @brief Gets the first element of the domain.
    //!
    //! @return The element at the lower bound of the domain along each dimension.
    KOKKOS_FUNCTION constexpr discrete_element_type front() const noexcept
    {
        return m_element_begin;
    }

    //! @brief Gets the last element of the domain.
    //!
    //! @details Along each dimension, it is `front + (extent - 1) * stride`.
    //!
    //! @return The element at the upper bound of the domain along each dimension.
    KOKKOS_FUNCTION constexpr discrete_element_type back() const noexcept
    {
        return discrete_element_type(
                (DiscreteElement<DDims>(m_element_begin)
                 + (get<DDims>(m_extents) - 1) * get<DDims>(m_strides))...);
    }

    //! @brief Takes the first `n` elements along each dimension.
    //!
    //! @details The front element and the strides are kept. The size `n` is not checked against the
    //! extents.
    //!
    //! @param[in] n The number of elements to keep along each dimension.
    //! @return The sub-domain made of the first `n` elements along each dimension.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_first(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front(), n, m_strides);
    }

    //! @brief Takes the last `n` elements along each dimension.
    //!
    //! @details The strides are kept. The size `n` is not checked against the extents.
    //!
    //! @param[in] n The number of elements to keep along each dimension.
    //! @return The sub-domain made of the last `n` elements along each dimension.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_last(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front() + prod(extents() - n, m_strides), n, m_strides);
    }

    //! @brief Removes the first `n` elements along each dimension.
    //!
    //! @details The strides are kept. The size `n` is not checked against the extents.
    //!
    //! @param[in] n The number of elements to remove at the beginning along each dimension.
    //! @return The sub-domain without the first `n` elements along each dimension.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_first(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front() + prod(n, m_strides), extents() - n, m_strides);
    }

    //! @brief Removes the last `n` elements along each dimension.
    //!
    //! @details The front element and the strides are kept. The size `n` is not checked against the
    //! extents.
    //!
    //! @param[in] n The number of elements to remove at the end along each dimension.
    //! @return The sub-domain without the last `n` elements along each dimension.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_last(discrete_vector_type n) const
    {
        return StridedDiscreteDomain(front(), extents() - n, m_strides);
    }

    //! @brief Removes elements at both ends of the domain along each dimension.
    //!
    //! @details The strides are kept. The sizes are not checked against the extents.
    //!
    //! @param[in] n1 The number of elements to remove at the beginning along each dimension.
    //! @param[in] n2 The number of elements to remove at the end along each dimension.
    //! @return The sub-domain without the first `n1` and the last `n2` elements along each
    //! dimension.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove(
            discrete_vector_type n1,
            discrete_vector_type n2) const
    {
        return StridedDiscreteDomain(front() + prod(n1, m_strides), extents() - n1 - n2, m_strides);
    }

    //! @brief Gets the element at a given position in the domain.
    //!
    //! @details The position is not checked against the extents.
    //!
    //! @param[in] dvect The position, as the number of strides from the front element along each
    //! dimension.
    //! @return The element `front + dvect * strides`, component-wise.
    KOKKOS_FUNCTION constexpr DiscreteElement<DDims...> operator()(
            DiscreteVector<DDims...> const& dvect) const noexcept
    {
        return m_element_begin + prod(dvect, m_strides);
    }

    //! @brief Checks whether an element belongs to the domain.
    //!
    //! @details The element is given as a list of DiscreteElement whose dimensions together are the
    //! dimensions of the domain (checked by a `static_assert`). Along each dimension, the element
    //! belongs to the domain if it lies between the front and the back elements and is aligned on
    //! the stride.
    //!
    //! @param[in] delems The discrete elements making up the element to look for.
    //! @return `true` if the element belongs to the domain, `false` otherwise.
    template <class... DElems>
    KOKKOS_FUNCTION bool contains(DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        auto const contains_1d = [](DiscreteElementType const i,
                                    DiscreteElementType const b,
                                    DiscreteVectorElement const n,
                                    DiscreteVectorElement const s) {
            return (i >= b) && (i < (b + (n - 1) * s + 1)) && ((i - b) % s == 0);
        };
        DiscreteElement<DDims...> const delem(delems...);
        for (std::size_t i = 0; i < rank(); ++i) {
            if (!contains_1d(
                        detail::array(delem)[i],
                        detail::array(m_element_begin)[i],
                        detail::array(m_extents)[i],
                        detail::array(m_strides)[i])) {
                return false;
            }
        }
        return true;
    }

    //! @brief Gets the position of an element in the domain.
    //!
    //! @details The element is given as a list of DiscreteElement whose dimensions together are the
    //! dimensions of the domain (checked by a `static_assert`).
    //!
    //! @param[in] delems The discrete elements making up an element of the domain.
    //! @return The number of strides between the front element and the given element along each
    //! dimension.
    //! @pre The domain contains the element (checked by `KOKKOS_ASSERT`).
    template <class... DElems>
    KOKKOS_FUNCTION DiscreteVector<DDims...> distance_from_front(
            DElems const&... delems) const noexcept
    {
        static_assert(
                sizeof...(DDims) == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        static_assert((is_discrete_element_v<DElems> && ...), "Expected DiscreteElements");
        KOKKOS_ASSERT(contains(delems...))
        return DiscreteVector<DDims...>(
                ((DiscreteElement<DDims>(take<DDims>(delems...))
                  - DiscreteElement<DDims>(m_element_begin))
                 / DiscreteVector<DDims>(m_strides))...);
    }

    //! @brief Checks whether the domain has no element.
    //!
    //! @return `true` if `size() == 0`, `false` otherwise.
    KOKKOS_FUNCTION constexpr bool empty() const noexcept
    {
        return size() == 0;
    }

    //! @brief Checks whether the domain is not empty.
    //!
    //! @return `true` if the domain is not empty, `false` otherwise.
    KOKKOS_FUNCTION constexpr explicit operator bool()
    {
        return !empty();
    }

    //! @brief Gets an iterator to the first element of a one-dimensional domain.
    //!
    //! @return A StridedDiscreteDomainIterator pointing to the front element.
    KOKKOS_FUNCTION auto begin() const
        requires(sizeof...(DDims) == 1)
    {
        return StridedDiscreteDomainIterator<
                std::tuple_element_t<0, std::tuple<DDims...>>>(front(), m_strides);
    }

    //! @brief Gets an iterator past the last element of a one-dimensional domain.
    //!
    //! @return A StridedDiscreteDomainIterator pointing to `front + extent * stride`.
    KOKKOS_FUNCTION auto end() const
        requires(sizeof...(DDims) == 1)
    {
        return StridedDiscreteDomainIterator<std::tuple_element_t<
                0,
                std::tuple<DDims...>>>(m_element_begin + m_extents * m_strides, m_strides);
    }

    //! @brief Gets a constant iterator to the first element of a one-dimensional domain.
    //!
    //! @return A StridedDiscreteDomainIterator pointing to the front element.
    KOKKOS_FUNCTION auto cbegin() const
        requires(sizeof...(DDims) == 1)
    {
        return StridedDiscreteDomainIterator<
                std::tuple_element_t<0, std::tuple<DDims...>>>(front(), m_strides);
    }

    //! @brief Gets a constant iterator past the last element of a one-dimensional domain.
    //!
    //! @return A StridedDiscreteDomainIterator pointing to `front + extent * stride`.
    KOKKOS_FUNCTION auto cend() const
        requires(sizeof...(DDims) == 1)
    {
        return StridedDiscreteDomainIterator<std::tuple_element_t<
                0,
                std::tuple<DDims...>>>(m_element_begin + m_extents * m_strides, m_strides);
    }

    //! @brief Accesses the `n`-th element of a one-dimensional domain.
    //!
    //! @param[in] n The position of the element; it is not checked against the extent.
    //! @return The element `begin()[n]`.
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n)
        requires(sizeof...(DDims) == 1)
    {
        return begin()[n];
    }

    //! @brief Accesses the `n`-th element of a one-dimensional domain.
    //!
    //! @param[in] n The position of the element; it is not checked against the extent.
    //! @return The element `begin()[n]`.
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n) const
        requires(sizeof...(DDims) == 1)
    {
        return begin()[n];
    }
};

//! @brief Specialization of StridedDiscreteDomain for zero dimensions.
//!
//! @details This domain holds exactly one (empty) element: it has size 1 and is never empty.
//! @ingroup ddc_domains
template <>
class StridedDiscreteDomain<>
{
    template <concepts::discrete_dimension...>
    friend class StridedDiscreteDomain;

public:
    //! @brief Type of the elements of the domain.
    using discrete_element_type = DiscreteElement<>;

    //! @brief Type of the vectors holding one value per dimension.
    using discrete_vector_type = DiscreteVector<>;

    //! @brief Gets the number of dimensions of the domain.
    //!
    //! @return Always 0.
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return 0;
    }

    //! @brief Default constructor.
    KOKKOS_DEFAULTED_FUNCTION constexpr StridedDiscreteDomain() = default;

    // Construct a StridedDiscreteDomain from a reordered copy of `domain`
    //! @brief Constructs a zero-dimensional domain from a domain of any dimensions.
    //!
    //! @details The argument is ignored: the result is the unique zero-dimensional domain.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr explicit StridedDiscreteDomain(
            StridedDiscreteDomain<ODDims...> const& /*domain*/)
    {
    }

    /** Construct a StridedDiscreteDomain starting from element_begin with size points.
     * @param element_begin the lower bound in each direction
     * @param size the number of points in each direction
     * @param strides the step between two elements
     */
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain(
            [[maybe_unused]] discrete_element_type const& element_begin,
            [[maybe_unused]] discrete_vector_type const& size,
            [[maybe_unused]] discrete_vector_type const& strides)
    {
        // Parameters `element_begin`, `size` and `strides` are truly unused
    }

    //! @brief Copy constructor.
    //!
    //! @param[in] x The domain to copy.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain const& x) = default;

    //! @brief Move constructor.
    //!
    //! @param[in] x The domain to move from.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain(StridedDiscreteDomain&& x) = default;

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~StridedDiscreteDomain() = default;

    //! @brief Copy assignment operator.
    //!
    //! @param[in] x The domain to copy.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain const& x)
            = default;

    //! @brief Move assignment operator.
    //!
    //! @param[in] x The domain to move from.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomain& operator=(StridedDiscreteDomain&& x) = default;

    //! @brief Compares two zero-dimensional domains.
    //!
    //! @return Always `true`.
    KOKKOS_FUNCTION constexpr bool operator==(StridedDiscreteDomain const& /*other*/) const
    {
        return true;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Checks whether two zero-dimensional domains are different.
    //!
    //! @param[in] other The domain to compare with.
    //! @return `!(*this == other)`, that is always `false`.
    KOKKOS_FUNCTION constexpr bool operator!=(StridedDiscreteDomain const& other) const
    {
        return !(*this == other);
    }
#endif

    //! @brief Gets the number of elements of the domain.
    //!
    //! @return Always 1.
    static KOKKOS_FUNCTION constexpr std::size_t size()
    {
        return 1;
    }

    //! @brief Gets the number of elements along each dimension.
    //!
    //! @return An empty DiscreteVector.
    static KOKKOS_FUNCTION constexpr discrete_vector_type extents() noexcept
    {
        return {};
    }

    //! @brief Gets the first element of the domain.
    //!
    //! @return An empty DiscreteElement.
    static KOKKOS_FUNCTION constexpr discrete_element_type front() noexcept
    {
        return {};
    }

    //! @brief Gets the last element of the domain.
    //!
    //! @return An empty DiscreteElement.
    static KOKKOS_FUNCTION constexpr discrete_element_type back() noexcept
    {
        return {};
    }

    //! @brief Takes the first elements of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_first(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Takes the last elements of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain take_last(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes the first elements of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_first(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes the last elements of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove_last(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes elements at both ends of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomain remove(
            discrete_vector_type /*n1*/,
            discrete_vector_type /*n2*/) const
    {
        return *this;
    }

    //! @brief Gets the element at a given position in the domain.
    //!
    //! @return An empty DiscreteElement.
    KOKKOS_FUNCTION constexpr DiscreteElement<> operator()(
            DiscreteVector<> const& /*dvect*/) const noexcept
    {
        return {};
    }

    //! @brief Checks whether the element belongs to the domain.
    //!
    //! @return Always `true`.
    static KOKKOS_FUNCTION bool contains() noexcept
    {
        return true;
    }

    //! @brief Checks whether an empty element belongs to the domain.
    //!
    //! @return Always `true`.
    static KOKKOS_FUNCTION bool contains(DiscreteElement<>) noexcept
    {
        return true;
    }

    //! @brief Gets the position of the element in the domain.
    //!
    //! @return An empty DiscreteVector.
    static KOKKOS_FUNCTION DiscreteVector<> distance_from_front() noexcept
    {
        return {};
    }

    //! @brief Gets the position of an empty element in the domain.
    //!
    //! @return An empty DiscreteVector.
    static KOKKOS_FUNCTION DiscreteVector<> distance_from_front(DiscreteElement<>) noexcept
    {
        return {};
    }

    //! @brief Checks whether the domain has no element.
    //!
    //! @return Always `false`.
    static KOKKOS_FUNCTION constexpr bool empty() noexcept
    {
        return false;
    }

    //! @brief Checks whether the domain is not empty.
    //!
    //! @return Always `true`.
    KOKKOS_FUNCTION constexpr explicit operator bool()
    {
        return true;
    }
};

//! @brief Extracts the sub-domain made of some dimensions of a domain.
//!
//! @details The dimensions of the result are `QueryDDims`, in this order.
//!
//! @tparam QueryDDims The dimensions to extract, among the dimensions of `domain`.
//! @param[in] domain The domain from which to extract.
//! @return The StridedDiscreteDomain restricted to the dimensions `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr StridedDiscreteDomain<QueryDDims...> select(
        StridedDiscreteDomain<DDims...> const& domain)
{
    return StridedDiscreteDomain<QueryDDims...>(domain);
}

namespace detail {

template <ddc::concepts::type_seq T>
struct ConvertTypeSeqToStridedDiscreteDomain
{
};

template <ddc::concepts::discrete_dimension... DDims>
struct ConvertTypeSeqToStridedDiscreteDomain<ddc::TypeSeq<DDims...>>
{
    using type = StridedDiscreteDomain<DDims...>;
};

template <ddc::concepts::type_seq T>
using convert_type_seq_to_strided_discrete_domain_t
        = ConvertTypeSeqToStridedDiscreteDomain<T>::type;

} // namespace detail

// Computes the subtraction DDom_a - DDom_b in the sense of linear spaces(retained dimensions are those in DDom_a which are not in DDom_b)
//! @brief Removes from a domain the dimensions of another domain.
//!
//! @details The dimensions of the result are the dimensions of `DDom_a` that are not dimensions
//! of the second domain. The second domain is only used for its type.
//!
//! @param[in] DDom_a The domain from which dimensions are removed.
//! @return The domain restricted to the dimensions of `DDom_a` not present in the second domain.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDimsA, concepts::discrete_dimension... DDimsB>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        StridedDiscreteDomain<DDimsA...> const& DDom_a,
        StridedDiscreteDomain<DDimsB...> const& /*DDom_b*/) noexcept
{
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_strided_discrete_domain_t<type_seq_r>(DDom_a);
}

//! Remove the dimensions DDimsB from DDom_a
//! @param[in] DDom_a The discrete domain on which to remove dimensions
//! @return The discrete domain without DDimsB dimensions
//! @tparam DDimsB The dimensions to remove from DDom_a.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDimsB, concepts::discrete_dimension... DDimsA>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        StridedDiscreteDomain<DDimsA...> const& DDom_a) noexcept
{
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_strided_discrete_domain_t<type_seq_r>(DDom_a);
}

namespace detail {

// Checks if dimension of DDom_a is DDim1. If not, returns restriction to DDim2 of DDom_b. May not be useful in its own, it helps for replace_dim_of
template <typename DDim1, typename DDim2, typename DDimA, typename... DDimsB>
KOKKOS_FUNCTION constexpr std::conditional_t<
        std::is_same_v<DDimA, DDim1>,
        ddc::StridedDiscreteDomain<DDim2>,
        ddc::StridedDiscreteDomain<DDimA>>
replace_dim_of_1d(
        [[maybe_unused]] StridedDiscreteDomain<DDimA> const& DDom_a,
        [[maybe_unused]] StridedDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    if constexpr (std::is_same_v<DDimA, DDim1>) {
        return ddc::StridedDiscreteDomain<DDim2>(DDom_b);
    } else {
        return DDom_a;
    }
}

} // namespace detail

// Replace in DDom_a the dimension Dim1 by the dimension Dim2 of DDom_b
//! @brief Replaces a dimension of a domain by a dimension taken from another domain.
//!
//! @details The result has the dimensions of `DDom_a`, where `DDim1` is replaced by `DDim2`, at
//! the same position. The part of the result along `DDim2` is the restriction of `DDom_b` to
//! `DDim2`. The other dimensions keep the front elements, extents and strides of `DDom_a`.
//!
//! @tparam DDim1 The dimension of `DDom_a` to replace.
//! @tparam DDim2 The dimension of `DDom_b` that replaces `DDim1`.
//! @param[in] DDom_a The domain in which a dimension is replaced.
//! @param[in] DDom_b The domain providing the replacing dimension.
//! @return The domain with `DDim1` replaced by `DDim2`.
//! @ingroup ddc_domains
template <typename DDim1, typename DDim2, typename... DDimsA, typename... DDimsB>
KOKKOS_FUNCTION constexpr auto replace_dim_of(
        StridedDiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] StridedDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    // TODO : static_asserts
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDim1>;
    using TagSeqC = TypeSeq<DDim2>;

    using type_seq_r = ddc::type_seq_replace_t<TagSeqA, TagSeqB, TagSeqC>;
    return ddc::detail::convert_type_seq_to_strided_discrete_domain_t<type_seq_r>(
            detail::replace_dim_of_1d<
                    DDim1,
                    DDim2,
                    DDimsA,
                    DDimsB...>(ddc::StridedDiscreteDomain<DDimsA>(DDom_a), DDom_b)...);
}

//! @brief Gets the number of elements of a domain along some dimensions.
//!
//! @tparam QueryDDims The dimensions to query, among the dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return The extents of `domain` along `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDims...> extents(
        StridedDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteVector<QueryDDims...>(StridedDiscreteDomain<QueryDDims>(domain).size()...);
}

//! @brief Gets the first element of a domain along some dimensions.
//!
//! @tparam QueryDDims The dimensions to query, among the dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return The front element of `domain` restricted to `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> front(
        StridedDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(StridedDiscreteDomain<QueryDDims>(domain).front()...);
}

//! @brief Gets the last element of a domain along some dimensions.
//!
//! @tparam QueryDDims The dimensions to query, among the dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return The back element of `domain` restricted to `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> back(
        StridedDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(StridedDiscreteDomain<QueryDDims>(domain).back()...);
}

//! @brief Random-access iterator over the elements of a one-dimensional StridedDiscreteDomain.
//!
//! @details The iterator holds the current element and the stride; incrementing it advances the
//! element by the stride.
//!
//! @tparam DDim The discrete dimension of the iterated elements.
//! @ingroup ddc_domains
template <concepts::discrete_dimension DDim>
struct StridedDiscreteDomainIterator
{
private:
    DiscreteElement<DDim> m_value = DiscreteElement<DDim>();

    DiscreteVector<DDim> m_stride = DiscreteVector<DDim>();

public:
    //! @brief Iterator category, see `std::random_access_iterator_tag`.
    using iterator_category = std::random_access_iterator_tag;

    //! @brief Type of the elements the iterator points to.
    using value_type = DiscreteElement<DDim>;

    //! @brief Signed type used for iterator arithmetic.
    using difference_type = std::ptrdiff_t;

    //! @brief Default constructor.
    KOKKOS_DEFAULTED_FUNCTION StridedDiscreteDomainIterator() = default;

    //! @brief Constructs an iterator pointing to an element.
    //!
    //! @param[in] value The element the iterator points to.
    //! @param[in] stride The step added to the element when the iterator is incremented.
    KOKKOS_FUNCTION constexpr explicit StridedDiscreteDomainIterator(
            DiscreteElement<DDim> value,
            DiscreteVector<DDim> stride)
        : m_value(value)
        , m_stride(stride)
    {
    }

    //! @brief Dereferences the iterator.
    //!
    //! @return The element the iterator points to.
    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator*() const noexcept
    {
        return m_value;
    }

    //! @brief Pre-increment: advances the iterator by one stride.
    //!
    //! @return A reference to this iterator.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator++()
    {
        m_value += m_stride;
        return *this;
    }

    //! @brief Post-increment: advances the iterator by one stride.
    //!
    //! @return A copy of the iterator before the increment.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator++(int)
    {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    //! @brief Pre-decrement: moves the iterator back by one stride.
    //!
    //! @return A reference to this iterator.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator--()
    {
        m_value -= m_stride;
        return *this;
    }

    //! @brief Post-decrement: moves the iterator back by one stride.
    //!
    //! @return A copy of the iterator before the decrement.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator--(int)
    {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    //! @brief Advances the iterator by `n` strides (backwards if `n` is negative).
    //!
    //! @param[in] n The number of strides.
    //! @return A reference to this iterator.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator+=(difference_type n)
    {
        if (n >= static_cast<difference_type>(0)) {
            m_value += static_cast<DiscreteElementType>(n) * m_stride;
        } else {
            m_value -= static_cast<DiscreteElementType>(-n) * m_stride;
        }
        return *this;
    }

    //! @brief Moves the iterator back by `n` strides (forwards if `n` is negative).
    //!
    //! @param[in] n The number of strides.
    //! @return A reference to this iterator.
    KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator& operator-=(difference_type n)
    {
        if (n >= static_cast<difference_type>(0)) {
            m_value -= static_cast<DiscreteElementType>(n) * m_stride;
        } else {
            m_value += static_cast<DiscreteElementType>(-n) * m_stride;
        }
        return *this;
    }

    //! @brief Gets the element `n` strides away from the iterator.
    //!
    //! @param[in] n The number of strides.
    //! @return The element `*it + n * stride`; the iterator is not modified.
    KOKKOS_FUNCTION constexpr DiscreteElement<DDim> operator[](difference_type n) const
    {
        return m_value + n * m_stride;
    }

    //! @brief Checks whether two iterators point to the same element.
    //!
    //! @details Only the pointed elements are compared, not the strides.
    //!
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if both iterators point to the same element.
    friend KOKKOS_FUNCTION constexpr bool operator==(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return xx.m_value == yy.m_value;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Checks whether two iterators point to different elements.
    //!
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if the iterators point to different elements.
    friend KOKKOS_FUNCTION constexpr bool operator!=(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return xx.m_value != yy.m_value;
    }
#endif

    //! @brief Checks whether an iterator points to an element before another one.
    //!
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if the element of `xx` is smaller than the element of `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator<(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return xx.m_value < yy.m_value;
    }

    //! @brief Checks whether an iterator points to an element after another one.
    //!
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if the element of `xx` is greater than the element of `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator>(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return yy < xx;
    }

    //! @brief Checks whether an iterator does not point after another one.
    //!
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if the element of `xx` is smaller than or equal to the element of `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator<=(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return !(yy < xx);
    }

    //! @brief Checks whether an iterator does not point before another one.
    //!
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return `true` if the element of `xx` is greater than or equal to the element of `yy`.
    friend KOKKOS_FUNCTION constexpr bool operator>=(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return !(xx < yy);
    }

    //! @brief Advances a copy of an iterator by `n` strides.
    //!
    //! @param[in] i The iterator.
    //! @param[in] n The number of strides.
    //! @return The advanced iterator.
    friend KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator+(
            StridedDiscreteDomainIterator i,
            difference_type n)
    {
        return i += n;
    }

    //! @brief Advances a copy of an iterator by `n` strides.
    //!
    //! @param[in] n The number of strides.
    //! @param[in] i The iterator.
    //! @return The advanced iterator.
    friend KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator+(
            difference_type n,
            StridedDiscreteDomainIterator i)
    {
        return i += n;
    }

    //! @brief Moves back a copy of an iterator by `n` strides.
    //!
    //! @param[in] i The iterator.
    //! @param[in] n The number of strides.
    //! @return The moved iterator.
    friend KOKKOS_FUNCTION constexpr StridedDiscreteDomainIterator operator-(
            StridedDiscreteDomainIterator i,
            difference_type n)
    {
        return i -= n;
    }

    //! @brief Computes the difference between the elements two iterators point to.
    //!
    //! @details The result is the difference between the pointed elements; it is not divided by the
    //! stride.
    //!
    //! @param[in] xx The first iterator.
    //! @param[in] yy The second iterator.
    //! @return The difference between the element of `xx` and the element of `yy`.
    friend KOKKOS_FUNCTION constexpr difference_type operator-(
            StridedDiscreteDomainIterator const& xx,
            StridedDiscreteDomainIterator const& yy)
    {
        return (yy.m_value > xx.m_value) ? (-static_cast<difference_type>(yy.m_value - xx.m_value))
                                         : (xx.m_value - yy.m_value);
    }
};

} // namespace ddc
