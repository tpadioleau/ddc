// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides SparseDiscreteDomain, a multidimensional domain of discrete elements listed
//! explicitly along each dimension, and its related free functions.
#pragma once

#include <cstddef>
#include <functional>
#include <iterator>
#include <string>
#include <tuple>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "detail/tagged_vector.hpp"

#include "discrete_dimension.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "type_seq.hpp"

namespace ddc {

template <concepts::discrete_dimension... DDims>
class SparseDiscreteDomain;

//! @brief Type trait that is true if `T` is a SparseDiscreteDomain.
//!
//! @details This primary template is `std::false_type`; it is specialized for
//! SparseDiscreteDomain.
//!
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
struct is_sparse_discrete_domain : std::false_type
{
};

//! @brief Specialization of is_sparse_discrete_domain for SparseDiscreteDomain (true).
//!
//! @tparam Tags The discrete dimensions of the SparseDiscreteDomain.
//! @ingroup ddc_domains
template <class... Tags>
struct is_sparse_discrete_domain<SparseDiscreteDomain<Tags...>> : std::true_type
{
};

//! @brief Shorthand for `is_sparse_discrete_domain<T>::value`.
//!
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
inline constexpr bool is_sparse_discrete_domain_v = is_sparse_discrete_domain<T>::value;

namespace concepts {

//! @brief Concept satisfied by the types that are a SparseDiscreteDomain.
//!
//! @tparam T The type to constrain.
//! @ingroup ddc_domains
template <class T>
concept sparse_discrete_domain = is_sparse_discrete_domain_v<T>;

} // namespace concepts

namespace detail {

template <class... Tags>
struct ToTypeSeq<SparseDiscreteDomain<Tags...>>
{
    using type = ddc::TypeSeq<Tags...>;
};

template <ddc::concepts::discrete_dimension... DDims, ddc::concepts::discrete_dimension... ODDims>
struct Rebind<SparseDiscreteDomain<DDims...>, ddc::TypeSeq<ODDims...>>
{
    using type = SparseDiscreteDomain<ODDims...>;
};

template <class InputIt1, class InputIt2>
KOKKOS_FUNCTION bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2)
{
    for (; first1 != last1; ++first1, ++first2) {
        if (!(*first1 == *first2)) {
            return false;
        }
    }

    return true;
}

template <class ForwardIt, class T = std::iterator_traits<ForwardIt>::value_type, class Compare>
KOKKOS_FUNCTION ForwardIt lower_bound(ForwardIt first, ForwardIt last, T const& value, Compare comp)
{
    ForwardIt it;
    typename std::iterator_traits<ForwardIt>::difference_type count = last - first;
    while (count > 0) {
        it = first;
        typename std::iterator_traits<ForwardIt>::difference_type const step = count / 2;
        it += step;

        if (comp(*it, value)) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }

    return first;
}

template <class ForwardIt, class T = std::iterator_traits<ForwardIt>::value_type, class Compare>
KOKKOS_FUNCTION bool binary_search(ForwardIt first, ForwardIt last, T const& value, Compare comp)
{
    first = ::ddc::detail::lower_bound(first, last, value, comp);
    return (!(first == last) && !(comp(value, *first)));
}

template <ddc::concepts::discrete_dimension DDim>
Kokkos::View<DiscreteElementType*, Kokkos::SharedSpace> extract_uid(
        Kokkos::DefaultExecutionSpace const& exec_space,
        Kokkos::View<DiscreteElement<DDim>*, Kokkos::SharedSpace> const& input)
{
    Kokkos::View<DiscreteElementType*, Kokkos::SharedSpace> output(input.label(), input.size());
    Kokkos::parallel_for(
            "SparseDiscreteDomainCtor",
            Kokkos::RangePolicy<
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(exec_space, 0, output.size()),
            KOKKOS_LAMBDA(std::size_t const i) { output(i) = input(i).uid(); });
    return output;
}

} // namespace detail

//! @brief A multidimensional domain whose discrete elements are listed explicitly along each
//! dimension.
//!
//! @details For each dimension, the domain stores a Kokkos::View (in `Kokkos::SharedSpace`) of the
//! identifiers (`uid`) of the elements along this dimension. The domain contains every
//! combination of these elements. The lists of identifiers must be sorted in ascending order:
//! this is what the binary searches of contains() and distance_from_front() rely on, but it is
//! not checked.
//!
//! The dimensions must be unique (checked by a `static_assert`). The domain does not own a unique
//! copy of the data: copies of a domain share the same views.
//!
//! @tparam DDims The discrete dimensions of the domain.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDims>
class SparseDiscreteDomain
{
    template <concepts::discrete_dimension...>
    friend class SparseDiscreteDomain;

    static_assert(
            type_seq_is_unique_v<TypeSeq<DDims...>>,
            "The dimensions of a SparseDiscreteDomain must be unique");

public:
    //! @brief Type of the elements of the domain.
    using discrete_element_type = DiscreteElement<DDims...>;

    //! @brief Type of the vectors holding one value (extent, offset) per dimension.
    using discrete_vector_type = DiscreteVector<DDims...>;

private:
    detail::TaggedVector<Kokkos::View<DiscreteElementType*, Kokkos::SharedSpace>, DDims...> m_views;

public:
    //! @brief Gets the number of dimensions of the domain.
    //!
    //! @return The number of discrete dimensions `sizeof...(DDims)`.
    static KOKKOS_FUNCTION constexpr std::size_t rank()
    {
        return sizeof...(DDims);
    }

    //! @brief Default constructor.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain() = default;

    /// Construct a SparseDiscreteDomain by copies and merge of domains
    ///
    /// @param[in] domains The domains whose dimensions are merged; the lists of elements of the
    /// result are the ones of the given domains.
    template <concepts::sparse_discrete_domain... DDoms>
    KOKKOS_FUNCTION constexpr explicit SparseDiscreteDomain(DDoms const&... domains)
        : m_views(domains.m_views...)
    {
    }

    /** Construct a SparseDiscreteDomain with Kokkos::View explicitly listing the discrete elements.
     * @param views list of Kokkos::View
     */
    explicit SparseDiscreteDomain(
            Kokkos::View<DiscreteElement<DDims>*, Kokkos::SharedSpace> const&... views)
    {
        Kokkos::DefaultExecutionSpace const exec_space;
        ((m_views[type_seq_rank_v<DDims, TypeSeq<DDims...>>]
          = detail::extract_uid(exec_space, views)),
         ...);
        exec_space.fence("SparseDiscreteDomainCtor");
    }

    //! @brief Copy constructor.
    //!
    //! @param[in] x The domain to copy.
    //!
    //! @note The copy shares the underlying data of the Kokkos::View with the original.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain(SparseDiscreteDomain const& x) = default;

    //! @brief Move constructor.
    //!
    //! @param[in] x The domain to move from.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain(SparseDiscreteDomain&& x) = default;

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~SparseDiscreteDomain() = default;

    //! @brief Copy assignment operator.
    //!
    //! @param[in] x The domain to copy.
    //! @return A reference to this domain.
    //!
    //! @note The assigned domain shares the underlying data of the Kokkos::View with `x`.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain& operator=(SparseDiscreteDomain const& x)
            = default;

    //! @brief Move assignment operator.
    //!
    //! @param[in] x The domain to move from.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain& operator=(SparseDiscreteDomain&& x) = default;

    //! @brief Checks whether two domains are equal.
    //!
    //! @details Two empty domains are always equal. Otherwise, the domains are equal if, along each
    //! dimension, they list the same identifiers in the same order. If the dimensions of `other`
    //! are the same as the ones of this domain in a different order, `other` is first reordered.
    //!
    //! @param[in] other The domain to compare with.
    //! @return `true` if the domains are equal, `false` otherwise.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr bool operator==(SparseDiscreteDomain<ODDims...> const& other) const
    {
        if constexpr ((std::is_same_v<DDims, ODDims> && ...)) {
            if (empty() && other.empty()) {
                return true;
            }
            for (std::size_t i = 0; i < m_views.size(); ++i) {
                if (m_views[i].size() != other.m_views[i].size()) {
                    return false;
                }
                if (!detail::
                            equal(m_views[i].data(),
                                  m_views[i].data() + m_views[i].size(),
                                  other.m_views[i].data())) {
                    return false;
                }
            }
            return true;
        } else {
            return *this == SparseDiscreteDomain(other);
        }
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Checks whether two domains are different.
    //!
    //! @param[in] other The domain to compare with.
    //! @return `!(*this == other)`.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr bool operator!=(SparseDiscreteDomain<ODDims...> const& other) const
    {
        return !(*this == other);
    }
#endif

    //! @brief Gets the number of elements of the domain.
    //!
    //! @return The product of the number of listed elements along all dimensions.
    KOKKOS_FUNCTION constexpr std::size_t size() const
    {
        return (1UL * ... * get<DDims>(m_views).size());
    }

    //! @brief Gets the number of listed elements along each dimension.
    //!
    //! @return The extents of the domain.
    KOKKOS_FUNCTION constexpr discrete_vector_type extents() const noexcept
    {
        return discrete_vector_type(get<DDims>(m_views).size()...);
    }

    //! @brief Gets the lists of element identifiers of the domain.
    //!
    //! @return A copy of the per-dimension Kokkos::View of identifiers (`uid`), which shares its
    //! data with the domain.
    KOKKOS_FUNCTION constexpr auto discrete_elements() const noexcept
    {
        return m_views;
    }

    //! @brief Gets the number of listed elements along one dimension.
    //!
    //! @tparam QueryDDim The dimension to query, one of `DDims`.
    //! @return The extent of the domain along `QueryDDim`.
    template <concepts::discrete_dimension QueryDDim>
    KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDim> extent() const noexcept
    {
        return DiscreteVector<QueryDDim>(get<QueryDDim>(m_views).size());
    }

    //! @brief Gets the first element of the domain.
    //!
    //! @details Along each dimension, it is the first element of the list.
    //!
    //! @return The element made of the first listed element along each dimension.
    KOKKOS_FUNCTION constexpr discrete_element_type front() const noexcept
    {
        return discrete_element_type(get<DDims>(m_views)(0)...);
    }

    //! @brief Gets the last element of the domain.
    //!
    //! @details Along each dimension, it is the last element of the list.
    //!
    //! @return The element made of the last listed element along each dimension.
    KOKKOS_FUNCTION constexpr discrete_element_type back() const noexcept
    {
        return discrete_element_type(get<DDims>(m_views)(get<DDims>(m_views).size() - 1)...);
    }

    //! @brief Takes the first `n` elements along each dimension.
    //!
    //! @param[in] n The number of elements to keep along each dimension.
    //! @return The sub-domain made of the first `n` elements along each dimension.
    //!
    //! @warning The implementation builds the result with a constructor taking a first element and
    //! extents, which SparseDiscreteDomain does not provide; it is not used in the tests of the
    //! repository. Not specified by the source whether it can be instantiated.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain take_first(discrete_vector_type n) const
    {
        return SparseDiscreteDomain(front(), n);
    }

    //! @brief Takes the last `n` elements along each dimension.
    //!
    //! @param[in] n The number of elements to keep along each dimension.
    //! @return The sub-domain made of the last `n` elements along each dimension.
    //!
    //! @warning Same limitation as take_first(): the implementation relies on a constructor and
    //! on a function `prod` that are not provided for SparseDiscreteDomain. Not specified by the
    //! source whether it can be instantiated.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain take_last(discrete_vector_type n) const
    {
        return SparseDiscreteDomain(front() + prod(extents() - n), n);
    }

    //! @brief Removes the first `n` elements along each dimension.
    //!
    //! @param[in] n The number of elements to remove at the beginning along each dimension.
    //! @return The sub-domain without the first `n` elements along each dimension.
    //!
    //! @warning Same limitation as take_first(). Not specified by the source whether it can be
    //! instantiated.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain remove_first(discrete_vector_type n) const
    {
        return SparseDiscreteDomain(front() + prod(n), extents() - n);
    }

    //! @brief Removes the last `n` elements along each dimension.
    //!
    //! @param[in] n The number of elements to remove at the end along each dimension.
    //! @return The sub-domain without the last `n` elements along each dimension.
    //!
    //! @warning Same limitation as take_first(). Not specified by the source whether it can be
    //! instantiated.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain remove_last(discrete_vector_type n) const
    {
        return SparseDiscreteDomain(front(), extents() - n);
    }

    //! @brief Removes elements at both ends of the domain along each dimension.
    //!
    //! @param[in] n1 The number of elements to remove at the beginning along each dimension.
    //! @param[in] n2 The number of elements to remove at the end along each dimension.
    //! @return The sub-domain without the first `n1` and the last `n2` elements along each
    //! dimension.
    //!
    //! @warning Same limitation as take_first(). Not specified by the source whether it can be
    //! instantiated.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain remove(
            discrete_vector_type n1,
            discrete_vector_type n2) const
    {
        return SparseDiscreteDomain(front() + prod(n1), extents() - n1 - n2);
    }

    //! @brief Gets the element at a given position in the domain.
    //!
    //! @details The position is not checked against the extents.
    //!
    //! @param[in] dvect The position, as the index in the list of elements along each dimension.
    //! @return The element made of the listed element at index `dvect` along each dimension.
    KOKKOS_FUNCTION constexpr DiscreteElement<DDims...> operator()(
            DiscreteVector<DDims...> const& dvect) const noexcept
    {
        return DiscreteElement<DDims...>(get<DDims>(m_views)(get<DDims>(dvect))...);
    }

    //! @brief Checks whether an element belongs to the domain.
    //!
    //! @details The element is given as a list of DiscreteElement whose dimensions together are the
    //! dimensions of the domain (checked by a `static_assert`). Along each dimension, a binary
    //! search is performed in the list of identifiers, which must therefore be sorted.
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
        DiscreteElement<DDims...> const delem(delems...);
        for (std::size_t i = 0; i < rank(); ++i) {
            if (!detail::binary_search(
                        m_views[i].data(),
                        m_views[i].data() + m_views[i].size(),
                        detail::array(delem)[i],
                        std::less {})) {
                return false;
            }
        }
        return true;
    }

    //! @brief Gets the position of an element in the domain.
    //!
    //! @details The element is given as a list of DiscreteElement whose dimensions together are the
    //! dimensions of the domain (checked by a `static_assert`). Along each dimension, the
    //! position is found by a binary search in the list of identifiers.
    //!
    //! @param[in] delems The discrete elements making up an element of the domain.
    //! @return The index of the element in the list of each dimension.
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
                (detail::lower_bound(
                         get<DDims>(m_views).data(),
                         get<DDims>(m_views).data() + get<DDims>(m_views).size(),
                         uid<DDims>(take<DDims>(delems...)),
                         std::less {})
                 - get<DDims>(m_views).data())...);
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

    //! @brief Gets a pointer to the first element identifier of a one-dimensional domain.
    //!
    //! @return A pointer to the first `DiscreteElementType` of the list of identifiers.
    KOKKOS_FUNCTION auto begin() const
        requires(sizeof...(DDims) == 1)
    {
        using DDim0 = std::tuple_element_t<0, std::tuple<DDims...>>;
        return get<DDim0>(m_views).data();
    }

    //! @brief Gets a pointer past the last element identifier of a one-dimensional domain.
    //!
    //! @return A pointer past the last `DiscreteElementType` of the list of identifiers.
    KOKKOS_FUNCTION auto end() const
        requires(sizeof...(DDims) == 1)
    {
        using DDim0 = std::tuple_element_t<0, std::tuple<DDims...>>;
        return get<DDim0>(m_views).data() + get<DDim0>(m_views).size();
    }

    //! @brief Gets a pointer to the first element identifier of a one-dimensional domain.
    //!
    //! @return A pointer to the first `DiscreteElementType` of the list of identifiers.
    KOKKOS_FUNCTION auto cbegin() const
        requires(sizeof...(DDims) == 1)
    {
        using DDim0 = std::tuple_element_t<0, std::tuple<DDims...>>;
        return get<DDim0>(m_views).data();
    }

    //! @brief Gets a pointer past the last element identifier of a one-dimensional domain.
    //!
    //! @return A pointer past the last `DiscreteElementType` of the list of identifiers.
    KOKKOS_FUNCTION auto cend() const
        requires(sizeof...(DDims) == 1)
    {
        using DDim0 = std::tuple_element_t<0, std::tuple<DDims...>>;
        return get<DDim0>(m_views).data() + get<DDim0>(m_views).size();
    }

    //! @brief Accesses the `n`-th element identifier of a one-dimensional domain.
    //!
    //! @param[in] n The position in the list of identifiers; it is not checked.
    //! @return The identifier `begin()[n]`.
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n)
        requires(sizeof...(DDims) == 1)
    {
        return begin()[n];
    }

    //! @brief Accesses the `n`-th element identifier of a one-dimensional domain.
    //!
    //! @param[in] n The position in the list of identifiers; it is not checked.
    //! @return The identifier `begin()[n]`.
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](std::size_t n) const
        requires(sizeof...(DDims) == 1)
    {
        return begin()[n];
    }
};

//! @brief Specialization of SparseDiscreteDomain for zero dimensions.
//!
//! @details This domain holds exactly one (empty) element: it has size 1 and is never empty.
//! @ingroup ddc_domains
template <>
class SparseDiscreteDomain<>
{
    template <concepts::discrete_dimension...>
    friend class SparseDiscreteDomain;

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
    KOKKOS_DEFAULTED_FUNCTION constexpr SparseDiscreteDomain() = default;

    // Construct a SparseDiscreteDomain from a reordered copy of `domain`
    //! @brief Constructs a zero-dimensional domain from a domain of any dimensions.
    //!
    //! @details The argument is ignored: the result is the unique zero-dimensional domain.
    template <concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION constexpr explicit SparseDiscreteDomain(
            SparseDiscreteDomain<ODDims...> const& /*domain*/)
    {
    }

    //! @brief Copy constructor.
    //!
    //! @param[in] x The domain to copy.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain(SparseDiscreteDomain const& x) = default;

    //! @brief Move constructor.
    //!
    //! @param[in] x The domain to move from.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain(SparseDiscreteDomain&& x) = default;

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~SparseDiscreteDomain() = default;

    //! @brief Copy assignment operator.
    //!
    //! @param[in] x The domain to copy.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain& operator=(SparseDiscreteDomain const& x)
            = default;

    //! @brief Move assignment operator.
    //!
    //! @param[in] x The domain to move from.
    //! @return A reference to this domain.
    KOKKOS_DEFAULTED_FUNCTION SparseDiscreteDomain& operator=(SparseDiscreteDomain&& x) = default;

    //! @brief Compares two zero-dimensional domains.
    //!
    //! @return Always `true`.
    KOKKOS_FUNCTION constexpr bool operator==(SparseDiscreteDomain const& /*other*/) const
    {
        return true;
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Checks whether two zero-dimensional domains are different.
    //!
    //! @param[in] other The domain to compare with.
    //! @return `!(*this == other)`, that is always `false`.
    KOKKOS_FUNCTION constexpr bool operator!=(SparseDiscreteDomain const& other) const
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
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain take_first(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Takes the last elements of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain take_last(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes the first elements of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain remove_first(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes the last elements of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain remove_last(discrete_vector_type /*n*/) const
    {
        return *this;
    }

    //! @brief Removes elements at both ends of the domain.
    //!
    //! @return A copy of the domain.
    KOKKOS_FUNCTION constexpr SparseDiscreteDomain remove(
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
//! @return The SparseDiscreteDomain restricted to the dimensions `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr SparseDiscreteDomain<QueryDDims...> select(
        SparseDiscreteDomain<DDims...> const& domain)
{
    return SparseDiscreteDomain<QueryDDims...>(domain);
}

namespace detail {

template <ddc::concepts::type_seq T>
struct ConvertTypeSeqToSparseDiscreteDomain
{
};

template <ddc::concepts::discrete_dimension... DDims>
struct ConvertTypeSeqToSparseDiscreteDomain<ddc::TypeSeq<DDims...>>
{
    using type = SparseDiscreteDomain<DDims...>;
};

template <ddc::concepts::type_seq T>
using convert_type_seq_to_sparse_discrete_domain_t = ConvertTypeSeqToSparseDiscreteDomain<T>::type;

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
        SparseDiscreteDomain<DDimsA...> const& DDom_a,
        SparseDiscreteDomain<DDimsB...> const& /*DDom_b*/) noexcept
{
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_sparse_discrete_domain_t<type_seq_r>(DDom_a);
}

//! Remove the dimensions DDimsB from DDom_a
//! @param[in] DDom_a The discrete domain on which to remove dimensions
//! @return The discrete domain without DDimsB dimensions
//! @tparam DDimsB The dimensions to remove from DDom_a.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... DDimsB, concepts::discrete_dimension... DDimsA>
KOKKOS_FUNCTION constexpr auto remove_dims_of(
        SparseDiscreteDomain<DDimsA...> const& DDom_a) noexcept
{
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDimsB...>;

    using type_seq_r = type_seq_remove_t<TagSeqA, TagSeqB>;
    return detail::convert_type_seq_to_sparse_discrete_domain_t<type_seq_r>(DDom_a);
}

namespace detail {

// Checks if dimension of DDom_a is DDim1. If not, returns restriction to DDim2 of DDom_b. May not be useful in its own, it helps for replace_dim_of
template <
        ddc::concepts::discrete_dimension DDim1,
        ddc::concepts::discrete_dimension DDim2,
        ddc::concepts::discrete_dimension DDimA,
        ddc::concepts::discrete_dimension... DDimsB>
KOKKOS_FUNCTION constexpr std::conditional_t<
        std::is_same_v<DDimA, DDim1>,
        ddc::SparseDiscreteDomain<DDim2>,
        ddc::SparseDiscreteDomain<DDimA>>
replace_dim_of_1d(
        [[maybe_unused]] SparseDiscreteDomain<DDimA> const& DDom_a,
        [[maybe_unused]] SparseDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    if constexpr (std::is_same_v<DDimA, DDim1>) {
        return ddc::SparseDiscreteDomain<DDim2>(DDom_b);
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
//! `DDim2`. The other dimensions keep the elements of `DDom_a`.
//!
//! @tparam DDim1 The dimension of `DDom_a` to replace.
//! @tparam DDim2 The dimension of `DDom_b` that replaces `DDim1`.
//! @param[in] DDom_a The domain in which a dimension is replaced.
//! @param[in] DDom_b The domain providing the replacing dimension.
//! @return The domain with `DDim1` replaced by `DDim2`.
//! @ingroup ddc_domains
template <
        concepts::discrete_dimension DDim1,
        concepts::discrete_dimension DDim2,
        concepts::discrete_dimension... DDimsA,
        concepts::discrete_dimension... DDimsB>
KOKKOS_FUNCTION constexpr auto replace_dim_of(
        SparseDiscreteDomain<DDimsA...> const& DDom_a,
        [[maybe_unused]] SparseDiscreteDomain<DDimsB...> const& DDom_b) noexcept
{
    // TODO : static_asserts
    using TagSeqA = TypeSeq<DDimsA...>;
    using TagSeqB = TypeSeq<DDim1>;
    using TagSeqC = TypeSeq<DDim2>;

    using type_seq_r = ddc::type_seq_replace_t<TagSeqA, TagSeqB, TagSeqC>;
    return ddc::detail::convert_type_seq_to_sparse_discrete_domain_t<type_seq_r>(
            detail::replace_dim_of_1d<
                    DDim1,
                    DDim2,
                    DDimsA,
                    DDimsB...>(ddc::SparseDiscreteDomain<DDimsA>(DDom_a), DDom_b)...);
}

//! @brief Gets the number of elements of a domain along some dimensions.
//!
//! @tparam QueryDDims The dimensions to query, among the dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return The extents of `domain` along `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryDDims...> extents(
        SparseDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteVector<QueryDDims...>(SparseDiscreteDomain<QueryDDims>(domain).size()...);
}

//! @brief Gets the first element of a domain along some dimensions.
//!
//! @tparam QueryDDims The dimensions to query, among the dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return The front element of `domain` restricted to `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> front(
        SparseDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(SparseDiscreteDomain<QueryDDims>(domain).front()...);
}

//! @brief Gets the last element of a domain along some dimensions.
//!
//! @tparam QueryDDims The dimensions to query, among the dimensions of `domain`.
//! @param[in] domain The domain to query.
//! @return The back element of `domain` restricted to `QueryDDims`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryDDims, concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryDDims...> back(
        SparseDiscreteDomain<DDims...> const& domain) noexcept
{
    return DiscreteElement<QueryDDims...>(SparseDiscreteDomain<QueryDDims>(domain).back()...);
}

} // namespace ddc
