// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ddc::DiscreteElement, a tagged index of an element of discrete dimensions.
#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <iosfwd>
#include <span>
#include <type_traits>
#include <utility>

#include <Kokkos_Macros.hpp>

#include "detail/macros.hpp"
#include "detail/utils.hpp"

#include "discrete_dimension.hpp"
#include "discrete_vector.hpp"
#include "type_seq.hpp"

namespace ddc {

template <concepts::discrete_dimension...>
class DiscreteElement;

//! @brief Type trait that is `std::true_type` if `T` is a ddc::DiscreteElement.
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
struct is_discrete_element : std::false_type
{
};

//! @brief Specialization of ddc::is_discrete_element for ddc::DiscreteElement.
//! @tparam Tags The discrete dimensions of the element.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags>
struct is_discrete_element<DiscreteElement<Tags...>> : std::true_type
{
};

//! @brief Helper variable template for ddc::is_discrete_element.
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
inline constexpr bool is_discrete_element_v = is_discrete_element<T>::value;

namespace concepts {

//! @brief Satisfied by the types that are a ddc::DiscreteElement.
//! @tparam T The type to check.
//! @ingroup ddc_domains
template <class T>
concept discrete_element = is_discrete_element_v<T>;

}

namespace detail {

template <ddc::concepts::discrete_dimension... Tags>
struct ToTypeSeq<DiscreteElement<Tags...>>
{
    using type = ddc::TypeSeq<Tags...>;
};

} // namespace detail

/** A DiscreteCoordElement is a scalar that identifies an element of the discrete dimension
 *
 * @ingroup ddc_domains
 */
using DiscreteElementType = std::size_t;

//! @brief Returns a const reference to the identifier of a one-dimensional DiscreteElement.
//! @param[in] tuple The discrete element.
//! @return A reference to the underlying scalar of `tuple`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr DiscreteElementType const& uid(DiscreteElement<Tag> const& tuple) noexcept
{
    return tuple.uid();
}

//! @brief Returns a mutable reference to the identifier of a one-dimensional DiscreteElement.
//! @param[in,out] tuple The discrete element.
//! @return A reference to the underlying scalar of `tuple`, which can be used to modify it.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr DiscreteElementType& uid(DiscreteElement<Tag>& tuple) noexcept
{
    return tuple.uid();
}

//! @brief Returns a const reference to the identifier along the discrete dimension `QueryTag`.
//! @tparam QueryTag The discrete dimension to query, it must be one of the dimensions of `tuple`.
//! @param[in] tuple The discrete element.
//! @return A reference to the component of `tuple` along `QueryTag`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension QueryTag, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteElementType const& uid(
        DiscreteElement<Tags...> const& tuple) noexcept
{
    return tuple.template uid<QueryTag>();
}

//! @brief Returns a mutable reference to the identifier along the discrete dimension `QueryTag`.
//! @tparam QueryTag The discrete dimension to query, it must be one of the dimensions of `tuple`.
//! @param[in,out] tuple The discrete element.
//! @return A reference to the component of `tuple` along `QueryTag`, usable to modify it.
//! @ingroup ddc_domains
template <concepts::discrete_dimension QueryTag, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteElementType& uid(DiscreteElement<Tags...>& tuple) noexcept
{
    return tuple.template uid<QueryTag>();
}

//! @brief Extracts the component along `QueryTag` if present, otherwise returns a default value.
//! @param[in] arr The discrete element to extract from.
//! @param[in] default_value The value returned when `QueryTag` is not a dimension of `arr`.
//! @return A one-dimensional DiscreteElement along `QueryTag`: the component of `arr` if `QueryTag`
//! is one of its dimensions, `default_value` otherwise.
//! @ingroup ddc_domains
template <concepts::discrete_dimension QueryTag, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryTag> select_or(
        DiscreteElement<Tags...> const& arr,
        DiscreteElement<QueryTag> const& default_value) noexcept
{
    if constexpr (in_tags_v<QueryTag, TypeSeq<Tags...>>) {
        return DiscreteElement<QueryTag>(arr);
    } else {
        return default_value;
    }
}

//! @brief Extracts a sub-element along the dimensions `QueryTags`, by copy.
//! @tparam QueryTags The discrete dimensions to extract, they must all be dimensions of `arr`.
//! @param[in] arr The discrete element to extract from.
//! @return A DiscreteElement along `QueryTags` containing the corresponding components of `arr`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryTags, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryTags...> select(
        DiscreteElement<Tags...> const& arr) noexcept
{
    return DiscreteElement<QueryTags...>(arr);
}

//! @brief Extracts a sub-element along the dimensions `QueryTags` from a temporary element.
//! @tparam QueryTags The discrete dimensions to extract, they must all be dimensions of `arr`.
//! @param[in] arr The discrete element to extract from.
//! @return A DiscreteElement along `QueryTags` containing the corresponding components of `arr`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryTags, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteElement<QueryTags...> select(
        DiscreteElement<Tags...>&& arr) noexcept
{
    return DiscreteElement<QueryTags...>(std::move(arr));
}

/// Returns a reference towards the DiscreteElement that contains the QueryTag
//! @tparam QueryTag The discrete dimension to look for.
//! @param[in] head The first DiscreteElement to search in.
//! @param[in] tail The remaining DiscreteElements to search in.
//! @return A const reference to the one of the arguments that has the dimension `QueryTag`.
//! @note A `static_assert` fails if `QueryTag` is absent from all arguments, or is present in more
//! than one of them.
//! @ingroup ddc_domains
template <
        concepts::discrete_dimension QueryTag,
        concepts::discrete_element HeadDElem,
        concepts::discrete_element... TailDElems>
KOKKOS_FUNCTION constexpr auto const& take(HeadDElem const& head, TailDElems const&... tail)
{
    DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
    if constexpr (type_seq_contains_v<TypeSeq<QueryTag>, to_type_seq_t<HeadDElem>>) {
        static_assert(
                (!type_seq_contains_v<TypeSeq<QueryTag>, to_type_seq_t<TailDElems>> && ...),
                "ERROR: tag redundant");
        return head;
    } else {
        static_assert(sizeof...(TailDElems) > 0, "ERROR: tag not found");
        return take<QueryTag>(tail...);
    }
    DDC_IF_NVCC_THEN_POP
}

namespace detail {

/// Returns a reference to the underlying `std::array`
template <ddc::concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)>& array(
        DiscreteElement<Tags...>& v) noexcept
{
    return v.m_values;
}

/// Returns a reference to the underlying `std::array`
template <ddc::concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)> const& array(
        DiscreteElement<Tags...> const& v) noexcept
{
    return v.m_values;
}

} // namespace detail

//! @brief Creates the reference element of a discrete dimension, whose identifier is 0.
//! @tparam DDim The discrete dimension.
//! @return A DiscreteElement along `DDim` with identifier 0.
//! @ingroup ddc_domains
template <concepts::discrete_dimension DDim>
constexpr DiscreteElement<DDim> create_reference_discrete_element() noexcept
{
    return DiscreteElement<DDim>(0);
}

/** A DiscreteElement identifies an element of the discrete dimension
 *
 * Each one is tagged by its associated dimensions.
 *
 * @tparam Tags The discrete dimensions of the element, they must be unique.
 * @ingroup ddc_domains
 */
template <concepts::discrete_dimension... Tags>
class DiscreteElement
{
    using tags_seq = TypeSeq<Tags...>;

    static_assert(
            type_seq_is_unique_v<tags_seq>,
            "The dimensions of a DiscreteElement must be unique");

    friend KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)>& detail::
            array<Tags...>(DiscreteElement<Tags...>& v) noexcept;

    friend KOKKOS_FUNCTION constexpr std::array<DiscreteElementType, sizeof...(Tags)> const&
    detail::array<Tags...>(DiscreteElement<Tags...> const& v) noexcept;

private:
    std::array<DiscreteElementType, sizeof...(Tags)> m_values;

public:
    //! @brief Type of the scalar identifier stored for each dimension.
    using value_type = DiscreteElementType;

    //! @brief Returns the number of dimensions of the element.
    //! @return The number of tags `sizeof...(Tags)`.
    static KOKKOS_FUNCTION constexpr std::size_t size() noexcept
    {
        return sizeof...(Tags);
    }

public:
    //! @brief Default constructor.
    //! @note The components are not initialized by default-initialization.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteElement() = default;

    //! @brief Copy constructor.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteElement(DiscreteElement const&) = default;

    //! @brief Move constructor.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteElement(DiscreteElement&&) = default;

    //! @brief Constructs a DiscreteElement by merging other DiscreteElements.
    //!
    //! @details Each dimension of `Tags` is taken from the argument that has it. The arguments may
    //! have further dimensions, which are ignored.
    //!
    //! @param[in] delems The DiscreteElements providing the components. Each dimension of `Tags`
    //! must appear in exactly one of them (checked by `static_assert`).
    template <concepts::discrete_element... DElems>
    KOKKOS_FUNCTION constexpr explicit DiscreteElement(DElems const&... delems) noexcept
        : m_values {take<Tags>(delems...).template uid<Tags>()...}
    {
    }

    //! @brief Constructs a DiscreteElement from an array of identifiers.
    //! @param[in] values The identifiers, in the order of `Tags`.
    template <std::convertible_to<DiscreteElementType> IntegerType>
    KOKKOS_FUNCTION constexpr explicit DiscreteElement(
            std::array<IntegerType, sizeof...(Tags)> const& values) noexcept
        : m_values(
                  detail::convert_array_to<
                          DiscreteElementType>(values, std::make_index_sequence<sizeof...(Tags)>()))
    {
    }

    //! @brief Constructs a DiscreteElement from one identifier per dimension.
    //! @param[in] params The identifiers, in the order of `Tags`. There must be one per tag.
    template <std::convertible_to<DiscreteElementType>... Params>
    KOKKOS_FUNCTION constexpr explicit DiscreteElement(Params const&... params) noexcept
        requires((!is_discrete_element_v<Params> && ...) && sizeof...(Params) == sizeof...(Tags))
        : m_values {static_cast<DiscreteElementType>(params)...}
    {
    }

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~DiscreteElement() = default;

    //! @brief Copy assignment operator.
    //! @param[in] other The element to copy.
    //! @return A reference to this element.
    KOKKOS_DEFAULTED_FUNCTION DiscreteElement& operator=(DiscreteElement const& other) = default;

    //! @brief Move assignment operator.
    //! @param[in] other The element to move from.
    //! @return A reference to this element.
    KOKKOS_DEFAULTED_FUNCTION DiscreteElement& operator=(DiscreteElement&& other) = default;

    //! @brief Returns a mutable reference to the identifier along the dimension `QueryTag`.
    //! @tparam QueryTag The discrete dimension to query, it must be in `Tags` (`static_assert`).
    //! @return A reference to the component along `QueryTag`.
    template <concepts::discrete_dimension QueryTag>
    KOKKOS_FUNCTION constexpr value_type& uid() noexcept
    {
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    //! @brief Returns a const reference to the identifier along the dimension `QueryTag`.
    //! @tparam QueryTag The discrete dimension to query, it must be in `Tags` (`static_assert`).
    //! @return A const reference to the component along `QueryTag`.
    template <concepts::discrete_dimension QueryTag>
    KOKKOS_FUNCTION constexpr value_type const& uid() const noexcept
    {
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteElement");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    //! @brief Returns a mutable reference to the identifier of a one-dimensional element.
    //! @return A reference to the only component.
    KOKKOS_FUNCTION constexpr value_type& uid() noexcept
        requires(sizeof...(Tags) == 1)
    {
        return m_values[0];
    }

    //! @brief Returns a const reference to the identifier of a one-dimensional element.
    //! @return A const reference to the only component.
    KOKKOS_FUNCTION constexpr value_type const& uid() const noexcept
        requires(sizeof...(Tags) == 1)
    {
        return m_values[0];
    }

    //! @brief Pre-increment: adds 1 to the identifier of a one-dimensional element.
    //! @return A reference to this element, after the increment.
    KOKKOS_FUNCTION constexpr DiscreteElement& operator++()
        requires(sizeof...(Tags) == 1)
    {
        ++m_values[0];
        return *this;
    }

    //! @brief Post-increment: adds 1 to the identifier of a one-dimensional element.
    //! @return A copy of this element, before the increment.
    KOKKOS_FUNCTION constexpr DiscreteElement operator++(int)
        requires(sizeof...(Tags) == 1)
    {
        DiscreteElement const tmp = *this;
        ++m_values[0];
        return tmp;
    }

    //! @brief Pre-decrement: subtracts 1 from the identifier of a one-dimensional element.
    //! @return A reference to this element, after the decrement.
    KOKKOS_FUNCTION constexpr DiscreteElement& operator--()
        requires(sizeof...(Tags) == 1)
    {
        --m_values[0];
        return *this;
    }

    //! @brief Post-decrement: subtracts 1 from the identifier of a one-dimensional element.
    //! @return A copy of this element, before the decrement.
    KOKKOS_FUNCTION constexpr DiscreteElement operator--(int)
        requires(sizeof...(Tags) == 1)
    {
        DiscreteElement const tmp = *this;
        --m_values[0];
        return tmp;
    }

    //! @brief Adds a DiscreteVector to this element, component by component.
    //! @param[in] rhs The vector to add. Its dimensions must be among those of this element
    //! (`static_assert`); the other components are unchanged.
    //! @return A reference to this element.
    template <concepts::discrete_dimension... OTags>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator+=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert((type_seq_contains_v<TypeSeq<OTags>, tags_seq> && ...));
        ((m_values[type_seq_rank_v<OTags, tags_seq>] += rhs.template get<OTags>()), ...);
        return *this;
    }

    //! @brief Adds an integer to the identifier of a one-dimensional element.
    //! @param[in] rhs The integer to add.
    //! @return A reference to this element.
    template <std::integral IntegralType>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator+=(IntegralType const& rhs)
        requires(sizeof...(Tags) == 1)
    {
        m_values[0] += rhs;
        return *this;
    }

    //! @brief Subtracts a DiscreteVector from this element, component by component.
    //! @param[in] rhs The vector to subtract. Its dimensions must be among those of this element
    //! (`static_assert`); the other components are unchanged.
    //! @return A reference to this element.
    template <concepts::discrete_dimension... OTags>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator-=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert((type_seq_contains_v<TypeSeq<OTags>, tags_seq> && ...));
        ((m_values[type_seq_rank_v<OTags, tags_seq>] -= rhs.template get<OTags>()), ...);
        return *this;
    }

    //! @brief Subtracts an integer from the identifier of a one-dimensional element.
    //! @param[in] rhs The integer to subtract.
    //! @return A reference to this element.
    template <std::integral IntegralType>
    KOKKOS_FUNCTION constexpr DiscreteElement& operator-=(IntegralType const& rhs)
        requires(sizeof...(Tags) == 1)
    {
        m_values[0] -= rhs;
        return *this;
    }
};

namespace detail {

void print_discrete_element(std::ostream& os, std::span<DiscreteElementType const> view);

} // namespace detail

//! @brief Prints a DiscreteElement to a stream, as a parenthesized, comma-separated list.
//! @param[in,out] os The output stream.
//! @param[in] arr The element to print, e.g. `(1, 2)`.
//! @return The stream `os`.
//! @ingroup ddc_output
template <concepts::discrete_dimension... Tags>
std::ostream& operator<<(std::ostream& os, DiscreteElement<Tags...> const& arr)
{
    detail::print_discrete_element(os, detail::array(arr));
    return os;
}


//! @brief Compares two DiscreteElements for equality.
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand. It must have all the dimensions of `lhs`.
//! @return `true` if all components of `lhs` equal the corresponding components of `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags, concepts::discrete_dimension... OTags>
KOKKOS_FUNCTION constexpr bool operator==(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs) noexcept
{
    return ((lhs.template uid<Tags>() == rhs.template uid<Tags>()) && ...);
}

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
// In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
//! @brief Compares two DiscreteElements for inequality.
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return `!(lhs == rhs)`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags, concepts::discrete_dimension... OTags>
KOKKOS_FUNCTION constexpr bool operator!=(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs) noexcept
{
    return !(lhs == rhs);
}
#endif

//! @brief Compares the identifiers of two one-dimensional DiscreteElements.
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return `true` if the identifier of `lhs` is strictly lower than that of `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr bool operator<(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() < rhs.uid();
}

//! @brief Compares the identifiers of two one-dimensional DiscreteElements.
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return `true` if the identifier of `lhs` is lower than or equal to that of `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr bool operator<=(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() <= rhs.uid();
}

//! @brief Compares the identifiers of two one-dimensional DiscreteElements.
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return `true` if the identifier of `lhs` is strictly greater than that of `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr bool operator>(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() > rhs.uid();
}

//! @brief Compares the identifiers of two one-dimensional DiscreteElements.
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return `true` if the identifier of `lhs` is greater than or equal to that of `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr bool operator>=(
        DiscreteElement<Tag> const& lhs,
        DiscreteElement<Tag> const& rhs)
{
    return lhs.uid() >= rhs.uid();
}

/// right external binary operators: +, -
//! @param[in] lhs The element.
//! @param[in] rhs The vector to add. Its dimensions must be among those of `lhs` (`static_assert`).
//! @return A copy of `lhs` shifted by `rhs`.
//! @ingroup ddc_domains

template <concepts::discrete_dimension... Tags, concepts::discrete_dimension... OTags>
KOKKOS_FUNCTION constexpr DiscreteElement<Tags...> operator+(
        DiscreteElement<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    static_assert((type_seq_contains_v<TypeSeq<OTags>, TypeSeq<Tags...>> && ...));
    DiscreteElement<Tags...> result(lhs);
    result += rhs;
    return result;
}

//! @brief Adds an integer to the identifier of a one-dimensional element.
//! @param[in] lhs The element.
//! @param[in] rhs The integer to add.
//! @return A copy of `lhs` whose identifier is increased by `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag, std::integral IntegralType>
KOKKOS_FUNCTION constexpr DiscreteElement<Tag> operator+(
        DiscreteElement<Tag> const& lhs,
        IntegralType const& rhs)
    requires(!is_discrete_vector_v<IntegralType>)
{
    DiscreteElement<Tag> result(lhs);
    result += rhs;
    return result;
}

//! @brief Subtracts a DiscreteVector from a DiscreteElement.
//! @param[in] lhs The element.
//! @param[in] rhs The vector to subtract. Its dimensions must be among those of `lhs`
//! (`static_assert`).
//! @return A copy of `lhs` shifted by `-rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags, concepts::discrete_dimension... OTags>
KOKKOS_FUNCTION constexpr DiscreteElement<Tags...> operator-(
        DiscreteElement<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    static_assert((type_seq_contains_v<TypeSeq<OTags>, TypeSeq<Tags...>> && ...));
    DiscreteElement<Tags...> result(lhs);
    result -= rhs;
    return result;
}

//! @brief Subtracts an integer from the identifier of a one-dimensional element.
//! @param[in] lhs The element.
//! @param[in] rhs The integer to subtract.
//! @return A copy of `lhs` whose identifier is decreased by `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag, std::integral IntegralType>
KOKKOS_FUNCTION constexpr DiscreteElement<Tag> operator-(
        DiscreteElement<Tag> const& lhs,
        IntegralType const& rhs)
    requires(!is_discrete_vector_v<IntegralType>)
{
    DiscreteElement<Tag> result(lhs);
    result -= rhs;
    return result;
}

/// binary operator: -
//! @param[in] lhs The element to subtract from.
//! @param[in] rhs The element to subtract. It must have exactly the same dimensions as `lhs`,
//! in any order (`static_assert`).
//! @return The DiscreteVector whose components are the differences of the identifiers.
//! @ingroup ddc_domains

template <concepts::discrete_dimension... Tags, concepts::discrete_dimension... OTags>
KOKKOS_FUNCTION constexpr DiscreteVector<Tags...> operator-(
        DiscreteElement<Tags...> const& lhs,
        DiscreteElement<OTags...> const& rhs)
{
    static_assert(type_seq_same_v<TypeSeq<Tags...>, TypeSeq<OTags...>>);
    return DiscreteVector<Tags...>((uid<Tags>(lhs) - uid<Tags>(rhs))...);
}

} // namespace ddc
