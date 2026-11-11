// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ddc::DiscreteVector, a tagged vector of offsets in discrete dimensions.
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
#include "type_seq.hpp"

namespace ddc {

template <concepts::discrete_dimension...>
class DiscreteVector;

//! @brief Type trait that is `std::true_type` if `T` is a ddc::DiscreteVector.
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
struct is_discrete_vector : std::false_type
{
};

//! @brief Specialization of ddc::is_discrete_vector for ddc::DiscreteVector.
//! @tparam Tags The discrete dimensions of the vector.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags>
struct is_discrete_vector<DiscreteVector<Tags...>> : std::true_type
{
};

//! @brief Helper variable template for ddc::is_discrete_vector.
//! @tparam T The type to inspect.
//! @ingroup ddc_domains
template <class T>
inline constexpr bool is_discrete_vector_v = is_discrete_vector<T>::value;

namespace concepts {

//! @brief Satisfied by the types that are a ddc::DiscreteVector.
//! @tparam T The type to check.
//! @ingroup ddc_domains
template <class T>
concept discrete_vector = is_discrete_vector_v<T>;

} // namespace concepts

namespace detail {

template <ddc::concepts::discrete_dimension... Tags>
struct ToTypeSeq<DiscreteVector<Tags...>>
{
    using type = ddc::TypeSeq<Tags...>;
};

} // namespace detail

/** A DiscreteVectorElement is a scalar that represents the difference between two coordinates.
 *
 * @ingroup ddc_domains
 */
using DiscreteVectorElement = std::ptrdiff_t;

//! @brief Returns a const reference to the component of a DiscreteVector along `QueryTag`.
//! @tparam QueryTag The discrete dimension to query, it must be in the dimensions of `tuple`.
//! @param[in] tuple The discrete vector.
//! @return A const reference to the component of `tuple` along `QueryTag`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension QueryTag, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVectorElement const& get(
        DiscreteVector<Tags...> const& tuple) noexcept
{
    return tuple.template get<QueryTag>();
}

//! @brief Returns a mutable reference to the component of a DiscreteVector along `QueryTag`.
//! @tparam QueryTag The discrete dimension to query, it must be in the dimensions of `tuple`.
//! @param[in,out] tuple The discrete vector.
//! @return A reference to the component of `tuple` along `QueryTag`, usable to modify it.
//! @ingroup ddc_domains
template <concepts::discrete_dimension QueryTag, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVectorElement& get(DiscreteVector<Tags...>& tuple) noexcept
{
    return tuple.template get<QueryTag>();
}

//! @brief Returns the component along `QueryTag` if present, otherwise a default value.
//! @tparam QueryTag The discrete dimension to query.
//! @param[in] tuple The discrete vector.
//! @param[in] default_value The value returned when `QueryTag` is not a dimension of `tuple`.
//! @return A const reference to the component of `tuple` along `QueryTag` if it exists, otherwise
//! to `default_value`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension QueryTag, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVectorElement const& get_or(
        DiscreteVector<Tags...> const& tuple,
        DiscreteVectorElement const& default_value) noexcept
{
    return tuple.template get_or<QueryTag>(default_value);
}

//! @brief Extracts the component along `QueryTag` if present, otherwise returns a default value.
//! @param[in] arr The discrete vector to extract from.
//! @param[in] default_value The value returned when `QueryTag` is not a dimension of `arr`.
//! @return A one-dimensional DiscreteVector along `QueryTag`: the component of `arr` if `QueryTag`
//! is one of its dimensions, `default_value` otherwise.
//! @ingroup ddc_domains
template <concepts::discrete_dimension QueryTag, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryTag> select_or(
        DiscreteVector<Tags...> const& arr,
        DiscreteVector<QueryTag> const& default_value) noexcept
{
    if constexpr (in_tags_v<QueryTag, TypeSeq<Tags...>>) {
        return DiscreteVector<QueryTag>(arr);
    } else {
        return default_value;
    }
}

/// Unary operators: +, -
//! @param[in] x The vector.
//! @return A copy of `x`.
//! @ingroup ddc_domains

template <concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVector<Tags...> operator+(DiscreteVector<Tags...> const& x)
{
    return x;
}

//! @brief Unary minus: negates every component of a DiscreteVector.
//! @param[in] x The vector.
//! @return A vector with the same dimensions as `x` and opposite components.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVector<Tags...> operator-(DiscreteVector<Tags...> const& x)
{
    return DiscreteVector<Tags...>((-get<Tags>(x))...);
}

/// Internal binary operators: +, -
//! @details The result has the dimensions of the operand with the most dimensions, whose dimensions
//! must include all those of the other operand (checked by `static_assert`). The components along
//! the dimensions present in only one operand are copied from it.
//!
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return The component-wise sum of `lhs` and `rhs`.
//! @ingroup ddc_domains

template <concepts::discrete_dimension... Tags, concepts::discrete_dimension... OTags>
KOKKOS_FUNCTION constexpr auto operator+(
        DiscreteVector<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    if constexpr (sizeof...(Tags) >= sizeof...(OTags)) {
        static_assert((type_seq_contains_v<TypeSeq<OTags>, TypeSeq<Tags...>> && ...));
        DiscreteVector<Tags...> result(lhs);
        result += rhs;
        return result;
    } else {
        static_assert((type_seq_contains_v<TypeSeq<Tags>, TypeSeq<OTags...>> && ...));
        DiscreteVector<OTags...> result(rhs);
        result += lhs;
        return result;
    }
}

//! @brief Subtracts two DiscreteVectors component by component.
//!
//! @details The result has the dimensions of the operand with the most dimensions, whose
//! dimensions must include all those of the other operand (checked by `static_assert`). Along
//! the dimensions present in only one operand, the component is copied from `lhs` or taken from
//! the opposite of `rhs`.
//!
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return The component-wise difference `lhs - rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... Tags, concepts::discrete_dimension... OTags>
KOKKOS_FUNCTION constexpr auto operator-(
        DiscreteVector<Tags...> const& lhs,
        DiscreteVector<OTags...> const& rhs)
{
    if constexpr (sizeof...(Tags) >= sizeof...(OTags)) {
        static_assert((type_seq_contains_v<TypeSeq<OTags>, TypeSeq<Tags...>> && ...));
        DiscreteVector<Tags...> result(lhs);
        result -= rhs;
        return result;
    } else {
        static_assert((type_seq_contains_v<TypeSeq<Tags>, TypeSeq<OTags...>> && ...));
        DiscreteVector<OTags...> result(-rhs);
        result += lhs;
        return result;
    }
}

//! @brief Adds an integer to a one-dimensional DiscreteVector.
//! @param[in] lhs The vector.
//! @param[in] rhs The integer to add.
//! @return A vector whose component is that of `lhs` plus `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag, std::integral IntegralType>
KOKKOS_FUNCTION constexpr DiscreteVector<Tag> operator+(
        DiscreteVector<Tag> const& lhs,
        IntegralType const& rhs)
{
    return DiscreteVector<Tag>(get<Tag>(lhs) + rhs);
}

//! @brief Adds a one-dimensional DiscreteVector to an integer.
//! @param[in] lhs The integer.
//! @param[in] rhs The vector.
//! @return A vector whose component is `lhs` plus that of `rhs`.
//! @ingroup ddc_domains
template <std::integral IntegralType, concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr DiscreteVector<Tag> operator+(
        IntegralType const& lhs,
        DiscreteVector<Tag> const& rhs)
{
    return DiscreteVector<Tag>(lhs + get<Tag>(rhs));
}

//! @brief Subtracts an integer from a one-dimensional DiscreteVector.
//! @param[in] lhs The vector.
//! @param[in] rhs The integer to subtract.
//! @return A vector whose component is that of `lhs` minus `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag, std::integral IntegralType>
KOKKOS_FUNCTION constexpr DiscreteVector<Tag> operator-(
        DiscreteVector<Tag> const& lhs,
        IntegralType const& rhs)
{
    return DiscreteVector<Tag>(get<Tag>(lhs) - rhs);
}

//! @brief Subtracts a one-dimensional DiscreteVector from an integer.
//! @param[in] lhs The integer.
//! @param[in] rhs The vector.
//! @return A vector whose component is `lhs` minus that of `rhs`.
//! @ingroup ddc_domains
template <std::integral IntegralType, concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr DiscreteVector<Tag> operator-(
        IntegralType const& lhs,
        DiscreteVector<Tag> const& rhs)
{
    return DiscreteVector<Tag>(lhs - get<Tag>(rhs));
}

/// external left binary operator: *
//! @param[in] lhs The integer scaling factor.
//! @param[in] rhs The vector to scale.
//! @return A vector with the dimensions of `rhs`, each component multiplied by `lhs`.
//! @ingroup ddc_domains

template <std::integral IntegralType, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr auto operator*(
        IntegralType const& lhs,
        DiscreteVector<Tags...> const& rhs)
{
    return DiscreteVector<Tags...>((lhs * get<Tags>(rhs))...);
}

//! @brief Extracts a sub-vector along the dimensions `QueryTags`, by copy.
//! @tparam QueryTags The discrete dimensions to extract, they must all be dimensions of `arr`.
//! @param[in] arr The discrete vector to extract from.
//! @return A DiscreteVector along `QueryTags` containing the corresponding components of `arr`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryTags, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryTags...> select(
        DiscreteVector<Tags...> const& arr) noexcept
{
    return DiscreteVector<QueryTags...>(arr);
}

//! @brief Extracts a sub-vector along the dimensions `QueryTags` from a temporary vector.
//! @tparam QueryTags The discrete dimensions to extract, they must all be dimensions of `arr`.
//! @param[in] arr The discrete vector to extract from.
//! @return A DiscreteVector along `QueryTags` containing the corresponding components of `arr`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension... QueryTags, concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr DiscreteVector<QueryTags...> select(
        DiscreteVector<Tags...>&& arr) noexcept
{
    return DiscreteVector<QueryTags...>(std::move(arr));
}

/// Returns a reference towards the DiscreteVector that contains the QueryTag
//! @tparam QueryTag The discrete dimension to look for.
//! @param[in] head The first DiscreteVector to search in.
//! @param[in] tail The remaining DiscreteVectors to search in.
//! @return A const reference to the one of the arguments that has the dimension `QueryTag`.
//! @note A `static_assert` fails if `QueryTag` is absent from all arguments, or is present in more
//! than one of them.
//! @ingroup ddc_domains
template <
        concepts::discrete_dimension QueryTag,
        concepts::discrete_vector HeadDVect,
        concepts::discrete_vector... TailDVects>
KOKKOS_FUNCTION constexpr auto const& take(HeadDVect const& head, TailDVects const&... tail)
{
    DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
    if constexpr (type_seq_contains_v<TypeSeq<QueryTag>, to_type_seq_t<HeadDVect>>) {
        static_assert(
                (!type_seq_contains_v<TypeSeq<QueryTag>, to_type_seq_t<TailDVects>> && ...),
                "ERROR: tag redundant");
        return head;
    } else {
        static_assert(sizeof...(TailDVects) > 0, "ERROR: tag not found");
        return take<QueryTag>(tail...);
    }
    DDC_IF_NVCC_THEN_POP
}

namespace detail {

template <class T>
class DiscreteVectorConversionOperators
{
};

template <ddc::concepts::discrete_dimension Tag>
class DiscreteVectorConversionOperators<DiscreteVector<Tag>>
{
public:
    // NOLINTBEGIN(google-explicit-constructor)
    KOKKOS_FUNCTION constexpr operator DiscreteVectorElement const&() const noexcept
    {
        return static_cast<DiscreteVector<Tag> const*>(this)->m_values[0];
    }

    KOKKOS_FUNCTION constexpr operator DiscreteVectorElement&() noexcept
    {
        return static_cast<DiscreteVector<Tag>*>(this)->m_values[0];
    }
    // NOLINTEND(google-explicit-constructor)
};

/// Returns a reference to the underlying `std::array`
template <ddc::concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr std::array<DiscreteVectorElement, sizeof...(Tags)>& array(
        DiscreteVector<Tags...>& v) noexcept
{
    return v.m_values;
}

/// Returns a reference to the underlying `std::array`
template <ddc::concepts::discrete_dimension... Tags>
KOKKOS_FUNCTION constexpr std::array<DiscreteVectorElement, sizeof...(Tags)> const& array(
        DiscreteVector<Tags...> const& v) noexcept
{
    return v.m_values;
}

} // namespace detail

/** A DiscreteVector is a vector in the discrete dimension
 *
 * Each is tagged by its associated dimensions.
 *
 * A one-dimensional DiscreteVector is implicitly convertible to a (const) reference to its
 * DiscreteVectorElement.
 *
 * @tparam Tags The discrete dimensions of the vector, they must be unique.
 * @ingroup ddc_domains
 */
template <concepts::discrete_dimension... Tags>
class DiscreteVector : public detail::DiscreteVectorConversionOperators<DiscreteVector<Tags...>>
{
    friend class detail::DiscreteVectorConversionOperators<DiscreteVector<Tags...>>;

    friend KOKKOS_FUNCTION constexpr std::array<DiscreteVectorElement, sizeof...(Tags)>& detail::
            array<Tags...>(DiscreteVector<Tags...>& v) noexcept;
    friend KOKKOS_FUNCTION constexpr std::array<DiscreteVectorElement, sizeof...(Tags)> const&
    detail::array<Tags...>(DiscreteVector<Tags...> const& v) noexcept;

    using tags_seq = TypeSeq<Tags...>;

    static_assert(
            type_seq_is_unique_v<tags_seq>,
            "The dimensions of a DiscreteVector must be unique");

private:
    std::array<DiscreteVectorElement, sizeof...(Tags)> m_values;

public:
    //! @brief Type of the scalar component stored for each dimension.
    using value_type = DiscreteVectorElement;

    //! @brief Returns the number of dimensions of the vector.
    //! @return The number of tags `sizeof...(Tags)`.
    static KOKKOS_FUNCTION constexpr std::size_t size() noexcept
    {
        return sizeof...(Tags);
    }

public:
    //! @brief Default constructor.
    //! @note The components are not initialized by default-initialization.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteVector() = default;

    //! @brief Copy constructor.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteVector(DiscreteVector const&) = default;

    //! @brief Move constructor.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteVector(DiscreteVector&&) = default;

    //! @brief Constructs a DiscreteVector by merging other DiscreteVectors.
    //!
    //! @details Each dimension of `Tags` is taken from the argument that has it. The arguments may
    //! have further dimensions, which are ignored.
    //!
    //! @param[in] delems The DiscreteVectors providing the components. Each dimension of `Tags`
    //! must appear in exactly one of them (checked by `static_assert`).
    template <concepts::discrete_vector... DVects>
    KOKKOS_FUNCTION constexpr explicit DiscreteVector(DVects const&... delems) noexcept
        : m_values {take<Tags>(delems...).template get<Tags>()...}
    {
    }

    //! @brief Constructs a DiscreteVector from an array of components.
    //! @param[in] values The components, in the order of `Tags`.
    template <std::convertible_to<DiscreteVectorElement> IntegerType>
    KOKKOS_FUNCTION constexpr explicit DiscreteVector(
            std::array<IntegerType, sizeof...(Tags)> const& values) noexcept
        : m_values(
                  detail::convert_array_to<DiscreteVectorElement>(
                          values,
                          std::make_index_sequence<sizeof...(Tags)>()))
    {
    }

    //! @brief Constructs a DiscreteVector from one component per dimension.
    //! @param[in] params The components, in the order of `Tags`. There must be one per tag.
    template <std::convertible_to<DiscreteVectorElement>... Params>
    KOKKOS_FUNCTION constexpr explicit DiscreteVector(Params const&... params) noexcept
        requires((!is_discrete_vector_v<Params> && ...) && sizeof...(Params) == sizeof...(Tags))
        : m_values {static_cast<DiscreteVectorElement>(params)...}
    {
    }

    //! @brief Destructor.
    KOKKOS_DEFAULTED_FUNCTION ~DiscreteVector() = default;

    //! @brief Copy assignment operator.
    //! @param[in] other The vector to copy.
    //! @return A reference to this vector.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteVector& operator=(DiscreteVector const& other)
            = default;

    //! @brief Move assignment operator.
    //! @param[in] other The vector to move from.
    //! @return A reference to this vector.
    KOKKOS_DEFAULTED_FUNCTION constexpr DiscreteVector& operator=(DiscreteVector&& other) = default;

    //! @brief Compares this vector with another one for equality.
    //! @param[in] rhs The vector to compare with. It must have all the dimensions of this vector.
    //! @return `true` if all components of this vector equal the corresponding ones of `rhs`.
    template <concepts::discrete_dimension... OTags>
    KOKKOS_FUNCTION constexpr bool operator==(DiscreteVector<OTags...> const& rhs) const noexcept
    {
        return ((m_values[type_seq_rank_v<Tags, tags_seq>] == rhs.template get<Tags>()) && ...);
    }

#if !defined(__cpp_impl_three_way_comparison) || __cpp_impl_three_way_comparison < 201902L
    // In C++20, `a!=b` shall be automatically translated by the compiler to `!(a==b)`
    //! @brief Compares this vector with another one for inequality.
    //! @param[in] rhs The vector to compare with.
    //! @return `!(*this == rhs)`.
    template <concepts::discrete_dimension... OTags>
    KOKKOS_FUNCTION constexpr bool operator!=(DiscreteVector<OTags...> const& rhs) const noexcept
    {
        return !(*this == rhs);
    }
#endif

    //! @brief Returns a mutable reference to the component along the dimension `QueryTag`.
    //! @tparam QueryTag The discrete dimension to query, it must be in `Tags` (`static_assert`).
    //! @return A reference to the component along `QueryTag`.
    template <concepts::discrete_dimension QueryTag>
    KOKKOS_FUNCTION constexpr DiscreteVectorElement& get() noexcept
    {
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteVector");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    //! @brief Returns a const reference to the component along the dimension `QueryTag`.
    //! @tparam QueryTag The discrete dimension to query, it must be in `Tags` (`static_assert`).
    //! @return A const reference to the component along `QueryTag`.
    template <concepts::discrete_dimension QueryTag>
    KOKKOS_FUNCTION constexpr DiscreteVectorElement const& get() const noexcept
    {
        static_assert(in_tags_v<QueryTag, tags_seq>, "requested Tag absent from DiscreteVector");
        return m_values[type_seq_rank_v<QueryTag, tags_seq>];
    }

    //! @brief Returns the component along `QueryTag` if present, otherwise a default value.
    //! @tparam QueryTag The discrete dimension to query.
    //! @param[in] default_value The value returned when `QueryTag` is not in `Tags`.
    //! @return A const reference to the component along `QueryTag` if it exists, otherwise to
    //! `default_value`.
    //! @warning The returned reference may refer to `default_value`: it must not outlive it.
    template <concepts::discrete_dimension QueryTag>
    KOKKOS_FUNCTION constexpr DiscreteVectorElement const& get_or(
            DiscreteVectorElement const& default_value) const&
    {
        DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
        if constexpr (in_tags_v<QueryTag, tags_seq>) {
            return m_values[type_seq_rank_v<QueryTag, tags_seq>];
        } else {
            return default_value;
        }
        DDC_IF_NVCC_THEN_POP
    }

    //! @brief Returns the component of a one-dimensional vector.
    //! @return A const reference to the only component.
    KOKKOS_FUNCTION constexpr DiscreteVectorElement const& value() const noexcept
        requires(sizeof...(Tags) == 1)
    {
        return m_values[0];
    }

    //! @brief Pre-increment: adds 1 to the component of a one-dimensional vector.
    //! @return A reference to this vector, after the increment.
    KOKKOS_FUNCTION constexpr DiscreteVector& operator++()
        requires(sizeof...(Tags) == 1)
    {
        ++m_values[0];
        return *this;
    }

    //! @brief Post-increment: adds 1 to the component of a one-dimensional vector.
    //! @return A copy of this vector, before the increment.
    KOKKOS_FUNCTION constexpr DiscreteVector operator++(int)
        requires(sizeof...(Tags) == 1)
    {
        DiscreteVector const tmp = *this;
        ++m_values[0];
        return tmp;
    }

    //! @brief Pre-decrement: subtracts 1 from the component of a one-dimensional vector.
    //! @return A reference to this vector, after the decrement.
    KOKKOS_FUNCTION constexpr DiscreteVector& operator--()
        requires(sizeof...(Tags) == 1)
    {
        --m_values[0];
        return *this;
    }

    //! @brief Post-decrement: subtracts 1 from the component of a one-dimensional vector.
    //! @return A copy of this vector, before the decrement.
    KOKKOS_FUNCTION constexpr DiscreteVector operator--(int)
        requires(sizeof...(Tags) == 1)
    {
        DiscreteVector const tmp = *this;
        --m_values[0];
        return tmp;
    }

    //! @brief Adds another DiscreteVector to this vector, component by component.
    //! @param[in] rhs The vector to add. Its dimensions must be among those of this vector
    //! (`static_assert`); the other components are unchanged.
    //! @return A reference to this vector.
    template <concepts::discrete_dimension... OTags>
    KOKKOS_FUNCTION constexpr DiscreteVector& operator+=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert((type_seq_contains_v<TypeSeq<OTags>, tags_seq> && ...));
        ((m_values[type_seq_rank_v<OTags, tags_seq>] += rhs.template get<OTags>()), ...);
        return *this;
    }

    //! @brief Adds an integer to the component of a one-dimensional vector.
    //! @param[in] rhs The integer to add.
    //! @return A reference to this vector.
    template <std::integral IntegralType>
    KOKKOS_FUNCTION constexpr DiscreteVector& operator+=(IntegralType const& rhs)
        requires(sizeof...(Tags) == 1)
    {
        m_values[0] += rhs;
        return *this;
    }

    //! @brief Subtracts another DiscreteVector from this vector, component by component.
    //! @param[in] rhs The vector to subtract. Its dimensions must be among those of this vector
    //! (`static_assert`); the other components are unchanged.
    //! @return A reference to this vector.
    template <concepts::discrete_dimension... OTags>
    KOKKOS_FUNCTION constexpr DiscreteVector& operator-=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert((type_seq_contains_v<TypeSeq<OTags>, tags_seq> && ...));
        ((m_values[type_seq_rank_v<OTags, tags_seq>] -= rhs.template get<OTags>()), ...);
        return *this;
    }

    //! @brief Subtracts an integer from the component of a one-dimensional vector.
    //! @param[in] rhs The integer to subtract.
    //! @return A reference to this vector.
    template <std::integral IntegralType>
    KOKKOS_FUNCTION constexpr DiscreteVector& operator-=(IntegralType const& rhs)
        requires(sizeof...(Tags) == 1)
    {
        m_values[0] -= rhs;
        return *this;
    }

    //! @brief Multiplies this vector by another one, component by component.
    //! @param[in] rhs The vector of factors. Its dimensions must be among those of this vector
    //! (`static_assert`); the other components are unchanged.
    //! @return A reference to this vector.
    template <concepts::discrete_dimension... OTags>
    KOKKOS_FUNCTION constexpr DiscreteVector& operator*=(DiscreteVector<OTags...> const& rhs)
    {
        static_assert((type_seq_contains_v<TypeSeq<OTags>, tags_seq> && ...));
        ((m_values[type_seq_rank_v<OTags, tags_seq>] *= rhs.template get<OTags>()), ...);
        return *this;
    }
};

//! @brief Compares the components of two one-dimensional DiscreteVectors.
//! @param[in] lhs The left operand.
//! @param[in] rhs The right operand.
//! @return `true` if the component of `lhs` is strictly lower than that of `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag>
KOKKOS_FUNCTION constexpr bool operator<(
        DiscreteVector<Tag> const& lhs,
        DiscreteVector<Tag> const& rhs)
{
    return lhs.value() < rhs.value();
}

//! @brief Compares the component of a one-dimensional DiscreteVector with a scalar.
//! @param[in] lhs The vector.
//! @param[in] rhs The scalar to compare with.
//! @return `true` if the component of `lhs` is strictly lower than `rhs`.
//! @ingroup ddc_domains
template <concepts::discrete_dimension Tag, class IntegralType>
KOKKOS_FUNCTION constexpr bool operator<(DiscreteVector<Tag> const& lhs, IntegralType const& rhs)
{
    return lhs.value() < rhs;
}

namespace detail {

void print_discrete_vector(std::ostream& os, std::span<DiscreteVectorElement const> view);

} // namespace detail

//! @brief Prints a DiscreteVector to a stream, as a parenthesized, comma-separated list.
//! @param[in,out] os The output stream.
//! @param[in] arr The vector to print, e.g. `(1, 2)`.
//! @return The stream `os`.
//! @ingroup ddc_output
template <concepts::discrete_dimension... Tags>
std::ostream& operator<<(std::ostream& os, DiscreteVector<Tags...> const& arr)
{
    detail::print_discrete_vector(os, detail::array(arr));
    return os;
}

} // namespace ddc
