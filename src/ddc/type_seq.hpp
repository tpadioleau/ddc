// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ddc::TypeSeq, a compile-time sequence of types, and metafunctions on it.

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddc/config.hpp>

namespace ddc {

/// @brief A compile-time sequence of types.
///
/// `TypeSeq` is used to represent an ordered collection of types at compile time.
/// It is primarily intended for manipulating lists of tags or types in generic
/// programming code.
///
/// The order of the types is significant for operations such as
/// @ref type_seq_rank_v, @ref type_seq_element_t, and the sequence manipulation
/// utilities.
///
/// @tparam Tags The types contained in the sequence.
///
/// @ingroup ddc_core_utils
template <class... Tags>
struct TypeSeq
{
};

namespace detail {

template <class T>
inline constexpr bool is_type_seq_v = false;

template <class... Tags>
inline constexpr bool is_type_seq_v<TypeSeq<Tags...>> = true;

} // namespace detail

namespace concepts {

//! @brief Concept satisfied by specializations of ddc::TypeSeq.
//!
//! @tparam T The type to check.
//!
//! @ingroup ddc_core_utils
template <class T>
concept type_seq = detail::is_type_seq_v<T>;

} // namespace concepts

namespace detail {

#if DDC_BUILD_DEPRECATED_CODE()
template <class... Tags>
using TypeSeq [[deprecated("Use `ddc::TypeSeq' instead")]] = ::ddc::TypeSeq<Tags...>;
#endif

template <class Tag>
struct SingleType
{
};

template <class QueryTagSeq, class TagSeq>
struct TypeSeqRank
{
};

template <class QueryTag>
struct TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<>>
{
    static constexpr bool present = false;
    static constexpr std::size_t val = std::numeric_limits<std::size_t>::max();
};

template <class QueryTag, class... TagsTail>
struct TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<QueryTag, TagsTail...>>
{
    static constexpr bool present = true;
    static constexpr std::size_t val = 0;
};

template <class QueryTag, class TagsHead, class... TagsTail>
struct TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<TagsHead, TagsTail...>>
{
    static constexpr bool present
            = TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<TagsTail...>>::present;
    static constexpr std::size_t val
            = present ? 1 + TypeSeqRank<SingleType<QueryTag>, ddc::TypeSeq<TagsTail...>>::val
                      : std::numeric_limits<std::size_t>::max();
};

template <class... QueryTags, class... Tags>
struct TypeSeqRank<ddc::TypeSeq<QueryTags...>, ddc::TypeSeq<Tags...>>
{
    using ValSeq = std::index_sequence<TypeSeqRank<QueryTags, ddc::TypeSeq<Tags...>>::val...>;
};

template <std::size_t I, class TagSeq>
struct TypeSeqElement
{
};

template <std::size_t I, class... Tags>
struct TypeSeqElement<I, ddc::TypeSeq<Tags...>>
{
    using type = std::tuple_element_t<I, std::tuple<Tags...>>;
};

/// R contains all elements in A that are not in B.
/// Remark 1: This operation preserves the order from A.
/// Remark 2: It is similar to the set difference in the set theory (R = A\\B).
/// Example: A = [a, b, c], B = [z, c, y], R = [a, b]
template <class TagSeqA, class TagSeqB, class TagSeqR>
struct TypeSeqRemove
{
};

template <class... TagsB, class... TagsR>
struct TypeSeqRemove<ddc::TypeSeq<>, ddc::TypeSeq<TagsB...>, ddc::TypeSeq<TagsR...>>
{
    using type = ddc::TypeSeq<TagsR...>;
};

template <class HeadTagsA, class... TailTagsA, class... TagsB, class... TagsR>
struct TypeSeqRemove<
        ddc::TypeSeq<HeadTagsA, TailTagsA...>,
        ddc::TypeSeq<TagsB...>,
        ddc::TypeSeq<TagsR...>>
    : std::conditional_t<
              TypeSeqRank<detail::SingleType<HeadTagsA>, ddc::TypeSeq<TagsB...>>::present,
              TypeSeqRemove<
                      ddc::TypeSeq<TailTagsA...>,
                      ddc::TypeSeq<TagsB...>,
                      ddc::TypeSeq<TagsR...>>,
              TypeSeqRemove<
                      ddc::TypeSeq<TailTagsA...>,
                      ddc::TypeSeq<TagsB...>,
                      ddc::TypeSeq<TagsR..., HeadTagsA>>>
{
};

/// R contains all elements in A and elements in B that are not in A.
/// Remark 1: This operation preserves the order from A.
/// Remark 2: It is similar to the set union in the set theory (R = AUB).
/// Example: A = [a, b, c], B = [z, c, y], R = [a, b, c, z, y]
template <class TagSeqA, class TagSeqB, class TagSeqR>
struct TypeSeqMerge
{
};

template <class... TagsA, class... TagsR>
struct TypeSeqMerge<ddc::TypeSeq<TagsA...>, ddc::TypeSeq<>, ddc::TypeSeq<TagsR...>>
{
    using type = ddc::TypeSeq<TagsR...>;
};

template <class... TagsA, class HeadTagsB, class... TailTagsB, class... TagsR>
struct TypeSeqMerge<
        ddc::TypeSeq<TagsA...>,
        ddc::TypeSeq<HeadTagsB, TailTagsB...>,
        ddc::TypeSeq<TagsR...>>
    : std::conditional_t<
              TypeSeqRank<detail::SingleType<HeadTagsB>, ddc::TypeSeq<TagsA...>>::present,
              TypeSeqMerge<
                      ddc::TypeSeq<TagsA...>,
                      ddc::TypeSeq<TailTagsB...>,
                      ddc::TypeSeq<TagsR...>>,
              TypeSeqMerge<
                      ddc::TypeSeq<TagsA...>,
                      ddc::TypeSeq<TailTagsB...>,
                      ddc::TypeSeq<TagsR..., HeadTagsB>>>
{
};

/// `type` contains all elements in A then all elements in B.
/// Example: A = [a, b, c], B = [z, c, y], returned type [a, b, c, z, c, y]
template <class TagSeqA, class TagSeqB>
struct TypeSeqCat
{
};

template <class... TagsA, class... TagsB>
struct TypeSeqCat<ddc::TypeSeq<TagsA...>, ddc::TypeSeq<TagsB...>>
{
    using type = ddc::TypeSeq<TagsA..., TagsB...>;
};

/// A is replaced by element of C at same position than the first element of B equal to A.
/// Remark : It may not be useful in its own, it is an helper for TypeSeqReplace
template <class TagA, class TagSeqB, class TagSeqC>
struct TypeSeqReplaceSingle
{
};

template <class TagA>
struct TypeSeqReplaceSingle<TagA, ddc::TypeSeq<>, ddc::TypeSeq<>>
{
    using type = TagA;
};

template <class TagA, class HeadTagsB, class... TailTagsB, class HeadTagsC, class... TailTagsC>
struct TypeSeqReplaceSingle<
        TagA,
        ddc::TypeSeq<HeadTagsB, TailTagsB...>,
        ddc::TypeSeq<HeadTagsC, TailTagsC...>>
    : std::conditional_t<
              std::is_same_v<TagA, HeadTagsB>,
              TypeSeqReplaceSingle<HeadTagsC, ddc::TypeSeq<>, ddc::TypeSeq<>>,
              TypeSeqReplaceSingle<TagA, ddc::TypeSeq<TailTagsB...>, ddc::TypeSeq<TailTagsC...>>>
{
};

/// R contains all elements of A except those of B which are replaced by those of C.
/// Remark : This operation preserves the orders.
template <class TagSeqA, class TagSeqB, class TagSeqC, class TagSeqR>
struct TypeSeqReplace
{
};

template <class... TagsB, class... TagsC, class... TagsR>
struct TypeSeqReplace<
        ddc::TypeSeq<>,
        ddc::TypeSeq<TagsB...>,
        ddc::TypeSeq<TagsC...>,
        ddc::TypeSeq<TagsR...>>
{
    using type = ddc::TypeSeq<TagsR...>;
};

template <class HeadTagsA, class... TailTagsA, class... TagsB, class... TagsC, class... TagsR>
struct TypeSeqReplace<
        ddc::TypeSeq<HeadTagsA, TailTagsA...>,
        ddc::TypeSeq<TagsB...>,
        ddc::TypeSeq<TagsC...>,
        ddc::TypeSeq<TagsR...>>
    : TypeSeqReplace<
              ddc::TypeSeq<TailTagsA...>,
              ddc::TypeSeq<TagsB...>,
              ddc::TypeSeq<TagsC...>,
              ddc::TypeSeq<
                      TagsR...,
                      typename TypeSeqReplaceSingle<
                              HeadTagsA,
                              ddc::TypeSeq<TagsB...>,
                              ddc::TypeSeq<TagsC...>>::type>>
{
};

template <class T>
struct ToTypeSeq
{
};

template <class T, concepts::type_seq TagSeq>
struct Rebind
{
};

template <class T, concepts::type_seq TagSeq>
using rebind_t = Rebind<T, TagSeq>::type;

} // namespace detail

//! @brief Number of types in a ddc::TypeSeq.
//!
//! @details The primary template is only reached for types not matched by the specialization on
//! ddc::TypeSeq and yields `std::numeric_limits<std::size_t>::max()`.
//!
//! @tparam TypeSeq A ddc::TypeSeq.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TypeSeq>
constexpr std::size_t type_seq_size_v = std::numeric_limits<std::size_t>::max();

/// Returns the number of types in a @ref TypeSeq.
///
/// @ingroup ddc_core_utils
template <class... Tags>
constexpr std::size_t type_seq_size_v<TypeSeq<Tags...>> = sizeof...(Tags);

//! @brief Position (starting from 0) of the first occurrence of a type in a ddc::TypeSeq.
//!
//! @details Equal to `std::numeric_limits<std::size_t>::max()` if `QueryTag` is absent from the
//! sequence.
//!
//! @tparam QueryTag The type to search for.
//! @tparam TypeSeq The ddc::TypeSeq in which the type is searched.
//!
//! @ingroup ddc_core_utils
template <class QueryTag, concepts::type_seq TypeSeq>
constexpr std::size_t type_seq_rank_v = std::numeric_limits<std::size_t>::max();

//! @brief Whether a type is present in a ddc::TypeSeq.
//!
//! @tparam QueryTag The type to search for.
//! @tparam OTypeSeq The ddc::TypeSeq in which the type is searched.
//!
//! @ingroup ddc_core_utils
template <class QueryTag, concepts::type_seq OTypeSeq>
constexpr bool in_tags_v = false;

//! @brief Whether every type of a ddc::TypeSeq is present in another one.
//!
//! @details Despite its name, this is true when all the types of `TypeSeq` are found in
//! `OTypeSeq` (that is, `TypeSeq` is a subset of `OTypeSeq`), whatever their order. It is true for
//! an empty `TypeSeq`.
//!
//! @tparam TypeSeq The ddc::TypeSeq whose types are searched.
//! @tparam OTypeSeq The ddc::TypeSeq in which they are searched.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TypeSeq, concepts::type_seq OTypeSeq>
constexpr bool type_seq_contains_v = false;

//! @brief Whether a ddc::TypeSeq contains no duplicated type.
//!
//! @tparam TypeSeq The ddc::TypeSeq to check.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TypeSeq>
constexpr bool type_seq_is_unique_v = false;

//! @brief Whether two ddc::TypeSeq contain the same set of types.
//!
//! @details The order and the multiplicity of the types are ignored (mutual inclusion according
//! to ddc::type_seq_contains_v).
//!
//! @tparam TypeSeq The first ddc::TypeSeq.
//! @tparam OTypeSeq The second ddc::TypeSeq.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TypeSeq, concepts::type_seq OTypeSeq>
constexpr bool type_seq_same_v
        = type_seq_contains_v<TypeSeq, OTypeSeq> && type_seq_contains_v<OTypeSeq, TypeSeq>;

//! @brief Specialization of ddc::in_tags_v for a ddc::TypeSeq.
//!
//! @ingroup ddc_core_utils
template <class QueryTag, class... Tags>
constexpr bool in_tags_v<QueryTag, TypeSeq<Tags...>>
        = detail::TypeSeqRank<detail::SingleType<QueryTag>, TypeSeq<Tags...>>::present;

//! @brief Specialization of ddc::type_seq_contains_v for a ddc::TypeSeq.
//!
//! @ingroup ddc_core_utils
template <class... Tags, class OTypeSeq>
constexpr bool type_seq_contains_v<TypeSeq<Tags...>, OTypeSeq>
        = (detail::TypeSeqRank<detail::SingleType<Tags>, OTypeSeq>::present && ...);

//! @brief Specialization of ddc::type_seq_rank_v for a ddc::TypeSeq.
//!
//! @ingroup ddc_core_utils
template <class QueryTag, class... Tags>
constexpr std::size_t type_seq_rank_v<QueryTag, TypeSeq<Tags...>>
        = detail::TypeSeqRank<detail::SingleType<QueryTag>, TypeSeq<Tags...>>::val;

//! @brief Type at a given position of a ddc::TypeSeq.
//!
//! @tparam I Position of the type, starting from 0.
//! @tparam TagSeq The ddc::TypeSeq.
//!
//! @ingroup ddc_core_utils
template <std::size_t I, concepts::type_seq TagSeq>
using type_seq_element_t = detail::TypeSeqElement<I, TagSeq>::type;

//! @brief ddc::TypeSeq of the types of `TagSeqA` that are not in `TagSeqB`.
//!
//! @details The order of `TagSeqA` is preserved (set difference). For example, removing
//! `[z, c, y]` from `[a, b, c]` gives `[a, b]`.
//!
//! @tparam TagSeqA The ddc::TypeSeq from which types are removed.
//! @tparam TagSeqB The ddc::TypeSeq of the types to remove.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TagSeqA, concepts::type_seq TagSeqB>
using type_seq_remove_t = detail::TypeSeqRemove<TagSeqA, TagSeqB, TypeSeq<>>::type;

//! @brief ddc::TypeSeq of the types of `TagSeqA` followed by those of `TagSeqB` absent from it.
//!
//! @details The order of `TagSeqA` is preserved (set union). For example, merging `[z, c, y]`
//! into `[a, b, c]` gives `[a, b, c, z, y]`.
//!
//! @tparam TagSeqA The first ddc::TypeSeq.
//! @tparam TagSeqB The ddc::TypeSeq whose missing types are appended.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TagSeqA, concepts::type_seq TagSeqB>
using type_seq_merge_t = detail::TypeSeqMerge<TagSeqA, TagSeqB, TagSeqA>::type;

//! @brief Concatenation of two ddc::TypeSeq, duplicates being kept.
//!
//! @tparam TagSeqA The ddc::TypeSeq placed first.
//! @tparam TagSeqB The ddc::TypeSeq placed last.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TagSeqA, concepts::type_seq TagSeqB>
using type_seq_cat_t = detail::TypeSeqCat<TagSeqA, TagSeqB>::type;

//! @brief ddc::TypeSeq obtained from `TagSeqA` by replacing the types of `TagSeqB` by those of
//! `TagSeqC`.
//!
//! @details Each type of `TagSeqA` equal to the type at position `i` of `TagSeqB` is replaced by
//! the type at position `i` of `TagSeqC`. The order of `TagSeqA` is preserved.
//!
//! @tparam TagSeqA The ddc::TypeSeq in which types are replaced.
//! @tparam TagSeqB The ddc::TypeSeq of the types to replace.
//! @tparam TagSeqC The ddc::TypeSeq of the replacing types, at the same positions as `TagSeqB`.
//!
//! @ingroup ddc_core_utils
template <concepts::type_seq TagSeqA, concepts::type_seq TagSeqB, concepts::type_seq TagSeqC>
using type_seq_replace_t = detail::TypeSeqReplace<TagSeqA, TagSeqB, TagSeqC, TypeSeq<>>::type;

//! @brief Specialization of ddc::type_seq_is_unique_v for a ddc::TypeSeq.
//!
//! @ingroup ddc_core_utils
template <class... Tags>
constexpr bool type_seq_is_unique_v<TypeSeq<Tags...>>
        = ((type_seq_size_v<type_seq_remove_t<TypeSeq<Tags...>, TypeSeq<Tags>>>
            == sizeof...(Tags) - 1)
           && ...);

//! @brief ddc::TypeSeq associated with a type through its `ToTypeSeq` specialization.
//!
//! @details Not specified by the source for a generic `T`: the result is the `type` member of
//! the (internal) `ToTypeSeq<T>` specialization, which other DDC types specialize.
//!
//! @tparam T The type to convert.
//!
//! @ingroup ddc_core_utils
template <class T>
using to_type_seq_t = detail::ToTypeSeq<T>::type;

} // namespace ddc
