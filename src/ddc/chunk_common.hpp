// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ChunkCommon, the base class of Chunk and ChunkSpan, and get_domain().
#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "chunk_traits.hpp"
#include "discrete_dimension.hpp"
#include "discrete_domain.hpp"
#include "discrete_vector.hpp"
#include "type_seq.hpp"

namespace ddc {

/** Access the domain (or subdomain) of a view
 * @param[in]  chunk the view whose domain to access
 * @return the domain of view in the queried dimensions
 * @tparam QueryDDims The discrete dimensions of the queried (sub)domain.
 * @ingroup ddc_containers
 */
template <concepts::discrete_dimension... QueryDDims, class ChunkType>
KOKKOS_FUNCTION auto get_domain(ChunkType const& chunk) noexcept
{
    static_assert(is_chunk_v<ChunkType>, "Not a chunk span type");
    return chunk.template domain<QueryDDims...>();
}

//! @brief Common base of Chunk and ChunkSpan, holding a data pointer and its support.
//!
//! @details It stores a `Kokkos::mdspan` on the memory allocation (indexed from 0, always strided)
//! together with the discrete domain (or other support) on which the values are defined. It does
//! not own the memory.
//!
//! @tparam ElementType The type of the stored elements, possibly `const`.
//! @tparam SupportType The type of the support on which the values are defined, e.g. a
//! DiscreteDomain.
//! @tparam LayoutStridedPolicy The `Kokkos::mdspan` layout policy, which must be always strided.
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class LayoutStridedPolicy>
class ChunkCommon
{
public:
    //! @brief Type of the support on which the values are defined.
    using discrete_domain_type = SupportType;

    /// The dereferenceable part of the co-domain but with a different domain, starting at 0
    using allocation_mdspan_type = Kokkos::mdspan<
            ElementType,
            Kokkos::dextents<std::size_t, SupportType::rank()>,
            LayoutStridedPolicy>;

    //! @brief Type of the `Kokkos::mdspan` on the allocation with `const` elements.
    using const_allocation_mdspan_type = Kokkos::mdspan<
            ElementType const,
            Kokkos::dextents<std::size_t, SupportType::rank()>,
            LayoutStridedPolicy>;

    //! @brief Type of the discrete elements of the support.
    using discrete_element_type = discrete_domain_type::discrete_element_type;

    //! @brief Type of the discrete vectors of the support.
    using discrete_vector_type = discrete_domain_type::discrete_vector_type;

    //! @brief Type of the extents of the underlying `Kokkos::mdspan`.
    using extents_type = allocation_mdspan_type::extents_type;

    //! @brief Layout policy of the underlying `Kokkos::mdspan`.
    using layout_type = allocation_mdspan_type::layout_type;

    //! @brief Accessor policy of the underlying `Kokkos::mdspan`.
    using accessor_type = allocation_mdspan_type::accessor_type;

    //! @brief Layout mapping of the underlying `Kokkos::mdspan`.
    using mapping_type = allocation_mdspan_type::mapping_type;

    //! @brief Type of the stored elements, possibly `const`.
    using element_type = allocation_mdspan_type::element_type;

    //! @brief Type of the stored elements without `const` qualification.
    using value_type = allocation_mdspan_type::value_type;

    //! @brief Type used to express sizes and extents.
    using size_type = allocation_mdspan_type::size_type;

    //! @brief Type of the data handle (a pointer) of the underlying `Kokkos::mdspan`.
    using data_handle_type = allocation_mdspan_type::data_handle_type;

    //! @brief Type of the references to the stored elements.
    using reference = allocation_mdspan_type::reference;

    // ChunkCommon, ChunkSpan and Chunk need to access to m_allocation_mdspan_mdspan and m_domain of other template versions
    template <class, class, class>
    friend class ChunkCommon;

    template <class, class, class, class>
    friend class ChunkSpan;

    template <class, class, class>
    friend class Chunk;

    static_assert(mapping_type::is_always_strided());

protected:
    /// The raw view of the data
    allocation_mdspan_type m_allocation_mdspan;

    /// The mesh on which this chunk is defined
    SupportType m_domain;

public:
    //! @brief Number of dimensions of the underlying `Kokkos::mdspan`.
    //!
    //! @return The rank, equal to the rank of the support.
    static KOKKOS_FUNCTION constexpr int rank() noexcept
    {
        return extents_type::rank();
    }

    //! @brief Number of dimensions whose extent is only known at run time.
    //!
    //! @return The number of dynamic extents of the underlying `Kokkos::mdspan`.
    static KOKKOS_FUNCTION constexpr int rank_dynamic() noexcept
    {
        return extents_type::rank_dynamic();
    }

    //! @brief Compile-time extent of a dimension.
    //!
    //! @param[in] r Index of the dimension, between 0 and `rank() - 1`.
    //! @return The static extent along `r`, or `Kokkos::dynamic_extent` if it is only known at
    //! run time.
    static KOKKOS_FUNCTION constexpr size_type static_extent(std::size_t r) noexcept
    {
        return extents_type::static_extent(r);
    }

    //! @brief Tells whether the layout mapping is unique for any valid extents.
    //!
    //! @return `true` if distinct valid indices always refer to distinct elements.
    static KOKKOS_FUNCTION constexpr bool is_always_unique() noexcept
    {
        return mapping_type::is_always_unique();
    }

    //! @brief Tells whether the layout mapping is exhaustive for any valid extents.
    //!
    //! @return `true` if the mapping covers the whole span of the allocation without gaps.
    static KOKKOS_FUNCTION constexpr bool is_always_exhaustive() noexcept
    {
        return mapping_type::is_always_exhaustive();
    }

    //! @brief Tells whether the layout mapping is strided for any valid extents.
    //!
    //! @return `true` if the mapping is strided; this is enforced by a `static_assert`.
    static KOKKOS_FUNCTION constexpr bool is_always_strided() noexcept
    {
        return mapping_type::is_always_strided();
    }

private:
    static KOKKOS_FUNCTION constexpr allocation_mdspan_type make_allocation_mdspan(
            ElementType* ptr,
            SupportType const& domain)
        requires(std::is_constructible_v<mapping_type, extents_type>)
    {
        return allocation_mdspan_type(ptr, detail::array(domain.extents()));
    }

public:
    //! @brief Gives the accessor of the underlying `Kokkos::mdspan`.
    //!
    //! @return A copy of the accessor.
    KOKKOS_FUNCTION constexpr accessor_type accessor() const
    {
        return m_allocation_mdspan.accessor();
    }

    //! @brief Gives the number of elements along each dimension.
    //!
    //! @return A discrete vector containing the extent of each dimension of the support.
    KOKKOS_FUNCTION constexpr SupportType::discrete_vector_type extents() const noexcept
    {
        return m_domain.extents();
    }

    //! @brief Gives the number of elements along one dimension.
    //!
    //! @tparam QueryDDim The discrete dimension to query.
    //! @return The extent along `QueryDDim`.
    template <concepts::discrete_dimension QueryDDim>
    KOKKOS_FUNCTION constexpr size_type extent() const noexcept
    {
        return m_domain.template extent<QueryDDim>();
    }

    //! @brief Gives the total number of elements.
    //!
    //! @return The number of elements of the underlying `Kokkos::mdspan`.
    KOKKOS_FUNCTION constexpr size_type size() const noexcept
    {
        return allocation_mdspan().size();
    }

    //! @brief Gives the layout mapping of the underlying `Kokkos::mdspan`.
    //!
    //! @return A copy of the mapping.
    KOKKOS_FUNCTION constexpr mapping_type mapping() const noexcept
    {
        return allocation_mdspan().mapping();
    }

    //! @brief Tells whether distinct valid indices refer to distinct elements.
    //!
    //! @return `true` if the mapping of this object is unique.
    KOKKOS_FUNCTION constexpr bool is_unique() const noexcept
    {
        return allocation_mdspan().is_unique();
    }

    //! @brief Tells whether the mapping covers the whole span of the allocation.
    //!
    //! @return `true` if the mapping of this object is exhaustive.
    KOKKOS_FUNCTION constexpr bool is_exhaustive() const noexcept
    {
        return allocation_mdspan().is_exhaustive();
    }

    //! @brief Tells whether the mapping is strided.
    //!
    //! @return `true` if the mapping of this object is strided.
    KOKKOS_FUNCTION constexpr bool is_strided() const noexcept
    {
        return allocation_mdspan().is_strided();
    }

    //! @brief Gives the stride along one dimension.
    //!
    //! @tparam QueryDDim The discrete dimension to query.
    //! @return The distance, in number of elements, between two consecutive elements along
    //! `QueryDDim` in memory.
    template <concepts::discrete_dimension QueryDDim>
    KOKKOS_FUNCTION constexpr size_type stride() const
    {
        return m_allocation_mdspan.stride(type_seq_rank_v<QueryDDim, to_type_seq_t<SupportType>>);
    }

    /** Provide access to the domain on which this chunk is defined
     * @return the domain on which this chunk is defined
     */
    KOKKOS_FUNCTION constexpr SupportType domain() const noexcept
    {
        return m_domain;
    }

    /** Provide access to the domain on which this chunk is defined
     * @return the domain on which this chunk is defined
     */
    template <concepts::discrete_dimension... QueryDDims>
    KOKKOS_FUNCTION constexpr DiscreteDomain<QueryDDims...> domain() const noexcept
    {
        return DiscreteDomain<QueryDDims...>(domain());
    }

protected:
    /// Empty ChunkCommon
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkCommon() = default;

    /** Constructs a new ChunkCommon from scratch
     * @param allocation_mdspan
     * @param domain
     */
    KOKKOS_FUNCTION constexpr ChunkCommon(
            allocation_mdspan_type allocation_mdspan,
            SupportType const& domain) noexcept
        : m_allocation_mdspan(std::move(allocation_mdspan))
        , m_domain(domain)
    {
    }

    /** Constructs a new ChunkCommon from scratch
     * @param ptr the allocation pointer to the data
     * @param domain the domain that sustains the view
     */
    KOKKOS_FUNCTION constexpr ChunkCommon(ElementType* ptr, SupportType const& domain) requires(std::is_constructible_v<mapping_type, extents_type>)
        : m_allocation_mdspan(make_allocation_mdspan(ptr, domain))
        , m_domain(domain)
    {
        // Handle the case where an allocation of size 0 returns a nullptr.
        KOKKOS_ASSERT(domain.empty() || ((ptr != nullptr) && !domain.empty()))
    }

    /** Constructs a new ChunkCommon by copy, yields a new view to the same data
     * @param other the ChunkCommon to copy
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkCommon(ChunkCommon const& other) = default;

    /** Constructs a new ChunkCommon by move
     * @param other the ChunkCommon to move
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkCommon(ChunkCommon&& other) noexcept = default;

    KOKKOS_DEFAULTED_FUNCTION ~ChunkCommon() noexcept = default;

    /** Copy-assigns a new value to this ChunkCommon, yields a new view to the same data
     * @param other the ChunkCommon to copy
     * @return *this
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkCommon& operator=(ChunkCommon const& other) = default;

    /** Move-assigns a new value to this ChunkCommon
     * @param other the ChunkCommon to move
     * @return *this
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkCommon& operator=(ChunkCommon&& other) noexcept
            = default;

    /** Access to the underlying allocation pointer
     * @return allocation pointer
     */
    KOKKOS_FUNCTION constexpr ElementType* data_handle() const
    {
        return m_allocation_mdspan.data_handle();
    }

    /** Provide a modifiable view of the data
     * @return a modifiable view of the data
     */
    KOKKOS_FUNCTION constexpr allocation_mdspan_type allocation_mdspan() const
    {
        return m_allocation_mdspan;
    }
};

} // namespace ddc
