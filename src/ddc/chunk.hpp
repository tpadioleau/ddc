// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides Chunk, an owning multi-dimensional array defined on a discrete support.
#pragma once

#include <cassert>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail/kokkos.hpp"

#include "chunk_common.hpp"
#include "chunk_span.hpp"
#include "chunk_traits.hpp"
#include "discrete_dimension.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "kokkos_allocator.hpp"

namespace ddc {

template <class ElementType, class, class Allocator = HostAllocator<ElementType>>
class Chunk;

//! @brief Marks Chunk as a chunk type, see enable_chunk.
//!
//! @tparam ElementType The type of the elements of the Chunk.
//! @tparam SupportType The support type of the Chunk.
//! @tparam Allocator The allocator type of the Chunk.
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class Allocator>
inline constexpr bool enable_chunk<Chunk<ElementType, SupportType, Allocator>> = true;

//! @brief Owning multi-dimensional array of values defined on a discrete support.
//!
//! @details The memory is allocated at construction with the `allocate(label, size)` member of the
//! allocator and released by the destructor with its `deallocate` member. The values are
//! uninitialized after construction and stored in `Kokkos::layout_right` order. A Chunk can not be
//! copied (use a deep copy instead) but can be moved: the moved-from Chunk no longer owns any
//! memory. Use span_view() or ChunkSpan to obtain a non-owning view on the data.
//!
//! @tparam ElementType The type of the stored elements.
//! @tparam SupportType The type of the support on which the values are defined, e.g. a
//! DiscreteDomain.
//! @tparam Allocator The allocator type, e.g. HostAllocator (default) or DeviceAllocator. It must
//! provide the `value_type` and `memory_space` types, `allocate(label, size)` and
//! `deallocate(pointer, size)`.
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class Allocator>
class Chunk : public ChunkCommon<ElementType, SupportType, Kokkos::layout_right>
{
protected:
    //! @brief Type of the base class, ChunkCommon.
    using base_type = ChunkCommon<ElementType, SupportType, Kokkos::layout_right>;

public:
    /// type of a span of this full chunk
    using span_type = ChunkSpan<
            ElementType,
            SupportType,
            Kokkos::layout_right,
            typename Allocator::memory_space>;

    /// type of a view of this full chunk
    using view_type = ChunkSpan<
            ElementType const,
            SupportType,
            Kokkos::layout_right,
            typename Allocator::memory_space>;

    /// The dereferenceable part of the co-domain but with indexing starting at 0
    using allocation_mdspan_type = base_type::allocation_mdspan_type;

    //! @brief Type of the `Kokkos::mdspan` on the allocation with `const` elements.
    using const_allocation_mdspan_type = base_type::const_allocation_mdspan_type;

    //! @brief Type of the support on which the values are defined.
    using discrete_domain_type = base_type::discrete_domain_type;

    //! @brief Kokkos memory space in which the data is allocated.
    using memory_space = Allocator::memory_space;

    //! @brief Type of the discrete elements of the support.
    using discrete_element_type = base_type::discrete_element_type;

    //! @brief Type of the discrete vectors of the support.
    using discrete_vector_type = base_type::discrete_vector_type;

    //! @brief Type of the extents of the underlying `Kokkos::mdspan`.
    using extents_type = base_type::extents_type;

    //! @brief Layout policy of the underlying `Kokkos::mdspan`, `Kokkos::layout_right`.
    using layout_type = base_type::layout_type;

    //! @brief Layout mapping of the underlying `Kokkos::mdspan`.
    using mapping_type = base_type::mapping_type;

    //! @brief Type of the stored elements.
    using element_type = base_type::element_type;

    //! @brief Type of the stored elements without `const` qualification.
    using value_type = base_type::value_type;

    //! @brief Type used to express sizes and extents.
    using size_type = base_type::size_type;

    //! @brief Type of the data handle (a pointer) of the underlying `Kokkos::mdspan`.
    using data_handle_type = base_type::data_handle_type;

    //! @brief Type of the references to the stored elements.
    using reference = base_type::reference;

    template <class, class, class>
    friend class Chunk;

private:
    Allocator m_allocator;

    std::string m_label;

public:
    /// Empty Chunk
    Chunk() = default;

    /// Construct a labeled Chunk on a domain with uninitialized values
    /// @param[in] label The label of the allocation, returned by label().
    /// @param[in] domain The support on which the values are defined.
    /// @param[in] allocator The allocator providing the memory, kept by the chunk to release it.
    explicit Chunk(
            std::string const& label,
            SupportType const& domain,
            Allocator allocator = Allocator())
        : base_type(allocator.allocate(label, domain.size()), domain)
        , m_allocator(std::move(allocator))
        , m_label(label)
    {
    }

    /// Construct a Chunk on a domain with uninitialized values
    /// @details The label of the chunk is `"no-label"`.
    /// @param[in] domain The support on which the values are defined.
    /// @param[in] allocator The allocator providing the memory, kept by the chunk to release it.
    explicit Chunk(SupportType const& domain, Allocator allocator = Allocator())
        : Chunk("no-label", domain, std::move(allocator))
    {
    }

    /// Deleted: use deepcopy instead
    Chunk(Chunk const& other) = delete;

    /** Constructs a new Chunk by move
     * @param other the Chunk to move
     */
    Chunk(Chunk&& other) noexcept
        : base_type(std::move(static_cast<base_type&>(other)))
        , m_allocator(std::move(other.m_allocator))
        , m_label(std::move(other.m_label))
    {
        other.m_allocation_mdspan
                = allocation_mdspan_type(nullptr, other.m_allocation_mdspan.mapping());
    }

    ~Chunk() noexcept
    {
        if (this->m_allocation_mdspan.data_handle()) {
            m_allocator.deallocate(this->data_handle(), this->size());
        }
    }

    /// Deleted: use deepcopy instead
    Chunk& operator=(Chunk const& other) = delete;

    /** Move-assigns a new value to this field
     * @param other the Chunk to move
     * @return *this
     */
    Chunk& operator=(Chunk&& other) noexcept
    {
        if (this->data_handle()) {
            m_allocator.deallocate(this->data_handle(), this->size());
        }
        static_cast<base_type&>(*this) = std::move(static_cast<base_type&>(other));
        m_allocator = std::move(other.m_allocator);
        m_label = std::move(other.m_label);
        other.m_allocation_mdspan
                = allocation_mdspan_type(nullptr, other.m_allocation_mdspan.mapping());

        return *this;
    }

    /// Slice out some dimensions
    /// @param[in] slice_spec The offsets, from the front of the support, at which to slice along
    /// the dimensions `QueryDDims`.
    /// @return A read-only ChunkSpan on the data of this chunk, without the dimensions
    /// `QueryDDims`.
    template <concepts::discrete_dimension... QueryDDims>
    auto operator[](DiscreteVector<QueryDDims...> const& slice_spec) const
    {
        return view_type(*this)[slice_spec];
    }

    /// Slice out some dimensions
    /// @param[in] slice_spec The offsets, from the front of the support, at which to slice along
    /// the dimensions `QueryDDims`.
    /// @return A ChunkSpan on the data of this chunk, without the dimensions `QueryDDims`.
    template <concepts::discrete_dimension... QueryDDims>
    auto operator[](DiscreteVector<QueryDDims...> const& slice_spec)
    {
        return span_view()[slice_spec];
    }

    /// Slice out some dimensions
    /// @param[in] slice_spec The position, in the support, at which to slice along the dimensions
    /// `QueryDDims`.
    /// @return A read-only ChunkSpan on the data of this chunk, without the dimensions
    /// `QueryDDims`.
    template <concepts::discrete_dimension... QueryDDims>
    auto operator[](DiscreteElement<QueryDDims...> const& slice_spec) const
    {
        return view_type(*this)[slice_spec];
    }

    /// Slice out some dimensions
    /// @param[in] slice_spec The position, in the support, at which to slice along the dimensions
    /// `QueryDDims`.
    /// @return A ChunkSpan on the data of this chunk, without the dimensions `QueryDDims`.
    template <concepts::discrete_dimension... QueryDDims>
    auto operator[](DiscreteElement<QueryDDims...> const& slice_spec)
    {
        return span_view()[slice_spec];
    }

    /// Slice out some dimensions
    /// @param[in] odomain The subdomain to which the chunk is restricted.
    /// @return A read-only ChunkSpan on the data of this chunk, restricted to `odomain`.
    template <concepts::discrete_dimension... QueryDDims>
    auto operator[](DiscreteDomain<QueryDDims...> const& odomain) const
        requires(is_discrete_domain_v<SupportType>)
    {
        return span_view()[odomain];
    }

    /// Slice out some dimensions
    /// @param[in] odomain The subdomain to which the chunk is restricted.
    /// @return A ChunkSpan on the data of this chunk, restricted to `odomain`.
    template <concepts::discrete_dimension... QueryDDims>
    auto operator[](DiscreteDomain<QueryDDims...> const& odomain)
        requires(is_discrete_domain_v<SupportType>)
    {
        return span_view()[odomain];
    }

    /** Element access using a list of DiscreteElement
     * @param delems discrete coordinates
     * @return const-reference to this element
     */
    template <concepts::discrete_element... DElems>
    element_type const& operator()(DElems const&... delems) const noexcept
    {
        static_assert(
                SupportType::rank() == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        assert(this->m_domain.contains(delems...));
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(this->m_domain.distance_from_front(delems...)));
    }

    /** Element access using a list of DiscreteVector
     * @param dvects discrete vectors
     * @return reference to this element
     */
    template <concepts::discrete_vector... DVects>
    element_type const& operator()(DVects const&... dvects) const noexcept
        requires(sizeof...(DVects) != 0)
    {
        static_assert(
                SupportType::rank() == (0 + ... + DVects::size()),
                "Invalid number of dimensions");
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(discrete_vector_type(dvects...)));
    }

    /** Element access using a list of DiscreteElement
     * @param delems discrete coordinates
     * @return reference to this element
     */
    template <concepts::discrete_element... DElems>
    element_type& operator()(DElems const&... delems) noexcept
    {
        static_assert(
                SupportType::rank() == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        assert(this->m_domain.contains(delems...));
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(this->m_domain.distance_from_front(delems...)));
    }

    /** Element access using a list of DiscreteVector
     * @param dvects discrete vectors
     * @return reference to this element
     */
    template <concepts::discrete_vector... DVects>
    element_type& operator()(DVects const&... dvects) noexcept
        requires(sizeof...(DVects) != 0)
    {
        static_assert(
                SupportType::rank() == (0 + ... + DVects::size()),
                "Invalid number of dimensions");
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(discrete_vector_type(dvects...)));
    }

    /** Returns the label of the Chunk
     * @return c-string
     */
    char const* label() const
    {
        return m_label.c_str();
    }

    /** Access to the underlying allocation pointer
     * @return read-only allocation pointer
     */
    ElementType const* data_handle() const
    {
        return base_type::data_handle();
    }

    /** Access to the underlying allocation pointer
     * @return allocation pointer
     */
    ElementType* data_handle()
    {
        return base_type::data_handle();
    }

    /** Provide a mdspan on the memory allocation
     * @return read-only allocation mdspan
     */
    const_allocation_mdspan_type allocation_mdspan() const
    {
        return base_type::allocation_mdspan();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    allocation_mdspan_type allocation_mdspan()
    {
        return base_type::allocation_mdspan();
    }

    /** Provide an unmanaged `Kokkos::View` on the memory allocation
     * @return allocation `Kokkos::View`
     */
    auto allocation_kokkos_view()
    {
        auto s = this->allocation_mdspan();
        auto kokkos_layout = detail::build_kokkos_layout(
                s.extents(),
                s.mapping(),
                std::make_index_sequence<SupportType::rank()> {});
        return Kokkos::View<
                detail::mdspan_to_kokkos_element_t<ElementType, SupportType::rank()>,
                decltype(kokkos_layout),
                typename Allocator::memory_space>(s.data_handle(), kokkos_layout);
    }

    /** Provide an unmanaged `Kokkos::View` on the memory allocation
     * @return read-only allocation `Kokkos::View`
     */
    auto allocation_kokkos_view() const
    {
        auto s = this->allocation_mdspan();
        auto kokkos_layout = detail::build_kokkos_layout(
                s.extents(),
                s.mapping(),
                std::make_index_sequence<SupportType::rank()> {});
        return Kokkos::View<
                detail::mdspan_to_kokkos_element_t<ElementType const, SupportType::rank()>,
                decltype(kokkos_layout),
                typename Allocator::memory_space>(s.data_handle(), kokkos_layout);
    }

    //! @brief Gives a read-only view on the data of this chunk.
    //!
    //! @return A ChunkSpan with `const` elements viewing the data of this chunk.
    //! @note The returned view does not own the data: it must not outlive this chunk.
    view_type span_cview() const
    {
        return view_type(*this);
    }

    //! @brief Gives a read-only view on the data of this chunk.
    //!
    //! @return A ChunkSpan with `const` elements viewing the data of this chunk.
    //! @note The returned view does not own the data: it must not outlive this chunk.
    view_type span_view() const
    {
        return view_type(*this);
    }

    //! @brief Gives a view on the data of this chunk.
    //!
    //! @return A ChunkSpan viewing the data of this chunk.
    //! @note The returned view does not own the data: it must not outlive this chunk.
    span_type span_view()
    {
        return span_type(*this);
    }
};

//! @brief Deduces the Chunk type from a label, a support and an allocator.
//!
//! @details The element type is the `value_type` of the allocator.
//! @ingroup ddc_containers
template <class SupportType, class Allocator>
Chunk(std::string const&, SupportType const&, Allocator)
        -> Chunk<typename Allocator::value_type, SupportType, Allocator>;

//! @brief Deduces the Chunk type from a support and an allocator.
//!
//! @details The element type is the `value_type` of the allocator.
//! @ingroup ddc_containers
template <class SupportType, class Allocator>
Chunk(SupportType const&, Allocator)
        -> Chunk<typename Allocator::value_type, SupportType, Allocator>;

} // namespace ddc
