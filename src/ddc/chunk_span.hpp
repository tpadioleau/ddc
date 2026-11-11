// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ChunkSpan, a non-owning view on multi-dimensional data, and ChunkView.
#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail/kokkos.hpp"
#include "detail/macros.hpp"

#include "chunk_common.hpp"
#include "chunk_traits.hpp"
#include "discrete_dimension.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "type_seq.hpp"

namespace ddc {

template <class, class, class>
class Chunk;

template <
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy = Kokkos::layout_right,
        class MemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
class ChunkSpan;

//! @brief Marks ChunkSpan as a chunk type, see enable_chunk.
//!
//! @tparam ElementType The type of the elements of the ChunkSpan.
//! @tparam SupportType The support type of the ChunkSpan.
//! @tparam LayoutStridedPolicy The layout policy of the ChunkSpan.
//! @tparam MemorySpace The memory space of the ChunkSpan.
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool
        enable_chunk<ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

//! @brief Marks ChunkSpan as a borrowed chunk, see enable_borrowed_chunk.
//!
//! @tparam ElementType The type of the elements of the ChunkSpan.
//! @tparam SupportType The support type of the ChunkSpan.
//! @tparam LayoutStridedPolicy The layout policy of the ChunkSpan.
//! @tparam MemorySpace The memory space of the ChunkSpan.
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool
        enable_borrowed_chunk<ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

//! @brief Non-owning view on multi-dimensional data defined on a discrete support.
//!
//! @details A ChunkSpan associates a `Kokkos::mdspan` on existing memory with the support
//! (usually a DiscreteDomain) on which the values are defined. Copying a ChunkSpan yields a new
//! view on the same data. The viewed memory is not owned: it must outlive the ChunkSpan. A
//! ChunkSpan cannot be built from a temporary Chunk. Most members are `KOKKOS_FUNCTION`, so they
//! can be called from Kokkos kernels.
//!
//! @tparam ElementType The type of the elements, `const` for a read-only view.
//! @tparam SupportType The type of the support on which the values are defined, e.g. a
//! DiscreteDomain.
//! @tparam LayoutStridedPolicy The layout policy: `Kokkos::layout_left`, `Kokkos::layout_right`
//! (default) or `Kokkos::layout_stride`.
//! @tparam MemorySpace The Kokkos memory space in which the data lives (default: the memory space
//! of `Kokkos::DefaultHostExecutionSpace`).
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
class ChunkSpan : public ChunkCommon<ElementType, SupportType, LayoutStridedPolicy>
{
    static_assert(
            std::is_same_v<LayoutStridedPolicy, Kokkos::layout_left>
                    || std::is_same_v<LayoutStridedPolicy, Kokkos::layout_right>
                    || std::is_same_v<LayoutStridedPolicy, Kokkos::layout_stride>,
            "ChunkSpan only supports layout_left, layout_right or layout_stride");
    static_assert(
            Kokkos::is_memory_space_v<MemorySpace>,
            "ChunkSpan requires a valid Kokkos memory space");

protected:
    //! @brief Type of the base class, ChunkCommon.
    using base_type = ChunkCommon<ElementType, SupportType, LayoutStridedPolicy>;

public:
    /// type of a span of this full chunk
    using span_type = ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>;

    /// type of a view of this full chunk
    using view_type = ChunkSpan<ElementType const, SupportType, LayoutStridedPolicy, MemorySpace>;

    //! @brief Type of the support on which the values are defined.
    using discrete_domain_type = base_type::discrete_domain_type;

    //! @brief Kokkos memory space in which the data lives.
    using memory_space = MemorySpace;

    /// The dereferenceable part of the co-domain but with a different domain, starting at 0
    using allocation_mdspan_type = base_type::allocation_mdspan_type;

    //! @brief Type of the `Kokkos::mdspan` on the allocation with `const` elements.
    using const_allocation_mdspan_type = base_type::const_allocation_mdspan_type;

    //! @brief Type of the discrete elements of the support.
    using discrete_element_type = discrete_domain_type::discrete_element_type;

    //! @brief Type of the discrete vectors of the support.
    using discrete_vector_type = discrete_domain_type::discrete_vector_type;

    //! @brief Type of the extents of the underlying `Kokkos::mdspan`.
    using extents_type = base_type::extents_type;

    //! @brief Layout policy of the underlying `Kokkos::mdspan`.
    using layout_type = base_type::layout_type;

    //! @brief Accessor policy of the underlying `Kokkos::mdspan`.
    using accessor_type = base_type::accessor_type;

    //! @brief Layout mapping of the underlying `Kokkos::mdspan`.
    using mapping_type = base_type::mapping_type;

    //! @brief Type of the viewed elements, possibly `const`.
    using element_type = base_type::element_type;

    //! @brief Type of the viewed elements without `const` qualification.
    using value_type = base_type::value_type;

    //! @brief Type used to express sizes and extents.
    using size_type = base_type::size_type;

    //! @brief Type of the data handle (a pointer) of the underlying `Kokkos::mdspan`.
    using data_handle_type = base_type::data_handle_type;

    //! @brief Type of the references to the viewed elements.
    using reference = base_type::reference;

    template <class, class, class, class>
    friend class ChunkSpan;

protected:
    //! @brief Gives the slice specifier of a dimension for a slicing by discrete vector.
    //!
    //! @tparam QueryDDim The dimension for which the specifier is requested.
    //! @param[in] c The offsets at which the slicing is done.
    //! @return The offset of `c` along `QueryDDim` if present in `c`, `Kokkos::full_extent`
    //! otherwise.
    template <concepts::discrete_dimension QueryDDim, concepts::discrete_dimension... ODDims>
    KOKKOS_FUNCTION static constexpr auto get_slicer_for(DiscreteVector<ODDims...> const& c)
    {
        DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
        if constexpr (in_tags_v<QueryDDim, TypeSeq<ODDims...>>) {
            return c.template get<QueryDDim>();
        } else {
            return Kokkos::full_extent;
        }
        DDC_IF_NVCC_THEN_POP
    }

    //! @brief Gives the slice specifier of a dimension for a slicing by sub-domain.
    //!
    //! @tparam QueryDDim The dimension for which the specifier is requested.
    //! @param[in] c The subdomain to which the view is restricted.
    //! @param[in] origin The domain of the view being sliced.
    //! @return The half-open range of indices of `c` relative to the front of `origin` along
    //! `QueryDDim` if present in `c`, `Kokkos::full_extent` otherwise.
    template <
            concepts::discrete_dimension QueryDDim,
            concepts::discrete_dimension... ODDims,
            concepts::discrete_dimension... OODDims>
    KOKKOS_FUNCTION static constexpr auto get_slicer_for(
            DiscreteDomain<ODDims...> const& c,
            DiscreteDomain<OODDims...> const& origin)
    {
        DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
        if constexpr (in_tags_v<QueryDDim, TypeSeq<ODDims...>>) {
            DiscreteDomain<QueryDDim> const c_slice(c);
            DiscreteDomain<QueryDDim> const origin_slice(origin);
            return std::pair<std::size_t, std::size_t>(
                    c_slice.front() - origin_slice.front(),
                    c_slice.back() + 1 - origin_slice.front());
        } else {
            return Kokkos::full_extent;
        }
        DDC_IF_NVCC_THEN_POP
    }

    template <class TypeSeq>
    struct Slicer
    {
    };

    //! @brief Applies `Kokkos::submdspan` with one slice specifier for each dimension `DDims`.
    //!
    //! @tparam DDims The dimensions of the support, in order.
    template <concepts::discrete_dimension... DDims>
    struct Slicer<TypeSeq<DDims...>>
    {
        //! @brief Slices an allocation mdspan by discrete vector.
        //!
        //! @param[in] span The allocation mdspan to slice.
        //! @param[in] c The offsets at which the slicing is done.
        //! @return The sub-mdspan, in which the dimensions of `c` are removed.
        template <concepts::discrete_dimension... ODDims>
        KOKKOS_FUNCTION constexpr auto operator()(
                allocation_mdspan_type const& span,
                DiscreteVector<ODDims...> const& c) const
        {
            return Kokkos::submdspan(span, get_slicer_for<DDims>(c)...);
        }

        //! @brief Slices an allocation mdspan by sub-domain.
        //!
        //! @param[in] span The allocation mdspan to slice.
        //! @param[in] c The subdomain to which the span is restricted.
        //! @param[in] origin The domain of the span being sliced.
        //! @return The sub-mdspan restricted to `c`.
        template <concepts::discrete_dimension... ODDims, concepts::discrete_dimension... OODDims>
        KOKKOS_FUNCTION constexpr auto operator()(
                allocation_mdspan_type const& span,
                DiscreteDomain<ODDims...> const& c,
                DiscreteDomain<OODDims...> const& origin) const
        {
            return Kokkos::submdspan(span, get_slicer_for<DDims>(c, origin)...);
        }
    };

public:
    /// Empty ChunkSpan
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan() = default;

    /** Constructs a new ChunkSpan by copy, yields a new view to the same data
     * @param other the ChunkSpan to copy
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan(ChunkSpan const& other) = default;

    /** Constructs a new ChunkSpan by move
     * @param other the ChunkSpan to move
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan(ChunkSpan&& other) noexcept = default;

    /** Forbids to construct a ChunkSpan from a rvalue of type Chunk.
     */
    template <class OElementType, class Allocator>
    ChunkSpan(Chunk<OElementType, SupportType, Allocator>&& other) noexcept
        requires(std::is_same_v<typename Allocator::memory_space, MemorySpace>)
    = delete;

    /** Constructs a new ChunkSpan from a Chunk, yields a new view to the same data
     * @param other the Chunk to view
     */
    template <class OElementType, class Allocator>
    KOKKOS_FUNCTION constexpr explicit ChunkSpan(
            Chunk<OElementType, SupportType, Allocator>& other) noexcept
        requires(std::is_same_v<typename Allocator::memory_space, MemorySpace>)
        : base_type(other.m_allocation_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan from a Chunk, yields a new view to the same data
     * @param other the Chunk to view
     */
    // Disabled in the case of `ElementType` is not `const` to avoid write access
    template <class OElementType, class Allocator>
    KOKKOS_FUNCTION constexpr explicit ChunkSpan(
            Chunk<OElementType, SupportType, Allocator> const& other) noexcept
        requires(
                std::is_const_v<ElementType>
                && std::is_same_v<typename Allocator::memory_space, MemorySpace>)
        : base_type(other.m_allocation_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan by copy of a chunk, yields a new view to the same data
     * @param other the ChunkSpan to move
     */
    template <class OElementType, class OLayout>
    KOKKOS_FUNCTION constexpr explicit ChunkSpan(
            ChunkSpan<OElementType, SupportType, OLayout, MemorySpace> const& other) noexcept
        requires(std::constructible_from<
                 allocation_mdspan_type,
                 Kokkos::mdspan<OElementType, extents_type, OLayout>>)
        : base_type(other.m_allocation_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan from scratch
     * @param ptr the allocation pointer to the data
     * @param domain the domain that sustains the view
     */
    KOKKOS_FUNCTION constexpr ChunkSpan(ElementType* const ptr, SupportType const& domain)
        requires(std::is_constructible_v<mapping_type, extents_type>)
        : base_type(ptr, domain)
    {
    }

    /** Constructs a new ChunkSpan from scratch
     * @param allocation_mdspan the allocation mdspan to the data
     * @param domain the domain that sustains the view
     */
    KOKKOS_FUNCTION constexpr ChunkSpan(
            allocation_mdspan_type allocation_mdspan,
            SupportType const& domain)
        : base_type(allocation_mdspan, domain)
    {
        if constexpr (SupportType::rank() > 0) {
            for (std::size_t i = 0; i < SupportType::rank(); ++i) {
                KOKKOS_ASSERT(
                        allocation_mdspan.extent(i)
                        == static_cast<std::size_t>(detail::array(domain.extents())[i]))
            }
        }
    }

    /** Constructs a new ChunkSpan from scratch
     * @param view the Kokkos view
     * @param domain the domain that sustains the view
     */
    template <class KokkosView>
    KOKKOS_FUNCTION constexpr ChunkSpan(KokkosView const& view, SupportType const& domain) noexcept
        requires(Kokkos::is_view_v<KokkosView>)
        : ChunkSpan(
                  detail::build_mdspan(view, std::make_index_sequence<SupportType::rank()> {}),
                  domain)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION ~ChunkSpan() noexcept = default;

    /** Copy-assigns a new value to this ChunkSpan, yields a new view to the same data
     * @param other the ChunkSpan to copy
     * @return *this
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan& operator=(ChunkSpan const& other) = default;

    /** Move-assigns a new value to this ChunkSpan
     * @param other the ChunkSpan to move
     * @return *this
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan& operator=(ChunkSpan&& other) noexcept = default;

    /** Slice out some dimensions
     * @param[in] slice_spec The offsets, from the front of the support, at which to slice along
     * the dimensions `QueryDDims`.
     * @return A ChunkSpan on the same data, without the dimensions `QueryDDims`.
     * @pre Each component of `slice_spec` is smaller than the extent of the corresponding dimension
     * (checked by `KOKKOS_ASSERT`).
     */
    template <concepts::discrete_dimension... QueryDDims>
    KOKKOS_FUNCTION constexpr auto operator[](DiscreteVector<QueryDDims...> const& slice_spec) const
    {
        KOKKOS_ASSERT(
                ((DiscreteVector<QueryDDims>(slice_spec)
                  < DiscreteVector<QueryDDims>(this->m_domain.extents()))
                 && ...))
        Slicer<to_type_seq_t<SupportType>> const slicer;
        auto subview = slicer(this->allocation_mdspan(), slice_spec);
        using layout_type = decltype(subview)::layout_type;
        using mapping_type = decltype(subview)::mapping_type;
        using extents_type = decltype(subview)::extents_type;
        using OutTypeSeqDDims
                = type_seq_remove_t<to_type_seq_t<SupportType>, TypeSeq<QueryDDims...>>;
        using OutDDom = detail::rebind_t<SupportType, OutTypeSeqDDims>;
        if constexpr (
                std::is_same_v<layout_type, Kokkos::layout_left>
                || std::is_same_v<layout_type, Kokkos::layout_right>
                || std::is_same_v<layout_type, Kokkos::layout_stride>) {
            return ChunkSpan<
                    ElementType,
                    OutDDom,
                    layout_type,
                    memory_space>(subview, OutDDom(this->m_domain));
        } else {
            static_assert(mapping_type::is_always_strided() && mapping_type::is_always_unique());
            Kokkos::layout_stride::mapping<extents_type> const mapping_stride(subview.mapping());
            Kokkos::mdspan<ElementType, extents_type, Kokkos::layout_stride> const
                    a(subview.data_handle(), mapping_stride);
            return ChunkSpan<
                    ElementType,
                    OutDDom,
                    Kokkos::layout_stride,
                    memory_space>(a, OutDDom(this->m_domain));
        }
    }

    /** Slice out some dimensions
     * @param[in] slice_spec The position, in the support, at which to slice along the dimensions
     * `QueryDDims`.
     * @return A ChunkSpan on the same data, without the dimensions `QueryDDims`.
     * @pre `slice_spec` belongs to the support (checked by `KOKKOS_ASSERT`).
     */
    template <concepts::discrete_dimension... QueryDDims>
    KOKKOS_FUNCTION constexpr auto operator[](
            DiscreteElement<QueryDDims...> const& slice_spec) const
    {
        using QueryDDom = detail::rebind_t<SupportType, TypeSeq<QueryDDims...>>;
        KOKKOS_ASSERT(QueryDDom(this->m_domain).contains(slice_spec))
        return (*this)[QueryDDom(this->m_domain).distance_from_front(slice_spec)];
    }

    /** Restrict to a subdomain, only valid when SupportType is a DiscreteDomain
     * @param[in] odomain The subdomain to which the view is restricted.
     * @return A ChunkSpan on the same data, defined on the restriction of the support to
     * `odomain`.
     * @pre `odomain` is empty or its front and back belong to the support (checked by
     * `KOKKOS_ASSERT`).
     */
    template <concepts::discrete_dimension... QueryDDims>
    KOKKOS_FUNCTION constexpr auto operator[](DiscreteDomain<QueryDDims...> const& odomain) const
        requires(is_discrete_domain_v<SupportType>)
    {
        KOKKOS_ASSERT(
                odomain.empty()
                || (DiscreteDomain<QueryDDims...>(this->m_domain).contains(odomain.front())
                    && DiscreteDomain<QueryDDims...>(this->m_domain).contains(odomain.back())))
        Slicer<to_type_seq_t<SupportType>> const slicer;
        auto subview = slicer(this->allocation_mdspan(), odomain, this->m_domain);
        using layout_type = decltype(subview)::layout_type;
        using mapping_type = decltype(subview)::mapping_type;
        using extents_type = decltype(subview)::extents_type;
        if constexpr (
                std::is_same_v<layout_type, Kokkos::layout_left>
                || std::is_same_v<layout_type, Kokkos::layout_right>
                || std::is_same_v<layout_type, Kokkos::layout_stride>) {
            return ChunkSpan<
                    ElementType,
                    decltype(this->m_domain.restrict_with(odomain)),
                    layout_type,
                    memory_space>(subview, this->m_domain.restrict_with(odomain));
        } else {
            static_assert(mapping_type::is_always_strided() && mapping_type::is_always_unique());
            Kokkos::layout_stride::mapping<extents_type> const mapping_stride(subview.mapping());
            Kokkos::mdspan<ElementType, extents_type, Kokkos::layout_stride> const
                    a(subview.data_handle(), mapping_stride);
            return ChunkSpan<
                    ElementType,
                    decltype(this->m_domain.restrict_with(odomain)),
                    Kokkos::layout_stride,
                    memory_space>(a, this->m_domain.restrict_with(odomain));
        }
    }

    /** Element access using a list of DiscreteElement
     * @param delems discrete elements
     * @return reference to this element
     */
    template <concepts::discrete_element... DElems>
    KOKKOS_FUNCTION constexpr reference operator()(DElems const&... delems) const noexcept
    {
        static_assert(
                SupportType::rank() == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        KOKKOS_ASSERT(this->m_domain.contains(delems...))
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(this->m_domain.distance_from_front(delems...)));
    }

    /** Element access using a list of DiscreteVector
     * @param dvects discrete vectors
     * @return reference to this element
     */
    template <concepts::discrete_vector... DVects>
    KOKKOS_FUNCTION constexpr reference operator()(DVects const&... dvects) const noexcept
        requires(sizeof...(DVects) != 0)
    {
        static_assert(
                SupportType::rank() == (0 + ... + DVects::size()),
                "Invalid number of dimensions");
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(discrete_vector_type(dvects...)));
    }

    /** Access to the underlying allocation pointer
     * @return allocation pointer
     */
    KOKKOS_FUNCTION constexpr ElementType* data_handle() const
    {
        return base_type::data_handle();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    KOKKOS_FUNCTION constexpr allocation_mdspan_type allocation_mdspan() const
    {
        return base_type::allocation_mdspan();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    KOKKOS_FUNCTION constexpr auto allocation_kokkos_view() const
    {
        auto s = this->allocation_mdspan();
        auto kokkos_layout = detail::build_kokkos_layout(
                s.extents(),
                s.mapping(),
                std::make_index_sequence<SupportType::rank()> {});
        return Kokkos::View<
                detail::mdspan_to_kokkos_element_t<ElementType, SupportType::rank()>,
                decltype(kokkos_layout),
                MemorySpace>(s.data_handle(), kokkos_layout);
    }

    //! @brief Gives a read-only view on the same data.
    //!
    //! @return A ChunkSpan with `const` elements viewing the same data.
    KOKKOS_FUNCTION constexpr view_type span_cview() const
    {
        return view_type(*this);
    }

    //! @brief Gives a view on the same data.
    //!
    //! @return A copy of this ChunkSpan.
    KOKKOS_FUNCTION constexpr span_type span_view() const
    {
        return *this;
    }
};

//! @brief Deduces a ChunkSpan type from a `Kokkos::View` and a support.
//!
//! @param[in] view The `Kokkos::View` to view.
//! @param[in] domain The support on which the values are defined.
//! @return The ChunkSpan type deduced from the `Kokkos::View` type (element type, layout and
//! memory space) and from the support type.
//! @ingroup ddc_containers
template <class DataType, class... Properties, class SupportType>
KOKKOS_DEDUCTION_GUIDE ChunkSpan(
        Kokkos::View<DataType, Properties...> const& view,
        SupportType domain)
        -> ChunkSpan<
                detail::kokkos_to_mdspan_element_t<
                        typename Kokkos::View<DataType, Properties...>::data_type>,
                SupportType,
                detail::kokkos_to_mdspan_layout_t<
                        typename Kokkos::View<DataType, Properties...>::array_layout>,
                typename Kokkos::View<DataType, Properties...>::memory_space>;

//! @brief Deduces the ChunkSpan type viewing a (non-const) Chunk.
//!
//! @param[in] other The Chunk to view.
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class Allocator>
ChunkSpan(Chunk<ElementType, SupportType, Allocator>& other) -> ChunkSpan<
        ElementType,
        SupportType,
        Kokkos::layout_right,
        typename Allocator::memory_space>;

//! @brief Deduces the read-only ChunkSpan type viewing a `const` Chunk.
//!
//! @param[in] other The Chunk to view.
//! @ingroup ddc_containers
template <class ElementType, class SupportType, class Allocator>
ChunkSpan(Chunk<ElementType, SupportType, Allocator> const& other) -> ChunkSpan<
        ElementType const,
        SupportType,
        Kokkos::layout_right,
        typename Allocator::memory_space>;

//! @brief Read-only ChunkSpan, i.e. a ChunkSpan whose elements are `const`.
//!
//! @tparam ElementType The type of the elements (`const` is added).
//! @tparam SupportType The type of the support on which the values are defined.
//! @tparam LayoutStridedPolicy The layout policy of the view.
//! @tparam MemorySpace The Kokkos memory space in which the data lives.
//! @ingroup ddc_containers
template <
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy = Kokkos::layout_right,
        class MemorySpace = Kokkos::HostSpace>
using ChunkView = ChunkSpan<ElementType const, SupportType, LayoutStridedPolicy, MemorySpace>;

} // namespace ddc
