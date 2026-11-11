// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ddc::NonUniformPointSampling, a discretization of a dimension into points at
//! arbitrary sorted coordinates, and its free functions.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "coordinate.hpp"
#include "discrete_dimension.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_space.hpp"
#include "discrete_vector.hpp"

namespace ddc {

namespace detail {

struct NonUniformPointSamplingBase
{
};

void print_non_uniform_point_samplig(std::ostream& os, std::size_t size);

} // namespace detail

/// `NonUniformPointSampling` models a non-uniform discretization of the `CDim` segment \f$[a, b]\f$.
///
/// @details The coordinates of the points are stored in a sorted array: the coordinate of the
/// discrete element `e` is the one at position `e - front` of that array.
///
/// @tparam CDim The continuous dimension (tag) that is discretized.
///
/// @ingroup ddc_samplings
template <class CDim>
class NonUniformPointSampling
    : detail::NonUniformPointSamplingBase
    , DiscreteDimension
{
public:
    //! @brief The continuous dimension that is discretized.
    using continuous_dimension_type = CDim;

    //! @brief The discrete dimension type itself, used to resolve the implementation class.
    using discrete_dimension_type = NonUniformPointSampling;

public:
    //! @brief Data and accessors of the sampling in a given memory space.
    //!
    //! @details The coordinates of the points are stored in a `Kokkos::View` of the memory space.
    //!
    //! @tparam DDim The discrete dimension deriving from NonUniformPointSampling.
    //! @tparam MemorySpace The Kokkos memory space in which the points are stored.
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

        Kokkos::View<Coordinate<CDim>*, MemorySpace> m_points;

        DiscreteElement<DDim> m_reference;

    public:
        //! @brief The discrete dimension type itself.
        using discrete_dimension_type = NonUniformPointSampling;

        //! @brief The type of the domains of this discrete dimension.
        using discrete_domain_type = DiscreteDomain<DDim>;

        //! @brief The type of the elements of this discrete dimension.
        using discrete_element_type = DiscreteElement<DDim>;

        //! @brief The type of the vectors of this discrete dimension.
        using discrete_vector_type = DiscreteVector<DDim>;

        //! @brief Constructs an empty sampling (no points, default-initialized reference element).
        Impl() = default;

        /// @brief Construct a `NonUniformPointSampling` using a brace-list, i.e. `NonUniformPointSampling mesh({0., 1.})`
        ///
        /// @param[in] points The coordinates of the points, in sorted order.
        /// @throws std::runtime_error If `points` is not sorted.
        Impl(std::initializer_list<Coordinate<CDim>> const points)
            : Impl(points.begin(), points.end())
        {
        }

        /// @brief Construct a `NonUniformPointSampling` using a C++20 "common range".
        ///
        /// @param[in] points The range of the coordinates of the points, in sorted order.
        /// @throws std::runtime_error If `points` is not sorted.
        template <class InputRange>
        explicit Impl(InputRange const& points) : Impl(points.begin(), points.end())
        {
        }

        /// @brief Construct a `NonUniformPointSampling` using a pair of iterators.
        ///
        /// @param[in] points_begin Iterator to the first coordinate.
        /// @param[in] points_end Iterator past the last coordinate.
        /// @throws std::runtime_error If the points are not sorted.
        /// @details The points are copied into a view of `MemorySpace`.
        template <class InputIt>
        Impl(InputIt const points_begin, InputIt const points_end)
            : m_reference(create_reference_discrete_element<DDim>())
        {
            using view_type = Kokkos::View<Coordinate<CDim>*, MemorySpace>;
            if (!std::is_sorted(points_begin, points_end)) {
                throw std::runtime_error("Input points must be sorted");
            }
            // Make a contiguous copy of [points_begin, points_end[
            std::vector<Coordinate<CDim>> host_points(points_begin, points_end);
            m_points = view_type("NonUniformPointSampling::points", host_points.size());
            Kokkos::deep_copy(m_points, view_type(host_points.data(), host_points.size()));
        }

        //! @brief Constructs a copy of a sampling stored in another memory space.
        //!
        //! @param[in] impl The sampling to copy, its points being copied to `MemorySpace`.
        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl)
            : m_points(Kokkos::create_mirror_view_and_copy(MemorySpace(), impl.m_points))
            , m_reference(impl.m_reference)
        {
        }

        //! @brief Copy constructor (deleted).
        //!
        //! @param[in] x The sampling to copy.
        Impl(Impl const& x) = delete;

        //! @brief Move constructor.
        //!
        //! @param[in,out] x The sampling to move from.
        Impl(Impl&& x) = default;

        //! @brief Destructor.
        ~Impl() = default;

        //! @brief Copy assignment operator (deleted).
        //!
        //! @param[in] x The sampling to copy.
        //! @return A reference to this object.
        Impl& operator=(Impl const& x) = delete;

        //! @brief Move assignment operator.
        //!
        //! @param[in,out] x The sampling to move from.
        //! @return A reference to this object.
        Impl& operator=(Impl&& x) = default;

        //! @brief Gets the number of points of the sampling.
        //!
        //! @return The number of stored points.
        KOKKOS_FUNCTION std::size_t size() const
        {
            return m_points.size();
        }

        /// @brief Lower bound index of the mesh
        ///
        /// @return The reference discrete element, whose coordinate is the first point.
        KOKKOS_FUNCTION discrete_element_type front() const noexcept
        {
            return m_reference;
        }

        /// @brief Convert a mesh index into a position in `CDim`
        ///
        /// @param[in] icoord The discrete element to convert.
        /// @return The coordinate of the point at position `icoord - front()`.
        KOKKOS_FUNCTION Coordinate<CDim> coordinate(
                discrete_element_type const& icoord) const noexcept
        {
            return m_points((icoord - front()).value());
        }
    };

    /** Construct an Impl<Kokkos::HostSpace> and associated discrete_domain_type from a range
     * containing the points coordinates along the `DDim` dimension.
     *
     * @param non_uniform_points a range (std::vector, std::array, ...) containing the coordinates of the points of the domain.
     *
     * @tparam InputRange Type of the range (it must provide `empty()` and `size()`).
     *
     * @return A tuple of the sampling implementation and the discrete domain of its points.
     *
     * @pre The range must not be empty (checked with `assert`).
     * @throws std::runtime_error If the coordinates are not sorted.
     */
    template <concepts::discrete_dimension DDim, class InputRange>
    static std::tuple<typename DDim::template Impl<DDim, Kokkos::HostSpace>, DiscreteDomain<DDim>>
    init(InputRange const& non_uniform_points)
    {
        assert(!non_uniform_points.empty());
        DiscreteVector<DDim> const n(non_uniform_points.size());
        typename DDim::template Impl<DDim, Kokkos::HostSpace> disc(non_uniform_points);
        DiscreteDomain<DDim> domain(disc.front(), n);
        return std::make_tuple(std::move(disc), std::move(domain));
    }

    /** Construct 4 non-uniform `DiscreteDomain` and an Impl<Kokkos::HostSpace> from 3 ranges containing the points coordinates along the `DDim` dimension.
     *
     * @param domain_r a range containing the coordinates of the points of the main domain along the DDim position
     * @param pre_ghost_r a range containing the positions of the ghost points before the main domain the DDim position
     * @param post_ghost_r a range containing the positions of the ghost points after the main domain the DDim position
     *
     * @tparam InputRange Type of the ranges (they must be of the same type and provide `empty()`,
     * `size()` and `value_type`).
     *
     * @return A tuple of, in this order: the sampling implementation (holding the points of the
     * pre-ghost, main and post-ghost ranges concatenated), the main domain, the whole domain
     * including ghost points, the domain of the ghost points before and the domain of the ghost
     * points after.
     *
     * @pre `domain_r` must not be empty (checked with `assert`).
     * @throws std::runtime_error If the concatenated coordinates are not sorted.
     */
    template <concepts::discrete_dimension DDim, class InputRange>
    static std::tuple<
            typename DDim::template Impl<DDim, Kokkos::HostSpace>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>>
    init_ghosted(
            InputRange const& domain_r,
            InputRange const& pre_ghost_r,
            InputRange const& post_ghost_r)
    {
        assert(!domain_r.empty());

        DiscreteVector<DDim> const n(domain_r.size());
        DiscreteVector<DDim> const n_ghosts_before(pre_ghost_r.size());
        DiscreteVector<DDim> const n_ghosts_after(post_ghost_r.size());

        std::vector<typename InputRange::value_type> full_domain;

        std::copy(pre_ghost_r.begin(), pre_ghost_r.end(), std::back_inserter(full_domain));
        std::copy(domain_r.begin(), domain_r.end(), std::back_inserter(full_domain));
        std::copy(post_ghost_r.begin(), post_ghost_r.end(), std::back_inserter(full_domain));

        typename DDim::template Impl<DDim, Kokkos::HostSpace> disc(full_domain);

        DiscreteDomain<DDim> ghosted_domain(disc.front(), n + n_ghosts_before + n_ghosts_after);
        DiscreteDomain<DDim> pre_ghost = ghosted_domain.take_first(n_ghosts_before);
        DiscreteDomain<DDim> main_domain = ghosted_domain.remove(n_ghosts_before, n_ghosts_after);
        DiscreteDomain<DDim> post_ghost = ghosted_domain.take_last(n_ghosts_after);
        return std::make_tuple(
                std::move(disc),
                std::move(main_domain),
                std::move(ghosted_domain),
                std::move(pre_ghost),
                std::move(post_ghost));
    }
};

//! @brief Trait that is true if `T` is a NonUniformPointSampling (or derives from one).
//!
//! @tparam T The type to check.
//!
//! @ingroup ddc_samplings
template <class T>
struct is_non_uniform_point_sampling
    : public std::is_base_of<detail::NonUniformPointSamplingBase, T>::type
{
};

//! @brief Shortcut for `is_non_uniform_point_sampling<T>::value`.
//!
//! @tparam T The type to check.
//!
//! @ingroup ddc_samplings
template <class T>
constexpr bool is_non_uniform_point_sampling_v = is_non_uniform_point_sampling<T>::value;

namespace concepts {

//! @brief Concept satisfied by discrete dimensions that are a NonUniformPointSampling.
//!
//! @tparam T The type to check.
//!
//! @ingroup ddc_samplings
template <class T>
concept non_uniform_point_sampling = discrete_dimension<T> && is_non_uniform_point_sampling_v<T>;

} // namespace concepts

//! @brief Prints a short description (the number of points) of a non-uniform sampling.
//!
//! @param[in,out] os The output stream.
//! @param[in] mesh The sampling implementation to print.
//! @return `os`.
//!
//! @ingroup ddc_samplings
template <class DDimImpl>
std::ostream& operator<<(std::ostream& os, DDimImpl const& mesh)
    requires(concepts::non_uniform_point_sampling<typename DDimImpl::discrete_dimension_type>)
{
    detail::print_non_uniform_point_samplig(os, mesh.size());
    return os;
}

//! @brief Gets the coordinate of a discrete element of a non-uniform sampling.
//!
//! @param[in] c The discrete element.
//! @return The coordinate of `c`, read from the discrete space of `DDim`.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::non_uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> coordinate(
        DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().coordinate(c);
}

//! @brief Distance between a point and the previous one.
//!
//! @param[in] i A discrete element that must not be the first one of the sampling.
//! @return `coordinate(i) - coordinate(i - 1)`.
//!
//! @ingroup ddc_samplings
template <concepts::non_uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> distance_at_left(
        DiscreteElement<DDim> i)
{
    return coordinate(i) - coordinate(i - 1);
}

//! @brief Distance between a point and the next one.
//!
//! @param[in] i A discrete element that must not be the last one of the sampling.
//! @return `coordinate(i + 1) - coordinate(i)`.
//!
//! @ingroup ddc_samplings
template <concepts::non_uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> distance_at_right(
        DiscreteElement<DDim> i)
{
    return coordinate(i + 1) - coordinate(i);
}

//! @brief Coordinate of the first point of a discrete domain.
//!
//! @details `d` is a non-empty discrete domain of a non-uniform sampling.
//!
//! @return The coordinate of `d.front()`.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::non_uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmin(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.front());
}

//! @brief Coordinate of the last point of a discrete domain.
//!
//! @details `d` is a non-empty discrete domain of a non-uniform sampling.
//!
//! @return The coordinate of `d.back()`.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::non_uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmax(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.back());
}

//! @brief Length of the segment spanned by a discrete domain.
//!
//! @details `d` is a non-empty discrete domain of a non-uniform sampling.
//!
//! @return The difference between rmax(d) and rmin(d).
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::non_uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rlength(
        DiscreteDomain<DDim> const& d)
{
    return rmax(d) - rmin(d);
}

} // namespace ddc
