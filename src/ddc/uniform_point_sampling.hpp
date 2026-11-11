// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//! @file
//! @brief Provides ddc::UniformPointSampling, a discretization of a dimension into regularly
//! spaced points, and its free functions.

#pragma once

#include <cassert>
#include <iosfwd>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "coordinate.hpp"
#include "discrete_dimension.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_space.hpp"
#include "discrete_vector.hpp"
#include "real_type.hpp"

namespace ddc {

namespace detail {

struct UniformPointSamplingBase
{
};

void print_uniform_point_sampling(std::ostream& os, CoordinateElement origin, Real step);

} // namespace detail

/** UniformPointSampling models a uniform discretization of the provided continuous dimension
 *
 * @details The coordinate of the discrete element `e` is `origin + (e - front) * step`, where
 * `front` is the reference element of the discrete dimension.
 *
 * @tparam CDim The continuous dimension (tag) that is discretized.
 *
 * @ingroup ddc_samplings
 */
template <class CDim>
class UniformPointSampling
    : detail::UniformPointSamplingBase
    , DiscreteDimension
{
public:
    //! @brief The continuous dimension that is discretized.
    using continuous_dimension_type = CDim;

    //! @brief The discrete dimension type itself, used to resolve the implementation class.
    using discrete_dimension_type = UniformPointSampling;

public:
    //! @brief Data and accessors of the sampling in a given memory space.
    //!
    //! @details The sampling is defined by the coordinate of the reference element (`origin`) and
    //! the constant `step` between two consecutive elements.
    //!
    //! @tparam DDim The discrete dimension deriving from UniformPointSampling.
    //! @tparam MemorySpace The Kokkos memory space in which the implementation is stored.
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

    private:
        Coordinate<CDim> m_origin;

        Real m_step;

        DiscreteElement<DDim> m_reference;

    public:
        //! @brief The discrete dimension type itself.
        using discrete_dimension_type = UniformPointSampling;

        //! @brief The type of the domains of this discrete dimension.
        using discrete_domain_type = DiscreteDomain<DDim>;

        //! @brief The type of the elements of this discrete dimension.
        using discrete_element_type = DiscreteElement<DDim>;

        //! @brief The type of the vectors of this discrete dimension.
        using discrete_vector_type = DiscreteVector<DDim>;

        //! @brief Constructs a sampling of origin 0 and step 1.
        Impl() noexcept
            : m_origin(0)
            , m_step(1)
            , m_reference(create_reference_discrete_element<DDim>())
        {
        }

        //! @brief Copy constructor (deleted).
        Impl(Impl const&) = delete;

        //! @brief Constructs a copy of a sampling stored in another memory space.
        //!
        //! @param[in] impl The sampling to copy, whose origin, step and reference element are
        //! copied.
        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl)
            : m_origin(impl.m_origin)
            , m_step(impl.m_step)
            , m_reference(impl.m_reference)
        {
        }

        //! @brief Move constructor.
        Impl(Impl&&) = default;

        /** @brief Construct a `Impl` from a point and a spacing step.
         *
         * @param origin the real coordinate of mesh coordinate 0
         * @param step   the real distance between two points of mesh distance 1
         *
         * @pre `step` must be strictly positive (checked with `assert`).
         */
        Impl(Coordinate<CDim> origin, Real step)
            : m_origin(origin)
            , m_step(step)
            , m_reference(create_reference_discrete_element<DDim>())
        {
            assert(step > 0);
        }

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

        /// @brief Lower bound index of the mesh
        ///
        /// @details Actually returns the coordinate of the reference element (see front()),
        /// that is the `origin` given at construction.
        ///
        /// @return The coordinate of the reference element of the sampling.
        KOKKOS_FUNCTION Coordinate<CDim> origin() const noexcept
        {
            return m_origin;
        }

        /// @brief Lower bound index of the mesh
        ///
        /// @return The reference discrete element, whose coordinate is origin().
        KOKKOS_FUNCTION discrete_element_type front() const noexcept
        {
            return m_reference;
        }

        /// @brief Spacing step of the mesh
        ///
        /// @return The distance between the coordinates of two consecutive elements.
        KOKKOS_FUNCTION Real step() const
        {
            return m_step;
        }

        /// @brief Convert a mesh index into a position in `CDim`
        ///
        /// @param[in] icoord The discrete element to convert.
        /// @return The coordinate of `icoord`.
        KOKKOS_FUNCTION Coordinate<CDim> coordinate(
                discrete_element_type const& icoord) const noexcept
        {
            return m_origin + Coordinate<CDim>((icoord - front()) * m_step);
        }
    };

    /** Construct a Impl<Kokkos::HostSpace> and associated discrete_domain_type from a segment
     *  \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
     *  Note that there is no guarantee that either the boundaries a or b will be exactly represented in the sampling.
     *  One should expect usual floating point rounding errors.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n number of points to map on the segment \f$[a, b]\f$ including a & b
     *
     * @return A tuple of the sampling implementation (of step `(b - a) / (n - 1)`) and the
     * discrete domain of `n` points starting at its `front()`.
     *
     * @pre `a < b` and `n > 1` (checked with `assert`).
     */
    template <concepts::discrete_dimension DDim>
    static std::tuple<typename DDim::template Impl<DDim, Kokkos::HostSpace>, DiscreteDomain<DDim>>
    init(Coordinate<CDim> a, Coordinate<CDim> b, DiscreteVector<DDim> n)
    {
        assert(a < b);
        assert(n > 1);
        typename DDim::template Impl<DDim, Kokkos::HostSpace>
                disc(a, Coordinate<CDim>((b - a) / (n - 1)));
        DiscreteDomain<DDim> domain(disc.front(), n);
        return std::make_tuple(std::move(disc), std::move(domain));
    }

    /** Construct a uniform `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
     *  number of points `n`.
     *  Note that there is no guarantee that either the boundaries a or b will be exactly represented in the sampling.
     *  One should expect usual floating point rounding errors.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     * @param n_ghosts_before number of additional "ghost" points before the segment
     * @param n_ghosts_after number of additional "ghost" points after the segment
     *
     * @return A tuple of, in this order: the sampling implementation (of step `(b - a) / (n - 1)`
     * and whose origin is `a` shifted by the ghost points before), the main domain of `n` points,
     * the whole domain including ghost points, the domain of the ghost points before and the
     * domain of the ghost points after.
     *
     * @pre `a < b` and `n > 1` (checked with `assert`).
     */
    template <concepts::discrete_dimension DDim>
    static std::tuple<
            typename DDim::template Impl<DDim, Kokkos::HostSpace>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>>
    init_ghosted(
            Coordinate<CDim> a,
            Coordinate<CDim> b,
            DiscreteVector<DDim> n,
            DiscreteVector<DDim> n_ghosts_before,
            DiscreteVector<DDim> n_ghosts_after)
    {
        assert(a < b);
        assert(n > 1);
        Real const discretization_step = (b - a) / (n - 1);
        typename DDim::template Impl<DDim, Kokkos::HostSpace>
                disc(a - n_ghosts_before.value() * discretization_step, discretization_step);
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

    /** Construct a uniform `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
     *  number of points `n`.
     *  Note that there is no guarantee that either the boundaries a or b will be exactly represented in the sampling.
     *  One should expect usual floating point rounding errors.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     * @param n_ghosts number of additional "ghost" points before and after the segment
     *
     * @return Same tuple as the overload with separate numbers of ghost points before and after.
     *
     * @pre `a < b` and `n > 1` (checked with `assert`).
     */
    template <concepts::discrete_dimension DDim>
    static std::tuple<
            typename DDim::template Impl<DDim, Kokkos::HostSpace>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>>
    init_ghosted(
            Coordinate<CDim> a,
            Coordinate<CDim> b,
            DiscreteVector<DDim> n,
            DiscreteVector<DDim> n_ghosts)
    {
        return init_ghosted(a, b, n, n_ghosts, n_ghosts);
    }
};

//! @brief Trait that is true if `T` is a UniformPointSampling (or derives from one).
//!
//! @tparam T The type to check.
//!
//! @ingroup ddc_samplings
template <class T>
struct is_uniform_point_sampling : public std::is_base_of<detail::UniformPointSamplingBase, T>::type
{
};

//! @brief Shortcut for `is_uniform_point_sampling<T>::value`.
//!
//! @tparam T The type to check.
//!
//! @ingroup ddc_samplings
template <class T>
constexpr bool is_uniform_point_sampling_v = is_uniform_point_sampling<T>::value;

namespace concepts {

//! @brief Concept satisfied by discrete dimensions that are a UniformPointSampling.
//!
//! @tparam T The type to check.
//!
//! @ingroup ddc_samplings
template <class T>
concept uniform_point_sampling = discrete_dimension<T> && is_uniform_point_sampling_v<T>;

}

//! @brief Prints the origin and step of a uniform sampling implementation to a stream.
//!
//! @param[in,out] os The output stream.
//! @param[in] mesh The sampling implementation to print.
//! @return `os`.
//!
//! @ingroup ddc_samplings
template <class DDimImpl>
std::ostream& operator<<(std::ostream& os, DDimImpl const& mesh)
    requires(concepts::uniform_point_sampling<typename DDimImpl::discrete_dimension_type>)
{
    print_uniform_point_sampling(os, mesh.origin(), mesh.step());
    return os;
}

//! @brief Gets the coordinate of a discrete element of a uniform sampling.
//!
//! @param[in] c The discrete element.
//! @return The coordinate of `c`, computed with the discrete space of `DDim`.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION constexpr Coordinate<typename DDim::continuous_dimension_type> coordinate(
        DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().coordinate(c);
}

/// @brief Lower bound index of the mesh
///
/// @tparam DDim The uniform sampling discrete dimension.
/// @return The coordinate of the reference element of the discrete space of `DDim`.
/// @pre The discrete space of `DDim` must be initialized.
///
/// @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> origin() noexcept
{
    return discrete_space<DDim>().origin();
}

/// @brief Lower bound index of the mesh
///
/// @tparam DDim The uniform sampling discrete dimension.
/// @return The reference discrete element of the discrete space of `DDim`.
/// @pre The discrete space of `DDim` must be initialized.
///
/// @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION DiscreteElement<DDim> front() noexcept
{
    return discrete_space<DDim>().front();
}

/// @brief Spacing step of the mesh
///
/// @tparam DDim The uniform sampling discrete dimension.
/// @return The distance between two consecutive points of the sampling.
/// @pre The discrete space of `DDim` must be initialized.
///
/// @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Real step() noexcept
{
    return discrete_space<DDim>().step();
}

//! @brief Distance to the previous point of a uniform sampling.
//!
//! @return The step of the discrete space of `DDim`, whatever the (unnamed) element argument.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> distance_at_left(
        DiscreteElement<DDim>)
{
    return Coordinate<typename DDim::continuous_dimension_type>(step<DDim>());
}

//! @brief Distance to the next point of a uniform sampling.
//!
//! @return The step of the discrete space of `DDim`, whatever the (unnamed) element argument.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> distance_at_right(
        DiscreteElement<DDim>)
{
    return Coordinate<typename DDim::continuous_dimension_type>(step<DDim>());
}

//! @brief Coordinate of the first point of a discrete domain.
//!
//! @param[in] d A non-empty discrete domain of a uniform sampling.
//! @return The coordinate of `d.front()`.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmin(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.front());
}

//! @brief Coordinate of the last point of a discrete domain.
//!
//! @param[in] d A non-empty discrete domain of a uniform sampling.
//! @return The coordinate of `d.back()`.
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmax(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.back());
}

//! @brief Length of the segment spanned by a discrete domain.
//!
//! @param[in] d A non-empty discrete domain of a uniform sampling.
//! @return The difference between rmax(d) and rmin(d).
//! @pre The discrete space of `DDim` must be initialized.
//!
//! @ingroup ddc_samplings
template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rlength(
        DiscreteDomain<DDim> const& d)
{
    return rmax(d) - rmin(d);
}

} // namespace ddc
