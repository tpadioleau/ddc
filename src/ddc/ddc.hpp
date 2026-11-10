// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#    if !defined(KOKKOS_ENABLE_CUDA_CONSTEXPR)
static_assert(false, "DDC requires option -DKokkos_ENABLE_CUDA_CONSTEXPR=ON");
#    endif

#    if !defined(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
static_assert(false, "DDC requires option -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON");
#    endif
#endif

#if defined(KOKKOS_ENABLE_HIP)
#    if !defined(KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
static_assert(false, "DDC requires option -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON");
#    endif
#endif

#if defined(KOKKOS_ENABLE_SYCL)
#    if !defined(KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE)
static_assert(false, "DDC requires option -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON");
#    endif
#endif

//! @brief The top-level namespace of DDC.
//! All DDC symbols are defined either in this namespace or in a nested namespace.
namespace ddc {
}

//! @file
//! @brief Umbrella header of the DDC core library, including all its public headers.
//!
//! Including this single header gives access to the whole `DDC::core` API. The optional components
//! (`DDC::fft`, `DDC::splines` and `DDC::pdi`) must be included separately.
//! @ingroup ddc_core_utils

//! @defgroup ddc_core_utils Type system and core utilities
//! @brief Compile-time type lists, tag utilities, the floating-point type selection and
//! the scope guard managing the lifetime of the global discretization store.

//! @defgroup ddc_containers Data containers and memory
//! @brief Multi-dimensional containers associating values to the elements of a discrete domain,
//! and the allocators used to manage their storage.

//! @defgroup ddc_discretizations Discretizations
//! @brief Discrete dimensions, their sampling of a continuous dimension, and the discrete domains
//! built on top of them.

//! @defgroup ddc_domains Discrete elements, vectors and domains
//! @ingroup ddc_discretizations
//! @brief Discrete elements, discrete vectors and the discrete domains that gather them.

//! @defgroup ddc_samplings Samplings and discrete space
//! @ingroup ddc_discretizations
//! @brief Discrete dimensions, their coordinates, and the global discrete space storing their
//! sampling.

//! @defgroup ddc_algorithms Algorithms
//! @brief Sequential and parallel algorithms iterating over discrete domains and containers.

//! @defgroup ddc_output Output
//! @brief Printing of containers and export to the NumPy format.

//! @defgroup ddc_experimental Experimental API
//! @brief Symbols defined in the `ddc::experimental` namespace.
//!
//! @warning The repository does not define a stability or compatibility policy for the
//! `ddc::experimental` namespace. These symbols must not be assumed to follow the compatibility
//! expectations of the rest of the library and may change between releases.

//! @defgroup ddc_fft Fourier transform (DDC::fft)
//! @brief Fast Fourier transform of containers, built on top of Kokkos-fft.
//!
//! This component is optional and provided by the `DDC::fft` CMake target.

//! @defgroup ddc_splines Splines (DDC::splines)
//! @brief B-splines, spline builders and evaluators, and the associated linear solvers.
//!
//! This component is optional and provided by the `DDC::splines` CMake target.

//! @defgroup ddc_pdi PDI wrapper (DDC::pdi)
//! @brief Exposure of DDC containers to PDI.
//!
//! This component is optional and provided by the `DDC::pdi` CMake target.

// Misc
#include <ddc/config.hpp>

#include "detail/dual_discretization.hpp"
#include "detail/kokkos.hpp"
#include "detail/macros.hpp"
#include "detail/tagged_vector.hpp"
#include "detail/tuple/tuple.hpp"
#include "detail/utils.hpp"

#include "ddc_to_kokkos_execution_policy.hpp"
#include "real_type.hpp"
#include "scope_guard.hpp"
#include "tagged_type_traits.hpp"
#include "type_seq.hpp"

// Containers
#include "aligned_allocator.hpp"
#include "chunk.hpp"
#include "chunk_common.hpp"
#include "chunk_span.hpp"
#include "chunk_traits.hpp"
#include "kokkos_allocator.hpp"

// Discretizations
#include "coordinate.hpp"
#include "discrete_dimension.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_space.hpp"
#include "discrete_vector.hpp"
#include "non_uniform_point_sampling.hpp"
#include "periodic_sampling.hpp"
#include "sparse_discrete_domain.hpp"
#include "strided_discrete_domain.hpp"
#include "trivial_space.hpp"
#include "uniform_point_sampling.hpp"

// Algorithms
#include "create_mirror.hpp"
#include "for_each.hpp"
#include "for_each_block.hpp"
#include "parallel_copy.hpp"
#include "parallel_deepcopy.hpp"
#include "parallel_fill.hpp"
#include "parallel_for_each.hpp"
#include "parallel_for_each_team.hpp"
#include "parallel_transform.hpp"
#include "parallel_transform_reduce.hpp"
#include "parallel_transform_scan.hpp"
#include "reducer.hpp"
#include "transform_reduce.hpp"

// Output
#include "print.hpp"
#include "save_npy.hpp"
