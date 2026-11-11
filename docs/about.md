# About {#mainpage}

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

%DDC is the discrete domain computation library, a C++20 library that aims to offer to the C++/MPI
world an equivalent to the
[`xarray.DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html)/[`dask.Array`](https://docs.dask.org/en/stable/array.html)
python environment.
Where these two libraries are based on [numpy](https://numpy.org/), DDC relies on
[Kokkos](https://github.com/kokkos/kokkos) and
[mdspan](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0009r10.html) to offer
CPU/GPU performance-portable multi-dimensional arrays and iterators.

DDC aims to offer abstractions to represent:

* tagged continuous dimensions,
* discretizations of these (multiple kinds of meshes, function spaces, Fourier, etc.),
* domains in these discretized dimensions,
* associating data to the elements of these discrete domains,
* efficient distribution and iteration over this data.

All these abstractions are handled at compilation using C++ template meta-programming to ensure zero
overhead at execution and enable developers to change their design choices (*eg.* from a regular
mesh to a non-uniform one) with performance portability.

## Features

* **Strongly typed indices.** Indices (ddc::DiscreteElement) and offsets (ddc::DiscreteVector) carry
  the discrete dimension they belong to, so indices of different dimensions cannot be mixed up.
* **Discretizations.** Uniform (ddc::UniformPointSampling), non-uniform
  (ddc::NonUniformPointSampling) and periodic (ddc::PeriodicSampling) samplings of continuous
  dimensions, registered once with ddc::init_discrete_space.
* **Domains.** Dense (ddc::DiscreteDomain), strided (ddc::StridedDiscreteDomain) and sparse
  (ddc::SparseDiscreteDomain) multi-dimensional domains.
* **Containers.** Owning arrays (ddc::Chunk) and non-owning views (ddc::ChunkSpan) indexed by
  domain elements and backed by Kokkos memory spaces.
* **Performance-portable algorithms.** ddc::parallel_for_each, ddc::parallel_transform_reduce,
  ddc::parallel_fill, ddc::parallel_deepcopy and their sequential counterparts, built on Kokkos.
* **Optional components.** Fourier transforms (`DDC::fft`), spline interpolation
  (`DDC::splines`) and I/O through PDI (`DDC::pdi`).

## Requirements

DDC core (`DDC::core`) requires:

* a C++20 compiler (the CMake targets require `cxx_std_20`; CI also builds with C++23),
* CMake 3.25 or later,
* Kokkos 4.4 or later, before 6.

The optional components add the following requirements:

| Component      | CMake option                 | Dependencies                                  |
|----------------|------------------------------|-----------------------------------------------|
| `DDC::fft`     | `DDC_BUILD_KERNELS_FFT`      | Kokkos-fft 0.3 or later, before 3             |
| `DDC::splines` | `DDC_BUILD_KERNELS_SPLINES`  | Ginkgo 1.9 or later, Kokkos Kernels 4.7 or later before 6, LAPACKE |
| `DDC::pdi`     | `DDC_BUILD_PDI_WRAPPER`      | PDI 1.10.1 or later, before 2                 |

The continuous integration of the project builds and tests DDC on Linux (CPU, CUDA and HIP
backends), macOS and Windows. See \ref installation "the installation guide" for details on how to
obtain these dependencies.

## Installation and compilation

DDC is built and installed with CMake, and consumed from another CMake project with
`find_package(DDC)`. The \ref installation "installation guide" describes a build from source and an
installation with Spack; \ref cmake_integration "CMake integration" describes how to link a project
against the `DDC::core`, `DDC::fft`, `DDC::splines` and `DDC::pdi` targets.

## Quick start

The following fragments are taken from `examples/uniform_heat_equation.cpp`, which is compiled as
part of the DDC build. The full tutorial is \ref first_steps "the uniform heat equation".

Include DDC and Kokkos:

\snippet{trimleft} uniform_heat_equation.cpp includes

Declare a continuous dimension as an empty tag type, and a discrete dimension that samples it:

\snippet{trimleft} uniform_heat_equation.cpp X-dimension

\snippet{trimleft} uniform_heat_equation.cpp X-discretization

Initialize the discrete dimension with ddc::init_discrete_space. Here the sampling is built with
ghost points, and the call returns the main domain, the domain including ghosts, and the two ghost
domains:

\snippet{trimleft} uniform_heat_equation.cpp X-global-domain

Allocate an owning, two-dimensional array in device memory on the product of the X and Y domains
(`ghosted_y_domain` is obtained in the same way as `ghosted_x_domain`):

\snippet{trimleft} uniform_heat_equation.cpp data allocation

## Architecture and public API

The \ref architecture "architecture overview" describes the modules of the library and how they
depend on each other. The public API is organized in the following modules:

* \ref ddc_core_utils
* \ref ddc_domains
* \ref ddc_samplings
* \ref ddc_containers
* \ref ddc_algorithms
* \ref ddc_output
* \ref ddc_experimental
* \ref ddc_fft
* \ref ddc_splines
* \ref ddc_pdi

The conventions that apply to the whole API (errors, ownership of data, threads) are described in
\ref error_handling_and_ownership "error handling, ownership and thread safety".

## Compatibility and limitations

* DDC is versioned according to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and its
  changes are recorded in the
  [changelog](https://github.com/CExA-project/ddc/blob/main/CHANGELOG.md).
* Before version 1.0, the installed CMake package only declares compatibility within the same minor
  version (`SameMinorVersion`): a project that requires DDC 0.17 does not accept DDC 0.18.
* Code that is deprecated is only compiled when the CMake option `DDC_BUILD_DEPRECATED_CODE` is
  `ON` (the default).
* The API of the namespace `ddc::experimental`, listed in \ref ddc_experimental, is not covered by
  any stability guarantee.
* Kokkos 4.5.0 embeds a version of mdspan that is not compatible with DDC. This issue is fixed in
  Kokkos 4.5.1.

## License

DDC is distributed under the [MIT License](https://github.com/CExA-project/ddc/blob/main/LICENSES/MIT.txt).
A small number of files do not originate from the DDC developers and carry their own license in
their header, as described in
[COPYRIGHT.md](https://github.com/CExA-project/ddc/blob/main/COPYRIGHT.md).

## Discover more

* First, you should \ref first_steps "have a look at our commented example / tutorial".
* To know more, you can [delve in the reference documentation](annotated.html).
* [Use the source, Luke](https://github.com/CExA-project/ddc).
* [You should definitely come see us on Slack](https://ddc-lib.slack.com/join/shared_invite/zt-33v61oiip-UHnWCavFC0cmff5a94HYwQ).

## Join us

If you like the project,
[leave us a ⭐star⭐ on Github](https://github.com/CExA-project/ddc).
And of course, we would love to talk with you on our [Slack channel ![Slack](slack.png) https://ddc-lib.slack.com/](https://ddc-lib.slack.com/join/shared_invite/zt-33v61oiip-UHnWCavFC0cmff5a94HYwQ).
