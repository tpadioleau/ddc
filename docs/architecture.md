# Architecture overview {#architecture}

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

[TOC]

This page describes how the library is organized: the CMake targets, the modules of the public API,
and how a typical program uses them. The concepts behind the API are introduced in
[DDC concepts](concepts.md).

## CMake targets

DDC is split into one core library and three optional components. Each component is a separate CMake
target, enabled by its own option, and links against `DDC::core`.

\dot
digraph targets {
    rankdir=BT;
    node [shape=box, fontname="Helvetica", fontsize=10];
    edge [fontname="Helvetica", fontsize=9];

    core    [label="DDC::core", URL="\ref ddc_core_utils"];
    fft     [label="DDC::fft", URL="\ref ddc_fft"];
    splines [label="DDC::splines", URL="\ref ddc_splines"];
    pdi     [label="DDC::pdi", URL="\ref ddc_pdi"];

    node [shape=ellipse, style=dashed];
    kokkos  [label="Kokkos"];
    kfft    [label="Kokkos-fft"];
    ginkgo  [label="Ginkgo,\nKokkos Kernels,\nLAPACKE"];
    pdilib  [label="PDI"];

    core    -> kokkos;
    fft     -> core;
    fft     -> kfft;
    splines -> core;
    splines -> ginkgo [label="private"];
    pdi     -> core;
    pdi     -> pdilib;
}
\enddot

| Target         | CMake option                | Public headers                       | Entry header                |
|----------------|-----------------------------|--------------------------------------|-----------------------------|
| `DDC::core`    | always built                | `ddc/*.hpp`                          | `<ddc/ddc.hpp>`             |
| `DDC::fft`     | `DDC_BUILD_KERNELS_FFT`     | `ddc/kernels/fft.hpp`                | `<ddc/kernels/fft.hpp>`     |
| `DDC::splines` | `DDC_BUILD_KERNELS_SPLINES` | `ddc/kernels/splines/*.hpp`          | `<ddc/kernels/splines.hpp>` |
| `DDC::pdi`     | `DDC_BUILD_PDI_WRAPPER`     | `ddc/pdi.hpp`                        | `<ddc/pdi.hpp>`             |

Ginkgo, Kokkos Kernels and LAPACKE are private dependencies of `DDC::splines`: they are used by its
compiled sources only and do not appear in its public headers' interface.

Most of the core library is made of templates defined in headers. The compiled part of `DDC::core`
holds the non-template code (for example the storage behind discrete spaces and ddc::ScopeGuard, the
printing options and the npy header writer). Headers under `ddc/detail/` and entities in a `detail`
namespace are implementation details: they are not part of the public API and are not documented
here.

## Modules of the core library

The public API of `DDC::core` is organized in the following modules. The arrows of the diagram show
which module headers include headers of another module.

\dot
digraph modules {
    rankdir=BT;
    node [shape=box, fontname="Helvetica", fontsize=10];

    utils      [label="Type system and\ncore utilities", URL="\ref ddc_core_utils"];
    domains    [label="Discrete elements,\nvectors and domains", URL="\ref ddc_domains"];
    samplings  [label="Samplings and\ndiscrete space", URL="\ref ddc_samplings"];
    containers [label="Data containers\nand memory", URL="\ref ddc_containers"];
    algorithms [label="Algorithms", URL="\ref ddc_algorithms"];
    output     [label="Output", URL="\ref ddc_output"];

    domains    -> utils;
    samplings  -> domains;
    samplings  -> utils;
    containers -> domains;
    containers -> utils;
    algorithms -> containers;
    algorithms -> domains;
    algorithms -> utils;
    output     -> containers;
    output     -> domains;
}
\enddot

| Module | Main entities | Responsibility |
|--------|---------------|----------------|
| \ref ddc_core_utils | ddc::TypeSeq, ddc::ScopeGuard, ddc::Real | Compile-time type sequences and tag traits used to label dimensions; lifetime of the global discrete spaces. |
| \ref ddc_domains | ddc::DiscreteElement, ddc::DiscreteVector, ddc::DiscreteDomain, ddc::StridedDiscreteDomain, ddc::SparseDiscreteDomain | Strongly typed multi-dimensional indices, offsets and sets of indices. |
| \ref ddc_samplings | ddc::Coordinate, ddc::UniformPointSampling, ddc::NonUniformPointSampling, ddc::PeriodicSampling, ddc::init_discrete_space | Map discrete elements to coordinates of a continuous dimension. |
| \ref ddc_containers | ddc::Chunk, ddc::ChunkSpan, ddc::KokkosAllocator, ddc::create_mirror | Arrays and views indexed by discrete elements, in a Kokkos memory space. |
| \ref ddc_algorithms | ddc::for_each, ddc::parallel_for_each, ddc::parallel_fill, ddc::parallel_deepcopy, ddc::parallel_transform_reduce | Sequential and Kokkos-parallel iteration over domains and containers. |
| \ref ddc_output | ddc::print, ddc::experimental::save_npy | Formatting and saving of container contents. |
| \ref ddc_experimental | `ddc::experimental` | API without stability guarantee. |

## Typical program structure

A program that uses DDC usually follows these steps, as in [the examples](examples.md):

1. **Initialize the runtimes.** Create a `Kokkos::ScopeGuard`, then a ddc::ScopeGuard. The examples
   declare them in this order, so that the DDC guard is destroyed first.
2. **Declare dimensions.** A continuous dimension is an empty tag type (for example `struct X {};`).
   A discrete dimension is a type that derives from a sampling, for example
   `struct DDimX : ddc::UniformPointSampling<X> {};`.
3. **Initialize discrete spaces.** Call ddc::init_discrete_space once for each discrete dimension.
   This stores the sampling in a global object owned by the library, and returns the domain(s)
   that the sampling covers.
4. **Build domains.** Combine one-dimensional domains into multi-dimensional ddc::DiscreteDomain
   objects, or take sub-domains of them.
5. **Allocate data.** Create ddc::Chunk objects on these domains with an allocator that selects the
   memory space (for example ddc::DeviceAllocator or ddc::HostAllocator), and pass non-owning
   ddc::ChunkSpan views to computations.
6. **Compute.** Iterate with ddc::parallel_for_each and the other algorithms, which execute on a
   Kokkos execution space, and access coordinates with ddc::coordinate.
7. **Exchange data.** Copy between memory spaces with ddc::parallel_deepcopy or the
   ddc::create_mirror family, and output data with ddc::print or the PDI wrapper.

## Related pages

* [DDC concepts](concepts.md)
* \ref error_handling_and_ownership "Error handling, ownership and thread safety"
* \ref first_steps "Tutorial: the uniform heat equation"
