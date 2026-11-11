# Error handling, ownership and thread safety {#error_handling_and_ownership}

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

[TOC]

This page gathers the conventions that apply to the whole public API. The reference documentation
of each entity gives the details that are specific to it. When a behaviour is not described here or
in the reference documentation, it is not specified by the library.

## Error handling

DDC reports errors in three ways, depending on when they can be detected.

### Compile-time errors

Most misuses are rejected at compile time. Dimensions are types, so mixing indices, offsets or
domains of different dimensions does not compile. Templates are constrained with C++20 concepts
(for example ddc::concepts::discrete_dimension or ddc::concepts::discrete_domain) and
`static_assert` declarations that state the expected property in their message.

### Run-time assertions

Preconditions that can only be checked at run time are mostly verified with `assert` and
`KOKKOS_ASSERT`. These checks are not part of the contract of a function: `assert` is disabled when
`NDEBUG` is defined, and the activation of `KOKKOS_ASSERT` depends on the configuration of Kokkos.
Calling a function with a violated precondition is a programming error, even in a build where the
check is disabled. The preconditions are given in the `Precondition` sections of the reference
documentation.

### Exceptions

A few functions throw `std::runtime_error` when they detect an error at run time. In the core
library:

| Function | Condition |
|----------|-----------|
| ddc::init_discrete_space | The discrete space of the dimension is already initialized. |
| ddc::init_discrete_space, with CUDA or HIP | The copy of the discrete space to device memory fails. |
| Construction of a ddc::NonUniformPointSampling from points (constructors of `NonUniformPointSampling::Impl`) | The input points are not sorted. |
| ddc::experimental::save_npy | The npy header does not fit in the npy 1.0 format. |

In the optional components, the spline builders throw when the interpolation problem is
inconsistent (for example an incorrect number of interpolation points), and the spline linear
solvers throw when the underlying LAPACKE or Ginkgo call fails.

Some functions are declared `noexcept` but call code that can throw (for example
ddc::host_for_each_block with a number of blocks, or ddc::parallel_for_each_team on a domain that is
too large). An exception that leaves a `noexcept` function calls `std::terminate`. The reference
documentation of these functions mentions it in a `Warning` section.

DDC does not define its own exception types.

## Ownership and lifetime

### Values

ddc::DiscreteElement, ddc::DiscreteVector, ddc::Coordinate and the domain types are small value
types: copying them copies their content and they do not refer to external memory, with the
exception of ddc::SparseDiscreteDomain, which stores its elements in Kokkos views.

### Containers and views

* ddc::Chunk **owns** its memory. The constructor allocates the memory with the given allocator and
  the destructor releases it. A ddc::Chunk can be moved but not copied: the copy constructor and the
  copy assignment operator are deleted.
* ddc::ChunkSpan is a **non-owning** view. It refers to memory owned by something else, for example
  a ddc::Chunk (see ddc::Chunk::span_view) or a `Kokkos::View`. Copying a ddc::ChunkSpan copies the
  view, not the data. A ddc::ChunkSpan must not be used after the memory it refers to has been
  released: DDC does not check this.
* The memory space of a container is selected by its allocator (ddc::HostAllocator,
  ddc::DeviceAllocator or another ddc::KokkosAllocator) or by the template parameters of the view.
  Data in device memory cannot be accessed directly from host code: use ddc::parallel_deepcopy or
  the ddc::create_mirror family of functions to copy it.

### Discrete spaces

ddc::init_discrete_space stores the discretization of a dimension in a global object owned by the
library, and registers it in a store that is created by the constructor of ddc::ScopeGuard. The
destructor of ddc::ScopeGuard destroys all the registered discrete spaces. As a consequence:

* a ddc::ScopeGuard must be alive when ddc::init_discrete_space is called;
* a discrete space can be initialized only once per dimension;
* functions that read a discrete space, such as ddc::coordinate or ddc::discrete_space, must not be
  called after the ddc::ScopeGuard has been destroyed.

The examples create a `Kokkos::ScopeGuard` before the ddc::ScopeGuard, so that the discrete spaces,
which can hold Kokkos views, are destroyed before Kokkos is finalized.

## Thread safety and asynchrony

The library does not specify thread-safety guarantees for its API. Apart from the printing
functions, it uses no locks to protect its global state. In particular, the initialization of
discrete spaces with ddc::init_discrete_space and the destruction of ddc::ScopeGuard use no
synchronization.

The printing functions (ddc::print and related functions) take a global lock, so that output
printed from several threads at the same time is not interleaved.

The parallel algorithms (ddc::parallel_for_each and the other `parallel_` functions) are thin
layers over Kokkos: each one calls `Kokkos::parallel_for`, `Kokkos::parallel_reduce`,
`Kokkos::parallel_scan` or `Kokkos::deep_copy`, and DDC adds no synchronization of its own. They
therefore have the synchronization semantics of the Kokkos call they make. For example,
ddc::parallel_for_each calls `Kokkos::parallel_for`, which can return before the kernel has
completed: call `fence()` on the execution space before reading the results from the host.

The only fence in the public headers is in the constructor of ddc::SparseDiscreteDomain, which
fences the execution space it receives before returning.
