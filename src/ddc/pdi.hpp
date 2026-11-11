// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

//! @file
//! @brief Utilities to expose DDC chunks, scalars and C-strings to PDI.
//! Provides `ddc::PdiEvent`, which shares data with PDI for the duration of an event,
//! and the `ddc::expose_to_pdi` helpers for exposing a single piece of data.

#include <any>
#include <array>
#include <cstddef>
#include <list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>

#include <pdi.h>

namespace ddc {

//! @addtogroup ddc_pdi
//! @{

//! Default PDI access mode deduced for a scalar passed as `T`:
//! `PDI_INOUT` for a non-const lvalue reference, `PDI_OUT` otherwise.
//! @tparam T The type (possibly a reference type) of the exposed scalar.
template <class T>
static constexpr PDI_inout_t default_access_v
        = (std::is_lvalue_reference_v<T> && !std::is_const_v<std::remove_reference_t<T>>)
                  ? PDI_INOUT
                  : PDI_OUT;

//! Default PDI access mode deduced for a chunk passed as `T`:
//! `PDI_INOUT` if `T` is a writable chunk (see `ddc::is_writable_chunk_v`), `PDI_OUT` otherwise.
//! @tparam T The type (possibly a reference type) of the exposed chunk.
template <class T>
static constexpr PDI_inout_t chunk_default_access_v = is_writable_chunk_v<T> ? PDI_INOUT : PDI_OUT;

//! Builds a PDI event: data registered with `with` are shared with PDI, and the event is
//! triggered when the `PdiEvent` object is destroyed, after which the shared names are reclaimed.
//!
//! The object is neither copyable nor movable. It is typically used as a temporary,
//! for example `ddc::PdiEvent("some_event").with("label", chunk.span_cview());`.
class PdiEvent
{
    std::string m_event_name;

    std::vector<std::string> m_names;

    std::list<std::any> m_metadata;

    char const* store_name(std::string&& name);

    char const* store_name(std::string const& name);

    template <class T>
    T* store_scalar(T t)
    {
        std::any& ref = m_metadata.emplace_back(std::in_place_type<T>, std::move(t));
        return std::any_cast<T>(&ref);
    }

    template <class T>
    T* store_array(std::vector<T> v)
    {
        std::any& ref = m_metadata.emplace_back(std::in_place_type<std::vector<T>>, std::move(v));
        return std::any_cast<std::vector<T>>(&ref)->data();
    }

public:
    //! Creates an event that does not share any data yet.
    //! @param[in] event_name Name of the PDI event triggered when this object is destroyed.
    explicit PdiEvent(std::string const& event_name);

    //! @brief Deleted copy constructor: an event cannot be copied.
    //! @param[in] rhs The event that cannot be copied.
    PdiEvent(PdiEvent const& rhs) = delete;

    //! @brief Deleted move constructor: an event cannot be moved.
    //! @param[in] rhs The event that cannot be moved.
    PdiEvent(PdiEvent&& rhs) noexcept = delete;

    //! Triggers the PDI event and reclaims every name shared through `with`.
    ~PdiEvent() noexcept;

    //! @brief Deleted copy assignment operator: an event cannot be copied.
    //! @param[in] rhs The event that cannot be copied.
    //! @return No value, the operator is deleted.
    PdiEvent& operator=(PdiEvent const& rhs) = delete;

    //! @brief Deleted move assignment operator: an event cannot be moved.
    //! @param[in] rhs The event that cannot be moved.
    //! @return No value, the operator is deleted.
    PdiEvent& operator=(PdiEvent&& rhs) noexcept = delete;

    /// @{
    /// API with access argument

    //! Shares a chunk with PDI using an explicit access mode.
    //! The chunk is shared under `name`, its rank under `name + "_rank"` and its extents
    //! (array of `std::size_t`) under `name + "_extents"`, both read-only for PDI.
    //! The data are not copied: PDI accesses the memory of the chunk.
    //! @tparam Access The PDI access mode (`PDI_IN`, `PDI_OUT` or `PDI_INOUT`) of the data.
    //! @param[in] name Name under which the chunk is shared with PDI.
    //! @param[in] data The borrowed chunk to share.
    //! @return A reference to this event, to chain calls.
    //! @pre If `Access` contains `PDI_IN`, the chunk must be writable (checked by `static_assert`).
    template <PDI_inout_t Access, concepts::borrowed_chunk BorrowedChunk>
    PdiEvent& with(std::string const& name, BorrowedChunk&& data)
    {
        static_assert(
                !(Access & PDI_IN) || (chunk_default_access_v<BorrowedChunk> & PDI_IN),
                "Invalid access for constant data");
        std::array const extents = detail::array(data.domain().extents());
        PDI_share(store_name(name + "_rank"), store_scalar(extents.size()), PDI_OUT);
        PDI_share(
                store_name(name + "_extents"),
                store_array(std::vector<std::size_t>(extents.begin(), extents.end())),
                PDI_OUT);
        PDI_share(
                store_name(name),
                const_cast<chunk_value_t<BorrowedChunk>*>(data.data_handle()),
                Access);
        return *this;
    }

    //! Shares an arithmetic scalar with PDI using an explicit access mode.
    //! If `Access` does not contain `PDI_IN`, a copy of the value is shared instead of the
    //! argument itself, which is then safe to pass as a temporary.
    //! @tparam Access The PDI access mode (`PDI_IN`, `PDI_OUT` or `PDI_INOUT`) of the data.
    //! @param[in] name Name under which the scalar is shared with PDI.
    //! @param[in] data The arithmetic value to share.
    //! @return A reference to this event, to chain calls.
    //! @pre If `Access` contains `PDI_IN`, the argument must be a non-const lvalue (checked by `static_assert`).
    template <PDI_inout_t Access, class Arithmetic>
    PdiEvent& with(std::string const& name, Arithmetic&& data)
        requires(std::is_arithmetic_v<std::remove_reference_t<Arithmetic>>)
    {
        static_assert(
                !(Access & PDI_IN) || (default_access_v<Arithmetic> & PDI_IN),
                "Invalid access for constant data");
        using value_type = std::remove_cvref_t<Arithmetic>;
        // NOLINTNEXTLINE(misc-const-correctness)
        value_type* data_ptr = const_cast<value_type*>(&data);
        // for read-only data, we share a copy instead of the data itself in case we received a ref on a temporary,
        if constexpr (!(Access & PDI_IN)) {
            data_ptr = store_scalar(data);
        }
        PDI_share(store_name(name), data_ptr, Access);
        return *this;
    }

    /// @}
    /// API with access deduction
    /// @{

    /// Borrowed chunk overload (Chunk (const)& or ChunkSpan&& or ChunkSpan (const)&)
    /// The access mode is `chunk_default_access_v`.
    /// @param[in] name Name under which the chunk is shared with PDI.
    /// @param[in] data The borrowed chunk to share.
    /// @return A reference to this event, to chain calls.
    template <concepts::borrowed_chunk BorrowedChunk>
    PdiEvent& with(std::string const& name, BorrowedChunk&& data)
    {
        return with<chunk_default_access_v<BorrowedChunk>>(name, std::forward<BorrowedChunk>(data));
    }

    /// Arithmetic overload
    /// The access mode is `default_access_v`.
    /// @param[in] name Name under which the scalar is shared with PDI.
    /// @param[in] data The arithmetic value to share.
    /// @return A reference to this event, to chain calls.
    template <class Arithmetic>
    PdiEvent& with(std::string const& name, Arithmetic&& data)
        requires(std::is_arithmetic_v<std::remove_reference_t<Arithmetic>>)
    {
        return with<default_access_v<Arithmetic>>(name, std::forward<Arithmetic>(data));
    }

    /// C-string overload
    /// The string is shared read-only under `name`, and its length (without the terminating
    /// null character) under `name + "_size"`. The characters are not copied.
    /// @param[in] name Name under which the string is shared with PDI.
    /// @param[in] c_string Null-terminated string; it must outlive the event.
    /// @return A reference to this event, to chain calls.
    PdiEvent& with(std::string const& name, char const* c_string);

    /// @}
};

//! Triggers a PDI event named `name` that shares `data` under the same name,
//! with an explicit access mode.
//! @tparam Access The PDI access mode (`PDI_IN`, `PDI_OUT` or `PDI_INOUT`) of the data.
//! @param[in] name Name of both the event and the shared data.
//! @param[in] data The chunk or arithmetic value to share.
//! @see PdiEvent::with
template <PDI_inout_t Access, class DataType>
void expose_to_pdi(std::string const& name, DataType&& data)
{
    PdiEvent(name).with<Access>(name, std::forward<DataType>(data));
}

//! Triggers a PDI event named `name` that shares `data` under the same name,
//! with an access mode deduced from the type of `data`.
//! @param[in] name Name of both the event and the shared data.
//! @param[in] data The chunk, arithmetic value or C-string to share.
//! @see PdiEvent::with
template <class DataType>
void expose_to_pdi(std::string const& name, DataType&& data)
{
    PdiEvent(name).with(name, std::forward<DataType>(data));
}

//! @}

} // namespace ddc
