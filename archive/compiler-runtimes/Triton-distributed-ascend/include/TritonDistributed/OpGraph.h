////////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////
// OpGraph: C++ mirror of the Python `core/op_graph.py` data model and a
// header-only reader/writer for the binary wire format produced by
// `ModelBuilder.build_opgraph()` / `serialize_opgraph()`.
//
// Self-contained (host-only, no CUDA, no pybind). Add `include/` to your
// target's include directories and `#include <TritonDistributed/OpGraph.h>`.
//
// The writer always emits little-endian (`endian=0`), matching the Python
// serializer; the reader adapts to the wire endianness declared in the header.
//
// Wire format (self-describing endianness; keep in sync with op_graph.py):
//
//   header (28 bytes):
//     magic        : 8 bytes  = "OPGRAPH1"
//     endian       : uint8_t  = 0 (LE wire) or 1 (BE wire); endian-independent
//     reserved     : 3 bytes  = 0
//     version      : uint32_t = 2   (read using `endian`)
//     num_tensors  : uint32_t
//     num_ops      : uint32_t
//     num_init_data: uint32_t = 0 (tensorInitData reserved)
//
//   tensor record (variable):
//     id           : int32_t
//     name_len:uint16 ; name : utf8
//     ndim         : uint8_t
//     shape        : int64_t[ndim]
//     stride       : int64_t[ndim]
//     dtype        : uint8_t
//     layout       : uint8_t
//     pool_hint    : uint8_t
//     external_addr: uint64_t  (wire encoding of TensorDesc::externalAddr /
//                               void*; in-memory type is void*, matching
//                               NewDataStructures.config)
//
//   op record (variable):
//     id              : int32_t
//     type_len:uint16 ; type : utf8
//     name_len:uint16 ; name : utf8
//     num_inputs:uint32 ; inputs : int32_t[num_inputs]
//     num_outputs:uint32 ; outputs : int32_t[num_outputs]
//     num_workspaces:uint32 ; workspaces : int32_t[num_workspaces]
//     coarse : uint8_t
//     num_depend_ops:uint32 ; depend_ops_id : int32_t[]
//     num_deped_ops:uint32 ; depended_ops_id : int32_t[]
//     num_attrs : uint32
//     attrs (repeated):
//       key_len:uint16 ; key : utf8
//       tag : uint8_t (0=int64,1=double,2=string,3=int64_list)
//       value:
//         0: int64_t
//         1: double(8)
//         2: val_len:uint16 ; val : utf8
//         3: count:uint32 ; int64_t[count]
//
// Multi-byte fields are stored in the endianness declared by `endian`. The
// reader detects the host endianness at compile time and the wire endianness
// at runtime, then byte-swaps each multi-byte field iff they differ. Every
// field is read with memcpy, so the buffer need not be aligned.

#pragma once

#include <TritonDistributed/WireEndian.h>

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace triton_dist {

// In-memory data model mirrors NewDataStructures.config (C++). Wire format
// may encode some fields differently (e.g. void* as uint64_t); see header
// comment above.

enum class DType : uint8_t {
    FP16 = 0, BF16 = 1, F32 = 2, I32 = 3, I64 = 4,
    FP8_E4M3 = 5, FP8_E8M0 = 6, I8 = 7, U8 = 8, UNKNOWN = 9,
};

enum class LayoutTag : uint8_t { RowMajor = 0, ColMajor = 1 };

enum class MemoryPool : uint8_t {
    Persistent = 0, Recyclable = 1, External = 2, ShmemRecyclable = 3,
};

using Shape = std::vector<int64_t>;
using Stride = std::vector<int64_t>;

struct TensorDesc {
    Shape shape;
    Stride stride;
    DType dtype{DType::UNKNOWN};
    LayoutTag layout{LayoutTag::RowMajor};
    MemoryPool poolHint{MemoryPool::Persistent};
    void* externalAddr{nullptr};  // Device address for external tensors
};

struct Tensor {
    int id{-1};
    std::string name;
    TensorDesc desc;
};

using AttrValue = std::variant<int64_t, double, std::string, std::vector<int64_t>>;

struct OpAttributes {
    std::unordered_map<std::string, AttrValue> items;
};

struct OpEdge {
    int src;
    int dst;
};

struct OpNode {
    int id{-1};
    std::string type;
    std::string name;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<int> workspaces;
    OpAttributes attrs;
    bool coarse{false};
    std::vector<int> dependOpsId;    // predecessors
    std::vector<int> dependedOpsId;  // successors
};

struct OpGraph {
    std::vector<Tensor> tensors;
    std::vector<OpNode> ops;
    std::unordered_map<int, std::vector<uint8_t>> tensorInitData;
};

namespace detail {

inline constexpr char kOpGraphMagic[8] = {'O','P','G','R','A','P','H','1'};
inline constexpr uint32_t kOpGraphVersion = 2;

// AttrValue wire tags (match op_graph.py)
inline constexpr uint8_t kAttrInt64 = 0;
inline constexpr uint8_t kAttrDouble = 1;
inline constexpr uint8_t kAttrString = 2;
inline constexpr uint8_t kAttrInt64List = 3;

// Endian / byteswap / append_bytes live in WireEndian.h (shared with TileGraph).

// Cursor over a byte buffer. `swap` is set once the wire endianness is known
// (from the header's `endian` byte) and is applied to every multi-byte read.
struct Cursor {
    const uint8_t* p;
    size_t remain;
    bool swap = false;

    bool read(void* dst, size_t n, std::string* err) {
        if (remain < n) {
            if (err) *err = "OpGraph: unexpected end of buffer";
            return false;
        }
        std::memcpy(dst, p, n);
        p += n;
        remain -= n;
        return true;
    }
};

template <typename T>
inline bool read_field(Cursor& c, T& out, std::string* err) {
    T v;
    if (!c.read(&v, sizeof(T), err)) return false;
    if (c.swap) v = byteswap(v);
    out = v;
    return true;
}

inline bool read_u16(Cursor& c, uint16_t& out, std::string* err) { return read_field<uint16_t>(c, out, err); }
inline bool read_u32(Cursor& c, uint32_t& out, std::string* err) { return read_field<uint32_t>(c, out, err); }
inline bool read_i32(Cursor& c, int32_t& out, std::string* err) { return read_field<int32_t>(c, out, err); }
inline bool read_u8 (Cursor& c, uint8_t& out, std::string* err) { return read_field<uint8_t>(c, out, err); }
inline bool read_i64(Cursor& c, int64_t& out, std::string* err) { return read_field<int64_t>(c, out, err); }
inline bool read_u64(Cursor& c, uint64_t& out, std::string* err) { return read_field<uint64_t>(c, out, err); }
inline bool read_f64(Cursor& c, double& out, std::string* err) { return read_field<double>(c, out, err); }

inline bool read_str(Cursor& c, std::string& out, std::string* err) {
    uint16_t n = 0;
    if (!read_u16(c, n, err)) return false;
    if (c.remain < n) { if (err) *err = "OpGraph: string truncated"; return false; }
    out.assign(reinterpret_cast<const char*>(c.p), n);
    c.p += n; c.remain -= n;
    return true;
}

inline bool read_i32_list(Cursor& c, std::vector<int>& out, std::string* err) {
    uint32_t n = 0;
    if (!read_u32(c, n, err)) return false;
    out.clear();
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        int32_t v = 0;
        if (!read_i32(c, v, err)) return false;
        out.push_back(static_cast<int>(v));
    }
    return true;
}

inline bool read_attr_value(Cursor& c, uint8_t tag, AttrValue& out, std::string* err) {
    switch (tag) {
        case kAttrInt64: {
            int64_t v = 0; if (!read_i64(c, v, err)) return false;
            out = v; return true;
        }
        case kAttrDouble: {
            double v = 0; if (!read_f64(c, v, err)) return false;
            out = v; return true;
        }
        case kAttrString: {
            std::string s; if (!read_str(c, s, err)) return false;
            out = std::move(s); return true;
        }
        case kAttrInt64List: {
            uint32_t n = 0; if (!read_u32(c, n, err)) return false;
            std::vector<int64_t> xs; xs.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                int64_t v = 0; if (!read_i64(c, v, err)) return false;
                xs.push_back(v);
            }
            out = std::move(xs); return true;
        }
        default:
            if (err) *err = "OpGraph: bad attr tag";
            return false;
    }
}

// Writer helpers. Always emit little-endian (`endian=0`), matching Python.
// append_bytes is provided by WireEndian.h.

template <typename T>
inline void write_field(std::vector<uint8_t>& out, T v, bool swap) {
    if (swap) v = byteswap(v);
    append_bytes(out, &v, sizeof(T));
}

inline void write_u8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }
inline void write_u16(std::vector<uint8_t>& out, uint16_t v, bool swap) { write_field(out, v, swap); }
inline void write_u32(std::vector<uint8_t>& out, uint32_t v, bool swap) { write_field(out, v, swap); }
inline void write_i32(std::vector<uint8_t>& out, int32_t v, bool swap) { write_field(out, v, swap); }
inline void write_i64(std::vector<uint8_t>& out, int64_t v, bool swap) { write_field(out, v, swap); }
inline void write_u64(std::vector<uint8_t>& out, uint64_t v, bool swap) { write_field(out, v, swap); }
inline void write_f64(std::vector<uint8_t>& out, double v, bool swap) { write_field(out, v, swap); }

inline bool write_str(std::vector<uint8_t>& out, const std::string& s, bool swap,
                      std::string* err) {
    if (s.size() > 0xFFFFu) {
        if (err) *err = "OpGraph: string longer than uint16 max";
        return false;
    }
    write_u16(out, static_cast<uint16_t>(s.size()), swap);
    append_bytes(out, s.data(), s.size());
    return true;
}

inline bool write_i32_list(std::vector<uint8_t>& out, const std::vector<int>& xs,
                           bool swap, std::string* err) {
    if (xs.size() > 0xFFFFFFFFu) {
        if (err) *err = "OpGraph: i32 list longer than uint32 max";
        return false;
    }
    write_u32(out, static_cast<uint32_t>(xs.size()), swap);
    for (int v : xs) write_i32(out, static_cast<int32_t>(v), swap);
    return true;
}

inline bool write_attr_value(std::vector<uint8_t>& out, const AttrValue& val, bool swap,
                             std::string* err) {
    return std::visit([&](const auto& v) -> bool {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            write_u8(out, kAttrInt64);
            write_i64(out, v, swap);
            return true;
        } else if constexpr (std::is_same_v<T, double>) {
            write_u8(out, kAttrDouble);
            write_f64(out, v, swap);
            return true;
        } else if constexpr (std::is_same_v<T, std::string>) {
            write_u8(out, kAttrString);
            return write_str(out, v, swap, err);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            if (v.size() > 0xFFFFFFFFu) {
                if (err) *err = "OpGraph: int64 list longer than uint32 max";
                return false;
            }
            write_u8(out, kAttrInt64List);
            write_u32(out, static_cast<uint32_t>(v.size()), swap);
            for (int64_t x : v) write_i64(out, x, swap);
            return true;
        } else {
            if (err) *err = "OpGraph: unsupported attr value type";
            return false;
        }
    }, val);
}

}  // namespace detail

// Deserialize an OpGraph from `data` of `size` bytes. Returns true on success
// and fills `out`. On failure returns false and writes a message to `err` (if
// non-null). Bounds-checked; will not read past the buffer end. Throws
// std::bad_alloc only if the recorded counts are impossibly large (guarded by
// `max_tensors` / `max_ops` / `max_init`).
inline bool deserialize_opgraph(const void* data, size_t size, OpGraph& out,
                                std::string* err = nullptr,
                                size_t max_tensors = 1u << 28,
                                size_t max_ops = 1u << 28,
                                size_t max_init = 1u << 28) {
    out.tensors.clear();
    out.ops.clear();
    if (data == nullptr || size < 12) {
        if (err) *err = "OpGraph: buffer too small for header";
        return false;
    }
    detail::Cursor c{static_cast<const uint8_t*>(data), size, false};

    char magic[8] = {0};
    if (!c.read(magic, 8, err)) return false;
    if (std::memcmp(magic, detail::kOpGraphMagic, 8) != 0) {
        if (err) *err = "OpGraph: bad magic";
        return false;
    }
    uint8_t endian = 0;
    if (!c.read(&endian, 1, err)) return false;
    if (endian != detail::kEndianLE && endian != detail::kEndianBE) {
        if (err) *err = "OpGraph: bad endian flag";
        return false;
    }
    uint8_t pad[3] = {0};
    if (!c.read(pad, 3, err)) return false;  // reserved
    // Wire is big-endian iff the flag says so; host is big-endian iff the
    // compile-time probe says so. Swap iff they differ.
    c.swap = ((endian == detail::kEndianBE) != detail::kHostBigEndian);

    uint32_t version = 0, num_tensors = 0, num_ops = 0, num_init = 0;
    if (!detail::read_u32(c, version, err)) return false;
    if (version != detail::kOpGraphVersion) {
        if (err) *err = "OpGraph: unsupported version";
        return false;
    }
    if (!detail::read_u32(c, num_tensors, err)) return false;
    if (!detail::read_u32(c, num_ops, err)) return false;
    if (!detail::read_u32(c, num_init, err)) return false;
    if (num_tensors > max_tensors || num_ops > max_ops || num_init > max_init) {
        if (err) *err = "OpGraph: counts exceed safety limit";
        return false;
    }
    // tensorInitData is reserved; if ever populated, its records would be read
    // here. Currently always 0.

    out.tensors.clear();
    out.tensors.reserve(num_tensors);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        Tensor t;
        int32_t tid = -1;
        if (!detail::read_i32(c, tid, err)) return false;
        t.id = static_cast<int>(tid);
        if (!detail::read_str(c, t.name, err)) return false;
        uint8_t ndim = 0;
        if (!detail::read_u8(c, ndim, err)) return false;
        t.desc.shape.resize(ndim);
        for (uint8_t d = 0; d < ndim; ++d)
            if (!detail::read_i64(c, t.desc.shape[d], err)) return false;
        t.desc.stride.resize(ndim);
        for (uint8_t d = 0; d < ndim; ++d)
            if (!detail::read_i64(c, t.desc.stride[d], err)) return false;
        uint8_t dt = 0, lay = 0, pool = 0;
        if (!detail::read_u8(c, dt, err)) return false;
        if (!detail::read_u8(c, lay, err)) return false;
        if (!detail::read_u8(c, pool, err)) return false;
        uint64_t addr = 0;
        if (!detail::read_u64(c, addr, err)) return false;
        t.desc.externalAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
        t.desc.dtype = static_cast<DType>(dt);
        t.desc.layout = static_cast<LayoutTag>(lay);
        t.desc.poolHint = static_cast<MemoryPool>(pool);
        out.tensors.push_back(std::move(t));
    }

    out.ops.clear();
    out.ops.reserve(num_ops);
    for (uint32_t i = 0; i < num_ops; ++i) {
        OpNode op;
        int32_t oid = -1;
        if (!detail::read_i32(c, oid, err)) return false;
        op.id = static_cast<int>(oid);
        if (!detail::read_str(c, op.type, err)) return false;
        if (!detail::read_str(c, op.name, err)) return false;
        if (!detail::read_i32_list(c, op.inputs, err)) return false;
        if (!detail::read_i32_list(c, op.outputs, err)) return false;
        if (!detail::read_i32_list(c, op.workspaces, err)) return false;
        uint8_t coarse = 0;
        if (!detail::read_u8(c, coarse, err)) return false;
        op.coarse = (coarse != 0);
        if (!detail::read_i32_list(c, op.dependOpsId, err)) return false;
        if (!detail::read_i32_list(c, op.dependedOpsId, err)) return false;
        uint32_t num_attrs = 0;
        if (!detail::read_u32(c, num_attrs, err)) return false;
        for (uint32_t a = 0; a < num_attrs; ++a) {
            std::string key;
            if (!detail::read_str(c, key, err)) return false;
            uint8_t tag = 0;
            if (!detail::read_u8(c, tag, err)) return false;
            AttrValue val;
            if (!detail::read_attr_value(c, tag, val, err)) return false;
            op.attrs.items.emplace(std::move(key), std::move(val));
        }
        out.ops.push_back(std::move(op));
    }

    out.tensorInitData.clear();
    // tensorInitData wire records reserved; currently always num_init==0.

    if (err) err->clear();
    return true;
}

// Throwing variant.
inline OpGraph deserialize_opgraph(const void* data, size_t size) {
    OpGraph out;
    std::string err;
    if (!deserialize_opgraph(data, size, out, &err)) {
        throw std::runtime_error("deserialize_opgraph: " + err);
    }
    return out;
}

template <typename Byte, typename = std::enable_if_t<sizeof(Byte) == 1>>
inline bool deserialize_opgraph(const std::vector<Byte>& buf, OpGraph& out,
                                std::string* err = nullptr) {
    return deserialize_opgraph(buf.data(), buf.size(), out, err);
}

// Serialize an OpGraph to the binary wire format consumed by
// `deserialize_opgraph` / the Python reader. Always emits little-endian
// (`endian=0`), matching `serialize_opgraph` in op_graph.py.
//
// Returns true on success and fills `out`. On failure returns false and writes
// a message to `err` (if non-null). `out` is cleared on entry.
inline bool serialize_opgraph(const OpGraph& in, std::vector<uint8_t>& out,
                              std::string* err = nullptr) {
    out.clear();
    if (in.tensors.size() > 0xFFFFFFFFu || in.ops.size() > 0xFFFFFFFFu) {
        if (err) *err = "OpGraph: tensor/op count exceeds uint32 max";
        return false;
    }

    // Write LE wire; swap multi-byte fields when the host is big-endian.
    const bool swap = detail::kHostBigEndian;

    out.reserve(28 + in.tensors.size() * 64 + in.ops.size() * 128);
    detail::append_bytes(out, detail::kOpGraphMagic, 8);
    detail::write_u8(out, detail::kEndianLE);
    const uint8_t pad[3] = {0, 0, 0};
    detail::append_bytes(out, pad, 3);
    detail::write_u32(out, detail::kOpGraphVersion, swap);
    detail::write_u32(out, static_cast<uint32_t>(in.tensors.size()), swap);
    detail::write_u32(out, static_cast<uint32_t>(in.ops.size()), swap);
    detail::write_u32(out, 0u, swap);  // num_init_data reserved

    for (const Tensor& t : in.tensors) {
        if (t.desc.shape.size() != t.desc.stride.size()) {
            if (err) *err = "OpGraph: tensor shape/stride ndim mismatch";
            return false;
        }
        if (t.desc.shape.size() > 0xFFu) {
            if (err) *err = "OpGraph: tensor ndim exceeds uint8 max";
            return false;
        }
        detail::write_i32(out, static_cast<int32_t>(t.id), swap);
        if (!detail::write_str(out, t.name, swap, err)) return false;
        detail::write_u8(out, static_cast<uint8_t>(t.desc.shape.size()));
        for (int64_t s : t.desc.shape) detail::write_i64(out, s, swap);
        for (int64_t s : t.desc.stride) detail::write_i64(out, s, swap);
        detail::write_u8(out, static_cast<uint8_t>(t.desc.dtype));
        detail::write_u8(out, static_cast<uint8_t>(t.desc.layout));
        detail::write_u8(out, static_cast<uint8_t>(t.desc.poolHint));
        detail::write_u64(out,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t.desc.externalAddr)),
            swap);
    }

    for (const OpNode& op : in.ops) {
        if (op.attrs.items.size() > 0xFFFFFFFFu) {
            if (err) *err = "OpGraph: attr count exceeds uint32 max";
            return false;
        }
        detail::write_i32(out, static_cast<int32_t>(op.id), swap);
        if (!detail::write_str(out, op.type, swap, err)) return false;
        if (!detail::write_str(out, op.name, swap, err)) return false;
        if (!detail::write_i32_list(out, op.inputs, swap, err)) return false;
        if (!detail::write_i32_list(out, op.outputs, swap, err)) return false;
        if (!detail::write_i32_list(out, op.workspaces, swap, err)) return false;
        detail::write_u8(out, op.coarse ? 1u : 0u);
        if (!detail::write_i32_list(out, op.dependOpsId, swap, err)) return false;
        if (!detail::write_i32_list(out, op.dependedOpsId, swap, err)) return false;
        detail::write_u32(out, static_cast<uint32_t>(op.attrs.items.size()), swap);
        for (const auto& kv : op.attrs.items) {
            if (!detail::write_str(out, kv.first, swap, err)) return false;
            if (!detail::write_attr_value(out, kv.second, swap, err)) return false;
        }
    }

    if (err) err->clear();
    return true;
}

// Throwing variant.
inline std::vector<uint8_t> serialize_opgraph(const OpGraph& in) {
    std::vector<uint8_t> out;
    std::string err;
    if (!serialize_opgraph(in, out, &err)) {
        throw std::runtime_error("serialize_opgraph: " + err);
    }
    return out;
}

// ---------------------------------------------------------------------------
// File helpers (read / write the same wire bytes as the buffer APIs above).
// ---------------------------------------------------------------------------

inline bool load_opgraph_from_file(const std::string& path, OpGraph& out,
                                   std::string* err = nullptr,
                                   size_t max_tensors = 1u << 28,
                                   size_t max_ops = 1u << 28,
                                   size_t max_init = 1u << 28) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (err) *err = "OpGraph: failed to open file for read: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff n = ifs.tellg();
    if (n < 0) {
        if (err) *err = "OpGraph: failed to get file size: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    if (n > 0 && !ifs.read(reinterpret_cast<char*>(buf.data()), n)) {
        if (err) *err = "OpGraph: failed to read file: " + path;
        return false;
    }
    return deserialize_opgraph(buf.data(), buf.size(), out, err,
                               max_tensors, max_ops, max_init);
}

inline OpGraph load_opgraph_from_file(const std::string& path) {
    OpGraph out;
    std::string err;
    if (!load_opgraph_from_file(path, out, &err)) {
        throw std::runtime_error("load_opgraph_from_file: " + err);
    }
    return out;
}

inline bool save_opgraph_to_file(const OpGraph& in, const std::string& path,
                                 std::string* err = nullptr) {
    std::vector<uint8_t> buf;
    if (!serialize_opgraph(in, buf, err)) return false;
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        if (err) *err = "OpGraph: failed to open file for write: " + path;
        return false;
    }
    if (!buf.empty() &&
        !ofs.write(reinterpret_cast<const char*>(buf.data()),
                   static_cast<std::streamsize>(buf.size()))) {
        if (err) *err = "OpGraph: failed to write file: " + path;
        return false;
    }
    if (err) err->clear();
    return true;
}

inline void save_opgraph_to_file(const OpGraph& in, const std::string& path) {
    std::string err;
    if (!save_opgraph_to_file(in, path, &err)) {
        throw std::runtime_error("save_opgraph_to_file: " + err);
    }
}

}  // namespace triton_dist
