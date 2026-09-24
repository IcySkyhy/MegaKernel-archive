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
// TileGraph: C++ mirror of the Python `core/tile_graph.py` data model and a
// header-only reader/writer for the binary wire format produced by
// `ModelBuilder.build_tilegraph()` / `serialize_tilegraph()`.
//
// This header is self-contained (host-only, no CUDA, no pybind). Add
// `include/` to your target's include directories and `#include
// <TritonDistributed/TileGraph.h>`.
//
// The writer always emits little-endian (`endian=0`), matching the Python
// serializer; the reader adapts to the wire endianness declared in the header.
//
// Wire format (self-describing endianness; keep in sync with tile_graph.py):
//
//   header (24 bytes):
//     magic       : 8 bytes  = "TILEGRPH"
//     endian      : uint8_t  = 0 (LE wire) or 1 (BE wire); endian-independent
//     reserved    : 3 bytes  = 0
//     version     : uint32_t = 2   (read using `endian`)
//     num_nodes   : uint32_t
//     num_edges   : uint32_t
//
//   nodes[num_nodes] (16 bytes each):
//     id           : uint32_t
//     op_id        : uint32_t
//     tile_id      : uint32_t
//     output_index : uint16_t
//     reserved     : uint16_t  = 0
//
//   edges[num_edges] (12 bytes each):
//     src     : uint32_t
//     dst     : uint32_t
//     origin  : uint8_t   (0=Exact, 1=Conservative, 2=Hidden)
//     reserved: 3 bytes  = 0
//
// All multi-byte fields are stored in the endianness declared by `endian`. The
// reader detects the host endianness at compile time and the wire endianness
// at runtime (from `endian`), then byte-swaps each field iff they differ. All
// records are 4-byte aligned; the buffer need not be aligned because every
// field is read with memcpy.

#pragma once

#include <TritonDistributed/WireEndian.h>

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace triton_dist {

enum class EdgeOrigin : uint8_t {
    Exact        = 0,
    Conservative = 1,
    Hidden       = 2,
};

struct TileNode {
    uint32_t id{0};
    uint32_t opId{0};
    uint32_t tileId{0};
    uint16_t outputIndex{0};
};

struct TileEdge {
    uint32_t src{0};
    uint32_t dst{0};
    EdgeOrigin origin{EdgeOrigin::Exact};
};

struct TileGraph {
    std::vector<TileNode> nodes;
    std::vector<TileEdge> edges;
};

namespace detail {

inline constexpr char kTileGraphMagic[8] = {'T','I','L','E','G','R','P','H'};
inline constexpr uint32_t kTileGraphVersion = 2;

// Endian / byteswap / load_field / store_field / append_bytes live in
// WireEndian.h (shared with OpGraph.h).

struct HeaderPacked {
    char magic[8];
    uint8_t endian;
    uint8_t reserved[3];
    uint32_t version;
    uint32_t num_nodes;
    uint32_t num_edges;
};
static_assert(sizeof(HeaderPacked) == 24, "TileGraph header must be 24 bytes");

struct NodePacked {
    uint32_t id;
    uint32_t op_id;
    uint32_t tile_id;
    uint16_t output_index;
    uint16_t reserved;
};
static_assert(sizeof(NodePacked) == 16, "TileGraph node record must be 16 bytes");

struct EdgePacked {
    uint32_t src;
    uint32_t dst;
    uint8_t  origin;
    uint8_t  reserved[3];
};
static_assert(sizeof(EdgePacked) == 12, "TileGraph edge record must be 12 bytes");

}  // namespace detail

// Deserialize a TileGraph from `data` of `size` bytes.
//
// Returns true on success and fills `out`. On failure returns false and, when
// `err` is non-null, writes a human-readable message. Does not allocate on
// failure. Throws std::bad_alloc only if the recorded counts are impossibly
// large (guarded by `max_nodes` / `max_edges`).
inline bool deserialize_tilegraph(const void* data, size_t size, TileGraph& out,
                                  std::string* err = nullptr,
                                  size_t max_nodes = 1u << 28,
                                  size_t max_edges = 1u << 28) {
    out.nodes.clear();
    out.edges.clear();
    if (data == nullptr || size < sizeof(detail::HeaderPacked)) {
        if (err) *err = "buffer too small for TileGraph header";
        return false;
    }
    const auto* base = static_cast<const char*>(data);

    detail::HeaderPacked hdr;
    std::memcpy(&hdr, base, sizeof(hdr));
    if (std::memcmp(hdr.magic, detail::kTileGraphMagic, 8) != 0) {
        if (err) *err = "bad TileGraph magic";
        return false;
    }
    if (hdr.endian != detail::kEndianLE && hdr.endian != detail::kEndianBE) {
        if (err) *err = "bad TileGraph endian flag";
        return false;
    }
    // Wire is big-endian iff the flag says so; host is big-endian iff the
    // compile-time probe says so. Swap iff they differ.
    const bool wire_be = (hdr.endian == detail::kEndianBE);
    const bool need_swap = (wire_be != detail::kHostBigEndian);

    const uint32_t version = detail::load_field<uint32_t>(&hdr.version, need_swap);
    if (version != detail::kTileGraphVersion) {
        if (err) *err = "unsupported TileGraph version";
        return false;
    }
    const uint32_t num_nodes = detail::load_field<uint32_t>(&hdr.num_nodes, need_swap);
    const uint32_t num_edges = detail::load_field<uint32_t>(&hdr.num_edges, need_swap);
    if (num_nodes > max_nodes || num_edges > max_edges) {
        if (err) *err = "TileGraph counts exceed safety limit";
        return false;
    }

    const size_t nodes_bytes = static_cast<size_t>(num_nodes) * sizeof(detail::NodePacked);
    const size_t edges_bytes = static_cast<size_t>(num_edges) * sizeof(detail::EdgePacked);
    const size_t needed = sizeof(detail::HeaderPacked) + nodes_bytes + edges_bytes;
    if (size < needed) {
        if (err) *err = "buffer truncated for declared TileGraph counts";
        return false;
    }

    out.nodes.resize(num_nodes);
    const char* p = base + sizeof(detail::HeaderPacked);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        detail::NodePacked np;
        std::memcpy(&np, p, sizeof(np));
        out.nodes[i].id           = detail::load_field<uint32_t>(&np.id, need_swap);
        out.nodes[i].opId         = detail::load_field<uint32_t>(&np.op_id, need_swap);
        out.nodes[i].tileId       = detail::load_field<uint32_t>(&np.tile_id, need_swap);
        out.nodes[i].outputIndex  = detail::load_field<uint16_t>(&np.output_index, need_swap);
        p += sizeof(detail::NodePacked);
    }

    out.edges.resize(num_edges);
    for (uint32_t i = 0; i < num_edges; ++i) {
        detail::EdgePacked ep;
        std::memcpy(&ep, p, sizeof(ep));
        out.edges[i].src    = detail::load_field<uint32_t>(&ep.src, need_swap);
        out.edges[i].dst    = detail::load_field<uint32_t>(&ep.dst, need_swap);
        out.edges[i].origin = static_cast<EdgeOrigin>(ep.origin);  // single byte
        p += sizeof(detail::EdgePacked);
    }

    if (err) err->clear();
    return true;
}

// Convenience overload for contiguous byte containers.
template <typename Byte,
          typename = std::enable_if_t<sizeof(Byte) == 1>>
inline bool deserialize_tilegraph(const std::vector<Byte>& buf, TileGraph& out,
                                  std::string* err = nullptr) {
    return deserialize_tilegraph(buf.data(), buf.size(), out, err);
}

// Throwing variant for callers that prefer exceptions.
inline TileGraph deserialize_tilegraph(const void* data, size_t size) {
    TileGraph out;
    std::string err;
    if (!deserialize_tilegraph(data, size, out, &err)) {
        throw std::runtime_error("deserialize_tilegraph: " + err);
    }
    return out;
}

// Serialize a TileGraph to the binary wire format consumed by
// `deserialize_tilegraph` / the Python reader. Always emits little-endian
// (`endian=0`), matching `serialize_tilegraph` in tile_graph.py.
//
// Returns true on success and fills `out`. On failure returns false and writes
// a message to `err` (if non-null). `out` is cleared on entry.
inline bool serialize_tilegraph(const TileGraph& in, std::vector<uint8_t>& out,
                                std::string* err = nullptr) {
    out.clear();
    if (in.nodes.size() > 0xFFFFFFFFu || in.edges.size() > 0xFFFFFFFFu) {
        if (err) *err = "TileGraph: node/edge count exceeds uint32 max";
        return false;
    }

    // Write LE wire; swap multi-byte fields when the host is big-endian.
    const bool need_swap = detail::kHostBigEndian;

    detail::HeaderPacked hdr{};
    std::memcpy(hdr.magic, detail::kTileGraphMagic, 8);
    hdr.endian = detail::kEndianLE;
    hdr.reserved[0] = hdr.reserved[1] = hdr.reserved[2] = 0;
    detail::store_field(&hdr.version, detail::kTileGraphVersion, need_swap);
    detail::store_field(&hdr.num_nodes, static_cast<uint32_t>(in.nodes.size()), need_swap);
    detail::store_field(&hdr.num_edges, static_cast<uint32_t>(in.edges.size()), need_swap);

    const size_t total = sizeof(detail::HeaderPacked)
        + in.nodes.size() * sizeof(detail::NodePacked)
        + in.edges.size() * sizeof(detail::EdgePacked);
    out.reserve(total);
    detail::append_bytes(out, &hdr, sizeof(hdr));

    for (const TileNode& n : in.nodes) {
        detail::NodePacked np{};
        detail::store_field(&np.id, n.id, need_swap);
        detail::store_field(&np.op_id, n.opId, need_swap);
        detail::store_field(&np.tile_id, n.tileId, need_swap);
        detail::store_field(&np.output_index, n.outputIndex, need_swap);
        np.reserved = 0;
        detail::append_bytes(out, &np, sizeof(np));
    }

    for (const TileEdge& e : in.edges) {
        detail::EdgePacked ep{};
        detail::store_field(&ep.src, e.src, need_swap);
        detail::store_field(&ep.dst, e.dst, need_swap);
        ep.origin = static_cast<uint8_t>(e.origin);
        ep.reserved[0] = ep.reserved[1] = ep.reserved[2] = 0;
        detail::append_bytes(out, &ep, sizeof(ep));
    }

    if (err) err->clear();
    return true;
}

// Throwing variant.
inline std::vector<uint8_t> serialize_tilegraph(const TileGraph& in) {
    std::vector<uint8_t> out;
    std::string err;
    if (!serialize_tilegraph(in, out, &err)) {
        throw std::runtime_error("serialize_tilegraph: " + err);
    }
    return out;
}

// ---------------------------------------------------------------------------
// File helpers (read / write the same wire bytes as the buffer APIs above).
// ---------------------------------------------------------------------------

inline bool load_tilegraph_from_file(const std::string& path, TileGraph& out,
                                     std::string* err = nullptr,
                                     size_t max_nodes = 1u << 28,
                                     size_t max_edges = 1u << 28) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (err) *err = "TileGraph: failed to open file for read: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff n = ifs.tellg();
    if (n < 0) {
        if (err) *err = "TileGraph: failed to get file size: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    if (n > 0 && !ifs.read(reinterpret_cast<char*>(buf.data()), n)) {
        if (err) *err = "TileGraph: failed to read file: " + path;
        return false;
    }
    return deserialize_tilegraph(buf.data(), buf.size(), out, err,
                                 max_nodes, max_edges);
}

inline TileGraph load_tilegraph_from_file(const std::string& path) {
    TileGraph out;
    std::string err;
    if (!load_tilegraph_from_file(path, out, &err)) {
        throw std::runtime_error("load_tilegraph_from_file: " + err);
    }
    return out;
}

inline bool save_tilegraph_to_file(const TileGraph& in, const std::string& path,
                                   std::string* err = nullptr) {
    std::vector<uint8_t> buf;
    if (!serialize_tilegraph(in, buf, err)) return false;
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        if (err) *err = "TileGraph: failed to open file for write: " + path;
        return false;
    }
    if (!buf.empty() &&
        !ofs.write(reinterpret_cast<const char*>(buf.data()),
                   static_cast<std::streamsize>(buf.size()))) {
        if (err) *err = "TileGraph: failed to write file: " + path;
        return false;
    }
    if (err) err->clear();
    return true;
}

inline void save_tilegraph_to_file(const TileGraph& in, const std::string& path) {
    std::string err;
    if (!save_tilegraph_to_file(in, path, &err)) {
        throw std::runtime_error("save_tilegraph_to_file: " + err);
    }
}

}  // namespace triton_dist
