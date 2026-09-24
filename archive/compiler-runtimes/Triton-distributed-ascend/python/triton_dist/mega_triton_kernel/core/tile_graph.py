################################################################################
#
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""TileGraph build + serialization.

A ``TileGraph`` is the tile-level view of the megakernel task topology: one
``TileNode`` per ``TaskBase`` (i.e. per tile), and one ``TileEdge`` per
producer-tile -> consumer-tile dependency, expanded from each
``TaskDependency``'s ``[start_tiles, end_tiles)`` interval.

The binary serialization produced by :func:`serialize_tilegraph` is the exact
wire format consumed by the header-only C++ reader/writer in
``include/TritonDistributed/TileGraph.h``. Both sides must stay in sync with
``_BINARY_MAGIC`` / ``_BINARY_VERSION`` and the record layouts documented
below.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Tuple
import struct

if TYPE_CHECKING:
    from .task_base import TaskBase


# ---------------------------------------------------------------------------
# Data model (mirrors the C++ structs in NewDataStructures.config / TileGraph.h)
# ---------------------------------------------------------------------------
class EdgeOrigin(IntEnum):
    Exact = 0          # precise producer/consumer edge (dep-opt graph)
    Conservative = 1   # order-registered, possibly over-approximated edge
    Hidden = 2         # implicit / barrier edge (not emitted from task deps)


@dataclass
class TileNode:
    id: int            # uint32; node id == index into TileGraph.nodes
    op_id: int         # uint32; which op (aggregated (layer_id, task_id)) this tile belongs to
    tile_id: int       # uint32; TaskBase.tile_id_or_start
    output_index: int  # uint16; which output of the op this tile produces (0 when single-output)


@dataclass
class TileEdge:
    src: int           # uint32; source TileNode.id
    dst: int           # uint32; destination TileNode.id
    origin: EdgeOrigin = EdgeOrigin.Exact  # uint8


@dataclass
class TileGraph:
    nodes: List[TileNode] = field(default_factory=list)
    edges: List[TileEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_tilegraph_from_tasks(tasks: "List[TaskBase]", enable_dep_opt: bool = False) -> TileGraph:
    """Build a :class:`TileGraph` from a list of ``TaskBase``.

    One node is created per task (node ``id`` == the task's position in
    ``tasks``). Edges are expanded from each task's ``dependency`` list: a
    dependency ``TaskDependency(layer_id, task_id, start_tiles, end_tiles)``
    means "wait for tiles ``[start_tiles, end_tiles)`` of op
    ``(layer_id, task_id)``", so one edge is emitted per referenced tile.

    The sentinel dependency ``layer_id == task_id == -1`` (the default
    ``TaskDependency()``) marks an entry node and produces no edges.

    ``output_index`` is not recorded per-tile by the task builders, so it
    defaults to 0 (correct for every Qwen3 op, which are all single-output).

    ``origin`` is set to ``Exact`` when ``enable_dep_opt`` is true (edges come
    from the producer/consumer graph) and ``Conservative`` otherwise (edges
    come from builder registration order and may over-approximate).
    """
    # op id per (layer_id, task_id), assigned in first-seen order
    op_index: Dict[Tuple[int, int], int] = {}
    for t in tasks:
        key = (t.layer_id, t.task_id)
        if key not in op_index:
            op_index[key] = len(op_index)

    nodes: List[TileNode] = []
    # op_id -> {tile_id_or_start: node_id}
    tile_to_node: Dict[int, Dict[int, int]] = {}
    for idx, t in enumerate(tasks):
        op_id = op_index[(t.layer_id, t.task_id)]
        nodes.append(TileNode(id=idx, op_id=op_id, tile_id=t.tile_id_or_start, output_index=0))
        tile_to_node.setdefault(op_id, {})[t.tile_id_or_start] = idx

    origin = EdgeOrigin.Exact if enable_dep_opt else EdgeOrigin.Conservative
    edges: List[TileEdge] = []
    for idx, t in enumerate(tasks):
        for dep in t.dependency:
            if dep.layer_id == -1 and dep.task_id == -1:
                continue  # entry sentinel
            src_op_id = op_index.get((dep.layer_id, dep.task_id))
            if src_op_id is None:
                continue  # unknown producer; skip silently
            tile_map = tile_to_node.get(src_op_id, {})
            for k in range(dep.start_tiles, dep.end_tiles):
                src = tile_map.get(k)
                if src is None:
                    continue  # interval references a tile that does not exist
                edges.append(TileEdge(src=src, dst=idx, origin=origin))

    return TileGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Binary serialization
# ---------------------------------------------------------------------------
# Layout (records 4-byte aligned). Multi-byte fields use the endianness declared
# by `endian` in the header; the Python writer always emits little-endian.
#
#   header (24 bytes):
#     magic       : 8 bytes  = b"TILEGRPH"
#     endian      : uint8    = 0 (LE wire) or 1 (BE wire); endian-independent
#     reserved    : 3 bytes  = 0
#     version     : uint32   = 2   (read using `endian`)
#     num_nodes   : uint32
#     num_edges   : uint32
#
#   nodes[num_nodes] (16 bytes each):
#     id           : uint32
#     op_id        : uint32
#     tile_id      : uint32
#     output_index : uint16
#     reserved     : uint16  = 0
#
#   edges[num_edges] (12 bytes each):
#     src     : uint32
#     dst     : uint32
#     origin  : uint8
#     reserved: 3 bytes = 0
#
# Keep in sync with include/TritonDistributed/TileGraph.h.
_BINARY_MAGIC = b"TILEGRPH"
_BINARY_VERSION = 2
_ENDIAN_LE = 0
_ENDIAN_BE = 1

# magic + endian + reserved[3] -- endian-independent prefix (read before knowing
# the wire endianness, since endian is a single byte).
_PRE_FMT = "<8sB3s"
_PRE_SIZE = struct.calcsize(_PRE_FMT)            # 12
_NODE_FMT_LE = "<IIIHH"                          # id, op_id, tile_id, output_index, reserved
_EDGE_FMT_LE = "<IIB3s"                          # src, dst, origin, reserved
_NODE_SIZE = struct.calcsize(_NODE_FMT_LE)       # 16
_EDGE_SIZE = struct.calcsize(_EDGE_FMT_LE)       # 12


def serialize_tilegraph(tilegraph: TileGraph) -> bytes:
    """Serialize a :class:`TileGraph` to the binary wire format consumed by
    the C++ reader/writer in ``include/TritonDistributed/TileGraph.h``.

    The Python writer always emits little-endian (``endian=0``); the C++ reader
    adapts to its host endianness at decode time."""
    parts = [struct.pack(_PRE_FMT, _BINARY_MAGIC, _ENDIAN_LE, b"\x00\x00\x00"),
             struct.pack("<III", _BINARY_VERSION, len(tilegraph.nodes), len(tilegraph.edges))]
    for n in tilegraph.nodes:
        parts.append(struct.pack(_NODE_FMT_LE, int(n.id) & 0xFFFFFFFF,
                                  int(n.op_id) & 0xFFFFFFFF,
                                  int(n.tile_id) & 0xFFFFFFFF,
                                  int(n.output_index) & 0xFFFF, 0))
    for e in tilegraph.edges:
        parts.append(struct.pack(_EDGE_FMT_LE, int(e.src) & 0xFFFFFFFF,
                                  int(e.dst) & 0xFFFFFFFF,
                                  int(e.origin) & 0xFF, b"\x00\x00\x00"))
    return b"".join(parts)


def serialize_tilegraph_to_dict(tilegraph: TileGraph) -> dict:
    """Convenience JSON-friendly view (for debugging / cross-checking with the
    topology JSON dump). Not used by the C++ reader."""
    return {
        "format": "tilegraph_v1",
        "num_nodes": len(tilegraph.nodes),
        "num_edges": len(tilegraph.edges),
        "nodes": [
            {"id": n.id, "op_id": n.op_id, "tile_id": n.tile_id,
             "output_index": n.output_index}
            for n in tilegraph.nodes
        ],
        "edges": [
            {"src": e.src, "dst": e.dst, "origin": int(e.origin)}
            for e in tilegraph.edges
        ],
    }


def deserialize_tilegraph(data: bytes) -> TileGraph:
    """Round-trip helper for the binary format (used by tests; the C++ side
    has its own reader). Adapts to the wire endianness declared in the header."""
    if len(data) < _PRE_SIZE + 12:
        raise ValueError("buffer too small for TileGraph header")
    magic, endian, _pad = struct.unpack_from(_PRE_FMT, data, 0)
    if magic != _BINARY_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if endian not in (_ENDIAN_LE, _ENDIAN_BE):
        raise ValueError(f"bad endian flag: {endian}")
    e = "<" if endian == _ENDIAN_LE else ">"
    version, num_nodes, num_edges = struct.unpack_from(e + "III", data, _PRE_SIZE)
    if version != _BINARY_VERSION:
        raise ValueError(f"unsupported TileGraph version {version}")

    node_fmt = e + "IIIHH"
    edge_fmt = e + "IIB3s"
    off = _PRE_SIZE + 12
    nodes: List[TileNode] = []
    for _ in range(num_nodes):
        nid, op_id, tile_id, out_idx, _r = struct.unpack_from(node_fmt, data, off)
        nodes.append(TileNode(id=nid, op_id=op_id, tile_id=tile_id, output_index=out_idx))
        off += _NODE_SIZE

    edges: List[TileEdge] = []
    for _ in range(num_edges):
        src, dst, origin, _r = struct.unpack_from(edge_fmt, data, off)
        edges.append(TileEdge(src=src, dst=dst, origin=EdgeOrigin(origin)))
        off += _EDGE_SIZE

    return TileGraph(nodes=nodes, edges=edges)


def save_tilegraph_to_file(tilegraph: TileGraph, path: str) -> None:
    """Serialize ``tilegraph`` and write the wire bytes to ``path``.

    Mirrors C++ ``save_tilegraph_to_file`` in ``TileGraph.h``.
    """
    with open(path, "wb") as f:
        f.write(serialize_tilegraph(tilegraph))


def load_tilegraph_from_file(path: str) -> TileGraph:
    """Read wire bytes from ``path`` and deserialize a :class:`TileGraph`.

    Mirrors C++ ``load_tilegraph_from_file`` in ``TileGraph.h``.
    """
    with open(path, "rb") as f:
        return deserialize_tilegraph(f.read())
