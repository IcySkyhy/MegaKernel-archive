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
"""OpGraph build + serialization (op-level view of the megakernel topology).

Unlike :mod:`tile_graph` (tile-level, built from ``TaskBase``), an
:class:`OpGraph` is built from :class:`triton_dist.mega_triton_kernel.core.graph.Graph`
which already records, per op, the real ``io_tensors`` (with ``data_ptr``-based
identity) and port-level ``input_producers``. That gives a faithful op graph
with real tensor identity and port-level producer/consumer links -- the
information that is *not* recoverable from the JSON topology dump.

The binary format produced by :func:`serialize_opgraph` is the exact wire
format consumed by the header-only C++ reader/writer in
``include/TritonDistributed/OpGraph.h``. Keep both sides in sync with
``_BINARY_MAGIC`` / ``_BINARY_VERSION`` and the record layouts documented
below.
"""
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import logging
import struct

if TYPE_CHECKING:
    from .graph import Graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums (mirror C++ enums in NewDataStructures.config)
# ---------------------------------------------------------------------------
class DType(IntEnum):
    FP16 = 0
    BF16 = 1
    F32 = 2
    I32 = 3
    I64 = 4
    FP8_E4M3 = 5
    FP8_E8M0 = 6
    I8 = 7
    U8 = 8
    UNKNOWN = 9


class LayoutTag(IntEnum):
    RowMajor = 0
    ColMajor = 1


class MemoryPool(IntEnum):
    Persistent = 0
    Recyclable = 1
    External = 2
    ShmemRecyclable = 3


# AttrValue wire tags (match C++ AttrValue variant order)
_ATTR_INT64 = 0
_ATTR_DOUBLE = 1
_ATTR_STRING = 2
_ATTR_INT64_LIST = 3


# ---------------------------------------------------------------------------
# Data model (mirrors C++ structs in NewDataStructures.config)
# ---------------------------------------------------------------------------
@dataclass
class TensorDesc:
    shape: List[int] = field(default_factory=list)
    stride: List[int] = field(default_factory=list)
    dtype: DType = DType.UNKNOWN
    layout: LayoutTag = LayoutTag.RowMajor
    pool_hint: MemoryPool = MemoryPool.Persistent
    external_addr: int = 0  # 0 == nullptr


@dataclass
class Tensor:
    id: int = -1
    name: str = ""
    desc: TensorDesc = field(default_factory=TensorDesc)


@dataclass
class OpAttributes:
    items: Dict[str, Any] = field(default_factory=dict)  # str -> (tag, value)


@dataclass
class OpNode:
    id: int = -1
    type: str = ""
    name: str = ""
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    workspaces: List[int] = field(default_factory=list)
    attrs: OpAttributes = field(default_factory=OpAttributes)
    coarse: bool = False
    depend_ops_id: List[int] = field(default_factory=list)    # predecessors
    depended_ops_id: List[int] = field(default_factory=list)  # successors


@dataclass
class OpGraph:
    tensors: List[Tensor] = field(default_factory=list)
    ops: List[OpNode] = field(default_factory=list)
    tensor_init_data: Dict[int, bytes] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# torch dtype -> DType
# ---------------------------------------------------------------------------
def _dtype_to_enum(t) -> DType:
    import torch
    mapping = {
        torch.float16: DType.FP16,
        torch.bfloat16: DType.BF16,
        torch.float32: DType.F32,
        torch.int32: DType.I32,
        torch.int64: DType.I64,
        torch.int8: DType.I8,
        torch.uint8: DType.U8,
    }
    name = str(t)
    if "float8_e4m3" in name:
        return DType.FP8_E4M3
    if "float8_e8m0" in name:
        return DType.FP8_E8M0
    return mapping.get(t, DType.UNKNOWN)


def _tensor_key(t) -> Tuple[int, int]:
    return (int(t.data_ptr()), int(t.nbytes))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_opgraph_from_graph(graph: "Graph") -> OpGraph:
    """Build an :class:`OpGraph` from a :class:`Graph`.

    One :class:`OpNode` per ``Graph._nodes`` entry. Tensors are deduped by
    ``(data_ptr, nbytes)`` across every op's inputs/outputs, so the same
    physical buffer (e.g. the output of op A reused as input of op B) gets a
    single tensor id -- the port-level link the JSON dump cannot recover.

    If ``graph._declared_tensors`` is non-empty, those entries are interned
    first (preserving declaration order, names, pool hints, and external
    addresses). Op export can be controlled via ``Node.extra_params``:

    - ``opgraph_op_name`` (str): OpNode.name (default ``L{layer}_T{task}_{type}``)
    - ``opgraph_omit_attrs`` (bool): leave attrs empty
    - ``opgraph_omit_deps`` (bool): leave dependOpsId / dependedOpsId empty

    Otherwise ``depend_ops_id`` / ``depended_ops_id`` come from each node's
    ``input_producers``.
    """
    nodes = graph._nodes
    op_id_of = {id(n): i for i, n in enumerate(nodes)}

    tensor_id_of: Dict[Tuple[int, int], int] = {}
    tensors: List[Tensor] = []
    declared_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def intern_tensor(t, hint_name: str) -> int:
        key = _tensor_key(t)
        tid = tensor_id_of.get(key)
        if tid is not None:
            return tid
        tid = len(tensors)
        meta = declared_meta.get(key)
        if meta is not None:
            name = meta["name"]
            pool = MemoryPool(meta["pool_hint"])
            addr = int(meta["external_addr"])
        else:
            name = hint_name
            pool = MemoryPool.Persistent
            addr = int(t.data_ptr())
        desc = TensorDesc(
            shape=[int(s) for s in t.shape],
            stride=[int(s) for s in t.stride()],
            dtype=_dtype_to_enum(t.dtype),
            layout=LayoutTag.RowMajor,
            pool_hint=pool,
            external_addr=addr,
        )
        tensor_id_of[key] = tid
        tensors.append(Tensor(id=tid, name=name, desc=desc))
        return tid

    for d in getattr(graph, "_declared_tensors", []) or []:
        key = _tensor_key(d.tensor)
        declared_meta[key] = {
            "name": d.name,
            "pool_hint": int(d.pool_hint),
            "external_addr": int(d.external_addr),
        }
        intern_tensor(d.tensor, d.name)

    op_nodes: List[OpNode] = []
    omit_deps_flags: List[bool] = []
    for i, n in enumerate(nodes):
        in_tensors = n.io_tensors[0]
        out_tensors = n.io_tensors[1]
        t0 = n.tasks[0] if n.tasks else None
        layer_id = t0.layer_id if t0 is not None else -1
        task_id = t0.task_id if t0 is not None else -1
        op_type = n.op_type
        extra = n.extra_params or {}
        default_name = f"L{layer_id}_T{task_id}_{op_type}"
        op_name = extra.get("opgraph_op_name", default_name)
        inputs = [intern_tensor(t, f"{default_name}_in{idx}") for idx, t in enumerate(in_tensors)]
        outputs = [intern_tensor(t, f"{default_name}_out{idx}") for idx, t in enumerate(out_tensors)]
        if extra.get("opgraph_omit_attrs"):
            attrs = OpAttributes(items={})
        else:
            attrs = _build_attrs(n, t0, op_type, layer_id, task_id, len(n.tasks))
        omit_deps_flags.append(bool(extra.get("opgraph_omit_deps")))
        op_nodes.append(OpNode(
            id=i, type=op_type, name=op_name,
            inputs=inputs, outputs=outputs, workspaces=[],
            attrs=attrs, coarse=False,
            depend_ops_id=[], depended_ops_id=[],
        ))

    for i, n in enumerate(nodes):
        if omit_deps_flags[i]:
            continue
        preds = set()
        for src_node, _ in n.input_producers.values():
            preds.add(op_id_of[id(src_node)])
        op_nodes[i].depend_ops_id = sorted(preds)
    for op in op_nodes:
        if omit_deps_flags[op.id]:
            continue
        for p in op.depend_ops_id:
            op_nodes[p].depended_ops_id.append(op.id)
    for op in op_nodes:
        if omit_deps_flags[op.id]:
            op.depend_ops_id = []
            op.depended_ops_id = []
        else:
            op.depended_ops_id.sort()

    return OpGraph(tensors=tensors, ops=op_nodes, tensor_init_data={})


def canonicalize_opgraph_bytes(data: bytes) -> bytes:
    """Re-serialize with runtime ``data_ptr`` addrs cleared for golden compares.

    Declared External addrs (non-zero and not looking like a fresh data_ptr
    export) are preserved when present in the blob: only addresses that were
    defaulted from ``data_ptr()`` need wiping. For Ascend-aligned dumps that
    already store fixed External addrs (and zeros elsewhere), prefer comparing
    ``build_opgraph()`` bytes directly against the expected blob.
    """
    og = deserialize_opgraph(data)
    for t in og.tensors:
        # Keep explicit External wire addrs; clear Persistent/Recyclable ptrs.
        if t.desc.pool_hint != MemoryPool.External:
            t.desc.external_addr = 0
    return serialize_opgraph(og)


def _coerce_attr_value(v: Any):
    if isinstance(v, bool):
        return (_ATTR_INT64, int(v))
    if isinstance(v, int):
        return (_ATTR_INT64, int(v))
    if isinstance(v, float):
        return (_ATTR_DOUBLE, float(v))
    if isinstance(v, str):
        return (_ATTR_STRING, v)
    if isinstance(v, (list, tuple)):
        ints = []
        for x in v:
            if isinstance(x, bool):
                ints.append(int(x))
            elif isinstance(x, int):
                ints.append(int(x))
            else:
                return None
        return (_ATTR_INT64_LIST, ints)
    return None


def _build_attrs(n, t0, op_type: str, layer_id: int, task_id: int, num_tasks: int) -> OpAttributes:
    items: Dict[str, Any] = {}
    items["op_type"] = (_ATTR_STRING, op_type)
    items["layer_id"] = (_ATTR_INT64, int(layer_id))
    items["task_id"] = (_ATTR_INT64, int(task_id))
    items["num_tiles"] = (_ATTR_INT64, int(num_tasks))
    if t0 is not None and is_dataclass(t0.config):
        try:
            for k, v in asdict(t0.config).items():
                cv = _coerce_attr_value(v)
                if cv is not None:
                    items[k] = cv
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("skipping task config attrs for op %s: %s", op_type, e)
    for k, v in (n.extra_params or {}).items():
        if str(k).startswith("opgraph_"):
            continue
        cv = _coerce_attr_value(v)
        if cv is not None:
            items[k] = cv
    return OpAttributes(items=items)


# ---------------------------------------------------------------------------
# Binary serialization
# ---------------------------------------------------------------------------
# Layout (records 4-byte aligned). Multi-byte fields use the endianness declared
# by `endian` in the header; the Python writer always emits little-endian.
# Keep in sync with include/TritonDistributed/OpGraph.h.
#
# header (28 bytes):
#   magic        : 8 bytes  = "OPGRAPH1"
#   endian       : uint8   = 0 (LE wire) or 1 (BE wire); endian-independent
#   reserved     : 3 bytes = 0
#   version      : uint32  = 2   (read using `endian`)
#   num_tensors  : uint32
#   num_ops      : uint32
#   num_init_data: uint32   = 0 (tensorInitData reserved; currently empty)
#
# tensor record (variable):
#   id           : int32
#   name_len:uint16 ; name : utf8
#   ndim         : uint8
#   shape        : int64[ndim]
#   stride       : int64[ndim]
#   dtype        : uint8
#   layout       : uint8
#   pool_hint    : uint8
#   external_addr: uint64
#
# op record (variable):
#   id              : int32
#   type_len:uint16 ; type : utf8
#   name_len:uint16 ; name : utf8
#   num_inputs:uint32 ; inputs : int32[num_inputs]
#   num_outputs:uint32 ; outputs : int32[num_outputs]
#   num_workspaces:uint32 ; workspaces : int32[num_workspaces]
#   coarse : uint8
#   num_depend_ops:uint32 ; depend_ops_id : int32[]
#   num_deped_ops:uint32 ; depended_ops_id : int32[]
#   num_attrs : uint32
#   attrs (repeated):
#     key_len:uint16 ; key : utf8
#     tag : uint8  (0=int64,1=double,2=string,3=int64_list)
#     value:
#       0: int64
#       1: double(8)
#       2: val_len:uint16 ; val : utf8
#       3: count:uint32 ; int64[count]
_BINARY_MAGIC = b"OPGRAPH1"
_BINARY_VERSION = 2
_ENDIAN_LE = 0
_ENDIAN_BE = 1

# magic + endian + reserved[3] -- endian-independent prefix (endian is a single
# byte, so it can be read before the wire endianness is known).
_PRE_FMT = "<8sB3s"
_PRE_SIZE = struct.calcsize(_PRE_FMT)   # 12

# Writer always emits little-endian; these constants are used by the writer and
# by the pack helpers (_pack_str / _pack_i32_list / _pack_attr_value).
_I32 = "<i"
_U16 = "<H"
_U32 = "<I"
_U8 = "<B"
_I64 = "<q"
_U64 = "<Q"
_F64 = "<d"


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack(_U16, len(b)) + b


def _pack_i32_list(xs) -> bytes:
    out = [struct.pack(_U32, len(xs))]
    for x in xs:
        out.append(struct.pack(_I32, int(x)))
    return b"".join(out)


def _pack_attr_value(tag: int, value) -> bytes:
    if tag == _ATTR_INT64:
        return struct.pack(_I64, int(value))
    if tag == _ATTR_DOUBLE:
        return struct.pack(_F64, float(value))
    if tag == _ATTR_STRING:
        return _pack_str(str(value))
    if tag == _ATTR_INT64_LIST:
        xs = value
        body = [struct.pack(_U32, len(xs))]
        for x in xs:
            body.append(struct.pack(_I64, int(x)))
        return b"".join(body)
    raise ValueError(f"bad attr tag {tag}")


def serialize_opgraph(opgraph: OpGraph) -> bytes:
    parts = [struct.pack(_PRE_FMT, _BINARY_MAGIC, _ENDIAN_LE, b"\x00\x00\x00"),
             struct.pack("<IIII", _BINARY_VERSION, len(opgraph.tensors),
                         len(opgraph.ops), len(opgraph.tensor_init_data))]
    for t in opgraph.tensors:
        d = t.desc
        b_name = t.name.encode("utf-8")
        parts.append(struct.pack(_I32, int(t.id)))
        parts.append(struct.pack(_U16, len(b_name)) + b_name)
        parts.append(struct.pack(_U8, len(d.shape)))
        for s in d.shape:
            parts.append(struct.pack(_I64, int(s)))
        for s in d.stride:
            parts.append(struct.pack(_I64, int(s)))
        parts.append(struct.pack(_U8, int(d.dtype)))
        parts.append(struct.pack(_U8, int(d.layout)))
        parts.append(struct.pack(_U8, int(d.pool_hint)))
        parts.append(struct.pack(_U64, int(d.external_addr)))
    for op in opgraph.ops:
        b_type = op.type.encode("utf-8")
        b_name = op.name.encode("utf-8")
        parts.append(struct.pack(_I32, int(op.id)))
        parts.append(struct.pack(_U16, len(b_type)) + b_type)
        parts.append(struct.pack(_U16, len(b_name)) + b_name)
        parts.append(_pack_i32_list(op.inputs))
        parts.append(_pack_i32_list(op.outputs))
        parts.append(_pack_i32_list(op.workspaces))
        parts.append(struct.pack(_U8, 1 if op.coarse else 0))
        parts.append(_pack_i32_list(op.depend_ops_id))
        parts.append(_pack_i32_list(op.depended_ops_id))
        parts.append(struct.pack(_U32, len(op.attrs.items)))
        for k, (tag, value) in op.attrs.items.items():
            parts.append(_pack_str(k))
            parts.append(struct.pack(_U8, tag))
            parts.append(_pack_attr_value(tag, value))
    return b"".join(parts)


def serialize_opgraph_to_dict(opgraph: OpGraph) -> dict:
    """JSON-friendly view for debugging / cross-checking."""
    return {
        "format": "opgraph_v1",
        "num_tensors": len(opgraph.tensors),
        "num_ops": len(opgraph.ops),
        "tensors": [
            {"id": t.id, "name": t.name,
             "shape": t.desc.shape, "stride": t.desc.stride,
             "dtype": int(t.desc.dtype), "layout": int(t.desc.layout),
             "pool_hint": int(t.desc.pool_hint),
             "external_addr": t.desc.external_addr}
            for t in opgraph.tensors
        ],
        "ops": [
            {"id": op.id, "type": op.type, "name": op.name,
             "inputs": op.inputs, "outputs": op.outputs,
             "workspaces": op.workspaces, "coarse": op.coarse,
             "depend_ops_id": op.depend_ops_id,
             "depended_ops_id": op.depended_ops_id,
             "attrs": {k: [tag, v] for k, (tag, v) in op.attrs.items.items()}}
            for op in opgraph.ops
        ],
    }


# ---------------------------------------------------------------------------
# Deserialize (round-trip helper for tests; the C++ side has its own reader)
# ---------------------------------------------------------------------------
class _Reader:
    def __init__(self, data: bytes, off: int = 0, e: str = "<"):
        self.data = data
        self.off = off
        self.e = e  # endianness prefix chosen from the header's `endian` byte

    def _r(self, fmt: str):
        full = self.e + fmt
        sz = struct.calcsize(full)
        v = struct.unpack_from(full, self.data, self.off)
        self.off += sz
        return v

    def u8(self):  return self._r("B")[0]
    def u16(self): return self._r("H")[0]
    def u32(self): return self._r("I")[0]
    def i32(self): return self._r("i")[0]
    def u64(self): return self._r("Q")[0]
    def i64(self): return self._r("q")[0]
    def f64(self): return self._r("d")[0]

    def read_str(self) -> str:
        n = self.u16()
        s = self.data[self.off:self.off + n].decode("utf-8")
        self.off += n
        return s

    def read_i32_list(self) -> List[int]:
        n = self.u32()
        return [self.i32() for _ in range(n)]

    def read_attr_value(self, tag: int):
        if tag == _ATTR_INT64:
            return self.i64()
        if tag == _ATTR_DOUBLE:
            return self.f64()
        if tag == _ATTR_STRING:
            return self.read_str()
        if tag == _ATTR_INT64_LIST:
            n = self.u32()
            return [self.i64() for _ in range(n)]
        raise ValueError(f"bad attr tag {tag}")


def deserialize_opgraph(data: bytes) -> OpGraph:
    if len(data) < _PRE_SIZE + 16:
        raise ValueError("buffer too small for OpGraph header")
    magic, endian, _pad = struct.unpack_from(_PRE_FMT, data, 0)
    if magic != _BINARY_MAGIC:
        raise ValueError(f"bad OpGraph magic: {magic!r}")
    if endian not in (_ENDIAN_LE, _ENDIAN_BE):
        raise ValueError(f"bad endian flag: {endian}")
    e = "<" if endian == _ENDIAN_LE else ">"
    version, num_tensors, num_ops, _num_init = struct.unpack_from(e + "IIII", data, _PRE_SIZE)
    if version != _BINARY_VERSION:
        raise ValueError(f"unsupported OpGraph version {version}")

    r = _Reader(data, off=_PRE_SIZE + 16, e=e)
    tensors: List[Tensor] = []
    for _ in range(num_tensors):
        tid = r.i32()
        name = r.read_str()
        ndim = r.u8()
        shape = [r.i64() for _ in range(ndim)]
        stride = [r.i64() for _ in range(ndim)]
        dt = r.u8()
        lay = r.u8()
        pool = r.u8()
        addr = r.u64()
        tensors.append(Tensor(id=tid, name=name, desc=TensorDesc(
            shape=shape, stride=stride, dtype=DType(dt), layout=LayoutTag(lay),
            pool_hint=MemoryPool(pool), external_addr=addr)))

    ops: List[OpNode] = []
    for _ in range(num_ops):
        oid = r.i32()
        otype = r.read_str()
        oname = r.read_str()
        inputs = r.read_i32_list()
        outputs = r.read_i32_list()
        workspaces = r.read_i32_list()
        coarse = r.u8()
        depend_ops_id = r.read_i32_list()
        depended_ops_id = r.read_i32_list()
        num_attrs = r.u32()
        items: Dict[str, Any] = {}
        for _ in range(num_attrs):
            k = r.read_str()
            tag = r.u8()
            items[k] = (tag, r.read_attr_value(tag))
        ops.append(OpNode(id=oid, type=otype, name=oname, inputs=inputs,
                          outputs=outputs, workspaces=workspaces,
                          attrs=OpAttributes(items=items), coarse=bool(coarse),
                          depend_ops_id=depend_ops_id,
                          depended_ops_id=depended_ops_id))

    return OpGraph(tensors=tensors, ops=ops, tensor_init_data={})


def save_opgraph_to_file(opgraph: OpGraph, path: str) -> None:
    """Serialize ``opgraph`` and write the wire bytes to ``path``.

    Mirrors C++ ``save_opgraph_to_file`` in ``OpGraph.h``.
    """
    with open(path, "wb") as f:
        f.write(serialize_opgraph(opgraph))


def load_opgraph_from_file(path: str) -> OpGraph:
    """Read wire bytes from ``path`` and deserialize an :class:`OpGraph`.

    Mirrors C++ ``load_opgraph_from_file`` in ``OpGraph.h``.
    """
    with open(path, "rb") as f:
        return deserialize_opgraph(f.read())
