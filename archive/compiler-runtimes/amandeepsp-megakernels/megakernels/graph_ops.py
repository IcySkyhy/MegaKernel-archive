import operator
from operator import getitem
from typing import Iterator
from dataclasses import field
import torch
from dataclasses import dataclass
from torch._subclasses import FakeTensor
import torch
from torch.fx.passes.shape_prop import TensorMetadata
from collections import defaultdict
from torch import fx
from .ir import ValueRef, TensorSpec
from torch.utils import _pytree as pytree

__all__ = ["GraphState"]

type MetaValue = torch.Tensor | tuple["MetaValue", ...] | list["MetaValue"] | None
type MetaScalar = (
     int
     | float
     | bool
     | str
     | torch.SymInt
     | torch.SymFloat
     | torch.SymBool
     | None
 )

_MISSING = object()


class GraphState:

    def __init__(self, gm: fx.GraphModule):
        self.storage_ids: dict[torch.UntypedStorage, int] = {}
        self.value_specs: dict[ValueRef, TensorSpec] = {}
        self.nodes_by_name: dict[str, fx.Node] = {}

        self.inputs: list[ValueRef] = []
        self.outputs: list[ValueRef] = []

        self._build(gm)

    def _build(self, gm: fx.GraphModule) -> None:
        self.storage_ids.clear()
        self.value_specs.clear()
        self.nodes_by_name.clear()

        self.inputs.clear()
        self.outputs.clear()

        for node in gm.graph.nodes:

            if node.name in self.nodes_by_name:
                raise RuntimeError(f"Duplicate FX node: {node.name} - should not happen")
            self.nodes_by_name[node.name] = node

            if node.op == "output":
                resolved_output = self.resolve_arg(node.args[0])
                leaves, _ = pytree.tree_flatten(resolved_output)

                self.outputs.extend(
                        leaf for leaf in leaves if isinstance(leaf, ValueRef)
                )
                continue

            value = node.meta.get("val", _MISSING)
            if value is _MISSING:
                raise RuntimeError(
                    f"{node.name}: missing node.meta['val']; "
                    "tensor metadata propagation may not have run"
                )

            node_outputs: list[ValueRef] = []
            for out_index, tensor in self._iter_tensor_outputs(value, node=node):
                ref = ValueRef(node=node.name, output_index=out_index)
                if ref in self.value_specs:
                    raise RuntimeError(f"duplicate value reference: {ref}")

                self.value_specs[ref] = self._make_tensor_spec(tensor)
                node_outputs.append(ref)

            if node.op == "placeholder":
                if len(node_outputs) != 1:
                    raise NotImplementedError(
                        f"{node.name}: expected one tensor placeholder output, "
                        f"got {len(node_outputs)}"
                    )
                self.inputs.append(node_outputs[0])


    @staticmethod
    def _iter_tensor_outputs(
        value: object,
        *,
        node: fx.Node,
    ):
        if isinstance(value, torch.Tensor):
            yield 0, value
            return

        if isinstance(value, (tuple, list)):
            for output_index, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    yield output_index, item
                elif item is None:
                    continue
                else:
                    raise NotImplementedError(
                        f"{node.name}: unsupported output {output_index}: "
                        f"{type(item).__qualname__}"
                    )
            return

        if value is None:
            return

        raise NotImplementedError(
            f"{node.name}: unsupported metadata value {type(value).__qualname__}"
        )

    def _intern_storage(self, tensor: torch.Tensor) -> int:
        storage = tensor.untyped_storage()
        try:
            return self.storage_ids[storage]
        except KeyError:
            storage_id = len(self.storage_ids)
            self.storage_ids[storage] = storage_id
            return storage_id

    def _make_tensor_spec(self, tensor: torch.Tensor) -> TensorSpec:
        return TensorSpec(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
            storage_offset=tensor.storage_offset(),
            storage_id=self._intern_storage(tensor),
        )

    def resolve_value(self, node: fx.Node) -> ValueRef:
        if node.op == "call_function" and node.target is operator.getitem:
            source, index = node.args
            source_value = source.meta.get("val")

            if isinstance(source, fx.Node) and isinstance(source_value, (tuple, list)):
                if not isinstance(index, int):
                    raise NotImplementedError("Dynamically indexed container, not yet implmented")
                return ValueRef(source.name, index)

        return ValueRef(node.name, 0)


    def resolve_arg(self, arg):
        return fx.map_arg(arg, self.resolve_value)

    def spec(self, value: ValueRef) -> TensorSpec:
        return self.value_specs[value]
