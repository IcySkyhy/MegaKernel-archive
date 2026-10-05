from megakernels.graph_ops import GraphState
from pprint import pprint
from numpy import conjugate
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

from .debug import save_graph_svg, save_graph_text


class MegakernelBackend:
    def __init__(self) -> None:
        self.backend = aot_autograd(
            fw_compiler=self.compile,
            inference_compiler=self.compile,
            bw_compiler=self._backward,
        )

        self.pass_manager = PassManager(run_checks_after_each_pass=True)

    @staticmethod
    def _backward(gm: GraphModule, example_inputs: list[torch.Tensor]):
        raise NotImplementedError(
            "Only forward modes are supported. Use `torch.no_grad()` or `torch.inference_mode()`"
        )

    def __call__(self, gm: GraphModule, example_inputs):
        return self.backend(gm, example_inputs)

    def compile(self, gm: GraphModule, example_inputs: list[torch.Tensor]):
        save_graph_svg(gm)
        save_graph_text(gm)

        graph_state = GraphState(gm)

        for node in gm.graph.nodes:
            for arg in node.args:
                ref = graph_state.resolve_arg(arg)
                print(ref, end="    ")
            print()

        return make_boxed_func(gm.forward)
