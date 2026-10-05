from pathlib import Path

import tabulate
from torch.fx import GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer


def save_graph_svg(gm: GraphModule, name: str = "graph") -> Path:
    path = Path(name).with_suffix(".svg")
    drawer = FxGraphDrawer(gm, path.stem)
    drawer.get_dot_graph().write_svg(path)  # pyrefly: ignore[missing-attribute]
    print("Saved FX graph to", path)
    return path


def save_graph_text(gm: GraphModule, name: str = "graph") -> tuple[Path, ...]:
    tabular_export = Path(f"{name}-table").with_suffix(".txt")
    with open(tabular_export, "w") as file:
        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes
        ]
        tabulated_specs = tabulate.tabulate(
            node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
        )
        file.write(tabulated_specs)
    return (tabular_export,)
