import torch

N_EMBD = 768
N_HEAD = 12
N_LAYER = 12
VOCAB_SIZE = 50257
HEAD_DIM = N_EMBD // N_HEAD  # 64
N_INNER = 4 * N_EMBD        # 3072
EPS = 1e-5
THREADS = 256
NUM_BLOCKS = 512
BATCH = 8

class Tensor:
    def __init__(self, name: str, t: torch.Tensor, is_weight=False, idx=0):
        self.name = name
        self.t = t
        self.is_weight = is_weight
        self.idx = idx

    def get_num_elements(self):
        return self.t.numel()
    
    def get_tensor(self):
        return self.t
    
    def emit(self):
        safe = self.name.replace('.', '_')
        return f"({safe})"

class TensorSlice:
    def __init__(self, name: str, parent: Tensor, access):
        self.name = name
        self.parent_name = parent.name
        self.parent = parent
        self.access = access
        self.is_weight = False
        self.t = parent.get_tensor()[access]
        

    def get_tensor(self):
        return self.t
    

    def emit(self):
        sz = self.t.numel()
        pptr = self.parent.emit()
        return f"(({pptr}) + ({self.access}) * {sz}))"

class Kernel:
    def __init__(self, name: str, num_blocks: int, inps: list[int], outs: list[int], consts: list[int] = []):
        self.name = name
        self.num_blocks = num_blocks
        self.inps = inps
        self.outs = outs
        self.consts = consts

    def emit(self, tensors: list[Tensor]):
        args_list = ','.join(
                [tensors[inp].emit() for inp in self.inps] + 
                [tensors[out].emit() for out in self.outs] + 
                [str(const) for const in self.consts])
        return (
            f"""
                for(int bidx = blockIdx.x; bidx < {self.num_blocks}; bidx += blockDim.x) {{
                    {self.name}({args_list}, bidx);
                }}
            """)


class Graph:
    def __init__(self):
        self.tensors = []
        self.tensor_dict = {}
        self.kernels = []
        self.edges = {}
        pass

    def add_tensor(self, name: str, t: torch.Tensor, is_weight=False):
        tensor = Tensor(name, t, is_weight, len(self.tensors))
        self.tensors.append(tensor)
        self.tensor_dict[name] = tensor.idx
        return tensor.idx

    def add_tensor_slice(self, name: str, parent_name: str, access):
        tensor = TensorSlice(name, self.tensors[self.tensor_dict[parent_name]], access)
        self.tensors.append(tensor)
        self.tensor_dict[name] = len(self.tensors) - 1
        return len(self.tensors) - 1

        
    
    def add_kernel_op(self, name: str, num_blocks: int, inps: list[int], outs: list[int], consts: list[int] = []):
        kernel = Kernel(name, num_blocks, inps, outs, consts)
        self.kernels.append(kernel)
        for inp in inps:
            if inp not in self.edges:
                self.edges[inp] = []
            self.edges[inp].append(len(self.kernels)-1)

    def get_tensor_idx(self, name: str):
        return self.tensor_dict[name]

    def emit(self, start_idx: list[int], end_idx: list[int]):

        res = ""

        self.vis = [False for _ in self.tensors]
        self.kern_vis = [False for _ in self.kernels]
        for idx in start_idx:
            self.vis[idx] = True

        is_output_produced = True
        for idx in end_idx:
            if not self.vis[idx]:
                is_output_produced = False
                break
        

        while not is_output_produced:
            ready_kernels = []
            for idx in range(len(self.kernels)):
                if self.kern_vis[idx]:
                    continue
                ready = True
                for inp in self.kernels[idx].inps:
                    if not self.tensors[inp].is_weight and not self.vis[inp]:
                        ready = False
                        break
                if ready:
                    ready_kernels.append(idx)
            
            for idx in ready_kernels:
                self.kern_vis[idx] = True
                res += f"""
                    {self.kernels[idx].emit(self.tensors)}
                    """
                for out in self.kernels[idx].outs:
                    self.vis[out] = True

            is_output_produced = True
            for idx in end_idx:
                if not self.vis[idx]:
                    is_output_produced = False
                    break
        return res
    
    def emit_weight_struct(self):
        attrs = '\n'.join(
            [f"float* {t.name};" for t in self.tensors if t.is_weight]
        )
        return f"""
                typedef struct {{
                    {attrs}
                }} W;
            """
    
    def emit_weight_allocator(self):
        wts = [t for t in self.tensors if t.is_weight]
        return ''.join([f"""
            CUDA_CHECK(cudaMalloc(&w.{wt.emit()}, sizeof(float) * {wt.get_num_elements()}));
        """ for wt in wts])

if __name__ == "__main__":
    g = Graph()

    g.add_tensor("input", torch.rand(1), is_weight=True)
    g.add_tensor("a", torch.rand(12, 15), is_weight=True)
    g.add_tensor("b", torch.rand(1), is_weight=True)
    g.add_tensor("c", torch.rand(1), is_weight=True)
    g.add_kernel_op("add", 20, 
                    [
                        g.get_tensor_idx("input"),
                        g.get_tensor_idx("a"),
                    ],
                    [
                        g.get_tensor_idx("b"),
                    ],
                    consts=[10, 100,12]
                    )
    g.add_tensor_slice("a_slice", "a", 0)

    g.add_kernel_op("sub", 100, 
                    [
                        g.get_tensor_idx("a_slice"),
                        g.get_tensor_idx("b"),
                    ],
                    [
                        g.get_tensor_idx("c"),
                    ],
                    consts=[10, 1]
                    )

    print(g.emit([g.get_tensor_idx("input"), g.get_tensor_idx("a"),  g.get_tensor_idx("a_slice")], [g.get_tensor_idx("c")]))
    print(g.emit_weight_struct())
    print(g.emit_weight_allocator())
                        



