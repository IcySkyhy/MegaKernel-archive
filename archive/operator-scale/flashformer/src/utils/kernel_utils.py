import os
from pathlib import Path
import cheetah.api as ch
from cheetah.api import ty
from argparse import ArgumentParser
from torch.utils.cpp_extension import load
import importlib
from cheetah.index_tools import Indices, DimName
from flashtransformer.utils.safe_data_index_ptr_utils import SDIPConfig
from flashtransformer.utils.cuda_utils import printf
from flashtransformer.utils.barrier_utils.local_barrier import LocalBarrier
from typing import Optional
import torch


def error_check():
    ch.raw_stmt(
        """
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error:\\n");
                printf("%s\\n", cudaGetErrorString(err));
                exit(1);
            }
        }
        """
    )


def make_cond_print(
    cond: ch.Expr,
    debug: bool = False,
    local_barrier: Optional[LocalBarrier] = None,
):
    def smart_print(fmt: str, *args: ch.Expr | str, **kwargs: ch.Expr | str):
        if debug:
            with ch.if_(cond):
                printf(fmt, *args, **kwargs)
            if local_barrier is not None:
                local_barrier.wait()
            else:
                ch.raw_stmt("__syncwarp();")

    return smart_print


def shmem_helper(sdips: list[SDIPConfig]):
    shared_mem = 0
    for sdip in sdips:
        if sdip.memtype == "shared":
            shared_mem += sdip.get_size()
    return shared_mem


def safe_extern_shared(ix: Indices, ty: ch.Type, thread_idx: DimName):
    shared_ptr = ch.alloc_extern_shared(ty)
    with ch.if_(ix[thread_idx] == 0):
        ch.raw_stmt("assert ($shared_ptr != nullptr);", shared_ptr=shared_ptr)
    ch.raw_stmt("__syncwarp();")
    return shared_ptr


def launch_helper(
    kernel: ch.Func,
    n_blocks: int,
    n_threads: int,
    arg_exprs: list[ch.Expr],
    shared_mem: int,
):
    if shared_mem >= 228 * 1024:
        print(f"Shared memory size too large: {shared_mem}")
    kernel = kernel.void_ptr()
    args = ch.alloc_array(ty.ptr_mut(None), len(arg_exprs))
    for i, arg_expr in enumerate(arg_exprs):
        arg_stack = ch.alloc(arg_expr.type(), arg_expr)
        args[i] = arg_stack.cast(ty.ptr_mut(None))

    n_blocks_expr = ch.const(n_blocks, ty.u32)
    n_threads_expr = ch.const(n_threads, ty.u32)
    shared_mem_expr = ch.const(shared_mem, ty.u32)

    ch.raw_stmt(
        "cudaFuncSetAttribute($kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, $shared_mem);",
        kernel=kernel,
        shared_mem=shared_mem_expr,
    )
    error_check()
    ch.raw_stmt(
        "cudaLaunchCooperativeKernel($kernel, $n_blocks, $n_threads, $args, $shared_mem);",
        kernel=kernel,
        n_blocks=n_blocks_expr,
        n_threads=n_threads_expr,
        args=args,
        shared_mem=shared_mem_expr,
    )
    error_check()


def get_gen_path(dir=""):
    return Path(__file__).parent.parent.parent / dir / "cuda_gen"


def get_build_path(dir=""):
    return Path(__file__).parent.parent.parent / dir / "build"


standard_headers = [
    "cuda_runtime.h",
    "cuda_bf16.h",
    "cuda/barrier",
    "cooperative_groups.h",
    "cooperative_groups/reduce.h",
    "torch/extension.h",
    "torch/torch.h",
]


def run_test_codegen(
    kernel: ch.Func,
    kernel_name: str,
    headers: list[str] = [],
    bind: bool = False,
    dir: str = "tests",
):
    gen_dir = get_gen_path(dir)
    if not gen_dir.exists():
        os.makedirs(gen_dir)
    test_path = gen_dir / f"{kernel_name}.cu"
    # test if the file exists
    # if so, compare the contents
    # if not, write the contents
    write_file = True
    new_contents = ch.render(
        (kernel_name, kernel), headers=headers, bind=bind, bind_name=kernel_name
    )
    if test_path.exists():
        with open(test_path, "r") as f:
            existing_contents = f.read()
        if existing_contents == new_contents:
            write_file = False
    if write_file:
        with open(test_path, "w") as f:
            f.write(new_contents)
    return test_path


def import_codegen(kernel_name: str, dir="tests", args=None):
    import os

    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    if not get_build_path().exists():
        # make build path
        os.makedirs(get_build_path())
    try:
        native_module = load(
            name=kernel_name,
            sources=[get_gen_path(dir) / f"{kernel_name}.cu"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-arch=sm_90a",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-O3",
                # "-lineinfo",
                # "-keep",
            ],
            build_directory=get_build_path(dir),
        )
    except IndexError as e:
        print("Likely didn't detect GPU")
        raise e

    module = importlib.import_module(kernel_name)
    print(kernel_name)
    return getattr(module, kernel_name)


def launch_args(
    default_warmup=5, default_runs=10, default_loop=20, return_parser=False
):
    parser = ArgumentParser()
    parser.add_argument("--no-codegen", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n-warmup", type=int, default=default_warmup)
    parser.add_argument("--n-runs", type=int, default=default_runs)
    parser.add_argument("--n-loop", type=int, default=default_loop)
    parser.add_argument("--init", type=str, default="randn", choices=["randn", "ones"])
    if return_parser:
        return parser
    args = parser.parse_args()
    assert args.test or args.benchmark, "Must test or benchmark"
    args.init_fn = torch.randn if args.init == "randn" else torch.ones
    return args
