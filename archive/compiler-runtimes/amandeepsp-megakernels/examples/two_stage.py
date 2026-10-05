from cutlass.cute.runtime import from_dlpack
import torch
import cutlass.cute as cute
from quack.sync import Semaphore


@cute.jit
def stage_a(inp, temp, tile, thread):
    if thread < inp.shape[1]:
        temp[tile, thread] = inp[tile, thread] * cute.Float(2.0)


@cute.jit
def stage_b(temp, out, tile, thread):
    if thread < temp.shape[1]:
        out[tile, thread] = temp[tile, thread] + cute.Float(1.0)


@cute.kernel
def two_stage_kernel(
    inp: cute.Tensor,
    temp: cute.Tensor,
    out: cute.Tensor,
    events: cute.Tensor,
    epoch: cute.Int32,
    workers: cute.Int32,
):
    thread, _, _ = cute.arch.thread_idx()
    worker, _, _ = cute.arch.block_idx()

    event = Semaphore(lock_ptr=events.iterator, thread_idx=thread, sync="cta")

    if thread == 0:
        cute.printf("A worker={}", worker)

    stage_a(inp, temp, worker, thread)

    event.release_store(epoch, flag_offset=worker)
    source = worker - 1
    if worker == 0:
        source = workers - 1

    event.wait_eq(epoch, flag_offset=source)

    if thread == 0:
        cute.printf("B worker={}, source={}", worker, source)

    stage_b(temp, out, source, thread)


@cute.jit
def launch(
    inp: cute.Tensor,
    temp: cute.Tensor,
    out: cute.Tensor,
    events: cute.Tensor,
    epoch: cute.Int32,
    workers: cute.Int32,
):
    two_stage_kernel(inp, temp, out, events, epoch, workers).launch(
        grid=(workers, 1, 1),
        block=(128, 1, 1),
    )


W = 32
T = 128
epoch = 1
workers = W

input = torch.randn((W, T), dtype=torch.float, device="cuda")
temp = torch.empty_like(input, device="cuda")
output = torch.empty_like(input, device="cuda")
events = torch.zeros((W,), dtype=torch.int32, device="cuda")


compiled = cute.compile(
    launch,
    from_dlpack(input),
    from_dlpack(temp),
    from_dlpack(output),
    from_dlpack(events),
    cute.Int32(epoch),
    cute.Int32(workers),
)


compiled(input, temp, output, events, epoch, workers)
torch.cuda.synchronize()
torch.testing.assert_close(output, input * 2 + 1)
