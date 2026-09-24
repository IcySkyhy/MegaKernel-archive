import cheetah.api as ch
from cheetah.api import ty

num_barriers = 16
already_initialized = [False] * num_barriers
track_inititalized = True


def reset_local_barriers():
    global already_initialized
    for i in range(num_barriers):
        already_initialized[i] = False


class LocalBarrier:
    def __init__(self, id: int, size: int):
        self.id = id
        global track_inititalized
        if track_inititalized:
            global already_initialized
            if already_initialized[id]:
                print(
                    "This local barrier has already been initialized. This might lead to errors"
                )
            already_initialized[id] = True
        self.size = size
        self.id_expr = ch.const(id, ty.i32)
        self.arrive_and_wait = self.wait

    def arrive(self):
        ch.asm_volatile(f"bar.arrive $id, {self.size};", id=self.id_expr)

    def wait(self):
        ch.asm_volatile(f"bar.sync $id, {self.size};", id=self.id_expr)


class LocalBarrierManager:
    def __init__(self):
        # local barriers are indexed by integer and represented by the number of threads that is checks for
        self.local_barriers = [-1] * 16
        # shared barriers are indexed by name and represented by a str (name)
        self.local_barrier_idx = 2  # save 0 and 1 for cuda code

    def get_local_barrier(self, size: int) -> LocalBarrier:
        assert self.local_barrier_idx < len(self.local_barriers)
        self.local_barriers[self.local_barrier_idx] = size
        self.local_barrier_idx += 1
        return LocalBarrier(self.local_barrier_idx - 1, size)

    def arrive(self, id_expr: ch.Expr, size: int):
        ch.asm(f"bar.arrive $id, {size};", id=id_expr)

    def sync(self, id_expr: ch.Expr, size: int):
        ch.asm(f"bar.sync $id, {size};", id=id_expr)
