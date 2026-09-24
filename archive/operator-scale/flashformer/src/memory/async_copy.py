from cheetah.index_tools import DimName, Dims, Indices
from cheetah.api import ty
import cheetah.api as ch
from flashtransformer.utils import (
    SharedMBarrier,
    SmartScopeFn,
    SDIPConfig,
    SafeDataIndexPtr,
)


class AsyncBarrierGTSCopyFn(SmartScopeFn):
    def __init__(
        self,
        dims: Dims,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
    ):
        self.sdip_configs = [src_cfg, dst_cfg]
        super().__init__(
            dims,
            src_cfg,
            dst_cfg,
            thread_idx,
            cpy_idx,
        )

    def setup(
        self,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
    ):
        self.src_cfg = src_cfg
        self.dst_cfg = dst_cfg
        self.thread_idx = thread_idx
        self.cpy_idx = cpy_idx
        assert src_cfg.memtype == "global" and dst_cfg.memtype == "shared"
        assert src_cfg.get_elem_size() == dst_cfg.get_elem_size()

    def __call__(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
        barrier: SharedMBarrier,
    ):
        src_ptr, dst_ptr = self.check_configs(src_ptr, dst_ptr)
        super().__call__(ix, src_ptr, dst_ptr, barrier)

    def generate(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
        barrier: SharedMBarrier,
    ):

        assert isinstance(barrier, SharedMBarrier)
        size = ix.size(self.cpy_idx) * dst_ptr.get_elem_size()
        ix.set_index(self.cpy_idx, ch.const(0, ty.i32))
        assert size % 16 == 0
        with ch.if_(ix[self.thread_idx] == 0):
            ch.asm(
                f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$dst_ptr], [$src_ptr], {size}, [$barrier];",
                dst_ptr=dst_ptr.idx_offset(),
                src_ptr=src_ptr.idx_offset(),
                barrier=barrier.ptr,
            )
            ch.asm(
                f"mbarrier.expect_tx.relaxed.cta.shared.b64 [$barrier], {size};",
                barrier=barrier.ptr,
            )
        ch.raw_stmt("__syncwarp();")


class AsyncBulkGroupSTGCopyFn(SmartScopeFn):
    def __init__(
        self,
        dims: Dims,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
        l2_cache_hint: bool = False,
    ):
        self.sdip_configs = [src_cfg, dst_cfg]
        super().__init__(dims, src_cfg, dst_cfg, thread_idx, cpy_idx, l2_cache_hint)

    def setup(
        self,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
        l2_cache_hint: bool = False,
    ):
        self.src_cfg = src_cfg
        self.dst_cfg = dst_cfg
        self.thread_idx = thread_idx
        self.cpy_idx = cpy_idx
        self.l2_cache_hint = l2_cache_hint
        assert src_cfg.memtype == "shared" and dst_cfg.memtype == "global"
        assert src_cfg.get_elem_size() == dst_cfg.get_elem_size()

    def __call__(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
    ):
        src_ptr, dst_ptr = self.check_configs(src_ptr, dst_ptr)
        super().__call__(ix, src_ptr, dst_ptr)

    def generate(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
    ):
        size = ix.size(self.cpy_idx) * src_ptr.get_elem_size()
        ix.set_index(self.cpy_idx, ch.const(0, ty.i32))
        assert size % 16 == 0
        with ch.if_(ix[self.thread_idx] == 0):
            l2_cache_hint = ".L2::cache_hint" if self.l2_cache_hint else ""
            ch.asm(
                f"cp.async.bulk.global.shared::cta.bulk_group{l2_cache_hint} [$dst_ptr], [$src_ptr], {size};",
                dst_ptr=dst_ptr.idx_offset(),
                src_ptr=src_ptr.idx_offset(),
            )
            ch.asm("cp.async.commit_group;")
        ch.raw_stmt("__syncwarp();")

    def wait(self, ix, local_barrier):
        with ix.scope(self.scope):
            with ch.if_(ix[self.thread_idx] == 0):
                ch.asm("cp.async.wait_all;")
            local_barrier.wait()
