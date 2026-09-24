from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from cheetah.index_tools import DimName, Indices, Dims
import cheetah.api as ch
from cheetah.api import ty
from flashtransformer.utils.scope_fn_utils import SmartScopeFn
from flashtransformer.utils.cuda_utils import (
    get_cast_dtype,
    get_default_elems_per_transaction,
)
from flashtransformer.utils.index_utils import make_overflow_idx, get_nondivisible_dim
from typing import Optional
from math import gcd


class ThreadVectorCopyFn(SmartScopeFn):

    def __init__(
        self,
        dims: Dims,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        cpy_idx: DimName,
        volatile: bool = False,
        override_types: bool = False,
    ):
        self.sdip_configs = [src_cfg, dst_cfg]
        super().__init__(
            dims,
            src_cfg,
            dst_cfg,
            cpy_idx,
            volatile,
            override_types,
        )

    def setup(
        self,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        cpy_idx: DimName,
        volatile: bool = False,
        override_types: bool = False,
    ):
        self.src_cfg = src_cfg
        self.dst_cfg = dst_cfg
        self.cpy_idx = cpy_idx
        if src_cfg.get_elem_size() * self.dims.size(cpy_idx) > 16:
            self.loop = True
            self.iter_idx = self.dims.new_dim("iter_idx")
            self.cast_size = gcd(
                self.dims.size(cpy_idx), int(16 / src_cfg.get_elem_size())
            )
            self.vector_idx = self.dims.new_dim("vector_idx", self.cast_size)
            self.dims.eq(cpy_idx, (self.iter_idx, self.vector_idx))
            self.cast_dtype = get_cast_dtype(self.cast_size * src_cfg.get_elem_size())
        else:
            self.loop = False
            self.cast_dtype = get_cast_dtype(
                self.dims.size(cpy_idx) * src_cfg.get_elem_size()
            )
        self.volatile = volatile
        assert (
            src_cfg.get_elem_type() == dst_cfg.get_elem_type() or override_types
        ), "src and dst must have same ptr type if override_types is False"

    def __call__(
        self, ix: Indices, src_ptr: SafeDataIndexPtr, dst_ptr: SafeDataIndexPtr
    ):
        src_ptr, dst_ptr = self.check_configs(src_ptr, dst_ptr)
        super().__call__(ix, src_ptr, dst_ptr)

    def generate(
        self, ix: Indices, src_ptr: SafeDataIndexPtr, dst_ptr: SafeDataIndexPtr
    ):
        if self.loop:
            ix.set_index(self.vector_idx, 0)
            with ix.loop(self.iter_idx):
                self.copy(ix, src_ptr, dst_ptr, self.cast_dtype)
        else:
            ix.set_index(self.cpy_idx, 0)
            self.copy(ix, src_ptr, dst_ptr, self.cast_dtype)

    def copy(self, ix, src_ptr, dst_ptr, cast_dtype):
        if src_ptr.is_const():
            cast_dst_ptr = dst_ptr.idx_offset().cast(ty.ptr_mut(cast_dtype))
            cast_src_ptr = src_ptr.idx_offset().cast(ty.ptr_const(cast_dtype))
            cast_dst_ptr[0] = cast_src_ptr[0]
        elif self.volatile:
            assert cast_dtype != ty.i4, "volatile copy of int4 is not supported"
            cast_dst_ptr = dst_ptr.idx_offset().cast(ty.ptr_mut(cast_dtype))
            cast_src_ptr = src_ptr.idx_offset().cast(ty.ptr_volatile(cast_dtype))
            cast_dst_ptr[0] = cast_src_ptr[0]
        else:
            cast_dst_ptr = dst_ptr.idx_offset().cast(ty.ptr_mut(cast_dtype))
            cast_src_ptr = src_ptr.idx_offset().cast(ty.ptr_mut(cast_dtype))
            cast_dst_ptr[0] = cast_src_ptr[0]

    def zero_out(self, ix, dst_ptr):
        with ix.scope(self.scope):
            dst_ptr_type = dst_ptr.type()
            assert (
                dst_ptr_type == self.dst_cfg.ptr_type
            ), f"dst_ptr: {dst_ptr} has type {dst_ptr_type}, expected {self.dst_cfg.ptr_type}"
            if self.loop:
                ix.set_index(self.vector_idx, 0)
                with ix.loop(self.iter_idx):
                    cast_dst_ptr = dst_ptr.idx_offset().cast(
                        ty.ptr_mut(self.cast_dtype)
                    )
                    cast_dst_ptr[0] = 0
            else:
                ix.set_index(self.cpy_idx, 0)
                cast_dst_ptr = dst_ptr.idx_offset().cast(ty.ptr_mut(self.cast_dtype))
                cast_dst_ptr[0] = 0


class WGSyncCopyFn(SmartScopeFn):
    def __init__(
        self,
        dims: Dims,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
        elems_per_load: Optional[int] = None,
        volatile: bool = False,
        use_whole_warp: Optional[bool] = None,
        warp_size: int = 32,
    ):
        self.sdip_configs = [src_cfg, dst_cfg]
        super().__init__(
            dims,
            src_cfg,
            dst_cfg,
            thread_idx,
            cpy_idx,
            elems_per_load,
            volatile,
            use_whole_warp,
            warp_size,
        )

    def setup(
        self,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
        elems_per_load: Optional[int] = None,
        volatile: bool = False,
        use_whole_warp: Optional[bool] = True,
        warp_size: int = 32,
    ):
        self.src_cfg = src_cfg
        self.dst_cfg = dst_cfg
        self.thread_idx = thread_idx
        self.cpy_idx = cpy_idx
        self.volatile = volatile
        self.use_whole_warp = use_whole_warp
        self.warp_size = warp_size
        # validate global loads are transaction sizd

        # Validate that copy size is divisible by elements per load
        if elems_per_load is None:
            # solve for elems_per_load
            if use_whole_warp:
                init_elems_per_load = self.dims.size(self.cpy_idx) // self.warp_size
            else:
                init_elems_per_load = self.dims.size(self.cpy_idx)
            # max(
            #     self.dims.size(self.cpy_idx) // self.dims.size(self.thread_idx), 1
            # )
            #
            if src_cfg.memtype == "global" or dst_cfg.memtype == "global":
                memtype = "global"
            elif src_cfg.memtype == "shared" or dst_cfg.memtype == "shared":
                memtype = "shared"
            else:
                memtype = "register"
            elems_per_load = gcd(
                init_elems_per_load,
                get_default_elems_per_transaction(self.src_cfg.ptr_type, memtype),
            )
        # Calculate total threads and elements that can be copied per iteration
        self.vector_idx = self.dims.new_dim("vector_idx", elems_per_load)
        check_dim = get_nondivisible_dim(
            self.dims, self.cpy_idx, (self.thread_idx, self.vector_idx)
        )
        self.iter_idx = self.dims.new_dim("iter_idx")
        # TODO: try doing smaller per warp transactions
        if check_dim is self.thread_idx:
            self.overflow_idx, self.overflow_check_and_set = make_overflow_idx(
                self.dims,
                self.cpy_idx,
                (self.iter_idx, self.thread_idx, self.vector_idx),
                self.thread_idx,
                self.iter_idx,
            )
            self.needs_to_check_size = True
        elif check_dim is None:
            self.dims.eq(
                self.cpy_idx, (self.iter_idx, self.thread_idx, self.vector_idx)
            )
            self.needs_to_check_size = False
        else:
            raise ValueError(f"Invalid check dimension: {check_dim}")

        # Create copy function for each thread's chunk
        self.thread_copy = ThreadVectorCopyFn(
            self.dims,
            src_cfg=self.src_cfg,
            dst_cfg=self.dst_cfg,
            cpy_idx=self.vector_idx,
            volatile=self.volatile,
        )

    def __call__(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
        local_barrier,
    ):
        src_ptr, dst_ptr = self.check_configs(src_ptr, dst_ptr)
        super().__call__(ix, src_ptr, dst_ptr, local_barrier)

    def generate(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
        local_barrier,
    ):

        # Copy either multi-iteration or single iteration
        if self.dims.size(self.iter_idx) > 1:
            # Multi-iteration copy
            with ix.loop(self.iter_idx):
                if self.needs_to_check_size:
                    with self.overflow_check_and_set(ix):
                        self.thread_copy(ix, src_ptr, dst_ptr)
                else:
                    self.thread_copy(ix, src_ptr, dst_ptr)
        else:
            # Single iteration copy - was missing this clause!
            if self.needs_to_check_size:
                with self.overflow_check_and_set(ix):
                    self.thread_copy(ix, src_ptr, dst_ptr)
            else:
                self.thread_copy(ix, src_ptr, dst_ptr)

        # Make sure all threads complete before continuing
        local_barrier.wait()
