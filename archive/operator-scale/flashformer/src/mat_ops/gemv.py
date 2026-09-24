from flashtransformer.utils.scope_fn_utils import SmartScopeFn
from flashtransformer.utils.safe_data_index_ptr_utils import (
    SDIPConfig,
    SafeDataIndexPtr,
)
from cheetah import api as ch
from cheetah.api import ty
from cheetah.index_tools import Dims, DimName, Indices
from flashtransformer.components.memory.sync_copy import ThreadVectorCopyFn
from flashtransformer.utils.cuda_utils import (
    get_default_elems_per_transaction,
    check_tile_size,
    group_reduce,
)


class WarpVecDotFn(SmartScopeFn):

    # called on two vectors, and returns their threadwise dot product

    def __init__(
        self, dims, vec_cfg: SDIPConfig, accum_type: ch.Type, thread_agg_idx: DimName
    ):
        self.sdip_configs = [vec_cfg, vec_cfg]
        super().__init__(dims, vec_cfg, accum_type, thread_agg_idx)

    def setup(self, vec_cfg: SDIPConfig, accum_type: ch.Type, thread_agg_idx: DimName):
        self.warp_iter_idx = self.dims.new_dim("warp_iter_idx")
        self.vec_cfg = vec_cfg
        self.accum_type = accum_type
        self.thread_agg_idx = thread_agg_idx
        self.elems_per_load = get_default_elems_per_transaction(
            self.vec_cfg.ptr_type, self.vec_cfg.memtype
        )
        self.vector_idx = self.dims.new_dim("vector_idx", self.elems_per_load)
        self.dims.eq(
            self.vec_cfg.idx,
            (self.warp_iter_idx, self.thread_agg_idx, self.vector_idx),
        )
        self.reg_vec_cfg = SDIPConfig(
            self.dims, self.vector_idx, self.vec_cfg.ptr_type, "register", "temp_vec"
        )

        self.vec_copy = ThreadVectorCopyFn(
            self.dims,
            self.vec_cfg,
            self.reg_vec_cfg,
            self.vector_idx,
        )

    def __call__(
        self,
        ix: Indices,
        vec1_ptr: SafeDataIndexPtr,
        vec2_ptr: SafeDataIndexPtr,
        tile,
    ):
        vec1_ptr, vec2_ptr = self.check_configs(vec1_ptr, vec2_ptr)
        return super().__call__(ix, vec1_ptr, vec2_ptr, tile)

    def generate(
        self,
        ix: Indices,
        vec1_ptr: SafeDataIndexPtr,
        vec2_ptr: SafeDataIndexPtr,
        tile,
    ):
        temp_vec1 = self.reg_vec_cfg.alloc_reg(ix)
        temp_vec2 = self.reg_vec_cfg.alloc_reg(ix)
        accum_val = ch.alloc(self.accum_type, 0.0)
        check_tile_size(tile, self.dims.size(self.thread_agg_idx))
        with ix.loop(self.warp_iter_idx):
            self.vec_copy(ix, vec1_ptr, temp_vec1)
            self.vec_copy(ix, vec2_ptr, temp_vec2)
            with ix.loop(self.vector_idx):
                accum_val.val += (temp_vec1.idx_get() * temp_vec2.idx_get()).cast(
                    self.accum_type
                )
        group_reduce(accum_val, tile)
        return accum_val

    def get_tile_size(self):
        return self.dims.size(self.thread_agg_idx)


class WarpGroupMatVecFn(SmartScopeFn):

    def __init__(
        self,
        dims: Dims,
        weight_cfg: SDIPConfig,
        in_cfg: SDIPConfig,
        out_cfg: SDIPConfig,
        warp_idx: DimName,
        lane_idx: DimName,
        d_in_idx: DimName,
        d_out_idx: DimName,
    ):
        self.sdip_configs = [weight_cfg, in_cfg, out_cfg]
        super().__init__(
            dims, weight_cfg, in_cfg, out_cfg, warp_idx, lane_idx, d_in_idx, d_out_idx
        )

    def setup(
        self,
        weight_cfg: SDIPConfig,
        in_cfg: SDIPConfig,
        out_cfg: SDIPConfig,
        warp_idx: DimName,
        lane_idx: DimName,
        d_in_idx: DimName,
        d_out_idx: DimName,
    ):
        self.weight_cfg = weight_cfg
        self.in_cfg = in_cfg
        self.out_cfg = out_cfg
        self.warp_idx = warp_idx
        self.lane_idx = lane_idx
        self.d_in_idx = d_in_idx
        self.d_out_idx = d_out_idx
        self.in_ptr_type = in_cfg.ptr_type
        self.out_ptr_type = out_cfg.ptr_type
        if (
            self.dims.size(d_out_idx) >= self.dims.size(warp_idx)
            and self.dims.size(d_out_idx) % self.dims.size(warp_idx) == 0
        ):
            self.d_out_iter_idx = self.dims.new_dim("d_out_iter_idx")
            # partition index is the same as warp index
            self.part_idx = self.warp_idx
            self.agg_idx = self.lane_idx
            self.dims.eq(self.d_out_idx, (self.d_out_iter_idx, self.part_idx))
        elif self.dims.size(warp_idx) % self.dims.size(d_out_idx) == 0:
            self.d_out_iter_idx = self.dims.new_dim("d_out_iter_idx", 1)
            self.part_idx = self.d_out_idx
            self.agg_idx = self.dims.new_dim("agg_idx")
            self.warp_subidx = self.dims.new_dim("warp_subidx")
            self.dims.eq(self.warp_idx, (self.part_idx, self.warp_subidx))
            self.dims.eq(self.agg_idx, (self.warp_subidx, self.lane_idx))
        else:
            raise ValueError(
                "Row index size must be either be a multiple of warp index size or a factor of it"
            )
        accum_type = self.out_cfg.ptr_type.try_to_ptr()[0]
        self.vecvec_fn = WarpVecDotFn(self.dims, self.in_cfg, accum_type, self.agg_idx)

    def __call__(
        self,
        ix: Indices,
        weight_ptr: SafeDataIndexPtr,
        in_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        tile,
        local_barrier,
    ):
        weight_ptr, in_ptr, out_ptr = self.check_configs(weight_ptr, in_ptr, out_ptr)
        return super().__call__(ix, weight_ptr, in_ptr, out_ptr, tile, local_barrier)

    def generate(
        self,
        ix: Indices,
        weight_ptr: SafeDataIndexPtr,
        in_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        tile,
        local_barrier,
    ):
        with ix.loop(self.d_out_iter_idx):
            weight_vec = weight_ptr.subidx(self.d_out_idx, self.d_in_idx)
            out_ptr.idx_set(self.vecvec_fn(ix, weight_vec, in_ptr, tile).val)
        local_barrier.wait()

    def get_tile_size(self):
        return self.vecvec_fn.get_tile_size()
