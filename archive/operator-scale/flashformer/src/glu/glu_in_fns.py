from cheetah import api as ch
from cheetah.api import ty
from cheetah.index_tools import Dims, DimName, Indices

from flashtransformer.components.mat_ops.gemv import WarpVecDotFn, WarpGroupMatVecFn
from flashtransformer.utils.scope_fn_utils import ProducerConsumerSmartFn, SmartScopeFn
from flashtransformer.utils.cuda_utils import (
    check_tile_size,
    Tile,
    mul_swiglu,
    mul_swiglu_bf16,
)
from flashtransformer.utils.kernel_utils import make_cond_print
from flashtransformer.utils.safe_data_index_ptr_utils import (
    SDIPConfig,
    SafeDataIndexPtr,
)
from flashtransformer.utils.barrier_utils import LocalBarrier


class WarpGroupGluInFn(WarpGroupMatVecFn):

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
        self.sdip_configs = [weight_cfg, weight_cfg, in_cfg, out_cfg]
        super(SmartScopeFn, self).__init__(
            dims, weight_cfg, in_cfg, out_cfg, warp_idx, lane_idx, d_in_idx, d_out_idx
        )

    def __call__(
        self,
        ix: Indices,
        weight_in_ptr: SafeDataIndexPtr,
        weight_gate_ptr: SafeDataIndexPtr,
        in_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        tile: Tile,
        local_barrier: LocalBarrier,
        debug=False,
    ):
        weight_in_ptr, weight_gate_ptr, in_ptr, out_ptr = self.check_configs(
            weight_in_ptr, weight_gate_ptr, in_ptr, out_ptr
        )
        self.debug_print = make_cond_print(
            ch.and_(ix[self.warp_idx] == 0, ix[self.lane_idx] == 0), debug
        )
        super(SmartScopeFn, self).__call__(
            ix, weight_in_ptr, weight_gate_ptr, in_ptr, out_ptr, tile, local_barrier
        )

    def generate(
        self,
        ix: Indices,
        weight_in_ptr: SafeDataIndexPtr,
        weight_gate_ptr: SafeDataIndexPtr,
        in_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        tile: Tile,
        local_barrier: LocalBarrier,
    ):
        with ix.loop(self.d_out_iter_idx):
            weight_in_vec = weight_in_ptr.subidx(self.d_out_idx, self.d_in_idx)
            weight_gate_vec = weight_gate_ptr.subidx(self.d_out_idx, self.d_in_idx)
            in_val = self.vecvec_fn(ix, weight_in_vec, in_ptr, tile)
            gate_val = self.vecvec_fn(ix, weight_gate_vec, in_ptr, tile)
            out_val = mul_swiglu(in_val, gate_val)
            self.debug_print(
                "in_val: %f, gate_val: %f, out_val: %f",
                in_val.val,
                gate_val.val,
                out_val,
            )
            out_ptr.idx_set(out_val)
        local_barrier.wait()
