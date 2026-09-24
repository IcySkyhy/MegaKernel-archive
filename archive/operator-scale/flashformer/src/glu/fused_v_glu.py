from typing import Any, Optional, Dict

import cheetah.api as ch
from cheetah.api import ty
import cheetah.index_tools as ixt
from cheetah.index_tools import Indices, DimName, Dims

from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from flashtransformer.utils.scope_fn_utils import ProducerConsumerSmartFn
from flashtransformer.components.memory import WGSyncCopyFn
from flashtransformer.components.memory.async_copy import AsyncBarrierGTSCopyFn
from flashtransformer.components.memory.async_red import AsyncBulkGroupSTGRedAddFn
from flashtransformer.components.mat_ops import WarpGroupMatVecFn
from flashtransformer.utils.memory_pipeline import MemoryPipeline


class FusedGLUGEMV(ProducerConsumerSmartFn):
    """Fused GLU (Gated Linear Unit) with GEMV computation.
    
    This performs the GLU out operation with reduction:
    y += GEMV(W_out, glu_activation)
    where glu_activation comes from previous GLU computation.
    The GEMV is computed in chunks and reduced-added to global memory.
    """
    
    def __init__(
        self,
        dims: Dims,
        in_cfg: SDIPConfig,  # Input from GLU activation
        out_cfg: SDIPConfig,  # Final output (for reduction)
        w_out_cfg: SDIPConfig,  # Output projection weights
        mem_pipeline: MemoryPipeline,
        block_idx: DimName,
        d_in_idx: DimName,  # GLU hidden dimension
        d_out_idx: DimName,  # Output dimension (d_model)
    ):
        # Set up SDIPConfig lists for validation
        self.consumer_sdip_configs = [in_cfg, out_cfg]
        self.producer_sdip_configs = [w_out_cfg]
        
        super().__init__(
            dims,
            in_cfg,
            out_cfg,
            w_out_cfg,
            mem_pipeline,
            block_idx,
            d_in_idx,
            d_out_idx,
        )
    
    def setup(
        self,
        in_cfg: SDIPConfig,
        out_cfg: SDIPConfig,
        w_out_cfg: SDIPConfig,
        mem_pipeline: MemoryPipeline,
        block_idx: DimName,
        d_in_idx: DimName,
        d_out_idx: DimName,
    ) -> None:
        """Set up the fused GLU GEMV function with reduction.
        
        Args:
            in_cfg: Configuration for GLU activation input
            out_cfg: Configuration for final output (global memory)
            w_out_cfg: Configuration for output projection weights
            mem_pipeline: Memory pipeline for producer-consumer coordination
            block_idx: Dimension name for block index
            d_in_idx: Dimension name for GLU hidden dimension
            d_out_idx: Dimension name for output dimension
        """
        # Initialize the ProducerConsumerSmartFn
        super().setup(mem_pipeline)
        
        # Store SDIPConfig instances
        self.in_cfg = in_cfg
        self.out_cfg = out_cfg
        self.w_out_cfg = w_out_cfg
        
        # Store memory pipeline and its components
        self.mem_pipeline = mem_pipeline
        self.buffer_idx = mem_pipeline.buffer_idx
        self.intra_buffer_idx = mem_pipeline.intra_buffer_idx
        self.buffer_cfg = mem_pipeline.buffer_cfg
        self.intra_buffer_cfg = mem_pipeline.intra_buffer_cfg
        self.lane_idx = mem_pipeline.lane_idx
        
        # Get thread indices from memory pipeline
        self.consumer_thread_idx = mem_pipeline.consumer_thread_idx
        self.producer_thread_idx = mem_pipeline.producer_thread_idx
        
        # Store dimension names
        self.block_idx = block_idx
        self.d_in_idx = d_in_idx  # GLU hidden dimension
        self.d_out_idx = d_out_idx  # Output dimension (d_model)
        
        # Set up pointer types - use out_cfg's type for consistency
        self.ptr_type = out_cfg.ptr_type
        self.accum_ptr_type = ty.f32  # Use f32 for accumulation
        
        # Setup dimensions for computation
        self.per_block_channel_idx = self.dims.new_dim("per_block_channel_idx")
        self.dims.eq(self.d_out_idx, (self.block_idx, self.per_block_channel_idx))
        
        # Buffer output row index
        self.buffer_out_row_idx = self.dims.new_dim("buffer_out_row_idx")
        self.dims.eq(
            self.intra_buffer_idx, (self.buffer_out_row_idx, self.d_in_idx)
        )
        
        # Setup iteration for computation
        self.glu_out_iter = self.dims.new_dim("glu_out_iter")
        self.dims.eq(
            self.per_block_channel_idx, (self.glu_out_iter, self.buffer_out_row_idx)
        )
        
        # Create SDIPConfig for shared input buffer (GLU activation)
        self.shared_inp_cfg = SDIPConfig(
            self.dims, self.d_in_idx, self.ptr_type, "shared", "glu_activation"
        )
        
        # Create SDIPConfig for shared output accumulation buffer
        self.shared_out_cfg = SDIPConfig(
            self.dims,
            idx=self.per_block_channel_idx,
            ptr_type=self.ptr_type,
            memtype="shared",
            name="glu_out_buffer",
        )
        
        # Create SDIPConfig for iteration output (intermediate results)
        self.iter_out_cfg = SDIPConfig(
            self.dims,
            idx=self.buffer_out_row_idx,
            ptr_type=self.ptr_type,
            memtype="shared",
            name="iter_glu_out",
        )
        
        # Setup input load function
        self.inp_load_fn = WGSyncCopyFn(
            dims=self.dims,
            src_cfg=in_cfg,
            dst_cfg=self.shared_inp_cfg,
            thread_idx=self.consumer_thread_idx,
            cpy_idx=self.d_in_idx,
        )
        
        # Setup GEMV computation function
        self.glu_out_gemv_fn = WarpGroupMatVecFn(
            dims=self.dims,
            weight_cfg=self.intra_buffer_cfg,
            in_cfg=self.shared_inp_cfg,
            out_cfg=self.iter_out_cfg,
            warp_idx=self.consumer_warp_idx,
            lane_idx=self.lane_idx,
            d_in_idx=self.d_in_idx,
            d_out_idx=self.buffer_out_row_idx,
        )
        
        # Setup bulk reduction function for accumulating to global memory
        # Note: AsyncBulkGroupSTGRedAddFn requires both configs to have same ptr type
        # Create a version of out_cfg with matching ptr_type for the reduction
        self.glu_out_reduce_fn = AsyncBulkGroupSTGRedAddFn(
            dims=self.dims,
            src_cfg=self.shared_out_cfg,
            dst_cfg=out_cfg,
            thread_idx=self.consumer_thread_idx,
            cpy_idx=self.per_block_channel_idx,
        )
        
        # Setup producer weight copy function
        self.w_out_copy_in_fn = AsyncBarrierGTSCopyFn(
            dims=self.dims,
            src_cfg=self.w_out_cfg,
            dst_cfg=self.buffer_cfg,
            thread_idx=self.producer_thread_idx,
            cpy_idx=self.intra_buffer_idx,
        )
    
    def setup_shmem(self, ix: Indices, shared_ptr: ch.Expr) -> ch.Expr:
        """Set up shared memory for GLU buffers.
        
        Args:
            ix: Cheetah indices
            shared_ptr: Shared memory pointer
            
        Returns:
            Updated shared_ptr
        """
        # Allocate shared memory for GLU activation input
        self.shared_inp_ptr, shared_ptr = self.shared_inp_cfg.alloc_dynamic_shared(
            ix, shared_ptr
        )
        # Allocate shared memory for output accumulation
        self.shared_out_ptr, shared_ptr = self.shared_out_cfg.alloc_dynamic_shared(
            ix, shared_ptr
        )
        
        return shared_ptr
    
    def get_shmem_size(self) -> int:
        """Get the shared memory size required.
        
        Returns:
            Size of required shared memory in bytes
        """
        return self.shared_inp_cfg.get_size() + self.shared_out_cfg.get_size()
    
    def child_consume(
        self,
        ix: Indices,
        global_glu_activation: SafeDataIndexPtr,
        global_output: SafeDataIndexPtr,
        groups: Dict[str, Any],
    ) -> None:
        """Consumer part of the fused GLU GEMV computation with reduction.
        
        Args:
            ix: Cheetah indices
            global_glu_activation: Global GLU activation input
            global_output: Global output buffer (for reduction)
            groups: Dictionary of cooperative groups
        """
        # Validate input pointers
        global_glu_activation, global_output = self.check_consumer_configs(
            global_glu_activation, global_output
        )
        
        # Load GLU activation to shared memory
        self.inp_load_fn(
            ix,
            src_ptr=global_glu_activation,
            dst_ptr=self.shared_inp_ptr,
            local_barrier=self.mem_pipeline.consumer_barrier,
        )
        
        # Get appropriate cooperative group tile
        glu_tile = groups[f"tile{self.glu_out_gemv_fn.get_tile_size()}"]
        
        # Process GEMV computation in chunks
        with ix.loop(self.glu_out_iter):
            # Get iteration specific pointer for output
            self.iter_glu_out_ptr = self.shared_out_ptr.subidx(
                self.glu_out_iter, self.buffer_out_row_idx
            )
            
            # Get buffer for output projection weights
            w_out_shared, stage = self.mem_pipeline.consumer_get_shared_buffer("glu_out")
            
            # Compute GEMV: out = W_out @ glu_activation
            self.glu_out_gemv_fn(
                ix,
                weight_ptr=w_out_shared,
                in_ptr=self.shared_inp_ptr,
                out_ptr=self.iter_glu_out_ptr,
                tile=glu_tile,
                local_barrier=self.mem_pipeline.consumer_barrier,
            )
            
            # Release buffer
            self.mem_pipeline.consumer_release_shared_buffer(stage)
        
        # Reduce-add accumulated results to global memory
        # This performs: global_output += shared_out
        self.glu_out_reduce_fn(
            ix,
            self.shared_out_ptr,
            global_output,
        )
        
        # Wait for async reduction to complete
        self.glu_out_reduce_fn.wait(ix, self.mem_pipeline.consumer_barrier)
    
    def child_produce(
        self,
        ix: Indices,
        buf_w_out: SafeDataIndexPtr,
    ) -> None:
        """Producer part of the fused GLU GEMV computation.
        
        Args:
            ix: Cheetah indices
            buf_w_out: Buffer for output projection weights
        """
        # Validate weight pointer
        buf_w_out = self.check_producer_configs(buf_w_out)
        
        # Copy weights in chunks
        with ix.loop(self.glu_out_iter):
            self.mem_pipeline.producer_copy(buf_w_out, self.w_out_copy_in_fn)
