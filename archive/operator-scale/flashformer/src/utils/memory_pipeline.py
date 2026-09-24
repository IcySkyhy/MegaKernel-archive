import cheetah.api as ch
from cheetah.api import ty
from cheetah.index_tools import Indices, Dims, DimName
from typing import Tuple, Optional, Type, Callable
from .safe_data_index_ptr_utils import SafeDataIndexPtr, SDIPConfig
from .barrier_utils.local_barrier import LocalBarrierManager
from .barrier_utils.pipeline_manager import PipelineManager
from .barrier_utils.shared_mbarrier import SharedMBarrierArray, SharedMBarrierToken
from .barrier_utils.global_barrier import GlobalBarrierManager
from .cuda_utils import get_elem_size


class MemoryPipeline:
    """
    Memory pipeline utility for efficient producer-consumer memory management.

    This class manages the shared memory pipeline for efficient producer-consumer
    pattern in CUDA kernels, providing buffer allocation, synchronization, and
    memory management utilities.
    """

    def __init__(
        self,
        dims: Dims,
        pipeline_stages: int,
        warp_idx: DimName,
        lane_idx: DimName,
        consumer_warp_idx: DimName,
        producer_warp_idx: DimName,
        local_barrier_manager: LocalBarrierManager,
        chunk_size: int,
        ptr_type: ch.Type,
        block_idx: Optional[DimName] = None,
    ):
        """
        Initialize the memory pipeline.

        Args:
            dims: Cheetah dimensions object
            pipeline_stages: Number of pipeline stages in the producer-consumer pipeline
            warp_idx: Warp index dimension for the full threadgroup
            lane_idx: Lane index dimension inside a warp
            consumer_warp_idx: Warp index for consumers
            producer_warp_idx: Warp index for producers
            chunk_size: Number of elements per buffer chunk (size of `intra_buffer_idx`)
            ptr_type: Element type for the pipeline buffers (e.g., ty.bf16)
            block_idx: Optional block index dimension
        """
        self.dims = dims
        self.pipeline_stages = pipeline_stages
        self.warp_idx = warp_idx
        self.lane_idx = lane_idx
        self.consumer_warp_idx = consumer_warp_idx
        self.producer_warp_idx = producer_warp_idx
        self.thread_idx = self.dims.new_dim_with_eq(
            "thread_idx", (self.warp_idx, self.lane_idx)
        )
        self.consumer_thread_idx = self.dims.new_dim_with_eq(
            "consumer_thread_idx", (self.consumer_warp_idx, self.lane_idx)
        )
        self.producer_thread_idx = self.dims.new_dim_with_eq(
            "producer_thread_idx", (self.producer_warp_idx, self.lane_idx)
        )
        self.inter_buffer_idx = dims.new_dim("inter_buffer_idx", pipeline_stages)
        self.intra_buffer_idx = dims.new_dim("intra_buffer_idx", chunk_size)
        self.block_idx = block_idx
        self.ptr_type = ptr_type

        # Set up pipeline infrastructure
        self.local_barrier_manager = local_barrier_manager
        self.pipeline_manager = PipelineManager(
            self.dims,
            self.pipeline_stages,
            self.dims.size(self.thread_idx),
            self.local_barrier_manager,
        )

        # Create buffer index
        self.buffer_idx = self.dims.new_dim_with_eq(
            "buffer_idx", (self.inter_buffer_idx, self.intra_buffer_idx)
        )
        self.buffer_cfg = SDIPConfig(
            self.dims,
            self.buffer_idx,
            ty.ptr_mut(self.ptr_type),
            "shared",
            "pipeline_buffer",
        )
        self.intra_buffer_cfg = SDIPConfig(
            self.dims,
            self.intra_buffer_idx,
            ty.ptr_mut(self.ptr_type),
            "shared",
            "intra_buffer",
        )

        self.stage = "uninitialized"

    def get_total_buffer_cfg(self) -> SDIPConfig:
        return self.buffer_cfg

    def get_single_buffer_cfg(self) -> SDIPConfig:
        return self.intra_buffer_cfg

    def initialize(
        self, ix: Indices, shared_ptr: ch.Expr
    ) -> Tuple[SafeDataIndexPtr, ch.Expr]:
        """
        Initialize memory pipeline resources.

        Args:
            ix: Cheetah indices object
            shared_ptr: Shared memory pointer to use for the pipeline buffer

        Returns:
            Tuple of (buffer, remaining_shared_ptr)
        """
        # Initialize consumer barrier

        assert (
            self.stage == "uninitialized"
        ), f"MemoryPipeline in state {self.stage}, should be uninitialized"
        self.consumer_barrier = self.local_barrier_manager.get_local_barrier(
            self.dims.size(self.consumer_thread_idx)
        )
        self.ix = ix

        # Initialize mbarriers for async operations
        self.raw_mbars = ch.alloc_shared_array(ty.u64, self.pipeline_stages)
        self.mbars = SharedMBarrierArray(
            self.dims, self.raw_mbars, self.pipeline_stages
        )

        # Allocate from provided shared memory
        buffer, remaining_shared_ptr = self.buffer_cfg.alloc_dynamic_shared(
            ix, shared_ptr
        )
        self.buffer = buffer

        # Initialize pipeline components
        self.pipeline_manager.pipeline_init()

        # Initialize memory barriers
        # Convert cheetah expression to int for barrier initialization
        thread_count = self.dims.size(self.consumer_thread_idx)
        self.mbars.init(ix, thread_count)

        # Synchronize threads
        ch.raw_stmt("__syncthreads();")

        self.stage = "initialized"

        return buffer, remaining_shared_ptr

    def get_shared_size(self) -> int:
        """
        Calculate required shared memory size for the pipeline buffer.

        Returns:
            Size of the pipeline buffer in bytes
        """
        # Pipeline buffer size
        return get_elem_size(self.ptr_type) * self.dims.size(self.buffer_idx)

    #
    # Consumer functions
    #

    def consumer_get_shared_buffer(self, name: str) -> Tuple[SafeDataIndexPtr, ch.Expr]:
        """
        [CONSUMER] Get a shared buffer from the pipeline for consumption.

        Args:
            ix: Cheetah indices object
            name: Buffer name for debugging

        Returns:
            Tuple of (shared_buffer, stage)
        """
        # Wait for available buffer
        assert (
            self.stage == "ready"
        ), f"MemoryPipeline in state {self.stage}, should be ready"
        stage = self.pipeline_manager.pipeline_consumer_wait()
        with self.ix.scope():
            self.ix.set_index(self.inter_buffer_idx, stage)
            # Wait on memory barrier
            self.mbars.wait(stage, self.mbars.arrive(stage))

            # Create SafeDataIndexPtr for buffer access
            shared_buffer = self.buffer.subidx(
                self.inter_buffer_idx, self.intra_buffer_idx, f"shared_{name}_buffer"
            )
        return shared_buffer, stage

    def consumer_release_shared_buffer(self, stage: ch.Expr):
        """
        [CONSUMER] Release a shared buffer back to the pipeline after consumption.

        Args:
            stage: Pipeline stage to release
        """
        assert (
            self.stage == "ready" or self.stage == "initialized"
        ), f"MemoryPipeline in state {self.stage}, should be ready or initialized (if during readying)"
        self.pipeline_manager.pipeline_consumer_arrive(stage)

    def consumer_pipeline_ready(self, ix: Indices):
        assert (
            self.stage == "initialized"
        ), f"MemoryPipeline in state {self.stage}, should be initialized"
        with ix.loop(self.inter_buffer_idx):
            self.consumer_release_shared_buffer(ix[self.inter_buffer_idx])
        self.stage = "ready"

    #
    # Producer functions
    #

    def producer_copy(self, weight: SafeDataIndexPtr, copy_fn: Callable) -> ch.Expr:
        """
        [PRODUCER] Producer copies data to a shared buffer.

        Args:
            ix: Cheetah indices object
            weight: Source data pointer
            copy_fn: Function to perform the copy
            name: Buffer name for debugging

        Returns:
            Pipeline stage used
        """
        assert (
            self.stage == "ready"
        ), f"MemoryPipeline in state {self.stage}, should be ready"
        # Wait for free producer stage
        next_stage = self.pipeline_manager.pipeline_producer_wait()

        # Perform copy with scope
        with self.ix.scope():
            self.ix.set_index(self.inter_buffer_idx, next_stage)

            # Use provided copy function
            copy_fn(
                self.ix,
                weight,
                self.buffer,
                self.mbars[next_stage],
            )

            # Signal completion
            self.pipeline_manager.pipeline_producer_arrive(next_stage)

        return next_stage

    def pipeline_ready(self, ix: Indices):
        """Ready the pipeline"""
        assert (
            self.stage == "initialized"
        ), f"MemoryPipeline in state {self.stage}, should be initialized"

        with ch.if_(ix[self.warp_idx] < self.dims.size(self.consumer_warp_idx)):
            self.consumer_pipeline_ready(ix)
