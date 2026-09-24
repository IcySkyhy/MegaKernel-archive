import cheetah.api as ch
from cheetah.api import ty
from cheetah.index_tools import Dims, Indices
from ..cuda_utils import printf

from .local_barrier import LocalBarrierManager


class PipelineManager:
    def __init__(
        self,
        dims: Dims,
        pipeline_stages: int,
        threads_per_block: int,
        local_barrier_manager: LocalBarrierManager,
    ):
        self.pipeline_scope = dims.new_scope("pipeline")
        self.pipeline_idx = dims.new_dim("pipeline_idx", pipeline_stages)
        self.pipeline_stages = pipeline_stages
        self.threads_per_block = threads_per_block
        self.local_barrier_manager = local_barrier_manager

        # Initialize pipeline barriers

    def pipeline_init(self):
        self.next_producer_stage = ch.alloc(ty.i32, 0)
        self.next_consumer_stage = ch.alloc(ty.i32, 0)
        self.stage_barriers = ch.alloc_array(ty.i32, self.pipeline_stages * 2)
        # for pipeline stage i, 2i is the producer barrier, 2i+1 is the consumer barrier
        for i in range(self.pipeline_stages):
            self.stage_barriers[2 * i] = self.local_barrier_manager.get_local_barrier(
                self.threads_per_block
            ).id_expr
            self.stage_barriers[2 * i + 1] = (
                self.local_barrier_manager.get_local_barrier(
                    self.threads_per_block
                ).id_expr
            )

    def pipeline_producer_arrive(self, stage: ch.Expr):
        self.local_barrier_manager.arrive(
            self.stage_barriers[2 * stage], self.threads_per_block
        )

    def pipeline_producer_wait(self) -> ch.Expr:
        self.local_barrier_manager.sync(
            self.stage_barriers[2 * self.next_producer_stage.val + 1],
            self.threads_per_block,
        )
        free_producer_stage = self.next_producer_stage.val
        self.next_producer_stage.val = (
            self.next_producer_stage.val + 1
        ) % self.pipeline_stages
        return free_producer_stage

    def pipeline_consumer_arrive(self, stage: ch.Expr):
        self.local_barrier_manager.arrive(
            self.stage_barriers[2 * stage + 1], self.threads_per_block
        )

    def pipeline_consumer_wait(self) -> ch.Expr:
        self.local_barrier_manager.sync(
            self.stage_barriers[2 * self.next_consumer_stage.val],
            self.threads_per_block,
        )
        free_consumer_stage = self.next_consumer_stage.val
        self.next_consumer_stage.val = (
            self.next_consumer_stage.val + 1
        ) % self.pipeline_stages
        return free_consumer_stage
