"""Split-phase scheduling: every step is prefill-only or decode-only, so each decode step is a
pure one-token-per-row batch the megakernel buckets cover. Subclasses non-public ``AsyncScheduler``.
"""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import create_request_queue

MAX_ROWS = 8
PREFILL = 'prefill'
DECODE = 'decode'


class MegakernelScheduler(AsyncScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'megakernel' not in (self.vllm_config.additional_config or {}):
            raise ValueError('MegakernelScheduler requires the megakernel additional_config')
        if self.max_num_running_reqs > MAX_ROWS:
            raise ValueError(f'max_num_seqs {self.max_num_running_reqs}: at most {MAX_ROWS} decode rows '
                             '(larger bucket unsupported; extra users queue at max_num_seqs)')
        self.last_step_kind = None
        # Free-block count at which a prefill found nothing it could admit; retried
        # only after blocks are released, never as alternating empty steps.
        self._megakernel_stalled_free = None

    def _megakernel_free_blocks(self) -> int:
        return self.kv_cache_manager.block_pool.get_num_free_blocks()

    def _megakernel_can_admit(self) -> bool:
        if self._pause_state != PauseState.UNPAUSED:
            return False
        queue = self._select_waiting_queue_for_scheduling()
        if queue is None:
            return False
        if len(self.running) + self.num_waiting_for_streaming_input >= self.max_num_running_reqs:
            return False
        free = self._megakernel_free_blocks()
        if self._megakernel_stalled_free is not None:
            if free <= self._megakernel_stalled_free:
                return False
            self._megakernel_stalled_free = None
        request = queue.peek_request()
        # The prompt plus the first sampled token's slot, whole blocks.
        needed = -(-(request.num_tokens + 1) // self.block_size)
        return free >= needed

    def schedule(self, throttle_prefills: bool = False):
        if self._megakernel_can_admit():
            output = self._megakernel_prefill_only(throttle_prefills)
            if output.num_scheduled_tokens:
                self.last_step_kind = PREFILL
                return output
            self._megakernel_stalled_free = self._megakernel_free_blocks()
            self.last_step_kind = None
            return output
        output = self._megakernel_decode_only(throttle_prefills)
        self.last_step_kind = DECODE if output.num_scheduled_tokens else None
        return output

    def _megakernel_prefill_only(self, throttle_prefills):
        hidden = self.running
        limit = self.max_num_running_reqs
        self.running = []
        self.max_num_running_reqs = limit - len(hidden)
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.running = hidden + self.running
            self.max_num_running_reqs = limit

    def _megakernel_decode_only(self, throttle_prefills):
        waiting, skipped = self.waiting, self.skipped_waiting
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)
        try:
            return super().schedule(throttle_prefills)
        finally:
            # Anything queued during the step (e.g. a preemption) goes ahead of
            # the requests that were already waiting.
            new_waiting, new_skipped = self.waiting, self.skipped_waiting
            self.waiting, self.skipped_waiting = waiting, skipped
            if new_waiting:
                self.waiting.prepend_requests(new_waiting)
            if new_skipped:
                self.skipped_waiting.prepend_requests(new_skipped)


def resolved_scheduler_is_active(vllm_config) -> bool:
    """True iff the engine resolved this split-phase scheduler with async output."""
    scheduler = vllm_config.scheduler_config
    configured = scheduler.scheduler_cls
    if configured is None:
        return False
    if isinstance(configured, str):
        from vllm.utils.import_utils import resolve_obj_by_qualname
        configured = resolve_obj_by_qualname(configured)
    return (isinstance(configured, type) and issubclass(configured, MegakernelScheduler)
            and scheduler.async_scheduling is True)
