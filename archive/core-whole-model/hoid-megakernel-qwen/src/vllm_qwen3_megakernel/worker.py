"""Pinned V2 worker, phase/capture ownership and guarded async sampler hooks."""
from functools import wraps

from .delivery import guard_sample


def install_delivery_hooks(runner, current_decode_workspace):
    """Use the stock I64 tensor internally and fresh guarded storage for AsyncOutput.

    current_decode_workspace returns None for an explicitly classified prefill or
    profile step, and the generation's Decoder for decode (including FULL replay).
    """
    import torch
    from vllm.v1.outputs import AsyncModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput

    if getattr(runner, '_megakernel_delivery_hooks', False):
        raise RuntimeError('delivery hooks already installed')
    ownership = object()
    runner._megakernel_delivery_hooks = ownership
    sample = runner.sample
    postprocess = runner.postprocess_sampled
    sample_tokens = runner.sample_tokens
    pending = None

    class CheckedOutput(AsyncModelRunnerOutput):
        def __init__(self, lease):
            self.lease = lease

        def get_output(self):
            return self.lease.get_output()

        def cancel(self):
            self.lease.cancel()

    @wraps(sample)
    def guarded_sample(hidden_states, input_batch, grammar_output):
        nonlocal pending
        if pending is not None:
            raise RuntimeError('previous sample lacks an AsyncOutput owner')
        workspace = current_decode_workspace()
        result = sample(hidden_states, input_batch, grammar_output)
        if workspace is not None:
            internal = result[0].sampled_token_ids
            lease = guard_sample(workspace, internal, torch.cuda.current_stream(runner.device))
            pending = (lease, internal)
            result[0].sampled_token_ids = lease.tokens
        return result

    @wraps(postprocess)
    def checked_postprocess(idx_mapping, sampled_tokens, num_sampled, num_rejected,
                            query_start_loc=None):
        if pending is not None:
            lease, internal = pending
            if sampled_tokens is not lease.tokens:
                lease.workspace.poisoned = True
                raise RuntimeError('unexpected sampled tensor in stock postprocess')
            sampled_tokens = internal
        return postprocess(idx_mapping, sampled_tokens, num_sampled, num_rejected,
                           query_start_loc)

    @wraps(sample_tokens)
    def checked_sample_tokens(*args, **kwargs):
        nonlocal pending
        try:
            output = sample_tokens(*args, **kwargs)
            if pending is None:
                return output
            lease, _ = pending
            if not isinstance(output, AsyncOutput) or output.sampler_output.sampled_token_ids is not lease.tokens:
                raise RuntimeError('pinned AsyncOutput must own the guarded delivery')
            lease.attach(output)
            pending = None
            return CheckedOutput(lease)
        except BaseException:
            if pending is not None:
                pending[0].workspace.poisoned = True
            raise

    runner.sample = guarded_sample
    runner.postprocess_sampled = checked_postprocess
    runner.sample_tokens = checked_sample_tokens
    return ownership, (
        ('sample', guarded_sample, sample),
        ('postprocess_sampled', checked_postprocess, postprocess),
        ('sample_tokens', checked_sample_tokens, sample_tokens),
    )


from vllm.v1.worker.gpu_worker import Worker


class StandaloneWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        import torch
        from .adapter import NativeAdapter
        if load_dummy_weights or not self.vllm_config.use_v2_model_runner:
            raise ValueError('actual checkpoint and pinned V2 runner required')
        super().load_model(load_dummy_weights=False)
        runner = self.model_runner
        if type(runner).__module__ != 'vllm.v1.worker.gpu.model_runner':
            raise ValueError('unexpected runner implementation')
        with torch.cuda.stream(runner.main_stream):
            adapter = NativeAdapter(runner.model, runner.model.megakernel_config,
                                    runner.device, runner.main_stream,
                                    max_num_seqs=self.vllm_config.scheduler_config.max_num_seqs)
        runner.model.megakernel_adapter = adapter
        install_runner_hooks(runner, adapter)

    def compile_or_warm_up_model(self):
        from .adapter import Phase
        adapter = self.model_runner.model.megakernel_adapter
        if getattr(self, '_megakernel_warmed', False):
            adapter.state.fail('startup warmup cannot repeat on serving cache')
        with adapter.state.scope(Phase.PROFILE):
            result = super().compile_or_warm_up_model()
        self._megakernel_warmed = True
        return result


def install_runner_hooks(runner, adapter):
    import torch
    import weakref
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from .adapter import Phase, guard_request_data

    state = adapter.state
    initialize = runner.initialize_kv_cache
    prepare = runner.prepare_inputs
    execute = runner.execute_model
    dummy = runner._dummy_run
    profile = runner.profile_cudagraph_memory
    shutdown = runner.shutdown

    @wraps(initialize)
    def initialize_kv_cache(kv_cache_config, is_profiling=False, kv_cache_allocation_context=None):
        state.check()
        if state.graphs_live:
            state.fail('cache replacement with live graphs')
        result = initialize(kv_cache_config, is_profiling, kv_cache_allocation_context)
        with torch.cuda.stream(runner.main_stream):
            adapter.bind_runner(runner, profiling=is_profiling)
        return result

    @wraps(prepare)
    def prepare_inputs(scheduler_output, batch_req_state, batch_desc):
        batch = prepare(scheduler_output, batch_req_state, batch_desc)
        state.prepared(has_prefill=batch_req_state.has_prefill,
                       num_reqs=len(batch_req_state.req_ids),
                       num_tokens=batch_req_state.num_tokens,
                       dummy=state.phase == Phase.PROFILE,
                       all_prefill=bool(batch_req_state.is_prefilling_np.all()),
                       graph_mode=batch_desc.cg_mode.name, graph_reqs=batch_desc.num_reqs)
        return batch

    @wraps(execute)
    def execute_model(scheduler_output, intermediate_tensors=None, dummy_run=False,
                      skip_attn_for_dummy_run=False, is_profile=False, context_len=0):
        state.check()
        startup = state.phase == Phase.PROFILE
        if not dummy_run and not startup:
            if scheduler_output.preempted_req_ids:
                state.fail('preemption/recompute unsupported')
            for request in scheduler_output.scheduled_new_reqs:
                guard_request_data(request, runner.max_model_len)
                if request.num_computed_tokens != 0:
                    state.fail('resumed/cache-prefilled request unsupported')
        with state.scope(Phase.PROFILE if dummy_run or startup else Phase.IDLE):
            result = execute(scheduler_output, intermediate_tensors, dummy_run,
                             skip_attn_for_dummy_run, is_profile, context_len)
            runner._megakernel_sample_bucket = state.bucket if state.phase == Phase.DECODE else None
        return result

    reload = getattr(runner, 'reload_weights', None)

    def reload_weights(*args, **kwargs):
        # The decoders borrow the loaded weight storages; a reload replaces them.
        adapter.token.advance()
        state.fail('weight reload is unsupported by the standalone decoder')

    @wraps(dummy)
    def dummy_run(*args, **kwargs):
        with state.scope(Phase.PROFILE):
            return dummy(*args, **kwargs)

    @wraps(profile)
    def profile_cudagraph_memory():
        with state.scope(Phase.PROFILE):
            try:
                return profile()
            finally:
                if runner.cudagraph_manager is not None:
                    state.fail('profiling did not detach its graph manager')
                with torch.cuda.stream(runner.main_stream):
                    adapter.generation.destroy_graphs()
                state.graphs_live = False
                adapter.bound = False

    capture_link = [CudaGraphManager.capture]
    owner_ref = weakref.ref(runner)

    @wraps(capture_link[0])
    def capture(manager, create_forward_fn, progress_bar_desc='Capturing CUDA graphs'):
        owner = owner_ref()
        if owner is None or manager is not owner.cudagraph_manager:
            return capture_link[0](manager, create_forward_fn, progress_bar_desc)
        with torch.cuda.stream(owner.main_stream):
            adapter.generation.own_graph_manager(manager)
        state.graphs_live = True

        def factory(desc, warmup):
            rows = desc.num_reqs
            if (desc.cg_mode.name != 'FULL' or rows not in adapter.buckets
                    or desc.num_tokens != rows or desc.uniform_token_count not in (None, 1)):
                state.fail(f'unexpected capture descriptor {desc}')
            capture_stream = torch.cuda.current_stream(owner.device)
            with torch.cuda.stream(owner.main_stream):
                try:
                    adapter.native.authorize_capture_stream(capture_stream)
                except RuntimeError as error:
                    state.fail(f'capture authorization failed: {error}')
            forward = create_forward_fn(desc, warmup)
            with torch.cuda.stream(owner.main_stream):
                # vLLM's dummy capture batch marks every row live (seq_len 1 over
                # null tables and PAD slots); the paged validators must see padding.
                owner.input_buffers.seq_lens[:rows].zero_()
            capture_stream.wait_stream(owner.main_stream)

            def call(mode):
                with state.scope(Phase.CAPTURE, bucket=rows):
                    if warmup:
                        owner.main_stream.wait_stream(capture_stream)
                        with torch.cuda.stream(owner.main_stream):
                            forward(mode)
                        capture_stream.wait_stream(owner.main_stream)
                    else:
                        forward(mode)
            return call
        return capture_link[0](manager, factory, progress_bar_desc)

    capture._megakernel_capture_link = capture_link

    runner_hooks = (
        ('initialize_kv_cache', initialize_kv_cache, initialize),
        ('prepare_inputs', prepare_inputs, prepare),
        ('execute_model', execute_model, execute),
        ('_dummy_run', dummy_run, dummy),
        ('profile_cudagraph_memory', profile_cudagraph_memory, profile),
    ) + ((('reload_weights', reload_weights, reload),) if reload is not None else ())

    def restore_capture_hook():
        current = CudaGraphManager.capture
        if current is capture:
            CudaGraphManager.capture = capture_link[0]
            return
        while (link := getattr(current, '_megakernel_capture_link', None)) is not None:
            if link[0] is capture:
                link[0] = capture_link[0]
                return
            current = link[0]

    def restore_runner_hooks(delivery):
        for name, installed, original in runner_hooks + delivery[1]:
            if getattr(runner, name) is installed:
                setattr(runner, name, original)
        if getattr(runner, '_megakernel_delivery_hooks', None) is delivery[0]:
            del runner._megakernel_delivery_hooks

    retirement_complete = False

    @wraps(shutdown)
    def checked_shutdown():
        nonlocal retirement_complete
        if not retirement_complete:
            with torch.cuda.stream(runner.main_stream):
                adapter.generation.retire()
            retirement_complete = True
        state.poisoned = True
        result = shutdown()
        restore_runner_hooks(delivery)
        restore_capture_hook()
        if runner.shutdown is checked_shutdown:
            runner.shutdown = shutdown
        adapter.release_owned_references()
        import gc
        gc.collect()
        torch.accelerator.empty_cache()
        return result

    runner.initialize_kv_cache = initialize_kv_cache
    runner.prepare_inputs = prepare_inputs
    runner.execute_model = execute_model
    runner._dummy_run = dummy_run
    runner.profile_cudagraph_memory = profile_cudagraph_memory
    runner.shutdown = checked_shutdown
    if reload is not None:
        runner.reload_weights = reload_weights
    CudaGraphManager.capture = capture
    def current_decode_workspace():
        bucket = getattr(runner, '_megakernel_sample_bucket', None)
        return None if bucket is None else adapter.workspace(bucket)

    delivery = install_delivery_hooks(runner, current_decode_workspace)
