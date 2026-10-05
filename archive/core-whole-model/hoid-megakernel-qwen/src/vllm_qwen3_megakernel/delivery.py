"""Per-output storage and completion leases for the pinned asynchronous runner.

Cancellation discards a result, not its device work or its raw fault checks.
Uncertain completion intentionally retains the lease in the workspace root.
"""


class DeliveryLease:
    def __init__(self, workspace, tokens, storage, rows=1):
        self.workspace = workspace
        self.tokens = tokens
        self.storage = storage
        self.rows = rows
        self.output = None
        self.finished = False
        workspace._deliveries[id(self)] = self

    def attach(self, output):
        if self.output is not None or self.finished:
            raise RuntimeError('delivery completion can only be attached once')
        if not hasattr(output, 'copy_event') or not hasattr(output, 'sampled_token_ids'):
            raise TypeError('pinned AsyncOutput completion required')
        self.output = output
        return self

    def get_output(self):
        if self.finished or self.output is None:
            raise RuntimeError('delivery is completed or lacks copy completion')
        try:
            self.output.copy_event.synchronize()
        except BaseException:
            self.workspace.poisoned = True
            raise
        try:
            rows = self.output.sampled_token_ids.tolist()
            if len(rows) != self.rows or any(
                    len(row) != 1 or type(row[0]) is not int or not 0 <= row[0] < 151936
                    for row in rows):
                raise RuntimeError('invalid raw asynchronous sampled token')
            if self.workspace.poisoned:
                raise RuntimeError('decoder generation is poisoned')
            return self.output.get_output()
        except BaseException:
            self.workspace.poisoned = True
            raise
        finally:
            # Completion is known even if validation or stock conversion failed.
            self.finished = True
            self.tokens = self.storage = None
            del self.workspace._deliveries[id(self)]

    def cancel(self):
        self.get_output()


def guard_sample(workspace, internal_tokens, stream):
    """Keep I64 stock history intact; guard a fresh I32 delivery allocation."""
    import torch

    workspace._check(stream)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('sample delivery must be outside forward capture')
    rows = internal_tokens.shape[0] if internal_tokens.ndim == 2 else 0
    if (internal_tokens.device != stream.device or internal_tokens.dtype != torch.int64
            or internal_tokens.ndim != 2 or internal_tokens.shape[1] != 1
            or not 1 <= rows <= getattr(workspace, 'batch', 1) or not internal_tokens.is_contiguous()):
        raise ValueError('stock sample must be contiguous I64 [live rows, 1] within the bucket')
    control = getattr(workspace, 'control', workspace)
    if control.delivery is not None:
        raise RuntimeError('fixed and asynchronous delivery cannot be mixed')
    tokens = internal_tokens.to(dtype=torch.int32)
    lease = DeliveryLease(workspace, tokens, tokens.untyped_storage(), rows)
    try:
        # cuLaunchKernel copies these argument values before returning. A new
        # delivery owns its storage until the copy-stream event has completed.
        control.guard.values[3].value = tokens.data_ptr()
        control.guard.values[4].value = rows
        control.guard.submit(stream)
    except BaseException:
        workspace.poisoned = True
        # Unknown submission outcome: retain storage, never speculate about free.
        raise
    return lease
