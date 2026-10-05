"""Shared bits of the serving-engine baselines.

Same workload, same GPU, prefill subtracted -- the only honest way to compare a
batch-1 megakernel with a general serving engine.  Run N tokens and 1 token and
divide the difference by N-1: that cancels prefill exactly and is immune to
prefix caching, which would otherwise make their prefill look free.  Their
engines are general -- paged KV, continuous batching, arbitrary sampling -- and
this one is not, so the comparison is on decode latency alone and that is how
it is reported.
"""
import json
import os


def prompt_tokens(model, n):
    """Random in-vocabulary ids.

    The benchmark is memory bound, so WHICH tokens only has to be legal -- but
    an out-of-range id crashes some engines, and a vocabulary is not always at
    the top level of the config.
    """
    import torch
    cfg = json.load(open(os.path.join(model, "config.json")))
    v = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size", 32000)
    g = torch.Generator(device="cpu").manual_seed(1234)
    return torch.randint(0, int(v) - 16, (n,), generator=g).tolist()
