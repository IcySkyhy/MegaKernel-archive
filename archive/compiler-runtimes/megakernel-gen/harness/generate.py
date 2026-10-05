"""
generate.py -- free-running greedy decode, end to end.

The gate is teacher-forced, which is the right way to measure numerical
agreement but never exercises the on-device sampler or the token feedback path.
This does: tokenise a prompt, run the megakernel with greedy sampling so that
each step's argmax becomes the next step's input entirely on the GPU, and
detokenise. No host round trip per token.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mkrun import Engine

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--model", required=True)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
ap.add_argument("--compare", action="store_true",
                help="also decode greedily with transformers and report where the "
                     "streams diverge (they eventually will: a near-tied argmax "
                     "sends the two down different paths, which is why the GATE is "
                     "teacher-forced and this is only a demonstration)")
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
if a.chat and getattr(tok, "chat_template", None):
    enc = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                  add_generation_prompt=True, tokenize=True)
    # newer transformers returns a BatchEncoding, older a plain list
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
else:
    ids = tok(a.prompt, add_special_tokens=True)["input_ids"]
ids = [int(x) for x in ids]

eng = Engine(a.out, a.model, device=a.device, max_len=len(ids) + a.n + 8)
buf = np.array(ids + [0] * (a.n + 1), dtype=np.int32)
eng.reset()
eng.set_tokens(buf)                    # also records the prompt length
eng.set_prompt_len(len(ids))           # ... which is shorter than the buffer here

t0 = time.perf_counter()
eng.run(len(ids) + a.n, greedy=1)      # prefill and decode; the kernel appends
                                       # only past the prompt, and never leaves
                                       # the GPU between tokens
dt = time.perf_counter() - t0

out = eng.tokens(len(ids) + a.n + 1)
gen = [int(x) for x in out[len(ids): len(ids) + a.n]]
print(f"prompt   : {tok.decode(ids)!r}")
print(f"generated: {tok.decode(gen)!r}")
print(f"ids      : {gen[:16]}{' ...' if len(gen) > 16 else ''}")
print(f"{len(ids)} prompt + {a.n} decode tokens in {dt*1e3:.0f} ms")

if a.compare:
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                             device_map=f"cuda:{a.device}").eval()
    with torch.no_grad():
        ref = m.generate(torch.tensor([ids], device=f"cuda:{a.device}"),
                         max_new_tokens=a.n, do_sample=False,
                         pad_token_id=tok.eos_token_id or 0)[0].tolist()[len(ids):]
    n = min(len(gen), len(ref))
    same = next((i for i, (x, y) in enumerate(zip(gen, ref)) if x != y), n)
    print(f"reference: {tok.decode(ref)!r}")
    # transformers stops at EOS; this engine is told to ignore it, so a shorter
    # reference is not a divergence
    note = " (reference stopped at EOS)" if same == n and len(ref) < len(gen) else ""
    print(f"identical for the first {same} of {n} comparable tokens{note}")

eng.free()
