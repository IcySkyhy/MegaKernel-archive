"""SGLang on the same workload as the megakernel: batch 1, 1024-token prompt,
128 decode steps, prefill subtracted by N-vs-1 differencing.

NB: this MUST be a file with a __main__ guard.  SGLang spawns its scheduler with
'spawn', which re-imports __main__; a script fed on stdin cannot be re-imported
and the scheduler dies with a bare "rank 0 died".
"""
import os
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MKGEN_MODEL_120B", ""))
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.80)
    a = ap.parse_args()

    import sglang as sgl
    llm = sgl.Engine(model_path=a.model, mem_fraction_static=a.mem, tp_size=1,
                     disable_radix_cache=True, log_level="warning")
    toks = prompt_tokens(a.model, a.prompt_len)

    def run(n):
        sp = {"temperature": 0.0, "max_new_tokens": n, "ignore_eos": True}
        t0 = time.perf_counter()
        llm.generate(input_ids=[toks], sampling_params=sp)
        return time.perf_counter() - t0

    run(4)
    t1 = run(1)
    best = min((run(a.decode_len) - t1) / (a.decode_len - 1) for _ in range(3))
    print(f"RESULT sglang {os.path.basename(a.model)} decode_ms_per_tok {best*1e3:.4f}  tok/s {1/best:.1f}")
    llm.shutdown()


if __name__ == "__main__":
    main()
