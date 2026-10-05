"""vLLM on the same workload as the megakernel: batch 1, a 1024-token prompt,
128 decode steps, prefill removed by N-vs-1 differencing.

Same shape as harness/run.py's decode measurement, so the two numbers compare:
CUDA graphs on, greedy, one sequence, the prompt already in the cache.
"""
import os
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MKGEN_MODEL_120B", ""))
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--mem", type=float, default=0.85)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=a.model, max_model_len=a.prompt_len + a.decode_len + 64,
              gpu_memory_utilization=a.mem, enforce_eager=False,
              tensor_parallel_size=1)
    prompt = TokensPrompt(prompt_token_ids=prompt_tokens(a.model, a.prompt_len))

    def run(n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate([prompt], sp, use_tqdm=False)
        return time.perf_counter() - t0

    run(4)
    t1 = run(1)
    best = min((run(a.decode_len) - t1) / (a.decode_len - 1) for _ in range(3))
    print(f"RESULT vllm {os.path.basename(a.model)} decode_ms_per_tok {best*1e3:.4f}"
          f"  tok/s {1/best:.1f}")


if __name__ == "__main__":
    main()
