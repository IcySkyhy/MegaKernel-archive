"""Independent FP32 reference for the correctness gate: Hugging Face transformers, FP32 weights,
activations and KV cache, TF32 off, greedy. For rows 0..7 at each context, and for both prompt sets
(``gate``: the natural-language gate prompts; ``bench``: the timed benchmark prompts), it records
the first ``GATED`` generated token ids plus the FP32 top-1/top-2 logit margin at each position,
so a mismatch at a near-tie can be told apart from a real divergence.

Row ``r``'s prompt is the same at every batch size, so one reference per (set, context, row) covers
every run. Prefill is chunked through the KV cache so a 32k prompt never builds a dense mask.
"""
import argparse
import json
from pathlib import Path
import sys

from workload import BATCHES, GATED, PROMPT_TOKENS, gate_prompt, prompt

CHUNK = 2048


def generate(model, ids, steps):
    import torch
    from transformers import DynamicCache
    cache = DynamicCache()
    device = model.device
    logits = None
    for start in range(0, len(ids), CHUNK):
        chunk = torch.tensor([ids[start:start + CHUNK]], device=device)
        positions = torch.arange(start, start + chunk.shape[1], device=device)[None]
        logits = model(input_ids=chunk, position_ids=positions, past_key_values=cache,
                       use_cache=True).logits[0, -1]
    tokens, margins = [], []
    position = len(ids)
    for _ in range(steps):
        top = torch.topk(logits.float(), 2)
        token = int(top.indices[0])
        tokens.append(token)
        margins.append(float(top.values[0] - top.values[1]))
        logits = model(input_ids=torch.tensor([[token]], device=device),
                       position_ids=torch.tensor([[position]], device=device),
                       past_key_values=cache, use_cache=True).logits[0, -1]
        position += 1
    return tokens, margins


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='reference JSON (never overwritten)')
    parser.add_argument('--contexts', default=','.join(PROMPT_TOKENS))
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f'{args.out} exists; references are never overwritten')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = AutoModelForCausalLM.from_pretrained(args.weights, torch_dtype=torch.float32,
                                                 attn_implementation='sdpa').cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    rows = {}
    with torch.inference_mode():
        for context in args.contexts.split(','):
            length = PROMPT_TOKENS[context]
            for row in range(max(BATCHES)):
                for name, ids in (('gate', gate_prompt(tokenizer, row, length)), ('bench', prompt(row, length))):
                    tokens, margins = generate(model, ids, GATED)
                    key = f'{name}/{context}/row{row}'
                    rows[key] = dict(tokens=tokens, top2_margin=margins, text=tokenizer.decode(tokens))
                    print(json.dumps({'case': key, 'text': rows[key]['text'], 'min_margin': min(margins)}),
                          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        schema='qwen3-4b-vllm-hoid.reference.v1', gated_tokens=GATED, compute_dtype='float32',
        kv_cache_dtype='float32', tf32=False, attention='sdpa', prefill_chunk=CHUNK,
        transformers=transformers.__version__, torch=torch.__version__, rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
