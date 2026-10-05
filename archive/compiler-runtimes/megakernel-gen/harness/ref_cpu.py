"""Reference logits on CPU, for models too large to dequantise onto a GPU.

gpt-oss-120b is MXFP4 on disk; `transformers` dequantises it to bf16, which is
~230 GB and does not fit in one H100.  It does fit in host memory, and a handful
of teacher-forced steps is all the gate needs.
"""
import argparse, json, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ids", required=True, help="npy file of int32 token ids")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import torch
from transformers import AutoModelForCausalLM

ids = np.load(a.ids).astype(np.int64)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                         device_map="cpu", attn_implementation="eager")
m.eval()
print(f"loaded in {time.time()-t0:.0f}s", flush=True)
with torch.no_grad():
    out = m(input_ids=torch.tensor([ids], dtype=torch.long))
np.save(a.out, out.logits[0].float().numpy())
print("wrote", a.out, flush=True)
