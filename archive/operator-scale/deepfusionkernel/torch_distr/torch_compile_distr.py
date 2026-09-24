"""
Evaluate the latency of LLaMA 3.1 70B with torch.compile. 
This script should be run with torchrun for distributed execution across 
multiple GPUs. The results are saved in a JSONL file for later analysis.
"""
import itertools
import json
import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3.1-70B"

# Initialize distributed
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

# Retrieve tensor parallel model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    tp_plan="auto",
)
print(model._tp_plan)
model = torch.compile(model)

# Prepare input tokens
tokenizer = AutoTokenizer.from_pretrained(model_id)


def latency_test_once(batch_size, seq_len):
    prompt = ["Can" for _ in range(batch_size)]
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Distributed run
    outputs = model(inputs)

    logits = outputs.logits
    past_kv = outputs.past_key_values
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([inputs, next_token], dim=1)

    latencies = []

    for i in range(seq_len - 1):
        start_time = time.time()
        outputs = model(next_token, past_key_values=past_kv, use_cache=True)
        end_time = time.time()

        latencies.append(end_time - start_time)

        logits = outputs.logits
        past_kv = outputs.past_key_values
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return np.median(latencies), sum(latencies)


batch_sizes = [1, 2, 4, 8, 16, 32, 64]
seq_lens = [512]

res = []

for bs, sl in itertools.product(batch_sizes, seq_lens):
    print(f"Running batch size {bs}, output len {sl}")
    median_decode, total_latency = latency_test_once(bs, sl)
    print(f"Median latency: {median_decode}")
    avg_decode = total_latency / (sl - 1)
    res.append(
        {
            "run_name": "Llama3.1-70B-torch-compile",
            "batch_size": bs,
            "input_len": 1,
            "output_len": sl,
            "prefill_latency": 0,
            "total_latency": total_latency,
            "avg_decode_throughput": bs * (sl - 1) / total_latency,
        }
    )

    with open("results-torch.jsonl", "a") as f:
        # for entry in res:
        entry = res[-1]
        json.dump(entry, f)
        f.write("\n")


