from typing import List
import torch
from transformers import AutoTokenizer
from datasets import load_dataset


def get_clm_example(
    data: List[str],
    tokenizer: AutoTokenizer,
    device,
):
    inputs = tokenizer(data, return_tensors="pt").input_ids.to(device)
    return inputs


def get_longbench_example(
    num_seq: int,
    seq_len: int,
    tokenizer: AutoTokenizer,
    device,
):
    inputs = []
    dataset = load_dataset("THUDM/LongBench", "gov_report", split="test")
    dataset_len = len(dataset)
    for i in range(num_seq):
        idx = i % dataset_len
        data = dataset[idx]["context"]
        inputs.append(data)
    input_ids = tokenizer(
        inputs,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        # return_tensor="pt",
    ).input_ids
    return torch.tensor(input_ids).to(device)
