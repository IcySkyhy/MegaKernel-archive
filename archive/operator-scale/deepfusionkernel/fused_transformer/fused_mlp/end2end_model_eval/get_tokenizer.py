import logging
from os import PathLike
from transformers import AutoTokenizer


def get_tokenizer(
    name: str = "meta-llama/Llama-3.2-1B", checkpoint: str | PathLike = None
):
    tokenizer = AutoTokenizer.from_pretrained(
        name if checkpoint is None else checkpoint
    )
    if "llama" in name:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Setting pad_token as eos_token")
    return tokenizer
