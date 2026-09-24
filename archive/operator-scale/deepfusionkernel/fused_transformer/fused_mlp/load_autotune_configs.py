import inspect
import logging
import pickle
from typing import Any, Callable, Dict, Iterable
import triton

from fused_transformer.utils import is_cuda, is_hip


def apply_autotuner_cache_records(
    best_config_pkl_path, autotuners: Iterable[triton.runtime.Autotuner]
):
    with open(best_config_pkl_path, "rb") as f:
        records: Dict[str, dict] = pickle.load(f)

    logging.info(f"Loading caches from {best_config_pkl_path}")
    for fn_name in records:
        if fn_name == "args":
            continue
        cache = records[fn_name]["cache"]
        keys = records[fn_name]["keys"]
        if cache == {}:
            logging.warning(
                f"Empty cache for {fn_name}, possibly because when running autotuning there is only one config."
            )
            continue
        for autotuner in autotuners:
            assert isinstance(autotuner, triton.runtime.Autotuner)
            if autotuner.base_fn.__name__ != fn_name:
                continue
            if autotuner.keys != keys:
                logging.warning(
                    f"Not loading autotuner of {fn_name} due to unmatched keys"
                )
                continue
            autotuner.cache.update(cache)
            logging.info(f"Loaded autotuner cache for {fn_name}")
