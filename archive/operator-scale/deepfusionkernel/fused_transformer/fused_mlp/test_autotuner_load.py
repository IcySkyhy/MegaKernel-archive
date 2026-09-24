import os
import sys
import triton

from ..utils import DATA_DIR
from .load_autotune_configs import apply_autotuner_cache_records
from . import KERNEL_LIST


def test_apply_autotuner_cache_records(best_config_pkl_path):
    best_config_pkl_path = os.path.join(DATA_DIR, best_config_pkl_path)
    autotuners = [k for k in KERNEL_LIST if isinstance(k, triton.runtime.Autotuner)]
    apply_autotuner_cache_records(best_config_pkl_path, autotuners)
    for autotuner in autotuners:
        print(f"{autotuner.base_fn.__name__}:")
        for key_values, config in autotuner.cache.items():
            print(
                f"    Keys: {[(name, k) for name, k in zip(autotuner.keys, key_values)]}"
            )
            print(f"    Config: {config.all_kwargs()}")


if __name__ == "__main__":
    path = sys.argv[1]  # e.g. "autotune-fused_gmlp-2025-04-06-20-30-58.pkl"
    test_apply_autotuner_cache_records(path)
