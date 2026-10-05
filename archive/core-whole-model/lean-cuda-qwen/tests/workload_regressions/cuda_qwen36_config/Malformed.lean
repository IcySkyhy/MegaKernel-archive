/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanCudaQwen.Qwen36

/-!
# Rejected Qwen3.8 config: divisibility violation

`Malformed.json` sets `num_attention_heads = 4` with `num_key_value_heads = 3`, violating the
GQA divisibility invariant; `hfconfig_type_provider` must fail elaboration via `Config.check`.
-/

hfconfig_type_provider "Malformed.json" as Bad
