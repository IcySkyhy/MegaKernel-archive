/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanCudaQwen.Qwen36

/-!
# Rejected Qwen3.8 config: missing field

`MalformedMissing.json` has no `hidden_size` in `text_config`; `hfconfig_type_provider` must
fail elaboration with a missing-field error.
-/

hfconfig_type_provider "MalformedMissing.json" as Bad
