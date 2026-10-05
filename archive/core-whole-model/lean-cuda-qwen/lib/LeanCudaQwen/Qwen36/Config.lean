/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import Init.Data.Float
public import Init.Data.OfScientific
import Init.Data.ToString.Macro

@[expose] public section

/-!
# Qwen3.8 text-model configuration

`Config` mirrors the HuggingFace `text_config` of Qwen3.8-27B (text `model_type` `qwen3_5_text`):
a hybrid stack of Gated DeltaNet (`linear_attention`) layers and gated full-attention
(`full_attention`) layers. `Config.check` enforces the positivity, divisibility, and tiling
invariants the CUDA kernels rely on; the derived-dimension helpers give the projection widths
used throughout the model. `Config.tiny` is the gate configuration from
`docs/QWEN36_MEGAKERNEL.md`; `Config.qwen38_27B` is the deployment shape.
-/

namespace Lean.Cuda.Qwen36

/-- Decoder layer kind, from HF `layer_types`. -/
inductive LayerType where
  /-- Gated DeltaNet (HF `linear_attention`). -/
  | linearAttention
  /-- Gated full attention with partial RoPE (HF `full_attention`). -/
  | fullAttention
  deriving Repr, BEq, Inhabited, DecidableEq

/-- Qwen3.8 text configuration; field names track the HF `text_config` keys. -/
structure Config where
  /-- Model width (`hidden_size`). -/
  hiddenSize : Nat
  /-- Number of decoder layers (`num_hidden_layers`). -/
  numHiddenLayers : Nat
  /-- Per-layer mixer kind (`layer_types`); length must equal `numHiddenLayers`. -/
  layerTypes : Array LayerType
  /-- DeltaNet key heads (`linear_num_key_heads`). -/
  linearNumKeyHeads : Nat
  /-- DeltaNet key head dim (`linear_key_head_dim`); multiple of 16. -/
  linearKeyHeadDim : Nat
  /-- DeltaNet value heads (`linear_num_value_heads`); multiple of `linearNumKeyHeads`. -/
  linearNumValueHeads : Nat
  /-- DeltaNet value head dim (`linear_value_head_dim`); multiple of 16. -/
  linearValueHeadDim : Nat
  /-- DeltaNet depthwise conv kernel width (`linear_conv_kernel_dim`). -/
  linearConvKernelDim : Nat
  /-- Full-attention query heads (`num_attention_heads`); multiple of `numKeyValueHeads`. -/
  numAttentionHeads : Nat
  /-- Full-attention key/value heads (`num_key_value_heads`). -/
  numKeyValueHeads : Nat
  /-- Full-attention head dim (`head_dim`); multiple of 16. -/
  headDim : Nat
  /-- RoPE fraction of `headDim` (`partial_rotary_factor`); `headDim * factor` must be
      a positive even integer. -/
  partialRotaryFactor : Float
  /-- SwiGLU FFN width (`intermediate_size`). -/
  intermediateSize : Nat
  /-- Vocabulary size (`vocab_size`). -/
  vocabSize : Nat
  /-- RMSNorm epsilon (`rms_norm_eps`). -/
  rmsNormEps : Float
  /-- RoPE base (`rope_theta`; nested under `rope_parameters` in the HF config). -/
  ropeTheta : Float
  /-- Whether embedding and lm_head share weights (`tie_word_embeddings`). -/
  tieWordEmbeddings : Bool
  deriving Repr, Inhabited

namespace Config

/-- DeltaNet key width: `linear_num_key_heads * linear_key_head_dim` (2048 at 27B). -/
def keyDim (cfg : Config) : Nat := cfg.linearNumKeyHeads * cfg.linearKeyHeadDim

/-- DeltaNet value width: `linear_num_value_heads * linear_value_head_dim` (6144 at 27B). -/
def valueDim (cfg : Config) : Nat := cfg.linearNumValueHeads * cfg.linearValueHeadDim

/-- Depthwise-conv channel count: `qkv` packs key, key, value (10240 at 27B). -/
def convDim (cfg : Config) : Nat := 2 * cfg.keyDim + cfg.valueDim

/-- Query rows per full-attention layer; also the swish output-gate width (6144 at 27B). -/
def attnQDim (cfg : Config) : Nat := cfg.numAttentionHeads * cfg.headDim

/-- `W_q` output width including the output gate: `2 * num_attention_heads * head_dim`
    (12288 at 27B). -/
def qProjDim (cfg : Config) : Nat := 2 * cfg.attnQDim

/-- `W_k` and `W_v` output width, each: `num_key_value_heads * head_dim` (1024 at 27B). -/
def kvProjDim (cfg : Config) : Nat := cfg.numKeyValueHeads * cfg.headDim

/-- GQA repeat factor: `num_attention_heads / num_key_value_heads` (6 at 27B). -/
def gqaRatio (cfg : Config) : Nat := cfg.numAttentionHeads / cfg.numKeyValueHeads

/-- DeltaNet q/k head repeat factor: `linear_num_value_heads / linear_num_key_heads` (3 at 27B). -/
def linearRepeatFactor (cfg : Config) : Nat := cfg.linearNumValueHeads / cfg.linearNumKeyHeads

/-- Partial-RoPE width: `head_dim * partial_rotary_factor` (64 at 27B). `check` validates that
    the product is a positive even integer, so the Float round-trip is exact. -/
def rotaryDim (cfg : Config) : Nat :=
  (cfg.headDim.toUInt64.toFloat * cfg.partialRotaryFactor).toUInt64.toNat

/-- Whether `layer` is a gated full-attention layer; out-of-range indices are `false`. -/
def isFullAttention (cfg : Config) (layer : Nat) : Bool :=
  match cfg.layerTypes[layer]? with
  | some .fullAttention => true
  | _ => false

/-- Validate the invariants the kernels rely on; returns the config unchanged on success. -/
def check (cfg : Config) : Except String Config := do
  if cfg.hiddenSize == 0 then
    throw "hidden_size must be positive"
  if cfg.numHiddenLayers == 0 then
    throw "num_hidden_layers must be positive"
  if cfg.layerTypes.size != cfg.numHiddenLayers then
    throw s!"layer_types length ({cfg.layerTypes.size}) must equal num_hidden_layers ({cfg.numHiddenLayers})"
  if cfg.linearNumKeyHeads == 0 then
    throw "linear_num_key_heads must be positive"
  if cfg.linearNumValueHeads == 0 then
    throw "linear_num_value_heads must be positive"
  if cfg.linearNumValueHeads % cfg.linearNumKeyHeads != 0 then
    throw s!"linear_num_value_heads ({cfg.linearNumValueHeads}) must be a multiple of linear_num_key_heads ({cfg.linearNumKeyHeads})"
  if cfg.linearKeyHeadDim == 0 || cfg.linearKeyHeadDim % 16 != 0 then
    throw s!"linear_key_head_dim ({cfg.linearKeyHeadDim}) must be a positive multiple of 16"
  if cfg.linearValueHeadDim == 0 || cfg.linearValueHeadDim % 16 != 0 then
    throw s!"linear_value_head_dim ({cfg.linearValueHeadDim}) must be a positive multiple of 16"
  if cfg.linearConvKernelDim == 0 then
    throw "linear_conv_kernel_dim must be positive"
  if cfg.numKeyValueHeads == 0 then
    throw "num_key_value_heads must be positive"
  if cfg.numAttentionHeads == 0 then
    throw "num_attention_heads must be positive"
  if cfg.numAttentionHeads % cfg.numKeyValueHeads != 0 then
    throw s!"num_attention_heads ({cfg.numAttentionHeads}) must be a multiple of num_key_value_heads ({cfg.numKeyValueHeads})"
  if cfg.headDim == 0 || cfg.headDim % 16 != 0 then
    throw s!"head_dim ({cfg.headDim}) must be a positive multiple of 16"
  if !(cfg.partialRotaryFactor > 0.0 && cfg.partialRotaryFactor <= 1.0) then
    throw s!"partial_rotary_factor ({cfg.partialRotaryFactor}) must be in (0, 1]"
  let rotaryProduct := cfg.headDim.toUInt64.toFloat * cfg.partialRotaryFactor
  if rotaryProduct != cfg.rotaryDim.toUInt64.toFloat then
    throw s!"head_dim ({cfg.headDim}) * partial_rotary_factor ({cfg.partialRotaryFactor}) must be an integer"
  if cfg.rotaryDim == 0 || cfg.rotaryDim % 2 != 0 then
    throw s!"rotary dimension ({cfg.rotaryDim}) must be positive and even"
  if cfg.intermediateSize == 0 then
    throw "intermediate_size must be positive"
  if cfg.vocabSize == 0 then
    throw "vocab_size must be positive"
  if !(cfg.rmsNormEps > 0.0) then
    throw s!"rms_norm_eps ({cfg.rmsNormEps}) must be positive"
  if !(cfg.ropeTheta > 0.0) then
    throw s!"rope_theta ({cfg.ropeTheta}) must be positive"
  pure cfg

end Config

/-- Tiny gate configuration from `docs/QWEN36_MEGAKERNEL.md`: exercises every code path at a
    tile-friendly shape (three DeltaNet layers then one full-attention layer). -/
def Config.tiny : Config := {
  hiddenSize := 256
  numHiddenLayers := 4
  layerTypes := #[.linearAttention, .linearAttention, .linearAttention, .fullAttention]
  linearNumKeyHeads := 2
  linearKeyHeadDim := 64
  linearNumValueHeads := 4
  linearValueHeadDim := 64
  linearConvKernelDim := 4
  numAttentionHeads := 4
  numKeyValueHeads := 2
  headDim := 64
  partialRotaryFactor := 0.25
  intermediateSize := 512
  vocabSize := 512
  rmsNormEps := 1e-6
  ropeTheta := 1e7
  tieWordEmbeddings := false
}

/-- Qwen3.8-27B deployment shape: 64 layers, `full_attention` iff `i % 4 == 3`. -/
def Config.qwen38_27B : Config := {
  hiddenSize := 5120
  numHiddenLayers := 64
  layerTypes := #[
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention,
    .linearAttention, .linearAttention, .linearAttention, .fullAttention
  ]
  linearNumKeyHeads := 16
  linearKeyHeadDim := 128
  linearNumValueHeads := 48
  linearValueHeadDim := 128
  linearConvKernelDim := 4
  numAttentionHeads := 24
  numKeyValueHeads := 4
  headDim := 256
  partialRotaryFactor := 0.25
  intermediateSize := 17408
  vocabSize := 248320
  rmsNormEps := 1e-6
  ropeTheta := 1e7
  tieWordEmbeddings := false
}

end Lean.Cuda.Qwen36
