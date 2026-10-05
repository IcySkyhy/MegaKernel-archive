/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Qwen36Checkpoint

/-!
# Qwen3.6-27B text checkpoint layout

This module turns a validated sharded safetensors manifest into the exact 64-layer text-model
layout consumed by the CUDA megakernel. It keeps shard-relative byte offsets, so model weights can
stay in the published 15-shard organization rather than being repacked or copied through another
framework.
-/

namespace Qwen36.Architecture

open Qwen36.Checkpoint

def hiddenSize : Nat := 5120
def intermediateSize : Nat := 17408
def vocabularySize : Nat := 248320
def layerCount : Nat := 64
def linearLayerCount : Nat := 48
def fullAttentionLayerCount : Nat := 16
def linearQkvSize : Nat := 10240
def linearValueSize : Nat := 6144
def linearHeadCount : Nat := 48
def linearHeadDimension : Nat := 128
def fullQueryProjectionSize : Nat := 12288
def fullKeyValueProjectionSize : Nat := 1024
def fullOutputSize : Nat := 6144
def fullHeadDimension : Nat := 256

inductive LayerKind where
  | linearAttention
  | fullAttention
  deriving Repr, BEq, DecidableEq

/-- The published model uses three linear-attention layers followed by one full-attention layer. -/
def layerKind (layer : Nat) : LayerKind :=
  if (layer + 1) % 4 == 0 then .fullAttention else .linearAttention

/-- A BF16 tensor location relative to the beginning of one safetensors shard's data section. -/
structure TensorRef where
  name : String
  shard : UInt32
  byteOffset : UInt64
  byteCount : UInt64
  deriving Repr, BEq

structure CommonLayerWeights where
  inputNorm : TensorRef
  postAttentionNorm : TensorRef
  mlpGate : TensorRef
  mlpUp : TensorRef
  mlpDown : TensorRef
  deriving Repr, BEq

structure LinearAttentionWeights where
  aLog : TensorRef
  convolution : TensorRef
  dtBias : TensorRef
  projectionA : TensorRef
  projectionB : TensorRef
  projectionQkv : TensorRef
  projectionZ : TensorRef
  norm : TensorRef
  outputProjection : TensorRef
  deriving Repr, BEq

structure FullAttentionWeights where
  queryNorm : TensorRef
  keyNorm : TensorRef
  queryProjection : TensorRef
  keyProjection : TensorRef
  valueProjection : TensorRef
  outputProjection : TensorRef
  deriving Repr, BEq

inductive AttentionWeights where
  | linear (weights : LinearAttentionWeights)
  | full (weights : FullAttentionWeights)
  deriving Repr, BEq

structure LayerWeights where
  index : Nat
  common : CommonLayerWeights
  attention : AttentionWeights
  deriving Repr, BEq

structure TextLayout where
  embedding : TensorRef
  layers : Array LayerWeights
  finalNorm : TensorRef
  lmHead : TensorRef
  deriving Repr, BEq

private def shardIndex (manifest : Manifest) (filename : String) : IO Nat := do
  for index in [:manifest.shards.size] do
    if let some shard := manifest.shards[index]? then
      if shard.filename == filename then
        return index
  throw <| IO.userError s!"checkpoint shard is not present in manifest: {filename}"

private def requireTensor (manifest : Manifest) (name : String)
    (shape : Array Nat) : IO TensorRef := do
  let some filename := manifest.index.findShard? name
    | throw <| IO.userError s!"checkpoint tensor is missing: {name}"
  let some shard := manifest.findShard? filename
    | throw <| IO.userError s!"checkpoint shard is missing: {filename}"
  let some tensor := shard.header.findTensor? name
    | throw <| IO.userError s!"checkpoint tensor '{name}' is absent from '{filename}'"
  unless tensor.dtype == "BF16" do
    throw <| IO.userError s!"checkpoint tensor '{name}' has dtype {tensor.dtype}, expected BF16"
  unless tensor.shape == shape do
    throw <| IO.userError
      s!"checkpoint tensor '{name}' has shape {tensor.shape}, expected {shape}"
  let index ← shardIndex manifest filename
  return {
    name
    shard := index.toUInt32
    byteOffset := tensor.dataStart.toUInt64
    byteCount := (tensor.dataEnd - tensor.dataStart).toUInt64
  }

private def layerPrefix (layer : Nat) : String :=
  s!"model.language_model.layers.{layer}"

private def readCommon (manifest : Manifest) (layer : Nat) : IO CommonLayerWeights := do
  let baseName := layerPrefix layer
  let inputNorm ← requireTensor manifest s!"{baseName}.input_layernorm.weight" #[hiddenSize]
  let postAttentionNorm ← requireTensor manifest
    s!"{baseName}.post_attention_layernorm.weight" #[hiddenSize]
  let mlpGate ← requireTensor manifest s!"{baseName}.mlp.gate_proj.weight"
    #[intermediateSize, hiddenSize]
  let mlpUp ← requireTensor manifest s!"{baseName}.mlp.up_proj.weight"
    #[intermediateSize, hiddenSize]
  let mlpDown ← requireTensor manifest s!"{baseName}.mlp.down_proj.weight"
    #[hiddenSize, intermediateSize]
  return { inputNorm, postAttentionNorm, mlpGate, mlpUp, mlpDown }

private def readLinear (manifest : Manifest) (layer : Nat) : IO LinearAttentionWeights := do
  let baseName := s!"{layerPrefix layer}.linear_attn"
  let aLog ← requireTensor manifest s!"{baseName}.A_log" #[linearHeadCount]
  let convolution ← requireTensor manifest s!"{baseName}.conv1d.weight"
    #[linearQkvSize, 1, 4]
  let dtBias ← requireTensor manifest s!"{baseName}.dt_bias" #[linearHeadCount]
  let projectionA ← requireTensor manifest s!"{baseName}.in_proj_a.weight"
    #[linearHeadCount, hiddenSize]
  let projectionB ← requireTensor manifest s!"{baseName}.in_proj_b.weight"
    #[linearHeadCount, hiddenSize]
  let projectionQkv ← requireTensor manifest s!"{baseName}.in_proj_qkv.weight"
    #[linearQkvSize, hiddenSize]
  let projectionZ ← requireTensor manifest s!"{baseName}.in_proj_z.weight"
    #[linearValueSize, hiddenSize]
  let norm ← requireTensor manifest s!"{baseName}.norm.weight" #[linearHeadDimension]
  let outputProjection ← requireTensor manifest s!"{baseName}.out_proj.weight"
    #[hiddenSize, linearValueSize]
  return {
    aLog, convolution, dtBias, projectionA, projectionB, projectionQkv, projectionZ, norm,
    outputProjection
  }

private def readFull (manifest : Manifest) (layer : Nat) : IO FullAttentionWeights := do
  let baseName := s!"{layerPrefix layer}.self_attn"
  let queryNorm ← requireTensor manifest s!"{baseName}.q_norm.weight" #[fullHeadDimension]
  let keyNorm ← requireTensor manifest s!"{baseName}.k_norm.weight" #[fullHeadDimension]
  let queryProjection ← requireTensor manifest s!"{baseName}.q_proj.weight"
    #[fullQueryProjectionSize, hiddenSize]
  let keyProjection ← requireTensor manifest s!"{baseName}.k_proj.weight"
    #[fullKeyValueProjectionSize, hiddenSize]
  let valueProjection ← requireTensor manifest s!"{baseName}.v_proj.weight"
    #[fullKeyValueProjectionSize, hiddenSize]
  let outputProjection ← requireTensor manifest s!"{baseName}.o_proj.weight"
    #[hiddenSize, fullOutputSize]
  return {
    queryNorm, keyNorm, queryProjection, keyProjection, valueProjection, outputProjection
  }

/-- Validate and resolve every tensor used by one-token text inference. -/
def readTextLayout (manifest : Manifest) : IO TextLayout := do
  let embedding ← requireTensor manifest "model.language_model.embed_tokens.weight"
    #[vocabularySize, hiddenSize]
  let mut layers := #[]
  for layer in [:layerCount] do
    let common ← readCommon manifest layer
    let attention ← match layerKind layer with
      | .linearAttention => do
        let weights ← readLinear manifest layer
        pure (.linear weights)
      | .fullAttention => do
        let weights ← readFull manifest layer
        pure (.full weights)
    layers := layers.push { index := layer, common, attention }
  let finalNorm ← requireTensor manifest "model.language_model.norm.weight" #[hiddenSize]
  let lmHead ← requireTensor manifest "lm_head.weight" #[vocabularySize, hiddenSize]
  return { embedding, layers, finalNorm, lmHead }

def CommonLayerWeights.tensorRefs (weights : CommonLayerWeights) : Array TensorRef :=
  #[weights.inputNorm, weights.postAttentionNorm, weights.mlpGate, weights.mlpUp, weights.mlpDown]

def AttentionWeights.tensorRefs : AttentionWeights → Array TensorRef
  | .linear weights => #[
      weights.aLog, weights.convolution, weights.dtBias, weights.projectionA, weights.projectionB,
      weights.projectionQkv, weights.projectionZ, weights.norm, weights.outputProjection]
  | .full weights => #[
      weights.queryNorm, weights.keyNorm, weights.queryProjection, weights.keyProjection,
      weights.valueProjection, weights.outputProjection]

def LayerWeights.tensorRefs (weights : LayerWeights) : Array TensorRef :=
  weights.common.tensorRefs ++ weights.attention.tensorRefs

def TextLayout.tensorRefs (layout : TextLayout) : Array TensorRef := Id.run do
  let mut tensors := #[layout.embedding]
  for layer in layout.layers do
    tensors := tensors ++ layer.tensorRefs
  return (tensors.push layout.finalNorm).push layout.lmHead

def TextLayout.linearLayers (layout : TextLayout) : Nat :=
  layout.layers.foldl (init := 0) fun count layer =>
    match layer.attention with
    | .linear _ => count + 1
    | .full _ => count

def TextLayout.fullAttentionLayers (layout : TextLayout) : Nat :=
  layout.layers.size - layout.linearLayers

def TextLayout.totalBytes (layout : TextLayout) : UInt64 :=
  layout.tensorRefs.foldl (init := 0) fun total tensor => total + tensor.byteCount

end Qwen36.Architecture
