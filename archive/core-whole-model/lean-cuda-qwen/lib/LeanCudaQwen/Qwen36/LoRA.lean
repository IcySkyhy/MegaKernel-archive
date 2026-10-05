/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Linear
public import Lean.Cuda.Random
public import LeanCudaQwen.Optimizer

public section

/-!
# Qwen3.6 mixed-adapter LoRA

Checked batch descriptors and deterministic Float32 numerical bodies for
`Y = X W^T + (alpha / rank) (X A^T) B^T`. Adapter identifiers belong to sequences, while
activations and recurrent state remain indexed by batch row. The no-adapter sentinel selects the
frozen base projection only.

The element bodies are exposed for resident training schedules. Ordinary kernel wrappers provide
the independent sequential route used for forward, VJP, gradient-isolation, and AdamW gates.
-/

namespace Cuda.Qwen36.LoRA

open Cuda.Qwen36.Primitives

/-- A sequence with this identifier uses only the frozen base projection. -/
def noAdapter : UInt32 := 0xffffffff

/-- Host-owned mixed-adapter batch description. -/
structure Descriptor where
  batchSize : Nat
  sequenceLength : Nat
  adapterCount : Nat
  rank : Nat
  inputFeatures : Nat
  outputFeatures : Nat
  alpha : Float32
  adapterIds : Array UInt32
  deriving Repr

/-- POD shape consumed by sequential and resident device schedules. -/
structure Shape where
  rows : UInt32
  sequenceLength : UInt32
  adapterCount : UInt32
  rank : UInt32
  inputFeatures : UInt32
  outputFeatures : UInt32
  scale : Float32
  deriving Cuda.POD

namespace Descriptor

/-- Total token rows in the batch. -/
def rows (descriptor : Descriptor) : Nat :=
  descriptor.batchSize * descriptor.sequenceLength

private def fitsUInt32 (value : Nat) : Bool :=
  value ≤ 0xffffffff

/-- Validate all indexing and production tile invariants. -/
def check (descriptor : Descriptor) : Except String Descriptor := do
  if descriptor.batchSize == 0 then
    throw "LoRA batch size must be positive"
  if descriptor.sequenceLength == 0 then
    throw "LoRA sequence length must be positive"
  if descriptor.adapterCount == 0 then
    throw "LoRA adapter count must be positive"
  if descriptor.rank == 0 || descriptor.rank % 16 != 0 then
    throw s!"LoRA rank ({descriptor.rank}) must be a positive multiple of 16"
  if descriptor.inputFeatures == 0 || descriptor.outputFeatures == 0 then
    throw "LoRA projection dimensions must be positive"
  if !(descriptor.alpha > 0) || descriptor.alpha.toBits &&& 0x7f800000 == 0x7f800000 then
    throw "LoRA alpha must be positive and finite"
  if descriptor.adapterIds.size != descriptor.batchSize then
    throw s!"LoRA adapter-id count ({descriptor.adapterIds.size}) must equal batch size ({descriptor.batchSize})"
  for adapterId in descriptor.adapterIds do
    if adapterId != noAdapter && adapterId.toNat ≥ descriptor.adapterCount then
      throw s!"LoRA adapter id {adapterId} is outside adapter count {descriptor.adapterCount}"
  let indexedSizes := #[
    descriptor.rows,
    descriptor.adapterCount * descriptor.rank * descriptor.inputFeatures,
    descriptor.adapterCount * descriptor.outputFeatures * descriptor.rank,
    descriptor.rows * descriptor.rank,
    descriptor.rows * descriptor.inputFeatures,
    descriptor.rows * descriptor.outputFeatures
  ]
  for size in indexedSizes do
    unless fitsUInt32 size do
      throw s!"LoRA indexed element count {size} exceeds UInt32"
  return descriptor

/-- Convert a checked host descriptor to its device POD shape. -/
def toShape (descriptor : Descriptor) : Except String Shape := do
  let descriptor ← descriptor.check
  return {
    rows := descriptor.rows.toUInt32
    sequenceLength := descriptor.sequenceLength.toUInt32
    adapterCount := descriptor.adapterCount.toUInt32
    rank := descriptor.rank.toUInt32
    inputFeatures := descriptor.inputFeatures.toUInt32
    outputFeatures := descriptor.outputFeatures.toUInt32
    scale := descriptor.alpha / descriptor.rank.toUInt32.toFloat32
  }

end Descriptor

namespace Shape

/-- Validate a directly constructed device shape before launching a sequential kernel. -/
def check (shape : Shape) : Except String Shape := do
  if shape.rows == 0 || shape.sequenceLength == 0 || shape.rows % shape.sequenceLength != 0 then
    throw "LoRA rows must be a positive multiple of sequence length"
  if shape.adapterCount == 0 then
    throw "LoRA adapter count must be positive"
  if shape.rank == 0 || shape.rank % 16 != 0 then
    throw "LoRA rank must be a positive multiple of 16"
  if shape.inputFeatures == 0 || shape.outputFeatures == 0 then
    throw "LoRA projection dimensions must be positive"
  if !(shape.scale > 0) || shape.scale.toBits &&& 0x7f800000 == 0x7f800000 then
    throw "LoRA scale must be positive and finite"
  return shape

end Shape

/-- Device pointers used by a fused LoRA projection and its VJP. -/
@[struct] structure Buffers where
  input : Cuda.DevicePtr Float32
  baseWeight : Cuda.DevicePtr Float32
  adapterA : Cuda.DevicePtr Float32
  adapterB : Cuda.DevicePtr Float32
  adapterIds : Cuda.DevicePtr UInt32
  rankActivation : Cuda.DevicePtr Float32
  output : Cuda.DevicePtr Float32
  outputGradient : Cuda.DevicePtr Float32
  inputGradient : Cuda.DevicePtr Float32
  adapterAGradient : Cuda.DevicePtr Float32
  adapterBGradient : Cuda.DevicePtr Float32

/-- Device-resident parameter and optimizer state for adapter matrices. -/
@[struct] structure OptimizerBuffers where
  adapterA : Cuda.DevicePtr Float32
  adapterB : Cuda.DevicePtr Float32
  adapterAGradient : Cuda.DevicePtr Float32
  adapterBGradient : Cuda.DevicePtr Float32
  adapterAFirstMoment : Cuda.DevicePtr Float32
  adapterASecondMoment : Cuda.DevicePtr Float32
  adapterBFirstMoment : Cuda.DevicePtr Float32
  adapterBSecondMoment : Cuda.DevicePtr Float32
  updateMask : Cuda.DevicePtr UInt32

namespace Internal

@[always_inline]
def adapterForRow (adapterIds : Cuda.DevicePtr UInt32) (shape : Shape) (row : UInt32) :
    Cuda.DeviceM UInt32 :=
  Cuda.loadUInt32 adapterIds (row / shape.sequenceLength).toUSize

@[always_inline]
def enabled (shape : Shape) (adapter : UInt32) : Bool :=
  adapter < shape.adapterCount

@[always_inline]
partial def baseForwardDot (input weight : Cuda.DevicePtr Float32) (shape : Shape)
    (row outputFeature inputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if inputFeature < shape.inputFeatures then
    let x ← Cuda.loadFloat32 input (row * shape.inputFeatures + inputFeature).toUSize
    let w ← Cuda.loadFloat32 weight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize
    baseForwardDot input weight shape row outputFeature (inputFeature + 1)
      (Cuda.fma x w accumulator)
  else
    return accumulator

@[always_inline]
partial def adapterAForwardDot (input adapterA : Cuda.DevicePtr Float32) (shape : Shape)
    (row adapter rankFeature inputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if inputFeature < shape.inputFeatures then
    let x ← Cuda.loadFloat32 input (row * shape.inputFeatures + inputFeature).toUSize
    let offset := (adapter * shape.rank + rankFeature) * shape.inputFeatures + inputFeature
    let weight ← Cuda.loadFloat32 adapterA offset.toUSize
    adapterAForwardDot input adapterA shape row adapter rankFeature (inputFeature + 1)
      (Cuda.fma x weight accumulator)
  else
    return accumulator

@[always_inline]
partial def adapterBForwardDot (rankActivation adapterB : Cuda.DevicePtr Float32)
    (shape : Shape) (row adapter outputFeature rankFeature : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if rankFeature < shape.rank then
    let value ← Cuda.loadFloat32 rankActivation (row * shape.rank + rankFeature).toUSize
    let offset := (adapter * shape.outputFeatures + outputFeature) * shape.rank + rankFeature
    let weight ← Cuda.loadFloat32 adapterB offset.toUSize
    adapterBForwardDot rankActivation adapterB shape row adapter outputFeature (rankFeature + 1)
      (Cuda.fma value weight accumulator)
  else
    return accumulator

@[always_inline]
partial def baseInputGradientDot (outputGradient baseWeight : Cuda.DevicePtr Float32)
    (shape : Shape) (row inputFeature outputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if outputFeature < shape.outputFeatures then
    let gradient ← Cuda.loadFloat32 outputGradient
      (row * shape.outputFeatures + outputFeature).toUSize
    let weight ← Cuda.loadFloat32 baseWeight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize
    baseInputGradientDot outputGradient baseWeight shape row inputFeature (outputFeature + 1)
      (Cuda.fma gradient weight accumulator)
  else
    return accumulator

@[always_inline]
partial def rankGradientDot (outputGradient adapterB : Cuda.DevicePtr Float32)
    (shape : Shape) (row adapter rankFeature outputFeature : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if outputFeature < shape.outputFeatures then
    let gradient ← Cuda.loadFloat32 outputGradient
      (row * shape.outputFeatures + outputFeature).toUSize
    let offset := (adapter * shape.outputFeatures + outputFeature) * shape.rank + rankFeature
    let weight ← Cuda.loadFloat32 adapterB offset.toUSize
    rankGradientDot outputGradient adapterB shape row adapter rankFeature (outputFeature + 1)
      (Cuda.fma gradient weight accumulator)
  else
    return accumulator * shape.scale

@[always_inline]
partial def adapterInputGradientDot (outputGradient adapterA adapterB : Cuda.DevicePtr Float32)
    (shape : Shape) (row adapter inputFeature rankFeature : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if rankFeature < shape.rank then
    let rankGradient ← rankGradientDot outputGradient adapterB shape row adapter rankFeature 0 0
    let offset := (adapter * shape.rank + rankFeature) * shape.inputFeatures + inputFeature
    let weight ← Cuda.loadFloat32 adapterA offset.toUSize
    adapterInputGradientDot outputGradient adapterA adapterB shape row adapter inputFeature
      (rankFeature + 1) (Cuda.fma rankGradient weight accumulator)
  else
    return accumulator

@[always_inline]
partial def adapterAGradientRows (input outputGradient adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (shape : Shape)
    (adapter rankFeature inputFeature row : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if row < shape.rows then
    let rowAdapter ← adapterForRow adapterIds shape row
    let accumulator ← if rowAdapter == adapter then
        let rankGradient ← rankGradientDot outputGradient adapterB shape row adapter rankFeature 0 0
        let x ← Cuda.loadFloat32 input (row * shape.inputFeatures + inputFeature).toUSize
        pure (Cuda.fma rankGradient x accumulator)
      else
        pure accumulator
    adapterAGradientRows input outputGradient adapterB adapterIds shape adapter rankFeature
      inputFeature (row + 1) accumulator
  else
    return accumulator

@[always_inline]
partial def adapterBGradientRows (rankActivation outputGradient : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (shape : Shape)
    (adapter outputFeature rankFeature row : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if row < shape.rows then
    let rowAdapter ← adapterForRow adapterIds shape row
    let accumulator ← if rowAdapter == adapter then
        let gradient ← Cuda.loadFloat32 outputGradient
          (row * shape.outputFeatures + outputFeature).toUSize
        let activation ← Cuda.loadFloat32 rankActivation
          (row * shape.rank + rankFeature).toUSize
        pure (Cuda.fma (gradient * shape.scale) activation accumulator)
      else
        pure accumulator
    adapterBGradientRows rankActivation outputGradient adapterIds shape adapter outputFeature
      rankFeature (row + 1) accumulator
  else
    return accumulator

@[always_inline]
def updateParameter (config : Cuda.Training.AdamW.Config)
    (parameter gradient firstMoment secondMoment : Cuda.DevicePtr Float32) (index : USize) :
    Cuda.DeviceM Unit := do
  let value ← Cuda.loadFloat32 parameter index
  let grad ← Cuda.loadFloat32 gradient index
  let first ← Cuda.loadFloat32 firstMoment index
  let second ← Cuda.loadFloat32 secondMoment index
  let result ← Cuda.Training.AdamW.stepFast config value grad { first, second }
  Cuda.storeFloat32 parameter index result.parameter
  Cuda.storeFloat32 firstMoment index result.firstMoment
  Cuda.storeFloat32 secondMoment index result.secondMoment

end Internal

/-- Publish one `X A^T` element, or zero for a no-adapter row. -/
@[expose, cuda_device, always_inline]
def rankForwardElementF32 (input adapterA : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (rankActivation : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.rank then
    let row := linear / shape.rank
    let rankFeature := linear % shape.rank
    let adapter ← Internal.adapterForRow adapterIds shape row
    let value ← if Internal.enabled shape adapter then
        Internal.adapterAForwardDot input adapterA shape row adapter rankFeature 0 0
      else
        pure 0
    Cuda.storeFloat32 rankActivation linear.toUSize value

/-- Publish one frozen-base plus adapter output element. -/
@[expose, cuda_device, always_inline]
def forwardElementF32 (input baseWeight adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (rankActivation output : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.outputFeatures then
    let row := linear / shape.outputFeatures
    let outputFeature := linear % shape.outputFeatures
    let base ← Internal.baseForwardDot input baseWeight shape row outputFeature 0 0
    let adapter ← Internal.adapterForRow adapterIds shape row
    let value ← if Internal.enabled shape adapter then
        let delta ← Internal.adapterBForwardDot rankActivation adapterB shape row adapter
          outputFeature 0 0
        pure (Cuda.fma shape.scale delta base)
      else
        pure base
    Cuda.storeFloat32 output linear.toUSize value

/-- Add one adapter contribution to an already-computed frozen-base output element. -/
@[expose, cuda_device, always_inline]
def addForwardElementF32 (adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (rankActivation output : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.outputFeatures then
    let row := linear / shape.outputFeatures
    let outputFeature := linear % shape.outputFeatures
    let adapter ← Internal.adapterForRow adapterIds shape row
    if Internal.enabled shape adapter then
      let delta ← Internal.adapterBForwardDot rankActivation adapterB shape row adapter
        outputFeature 0 0
      let base ← Cuda.loadFloat32 output linear.toUSize
      Cuda.storeFloat32 output linear.toUSize (Cuda.fma shape.scale delta base)

/-- Publish one input VJP element, including both frozen-base and adapter paths. -/
@[expose, cuda_device, always_inline]
def backwardInputElementF32 (outputGradient baseWeight adapterA adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (inputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.inputFeatures then
    let row := linear / shape.inputFeatures
    let inputFeature := linear % shape.inputFeatures
    let base ← Internal.baseInputGradientDot outputGradient baseWeight shape row inputFeature 0 0
    let adapter ← Internal.adapterForRow adapterIds shape row
    let value ← if Internal.enabled shape adapter then
        let delta ← Internal.adapterInputGradientDot outputGradient adapterA adapterB shape row
          adapter inputFeature 0 0
        pure (base + delta)
      else
        pure base
    Cuda.storeFloat32 inputGradient linear.toUSize value

/-- Add only the adapter input VJP to a separately computed frozen-base input gradient. -/
@[expose, cuda_device, always_inline]
def addBackwardInputElementF32 (outputGradient adapterA adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (inputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.inputFeatures then
    let row := linear / shape.inputFeatures
    let inputFeature := linear % shape.inputFeatures
    let adapter ← Internal.adapterForRow adapterIds shape row
    if Internal.enabled shape adapter then
      let delta ← Internal.adapterInputGradientDot outputGradient adapterA adapterB shape row
        adapter inputFeature 0 0
      let base ← Cuda.loadFloat32 inputGradient linear.toUSize
      Cuda.storeFloat32 inputGradient linear.toUSize (base + delta)

/-- Deterministically reduce one adapter-A gradient element across matching token rows. -/
@[expose, cuda_device, always_inline]
def backwardAElementF32 (input outputGradient adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (adapterAGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  let count := shape.adapterCount * shape.rank * shape.inputFeatures
  if linear < count then
    let adapterStride := shape.rank * shape.inputFeatures
    let adapter := linear / adapterStride
    let remainder := linear % adapterStride
    let rankFeature := remainder / shape.inputFeatures
    let inputFeature := remainder % shape.inputFeatures
    let gradient ← Internal.adapterAGradientRows input outputGradient adapterB adapterIds shape
      adapter rankFeature inputFeature 0 0
    Cuda.storeFloat32 adapterAGradient linear.toUSize gradient

/-- Deterministically reduce one adapter-B gradient element across matching token rows. -/
@[expose, cuda_device, always_inline]
def backwardBElementF32 (rankActivation outputGradient : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (adapterBGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  let count := shape.adapterCount * shape.outputFeatures * shape.rank
  if linear < count then
    let adapterStride := shape.outputFeatures * shape.rank
    let adapter := linear / adapterStride
    let remainder := linear % adapterStride
    let outputFeature := remainder / shape.rank
    let rankFeature := remainder % shape.rank
    let gradient ← Internal.adapterBGradientRows rankActivation outputGradient adapterIds shape
      adapter outputFeature rankFeature 0 0
    Cuda.storeFloat32 adapterBGradient linear.toUSize gradient

/-- Apply one masked adapter-local AdamW update; a zero mask preserves parameters and moments. -/
@[expose, cuda_device, always_inline]
def adamWElementF32 (config : Cuda.Training.AdamW.Config) (buffers : OptimizerBuffers)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  let aCount := shape.adapterCount * shape.rank * shape.inputFeatures
  let bCount := shape.adapterCount * shape.outputFeatures * shape.rank
  if linear < aCount + bCount then
    let isA := linear < aCount
    let localIndex := if isA then linear else linear - aCount
    let stride := if isA then
        shape.rank * shape.inputFeatures
      else
        shape.outputFeatures * shape.rank
    let adapter := localIndex / stride
    let enabled ← Cuda.loadUInt32 buffers.updateMask adapter.toUSize
    if enabled != 0 then
      if isA then
        Internal.updateParameter config buffers.adapterA buffers.adapterAGradient
          buffers.adapterAFirstMoment buffers.adapterASecondMoment localIndex.toUSize
      else
        Internal.updateParameter config buffers.adapterB buffers.adapterBGradient
          buffers.adapterBFirstMoment buffers.adapterBSecondMoment localIndex.toUSize

/--
Initialize a complete adapter allocation in one launch. A receives deterministic centered Philox
values; B, gradients, optimizer moments, and rank scratch start at zero. Zero-initialized B makes
the initial adapter an exact frozen-base identity while preserving a nonzero gradient path into B.
-/
@[cuda_kernel]
def initializeAdapterF32Kernel
    (adapterA adapterB rankActivation adapterAGradient adapterBGradient adapterAFirstMoment
      adapterASecondMoment adapterBFirstMoment adapterBSecondMoment : Cuda.DevicePtr Float32)
    (seed : UInt64) (initScale : Float32) (shape : Shape) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let aCount := shape.adapterCount * shape.rank * shape.inputFeatures
  let bCount := shape.adapterCount * shape.outputFeatures * shape.rank
  let rankCount := shape.rows * shape.rank
  if linear < aCount then
    let word := Cuda.philox4x32Word seed linear 0 0 0 0
    let centered := (word &&& 0x00ffffff).toFloat32 / 8388608 - 1
    Cuda.storeFloat32 adapterA linear.toUSize (centered * initScale)
    Cuda.storeFloat32 adapterAGradient linear.toUSize 0
    Cuda.storeFloat32 adapterAFirstMoment linear.toUSize 0
    Cuda.storeFloat32 adapterASecondMoment linear.toUSize 0
  if linear < bCount then
    Cuda.storeFloat32 adapterB linear.toUSize 0
    Cuda.storeFloat32 adapterBGradient linear.toUSize 0
    Cuda.storeFloat32 adapterBFirstMoment linear.toUSize 0
    Cuda.storeFloat32 adapterBSecondMoment linear.toUSize 0
  if linear < rankCount then
    Cuda.storeFloat32 rankActivation linear.toUSize 0

@[cuda_kernel]
def rankForwardF32Kernel (input adapterA : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (rankActivation : Cuda.DevicePtr Float32)
    (shape : Shape) : Cuda.DeviceM Unit := do
  rankForwardElementF32 input adapterA adapterIds rankActivation shape (← elementIndex)

@[cuda_kernel]
def forwardF32Kernel (input baseWeight adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (rankActivation output : Cuda.DevicePtr Float32)
    (shape : Shape) : Cuda.DeviceM Unit := do
  forwardElementF32 input baseWeight adapterB adapterIds rankActivation output shape
    (← elementIndex)

@[cuda_kernel]
def addForwardF32Kernel (adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (rankActivation output : Cuda.DevicePtr Float32)
    (shape : Shape) : Cuda.DeviceM Unit := do
  addForwardElementF32 adapterB adapterIds rankActivation output shape (← elementIndex)

@[cuda_kernel]
def backwardInputF32Kernel (outputGradient baseWeight adapterA adapterB :
    Cuda.DevicePtr Float32) (adapterIds : Cuda.DevicePtr UInt32)
    (inputGradient : Cuda.DevicePtr Float32) (shape : Shape) : Cuda.DeviceM Unit := do
  backwardInputElementF32 outputGradient baseWeight adapterA adapterB adapterIds inputGradient shape
    (← elementIndex)

@[cuda_kernel]
def addBackwardInputF32Kernel (outputGradient adapterA adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (inputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) : Cuda.DeviceM Unit := do
  addBackwardInputElementF32 outputGradient adapterA adapterB adapterIds inputGradient shape
    (← elementIndex)

@[cuda_kernel]
def backwardAF32Kernel (input outputGradient adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (adapterAGradient : Cuda.DevicePtr Float32)
    (shape : Shape) : Cuda.DeviceM Unit := do
  backwardAElementF32 input outputGradient adapterB adapterIds adapterAGradient shape
    (← elementIndex)

@[cuda_kernel]
def backwardBF32Kernel (rankActivation outputGradient : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32) (adapterBGradient : Cuda.DevicePtr Float32)
    (shape : Shape) : Cuda.DeviceM Unit := do
  backwardBElementF32 rankActivation outputGradient adapterIds adapterBGradient shape
    (← elementIndex)

@[cuda_kernel]
def adamWF32Kernel (config : Cuda.Training.AdamW.Config)
    (adapterA adapterB adapterAGradient adapterBGradient adapterAFirstMoment adapterASecondMoment
      adapterBFirstMoment adapterBSecondMoment : Cuda.DevicePtr Float32)
    (updateMask : Cuda.DevicePtr UInt32) (shape : Shape) : Cuda.DeviceM Unit := do
  adamWElementF32 config {
    adapterA, adapterB, adapterAGradient, adapterBGradient,
    adapterAFirstMoment, adapterASecondMoment, adapterBFirstMoment, adapterBSecondMoment,
    updateMask
  } shape (← elementIndex)

private def requireShape (shape : Shape) : IO Unit :=
  match shape.check with
  | .ok _ => pure ()
  | .error error => throw <| IO.userError error

/-- Launch deterministic zero-delta initialization for a fully allocated adapter. -/
def initializeAdapterF32 (stream : @& Cuda.Stream)
    (adapterA adapterB rankActivation adapterAGradient adapterBGradient adapterAFirstMoment
      adapterASecondMoment adapterBFirstMoment adapterBSecondMoment : @& Cuda.Buffer Float32)
    (seed : UInt64) (initScale : Float32) (shape : Shape) : IO Unit := do
  requireShape shape
  unless initScale > 0 && initScale.toBits &&& 0x7f800000 != 0x7f800000 do
    throw <| IO.userError
      "Qwen3.6 LoRA adapter-A initialization scale must be positive and finite"
  let aCount := shape.adapterCount * shape.rank * shape.inputFeatures
  let bCount := shape.adapterCount * shape.outputFeatures * shape.rank
  let rankCount := shape.rows * shape.rank
  let elements := max aCount (max bCount rankCount)
  (← initializeAdapterF32Kernel.launchOn stream (elementConfig elements)
    adapterA adapterB rankActivation adapterAGradient adapterBGradient adapterAFirstMoment
    adapterASecondMoment adapterBFirstMoment adapterBSecondMoment seed initScale shape).waitChecked
    "Qwen3.6 LoRA adapter initialization"

/-! ## Shared sequential/device-graph submissions -/

/-- Submit frozen-base plus mixed-adapter forward to either a stream or CUDA graph. -/
def submitForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input baseWeight adapterA adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32) (rankActivation output : @& Cuda.Buffer Float32)
    (shape : Shape) : IO Unit := do
  requireShape shape
  let rankLaunch := elementConfig (shape.rows * shape.rank)
  executor.submit (label ++ " rank projection")
    (fun stream => rankForwardF32Kernel.launchOn stream rankLaunch input adapterA adapterIds
      rankActivation shape)
    (fun builder => rankForwardF32Kernel.addToGraph builder rankLaunch input adapterA adapterIds
      rankActivation shape)
  let outputLaunch := elementConfig (shape.rows * shape.outputFeatures)
  executor.submit label
    (fun stream => forwardF32Kernel.launchOn stream outputLaunch input baseWeight adapterB adapterIds
      rankActivation output shape)
    (fun builder => forwardF32Kernel.addToGraph builder outputLaunch input baseWeight adapterB
      adapterIds rankActivation output shape)

/-- Submit an adapter contribution on top of a separately computed frozen-base projection. -/
def submitAddForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input adapterA adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32) (rankActivation output : @& Cuda.Buffer Float32)
    (shape : Shape) : IO Unit := do
  requireShape shape
  let rankLaunch := elementConfig (shape.rows * shape.rank)
  executor.submit (label ++ " rank projection")
    (fun stream => rankForwardF32Kernel.launchOn stream rankLaunch input adapterA adapterIds
      rankActivation shape)
    (fun builder => rankForwardF32Kernel.addToGraph builder rankLaunch input adapterA adapterIds
      rankActivation shape)
  let outputLaunch := elementConfig (shape.rows * shape.outputFeatures)
  executor.submit label
    (fun stream => addForwardF32Kernel.launchOn stream outputLaunch adapterB adapterIds
      rankActivation output shape)
    (fun builder => addForwardF32Kernel.addToGraph builder outputLaunch adapterB adapterIds
      rankActivation output shape)

/-- Submit the mixed-adapter input/A/B VJPs. The frozen base intentionally has no gradient. -/
def submitBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input baseWeight adapterA adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation outputGradient inputGradient adapterAGradient adapterBGradient :
      @& Cuda.Buffer Float32) (shape : Shape) : IO Unit := do
  requireShape shape
  let inputLaunch := elementConfig (shape.rows * shape.inputFeatures)
  executor.submit (label ++ " input VJP")
    (fun stream => backwardInputF32Kernel.launchOn stream inputLaunch outputGradient baseWeight
      adapterA adapterB adapterIds inputGradient shape)
    (fun builder => backwardInputF32Kernel.addToGraph builder inputLaunch outputGradient baseWeight
      adapterA adapterB adapterIds inputGradient shape)
  let aLaunch := elementConfig (shape.adapterCount * shape.rank * shape.inputFeatures)
  executor.submit (label ++ " adapter-A VJP")
    (fun stream => backwardAF32Kernel.launchOn stream aLaunch input outputGradient adapterB
      adapterIds adapterAGradient shape)
    (fun builder => backwardAF32Kernel.addToGraph builder aLaunch input outputGradient adapterB
      adapterIds adapterAGradient shape)
  let bLaunch := elementConfig (shape.adapterCount * shape.outputFeatures * shape.rank)
  executor.submit (label ++ " adapter-B VJP")
    (fun stream => backwardBF32Kernel.launchOn stream bLaunch rankActivation outputGradient
      adapterIds adapterBGradient shape)
    (fun builder => backwardBF32Kernel.addToGraph builder bLaunch rankActivation outputGradient
      adapterIds adapterBGradient shape)

/-- Submit only trainable adapter VJPs when the frozen base input gradient is not required. -/
def submitBackwardAdaptersF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input adapterB : @& Cuda.Buffer Float32) (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation outputGradient adapterAGradient adapterBGradient : @& Cuda.Buffer Float32)
    (shape : Shape) : IO Unit := do
  requireShape shape
  let aLaunch := elementConfig (shape.adapterCount * shape.rank * shape.inputFeatures)
  executor.submit (label ++ " adapter-A VJP")
    (fun stream => backwardAF32Kernel.launchOn stream aLaunch input outputGradient adapterB
      adapterIds adapterAGradient shape)
    (fun builder => backwardAF32Kernel.addToGraph builder aLaunch input outputGradient adapterB
      adapterIds adapterAGradient shape)
  let bLaunch := elementConfig (shape.adapterCount * shape.outputFeatures * shape.rank)
  executor.submit (label ++ " adapter-B VJP")
    (fun stream => backwardBF32Kernel.launchOn stream bLaunch rankActivation outputGradient
      adapterIds adapterBGradient shape)
    (fun builder => backwardBF32Kernel.addToGraph builder bLaunch rankActivation outputGradient
      adapterIds adapterBGradient shape)

/--
Add the adapter input VJP and publish dA/dB after a separate frozen-base implementation has
already written the base input gradient.
-/
def submitAddBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input adapterA adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation outputGradient inputGradient adapterAGradient adapterBGradient :
      @& Cuda.Buffer Float32) (shape : Shape) : IO Unit := do
  requireShape shape
  let inputLaunch := elementConfig (shape.rows * shape.inputFeatures)
  executor.submit (label ++ " adapter input VJP")
    (fun stream => addBackwardInputF32Kernel.launchOn stream inputLaunch outputGradient adapterA
      adapterB adapterIds inputGradient shape)
    (fun builder => addBackwardInputF32Kernel.addToGraph builder inputLaunch outputGradient adapterA
      adapterB adapterIds inputGradient shape)
  submitBackwardAdaptersF32 executor label input adapterB adapterIds rankActivation outputGradient
    adapterAGradient adapterBGradient shape

/-- Sequential two-launch fused-base LoRA forward. -/
def forwardF32 (stream : @& Cuda.Stream) (input baseWeight adapterA adapterB :
    @& Cuda.Buffer Float32) (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation output : @& Cuda.Buffer Float32) (shape : Shape) : IO Unit := do
  requireShape shape
  let rankHandle ← rankForwardF32Kernel.launchOn stream (elementConfig (shape.rows * shape.rank))
    input adapterA adapterIds rankActivation shape
  rankHandle.waitChecked "Qwen3.6 LoRA rank projection"
  let outputHandle ← forwardF32Kernel.launchOn stream
    (elementConfig (shape.rows * shape.outputFeatures)) input baseWeight adapterB adapterIds
    rankActivation output shape
  outputHandle.waitChecked "Qwen3.6 fused LoRA projection"

/-- Sequential adapter contribution on top of a separately computed frozen-base projection. -/
def addForwardF32 (stream : @& Cuda.Stream) (input adapterA adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32) (rankActivation output : @& Cuda.Buffer Float32)
    (shape : Shape) : IO Unit :=
  submitAddForwardF32 (.sequential stream) "Qwen3.6 additive LoRA projection" input adapterA
    adapterB adapterIds rankActivation output shape

/-- Sequential adapter A/B VJPs without computing a frozen-base input gradient. -/
def backwardAdaptersF32 (stream : @& Cuda.Stream) (input adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation outputGradient adapterAGradient adapterBGradient : @& Cuda.Buffer Float32)
    (shape : Shape) : IO Unit :=
  submitBackwardAdaptersF32 (.sequential stream) "Qwen3.6 adapter-only LoRA VJP" input adapterB
    adapterIds rankActivation outputGradient adapterAGradient adapterBGradient shape

/-- Sequential deterministic LoRA input and adapter VJPs. The frozen base has no gradient output. -/
def backwardF32 (stream : @& Cuda.Stream) (input baseWeight adapterA adapterB :
    @& Cuda.Buffer Float32) (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation outputGradient inputGradient adapterAGradient adapterBGradient :
      @& Cuda.Buffer Float32) (shape : Shape) : IO Unit := do
  requireShape shape
  let inputHandle ← backwardInputF32Kernel.launchOn stream
    (elementConfig (shape.rows * shape.inputFeatures)) outputGradient baseWeight adapterA adapterB
    adapterIds inputGradient shape
  inputHandle.waitChecked "Qwen3.6 LoRA input VJP"
  let aHandle ← backwardAF32Kernel.launchOn stream
    (elementConfig (shape.adapterCount * shape.rank * shape.inputFeatures)) input outputGradient
    adapterB adapterIds adapterAGradient shape
  aHandle.waitChecked "Qwen3.6 LoRA A VJP"
  let bHandle ← backwardBF32Kernel.launchOn stream
    (elementConfig (shape.adapterCount * shape.outputFeatures * shape.rank)) rankActivation
    outputGradient adapterIds adapterBGradient shape
  bHandle.waitChecked "Qwen3.6 LoRA B VJP"

/-- Apply one device-resident, adapter-masked AdamW update to A and B. -/
def adamWF32 (stream : @& Cuda.Stream) (config : Cuda.Training.AdamW.Config)
    (adapterA adapterB adapterAGradient adapterBGradient adapterAFirstMoment adapterASecondMoment
      adapterBFirstMoment adapterBSecondMoment : @& Cuda.Buffer Float32)
    (updateMask : @& Cuda.Buffer UInt32) (shape : Shape) : IO Cuda.KernelHandle := do
  requireShape shape
  adamWF32Kernel.launchOn stream
    (elementConfig (shape.adapterCount *
      (shape.rank * shape.inputFeatures + shape.outputFeatures * shape.rank)))
    config adapterA adapterB adapterAGradient adapterBGradient adapterAFirstMoment
    adapterASecondMoment adapterBFirstMoment adapterBSecondMoment updateMask shape

end Cuda.Qwen36.LoRA
