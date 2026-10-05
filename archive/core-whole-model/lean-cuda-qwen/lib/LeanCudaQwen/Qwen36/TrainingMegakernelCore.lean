/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Model
public import LeanCudaQwen.Qwen36.LoRA

public section

/-!
# Shared projection-training megakernel core

Device-only phases shared by the dedicated pretraining and GRPO programs. One cooperative launch
performs LoRA forward, the objective's logit VJP, deterministic adapter VJPs, and AdamW. The
objective-specific entry points intentionally live in separate modules so their generated kernels
contain no runtime objective branch.

This is also the exact incremental fusion boundary: it trains one complete frozen-base plus LoRA
projection. The modular full-model trainer remains the numerical oracle while the same phases are
lifted projection by projection into the model-wide schedule.
-/

namespace Cuda.Qwen36.TrainingMegakernel

open Cuda.Qwen36.Primitives

/-- POD launch descriptor shared by both objective-specific programs. -/
structure ProjectionStepF32 where
  shape : Cuda.Qwen36.LoRA.Shape
  optimizer : Cuda.Training.AdamW.Config
  deriving Cuda.POD

/-- POD-only GRPO choices embedded directly in the GRPO program's launch arguments. -/
structure GRPOConfigF32 where
  groupSize : UInt32
  clipEpsilon : Float32
  klBeta : Float32
  advantageEpsilon : Float32
  deriving Cuda.POD

/-- Native register representation of one row's stable-softmax statistics. -/
@[struct] structure RowSoftmaxF32 where
  maximum : Float32
  inverseSum : Float32

/-- Device pointers shared by projection forward, reverse, and update phases. -/
@[struct] structure ProjectionBuffersF32 where
  input : Cuda.DevicePtr Float32
  baseWeight : Cuda.DevicePtr Float32
  adapterA : Cuda.DevicePtr Float32
  adapterB : Cuda.DevicePtr Float32
  adapterIds : Cuda.DevicePtr UInt32
  rankActivation : Cuda.DevicePtr Float32
  logits : Cuda.DevicePtr Float32
  logitGradient : Cuda.DevicePtr Float32
  adapterAGradient : Cuda.DevicePtr Float32
  adapterBGradient : Cuda.DevicePtr Float32
  adapterAFirstMoment : Cuda.DevicePtr Float32
  adapterASecondMoment : Cuda.DevicePtr Float32
  adapterBFirstMoment : Cuda.DevicePtr Float32
  adapterBSecondMoment : Cuda.DevicePtr Float32
  updateMask : Cuda.DevicePtr UInt32

private def isFinite (value : Float32) : Bool :=
  (value.toBits &&& 0x7f800000) != 0x7f800000

/-- Check the POD descriptor before either cooperative launch. -/
def ProjectionStepF32.check (step : ProjectionStepF32) : Except String ProjectionStepF32 := do
  let shape ← step.shape.check
  let optimizer := step.optimizer
  unless optimizer.learningRate > 0 && isFinite optimizer.learningRate do
    throw "training-megakernel learning rate must be positive and finite"
  unless optimizer.beta1 >= 0 && optimizer.beta1 < 1 && isFinite optimizer.beta1 do
    throw "training-megakernel beta1 must be finite and lie in [0, 1)"
  unless optimizer.beta2 >= 0 && optimizer.beta2 < 1 && isFinite optimizer.beta2 do
    throw "training-megakernel beta2 must be finite and lie in [0, 1)"
  unless optimizer.inverseBiasCorrection1 > 0 && isFinite optimizer.inverseBiasCorrection1 do
    throw "training-megakernel first bias correction must be positive and finite"
  unless optimizer.inverseBiasCorrection2 > 0 && isFinite optimizer.inverseBiasCorrection2 do
    throw "training-megakernel second bias correction must be positive and finite"
  unless optimizer.epsilon > 0 && isFinite optimizer.epsilon do
    throw "training-megakernel epsilon must be positive and finite"
  unless optimizer.weightDecay >= 0 && isFinite optimizer.weightDecay do
    throw "training-megakernel weight decay must be nonnegative and finite"
  return { step with shape }

/-- Check group geometry and GRPO scalars against a checked projection batch. -/
def GRPOConfigF32.check (config : GRPOConfigF32) (shape : Cuda.Qwen36.LoRA.Shape) :
    Except String GRPOConfigF32 := do
  let sequences := shape.rows / shape.sequenceLength
  unless config.groupSize >= 2 && sequences % config.groupSize == 0 do
    throw "GRPO megakernel requires adjacent groups of at least two sequences dividing the batch"
  unless config.clipEpsilon > 0 && config.clipEpsilon < 1 && isFinite config.clipEpsilon do
    throw "GRPO megakernel clip epsilon must be finite and lie in (0, 1)"
  unless config.klBeta >= 0 && isFinite config.klBeta do
    throw "GRPO megakernel KL beta must be nonnegative and finite"
  unless config.advantageEpsilon > 0 && isFinite config.advantageEpsilon do
    throw "GRPO megakernel advantage epsilon must be positive and finite"
  return config

/-- Cooperative launch geometry used by both separate programs. -/
def cooperativeConfig (blocks : UInt32) : Except String Cuda.LaunchConfig := do
  if blocks == 0 then
    throw "training-megakernel cooperative block count must be positive"
  return {
    grid := { x := blocks }
    block := { x := 128 }
    blockArenaBytes := 0
    cooperative := true
  }

namespace Internal

@[expose, cuda_device, always_inline]
def zero : Float32 := Float32.ofBits 0

@[expose, cuda_device, always_inline]
def one : Float32 := Float32.ofBits 0x3f800000

@[expose, cuda_device, always_inline]
def negativeInfinity : Float32 := Float32.ofBits 0xff800000

@[always_inline]
partial def rowMaximumLoop (logits : Cuda.DevicePtr Float32) (row classes column : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < classes then
    let value ← Cuda.loadFloat32 logits (row * classes + column).toUSize
    rowMaximumLoop logits row classes (column + 1) (max accumulator value)
  else
    return accumulator

@[always_inline]
partial def rowExponentSumLoop (logits : Cuda.DevicePtr Float32)
    (row classes column : UInt32) (maximum accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < classes then
    let value ← Cuda.loadFloat32 logits (row * classes + column).toUSize
    rowExponentSumLoop logits row classes (column + 1) maximum
      (accumulator + Float32.exp (value - maximum))
  else
    return accumulator

@[expose, cuda_device, always_inline]
def calculateRowSoftmax (logits : Cuda.DevicePtr Float32) (row classes : UInt32) :
    Cuda.DeviceM RowSoftmaxF32 := do
  let maximum ← rowMaximumLoop logits row classes 0 negativeInfinity
  let denominator ← rowExponentSumLoop logits row classes 0 maximum zero
  return { maximum, inverseSum := one / denominator }

@[expose, cuda_device, always_inline]
def storeRowSoftmax (statistics : Cuda.DevicePtr Float32) (row : UInt32)
    (value : RowSoftmaxF32) : Cuda.DeviceM Unit := do
  Cuda.storeFloat32 statistics (row * 2).toUSize value.maximum
  Cuda.storeFloat32 statistics (row * 2 + 1).toUSize value.inverseSum

@[expose, cuda_device, always_inline]
def loadRowSoftmax (statistics : Cuda.DevicePtr Float32) (row : UInt32) :
    Cuda.DeviceM RowSoftmaxF32 := do
  return {
    maximum := ← Cuda.loadFloat32 statistics (row * 2).toUSize
    inverseSum := ← Cuda.loadFloat32 statistics (row * 2 + 1).toUSize
  }

@[always_inline]
partial def rankForwardLoop (buffers : ProjectionBuffersF32) (shape : Cuda.Qwen36.LoRA.Shape)
    (linear stride : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.rank then
    Cuda.Qwen36.LoRA.rankForwardElementF32 buffers.input buffers.adapterA buffers.adapterIds
      buffers.rankActivation shape linear
    rankForwardLoop buffers shape (linear + stride) stride

@[always_inline]
partial def projectionForwardLoop (buffers : ProjectionBuffersF32)
    (shape : Cuda.Qwen36.LoRA.Shape) (linear stride : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.outputFeatures then
    Cuda.Qwen36.LoRA.forwardElementF32 buffers.input buffers.baseWeight buffers.adapterB
      buffers.adapterIds buffers.rankActivation buffers.logits shape linear
    projectionForwardLoop buffers shape (linear + stride) stride

@[always_inline]
partial def pretrainRowLoop (logits rowStatistics rowLoss : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32) (rows classes row stride : UInt32) : Cuda.DeviceM Unit := do
  if row < rows then
    let stats ← calculateRowSoftmax logits row classes
    storeRowSoftmax rowStatistics row stats
    let label ← Cuda.loadUInt32 targets row.toUSize
    let target ← Cuda.loadFloat32 logits (row * classes + label).toUSize
    let scaledLoss := (stats.maximum - Float32.log stats.inverseSum - target) / rows.toFloat32
    Cuda.storeFloat32 rowLoss row.toUSize scaledLoss
    pretrainRowLoop logits rowStatistics rowLoss targets rows classes (row + stride) stride

@[always_inline]
partial def grpoRowLoop (logits rowStatistics policy : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (rows classes row stride : UInt32) :
    Cuda.DeviceM Unit := do
  if row < rows then
    let stats ← calculateRowSoftmax logits row classes
    storeRowSoftmax rowStatistics row stats
    let enabled ← Cuda.loadUInt32 masks row.toUSize
    let logProbability ← if enabled == 0 then pure zero else do
      let label ← Cuda.loadUInt32 targets row.toUSize
      let target ← Cuda.loadFloat32 logits (row * classes + label).toUSize
      pure (target - stats.maximum + Float32.log stats.inverseSum)
    Cuda.storeFloat32 policy row.toUSize logProbability
    grpoRowLoop logits rowStatistics policy targets masks rows classes (row + stride) stride

@[expose, cuda_device, always_inline]
def storeGradientElement (logits rowStatistics gradient : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32) (classes linear : UInt32) (coefficient : Float32) :
    Cuda.DeviceM Unit := do
  let row := linear / classes
  let column := linear % classes
  let stats ← loadRowSoftmax rowStatistics row
  let value ← Cuda.loadFloat32 logits linear.toUSize
  let probability := Float32.exp (value - stats.maximum) * stats.inverseSum
  let label ← Cuda.loadUInt32 targets row.toUSize
  let target := if column == label then one else zero
  Cuda.storeFloat32 gradient linear.toUSize ((probability - target) * coefficient)

@[always_inline]
partial def pretrainGradientLoop (logits rowStatistics gradient : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32) (rows classes linear stride : UInt32) : Cuda.DeviceM Unit := do
  if linear < rows * classes then
    storeGradientElement logits rowStatistics gradient targets classes linear
      (one / rows.toFloat32)
    pretrainGradientLoop logits rowStatistics gradient targets rows classes (linear + stride) stride

@[always_inline]
partial def grpoGradientLoop (logits rowStatistics coefficients gradient : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (rows classes linear stride : UInt32) :
    Cuda.DeviceM Unit := do
  if linear < rows * classes then
    let row := linear / classes
    let enabled ← Cuda.loadUInt32 masks row.toUSize
    if enabled == 0 then
      Cuda.storeFloat32 gradient linear.toUSize zero
    else
      let coefficient ← Cuda.loadFloat32 coefficients row.toUSize
      storeGradientElement logits rowStatistics gradient targets classes linear coefficient
    grpoGradientLoop logits rowStatistics coefficients gradient targets masks rows classes
      (linear + stride) stride

@[always_inline]
partial def sumRowsLoop (values : Cuda.DevicePtr Float32) (rows row : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < rows then
    sumRowsLoop values rows (row + 1) (accumulator + (← Cuda.loadFloat32 values row.toUSize))
  else
    return accumulator

@[always_inline]
partial def adapterGradientLoop (buffers : ProjectionBuffersF32)
    (shape : Cuda.Qwen36.LoRA.Shape) (linear stride : UInt32) : Cuda.DeviceM Unit := do
  let aCount := shape.adapterCount * shape.rank * shape.inputFeatures
  let bCount := shape.adapterCount * shape.outputFeatures * shape.rank
  if linear < max aCount bCount then
    if linear < aCount then
      Cuda.Qwen36.LoRA.backwardAElementF32 buffers.input buffers.logitGradient buffers.adapterB
        buffers.adapterIds buffers.adapterAGradient shape linear
    if linear < bCount then
      Cuda.Qwen36.LoRA.backwardBElementF32 buffers.rankActivation buffers.logitGradient
        buffers.adapterIds buffers.adapterBGradient shape linear
    adapterGradientLoop buffers shape (linear + stride) stride

@[always_inline]
partial def optimizerLoop (buffers : ProjectionBuffersF32) (step : ProjectionStepF32)
    (linear stride : UInt32) : Cuda.DeviceM Unit := do
  let shape := step.shape
  let count := shape.adapterCount *
    (shape.rank * shape.inputFeatures + shape.outputFeatures * shape.rank)
  if linear < count then
    Cuda.Qwen36.LoRA.adamWElementF32 step.optimizer {
      adapterA := buffers.adapterA
      adapterB := buffers.adapterB
      adapterAGradient := buffers.adapterAGradient
      adapterBGradient := buffers.adapterBGradient
      adapterAFirstMoment := buffers.adapterAFirstMoment
      adapterASecondMoment := buffers.adapterASecondMoment
      adapterBFirstMoment := buffers.adapterBFirstMoment
      adapterBSecondMoment := buffers.adapterBSecondMoment
      updateMask := buffers.updateMask
    } shape linear
    optimizerLoop buffers step (linear + stride) stride

/-! Cross-module device entry bodies. Recursive loops remain implementation details because Lean's
`partial` definitions are not valid direct `@[cuda_device]` roots. -/

@[expose, cuda_device, always_inline]
def rankForward (buffers : ProjectionBuffersF32) (shape : Cuda.Qwen36.LoRA.Shape)
    (linear stride : UInt32) : Cuda.DeviceM Unit :=
  rankForwardLoop buffers shape linear stride

@[expose, cuda_device, always_inline]
def projectionForward (buffers : ProjectionBuffersF32) (shape : Cuda.Qwen36.LoRA.Shape)
    (linear stride : UInt32) : Cuda.DeviceM Unit :=
  projectionForwardLoop buffers shape linear stride

@[expose, cuda_device, always_inline]
def pretrainRows (logits rowStatistics rowLoss : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32) (rows classes row stride : UInt32) : Cuda.DeviceM Unit :=
  pretrainRowLoop logits rowStatistics rowLoss targets rows classes row stride

@[expose, cuda_device, always_inline]
def grpoRows (logits rowStatistics policy : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (rows classes row stride : UInt32) :
    Cuda.DeviceM Unit :=
  grpoRowLoop logits rowStatistics policy targets masks rows classes row stride

@[expose, cuda_device, always_inline]
def pretrainGradient (logits rowStatistics gradient : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32) (rows classes linear stride : UInt32) : Cuda.DeviceM Unit :=
  pretrainGradientLoop logits rowStatistics gradient targets rows classes linear stride

@[expose, cuda_device, always_inline]
def grpoGradient (logits rowStatistics coefficients gradient : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (rows classes linear stride : UInt32) :
    Cuda.DeviceM Unit :=
  grpoGradientLoop logits rowStatistics coefficients gradient targets masks rows classes linear stride

@[expose, cuda_device, always_inline]
def sumRows (values : Cuda.DevicePtr Float32) (rows : UInt32) : Cuda.DeviceM Float32 :=
  sumRowsLoop values rows 0 zero

@[expose, cuda_device, always_inline]
def adapterGradients (buffers : ProjectionBuffersF32) (shape : Cuda.Qwen36.LoRA.Shape)
    (linear stride : UInt32) : Cuda.DeviceM Unit :=
  adapterGradientLoop buffers shape linear stride

@[expose, cuda_device, always_inline]
def optimize (buffers : ProjectionBuffersF32) (step : ProjectionStepF32)
    (linear stride : UInt32) : Cuda.DeviceM Unit :=
  optimizerLoop buffers step linear stride

end Internal

end Cuda.Qwen36.TrainingMegakernel
