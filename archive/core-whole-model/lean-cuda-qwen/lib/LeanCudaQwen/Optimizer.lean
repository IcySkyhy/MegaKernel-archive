/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import Lean.Cuda.Half
public import Lean.Compiler.StructAttr

public section

/-!
# Device-native optimizer components

Optimizer algebra lives in Lean and is independent of whether a caller is an ordinary kernel, a
persistent worker, or a larger training megakernel. The CUDA boundary is limited to memory access
and reciprocal square root. Register state uses native Lean structures so composing an update does
not allocate Lean heap objects on the device.
-/

namespace Cuda.Training

namespace AdamW

/-- Per-step AdamW coefficients. Bias corrections are computed by the scheduler, not per element. -/
structure Config where
  /-- Learning rate for this step. -/
  learningRate : Float32
  /-- First-moment decay. -/
  beta1 : Float32
  /-- Second-moment decay. -/
  beta2 : Float32
  /-- Reciprocal of `1 - beta1 ^ step`. -/
  inverseBiasCorrection1 : Float32
  /-- Reciprocal of `1 - beta2 ^ step`. -/
  inverseBiasCorrection2 : Float32
  /-- Denominator stabilizer. -/
  epsilon : Float32
  /-- Decoupled weight decay. -/
  weightDecay : Float32
  deriving Cuda.POD

/-- First and second moments held in registers while one parameter is updated. -/
@[struct] structure Moments where
  /-- Exponential moving average of gradients. -/
  first : Float32
  /-- Exponential moving average of squared gradients. -/
  second : Float32

/-- All observable scalar results of one AdamW update. -/
@[struct] structure Result where
  /-- Updated FP32 master parameter. -/
  parameter : Float32
  /-- Updated first moment. -/
  firstMoment : Float32
  /-- Updated second moment. -/
  secondMoment : Float32
  /-- Bias-corrected Adam direction before decoupled decay is added. -/
  normalizedGradient : Float32

/-- Device buffers needed by a mixed-precision AdamW update. -/
@[struct] structure BFloat16Buffers where
  /-- FP32 master parameters. -/
  master : Cuda.DevicePtr Float32
  /-- BF16 parameters consumed by forward and backward kernels. -/
  model : Cuda.DevicePtr Cuda.BFloat16
  /-- BF16 gradients produced by backward kernels. -/
  gradient : Cuda.DevicePtr Cuda.BFloat16
  /-- FP32 first moments. -/
  firstMoment : Cuda.DevicePtr Float32
  /-- FP32 second moments. -/
  secondMoment : Cuda.DevicePtr Float32

/-- Device buffers for an AdamW update that preserves an FP32 gradient. -/
@[struct] structure Float32GradientBuffers where
  /-- FP32 master parameters. -/
  master : Cuda.DevicePtr Float32
  /-- BF16 parameters consumed by forward and backward kernels. -/
  model : Cuda.DevicePtr Cuda.BFloat16
  /-- FP32 gradients preserved from the training reduction. -/
  gradient : Cuda.DevicePtr Float32
  /-- FP32 first moments. -/
  firstMoment : Cuda.DevicePtr Float32
  /-- FP32 second moments. -/
  secondMoment : Cuda.DevicePtr Float32

/-- Update Adam's moments with two fused operations. -/
@[expose, cuda_device, always_inline] def updateMoments (config : Config)
    (gradient : Float32) (moments : Moments) : Moments :=
  let gradientSquared := gradient * gradient
  {
    first := Cuda.fma config.beta1 (moments.first - gradient) gradient
    second := Cuda.fma config.beta2 (moments.second - gradientSquared) gradientSquared
  }

/-- Finish an AdamW step from an already normalized gradient. -/
@[expose, cuda_device, always_inline] def finishNormalized (config : Config)
    (parameter : Float32) (moments : Moments) (normalized : Float32) : Result :=
  let update := Cuda.fma config.weightDecay parameter normalized
  {
    parameter := Cuda.fma (-config.learningRate) update parameter
    firstMoment := moments.first
    secondMoment := moments.second
    normalizedGradient := normalized
  }

/-- Finish an AdamW step from an inverse square root of the corrected second moment. -/
@[expose, cuda_device, always_inline] def finish (config : Config) (parameter : Float32)
    (moments : Moments) (inverseRoot : Float32) : Result :=
  let correctedFirst := moments.first * config.inverseBiasCorrection1
  let correctedSecond := moments.second * config.inverseBiasCorrection2
  let normalized := if correctedSecond == Float32.ofBits 0 then
      Float32.ofBits 0
    else
      correctedFirst * inverseRoot /
        (Float32.ofBits 0x3f800000 + config.epsilon * inverseRoot)
  finishNormalized config parameter moments normalized

/-- Device-only AdamW finish using CUDA's fast division instruction. -/
@[expose, cuda_device, always_inline] def finishFast (config : Config) (parameter : Float32)
    (moments : Moments) (inverseRoot : Float32) : Cuda.DeviceM Result := do
  let correctedFirst := moments.first * config.inverseBiasCorrection1
  let correctedSecond := moments.second * config.inverseBiasCorrection2
  let normalized ← if correctedSecond == Float32.ofBits 0 then
      pure (Float32.ofBits 0)
    else
      Cuda.fastDivide (correctedFirst * inverseRoot)
        (Float32.ofBits 0x3f800000 + config.epsilon * inverseRoot)
  return finishNormalized config parameter moments normalized

/-- Host-testable AdamW algebra using an exact square root. -/
@[expose, cuda_device, always_inline] def step (config : Config) (parameter gradient : Float32)
    (moments : Moments) : Result :=
  let moments := updateMoments config gradient moments
  let correctedSecond := moments.second * config.inverseBiasCorrection2
  let inverseRoot := Float32.ofBits 0x3f800000 / Float32.sqrt correctedSecond
  finish config parameter moments inverseRoot

/-- Device AdamW algebra using the CUDA reciprocal-square-root instruction. -/
@[expose, cuda_device, always_inline] def stepFast (config : Config)
    (parameter gradient : Float32) (moments : Moments) : Cuda.DeviceM Result := do
  let moments := updateMoments config gradient moments
  let correctedSecond := moments.second * config.inverseBiasCorrection2
  let inverseRoot ← Cuda.fastRsqrt correctedSecond
  finishFast config parameter moments inverseRoot

/-- Apply one scalar update and publish FP32 state plus the rounded BF16 model value. -/
@[expose, cuda_device, always_inline] def updateAndPublish (config : Config)
    (master : Cuda.DevicePtr Float32) (model : Cuda.DevicePtr Cuda.BFloat16)
    (firstMoment secondMoment : Cuda.DevicePtr Float32) (index : USize)
    (gradient : Float32) : Cuda.DeviceM Unit := do
  let parameter ← Cuda.loadFloat32 master index
  let first ← Cuda.loadFloat32 firstMoment index
  let second ← Cuda.loadFloat32 secondMoment index
  let result ← stepFast config parameter gradient { first, second }
  Cuda.storeFloat32 master index result.parameter
  Cuda.storeBFloat16 model index (Cuda.BFloat16.ofFloat32 result.parameter)
  Cuda.storeFloat32 firstMoment index result.firstMoment
  Cuda.storeFloat32 secondMoment index result.secondMoment

/-- Update one mixed-precision parameter from a BF16 gradient. -/
@[expose, cuda_device, always_inline] def updateBFloat16At (config : Config)
    (buffers : BFloat16Buffers) (index : USize) : Cuda.DeviceM Unit := do
  let gradient := (← Cuda.loadBFloat16 buffers.gradient index).toFloat32
  updateAndPublish config buffers.master buffers.model buffers.firstMoment buffers.secondMoment index
    gradient

/-- Update one mixed-precision parameter without rounding its FP32 gradient. -/
@[expose, cuda_device, always_inline] def updateFloat32GradientAt (config : Config)
    (buffers : Float32GradientBuffers) (index : USize) : Cuda.DeviceM Unit := do
  let gradient ← Cuda.loadFloat32 buffers.gradient index
  updateAndPublish config buffers.master buffers.model buffers.firstMoment buffers.secondMoment index
    gradient

end AdamW

namespace Momentum

/-- Momentum and optional Nesterov coefficients shared by SGD and Muon. -/
structure Config where
  /-- Exponential decay of the previous momentum. -/
  beta : Float32
  /-- Blend of the updated momentum into the direction; zero selects ordinary momentum. -/
  nesterov : Float32 := Float32.ofBits 0
  deriving Cuda.POD

/-- Update momentum and return the direction consumed by the outer optimizer. -/
@[struct] structure Result where
  /-- Updated momentum state. -/
  momentum : Float32
  /-- Ordinary or Nesterov-blended direction. -/
  direction : Float32

/-- One scalar momentum update, shared by elementwise optimizers and matrix-valued Muon phases. -/
@[expose, cuda_device, always_inline] def step (config : Config)
    (gradient previous : Float32) : Result :=
  let momentum := Cuda.fma config.beta (previous - gradient) gradient
  let direction := Cuda.fma config.nesterov (momentum - gradient) gradient
  { momentum, direction }

end Momentum

end Cuda.Training
