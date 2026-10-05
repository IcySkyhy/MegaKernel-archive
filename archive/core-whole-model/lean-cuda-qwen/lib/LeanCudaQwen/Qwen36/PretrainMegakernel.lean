/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.TrainingMegakernelCore

public section

/-!
# Dedicated pretraining projection megakernel

A single cooperative launch executes frozen-base plus LoRA forward, stable mean cross-entropy,
the logit VJP, both adapter VJPs, and a masked AdamW update. There is no runtime objective tag or
GRPO branch in the generated CUDA program.
-/

namespace Cuda.Qwen36.PretrainMegakernel

open Cuda.Qwen36.TrainingMegakernel

/-- One-launch pretraining step for a complete trainable LoRA projection. -/
@[cuda_kernel]
def pretrainingProjectionStepF32Kernel
    (input baseWeight adapterA adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32)
    (rankActivation logits rowStatistics logitGradient rowLoss loss adapterAGradient
      adapterBGradient adapterAFirstMoment adapterASecondMoment adapterBFirstMoment
      adapterBSecondMoment : Cuda.DevicePtr Float32)
    (targets updateMask : Cuda.DevicePtr UInt32)
    (step : ProjectionStepF32) : Cuda.DeviceM Unit := do
  let linear ← Cuda.Qwen36.Primitives.elementIndex
  let stride := (← Cuda.gridDimX) * (← Cuda.blockDimX)
  let buffers : ProjectionBuffersF32 := {
    input, baseWeight, adapterA, adapterB, adapterIds, rankActivation, logits, logitGradient,
    adapterAGradient, adapterBGradient, adapterAFirstMoment, adapterASecondMoment,
    adapterBFirstMoment, adapterBSecondMoment, updateMask
  }
  let shape := step.shape
  TrainingMegakernel.Internal.rankForward buffers shape linear stride
  Cuda.gridSync
  TrainingMegakernel.Internal.projectionForward buffers shape linear stride
  Cuda.gridSync
  TrainingMegakernel.Internal.pretrainRows logits rowStatistics rowLoss targets
    shape.rows shape.outputFeatures linear stride
  Cuda.gridSync
  TrainingMegakernel.Internal.pretrainGradient logits rowStatistics logitGradient targets
    shape.rows shape.outputFeatures linear stride
  if linear == 0 then
    Cuda.storeFloat32 loss 0
      (← TrainingMegakernel.Internal.sumRows rowLoss shape.rows)
  Cuda.gridSync
  TrainingMegakernel.Internal.adapterGradients buffers shape linear stride
  Cuda.gridSync
  TrainingMegakernel.Internal.optimize buffers step linear stride

/-- Launch one cooperative pretraining projection step. -/
def launchProjectionStepF32 (stream : @& Cuda.Stream)
    (input baseWeight adapterA adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation logits rowStatistics logitGradient rowLoss loss adapterAGradient
      adapterBGradient adapterAFirstMoment adapterASecondMoment adapterBFirstMoment
      adapterBSecondMoment : @& Cuda.Buffer Float32)
    (targets updateMask : @& Cuda.Buffer UInt32) (step : ProjectionStepF32)
    (blocks : UInt32 := 1) : IO Cuda.KernelHandle := do
  let step ← match step.check with
    | .ok step => pure step
    | .error message => throw <| IO.userError message
  let launch ← match cooperativeConfig blocks with
    | .ok launch => pure launch
    | .error message => throw <| IO.userError message
  pretrainingProjectionStepF32Kernel.launchOn stream launch input baseWeight adapterA adapterB
    adapterIds rankActivation logits rowStatistics logitGradient rowLoss loss adapterAGradient
    adapterBGradient adapterAFirstMoment adapterASecondMoment adapterBFirstMoment
    adapterBSecondMoment targets updateMask step

end Cuda.Qwen36.PretrainMegakernel
