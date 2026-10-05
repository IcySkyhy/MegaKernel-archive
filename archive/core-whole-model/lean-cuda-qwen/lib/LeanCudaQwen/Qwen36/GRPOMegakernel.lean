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
# Dedicated GRPO projection megakernel

A single cooperative launch executes current-policy LoRA forward, sampled-token scoring, the
clipped group-relative objective and direct reference KL, the completion-masked logit VJP, both
adapter VJPs, and masked AdamW. Old-policy and reference scores are immutable inputs to the step.
There is no pretraining branch in the generated CUDA program.
-/

namespace Cuda.Qwen36.GRPOMegakernel

open Cuda.Qwen36.TrainingMegakernel

/-- One-launch GRPO step for a complete trainable LoRA projection. -/
@[cuda_kernel]
def grpoProjectionStepF32Kernel
    (input baseWeight adapterA adapterB : Cuda.DevicePtr Float32)
    (adapterIds : Cuda.DevicePtr UInt32)
    (rankActivation logits rowStatistics logitGradient policy oldPolicy reference rewards
      advantages rowCoefficients statistics loss adapterAGradient adapterBGradient
      adapterAFirstMoment adapterASecondMoment adapterBFirstMoment adapterBSecondMoment :
      Cuda.DevicePtr Float32)
    (targets masks updateMask : Cuda.DevicePtr UInt32)
    (step : ProjectionStepF32) (objective : GRPOConfigF32) : Cuda.DeviceM Unit := do
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
  TrainingMegakernel.Internal.grpoRows logits rowStatistics policy targets masks shape.rows
    shape.outputFeatures linear stride
  Cuda.gridSync
  Cuda.Qwen36.Model.grpoObjectiveElementF32 policy oldPolicy reference rewards advantages
    rowCoefficients statistics loss masks {
      sequences := shape.rows / shape.sequenceLength
      sequenceLength := shape.sequenceLength
      groupSize := objective.groupSize
      clipEpsilon := objective.clipEpsilon
      klBeta := objective.klBeta
      advantageEpsilon := objective.advantageEpsilon
    } linear
  Cuda.gridSync
  TrainingMegakernel.Internal.grpoGradient logits rowStatistics rowCoefficients logitGradient
    targets masks shape.rows shape.outputFeatures linear stride
  Cuda.gridSync
  TrainingMegakernel.Internal.adapterGradients buffers shape linear stride
  Cuda.gridSync
  TrainingMegakernel.Internal.optimize buffers step linear stride

/-- Launch one cooperative GRPO projection step. -/
def launchProjectionStepF32 (stream : @& Cuda.Stream)
    (input baseWeight adapterA adapterB : @& Cuda.Buffer Float32)
    (adapterIds : @& Cuda.Buffer UInt32)
    (rankActivation logits rowStatistics logitGradient policy oldPolicy reference rewards
      advantages rowCoefficients statistics loss adapterAGradient adapterBGradient
      adapterAFirstMoment adapterASecondMoment adapterBFirstMoment adapterBSecondMoment :
      @& Cuda.Buffer Float32)
    (targets masks updateMask : @& Cuda.Buffer UInt32)
    (step : ProjectionStepF32) (objective : GRPOConfigF32)
    (blocks : UInt32 := 1) : IO Cuda.KernelHandle := do
  let step ← match step.check with
    | .ok step => pure step
    | .error message => throw <| IO.userError message
  let objective ← match objective.check step.shape with
    | .ok objective => pure objective
    | .error message => throw <| IO.userError message
  let launch ← match cooperativeConfig blocks with
    | .ok launch => pure launch
    | .error message => throw <| IO.userError message
  grpoProjectionStepF32Kernel.launchOn stream launch input baseWeight adapterA adapterB adapterIds
    rankActivation logits rowStatistics logitGradient policy oldPolicy reference rewards advantages
    rowCoefficients statistics loss adapterAGradient adapterBGradient adapterAFirstMoment
    adapterASecondMoment adapterBFirstMoment adapterBSecondMoment targets masks updateMask step
    objective

end Cuda.Qwen36.GRPOMegakernel
