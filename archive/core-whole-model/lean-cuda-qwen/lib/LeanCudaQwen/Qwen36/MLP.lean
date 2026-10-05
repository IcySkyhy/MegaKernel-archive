/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Projection

public section

/-!
# Qwen3.6 SwiGLU MLP

Numerical Float32 forward and reverse-mode composition for
`RMSNorm -> gate/up -> SiLU(gate) * up -> down`. A single checked operation schedule targets
either ordinary stream submission or CUDA graph construction, so the independent sequential
oracle and the replayable training route cannot drift apart.
-/

namespace Cuda.Qwen36.MLP

open Cuda.Qwen36.Primitives

/-- Common forward schedule for sequential validation and CUDA graph construction. -/
def submitForwardF32 (executor : @& Cuda.Qwen36.Executor)
    (input normWeight normalized inverseRms gate up activation output : @& Cuda.Buffer Float32)
    (gateWeight upWeight downWeight : @& Projection.WeightsF32)
    (rows hidden intermediate : UInt32) (epsilon : Float32) : IO Unit := do
  submitRmsNormForwardF32 executor "Qwen3.6 MLP RMSNorm" input normWeight normalized inverseRms
    rows hidden epsilon
  Projection.submitForwardF32 executor "Qwen3.6 MLP gate projection" normalized gateWeight gate rows
    hidden intermediate
  Projection.submitForwardF32 executor "Qwen3.6 MLP up projection" normalized upWeight up rows hidden
    intermediate
  submitSwigluForwardF32 executor "Qwen3.6 MLP SwiGLU" gate up activation
    (rows * intermediate)
  Projection.submitForwardF32 executor "Qwen3.6 MLP down projection" activation downWeight output rows
    intermediate hidden

/-- Sequential numerical forward baseline for a `[rows, hidden]` Qwen3.6 MLP. -/
def forwardF32 (stream : @& Cuda.Stream)
    (input normWeight normalized inverseRms gate up activation output : @& Cuda.Buffer Float32)
    (gateWeight upWeight downWeight : @& Projection.WeightsF32)
    (rows hidden intermediate : UInt32) (epsilon : Float32) : IO Unit :=
  submitForwardF32 (.sequential stream) input normWeight normalized inverseRms gate up activation
    output gateWeight upWeight downWeight rows hidden intermediate epsilon

/-- Common reverse-mode schedule for sequential validation and CUDA graph construction. -/
def submitBackwardF32 (executor : @& Cuda.Qwen36.Executor)
    (input normWeight normalized inverseRms gate up activation outputGradient activationGradient
      gateGradient upGradient normGradientGate normGradientUp normGradient inputGradient
      normWeightGradient gateWeightGradient upWeightGradient downWeightGradient :
      @& Cuda.Buffer Float32)
    (gateWeight upWeight downWeight : @& Projection.WeightsF32)
    (rows hidden intermediate : UInt32) : IO Unit := do
  Projection.submitBackwardF32 executor "Qwen3.6 MLP down projection" activation downWeight
    outputGradient activationGradient downWeightGradient rows intermediate hidden
  submitSwigluBackwardF32 executor "Qwen3.6 MLP SwiGLU VJP" gate up activationGradient
    gateGradient upGradient (rows * intermediate)
  Projection.submitBackwardF32 executor "Qwen3.6 MLP gate projection" normalized gateWeight
    gateGradient normGradientGate gateWeightGradient rows hidden intermediate
  Projection.submitBackwardF32 executor "Qwen3.6 MLP up projection" normalized upWeight upGradient
    normGradientUp upWeightGradient rows hidden intermediate
  Linear.submitAddF32 executor "Qwen3.6 MLP projection-gradient sum" normGradientGate
    normGradientUp normGradient (rows * hidden)
  submitRmsNormBackwardInputF32 executor "Qwen3.6 MLP RMSNorm input VJP" input normWeight
    normGradient inputGradient inverseRms rows hidden
  submitRmsNormBackwardWeightF32 executor "Qwen3.6 MLP RMSNorm weight VJP" input normGradient
    inverseRms normWeightGradient rows hidden

/-- Sequential numerical VJP baseline for the Qwen3.6 MLP. -/
def backwardF32 (stream : @& Cuda.Stream)
    (input normWeight normalized inverseRms gate up activation outputGradient activationGradient
      gateGradient upGradient normGradientGate normGradientUp normGradient inputGradient
      normWeightGradient gateWeightGradient upWeightGradient downWeightGradient :
      @& Cuda.Buffer Float32)
    (gateWeight upWeight downWeight : @& Projection.WeightsF32)
    (rows hidden intermediate : UInt32) : IO Unit :=
  submitBackwardF32 (.sequential stream) input normWeight normalized inverseRms gate up activation
    outputGradient activationGradient gateGradient upGradient normGradientGate normGradientUp
    normGradient inputGradient normWeightGradient gateWeightGradient upWeightGradient
    downWeightGradient gateWeight upWeight downWeight rows hidden intermediate

end Cuda.Qwen36.MLP
