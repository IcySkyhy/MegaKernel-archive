/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.FullTrainingMegakernelCore

public section

/-!
# Complete-model GRPO megakernel

This is a separate generated CUDA program from pretraining: model forward, target-token policy
scoring, clipped group-relative objective with direct reference KL, masked logit VJP, complete
model reverse, all LoRA reductions, and AdamW execute in one cooperative launch without a runtime
objective tag.
-/

namespace Cuda.Qwen36.FullGRPOMegakernel

open Cuda.Qwen36.FullTrainingMegakernel

@[cuda_kernel]
def fullGRPOStepF32Kernel (operations : Cuda.DevicePtr UInt32)
    (activations adapters : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (indices : Cuda.DevicePtrTable UInt32)
    (projectionMetadata : Cuda.DevicePtr UInt32)
    (logits logitGradient policy oldPolicy reference rewards advantages coefficients statistics
      loss : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32)
    (scheduleStep : Cuda.DevicePtr UInt32)
    (beta1Power beta2Power inverseBiasCorrection1 inverseBiasCorrection2 :
      Cuda.DevicePtr Float32)
    (descriptor : ProgramDescriptor)
    (hyperparameters : Cuda.Qwen36.Training.AdamWHyperparametersF32)
    (objective : Cuda.Qwen36.Model.GRPOObjectiveF32)
    (rows classes : UInt32) : Cuda.DeviceM Unit := do
  FullTrainingMegakernel.Internal.runForward operations activations adapters matrices indices
    projectionMetadata descriptor
  Cuda.gridSync
  let block ← Cuda.blockIdxX
  FullTrainingMegakernel.Internal.policyRows logits policy targets masks rows classes block
    (← Cuda.gridDimX)
  Cuda.gridSync
  let linear := block * (← Cuda.blockDimX) + (← Cuda.threadIdxX)
  Cuda.Qwen36.Model.grpoObjectiveElementF32 policy oldPolicy reference rewards advantages
    coefficients statistics loss masks objective linear
  Cuda.gridSync
  FullTrainingMegakernel.Internal.grpoGradientRows logits logitGradient coefficients targets masks
    rows classes block (← Cuda.gridDimX)
  Cuda.gridSync
  FullTrainingMegakernel.Internal.runBackward operations activations adapters matrices indices
    projectionMetadata descriptor
  Cuda.gridSync
  FullTrainingMegakernel.Internal.advanceAndOptimize adapters indices projectionMetadata descriptor
    hyperparameters {
      step := scheduleStep, beta1Power, beta2Power, inverseBiasCorrection1,
      inverseBiasCorrection2
    }

/-- Launch one complete outcome-GRPO update as exactly one cooperative CUDA kernel. -/
def launch (stream : @& Cuda.Stream) (program : @& Program)
    (policy oldPolicy reference rewards advantages coefficients statistics :
      @& Cuda.Buffer Float32) (masks : @& Cuda.Buffer UInt32)
    (objective : Cuda.Qwen36.Model.GRPOObjectiveF32)
    (hyperparameters : Cuda.Qwen36.Training.AdamWHyperparametersF32)
    (schedule : @& Cuda.Qwen36.Training.AdamWScheduleF32) : IO Cuda.KernelHandle := do
  hyperparameters.check
  schedule.check
  let occupancy ← fullGRPOStepF32Kernel.occupancy 32 program.sharedMemoryBytes
  let residentBlocks := occupancy.multiprocessorCount * occupancy.activeBlocksPerMultiprocessor
  unless residentBlocks > 0 do
    throw <| IO.userError "Qwen3.8 full GRPO megakernel has zero resident blocks"
  let config := { Program.launchConfig program with
    grid := { x := min program.blocks residentBlocks } }
  fullGRPOStepF32Kernel.launchOn stream config
    program.operations program.activations program.adapters program.matrices program.indices
    program.projectionMetadata program.logits program.logitGradient policy oldPolicy reference
    rewards advantages coefficients statistics program.loss program.targets masks schedule.step
    schedule.beta1Power schedule.beta2Power schedule.inverseBiasCorrection1
    schedule.inverseBiasCorrection2 program.descriptor hyperparameters objective program.rows
    program.classes

end Cuda.Qwen36.FullGRPOMegakernel
