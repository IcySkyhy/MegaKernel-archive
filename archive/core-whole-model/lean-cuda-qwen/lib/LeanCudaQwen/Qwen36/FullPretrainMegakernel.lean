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
# Complete-model pretraining megakernel

One cooperative launch executes the entire lowered model forward, stable mean cross-entropy,
the complete hybrid-decoder VJP, all 497 projection-local LoRA reductions in the production model,
one schedule advance, and every masked AdamW publication.
-/

namespace Cuda.Qwen36.FullPretrainMegakernel

open Cuda.Qwen36.FullTrainingMegakernel

@[cuda_kernel]
def fullPretrainingStepF32Kernel (operations : Cuda.DevicePtr UInt32)
    (activations adapters : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (indices : Cuda.DevicePtrTable UInt32)
    (projectionMetadata : Cuda.DevicePtr UInt32)
    (logits logitGradient rowLoss loss : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32)
    (scheduleStep : Cuda.DevicePtr UInt32)
    (beta1Power beta2Power inverseBiasCorrection1 inverseBiasCorrection2 :
      Cuda.DevicePtr Float32)
    (descriptor : ProgramDescriptor)
    (hyperparameters : Cuda.Qwen36.Training.AdamWHyperparametersF32)
    (rows classes : UInt32) : Cuda.DeviceM Unit := do
  FullTrainingMegakernel.Internal.runForward operations activations adapters matrices indices
    projectionMetadata descriptor
  Cuda.gridSync
  let block ← Cuda.blockIdxX
  FullTrainingMegakernel.Internal.pretrainingRows logits logitGradient rowLoss targets rows classes
    block (← Cuda.gridDimX)
  Cuda.gridSync
  if block == 0 && (← Cuda.threadIdxX) == 0 then
    Cuda.storeFloat32 loss 0 (← FullTrainingMegakernel.Internal.sumRows rowLoss rows 0 0)
  Cuda.gridSync
  FullTrainingMegakernel.Internal.runBackward operations activations adapters matrices indices
    projectionMetadata descriptor
  Cuda.gridSync
  FullTrainingMegakernel.Internal.advanceAndOptimize adapters indices projectionMetadata descriptor
    hyperparameters {
      step := scheduleStep, beta1Power, beta2Power, inverseBiasCorrection1,
      inverseBiasCorrection2
    }

/-- Launch one complete pretraining update as exactly one cooperative CUDA kernel. -/
def launch (stream : @& Cuda.Stream) (program : @& Program)
    (hyperparameters : Cuda.Qwen36.Training.AdamWHyperparametersF32)
    (schedule : @& Cuda.Qwen36.Training.AdamWScheduleF32) : IO Cuda.KernelHandle := do
  hyperparameters.check
  schedule.check
  let occupancy ← fullPretrainingStepF32Kernel.occupancy 32 program.sharedMemoryBytes
  let residentBlocks := occupancy.multiprocessorCount * occupancy.activeBlocksPerMultiprocessor
  unless residentBlocks > 0 do
    throw <| IO.userError "Qwen3.8 full pretraining megakernel has zero resident blocks"
  let config := { Program.launchConfig program with
    grid := { x := min program.blocks residentBlocks } }
  fullPretrainingStepF32Kernel.launchOn stream config
    program.operations program.activations program.adapters program.matrices program.indices
    program.projectionMetadata program.logits program.logitGradient program.rowLoss program.loss
    program.targets schedule.step schedule.beta1Power schedule.beta2Power
    schedule.inverseBiasCorrection1 schedule.inverseBiasCorrection2 program.descriptor
    hyperparameters program.rows program.classes

end Cuda.Qwen36.FullPretrainMegakernel
