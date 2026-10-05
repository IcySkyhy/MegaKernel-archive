/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.LoRA
public import LeanCudaQwen.Qwen36.Model

public section

/-!
# Qwen3.6 resident training

Checked batch geometry, mixed-adapter profile metadata, reusable Float32 AdamW publication, and
the device-tail control node for recurrent CUDA graphs. Batch size and sequence length are
separate invariants: recurrent and attention schedules must never infer sequence boundaries from
the flattened token count.

The graph composer below this boundary uses the same finite Lean CUDA kernels as the independent
Model route. A sequential executor waits after each node; a graph executor appends the identical
linear node sequence and lets tailTrainingGraph start the next step without returning to the host.
-/

namespace Cuda.Qwen36.Training

/-- Host-owned adapter metadata shared by every enabled projection in one training graph. -/
structure LoRAProfile where
  adapterCount : Nat
  rank : Nat
  alpha : Float32
  adapterIds : Array UInt32
  deriving Repr

/-- Which parameter family a resident training graph updates. -/
inductive Profile where
  /-- Update the complete model parameter registry. -/
  | fullModel
  /-- Freeze base weights and update projection-local adapters. -/
  | lora (metadata : LoRAProfile)
  deriving Repr

/-- Host-side shape and recurrence contract for one resident graph. -/
structure Descriptor where
  steps : Nat
  batchSize : Nat
  sequenceLength : Nat
  model : Model.ShapeF32
  profile : Profile

namespace Descriptor

private def checkModelShape (shape : Model.ShapeF32) : Except String Unit := do
  if shape.tokens == 0 || shape.hidden == 0 || shape.intermediate == 0 ||
      shape.vocabulary == 0 then
    throw "Qwen3.6 training model dimensions must be positive"
  if shape.linearKeyHeads == 0 || shape.linearValueHeads == 0 ||
      shape.linearKeyWidth == 0 || shape.linearValueWidth == 0 then
    throw "Qwen3.6 training DeltaNet dimensions must be positive"
  if shape.linearValueHeads % shape.linearKeyHeads != 0 then
    throw "Qwen3.6 training value heads must be divisible by key heads"
  if shape.attentionQueryHeads == 0 || shape.attentionKeyValueHeads == 0 ||
      shape.attentionHeadWidth == 0 then
    throw "Qwen3.6 training attention dimensions must be positive"
  if shape.attentionQueryHeads % shape.attentionKeyValueHeads != 0 then
    throw "Qwen3.6 training query heads must be divisible by key/value heads"
  if shape.rotaryHalf * 2 > shape.attentionHeadWidth then
    throw "Qwen3.6 training rotary width exceeds the attention head width"
  let finite := fun value : Float32 => value.toBits &&& 0x7f800000 != 0x7f800000
  if !(shape.epsilon > 0) || !(shape.queryScale > 0) || !(shape.attentionScale > 0) ||
      !finite shape.epsilon || !finite shape.queryScale || !finite shape.attentionScale then
    throw "Qwen3.6 training numerical scales must be positive and finite"
  return ()

/-- Recheck recurrence, flattened-token, architecture, and adapter invariants. -/
def check (descriptor : Descriptor) : Except String Descriptor := do
  if descriptor.steps == 0 then
    throw "Qwen3.6 resident training requires at least one step"
  if descriptor.steps > 0xffffffff then
    throw "Qwen3.6 resident training step count exceeds UInt32"
  if descriptor.batchSize == 0 || descriptor.sequenceLength == 0 then
    throw "Qwen3.6 resident training batch and sequence dimensions must be positive"
  let rows := descriptor.batchSize * descriptor.sequenceLength
  if rows > 0xffffffff then
    throw "Qwen3.6 resident training token count exceeds UInt32"
  if rows != descriptor.model.tokens.toNat then
    throw <| s!"Qwen3.6 flattened token count {descriptor.model.tokens} does not equal " ++
      s!"batch x sequence ({descriptor.batchSize} x {descriptor.sequenceLength})"
  checkModelShape descriptor.model
  match descriptor.profile with
  | .fullModel => pure ()
  | .lora metadata =>
      discard <| (LoRA.Descriptor.check {
        batchSize := descriptor.batchSize
        sequenceLength := descriptor.sequenceLength
        adapterCount := metadata.adapterCount
        rank := metadata.rank
        inputFeatures := descriptor.model.hidden.toNat
        outputFeatures := descriptor.model.hidden.toNat
        alpha := metadata.alpha
        adapterIds := metadata.adapterIds
      })
  return descriptor

end Descriptor

/-- Device buffers for one Float32 AdamW parameter group. -/
@[struct] structure AdamWBuffersF32 where
  parameter : Cuda.DevicePtr Float32
  gradient : Cuda.DevicePtr Float32
  firstMoment : Cuda.DevicePtr Float32
  secondMoment : Cuda.DevicePtr Float32

/-- Step-invariant AdamW coefficients for a resident schedule. -/
structure AdamWHyperparametersF32 where
  learningRate : Float32
  beta1 : Float32
  beta2 : Float32
  epsilon : Float32
  deriving Cuda.POD

/-- Device-resident AdamW step, running powers, and published bias corrections. -/
@[struct] structure AdamWScheduleBuffersF32 where
  step : Cuda.DevicePtr UInt32
  beta1Power : Cuda.DevicePtr Float32
  beta2Power : Cuda.DevicePtr Float32
  inverseBiasCorrection1 : Cuda.DevicePtr Float32
  inverseBiasCorrection2 : Cuda.DevicePtr Float32

/-- Update and publish one Float32 parameter and its two moments. -/
@[expose, cuda_device, always_inline]
def adamWElementF32 (config : Cuda.Training.AdamW.Config) (buffers : AdamWBuffersF32)
    (elements linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < elements then
    let index := linear.toUSize
    let parameter ← Cuda.loadFloat32 buffers.parameter index
    let gradient ← Cuda.loadFloat32 buffers.gradient index
    let first ← Cuda.loadFloat32 buffers.firstMoment index
    let second ← Cuda.loadFloat32 buffers.secondMoment index
    let result ← Cuda.Training.AdamW.stepFast config parameter gradient { first, second }
    Cuda.storeFloat32 buffers.parameter index result.parameter
    Cuda.storeFloat32 buffers.firstMoment index result.firstMoment
    Cuda.storeFloat32 buffers.secondMoment index result.secondMoment

/-- Advance one resident AdamW step and publish its two bias corrections. -/
@[expose, cuda_device, always_inline]
def advanceAdamWScheduleF32 (hyperparameters : AdamWHyperparametersF32)
    (schedule : AdamWScheduleBuffersF32) : Cuda.DeviceM Unit := do
  let step ← Cuda.loadUInt32 schedule.step 0
  let beta1Power := (← Cuda.loadFloat32 schedule.beta1Power 0) * hyperparameters.beta1
  let beta2Power := (← Cuda.loadFloat32 schedule.beta2Power 0) * hyperparameters.beta2
  let one := Primitives.oneF32
  Cuda.storeUInt32 schedule.step 0 (step + 1)
  Cuda.storeFloat32 schedule.beta1Power 0 beta1Power
  Cuda.storeFloat32 schedule.beta2Power 0 beta2Power
  Cuda.storeFloat32 schedule.inverseBiasCorrection1 0
    (← Cuda.fastDivide one (one - beta1Power))
  Cuda.storeFloat32 schedule.inverseBiasCorrection2 0
    (← Cuda.fastDivide one (one - beta2Power))

/-- Apply one element using bias corrections published by the resident schedule node. -/
@[expose, cuda_device, always_inline]
def residentAdamWElementF32 (hyperparameters : AdamWHyperparametersF32)
    (schedule : AdamWScheduleBuffersF32) (buffers : AdamWBuffersF32)
    (elements : UInt32) (weightDecay : Float32) (linear : UInt32) : Cuda.DeviceM Unit := do
  let config : Cuda.Training.AdamW.Config := {
    learningRate := hyperparameters.learningRate
    beta1 := hyperparameters.beta1
    beta2 := hyperparameters.beta2
    inverseBiasCorrection1 := ← Cuda.loadFloat32 schedule.inverseBiasCorrection1 0
    inverseBiasCorrection2 := ← Cuda.loadFloat32 schedule.inverseBiasCorrection2 0
    epsilon := hyperparameters.epsilon
    weightDecay
  }
  adamWElementF32 config buffers elements linear

/-- Element-parallel Float32 AdamW update used by sequential and resident full-model schedules. -/
@[cuda_kernel]
def adamWF32Kernel (config : Cuda.Training.AdamW.Config)
    (parameter gradient firstMoment secondMoment : Cuda.DevicePtr Float32)
    (elements : UInt32) : Cuda.DeviceM Unit := do
  adamWElementF32 config { parameter, gradient, firstMoment, secondMoment } elements
    (← Primitives.elementIndex)

/-- One-thread schedule node placed between reverse-mode gradients and parameter publication. -/
@[cuda_kernel]
def advanceAdamWScheduleF32Kernel (hyperparameters : AdamWHyperparametersF32)
    (step : Cuda.DevicePtr UInt32) (beta1Power beta2Power inverseBiasCorrection1
      inverseBiasCorrection2 : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  if (← Cuda.blockIdxX) == 0 && (← Cuda.threadIdxX) == 0 then
    advanceAdamWScheduleF32 hyperparameters {
      step, beta1Power, beta2Power, inverseBiasCorrection1, inverseBiasCorrection2
    }

/-- Element-parallel AdamW update driven by the device-resident schedule. -/
@[cuda_kernel]
def residentAdamWF32Kernel (hyperparameters : AdamWHyperparametersF32)
    (step : Cuda.DevicePtr UInt32) (beta1Power beta2Power inverseBiasCorrection1
      inverseBiasCorrection2 : Cuda.DevicePtr Float32)
    (parameter gradient firstMoment secondMoment : Cuda.DevicePtr Float32)
    (elements : UInt32) (weightDecay : Float32) : Cuda.DeviceM Unit := do
  residentAdamWElementF32 hyperparameters {
    step, beta1Power, beta2Power, inverseBiasCorrection1, inverseBiasCorrection2
  } { parameter, gradient, firstMoment, secondMoment } elements weightDecay
    (← Primitives.elementIndex)

/-- Adapter-masked resident AdamW using the same device-published bias corrections as base groups. -/
@[cuda_kernel]
def residentLoRAAdamWF32Kernel (hyperparameters : AdamWHyperparametersF32)
    (_step : Cuda.DevicePtr UInt32) (_beta1Power _beta2Power inverseBiasCorrection1
      inverseBiasCorrection2 : Cuda.DevicePtr Float32)
    (adapterA adapterB adapterAGradient adapterBGradient adapterAFirstMoment adapterASecondMoment
      adapterBFirstMoment adapterBSecondMoment : Cuda.DevicePtr Float32)
    (updateMask : Cuda.DevicePtr UInt32) (shape : LoRA.Shape) (weightDecay : Float32) :
    Cuda.DeviceM Unit := do
  let config : Cuda.Training.AdamW.Config := {
    learningRate := hyperparameters.learningRate
    beta1 := hyperparameters.beta1
    beta2 := hyperparameters.beta2
    inverseBiasCorrection1 := ← Cuda.loadFloat32 inverseBiasCorrection1 0
    inverseBiasCorrection2 := ← Cuda.loadFloat32 inverseBiasCorrection2 0
    epsilon := hyperparameters.epsilon
    weightDecay
  }
  LoRA.adamWElementF32 config {
    adapterA, adapterB, adapterAGradient, adapterBGradient,
    adapterAFirstMoment, adapterASecondMoment, adapterBFirstMoment, adapterBSecondMoment,
    updateMask
  } shape (← Primitives.elementIndex)

/--
Finish one graph step and relaunch the same executable graph from the device when work remains.

remaining[0] is initialized to the checked step count. Exactly one one-thread tail node appears
after publication of every parameter and optimizer state.
-/
@[cuda_kernel]
def tailTrainingGraph (graphHandle : Cuda.DevicePtr UInt64)
    (remaining : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  if (← Cuda.blockIdxX) == 0 && (← Cuda.threadIdxX) == 0 then
    let before ← Cuda.loadUInt32 remaining 0
    if before == 0 then
      Cuda.panic 0x3636 0
    else
      Cuda.storeUInt32 remaining 0 (before - 1)
      if before > 1 then
        Cuda.DeviceGraph.tailLaunch (← Cuda.loadUInt64 graphHandle 0)

/-- One mutable Float32 parameter family and its device-resident optimizer state. -/
structure ParameterGroupF32 where
  parameter : Cuda.Buffer Float32
  gradient : Cuda.Buffer Float32
  firstMoment : Cuda.Buffer Float32
  secondMoment : Cuda.Buffer Float32
  elements : UInt32
  weightDecay : Float32

namespace ParameterGroupF32

/-- Reject empty or differently-sized parameter, gradient, and moment allocations. -/
def check (group : ParameterGroupF32) : IO Unit := do
  if group.elements == 0 then
    throw <| IO.userError "Qwen3.6 optimizer parameter group must be nonempty"
  if !(group.weightDecay ≥ 0) then
    throw <| IO.userError "Qwen3.6 optimizer weight decay must be nonnegative"
  let expected := group.elements.toUSize * 4
  unless (← group.parameter.byteSize) == expected do
    throw <| IO.userError "Qwen3.6 optimizer parameter allocation size mismatch"
  unless (← group.gradient.byteSize) == expected do
    throw <| IO.userError "Qwen3.6 optimizer gradient allocation size mismatch"
  unless (← group.firstMoment.byteSize) == expected do
    throw <| IO.userError "Qwen3.6 optimizer first-moment allocation size mismatch"
  unless (← group.secondMoment.byteSize) == expected do
    throw <| IO.userError "Qwen3.6 optimizer second-moment allocation size mismatch"

end ParameterGroupF32

/-- Projection-independent AdamW state paired with one entry in the canonical model registry. -/
structure ParameterStateF32 where
  firstMoment : Cuda.Buffer Float32
  secondMoment : Cuda.Buffer Float32
  weightDecay : Float32

/-- One externally allocatable entry in the canonical full-model parameter order. -/
structure FullModelParameterSpecF32 where
  label : String
  elements : UInt32
  deriving Repr

/-- Optimizer state for every parameter in the canonical full-model registry. -/
structure FullModelOptimizerF32 where
  states : Array ParameterStateF32

/-- Profile-matched mutable optimizer ownership for one model step. -/
inductive UpdateStateF32 where
  | fullModel (optimizer : FullModelOptimizerF32)
  | lora

/-- Host-owned handles for the device-resident AdamW schedule. -/
structure AdamWScheduleF32 where
  step : Cuda.Buffer UInt32
  beta1Power : Cuda.Buffer Float32
  beta2Power : Cuda.Buffer Float32
  inverseBiasCorrection1 : Cuda.Buffer Float32
  inverseBiasCorrection2 : Cuda.Buffer Float32

/-- Host-checkpointable state of the device-resident AdamW bias-correction schedule. -/
structure AdamWScheduleSnapshotF32 where
  step : ByteArray
  beta1Power : ByteArray
  beta2Power : ByteArray
  inverseBiasCorrection1 : ByteArray
  inverseBiasCorrection2 : ByteArray
  deriving BEq

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def pushUInt64 (bytes : ByteArray) (value : UInt64) : ByteArray := Id.run do
  let mut result := bytes
  for shift in [:8] do
    result := result.push (value >>> (8 * shift).toUInt64).toUInt8
  return result

private def scalarFloat32Bytes (value : Float32) : ByteArray :=
  pushUInt32 ByteArray.empty value.toBits

private def readUInt32 (bytes : ByteArray) : UInt32 :=
  bytes[0]!.toUInt32 ||| (bytes[1]!.toUInt32 <<< 8) |||
    (bytes[2]!.toUInt32 <<< 16) ||| (bytes[3]!.toUInt32 <<< 24)

private def finiteFloat32 (value : Float32) : Bool :=
  value.toBits &&& 0x7f800000 != 0x7f800000

namespace AdamWHyperparametersF32

/-- Reject invalid step-invariant AdamW coefficients. -/
def check (hyperparameters : AdamWHyperparametersF32) : IO Unit := do
  unless hyperparameters.learningRate > 0 && finiteFloat32 hyperparameters.learningRate do
    throw <| IO.userError "Qwen3.6 AdamW learning rate must be positive and finite"
  unless hyperparameters.beta1 > 0 && hyperparameters.beta1 < 1 &&
      finiteFloat32 hyperparameters.beta1 do
    throw <| IO.userError "Qwen3.6 AdamW beta1 must be finite and lie in (0, 1)"
  unless hyperparameters.beta2 > 0 && hyperparameters.beta2 < 1 &&
      finiteFloat32 hyperparameters.beta2 do
    throw <| IO.userError "Qwen3.6 AdamW beta2 must be finite and lie in (0, 1)"
  unless hyperparameters.epsilon > 0 && finiteFloat32 hyperparameters.epsilon do
    throw <| IO.userError "Qwen3.6 AdamW epsilon must be positive and finite"

end AdamWHyperparametersF32

namespace AdamWScheduleF32

/-- Allocate the canonical zero-step AdamW schedule directly on `stream`. -/
def allocate (stream : @& Cuda.Stream) : IO AdamWScheduleF32 := do
  let step ← Cuda.Buffer.alloc UInt32 1
  let beta1Power ← Cuda.Buffer.alloc Float32 1
  let beta2Power ← Cuda.Buffer.alloc Float32 1
  let inverseBiasCorrection1 ← Cuda.Buffer.alloc Float32 1
  let inverseBiasCorrection2 ← Cuda.Buffer.alloc Float32 1
  step.copyFrom (pushUInt32 ByteArray.empty 0) stream
  beta1Power.copyFrom (scalarFloat32Bytes 1) stream
  beta2Power.copyFrom (scalarFloat32Bytes 1) stream
  inverseBiasCorrection1.copyFrom (scalarFloat32Bytes 0) stream
  inverseBiasCorrection2.copyFrom (scalarFloat32Bytes 0) stream
  return { step, beta1Power, beta2Power, inverseBiasCorrection1, inverseBiasCorrection2 }

/-- Require five one-element device allocations for a resident AdamW schedule. -/
def check (schedule : AdamWScheduleF32) : IO Unit := do
  unless (← schedule.step.byteSize) == 4 do
    throw <| IO.userError "Qwen3.6 AdamW step allocation must contain one UInt32"
  for (label, buffer) in #[
      ("beta1 power", schedule.beta1Power),
      ("beta2 power", schedule.beta2Power),
      ("inverse beta1 correction", schedule.inverseBiasCorrection1),
      ("inverse beta2 correction", schedule.inverseBiasCorrection2)
    ] do
    unless (← buffer.byteSize) == 4 do
      throw <| IO.userError s!"Qwen3.6 AdamW {label} allocation must contain one Float32"

/-- Copy the complete bias-correction schedule to host-owned checkpoint bytes. -/
def snapshot (schedule : @& AdamWScheduleF32) (stream : @& Cuda.Stream) :
    IO AdamWScheduleSnapshotF32 := do
  schedule.check
  return {
    step := ← schedule.step.copyTo stream
    beta1Power := ← schedule.beta1Power.copyTo stream
    beta2Power := ← schedule.beta2Power.copyTo stream
    inverseBiasCorrection1 := ← schedule.inverseBiasCorrection1.copyTo stream
    inverseBiasCorrection2 := ← schedule.inverseBiasCorrection2.copyTo stream
  }

/-- Validate a schedule snapshot against its fixed destination geometry without restoring it. -/
def checkSnapshot (schedule : @& AdamWScheduleF32)
    (snapshot : @& AdamWScheduleSnapshotF32) : IO Unit := do
  schedule.check
  for (label, bytes) in #[
      ("step", snapshot.step),
      ("beta1 power", snapshot.beta1Power),
      ("beta2 power", snapshot.beta2Power),
      ("inverse beta1 correction", snapshot.inverseBiasCorrection1),
      ("inverse beta2 correction", snapshot.inverseBiasCorrection2)
    ] do
    unless bytes.size == 4 do
      throw <| IO.userError s!"Qwen3.6 AdamW snapshot {label} size mismatch"

/-- Restore a previously snapshotted AdamW schedule on `stream`. -/
def restore (schedule : @& AdamWScheduleF32) (snapshot : @& AdamWScheduleSnapshotF32)
    (stream : @& Cuda.Stream) : IO Unit := do
  schedule.checkSnapshot snapshot
  schedule.step.copyFrom snapshot.step stream
  schedule.beta1Power.copyFrom snapshot.beta1Power stream
  schedule.beta2Power.copyFrom snapshot.beta2Power stream
  schedule.inverseBiasCorrection1.copyFrom snapshot.inverseBiasCorrection1 stream
  schedule.inverseBiasCorrection2.copyFrom snapshot.inverseBiasCorrection2 stream

end AdamWScheduleF32

/-- Training-facing alias of the shared finite-kernel executor. -/
abbrev Executor := Cuda.Qwen36.Executor

/-- Submit one checked Float32 AdamW parameter group to either execution mode. -/
def submitAdamWF32 (executor : @& Executor) (config : Cuda.Training.AdamW.Config)
    (group : @& ParameterGroupF32) : IO Unit := do
  let config := { config with weightDecay := group.weightDecay }
  let launch := Primitives.elementConfig group.elements
  match executor with
  | .sequential stream =>
      (← adamWF32Kernel.launchOn stream launch config group.parameter group.gradient
        group.firstMoment group.secondMoment group.elements).waitChecked
        "Qwen3.6 Float32 AdamW"
  | .graph builder =>
      adamWF32Kernel.addToGraph builder launch config group.parameter group.gradient
        group.firstMoment group.secondMoment group.elements

/-- Advance the resident AdamW bias-correction schedule once. -/
def submitAdvanceAdamWScheduleF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32) : IO Unit :=
  executor.submit "Qwen3.6 resident AdamW schedule"
    (fun stream => advanceAdamWScheduleF32Kernel.launchOn stream {
      grid := { x := 1 }
      block := { x := 1 }
      blockArenaBytes := 0
    } hyperparameters schedule.step schedule.beta1Power schedule.beta2Power
      schedule.inverseBiasCorrection1 schedule.inverseBiasCorrection2)
    (fun builder => advanceAdamWScheduleF32Kernel.addToGraph builder {
      grid := { x := 1 }
      block := { x := 1 }
      blockArenaBytes := 0
    } hyperparameters schedule.step schedule.beta1Power schedule.beta2Power
      schedule.inverseBiasCorrection1 schedule.inverseBiasCorrection2)

/-- Update one parameter group using bias corrections published on device for this step. -/
def submitResidentAdamWF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (group : @& ParameterGroupF32) : IO Unit :=
  let launch := Primitives.elementConfig group.elements
  executor.submit "Qwen3.6 resident Float32 AdamW"
    (fun stream => residentAdamWF32Kernel.launchOn stream launch hyperparameters schedule.step
      schedule.beta1Power schedule.beta2Power schedule.inverseBiasCorrection1
      schedule.inverseBiasCorrection2 group.parameter group.gradient group.firstMoment
      group.secondMoment group.elements group.weightDecay)
    (fun builder => residentAdamWF32Kernel.addToGraph builder launch hyperparameters schedule.step
      schedule.beta1Power schedule.beta2Power schedule.inverseBiasCorrection1
      schedule.inverseBiasCorrection2 group.parameter group.gradient group.firstMoment
      group.secondMoment group.elements group.weightDecay)

/-- Update one projection-local adapter pair while preserving every masked adapter and moment. -/
def submitResidentLoRAAdamWF32 (executor : @& Executor) (label : String)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (adapter : @& Projection.AdapterF32) : IO Unit := do
  let shape := adapter.shape
  let elements := shape.adapterCount *
    (shape.rank * shape.inputFeatures + shape.outputFeatures * shape.rank)
  let launch := Primitives.elementConfig elements
  executor.submit label
    (fun stream => residentLoRAAdamWF32Kernel.launchOn stream launch hyperparameters schedule.step
      schedule.beta1Power schedule.beta2Power schedule.inverseBiasCorrection1
      schedule.inverseBiasCorrection2 adapter.adapterA adapter.adapterB adapter.adapterAGradient
      adapter.adapterBGradient adapter.adapterAFirstMoment adapter.adapterASecondMoment
      adapter.adapterBFirstMoment adapter.adapterBSecondMoment adapter.updateMask shape
      adapter.weightDecay)
    (fun builder => residentLoRAAdamWF32Kernel.addToGraph builder launch hyperparameters
      schedule.step schedule.beta1Power schedule.beta2Power schedule.inverseBiasCorrection1
      schedule.inverseBiasCorrection2 adapter.adapterA adapter.adapterB adapter.adapterAGradient
      adapter.adapterBGradient adapter.adapterAFirstMoment adapter.adapterASecondMoment
      adapter.adapterBFirstMoment adapter.adapterBSecondMoment adapter.updateMask shape
      adapter.weightDecay)

private def checkProjectionProfile (descriptor : @& Descriptor) (label : String)
    (projection : @& Projection.WeightsF32) (inputFeatures outputFeatures : UInt32) : IO Unit :=
  match descriptor.profile, projection with
  | .fullModel, .base _ => pure ()
  | .fullModel, .lora _ _ =>
      throw <| IO.userError s!"Qwen3.6 full-model profile contains LoRA projection {label}"
  | .fullModel, .frozenBF16 _ =>
      throw <| IO.userError s!"Qwen3.6 full-model profile contains frozen BF16 projection {label}"
  | .fullModel, .loraFrozenBF16 _ _ =>
      throw <| IO.userError s!"Qwen3.6 full-model profile contains frozen LoRA projection {label}"
  | .lora _, .base _ =>
      throw <| IO.userError s!"Qwen3.6 LoRA profile is missing adapter projection {label}"
  | .lora _, .frozenBF16 _ =>
      throw <| IO.userError s!"Qwen3.6 LoRA profile is missing adapter projection {label}"
  | .lora metadata, .lora baseWeight adapter => do
      adapter.check descriptor.model.tokens inputFeatures outputFeatures
      let shape := adapter.shape
      unless shape.sequenceLength == descriptor.sequenceLength.toUInt32 &&
          shape.adapterCount == metadata.adapterCount.toUInt32 &&
          shape.rank == metadata.rank.toUInt32 &&
          shape.scale == metadata.alpha / metadata.rank.toUInt32.toFloat32 do
        throw <| IO.userError s!"Qwen3.6 LoRA metadata mismatch at projection {label}"
      unless (← baseWeight.byteSize) == (outputFeatures * inputFeatures * 4).toUSize do
        throw <| IO.userError s!"Qwen3.6 frozen base size mismatch at projection {label}"
  | .lora metadata, .loraFrozenBF16 baseWeight adapter => do
      adapter.check descriptor.model.tokens inputFeatures outputFeatures
      baseWeight.check descriptor.model.tokens inputFeatures outputFeatures
      let shape := adapter.shape
      unless shape.sequenceLength == descriptor.sequenceLength.toUInt32 &&
          shape.adapterCount == metadata.adapterCount.toUInt32 &&
          shape.rank == metadata.rank.toUInt32 &&
          shape.scale == metadata.alpha / metadata.rank.toUInt32.toFloat32 do
        throw <| IO.userError s!"Qwen3.6 LoRA metadata mismatch at projection {label}"

private def checkModelProjectionProfiles (descriptor : @& Descriptor)
    (weights : @& Model.WeightsF32) : IO Unit := do
  let shape := descriptor.model
  for index in [:weights.layers.size] do
    let some layer := weights.layers[index]?
      | throw <| IO.userError s!"Qwen3.6 missing layer {index} during projection audit"
    match layer.mixer with
    | .deltaNet mixer =>
        let convolutionWidth :=
          2 * shape.linearKeyHeads * shape.linearKeyWidth +
            shape.linearValueHeads * shape.linearValueWidth
        checkProjectionProfile descriptor s!"layer {index} DeltaNet QKV" mixer.queryKeyValue
          shape.hidden convolutionWidth
        checkProjectionProfile descriptor s!"layer {index} DeltaNet z" mixer.z shape.hidden
          (shape.linearValueHeads * shape.linearValueWidth)
        checkProjectionProfile descriptor s!"layer {index} DeltaNet b" mixer.b shape.hidden
          shape.linearValueHeads
        checkProjectionProfile descriptor s!"layer {index} DeltaNet a" mixer.a shape.hidden
          shape.linearValueHeads
        checkProjectionProfile descriptor s!"layer {index} DeltaNet output" mixer.output
          (shape.linearValueHeads * shape.linearValueWidth) shape.hidden
    | .attention mixer =>
        checkProjectionProfile descriptor s!"layer {index} attention query" mixer.query
          shape.hidden (2 * shape.attentionQueryHeads * shape.attentionHeadWidth)
        checkProjectionProfile descriptor s!"layer {index} attention key" mixer.key shape.hidden
          (shape.attentionKeyValueHeads * shape.attentionHeadWidth)
        checkProjectionProfile descriptor s!"layer {index} attention value" mixer.value
          shape.hidden (shape.attentionKeyValueHeads * shape.attentionHeadWidth)
        checkProjectionProfile descriptor s!"layer {index} attention output" mixer.output
          (shape.attentionQueryHeads * shape.attentionHeadWidth) shape.hidden
    checkProjectionProfile descriptor s!"layer {index} MLP gate" layer.mlp.gate shape.hidden
      shape.intermediate
    checkProjectionProfile descriptor s!"layer {index} MLP up" layer.mlp.up shape.hidden
      shape.intermediate
    checkProjectionProfile descriptor s!"layer {index} MLP down" layer.mlp.down
      shape.intermediate shape.hidden
  checkProjectionProfile descriptor "language-model head" weights.lmHead shape.hidden
    shape.vocabulary

private structure BoundParameterF32 where
  label : String
  parameter : Cuda.Buffer Float32
  gradient : Cuda.Buffer Float32
  elements : UInt32

private def requireBaseProjection (label : String) (projection : @& Projection.WeightsF32) :
    IO (Cuda.Buffer Float32) :=
  match projection with
  | .base weight => pure weight
  | .lora _ _ =>
      throw <| IO.userError s!"Qwen3.6 full-model registry contains LoRA projection {label}"
  | .frozenBF16 _ | .loraFrozenBF16 _ _ =>
      throw <| IO.userError s!"Qwen3.6 full-model registry contains frozen projection {label}"

private def fullModelParameterBindingsF32 (weights : @& Model.WeightsF32)
    (backward : @& Model.BackwardF32) (shape : @& Model.ShapeF32) :
    IO (Array BoundParameterF32) := do
  unless weights.layers.size == backward.layers.size do
    throw <| IO.userError "Qwen3.6 full-model registry layer-count mismatch"
  let embedding ← match weights.embedding with
    | .f32 weight => pure weight
    | .frozenBF16 _ =>
        throw <| IO.userError "Qwen3.6 full-model registry contains frozen token embeddings"
  let mut bindings : Array BoundParameterF32 := #[{
    label := "token embedding"
    parameter := embedding
    gradient := backward.embeddingWeightGradient
    elements := shape.vocabulary * shape.hidden
  }]
  for index in [:weights.layers.size] do
    let some layer := weights.layers[index]?
      | throw <| IO.userError s!"Qwen3.6 full-model registry missing layer {index}"
    let some reverse := backward.layers[index]?
      | throw <| IO.userError s!"Qwen3.6 full-model registry missing reverse layer {index}"
    match layer.mixer, reverse.mixer with
    | .deltaNet mixer, .deltaNet gradient =>
        let convolutionWidth :=
          2 * shape.linearKeyHeads * shape.linearKeyWidth +
            shape.linearValueHeads * shape.linearValueWidth
        let valueWidth := shape.linearValueHeads * shape.linearValueWidth
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet input norm"
          parameter := mixer.inputNorm
          gradient := gradient.inputNormWeightGradient
          elements := shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet QKV"
          parameter := ← requireBaseProjection s!"layer {index} DeltaNet QKV"
            mixer.queryKeyValue
          gradient := gradient.queryKeyValueWeightGradient
          elements := convolutionWidth * shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet z"
          parameter := ← requireBaseProjection s!"layer {index} DeltaNet z" mixer.z
          gradient := gradient.zWeightGradient
          elements := valueWidth * shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet b"
          parameter := ← requireBaseProjection s!"layer {index} DeltaNet b" mixer.b
          gradient := gradient.bWeightGradient
          elements := shape.linearValueHeads * shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet a"
          parameter := ← requireBaseProjection s!"layer {index} DeltaNet a" mixer.a
          gradient := gradient.aWeightGradient
          elements := shape.linearValueHeads * shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet convolution"
          parameter := mixer.convolution
          gradient := gradient.convolutionWeightGradient
          elements := convolutionWidth * 4
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet A-log"
          parameter := mixer.aLog
          gradient := gradient.aLogGradient
          elements := shape.linearValueHeads
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet dt bias"
          parameter := mixer.dtBias
          gradient := gradient.dtBiasGradient
          elements := shape.linearValueHeads
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet gated norm"
          parameter := mixer.gatedNorm
          gradient := gradient.gatedNormWeightGradient
          elements := shape.linearValueWidth
        }
        bindings := bindings.push {
          label := s!"layer {index} DeltaNet output"
          parameter := ← requireBaseProjection s!"layer {index} DeltaNet output" mixer.output
          gradient := gradient.outputWeightGradient
          elements := shape.hidden * valueWidth
        }
    | .attention mixer, .attention gradient =>
        let queryWidth := shape.attentionQueryHeads * shape.attentionHeadWidth
        let keyValueWidth := shape.attentionKeyValueHeads * shape.attentionHeadWidth
        bindings := bindings.push {
          label := s!"layer {index} attention input norm"
          parameter := mixer.inputNorm
          gradient := gradient.inputNormWeightGradient
          elements := shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} attention query norm"
          parameter := mixer.queryNorm
          gradient := gradient.queryNormWeightGradient
          elements := shape.attentionHeadWidth
        }
        bindings := bindings.push {
          label := s!"layer {index} attention key norm"
          parameter := mixer.keyNorm
          gradient := gradient.keyNormWeightGradient
          elements := shape.attentionHeadWidth
        }
        bindings := bindings.push {
          label := s!"layer {index} attention query"
          parameter := ← requireBaseProjection s!"layer {index} attention query" mixer.query
          gradient := gradient.queryWeightGradient
          elements := 2 * queryWidth * shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} attention key"
          parameter := ← requireBaseProjection s!"layer {index} attention key" mixer.key
          gradient := gradient.keyWeightGradient
          elements := keyValueWidth * shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} attention value"
          parameter := ← requireBaseProjection s!"layer {index} attention value" mixer.value
          gradient := gradient.valueWeightGradient
          elements := keyValueWidth * shape.hidden
        }
        bindings := bindings.push {
          label := s!"layer {index} attention output"
          parameter := ← requireBaseProjection s!"layer {index} attention output" mixer.output
          gradient := gradient.outputWeightGradient
          elements := shape.hidden * queryWidth
        }
    | _, _ =>
        throw <| IO.userError s!"Qwen3.6 full-model registry mixer mismatch at layer {index}"
    bindings := bindings.push {
      label := s!"layer {index} MLP norm"
      parameter := layer.mlp.norm
      gradient := reverse.mlp.normWeightGradient
      elements := shape.hidden
    }
    bindings := bindings.push {
      label := s!"layer {index} MLP gate"
      parameter := ← requireBaseProjection s!"layer {index} MLP gate" layer.mlp.gate
      gradient := reverse.mlp.gateWeightGradient
      elements := shape.intermediate * shape.hidden
    }
    bindings := bindings.push {
      label := s!"layer {index} MLP up"
      parameter := ← requireBaseProjection s!"layer {index} MLP up" layer.mlp.up
      gradient := reverse.mlp.upWeightGradient
      elements := shape.intermediate * shape.hidden
    }
    bindings := bindings.push {
      label := s!"layer {index} MLP down"
      parameter := ← requireBaseProjection s!"layer {index} MLP down" layer.mlp.down
      gradient := reverse.mlp.downWeightGradient
      elements := shape.hidden * shape.intermediate
    }
  bindings := bindings.push {
    label := "final norm"
    parameter := weights.finalNorm
    gradient := backward.finalNormWeightGradient
    elements := shape.hidden
  }
  bindings := bindings.push {
    label := "language-model head"
    parameter := ← requireBaseProjection "language-model head" weights.lmHead
    gradient := backward.lmHeadWeightGradient
    elements := shape.vocabulary * shape.hidden
  }
  return bindings

/-- Report the exact allocation order required by the complete full-model AdamW registry. -/
def fullModelParameterSpecsF32 (weights : @& Model.WeightsF32)
    (backward : @& Model.BackwardF32) (shape : @& Model.ShapeF32) :
    IO (Array FullModelParameterSpecF32) := do
  let bindings ← fullModelParameterBindingsF32 weights backward shape
  return bindings.map fun binding => { label := binding.label, elements := binding.elements }

/-- Bind canonical model parameters and gradients to their checked optimizer-state allocations. -/
def fullModelParameterGroupsF32 (weights : @& Model.WeightsF32)
    (backward : @& Model.BackwardF32) (shape : @& Model.ShapeF32)
    (optimizer : @& FullModelOptimizerF32) : IO (Array ParameterGroupF32) := do
  let bindings ← fullModelParameterBindingsF32 weights backward shape
  unless bindings.size == optimizer.states.size do
    throw <| IO.userError <|
      s!"Qwen3.6 full-model optimizer expected {bindings.size} states, found " ++
      s!"{optimizer.states.size}"
  let mut groups : Array ParameterGroupF32 := #[]
  for index in [:bindings.size] do
    match bindings[index]?, optimizer.states[index]? with
    | some binding, some state =>
        let group : ParameterGroupF32 := {
          parameter := binding.parameter
          gradient := binding.gradient
          firstMoment := state.firstMoment
          secondMoment := state.secondMoment
          elements := binding.elements
          weightDecay := state.weightDecay
        }
        try group.check catch error =>
          throw <| IO.userError s!"Qwen3.6 {binding.label}: {error}"
        groups := groups.push group
    | _, _ =>
        throw <| IO.userError s!"Qwen3.6 full-model optimizer index mismatch at {index}"
  return groups

/-- All buffers needed by one complete full-model training step. -/
structure FullModelStepF32 where
  weights : Model.WeightsF32
  forward : Model.ForwardF32
  backward : Model.BackwardF32
  tokenIds : Cuda.Buffer UInt32
  targets : Cuda.Buffer UInt32
  updateState : UpdateStateF32

namespace FullModelStepF32

/-- Validate host-visible model, batch, target, optimizer, and schedule allocations. -/
def check (descriptor : Descriptor) (hyperparameters : AdamWHyperparametersF32)
    (schedule : @& AdamWScheduleF32) (step : @& FullModelStepF32) : IO Descriptor := do
  let descriptor ← match descriptor.check with
    | .ok descriptor => pure descriptor
    | .error message => throw <| IO.userError message
  hyperparameters.check
  schedule.check
  unless step.weights.layers.size == step.forward.layers.size &&
      step.weights.layers.size == step.backward.layers.size do
    throw <| IO.userError "Qwen3.6 full-model step layer-count mismatch"
  checkModelProjectionProfiles descriptor step.weights
  let idBytes := descriptor.model.tokens.toUSize * 4
  unless (← step.tokenIds.byteSize) == idBytes do
    throw <| IO.userError "Qwen3.6 full-model token allocation size mismatch"
  unless (← step.targets.byteSize) == idBytes do
    throw <| IO.userError "Qwen3.6 full-model target allocation size mismatch"
  match descriptor.profile, step.updateState with
  | .fullModel, .fullModel optimizer =>
      discard <| fullModelParameterGroupsF32 step.weights step.backward descriptor.model optimizer
  | .lora _, .lora => pure ()
  | .fullModel, .lora =>
      throw <| IO.userError "Qwen3.6 full-model step requires the complete optimizer registry"
  | .lora _, .fullModel _ =>
      throw <| IO.userError "Qwen3.6 LoRA step must not contain base optimizer state"
  return descriptor

end FullModelStepF32

private partial def submitResidentParameterGroupsF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (groups : @& Array ParameterGroupF32) (index : Nat) : IO Unit := do
  if index < groups.size then
    match groups[index]? with
    | some group =>
        submitResidentAdamWF32 executor hyperparameters schedule group
        submitResidentParameterGroupsF32 executor hyperparameters schedule groups (index + 1)
    | none =>
        throw <| IO.userError "Qwen3.6 full-model parameter-group index mismatch"

private def submitProjectionLoRAUpdateF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (label : String) (projection : @& Projection.WeightsF32) : IO Unit :=
  match projection with
  | .lora _ adapter =>
      submitResidentLoRAAdamWF32 executor label hyperparameters schedule adapter
  | .loraFrozenBF16 _ adapter =>
      submitResidentLoRAAdamWF32 executor label hyperparameters schedule adapter
  | .base _ | .frozenBF16 _ =>
      throw <| IO.userError s!"Qwen3.6 LoRA update encountered base projection {label}"

private def submitLayerLoRAUpdatesF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (index : Nat) (layer : @& Model.LayerWeightsF32) : IO Unit := do
  match layer.mixer with
  | .deltaNet mixer =>
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} DeltaNet QKV adapter AdamW" mixer.queryKeyValue
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} DeltaNet z adapter AdamW" mixer.z
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} DeltaNet b adapter AdamW" mixer.b
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} DeltaNet a adapter AdamW" mixer.a
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} DeltaNet output adapter AdamW" mixer.output
  | .attention mixer =>
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} attention query adapter AdamW" mixer.query
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} attention key adapter AdamW" mixer.key
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} attention value adapter AdamW" mixer.value
      submitProjectionLoRAUpdateF32 executor hyperparameters schedule
        s!"Qwen3.6 layer {index} attention output adapter AdamW" mixer.output
  submitProjectionLoRAUpdateF32 executor hyperparameters schedule
    s!"Qwen3.6 layer {index} MLP gate adapter AdamW" layer.mlp.gate
  submitProjectionLoRAUpdateF32 executor hyperparameters schedule
    s!"Qwen3.6 layer {index} MLP up adapter AdamW" layer.mlp.up
  submitProjectionLoRAUpdateF32 executor hyperparameters schedule
    s!"Qwen3.6 layer {index} MLP down adapter AdamW" layer.mlp.down

private partial def submitLayerArrayLoRAUpdatesF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (layers : @& Array Model.LayerWeightsF32) (index : Nat) : IO Unit := do
  if index < layers.size then
    match layers[index]? with
    | some layer =>
        submitLayerLoRAUpdatesF32 executor hyperparameters schedule index layer
        submitLayerArrayLoRAUpdatesF32 executor hyperparameters schedule layers (index + 1)
    | none =>
        throw <| IO.userError "Qwen3.6 LoRA update layer index mismatch"

private def submitModelLoRAUpdatesF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (weights : @& Model.WeightsF32) : IO Unit := do
  submitLayerArrayLoRAUpdatesF32 executor hyperparameters schedule weights.layers 0
  submitProjectionLoRAUpdateF32 executor hyperparameters schedule
    "Qwen3.6 language-model-head adapter AdamW" weights.lmHead

private def submitCheckedFullModelBackwardUpdateF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (descriptor : Descriptor) (step : @& FullModelStepF32) : IO Unit := do
  let layout : Cuda.Qwen36.SequenceLayout := {
    batchSize := descriptor.batchSize.toUInt32
    sequenceLength := descriptor.sequenceLength.toUInt32
  }
  Model.submitBatchedBackwardF32 executor step.weights step.forward step.backward step.tokenIds
    layout descriptor.model
  submitAdvanceAdamWScheduleF32 executor hyperparameters schedule
  match descriptor.profile, step.updateState with
  | .fullModel, .fullModel optimizer =>
      let groups ← fullModelParameterGroupsF32 step.weights step.backward descriptor.model optimizer
      submitResidentParameterGroupsF32 executor hyperparameters schedule groups 0
  | .lora _, .lora =>
      submitModelLoRAUpdatesF32 executor hyperparameters schedule step.weights
  | _, _ =>
      throw <| IO.userError "Qwen3.6 checked training profile changed during submission"

/--
Submit the objective-independent `backward -> AdamW` suffix after a caller has populated the
full-model logit VJP. This permits cross-entropy, DPO, and verifier-driven objectives to share the
same checked reverse/update implementation.
-/
def submitFullModelBackwardUpdateF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (descriptor : Descriptor) (step : @& FullModelStepF32) : IO Unit := do
  let descriptor ← step.check descriptor hyperparameters schedule
  submitCheckedFullModelBackwardUpdateF32 executor hyperparameters schedule descriptor step

/--
Submit one complete `forward -> loss -> backward -> AdamW` step to a stream or CUDA graph.

The schedule node advances bias corrections exactly once after all gradients are published; every
parameter group then consumes the same corrections before the optional graph-tail node.
-/
def submitFullModelStepF32 (executor : @& Executor)
    (hyperparameters : AdamWHyperparametersF32) (schedule : @& AdamWScheduleF32)
    (descriptor : Descriptor) (step : @& FullModelStepF32) : IO Unit := do
  let descriptor ← step.check descriptor hyperparameters schedule
  let layout : Cuda.Qwen36.SequenceLayout := {
    batchSize := descriptor.batchSize.toUInt32
    sequenceLength := descriptor.sequenceLength.toUInt32
  }
  Model.submitBatchedForwardLossF32 executor step.weights step.forward step.tokenIds step.targets
    layout descriptor.model
  submitCheckedFullModelBackwardUpdateF32 executor hyperparameters schedule descriptor step

/-- Append the one-thread device-tail transition after a complete training step. -/
def appendTailTrainingGraph (builder : @& Cuda.GraphBuilder)
    (graphHandle : @& Cuda.Buffer UInt64) (remaining : @& Cuda.Buffer UInt32) : IO Unit :=
  tailTrainingGraph.addToGraph builder {
    grid := { x := 1 }
    block := { x := 1 }
    blockArenaBytes := 0
  } graphHandle remaining

/--
An instantiated complete-model training graph whose checked step count is driven entirely by its
device-tail node. The retained schedule makes repeated and resumed runs observable and verifiable.
-/
structure ResidentTrainerF32 where
  graph : Cuda.DeviceGraph
  graphHandle : Cuda.Buffer UInt64
  remaining : Cuda.Buffer UInt32
  schedule : AdamWScheduleF32
  steps : UInt32

namespace ResidentTrainerF32

/-- Build one complete `forward -> loss -> backward -> update -> device-tail` resident trainer. -/
def build (hyperparameters : AdamWHyperparametersF32) (schedule : AdamWScheduleF32)
    (descriptor : Descriptor) (step : @& FullModelStepF32) : IO ResidentTrainerF32 := do
  let descriptor ← step.check descriptor hyperparameters schedule
  let graphHandle ← Cuda.Buffer.alloc UInt64 1
  let remaining ← Cuda.Buffer.alloc UInt32 1
  let builder ← Cuda.GraphBuilder.create
  submitFullModelStepF32 (.graph builder) hyperparameters schedule descriptor step
  appendTailTrainingGraph builder graphHandle remaining
  let graph ← builder.instantiate
  return { graph, graphHandle, remaining, schedule, steps := descriptor.steps.toUInt32 }

/-- Run all checked recurrent steps with one host graph launch and verify device-tail completion. -/
def run (trainer : @& ResidentTrainerF32) (stream : @& Cuda.Stream) : IO Unit := do
  trainer.schedule.check
  let before := readUInt32 (← trainer.schedule.step.copyTo stream)
  if before.toUInt64 + trainer.steps.toUInt64 > 0xffffffff then
    throw <| IO.userError "Qwen3.6 resident AdamW step counter would overflow UInt32"
  trainer.graph.upload stream
  trainer.graphHandle.copyFrom (pushUInt64 ByteArray.empty (← trainer.graph.deviceHandle)) stream
  trainer.remaining.copyFrom (pushUInt32 ByteArray.empty trainer.steps) stream
  (← trainer.graph.launchOn stream).waitChecked "Qwen3.6 resident full-model training graph"
  unless readUInt32 (← trainer.remaining.copyTo stream) == 0 do
    throw <| IO.userError "Qwen3.6 resident training graph did not consume every recurrent step"
  let after := readUInt32 (← trainer.schedule.step.copyTo stream)
  unless after == before + trainer.steps do
    throw <| IO.userError <|
      s!"Qwen3.6 resident training schedule advanced from {before} to {after}, expected " ++
      s!"{before + trainer.steps}"

end ResidentTrainerF32

end Cuda.Qwen36.Training
