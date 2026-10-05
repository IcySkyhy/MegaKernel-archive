/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Megakernel
public import LeanCudaQwen.Qwen36.Training
public import LeanCudaQwen.Qwen36.FullPretrainMegakernel
public import LeanCudaQwen.Qwen36.FullGRPOMegakernel

public section

/-!
# Real-checkpoint Qwen3.6 LoRA training

Bridges the immutable BF16 persistent checkpoint to the existing Float32 adapter and AdamW
abstractions. The first production slice trains the language-model head while retaining the final
normalized hidden states on device. Base checkpoint shards are never exposed to an update kernel.
-/

namespace Cuda.Qwen36.CheckpointLoRA

structure Config where
  rank : Nat := 16
  alpha : Float32 := 16
  learningRate : Float32 := 0.001
  enableDPO : Bool := false
  dpoBeta : Float32 := 0.1
  enableGRPO : Bool := false
  grpoGroupSize : Nat := 2
  grpoClipEpsilon : Float32 := 0.2
  grpoKLBeta : Float32 := 0.04
  grpoAdvantageEpsilon : Float32 := 1e-6
  beta1 : Float32 := 0.9
  beta2 : Float32 := 0.999
  epsilon : Float32 := 1e-8
  seed : UInt64 := 0x3636
  initScale : Float32 := 0.01
  weightDecay : Float32 := 0
  deriving Repr

/-- Fixed-shape mixed-adapter batch routed by one adapter identifier per sequence. -/
structure Descriptor where
  batchSize : Nat
  sequenceLength : Nat
  adapterCount : Nat
  adapterIds : Array UInt32
  updateMask : Array UInt32
  deriving Repr

namespace Descriptor

/-- Flattened token-row count consumed by the LM-head adapter. -/
def rows (descriptor : Descriptor) : Nat :=
  descriptor.batchSize * descriptor.sequenceLength

/-- Validate the sequence boundary, adapter routing, and optimizer mask before loading weights. -/
def check (descriptor : Descriptor) : Except String Descriptor := do
  if descriptor.batchSize == 0 then
    throw "Qwen3.6 checkpoint LoRA batch size must be positive"
  if descriptor.sequenceLength == 0 ||
      descriptor.sequenceLength > Megakernel.maxTrainingSequenceTokens then
    throw <|
      s!"Qwen3.6 checkpoint LoRA sequence length must lie in " ++
      s!"[1, {Megakernel.maxTrainingSequenceTokens}]"
  if descriptor.adapterCount == 0 then
    throw "Qwen3.6 checkpoint LoRA adapter count must be positive"
  if descriptor.adapterIds.size != descriptor.batchSize then
    throw "Qwen3.6 checkpoint LoRA requires one adapter ID per sequence"
  for adapterId in descriptor.adapterIds do
    if adapterId != LoRA.noAdapter && adapterId.toNat ≥ descriptor.adapterCount then
      throw s!"Qwen3.6 checkpoint LoRA adapter ID {adapterId} is out of range"
  if descriptor.updateMask.size != descriptor.adapterCount then
    throw "Qwen3.6 checkpoint LoRA requires one update-mask entry per adapter"
  if !descriptor.updateMask.all fun value => value == 0 || value == 1 then
    throw "Qwen3.6 checkpoint LoRA update mask must contain only zero or one"
  return descriptor

end Descriptor

structure Snapshot where
  adapter : Projection.AdapterSnapshotF32
  schedule : Training.AdamWScheduleSnapshotF32
  deriving BEq

/-- Host-visible scale summary for one trainable parameter tensor. -/
structure GradientStats where
  elements : Nat
  finiteElements : Nat
  nonzeroElements : Nat
  maxAbs : Float32
  l2Norm : Float32
  rms : Float32
  deriving Repr

/-- Per-adapter diagnostics for both LoRA parameter matrices. -/
structure AdapterGradientStats where
  adapterId : UInt32
  adapterA : GradientStats
  adapterB : GradientStats
  deriving Repr

/-- Loss and pre-update gradients returned by an observable optimizer step. -/
structure StepResult where
  loss : Float32
  gradients : Array AdapterGradientStats
  deriving Repr

/-- Projection identity attached to the compact diagnostics for one routed adapter. -/
structure NamedAdapterGradientStats where
  projection : String
  adapterId : UInt32
  adapterA : GradientStats
  adapterB : GradientStats
  deriving Repr

/-- Loss and bounded-size diagnostics returned by a projection-wide optimizer step. -/
structure ProjectionStepResult where
  loss : Float32
  gradients : Array NamedAdapterGradientStats
  deriving Repr

/-- Pre-update DPO objective and implicit-reward summary for one chosen/rejected batch. -/
structure DPOResult where
  loss : Float32
  rewardAccuracy : Float32
  meanChosenReward : Float32
  meanRejectedReward : Float32
  meanRewardMargin : Float32
  deriving Repr

/-- DPO metrics plus bounded projection-wide gradient diagnostics. -/
structure DPOProjectionStepResult where
  loss : Float32
  rewardAccuracy : Float32
  meanChosenReward : Float32
  meanRejectedReward : Float32
  meanRewardMargin : Float32
  gradients : Array NamedAdapterGradientStats
  deriving Repr

/-- Pre-update clipped outcome-GRPO metrics for one grouped rollout batch. -/
structure GRPOResult where
  loss : Float32
  meanReward : Float32
  rewardStd : Float32
  meanKL : Float32
  clipFraction : Float32
  meanAbsoluteAdvantage : Float32
  deriving Repr

/-- GRPO metrics plus bounded projection-wide gradient diagnostics. -/
structure GRPOProjectionStepResult where
  loss : Float32
  meanReward : Float32
  rewardStd : Float32
  meanKL : Float32
  clipFraction : Float32
  meanAbsoluteAdvantage : Float32
  gradients : Array NamedAdapterGradientStats
  deriving Repr


structure Trainer where
  checkpoint : Megakernel.Checkpoint
  trainable : Megakernel.TrainableCheckpoint
  adapter : Projection.AdapterF32
  schedule : Training.AdamWScheduleF32
  hyperparameters : Training.AdamWHyperparametersF32
  stream : Cuda.Stream
  batchSize : UInt32
  sequenceLength : UInt32
  rows : UInt32
  input : Cuda.Buffer Float32
  targets : Cuda.Buffer UInt32
  gradient : Cuda.Buffer Float32
  rowLoss : Cuda.Buffer Float32
  loss : Cuda.Buffer Float32

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def pushUInt64 (bytes : ByteArray) (value : UInt64) : ByteArray := Id.run do
  let mut result := bytes
  for shift in [:8] do
    result := result.push (value >>> (8 * shift).toUInt64).toUInt8
  return result

private def readPackedUInt32At (bytes : ByteArray) (offset : Nat) : UInt32 :=
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def readPackedUInt64At (bytes : ByteArray) (offset : Nat) : UInt64 := Id.run do
  let mut result : UInt64 := 0
  for shift in [:8] do
    result := result ||| bytes[offset + shift]!.toUInt64 <<< (8 * shift).toUInt64
  return result

private def fnvOffset : UInt64 := 14695981039346656037
private def fnvPrime : UInt64 := 1099511628211

private def updateFingerprint (initial : UInt64) (bytes : ByteArray) : UInt64 :=
  let combined := (initial ^^^ bytes.hash ^^^ bytes.size.toUInt64) * fnvPrime
  (combined ^^^ (combined >>> 32)) * fnvPrime

private def packUInt32 (values : Array UInt32) : ByteArray :=
  values.foldl (init := ByteArray.emptyWithCapacity (values.size * 4)) pushUInt32

private def packFloat32 (values : Array Float32) : ByteArray :=
  packUInt32 (values.map Float32.toBits)

private def readFloat32 (bytes : ByteArray) : Float32 :=
  Float32.ofBits <| bytes[0]!.toUInt32 ||| (bytes[1]!.toUInt32 <<< 8) |||
    (bytes[2]!.toUInt32 <<< 16) ||| (bytes[3]!.toUInt32 <<< 24)

private def readFloat32At (bytes : ByteArray) (offset : Nat) : Float32 :=
  Float32.ofBits <| bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def finiteFloat32 (value : Float32) : Bool :=
  value.toBits &&& 0x7f800000 != 0x7f800000

private def gradientStats (bytes : ByteArray) (start elements : Nat) : GradientStats := Id.run do
  let mut finiteElements := 0
  let mut nonzeroElements := 0
  let mut maxAbs : Float32 := 0
  let mut sumSquares : Float32 := 0
  for index in [:elements] do
    let value := readFloat32At bytes (start + index * 4)
    let magnitude := Float32.ofBits (value.toBits &&& 0x7fffffff)
    if finiteFloat32 value then
      finiteElements := finiteElements + 1
      if magnitude != 0 then
        nonzeroElements := nonzeroElements + 1
      if magnitude > maxAbs then
        maxAbs := magnitude
      sumSquares := sumSquares + value * value
  return {
    elements
    finiteElements
    nonzeroElements
    maxAbs
    l2Norm := Float32.sqrt sumSquares
    rms := if elements == 0 then 0 else Float32.sqrt (sumSquares / elements.toUInt32.toFloat32)
  }

private def snapshotMagic : ByteArray := "LCQLORA1".toUTF8
private def snapshotVersion : UInt32 := 1
private def snapshotFieldCount : UInt32 := 13

private def readUInt32Checked (bytes : ByteArray) (offset : Nat) : Except String UInt32 := do
  if offset + 4 > bytes.size then
    throw "truncated Qwen3.6 LoRA checkpoint"
  return bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def appendSnapshotField (bytes field : ByteArray) : ByteArray :=
  pushUInt32 bytes field.size.toUInt32 ++ field

/-- Encode all adapter parameters, routing, optimizer moments, mask, and schedule state. -/
def Snapshot.toBytes (snapshot : Snapshot) : ByteArray :=
  let fields := #[
    snapshot.adapter.adapterA,
    snapshot.adapter.adapterB,
    snapshot.adapter.adapterIds,
    snapshot.adapter.adapterAFirstMoment,
    snapshot.adapter.adapterASecondMoment,
    snapshot.adapter.adapterBFirstMoment,
    snapshot.adapter.adapterBSecondMoment,
    snapshot.adapter.updateMask,
    snapshot.schedule.step,
    snapshot.schedule.beta1Power,
    snapshot.schedule.beta2Power,
    snapshot.schedule.inverseBiasCorrection1,
    snapshot.schedule.inverseBiasCorrection2
  ]
  fields.foldl (init :=
    pushUInt32 (pushUInt32 snapshotMagic snapshotVersion) snapshotFieldCount)
    appendSnapshotField

private def decodeSnapshotFields (bytes : ByteArray) : Except String (Array ByteArray) := do
  if bytes.size < 16 || bytes.extract 0 snapshotMagic.size != snapshotMagic then
    throw "invalid Qwen3.6 LoRA checkpoint magic"
  let version ← readUInt32Checked bytes snapshotMagic.size
  unless version == snapshotVersion do
    throw s!"unsupported Qwen3.6 LoRA checkpoint version {version}"
  let count ← readUInt32Checked bytes (snapshotMagic.size + 4)
  unless count == snapshotFieldCount do
    throw s!"Qwen3.6 LoRA checkpoint has {count} fields, expected {snapshotFieldCount}"
  let mut cursor := snapshotMagic.size + 8
  let mut fields := #[]
  for _ in [:count.toNat] do
    let length ← readUInt32Checked bytes cursor
    cursor := cursor + 4
    if length.toNat > bytes.size - cursor then
      throw "truncated Qwen3.6 LoRA checkpoint field"
    fields := fields.push (bytes.extract cursor (cursor + length.toNat))
    cursor := cursor + length.toNat
  unless cursor == bytes.size do
    throw "Qwen3.6 LoRA checkpoint has trailing bytes"
  return fields

/-- Decode a versioned on-disk adapter and AdamW snapshot. -/
def Snapshot.ofBytes (bytes : ByteArray) : Except String Snapshot := do
  let fields ← decodeSnapshotFields bytes
  return {
    adapter := {
      adapterA := fields[0]!
      adapterB := fields[1]!
      adapterIds := fields[2]!
      adapterAFirstMoment := fields[3]!
      adapterASecondMoment := fields[4]!
      adapterBFirstMoment := fields[5]!
      adapterBSecondMoment := fields[6]!
      updateMask := fields[7]!
    }
    schedule := {
      step := fields[8]!
      beta1Power := fields[9]!
      beta2Power := fields[10]!
      inverseBiasCorrection1 := fields[11]!
      inverseBiasCorrection2 := fields[12]!
    }
  }

private def Config.hyperparameters (config : Config) : Training.AdamWHyperparametersF32 := {
  learningRate := config.learningRate
  beta1 := config.beta1
  beta2 := config.beta2
  epsilon := config.epsilon
}

private def Config.check (config : Config) : IO Unit := do
  unless config.rank > 0 && config.rank % 16 == 0 do
    throw <| IO.userError "Qwen3.6 checkpoint LoRA rank must be a positive multiple of 16"
  unless config.alpha > 0 && finiteFloat32 config.alpha do
    throw <| IO.userError "Qwen3.6 checkpoint LoRA alpha must be positive and finite"
  if config.enableDPO && config.enableGRPO then
    throw <| IO.userError "Qwen3.6 checkpoint LoRA cannot enable DPO and GRPO together"
  unless config.dpoBeta > 0 && (config.dpoBeta.toBits &&& 0x7f800000) != 0x7f800000 do
    throw <| IO.userError "Qwen3.6 checkpoint LoRA DPO beta must be positive and finite"
  unless config.grpoGroupSize >= 2 do
    throw <| IO.userError "Qwen3.6 checkpoint LoRA GRPO group size must be at least two"
  unless config.grpoClipEpsilon > 0 && config.grpoClipEpsilon < 1 &&
      finiteFloat32 config.grpoClipEpsilon do
    throw <| IO.userError
      "Qwen3.6 checkpoint LoRA GRPO clip epsilon must be finite and lie in (0, 1)"
  unless config.grpoKLBeta >= 0 && finiteFloat32 config.grpoKLBeta do
    throw <| IO.userError
      "Qwen3.6 checkpoint LoRA GRPO KL beta must be nonnegative and finite"
  unless config.grpoAdvantageEpsilon > 0 && finiteFloat32 config.grpoAdvantageEpsilon do
    throw <| IO.userError
      "Qwen3.6 checkpoint LoRA GRPO advantage epsilon must be positive and finite"
  unless config.initScale > 0 && finiteFloat32 config.initScale do
    throw <| IO.userError "Qwen3.6 checkpoint LoRA initialization scale must be positive and finite"
  unless config.weightDecay ≥ 0 && finiteFloat32 config.weightDecay do
    throw <| IO.userError "Qwen3.6 checkpoint LoRA weight decay must be nonnegative and finite"
  config.hyperparameters.check

/-- Load the immutable checkpoint once and allocate a routed LM-head adapter bank plus state. -/
def load (modelDirectory : System.FilePath) (descriptor : Descriptor) (config : Config := {}) :
    IO Trainer := do
  let descriptor ← match descriptor.check with
    | .ok descriptor => pure descriptor
    | .error message => throw <| IO.userError message
  config.check
  let trainable ← Megakernel.loadTrainableCheckpoint modelDirectory
  let stream := trainable.stream
  let adapterDescriptor : LoRA.Descriptor := {
    batchSize := descriptor.batchSize
    sequenceLength := descriptor.sequenceLength
    adapterCount := descriptor.adapterCount
    rank := config.rank
    inputFeatures := trainable.inputFeatures.toNat
    outputFeatures := trainable.outputFeatures.toNat
    alpha := config.alpha
    adapterIds := descriptor.adapterIds
  }
  let adapter ← Projection.AdapterF32.allocate adapterDescriptor descriptor.updateMask config.seed
    config.initScale config.weightDecay stream
  let schedule ← Training.AdamWScheduleF32.allocate stream
  let rows := descriptor.rows
  let input ← Cuda.Buffer.alloc Float32
    (rows * trainable.inputFeatures.toNat).toUSize
  let targets ← Cuda.Buffer.alloc UInt32 rows.toUSize
  let gradient ← Cuda.Buffer.alloc Float32
    (rows * trainable.outputFeatures.toNat).toUSize
  let rowLoss ← Cuda.Buffer.alloc Float32 rows.toUSize
  let loss ← Cuda.Buffer.alloc Float32 1
  return {
    checkpoint := trainable.checkpoint
    trainable
    adapter
    schedule
    hyperparameters := config.hyperparameters
    stream
    batchSize := descriptor.batchSize.toUInt32
    sequenceLength := descriptor.sequenceLength.toUInt32
    rows := rows.toUInt32
    input, targets, gradient, rowLoss, loss
  }

namespace Trainer

private def checkBatch (trainer : @& Trainer) (tokens targets : Array (Array UInt32)) :
    IO Unit := do
  unless tokens.size == trainer.batchSize.toNat && targets.size == trainer.batchSize.toNat do
    throw <| IO.userError
      s!"Qwen3.6 checkpoint LoRA expected {trainer.batchSize} token and target sequences"
  for batch in [:tokens.size] do
    unless tokens[batch]!.size == trainer.sequenceLength.toNat &&
        targets[batch]!.size == trainer.sequenceLength.toNat do
      throw <| IO.userError <|
        s!"Qwen3.6 checkpoint LoRA expected sequence length {trainer.sequenceLength}"

private def forwardLoss (trainer : @& Trainer) (tokens targets : Array (Array UInt32)) :
    IO Float32 := do
  trainer.checkBatch tokens targets
  let batch ← trainer.trainable.forwardBatch tokens targets
  unless batch.rows == trainer.rows && batch.inputFeatures == trainer.adapter.shape.inputFeatures &&
      batch.outputFeatures == trainer.adapter.shape.outputFeatures do
    throw <| IO.userError "Qwen3.6 checkpoint LoRA forward geometry changed"
  (← Primitives.castBF16ToF32 trainer.stream batch.finalHidden trainer.input
    (batch.rows * batch.inputFeatures)).waitChecked
    "Qwen3.6 checkpoint LoRA final-hidden cast"
  LoRA.addForwardF32 trainer.stream trainer.input trainer.adapter.adapterA trainer.adapter.adapterB
    trainer.adapter.adapterIds trainer.adapter.rankActivation batch.logits trainer.adapter.shape
  let mut flatTargets := #[]
  for sequence in targets do
    for target in sequence do
      flatTargets := flatTargets.push target
  trainer.targets.copyFrom (packUInt32 flatTargets) trainer.stream
  (← Model.crossEntropyF32 trainer.stream batch.logits trainer.targets trainer.gradient
    trainer.rowLoss batch.rows batch.outputFeatures).waitChecked
    "Qwen3.6 checkpoint LoRA cross-entropy"
  (← Model.reduceLossF32 trainer.stream trainer.rowLoss trainer.loss batch.rows).waitChecked
    "Qwen3.6 checkpoint LoRA loss reduction"
  return readFloat32 (← trainer.loss.copyTo trainer.stream)

/-- Evaluate mean cross-entropy with the current adapter without updating any state. -/
def evaluate (trainer : @& Trainer) (tokens targets : Array (Array UInt32)) : IO Float32 :=
  trainer.forwardLoss tokens targets

/-- Copy pre-update LoRA gradients to host and summarize every adapter-local parameter tensor. -/
def gradientDiagnostics (trainer : @& Trainer) : IO (Array AdapterGradientStats) := do
  let aBytes ← trainer.adapter.adapterAGradient.copyTo trainer.stream
  let bBytes ← trainer.adapter.adapterBGradient.copyTo trainer.stream
  let aElements := trainer.adapter.shape.rank.toNat * trainer.adapter.shape.inputFeatures.toNat
  let bElements := trainer.adapter.shape.outputFeatures.toNat * trainer.adapter.shape.rank.toNat
  let mut diagnostics := #[]
  for adapter in [:trainer.adapter.shape.adapterCount.toNat] do
    diagnostics := diagnostics.push {
      adapterId := adapter.toUInt32
      adapterA := gradientStats aBytes (adapter * aElements * 4) aElements
      adapterB := gradientStats bBytes (adapter * bElements * 4) bElements
    }
  return diagnostics

/--
Run one real BF16-base forward and adapter VJP, expose finite scale diagnostics before mutation,
then apply one bias-corrected AdamW update.
-/
def stepWithDiagnostics (trainer : @& Trainer) (tokens targets : Array (Array UInt32)) :
    IO StepResult := do
  let loss ← trainer.forwardLoss tokens targets
  LoRA.backwardAdaptersF32 trainer.stream trainer.input trainer.adapter.adapterB
    trainer.adapter.adapterIds trainer.adapter.rankActivation trainer.gradient
    trainer.adapter.adapterAGradient trainer.adapter.adapterBGradient trainer.adapter.shape
  let gradients ← trainer.gradientDiagnostics
  Training.submitAdvanceAdamWScheduleF32 (.sequential trainer.stream) trainer.hyperparameters
    trainer.schedule
  Training.submitResidentLoRAAdamWF32 (.sequential trainer.stream)
    "Qwen3.6 checkpoint LM-head LoRA AdamW" trainer.hyperparameters trainer.schedule
    trainer.adapter
  return { loss, gradients }

/-- Run one observable optimizer step and return its pre-update mean cross-entropy. -/
def step (trainer : @& Trainer) (tokens targets : Array (Array UInt32)) : IO Float32 :=
  return (← trainer.stepWithDiagnostics tokens targets).loss

/-- Snapshot adapter parameters, moments, routing, mask, and optimizer schedule for exact resume. -/
def snapshot (trainer : @& Trainer) : IO Snapshot := do
  return {
    adapter := ← trainer.adapter.snapshot trainer.stream
    schedule := ← trainer.schedule.snapshot trainer.stream
  }

/-- Restore an in-memory snapshot after allocation-size compatibility checks. -/
def restore (trainer : @& Trainer) (state : @& Snapshot) : IO Unit := do
  trainer.adapter.restore state.adapter trainer.stream
  trainer.schedule.restore state.schedule trainer.stream

/-- Persist an exact, versioned adapter and AdamW checkpoint to a closed binary file. -/
def saveCheckpoint (trainer : @& Trainer) (path : System.FilePath) : IO Unit := do
  IO.FS.writeBinFile path (← trainer.snapshot).toBytes

/-- Read and validate a versioned adapter and AdamW checkpoint from disk. -/
def loadSnapshot (path : System.FilePath) : IO Snapshot := do
  let bytes ← IO.FS.readBinFile path
  match Snapshot.ofBytes bytes with
  | .ok snapshot => return snapshot
  | .error message => throw <| IO.userError message

/--
Resume parameters, routing, mask, moments, and the bias-correction schedule. The destination
trainer's checked allocation geometry is the compatibility contract.
-/
def resumeCheckpoint (trainer : @& Trainer) (path : System.FilePath) : IO Unit := do

  trainer.restore (← loadSnapshot path)
end Trainer

/-! ## Projection-wide real-checkpoint trainer -/

/-- One canonical projection name paired with its trainable adapter state. -/
structure NamedAdapterF32 where
  name : String
  adapter : Projection.AdapterF32

structure ProjectionGradientSlice where
  projection : String
  adapterId : UInt32
  isAdapterA : Bool
  elements : Nat
  deriving Inhabited

@[struct]
private structure DeviceGradientAccumulator where
  finite : Float32
  nonzero : Float32
  maxAbs : Float32
  sumSquares : Float32

@[always_inline]
private partial def accumulateGradientSlice (gradient : Cuda.DevicePtr Float32)
    (elements index stride : UInt32) (accumulator : DeviceGradientAccumulator) :
    Cuda.DeviceM DeviceGradientAccumulator := do
  if index < elements then
    let value ← Cuda.loadFloat32 gradient index.toUSize
    let bits := value.toBits
    let finite := bits &&& 0x7f800000 != 0x7f800000
    let magnitude := Float32.ofBits (bits &&& 0x7fffffff)
    accumulateGradientSlice gradient elements (index + stride) stride {
      finite := accumulator.finite + if finite then 1 else 0
      nonzero := accumulator.nonzero + if finite && magnitude != 0 then 1 else 0
      maxAbs := if finite then max accumulator.maxAbs magnitude else accumulator.maxAbs
      sumSquares := accumulator.sumSquares + if finite then value * value else 0
    }
  else
    return accumulator

/-- Copy one policy/reference adapter route to every projection-local routing buffer. -/
@[cuda_kernel]
def projectionAdapterRoutingKernel (destinations : Cuda.DevicePtrTable UInt32)
    (source : Cuda.DevicePtr UInt32) (projections batchSize : UInt32) : Cuda.DeviceM Unit := do
  let linear ← Primitives.elementIndex
  if linear < projections * batchSize then
    let projection := linear / batchSize
    let sequence := linear % batchSize
    let destination ← destinations.get projection.toUSize
    Cuda.storeUInt32 destination sequence.toUSize (← Cuda.loadUInt32 source sequence.toUSize)

/-- Device buffers and reusable graphs for reference-scored direct preference optimization. -/
structure DPOState where
  masks : Cuda.Buffer UInt32
  referenceSequenceLogProbability : Cuda.Buffer Float32
  policySequenceLogProbability : Cuda.Buffer Float32
  sequenceCoefficients : Cuda.Buffer Float32
  adapterIdTable : Cuda.BufferTable UInt32
  policyAdapterIds : Cuda.Buffer UInt32
  referenceAdapterIds : Cuda.Buffer UInt32
  beta : Float32
  referenceGraph : Cuda.DeviceGraph
  evaluationGraph : Cuda.DeviceGraph
  resident : Training.ResidentTrainerF32

/-- Device buffers and reusable graphs for clipped outcome-supervised GRPO. -/
structure GRPOState where
  masks : Cuda.Buffer UInt32
  rewards : Cuda.Buffer Float32
  advantages : Cuda.Buffer Float32
  oldTokenLogProbability : Cuda.Buffer Float32
  referenceTokenLogProbability : Cuda.Buffer Float32
  policyTokenLogProbability : Cuda.Buffer Float32
  rowCoefficients : Cuda.Buffer Float32
  statistics : Cuda.Buffer Float32
  adapterIdTable : Cuda.BufferTable UInt32
  policyAdapterIds : Cuda.Buffer UInt32
  referenceAdapterIds : Cuda.Buffer UInt32
  groupSize : UInt32
  clipEpsilon : Float32
  klBeta : Float32
  advantageEpsilon : Float32
  referenceGraph : Cuda.DeviceGraph
  evaluationGraph : Cuda.DeviceGraph
  resident : Training.ResidentTrainerF32

/--
Reduce every adapter-local A/B gradient slice to four Float32 scalars on device. One block owns
one slice, so diagnostics copy O(projections * adapters) values instead of multi-gigabyte tensors.
-/
@[cuda_kernel]
def projectionGradientStatsKernel (gradients : Cuda.DevicePtrTable Float32)
    (elements : Cuda.DevicePtr UInt32) (statistics : Cuda.DevicePtr Float32) :
    Cuda.DeviceM Unit := do
  let tensor ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let count ← Cuda.loadUInt32 elements tensor.toUSize
  let gradient ← gradients.get tensor.toUSize
  let accumulated ← accumulateGradientSlice gradient count thread 256 {
    finite := 0, nonzero := 0, maxAbs := 0, sumSquares := 0
  }
  let scratchRaw ← Cuda.dynamicShared (α := Float32) 8
  let scratch : Cuda.Collective.BlockScratch 8 := scratchRaw
  let finite ← Cuda.Collective.blockSum scratch accumulated.finite
  let nonzero ← Cuda.Collective.blockSum scratch accumulated.nonzero
  let maxAbs ← Cuda.Collective.blockMax scratch accumulated.maxAbs
  let sumSquares ← Cuda.Collective.blockSum scratch accumulated.sumSquares
  if thread == 0 then
    let base := tensor * 4
    Cuda.storeFloat32 statistics base.toUSize finite
    Cuda.storeFloat32 statistics (base + 1).toUSize nonzero
    Cuda.storeFloat32 statistics (base + 2).toUSize maxAbs
    Cuda.storeFloat32 statistics (base + 3).toUSize sumSquares
/-- Immutable facts that must match before a projection-wide checkpoint can be resumed. -/
structure ProjectionCheckpointCompatibility where
  baseModelFingerprint : UInt64
  batchSize : UInt32
  sequenceLength : UInt32
  adapterCount : UInt32
  rank : UInt32
  alphaBits : UInt32
  learningRateBits : UInt32
  beta1Bits : UInt32
  beta2Bits : UInt32
  epsilonBits : UInt32
  initScaleBits : UInt32
  weightDecayBits : UInt32
  objective : UInt32
  dpoBetaBits : UInt32
  grpoGroupSize : UInt32
  grpoClipEpsilonBits : UInt32
  grpoKLBetaBits : UInt32
  grpoAdvantageEpsilonBits : UInt32
  deriving BEq, Repr

private def compatibilityBytes : Nat := 8 + 17 * 4

private def ProjectionCheckpointCompatibility.toBytes
    (compatibility : ProjectionCheckpointCompatibility) : ByteArray :=
  pushUInt64 ByteArray.empty compatibility.baseModelFingerprint ++ packUInt32 #[
    compatibility.batchSize, compatibility.sequenceLength, compatibility.adapterCount,
    compatibility.rank, compatibility.alphaBits, compatibility.learningRateBits,
    compatibility.beta1Bits, compatibility.beta2Bits, compatibility.epsilonBits,
    compatibility.initScaleBits, compatibility.weightDecayBits, compatibility.objective,
    compatibility.dpoBetaBits, compatibility.grpoGroupSize, compatibility.grpoClipEpsilonBits,
    compatibility.grpoKLBetaBits, compatibility.grpoAdvantageEpsilonBits
  ]

private def ProjectionCheckpointCompatibility.ofBytes (bytes : ByteArray) :
    Except String ProjectionCheckpointCompatibility := do
  unless bytes.size == compatibilityBytes do
    throw s!"checkpoint compatibility record has {bytes.size} bytes, expected {compatibilityBytes}"
  let word (index : Nat) := readPackedUInt32At bytes (8 + index * 4)
  return {
    baseModelFingerprint := readPackedUInt64At bytes 0
    batchSize := word 0
    sequenceLength := word 1
    adapterCount := word 2
    rank := word 3
    alphaBits := word 4
    learningRateBits := word 5
    beta1Bits := word 6
    beta2Bits := word 7
    epsilonBits := word 8
    initScaleBits := word 9
    weightDecayBits := word 10
    objective := word 11
    dpoBetaBits := word 12
    grpoGroupSize := word 13
    grpoClipEpsilonBits := word 14
    grpoKLBetaBits := word 15
    grpoAdvantageEpsilonBits := word 16
  }

private def baseModelFingerprint (modelDirectory : System.FilePath) : IO UInt64 := do
  let mut result := fnvOffset
  for filename in #["config.json", "model.safetensors.index.json"] do
    let path := modelDirectory / filename
    unless (← path.pathExists) do
      throw <| IO.userError s!"cannot fingerprint Qwen checkpoint; missing {path}"
    result := updateFingerprint result filename.toUTF8
    result := updateFingerprint result (← IO.FS.readBinFile path)
  return result

private def projectionCheckpointCompatibility (modelDirectory : System.FilePath)
    (descriptor : Descriptor) (config : Config) : IO ProjectionCheckpointCompatibility := do
  return {
    baseModelFingerprint := ← baseModelFingerprint modelDirectory
    batchSize := descriptor.batchSize.toUInt32
    sequenceLength := descriptor.sequenceLength.toUInt32
    adapterCount := descriptor.adapterCount.toUInt32
    rank := config.rank.toUInt32
    alphaBits := config.alpha.toBits
    learningRateBits := config.learningRate.toBits
    beta1Bits := config.beta1.toBits
    beta2Bits := config.beta2.toBits
    epsilonBits := config.epsilon.toBits
    initScaleBits := config.initScale.toBits
    weightDecayBits := config.weightDecay.toBits
    objective := if config.enableDPO then 1 else if config.enableGRPO then 2 else 0
    dpoBetaBits := config.dpoBeta.toBits
    grpoGroupSize := config.grpoGroupSize.toUInt32
    grpoClipEpsilonBits := config.grpoClipEpsilon.toBits
    grpoKLBetaBits := config.grpoKLBeta.toBits
    grpoAdvantageEpsilonBits := config.grpoAdvantageEpsilon.toBits
  }


/-- Complete frozen-checkpoint reverse graph and reusable one-step CUDA graph. -/
structure ProjectionTrainer where
  checkpoint : Megakernel.Checkpoint
  trainable : Megakernel.TrainableCheckpoint
  descriptor : Training.Descriptor
  stepState : Training.FullModelStepF32
  schedule : Training.AdamWScheduleF32
  resident : Training.ResidentTrainerF32
  evaluationGraph : Cuda.DeviceGraph
  dpo : Option DPOState
  grpo : Option GRPOState
  hyperparameters : Training.AdamWHyperparametersF32
  compatibility : ProjectionCheckpointCompatibility
  adapters : Array NamedAdapterF32
  gradientSlices : Array ProjectionGradientSlice
  gradientTable : Cuda.BufferTable Float32
  gradientElements : Cuda.Buffer UInt32
  gradientStatistics : Cuda.Buffer Float32
  stream : Cuda.Stream
  batchSize : UInt32
  sequenceLength : UInt32

private def allocateF32 (elements : UInt32) : IO (Cuda.Buffer Float32) :=
  Cuda.Buffer.alloc Float32 elements.toUSize

private def allocateZeroF32 (elements : UInt32) (stream : @& Cuda.Stream) :
    IO (Cuda.Buffer Float32) := do
  let buffer ← allocateF32 elements
  (← Primitives.zeroBufferF32 stream buffer elements).waitChecked
    "Qwen3.8 projection-wide zero initialization"
  return buffer

private def allocateMLPForward (tokens hidden intermediate : UInt32) :
    IO Model.MLPForwardF32 := do
  let normalized ← allocateF32 (tokens * hidden)
  let inverseRms ← allocateF32 tokens
  let gate ← allocateF32 (tokens * intermediate)
  let up ← allocateF32 (tokens * intermediate)
  let activation ← allocateF32 (tokens * intermediate)
  let output ← allocateF32 (tokens * hidden)
  return { normalized, inverseRms, gate, up, activation, output }

private def allocateMLPBackward (tokens hidden intermediate : UInt32) :
    IO Model.MLPBackwardF32 := do
  let activationGradient ← allocateF32 (tokens * intermediate)
  let gateGradient ← allocateF32 (tokens * intermediate)
  let upGradient ← allocateF32 (tokens * intermediate)
  let normGradientGate ← allocateF32 (tokens * hidden)
  let normGradientUp ← allocateF32 (tokens * hidden)
  let normGradient ← allocateF32 (tokens * hidden)
  let inputGradient ← allocateF32 (tokens * hidden)
  let normWeightGradient ← allocateF32 hidden
  let dummy ← allocateF32 1
  return {
    activationGradient, gateGradient, upGradient, normGradientGate, normGradientUp,
    normGradient, inputGradient, normWeightGradient,
    gateWeightGradient := dummy, upWeightGradient := dummy, downWeightGradient := dummy
  }

private def allocateDeltaForward (batchSize sequenceLength hidden keyHeads valueHeads keyWidth
    valueWidth : UInt32) (stream : @& Cuda.Stream) : IO DeltaNet.StageForwardF32 := do
  let tokens := batchSize * sequenceLength
  let convolutionWidth := 2 * keyHeads * keyWidth + valueHeads * valueWidth
  let compactElements := tokens * keyHeads * keyWidth
  let valueElements := tokens * valueHeads * valueWidth
  let stateElements := batchSize * valueHeads * keyWidth * valueWidth
  let input ← allocateF32 (tokens * hidden)
  let normalizedInput ← allocateF32 (tokens * hidden)
  let inputInverseRms ← allocateF32 tokens
  let queryKeyValuePreConv ← allocateF32 (tokens * convolutionWidth)
  let z ← allocateF32 valueElements
  let b ← allocateF32 (tokens * valueHeads)
  let a ← allocateF32 (tokens * valueHeads)
  let queryKeyValuePostConv ← allocateF32 (tokens * convolutionWidth)
  let queryCompact ← allocateF32 compactElements
  let keyCompact ← allocateF32 compactElements
  let value ← allocateF32 valueElements
  let queryRepeated ← allocateF32 valueElements
  let keyRepeated ← allocateF32 valueElements
  let queryNormed ← allocateF32 valueElements
  let keyNormed ← allocateF32 valueElements
  let decayLog ← allocateF32 (tokens * valueHeads)
  let beta ← allocateF32 (tokens * valueHeads)
  let stateInput ← allocateZeroF32 stateElements stream
  let stateOutput ← allocateF32 stateElements
  let stateHistory ← allocateF32 (batchSize * (sequenceLength + 1) * valueHeads * keyWidth * valueWidth)
  let deltaOutput ← allocateF32 valueElements
  let gatedInverseRms ← allocateF32 (tokens * valueHeads)
  let gatedOutput ← allocateF32 valueElements
  let output ← allocateF32 (tokens * hidden)
  return {
    input, normalizedInput, inputInverseRms, queryKeyValuePreConv, z, b, a,
    queryKeyValuePostConv, queryCompact, keyCompact, value, queryRepeated, keyRepeated,
    queryNormed, keyNormed, decayLog, beta, stateInput, stateOutput, stateHistory,
    deltaOutput, gatedInverseRms, gatedOutput, output
  }

private def allocateDeltaBackward (batchSize sequenceLength hidden keyHeads valueHeads keyWidth
    valueWidth : UInt32) : IO DeltaNet.StageBackwardF32 := do
  let tokens := batchSize * sequenceLength
  let convolutionWidth := 2 * keyHeads * keyWidth + valueHeads * valueWidth
  let compactElements := tokens * keyHeads * keyWidth
  let valueElements := tokens * valueHeads * valueWidth
  let stateElements := batchSize * valueHeads * keyWidth * valueWidth
  let outputGradient ← allocateF32 (tokens * hidden)
  let gatedOutputGradient ← allocateF32 valueElements
  let deltaOutputGradient ← allocateF32 valueElements
  let zGradient ← allocateF32 valueElements
  let gatedNormWeightGradient ← allocateF32 valueWidth
  let queryNormGradient ← allocateF32 valueElements
  let keyNormGradient ← allocateF32 valueElements
  let valueGradient ← allocateF32 valueElements
  let decayLogGradient ← allocateF32 (tokens * valueHeads)
  let betaGradient ← allocateF32 (tokens * valueHeads)
  let stateGradient ← allocateF32 stateElements
  let queryRepeatedGradient ← allocateF32 valueElements
  let keyRepeatedGradient ← allocateF32 valueElements
  let queryCompactGradient ← allocateF32 compactElements
  let keyCompactGradient ← allocateF32 compactElements
  let queryKeyValuePostConvGradient ← allocateF32 (tokens * convolutionWidth)
  let queryKeyValuePreConvGradient ← allocateF32 (tokens * convolutionWidth)
  let convolutionWeightGradient ← allocateF32 (convolutionWidth * 4)
  let aGradient ← allocateF32 (tokens * valueHeads)
  let bGradient ← allocateF32 (tokens * valueHeads)
  let aLogGradient ← allocateF32 valueHeads
  let dtBiasGradient ← allocateF32 valueHeads
  let queryKeyValueInputGradient ← allocateF32 (tokens * hidden)
  let zInputGradient ← allocateF32 (tokens * hidden)
  let bInputGradient ← allocateF32 (tokens * hidden)
  let aInputGradient ← allocateF32 (tokens * hidden)
  let queryKeyValueZInputGradient ← allocateF32 (tokens * hidden)
  let queryKeyValueZBInputGradient ← allocateF32 (tokens * hidden)
  let normalizedInputGradient ← allocateF32 (tokens * hidden)
  let inputGradient ← allocateF32 (tokens * hidden)
  let inputNormWeightGradient ← allocateF32 hidden
  let dummy ← allocateF32 1
  return {
    outputGradient, gatedOutputGradient, deltaOutputGradient, zGradient,
    gatedNormWeightGradient, queryNormGradient, keyNormGradient, valueGradient,
    decayLogGradient, betaGradient, stateGradient, queryRepeatedGradient,
    keyRepeatedGradient, queryCompactGradient, keyCompactGradient,
    queryKeyValuePostConvGradient, queryKeyValuePreConvGradient,
    convolutionWeightGradient, aGradient, bGradient, aLogGradient, dtBiasGradient,
    queryKeyValueInputGradient, zInputGradient, bInputGradient, aInputGradient,
    queryKeyValueZInputGradient, queryKeyValueZBInputGradient, normalizedInputGradient,
    inputGradient, inputNormWeightGradient,
    queryKeyValueWeightGradient := dummy, zWeightGradient := dummy,
    bWeightGradient := dummy, aWeightGradient := dummy, outputWeightGradient := dummy
  }

private def allocateAttentionForward (batchSize sequenceLength hidden queryHeads keyValueHeads
    width : UInt32) : IO Attention.StageForwardF32 := do
  let tokens := batchSize * sequenceLength
  let queryElements := tokens * queryHeads * width
  let keyValueElements := tokens * keyValueHeads * width
  let probabilityElements := batchSize * queryHeads * sequenceLength * sequenceLength
  let input ← allocateF32 (tokens * hidden)
  let normalizedInput ← allocateF32 (tokens * hidden)
  let inputInverseRms ← allocateF32 tokens
  let queryGateProjection ← allocateF32 (2 * queryElements)
  let queryPreNorm ← allocateF32 queryElements
  let gate ← allocateF32 queryElements
  let keyPreNorm ← allocateF32 keyValueElements
  let value ← allocateF32 keyValueElements
  let queryNormed ← allocateF32 queryElements
  let queryInverseRms ← allocateF32 (tokens * queryHeads)
  let keyNormed ← allocateF32 keyValueElements
  let keyInverseRms ← allocateF32 (tokens * keyValueHeads)
  let queryRope ← allocateF32 queryElements
  let keyRope ← allocateF32 keyValueElements
  let probabilities ← allocateF32 probabilityElements
  let preGate ← allocateF32 queryElements
  let postGate ← allocateF32 queryElements
  let output ← allocateF32 (tokens * hidden)
  return {
    input, normalizedInput, inputInverseRms, queryGateProjection, queryPreNorm, gate,
    keyPreNorm, value, queryNormed, queryInverseRms, keyNormed, keyInverseRms,
    queryRope, keyRope, probabilities, preGate, postGate, output
  }

private def allocateAttentionBackward (batchSize sequenceLength hidden queryHeads keyValueHeads
    width : UInt32) : IO Attention.StageBackwardF32 := do
  let tokens := batchSize * sequenceLength
  let queryElements := tokens * queryHeads * width
  let keyValueElements := tokens * keyValueHeads * width
  let probabilityElements := batchSize * queryHeads * sequenceLength * sequenceLength
  let outputGradient ← allocateF32 (tokens * hidden)
  let postGateGradient ← allocateF32 queryElements
  let preGateGradient ← allocateF32 queryElements
  let gateGradient ← allocateF32 queryElements
  let probabilityGradient ← allocateF32 probabilityElements
  let scoreGradient ← allocateF32 probabilityElements
  let queryRopeGradient ← allocateF32 queryElements
  let keyRopeGradient ← allocateF32 keyValueElements
  let valueGradient ← allocateF32 keyValueElements
  let queryNormGradient ← allocateF32 queryElements
  let keyNormGradient ← allocateF32 keyValueElements
  let queryPreNormGradient ← allocateF32 queryElements
  let keyPreNormGradient ← allocateF32 keyValueElements
  let queryNormWeightGradient ← allocateF32 width
  let keyNormWeightGradient ← allocateF32 width
  let queryGateProjectionGradient ← allocateF32 (2 * queryElements)
  let queryInputGradient ← allocateF32 (tokens * hidden)
  let keyInputGradient ← allocateF32 (tokens * hidden)
  let valueInputGradient ← allocateF32 (tokens * hidden)
  let queryKeyInputGradient ← allocateF32 (tokens * hidden)
  let normalizedInputGradient ← allocateF32 (tokens * hidden)
  let inputGradient ← allocateF32 (tokens * hidden)
  let inputNormWeightGradient ← allocateF32 hidden
  let dummy ← allocateF32 1
  return {
    outputGradient, postGateGradient, preGateGradient, gateGradient,
    probabilityGradient, scoreGradient, queryRopeGradient, keyRopeGradient,
    valueGradient, queryNormGradient, keyNormGradient, queryPreNormGradient,
    keyPreNormGradient, queryNormWeightGradient, keyNormWeightGradient,
    queryGateProjectionGradient, queryInputGradient, keyInputGradient,
    valueInputGradient, queryKeyInputGradient, normalizedInputGradient,
    inputGradient, inputNormWeightGradient,
    queryWeightGradient := dummy, keyWeightGradient := dummy,
    valueWeightGradient := dummy, outputWeightGradient := dummy
  }

private def allocateLayerForward (mixer : Model.MixerForwardF32)
    (tokens hidden intermediate : UInt32) : IO Model.LayerForwardF32 := do
  let mixerResidual ← allocateF32 (tokens * hidden)
  let mlp ← allocateMLPForward tokens hidden intermediate
  let output ← allocateF32 (tokens * hidden)
  return { mixer, mixerResidual, mlp, output }

private def allocateLayerBackward (mixer : Model.MixerBackwardF32)
    (tokens hidden intermediate : UInt32) : IO Model.LayerBackwardF32 := do
  let mlp ← allocateMLPBackward tokens hidden intermediate
  let mixerResidualGradient ← allocateF32 (tokens * hidden)
  let inputGradient ← allocateF32 (tokens * hidden)
  return { mlp, mixerResidualGradient, mixer, inputGradient }

private structure ProjectionBuild where
  weights : Projection.WeightsF32
  named : NamedAdapterF32

private def buildProjection (weight : Primitives.FrozenBFloat16Weight)
    (inputScratch outputGradientScratch : Cuda.Buffer Cuda.BFloat16)
    (descriptor : @& Descriptor) (config : @& Config) (stream : @& Cuda.Stream)
    (name : String) (inputFeatures outputFeatures : UInt32) (salt : UInt64) :
    IO ProjectionBuild := do
  let adapterDescriptor : LoRA.Descriptor := {
    batchSize := descriptor.batchSize
    sequenceLength := descriptor.sequenceLength
    adapterCount := descriptor.adapterCount
    rank := config.rank
    inputFeatures := inputFeatures.toNat
    outputFeatures := outputFeatures.toNat
    alpha := config.alpha
    adapterIds := descriptor.adapterIds
  }
  let adapter ← Projection.AdapterF32.allocate adapterDescriptor descriptor.updateMask
    (config.seed + salt) config.initScale config.weightDecay stream
  let base : Linear.FrozenBFloat16F32 := {
    weight, inputBF16 := inputScratch, outputGradientBF16 := outputGradientScratch
  }
  return { weights := .loraFrozenBF16 base adapter, named := { name, adapter } }

private structure LayerBuild where
  weights : Model.LayerWeightsF32
  forward : Model.LayerForwardF32
  backward : Model.LayerBackwardF32
  adapters : Array NamedAdapterF32

private def buildLayer (table : @& Megakernel.TrainingWeightTable)
    (architecture : @& Megakernel.TrainingArchitecture) (descriptor : @& Descriptor)
    (config : @& Config) (inputScratch outputGradientScratch : Cuda.Buffer Cuda.BFloat16)
    (inverseFrequency : Cuda.Buffer Float32) (stream : @& Cuda.Stream) (index : Nat) :
    IO LayerBuild := do
  let tokens := descriptor.rows.toUInt32
  let inputNorm ← Primitives.copyFrozenBF16ToF32 (← table.layer index 0)
    architecture.hidden stream
  let mlpNorm ← Primitives.copyFrozenBF16ToF32 (← table.layer index 1)
    architecture.hidden stream
  let gate ← buildProjection (← table.layer index 2) inputScratch outputGradientScratch
    descriptor config stream s!"layers.{index}.mlp.gate_proj" architecture.hidden
    architecture.intermediate (index * 16 + 2).toUInt64
  let up ← buildProjection (← table.layer index 3) inputScratch outputGradientScratch
    descriptor config stream s!"layers.{index}.mlp.up_proj" architecture.hidden
    architecture.intermediate (index * 16 + 3).toUInt64
  let down ← buildProjection (← table.layer index 4) inputScratch outputGradientScratch
    descriptor config stream s!"layers.{index}.mlp.down_proj" architecture.intermediate
    architecture.hidden (index * 16 + 4).toUInt64
  let mlpWeights : Model.MLPWeightsF32 := {
    norm := mlpNorm, gate := gate.weights, up := up.weights, down := down.weights
  }
  let mlpAdapters := #[gate.named, up.named, down.named]
  if architecture.layerUsesFullAttention index then
    let queryNorm ← Primitives.copyFrozenBF16ToF32 (← table.layer index 5)
      architecture.attentionHeadWidth stream
    let keyNorm ← Primitives.copyFrozenBF16ToF32 (← table.layer index 6)
      architecture.attentionHeadWidth stream
    let queryWidth := architecture.attentionQueryHeads * architecture.attentionHeadWidth
    let keyValueWidth := architecture.attentionKeyValueHeads * architecture.attentionHeadWidth
    let query ← buildProjection (← table.layer index 7) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.self_attn.q_proj" architecture.hidden
      (2 * queryWidth) (index * 16 + 7).toUInt64
    let key ← buildProjection (← table.layer index 8) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.self_attn.k_proj" architecture.hidden
      keyValueWidth (index * 16 + 8).toUInt64
    let value ← buildProjection (← table.layer index 9) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.self_attn.v_proj" architecture.hidden
      keyValueWidth (index * 16 + 9).toUInt64
    let output ← buildProjection (← table.layer index 10) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.self_attn.o_proj" queryWidth
      architecture.hidden (index * 16 + 10).toUInt64
    let mixerWeights : Attention.StageWeightsF32 := {
      inputNorm, query := query.weights, key := key.weights, value := value.weights,
      queryNorm, keyNorm, output := output.weights, inverseFrequency
    }
    let mixerForward ← allocateAttentionForward descriptor.batchSize.toUInt32
      descriptor.sequenceLength.toUInt32 architecture.hidden architecture.attentionQueryHeads
      architecture.attentionKeyValueHeads architecture.attentionHeadWidth
    let mixerBackward ← allocateAttentionBackward descriptor.batchSize.toUInt32
      descriptor.sequenceLength.toUInt32 architecture.hidden architecture.attentionQueryHeads
      architecture.attentionKeyValueHeads architecture.attentionHeadWidth
    let forward ← allocateLayerForward (.attention mixerForward) tokens architecture.hidden
      architecture.intermediate
    let backward ← allocateLayerBackward (.attention mixerBackward) tokens architecture.hidden
      architecture.intermediate
    return {
      weights := { mixer := .attention mixerWeights, mlp := mlpWeights }
      forward, backward
      adapters := #[query.named, key.named, value.named, output.named] ++ mlpAdapters
    }
  else
    let convolutionWidth := 2 * architecture.linearKeyHeads * architecture.linearKeyWidth +
      architecture.linearValueHeads * architecture.linearValueWidth
    let valueWidth := architecture.linearValueHeads * architecture.linearValueWidth
    let aLog ← Primitives.copyFrozenBF16ToF32 (← table.layer index 5)
      architecture.linearValueHeads stream
    let convolution ← Primitives.copyFrozenBF16ToF32 (← table.layer index 6)
      (convolutionWidth * 4) stream
    let dtBias ← Primitives.copyFrozenBF16ToF32 (← table.layer index 7)
      architecture.linearValueHeads stream
    let a ← buildProjection (← table.layer index 8) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.linear_attn.in_proj_a" architecture.hidden
      architecture.linearValueHeads (index * 16 + 8).toUInt64
    let b ← buildProjection (← table.layer index 9) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.linear_attn.in_proj_b" architecture.hidden
      architecture.linearValueHeads (index * 16 + 9).toUInt64
    let qkv ← buildProjection (← table.layer index 10) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.linear_attn.in_proj_qkv" architecture.hidden
      convolutionWidth (index * 16 + 10).toUInt64
    let z ← buildProjection (← table.layer index 11) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.linear_attn.in_proj_z" architecture.hidden
      valueWidth (index * 16 + 11).toUInt64
    let gatedNorm ← Primitives.copyFrozenBF16ToF32 (← table.layer index 12)
      architecture.linearValueWidth stream
    let output ← buildProjection (← table.layer index 13) inputScratch outputGradientScratch
      descriptor config stream s!"layers.{index}.linear_attn.out_proj" valueWidth
      architecture.hidden (index * 16 + 13).toUInt64
    let mixerWeights : DeltaNet.StageWeightsF32 := {
      inputNorm, queryKeyValue := qkv.weights, z := z.weights, b := b.weights, a := a.weights,
      convolution, aLog, dtBias, gatedNorm, output := output.weights
    }
    let mixerForward ← allocateDeltaForward descriptor.batchSize.toUInt32
      descriptor.sequenceLength.toUInt32 architecture.hidden architecture.linearKeyHeads
      architecture.linearValueHeads architecture.linearKeyWidth architecture.linearValueWidth stream
    let mixerBackward ← allocateDeltaBackward descriptor.batchSize.toUInt32
      descriptor.sequenceLength.toUInt32 architecture.hidden architecture.linearKeyHeads
      architecture.linearValueHeads architecture.linearKeyWidth architecture.linearValueWidth
    let forward ← allocateLayerForward (.deltaNet mixerForward) tokens architecture.hidden
      architecture.intermediate
    let backward ← allocateLayerBackward (.deltaNet mixerBackward) tokens architecture.hidden
      architecture.intermediate
    return {
      weights := { mixer := .deltaNet mixerWeights, mlp := mlpWeights }
      forward, backward
      adapters := #[qkv.named, z.named, b.named, a.named, output.named] ++ mlpAdapters
    }

/--
Load the real BF16 checkpoint and build LoRA adapters on every documented transformer projection.
The frozen matrices remain zero-copy; only small norms, convolution parameters, and saved training
activations use Float32 storage.
-/
def loadProjectionWide (modelDirectory : System.FilePath) (descriptor : Descriptor)
    (config : Config := {}) : IO ProjectionTrainer := do
  let descriptor ← match descriptor.check with
    | .ok descriptor => pure descriptor
    | .error message => throw <| IO.userError message
  config.check
  let compatibility ← projectionCheckpointCompatibility modelDirectory descriptor config
  let rows := descriptor.rows
  unless rows ≤ 128 && descriptor.batchSize * (descriptor.sequenceLength + 1) ≤ 132 do
    throw <| IO.userError
      "Qwen3.8 projection-wide trainer currently requires at most 128 token rows and 132 recurrent state rows"
  let trainable ← Megakernel.loadTrainableCheckpoint modelDirectory
  let stream := trainable.stream
  let architecture := Megakernel.trainingArchitecture
  let inputScratch ← Cuda.Buffer.alloc Cuda.BFloat16
    (rows * architecture.intermediate.toNat).toUSize
  let outputGradientScratch ← Cuda.Buffer.alloc Cuda.BFloat16
    (rows * architecture.vocabulary.toNat).toUSize
  let inverseFrequency ← allocateF32 architecture.rotaryHalf
  (← Primitives.ropeInvFreq stream inverseFrequency architecture.rotaryHalf
    (architecture.rotaryHalf * 2) 1e7).waitChecked "Qwen3.8 projection-wide inverse frequency"
  let mut layerWeights := #[]
  let mut layerForward := #[]
  let mut layerBackward := #[]
  let mut adapters := #[]
  for index in [:architecture.layers] do
    let layer ← buildLayer trainable.weights architecture descriptor config inputScratch
      outputGradientScratch inverseFrequency stream index
    layerWeights := layerWeights.push layer.weights
    layerForward := layerForward.push layer.forward
    layerBackward := layerBackward.push layer.backward
    adapters := adapters ++ layer.adapters
  let lmHead ← buildProjection (← trainable.weights.lmHead) inputScratch outputGradientScratch
    descriptor config stream "lm_head" architecture.hidden architecture.vocabulary 0x36360000
  adapters := adapters.push lmHead.named
  let finalNorm ← Primitives.copyFrozenBF16ToF32 (← trainable.weights.finalNorm)
    architecture.hidden stream
  let weights : Model.WeightsF32 := {
    embedding := .frozenBF16 (← trainable.weights.embedding)
    layers := layerWeights
    finalNorm
    lmHead := lmHead.weights
  }
  let tokenIds ← Cuda.Buffer.alloc UInt32 rows.toUSize
  let targets ← Cuda.Buffer.alloc UInt32 rows.toUSize
  let embedding ← allocateF32 (rows.toUInt32 * architecture.hidden)
  let finalHidden ← allocateF32 (rows.toUInt32 * architecture.hidden)
  let finalInverseRms ← allocateF32 rows.toUInt32
  let logits ← allocateF32 (rows.toUInt32 * architecture.vocabulary)
  let rowLoss ← allocateF32 rows.toUInt32
  let loss ← allocateF32 1
  let logitGradient ← allocateF32 (rows.toUInt32 * architecture.vocabulary)
  let forward : Model.ForwardF32 := {
    embedding, layers := layerForward, finalHidden, finalInverseRms,
    logits, rowLoss, loss, logitGradient
  }
  let finalHiddenGradient ← allocateF32 (rows.toUInt32 * architecture.hidden)
  let dummy ← allocateF32 1
  let decoderOutputGradient ← allocateF32 (rows.toUInt32 * architecture.hidden)
  let finalNormWeightGradient ← allocateF32 architecture.hidden
  let backward : Model.BackwardF32 := {
    finalHiddenGradient, lmHeadWeightGradient := dummy, decoderOutputGradient,
    finalNormWeightGradient, layers := layerBackward, embeddingWeightGradient := dummy
  }
  let modelShape : Model.ShapeF32 := {
    tokens := rows.toUInt32
    hidden := architecture.hidden
    intermediate := architecture.intermediate
    vocabulary := architecture.vocabulary
    linearKeyHeads := architecture.linearKeyHeads
    linearValueHeads := architecture.linearValueHeads
    linearKeyWidth := architecture.linearKeyWidth
    linearValueWidth := architecture.linearValueWidth
    attentionQueryHeads := architecture.attentionQueryHeads
    attentionKeyValueHeads := architecture.attentionKeyValueHeads
    attentionHeadWidth := architecture.attentionHeadWidth
    rotaryHalf := architecture.rotaryHalf
    epsilon := architecture.epsilon
    queryScale := architecture.queryScale
    attentionScale := architecture.attentionScale
  }
  let trainingDescriptor : Training.Descriptor := {
    steps := 1
    batchSize := descriptor.batchSize
    sequenceLength := descriptor.sequenceLength
    model := modelShape
    profile := .lora {
      adapterCount := descriptor.adapterCount
      rank := config.rank
      alpha := config.alpha
      adapterIds := descriptor.adapterIds
    }
  }
  let stepState : Training.FullModelStepF32 := {
    weights, forward, backward, tokenIds, targets, updateState := .lora
  }
  let schedule ← Training.AdamWScheduleF32.allocate stream
  let hyperparameters := config.hyperparameters
  let resident ← Training.ResidentTrainerF32.build hyperparameters schedule trainingDescriptor
    stepState
  let builder ← Cuda.GraphBuilder.create
  let layout : Cuda.Qwen36.SequenceLayout := {
    batchSize := descriptor.batchSize.toUInt32
    sequenceLength := descriptor.sequenceLength.toUInt32
  }
  Model.submitBatchedForwardLossF32 (.graph builder) weights forward tokenIds targets layout
    modelShape
  let evaluationGraph ← builder.instantiate

  let dpo : Option DPOState ← if config.enableDPO then do
    unless descriptor.batchSize ≥ 2 && descriptor.batchSize % 2 == 0 do
      throw <| IO.userError
        "Qwen3.8 projection-wide DPO requires an even chosen/rejected sequence batch"
    let masks ← Cuda.Buffer.alloc UInt32 rows.toUSize
    let referenceSequenceLogProbability ← allocateF32 descriptor.batchSize.toUInt32
    let policySequenceLogProbability ← allocateF32 descriptor.batchSize.toUInt32
    let sequenceCoefficients ← allocateF32 descriptor.batchSize.toUInt32
    let adapterIdOwners := adapters.map fun named => named.adapter.adapterIds
    let adapterIdOffsets : Array USize := Array.replicate adapters.size 0
    let adapterIdTable ← Cuda.BufferTable.createAtByteOffsets adapterIdOwners adapterIdOffsets
    let policyAdapterIds ← Cuda.Buffer.alloc UInt32 descriptor.batchSize.toUSize
    let referenceAdapterIds ← Cuda.Buffer.alloc UInt32 descriptor.batchSize.toUSize
    policyAdapterIds.copyFrom (packUInt32 descriptor.adapterIds) stream
    referenceAdapterIds.copyFrom
      (packUInt32 (Array.replicate descriptor.batchSize LoRA.noAdapter)) stream
    let routingLaunch := Primitives.elementConfig
      (adapters.size.toUInt32 * descriptor.batchSize.toUInt32)

    let referenceBuilder ← Cuda.GraphBuilder.create
    projectionAdapterRoutingKernel.addToGraph referenceBuilder routingLaunch adapterIdTable
      referenceAdapterIds adapters.size.toUInt32 descriptor.batchSize.toUInt32
    Model.submitBatchedSequenceLogProbabilityF32 (.graph referenceBuilder) weights forward tokenIds
      targets masks referenceSequenceLogProbability layout modelShape
    let referenceGraph ← referenceBuilder.instantiate

    let dpoEvaluationBuilder ← Cuda.GraphBuilder.create
    projectionAdapterRoutingKernel.addToGraph dpoEvaluationBuilder routingLaunch adapterIdTable
      policyAdapterIds adapters.size.toUInt32 descriptor.batchSize.toUInt32
    Model.submitBatchedDPOForwardF32 (.graph dpoEvaluationBuilder) weights forward tokenIds targets
      masks referenceSequenceLogProbability policySequenceLogProbability sequenceCoefficients layout
      modelShape config.dpoBeta
    let dpoEvaluationGraph ← dpoEvaluationBuilder.instantiate

    let dpoGraphHandle ← Cuda.Buffer.alloc UInt64 1
    let dpoRemaining ← Cuda.Buffer.alloc UInt32 1
    let dpoBuilder ← Cuda.GraphBuilder.create
    projectionAdapterRoutingKernel.addToGraph dpoBuilder routingLaunch adapterIdTable
      policyAdapterIds adapters.size.toUInt32 descriptor.batchSize.toUInt32
    Model.submitBatchedDPOForwardF32 (.graph dpoBuilder) weights forward tokenIds targets masks
      referenceSequenceLogProbability policySequenceLogProbability sequenceCoefficients layout
      modelShape config.dpoBeta
    Training.submitFullModelBackwardUpdateF32 (.graph dpoBuilder) hyperparameters schedule
      trainingDescriptor stepState
    Training.appendTailTrainingGraph dpoBuilder dpoGraphHandle dpoRemaining
    let dpoGraph ← dpoBuilder.instantiate
    let dpoResident : Training.ResidentTrainerF32 := {
      graph := dpoGraph
      graphHandle := dpoGraphHandle
      remaining := dpoRemaining
      schedule
      steps := 1
    }
    let state : DPOState := {
      masks, referenceSequenceLogProbability, policySequenceLogProbability, sequenceCoefficients,
      adapterIdTable, policyAdapterIds, referenceAdapterIds, beta := config.dpoBeta,
      referenceGraph, evaluationGraph := dpoEvaluationGraph, resident := dpoResident
    }
    pure (some state)
  else
    pure none

  let grpo : Option GRPOState ← if config.enableGRPO then do
    unless descriptor.batchSize >= config.grpoGroupSize &&
        descriptor.batchSize % config.grpoGroupSize == 0 do
      throw <| IO.userError
        "Qwen3.8 projection-wide GRPO group size must divide the sequence batch"
    let masks ← Cuda.Buffer.alloc UInt32 rows.toUSize
    let rewards ← allocateF32 descriptor.batchSize.toUInt32
    let advantages ← allocateF32 descriptor.batchSize.toUInt32
    let oldTokenLogProbability ← allocateF32 rows.toUInt32
    let referenceTokenLogProbability ← allocateF32 rows.toUInt32
    let policyTokenLogProbability ← allocateF32 rows.toUInt32
    let rowCoefficients ← allocateF32 rows.toUInt32
    let statistics ← allocateF32 2
    let adapterIdOwners := adapters.map fun named => named.adapter.adapterIds
    let adapterIdOffsets : Array USize := Array.replicate adapters.size 0
    let adapterIdTable ← Cuda.BufferTable.createAtByteOffsets adapterIdOwners adapterIdOffsets
    let policyAdapterIds ← Cuda.Buffer.alloc UInt32 descriptor.batchSize.toUSize
    let referenceAdapterIds ← Cuda.Buffer.alloc UInt32 descriptor.batchSize.toUSize
    policyAdapterIds.copyFrom (packUInt32 descriptor.adapterIds) stream
    referenceAdapterIds.copyFrom
      (packUInt32 (Array.replicate descriptor.batchSize LoRA.noAdapter)) stream
    let routingLaunch := Primitives.elementConfig
      (adapters.size.toUInt32 * descriptor.batchSize.toUInt32)

    let referenceBuilder ← Cuda.GraphBuilder.create
    projectionAdapterRoutingKernel.addToGraph referenceBuilder routingLaunch adapterIdTable
      referenceAdapterIds adapters.size.toUInt32 descriptor.batchSize.toUInt32
    Model.submitBatchedForwardF32 (.graph referenceBuilder) weights forward tokenIds layout modelShape
    Model.submitTokenLogProbabilityF32 (.graph referenceBuilder)
      "Qwen3.8 GRPO reference token log probability" forward.logits targets masks
      referenceTokenLogProbability modelShape.tokens modelShape.vocabulary
    let referenceGraph ← referenceBuilder.instantiate

    let grpoEvaluationBuilder ← Cuda.GraphBuilder.create
    projectionAdapterRoutingKernel.addToGraph grpoEvaluationBuilder routingLaunch adapterIdTable
      policyAdapterIds adapters.size.toUInt32 descriptor.batchSize.toUInt32
    Model.submitBatchedGRPOForwardF32 (.graph grpoEvaluationBuilder) weights forward tokenIds targets
      masks oldTokenLogProbability referenceTokenLogProbability policyTokenLogProbability rewards
      advantages rowCoefficients statistics layout modelShape config.grpoGroupSize.toUInt32
      config.grpoClipEpsilon config.grpoKLBeta config.grpoAdvantageEpsilon
    let grpoEvaluationGraph ← grpoEvaluationBuilder.instantiate

    let grpoGraphHandle ← Cuda.Buffer.alloc UInt64 1
    let grpoRemaining ← Cuda.Buffer.alloc UInt32 1
    let grpoBuilder ← Cuda.GraphBuilder.create
    projectionAdapterRoutingKernel.addToGraph grpoBuilder routingLaunch adapterIdTable
      policyAdapterIds adapters.size.toUInt32 descriptor.batchSize.toUInt32
    Model.submitBatchedGRPOForwardF32 (.graph grpoBuilder) weights forward tokenIds targets masks
      oldTokenLogProbability referenceTokenLogProbability policyTokenLogProbability rewards
      advantages rowCoefficients statistics layout modelShape config.grpoGroupSize.toUInt32
      config.grpoClipEpsilon config.grpoKLBeta config.grpoAdvantageEpsilon
    Training.submitFullModelBackwardUpdateF32 (.graph grpoBuilder) hyperparameters schedule
      trainingDescriptor stepState
    Training.appendTailTrainingGraph grpoBuilder grpoGraphHandle grpoRemaining
    let grpoGraph ← grpoBuilder.instantiate
    let grpoResident : Training.ResidentTrainerF32 := {
      graph := grpoGraph
      graphHandle := grpoGraphHandle
      remaining := grpoRemaining
      schedule
      steps := 1
    }
    pure (some {
      masks, rewards, advantages, oldTokenLogProbability, referenceTokenLogProbability,
      policyTokenLogProbability, rowCoefficients, statistics, adapterIdTable, policyAdapterIds,
      referenceAdapterIds, groupSize := config.grpoGroupSize.toUInt32,
      clipEpsilon := config.grpoClipEpsilon, klBeta := config.grpoKLBeta,
      advantageEpsilon := config.grpoAdvantageEpsilon, referenceGraph,
      evaluationGraph := grpoEvaluationGraph, resident := grpoResident
    })
  else
    pure none

  let mut gradientOwners : Array (Cuda.Buffer Float32) := #[]
  let mut gradientOffsets : Array USize := #[]
  let mut gradientCounts : Array UInt32 := #[]
  let mut gradientSlices : Array ProjectionGradientSlice := #[]
  for named in adapters do
    let shape := named.adapter.shape
    let aElements := shape.rank * shape.inputFeatures
    let bElements := shape.outputFeatures * shape.rank
    for adapterId in [:shape.adapterCount.toNat] do
      gradientOwners := gradientOwners.push named.adapter.adapterAGradient
      gradientOffsets := gradientOffsets.push ((adapterId.toUInt32 * aElements * 4).toUSize)
      gradientCounts := gradientCounts.push aElements
      gradientSlices := gradientSlices.push {
        projection := named.name, adapterId := adapterId.toUInt32,
        isAdapterA := true, elements := aElements.toNat
      }
      gradientOwners := gradientOwners.push named.adapter.adapterBGradient
      gradientOffsets := gradientOffsets.push ((adapterId.toUInt32 * bElements * 4).toUSize)
      gradientCounts := gradientCounts.push bElements
      gradientSlices := gradientSlices.push {
        projection := named.name, adapterId := adapterId.toUInt32,
        isAdapterA := false, elements := bElements.toNat
      }
  let gradientTable ← Cuda.BufferTable.createAtByteOffsets gradientOwners gradientOffsets
  let gradientElements ← Cuda.Buffer.alloc UInt32 gradientCounts.size.toUSize
  gradientElements.copyFrom (packUInt32 gradientCounts) stream
  let gradientStatistics ← allocateF32 (gradientCounts.size.toUInt32 * 4)
  return {
    checkpoint := trainable.checkpoint, trainable, descriptor := trainingDescriptor,
    stepState, schedule, resident, evaluationGraph, dpo, grpo, hyperparameters, compatibility, adapters,
    gradientSlices, gradientTable, gradientElements, gradientStatistics, stream,
    batchSize := descriptor.batchSize.toUInt32,
    sequenceLength := descriptor.sequenceLength.toUInt32
  }

namespace ProjectionTrainer

private def uploadBatch (trainer : @& ProjectionTrainer)
    (tokens targets : Array (Array UInt32)) : IO Unit := do
  unless tokens.size == trainer.batchSize.toNat && targets.size == trainer.batchSize.toNat do
    throw <| IO.userError s!"Qwen3.8 projection-wide trainer expected {trainer.batchSize} sequences"
  let mut flatTokens := #[]
  let mut flatTargets := #[]
  for batch in [:tokens.size] do
    unless tokens[batch]!.size == trainer.sequenceLength.toNat &&
        targets[batch]!.size == trainer.sequenceLength.toNat do
      throw <| IO.userError
        s!"Qwen3.8 projection-wide trainer expected sequence length {trainer.sequenceLength}"
    flatTokens := flatTokens ++ tokens[batch]!
    flatTargets := flatTargets ++ targets[batch]!
  trainer.stepState.tokenIds.copyFrom (packUInt32 flatTokens) trainer.stream
  trainer.stepState.targets.copyFrom (packUInt32 flatTargets) trainer.stream

private def requireDPO (trainer : @& ProjectionTrainer) : IO DPOState :=
  match trainer.dpo with
  | some dpo => pure dpo
  | none => throw <| IO.userError
      "Qwen3.8 projection-wide trainer was not loaded with DPO enabled"

private def requireGRPO (trainer : @& ProjectionTrainer) : IO GRPOState :=
  match trainer.grpo with
  | some grpo => pure grpo
  | none => throw <| IO.userError
      "Qwen3.8 projection-wide trainer was not loaded with GRPO enabled"

private def uploadDPOBatch (trainer : @& ProjectionTrainer) (dpo : @& DPOState)
    (tokens targets masks : Array (Array UInt32)) : IO Unit := do
  trainer.uploadBatch tokens targets
  unless masks.size == trainer.batchSize.toNat do
    throw <| IO.userError s!"Qwen3.8 projection-wide DPO expected {trainer.batchSize} masks"
  let mut flatMasks := #[]
  for sequence in masks do
    unless sequence.size == trainer.sequenceLength.toNat do
      throw <| IO.userError
        s!"Qwen3.8 projection-wide DPO expected mask length {trainer.sequenceLength}"
    unless sequence.all fun value => value == 0 || value == 1 do
      throw <| IO.userError "Qwen3.8 projection-wide DPO masks must contain only zero or one"
    unless sequence.any fun value => value == 1 do
      throw <| IO.userError "Qwen3.8 projection-wide DPO requires a nonempty completion mask"
    flatMasks := flatMasks ++ sequence
  dpo.masks.copyFrom (packUInt32 flatMasks) trainer.stream

private def dpoMetrics (trainer : @& ProjectionTrainer) (dpo : @& DPOState) : IO DPOResult := do
  let policy ← dpo.policySequenceLogProbability.copyTo trainer.stream
  let reference ← dpo.referenceSequenceLogProbability.copyTo trainer.stream
  let pairs := trainer.batchSize.toNat / 2
  let mut correct := 0
  let mut chosenTotal : Float32 := 0
  let mut rejectedTotal : Float32 := 0
  for pair in [:pairs] do
    let chosen := pair * 2
    let rejected := chosen + 1
    let chosenReward := dpo.beta *
      (readFloat32At policy (chosen * 4) - readFloat32At reference (chosen * 4))
    let rejectedReward := dpo.beta *
      (readFloat32At policy (rejected * 4) - readFloat32At reference (rejected * 4))
    if chosenReward > rejectedReward then
      correct := correct + 1
    chosenTotal := chosenTotal + chosenReward
    rejectedTotal := rejectedTotal + rejectedReward
  let inversePairs := 1 / pairs.toUInt32.toFloat32
  let meanChosenReward := chosenTotal * inversePairs
  let meanRejectedReward := rejectedTotal * inversePairs
  return {
    loss := readFloat32 (← trainer.stepState.forward.loss.copyTo trainer.stream)
    rewardAccuracy := correct.toUInt32.toFloat32 * inversePairs
    meanChosenReward
    meanRejectedReward
    meanRewardMargin := meanChosenReward - meanRejectedReward
  }

private def runDPOReference (trainer : @& ProjectionTrainer) (dpo : @& DPOState) : IO Unit := do
  dpo.referenceGraph.upload trainer.stream
  (← dpo.referenceGraph.launchOn trainer.stream).waitChecked
    "Qwen3.8 projection-wide DPO reference graph"

/-- Evaluate reference-scored DPO without mutating adapter or optimizer state. -/
def evaluateDPO (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) : IO DPOResult := do
  let dpo ← trainer.requireDPO
  trainer.uploadDPOBatch dpo tokens targets masks
  trainer.runDPOReference dpo
  dpo.evaluationGraph.upload trainer.stream
  (← dpo.evaluationGraph.launchOn trainer.stream).waitChecked
    "Qwen3.8 projection-wide DPO evaluation graph"
  trainer.dpoMetrics dpo

/-- Run frozen-reference scoring followed by one complete DPO reverse/AdamW graph. -/
def stepDPO (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) : IO DPOResult := do
  let dpo ← trainer.requireDPO
  trainer.uploadDPOBatch dpo tokens targets masks
  trainer.runDPOReference dpo
  if (← IO.getEnv "QWEN_TRAIN_SEQUENTIAL_DEBUG") == some "1" then
    throw <| IO.userError "QWEN_TRAIN_SEQUENTIAL_DEBUG is not implemented for DPO"
  dpo.resident.run trainer.stream
  trainer.dpoMetrics dpo

private def uploadGRPOBatch (trainer : @& ProjectionTrainer) (grpo : @& GRPOState)
    (tokens targets masks : Array (Array UInt32))
    (behaviorLogProbabilities : Array (Array Float32)) (rewards : Array Float32) : IO Unit := do
  trainer.uploadBatch tokens targets
  unless masks.size == trainer.batchSize.toNat do
    throw <| IO.userError s!"Qwen3.8 projection-wide GRPO expected {trainer.batchSize} masks"
  unless behaviorLogProbabilities.size == trainer.batchSize.toNat do
    throw <| IO.userError
      s!"Qwen3.8 projection-wide GRPO expected {trainer.batchSize} behavior-score sequences"
  unless rewards.size == trainer.batchSize.toNat do
    throw <| IO.userError s!"Qwen3.8 projection-wide GRPO expected {trainer.batchSize} rewards"
  unless rewards.all finiteFloat32 do
    throw <| IO.userError "Qwen3.8 projection-wide GRPO rewards must be finite"
  let mut flatMasks := #[]
  let mut flatBehavior := #[]
  for batch in [:masks.size] do
    let sequence := masks[batch]!
    let behavior := behaviorLogProbabilities[batch]!
    unless sequence.size == trainer.sequenceLength.toNat &&
        behavior.size == trainer.sequenceLength.toNat do
      throw <| IO.userError
        s!"Qwen3.8 projection-wide GRPO expected mask and behavior-score length {trainer.sequenceLength}"
    unless sequence.all fun value => value == 0 || value == 1 do
      throw <| IO.userError "Qwen3.8 projection-wide GRPO masks must contain only zero or one"
    unless sequence.any fun value => value == 1 do
      throw <| IO.userError "Qwen3.8 projection-wide GRPO requires nonempty action masks"
    for row in [:sequence.size] do
      unless finiteFloat32 behavior[row]! do
        throw <| IO.userError "Qwen3.8 projection-wide GRPO behavior log probabilities must be finite"
      if behavior[row]! > 0 then
        throw <| IO.userError
          "Qwen3.8 projection-wide GRPO behavior log probabilities must be at most zero"
      if sequence[row]! == 0 && behavior[row]! != 0 then
        throw <| IO.userError "Qwen3.8 projection-wide GRPO masked behavior scores must be zero"
    flatMasks := flatMasks ++ sequence
    flatBehavior := flatBehavior ++ behavior
  grpo.masks.copyFrom (packUInt32 flatMasks) trainer.stream
  grpo.oldTokenLogProbability.copyFrom (packFloat32 flatBehavior) trainer.stream
  grpo.rewards.copyFrom (packFloat32 rewards) trainer.stream

private def grpoMetrics (trainer : @& ProjectionTrainer) (grpo : @& GRPOState) :
    IO GRPOResult := do
  let rewardBytes ← grpo.rewards.copyTo trainer.stream
  let advantageBytes ← grpo.advantages.copyTo trainer.stream
  let statistics ← grpo.statistics.copyTo trainer.stream
  let count := trainer.batchSize.toNat
  let inverse := 1 / trainer.batchSize.toFloat32
  let mut rewardTotal : Float32 := 0
  let mut absoluteAdvantageTotal : Float32 := 0
  for index in [:count] do
    rewardTotal := rewardTotal + readFloat32At rewardBytes (index * 4)
    let advantage := readFloat32At advantageBytes (index * 4)
    absoluteAdvantageTotal := absoluteAdvantageTotal +
      Float32.ofBits (advantage.toBits &&& 0x7fffffff)
  let meanReward := rewardTotal * inverse
  let mut squaredDeviationTotal : Float32 := 0
  for index in [:count] do
    let difference := readFloat32At rewardBytes (index * 4) - meanReward
    squaredDeviationTotal := squaredDeviationTotal + difference * difference
  return {
    loss := readFloat32 (← trainer.stepState.forward.loss.copyTo trainer.stream)
    meanReward
    rewardStd := Float32.sqrt (squaredDeviationTotal * inverse)
    meanKL := readFloat32At statistics 0
    clipFraction := readFloat32At statistics 4
    meanAbsoluteAdvantage := absoluteAdvantageTotal * inverse
  }

private def prepareGRPO (trainer : @& ProjectionTrainer) (grpo : @& GRPOState)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32) : IO Unit := do
  trainer.uploadGRPOBatch grpo tokens targets masks behavior rewards
  grpo.referenceGraph.upload trainer.stream
  (← grpo.referenceGraph.launchOn trainer.stream).waitChecked
    "Qwen3.8 projection-wide GRPO reference graph"

/-- Evaluate one freshly prepared grouped rollout batch without updating policy or optimizer state. -/
def evaluateGRPO (trainer : @& ProjectionTrainer) (tokens targets masks : Array (Array UInt32))
    (behavior : Array (Array Float32)) (rewards : Array Float32) : IO GRPOResult := do
  let grpo ← trainer.requireGRPO
  trainer.prepareGRPO grpo tokens targets masks behavior rewards
  grpo.evaluationGraph.upload trainer.stream
  (← grpo.evaluationGraph.launchOn trainer.stream).waitChecked
    "Qwen3.8 projection-wide GRPO evaluation graph"
  trainer.grpoMetrics grpo

/--
Load collector-recorded behavior scores, cache frozen-reference scores once, then run one or more
clipped GRPO updates. Each update advances the shared AdamW schedule exactly once.
-/
def stepGRPOUpdates (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32)
    (updates : Nat) : IO GRPOResult := do
  unless updates > 0 do
    throw <| IO.userError "Qwen3.8 projection-wide GRPO updates must be positive"
  let grpo ← trainer.requireGRPO
  trainer.prepareGRPO grpo tokens targets masks behavior rewards
  if (← IO.getEnv "QWEN_TRAIN_SEQUENTIAL_DEBUG") == some "1" then
    throw <| IO.userError "QWEN_TRAIN_SEQUENTIAL_DEBUG is not implemented for GRPO"
  for _ in [:updates] do
    grpo.resident.run trainer.stream
  trainer.grpoMetrics grpo

/-- Run one clipped outcome-GRPO update on a freshly prepared grouped rollout batch. -/
def stepGRPO (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32) : IO GRPOResult :=
  trainer.stepGRPOUpdates tokens targets masks behavior rewards 1

/-- Build the device-resident model-wide tape used by both one-launch training objectives. -/
def buildFullMegakernelProgram (trainer : @& ProjectionTrainer) :
    IO FullTrainingMegakernel.Program :=
  FullTrainingMegakernel.Program.build trainer.stream trainer.descriptor trainer.stepState

/-- Run one complete pretraining update with a previously lowered model-wide program. -/
def stepFullMegakernelProgram (trainer : @& ProjectionTrainer)
    (program : @& FullTrainingMegakernel.Program)
    (tokens targets : Array (Array UInt32)) : IO Float32 := do
  trainer.uploadBatch tokens targets
  (← FullPretrainMegakernel.launch trainer.stream program trainer.hyperparameters
    trainer.schedule).waitChecked "Qwen3.8 full-model pretraining megakernel"
  return readFloat32 (← trainer.stepState.forward.loss.copyTo trainer.stream)

/-- Lower once and run one complete projection-wide pretraining update in one CUDA launch. -/
def stepFullMegakernel (trainer : @& ProjectionTrainer)
    (tokens targets : Array (Array UInt32)) : IO Float32 := do
  let program ← trainer.buildFullMegakernelProgram
  trainer.stepFullMegakernelProgram program tokens targets

/-- Load behavior/reference scores, then use a cached program for one-launch GRPO policy updates. -/
def stepGRPOFullMegakernelProgramUpdates (trainer : @& ProjectionTrainer)
    (program : @& FullTrainingMegakernel.Program)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32)
    (updates : Nat) : IO GRPOResult := do
  unless updates > 0 do
    throw <| IO.userError "Qwen3.8 full-model GRPO updates must be positive"
  let grpo ← trainer.requireGRPO
  trainer.prepareGRPO grpo tokens targets masks behavior rewards
  let objective : Model.GRPOObjectiveF32 := {
    sequences := trainer.batchSize
    sequenceLength := trainer.sequenceLength
    groupSize := grpo.groupSize
    clipEpsilon := grpo.clipEpsilon
    klBeta := grpo.klBeta
    advantageEpsilon := grpo.advantageEpsilon
  }
  for _ in [:updates] do
    (← FullGRPOMegakernel.launch trainer.stream program grpo.policyTokenLogProbability
      grpo.oldTokenLogProbability grpo.referenceTokenLogProbability grpo.rewards grpo.advantages
      grpo.rowCoefficients grpo.statistics grpo.masks objective trainer.hyperparameters
      trainer.schedule).waitChecked "Qwen3.8 full-model GRPO megakernel"
  trainer.grpoMetrics grpo

/-- Lower once, load behavior/reference scores, then execute complete GRPO updates in one launch each. -/
def stepGRPOFullMegakernelUpdates (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32)
    (updates : Nat) : IO GRPOResult := do
  let program ← trainer.buildFullMegakernelProgram
  trainer.stepGRPOFullMegakernelProgramUpdates program tokens targets masks behavior rewards updates

/-- Run one freshly prepared outcome-GRPO update in the complete-model training megakernel. -/
def stepGRPOFullMegakernel (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32) : IO GRPOResult :=
  trainer.stepGRPOFullMegakernelUpdates tokens targets masks behavior rewards 1

/-- Evaluate current projection-wide adapters without mutating parameters or optimizer state. -/
def evaluate (trainer : @& ProjectionTrainer) (tokens targets : Array (Array UInt32)) :
    IO Float32 := do
  trainer.uploadBatch tokens targets
  trainer.evaluationGraph.upload trainer.stream
  (← trainer.evaluationGraph.launchOn trainer.stream).waitChecked
    "Qwen3.8 projection-wide evaluation graph"
  return readFloat32 (← trainer.stepState.forward.loss.copyTo trainer.stream)

/-- Run one complete projection-wide forward/backward/AdamW graph and return pre-update loss. -/
def step (trainer : @& ProjectionTrainer) (tokens targets : Array (Array UInt32)) : IO Float32 := do
  trainer.uploadBatch tokens targets
  if (← IO.getEnv "QWEN_TRAIN_SEQUENTIAL_DEBUG") == some "1" then
    Training.submitFullModelStepF32 (.sequential trainer.stream) trainer.hyperparameters
      trainer.schedule trainer.descriptor trainer.stepState
  else
    trainer.resident.run trainer.stream
  return readFloat32 (← trainer.stepState.forward.loss.copyTo trainer.stream)

private def gradientStatsAt (slice : ProjectionGradientSlice) (bytes : ByteArray)
    (index : Nat) : GradientStats :=
  let finite := readFloat32At bytes ((index * 4) * 4)
  let nonzero := readFloat32At bytes ((index * 4 + 1) * 4)
  let maxAbs := readFloat32At bytes ((index * 4 + 2) * 4)
  let sumSquares := readFloat32At bytes ((index * 4 + 3) * 4)
  {
    elements := slice.elements
    finiteElements := finite.toUInt32.toNat
    nonzeroElements := nonzero.toUInt32.toNat
    maxAbs
    l2Norm := Float32.sqrt sumSquares
    rms := if slice.elements == 0 then 0 else
      Float32.sqrt (sumSquares / slice.elements.toUInt32.toFloat32)
  }

/--
Summarize every projection-local A/B gradient on device and copy only four scalars per tensor.
The result is ordered by canonical projection name and then routed adapter ID.
-/
def gradientDiagnostics (trainer : @& ProjectionTrainer) :
    IO (Array NamedAdapterGradientStats) := do
  let tensorCount := trainer.gradientSlices.size.toUInt32
  let launch : Cuda.LaunchConfig := {
    grid := { x := tensorCount }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  }
  (← projectionGradientStatsKernel.launchOn trainer.stream launch trainer.gradientTable
    trainer.gradientElements trainer.gradientStatistics).waitChecked
    "Qwen3.8 projection-wide gradient diagnostics"
  let bytes ← trainer.gradientStatistics.copyTo trainer.stream
  let mut diagnostics := #[]
  for pair in [:trainer.gradientSlices.size / 2] do
    let aIndex := pair * 2
    let bIndex := aIndex + 1
    let aSlice := trainer.gradientSlices[aIndex]!
    let bSlice := trainer.gradientSlices[bIndex]!
    unless aSlice.isAdapterA && !bSlice.isAdapterA &&
        aSlice.projection == bSlice.projection && aSlice.adapterId == bSlice.adapterId do
      throw <| IO.userError "Qwen3.8 projection-wide gradient diagnostic table is inconsistent"
    diagnostics := diagnostics.push {
      projection := aSlice.projection
      adapterId := aSlice.adapterId
      adapterA := gradientStatsAt aSlice bytes aIndex
      adapterB := gradientStatsAt bSlice bytes bIndex
    }
  return diagnostics

/--
Run one complete real-checkpoint forward/backward/AdamW graph and return bounded pre-update
diagnostics for all projection-local adapters.
-/
def stepWithDiagnostics (trainer : @& ProjectionTrainer)
    (tokens targets : Array (Array UInt32)) : IO ProjectionStepResult := do
  let loss ← trainer.step tokens targets
  return { loss, gradients := ← trainer.gradientDiagnostics }

/-- Run one DPO step and return its pair metrics plus bounded projection-gradient diagnostics. -/
def stepDPOWithDiagnostics (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) : IO DPOProjectionStepResult := do
  let result ← trainer.stepDPO tokens targets masks
  return {
    loss := result.loss
    rewardAccuracy := result.rewardAccuracy
    meanChosenReward := result.meanChosenReward
    meanRejectedReward := result.meanRejectedReward
    meanRewardMargin := result.meanRewardMargin
    gradients := ← trainer.gradientDiagnostics
  }

/-- Run clipped GRPO updates and return metrics plus bounded projection-gradient diagnostics. -/
def stepGRPOWithDiagnostics (trainer : @& ProjectionTrainer)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32)
    (updates : Nat := 1) : IO GRPOProjectionStepResult := do
  let result ← trainer.stepGRPOUpdates tokens targets masks behavior rewards updates
  return {
    loss := result.loss
    meanReward := result.meanReward
    rewardStd := result.rewardStd
    meanKL := result.meanKL
    clipFraction := result.clipFraction
    meanAbsoluteAdvantage := result.meanAbsoluteAdvantage
    gradients := ← trainer.gradientDiagnostics
  }

private def projectionSnapshotMagic : ByteArray := "LCQPROJ2".toUTF8
private def projectionSnapshotVersion : UInt32 := 2

private def writeChecked (handle : @& IO.FS.Handle) (checksum : IO.Ref UInt64)
    (bytes : @& ByteArray) : IO Unit := do
  handle.write bytes
  checksum.modify fun current => updateFingerprint current bytes

private def writeFieldChecked (handle : @& IO.FS.Handle) (checksum : IO.Ref UInt64)
    (bytes : @& ByteArray) : IO Unit := do
  writeChecked handle checksum (pushUInt32 ByteArray.empty bytes.size.toUInt32)
  writeChecked handle checksum bytes

private partial def readExact (handle : @& IO.FS.Handle) (remaining : Nat)
    (bytes : ByteArray := ByteArray.empty) : IO ByteArray := do
  if remaining == 0 then
    return bytes
  let chunk ← handle.read (min remaining (64 * 1024 * 1024)).toUSize
  if chunk.isEmpty then
    throw <| IO.userError "truncated Qwen3.8 projection-wide adapter checkpoint"
  readExact handle (remaining - chunk.size) (bytes ++ chunk)

private def readExactChecked (handle : @& IO.FS.Handle) (checksum : IO.Ref UInt64)
    (remaining : Nat) : IO ByteArray := do
  let bytes ← readExact handle remaining
  checksum.modify fun current => updateFingerprint current bytes
  return bytes


private def readUInt32CheckedFromHandle (handle : @& IO.FS.Handle)
    (checksum : IO.Ref UInt64) : IO UInt32 :=
  return readPackedUInt32At (← readExactChecked handle checksum 4) 0

private def readFieldChecked (handle : @& IO.FS.Handle) (checksum : IO.Ref UInt64) :
    IO ByteArray := do
  readExactChecked handle checksum (← readUInt32CheckedFromHandle handle checksum).toNat

/--
Stream all 497 adapters and their moments to a temporary file, then atomically publish it. The v2
format includes exact model/trainer compatibility metadata and a streaming integrity checksum.
-/
def saveCheckpoint (trainer : @& ProjectionTrainer) (path : System.FilePath) : IO Unit := do
  let temporary : System.FilePath := path.toString ++ ".tmp"
  IO.FS.withFile temporary .write fun handle => do
    let checksum ← IO.mkRef fnvOffset
    writeChecked handle checksum projectionSnapshotMagic
    writeChecked handle checksum (pushUInt32 ByteArray.empty projectionSnapshotVersion)
    writeChecked handle checksum (pushUInt32 ByteArray.empty trainer.adapters.size.toUInt32)
    writeFieldChecked handle checksum trainer.compatibility.toBytes
    let schedule ← trainer.schedule.snapshot trainer.stream
    for field in #[schedule.step, schedule.beta1Power, schedule.beta2Power,
        schedule.inverseBiasCorrection1, schedule.inverseBiasCorrection2] do
      writeFieldChecked handle checksum field
    for named in trainer.adapters do
      writeFieldChecked handle checksum named.name.toUTF8
      let snapshot ← named.adapter.snapshot trainer.stream
      for field in #[snapshot.adapterA, snapshot.adapterB, snapshot.adapterIds,
          snapshot.adapterAFirstMoment, snapshot.adapterASecondMoment,
          snapshot.adapterBFirstMoment, snapshot.adapterBSecondMoment,
          snapshot.updateMask] do
        writeFieldChecked handle checksum field
    handle.write (pushUInt64 ByteArray.empty (← checksum.get))
    handle.flush
  IO.FS.rename temporary path

private def readCheckpoint (trainer : @& ProjectionTrainer) (path : System.FilePath)
    (restore : Bool) : IO Unit := do
  IO.FS.withFile path .read fun handle => do
    let checksum ← IO.mkRef fnvOffset
    unless (← readExactChecked handle checksum projectionSnapshotMagic.size) ==
        projectionSnapshotMagic do
      throw <| IO.userError "invalid Qwen3.8 projection-wide adapter checkpoint magic"
    unless (← readUInt32CheckedFromHandle handle checksum) == projectionSnapshotVersion do
      throw <| IO.userError "unsupported Qwen3.8 projection-wide adapter checkpoint version"
    let adapterCount ← readUInt32CheckedFromHandle handle checksum
    unless adapterCount.toNat == trainer.adapters.size do
      throw <| IO.userError s!"Qwen3.8 projection-wide checkpoint has {adapterCount} adapters, expected {trainer.adapters.size}"
    let compatibility ← match ProjectionCheckpointCompatibility.ofBytes
        (← readFieldChecked handle checksum) with
      | .ok compatibility => pure compatibility
      | .error message => throw <| IO.userError message
    unless compatibility == trainer.compatibility do
      throw <| IO.userError <|
        s!"Qwen3.8 projection-wide checkpoint is incompatible with this trainer; " ++
        s!"checkpoint={repr compatibility}, trainer={repr trainer.compatibility}"
    let schedule : Training.AdamWScheduleSnapshotF32 := {
      step := ← readFieldChecked handle checksum
      beta1Power := ← readFieldChecked handle checksum
      beta2Power := ← readFieldChecked handle checksum
      inverseBiasCorrection1 := ← readFieldChecked handle checksum
      inverseBiasCorrection2 := ← readFieldChecked handle checksum
    }
    trainer.schedule.checkSnapshot schedule
    if restore then
      trainer.schedule.restore schedule trainer.stream
    for named in trainer.adapters do
      let nameBytes ← readFieldChecked handle checksum
      unless nameBytes == named.name.toUTF8 do
        throw <| IO.userError s!"Qwen3.8 projection-wide checkpoint order mismatch at {named.name}"
      let snapshot : Projection.AdapterSnapshotF32 := {
        adapterA := ← readFieldChecked handle checksum
        adapterB := ← readFieldChecked handle checksum
        adapterIds := ← readFieldChecked handle checksum
        adapterAFirstMoment := ← readFieldChecked handle checksum
        adapterASecondMoment := ← readFieldChecked handle checksum
        adapterBFirstMoment := ← readFieldChecked handle checksum
        adapterBSecondMoment := ← readFieldChecked handle checksum
        updateMask := ← readFieldChecked handle checksum
      }
      named.adapter.checkSnapshot snapshot
      if restore then
        named.adapter.restore snapshot trainer.stream
    let expectedChecksum := readPackedUInt64At (← readExact handle 8) 0
    unless expectedChecksum == (← checksum.get) do
      throw <| IO.userError "Qwen3.8 projection-wide adapter checkpoint checksum mismatch"
    unless (← handle.read 1).isEmpty do
      throw <| IO.userError "Qwen3.8 projection-wide adapter checkpoint has trailing bytes"

/-- Validate the entire file before restoring any device state, then restore the checked payload. -/
def resumeCheckpoint (trainer : @& ProjectionTrainer) (path : System.FilePath) : IO Unit := do
  readCheckpoint trainer path false
  readCheckpoint trainer path true

end ProjectionTrainer

end Cuda.Qwen36.CheckpointLoRA
