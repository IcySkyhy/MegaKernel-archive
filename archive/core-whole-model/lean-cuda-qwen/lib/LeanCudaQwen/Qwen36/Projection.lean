/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.LoRA

public section

/-!
# Qwen3.6 projection dispatch

A model projection is either an ordinary trainable dense matrix or a frozen base matrix with
mixed-batch LoRA state. This is the shared host scheduling boundary used by every Qwen projection;
component schedules no longer need a parallel LoRA implementation.
-/

namespace Cuda.Qwen36.Projection

/-- Projection-local adapter values, VJP scratch, optimizer moments, and update mask. -/
structure AdapterF32 where
  adapterA : Cuda.Buffer Float32
  adapterB : Cuda.Buffer Float32
  adapterIds : Cuda.Buffer UInt32
  rankActivation : Cuda.Buffer Float32
  adapterAGradient : Cuda.Buffer Float32
  adapterBGradient : Cuda.Buffer Float32
  adapterAFirstMoment : Cuda.Buffer Float32
  adapterASecondMoment : Cuda.Buffer Float32
  adapterBFirstMoment : Cuda.Buffer Float32
  adapterBSecondMoment : Cuda.Buffer Float32
  updateMask : Cuda.Buffer UInt32
  weightDecay : Float32
  shape : LoRA.Shape

/-- Serializable parameter, routing, mask, and AdamW state for one projection-local adapter. -/
structure AdapterSnapshotF32 where
  adapterA : ByteArray
  adapterB : ByteArray
  adapterIds : ByteArray
  adapterAFirstMoment : ByteArray
  adapterASecondMoment : ByteArray
  adapterBFirstMoment : ByteArray
  adapterBSecondMoment : ByteArray
  updateMask : ByteArray
  deriving BEq

namespace AdapterF32

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def packUInt32 (values : Array UInt32) : ByteArray :=
  values.foldl (init := ByteArray.emptyWithCapacity (values.size * 4)) pushUInt32

private def requireBytes (label : String) (buffer : @& Cuda.Buffer α) (bytes : USize) : IO Unit := do
  unless (← buffer.byteSize) == bytes do
    throw <| IO.userError s!"Qwen3.6 {label} allocation size mismatch"

/-- Recheck projection geometry and every adapter-owned allocation before graph construction. -/
def check (adapter : @& AdapterF32) (rows inputFeatures outputFeatures : UInt32) : IO Unit := do
  let shape ← match adapter.shape.check with
    | .ok shape => pure shape
    | .error message => throw <| IO.userError message
  unless shape.rows == rows && shape.inputFeatures == inputFeatures &&
      shape.outputFeatures == outputFeatures do
    throw <| IO.userError "Qwen3.6 LoRA shape does not match its projection"
  let floatBytes (elements : UInt32) : USize := (elements * 4).toUSize
  let aElements := shape.adapterCount * shape.rank * shape.inputFeatures
  let bElements := shape.adapterCount * shape.outputFeatures * shape.rank
  requireBytes "LoRA adapter-A" adapter.adapterA (floatBytes aElements)
  requireBytes "LoRA adapter-B" adapter.adapterB (floatBytes bElements)
  requireBytes "LoRA adapter IDs" adapter.adapterIds
    ((shape.rows / shape.sequenceLength * 4).toUSize)
  requireBytes "LoRA rank activation" adapter.rankActivation
    (floatBytes (shape.rows * shape.rank))
  requireBytes "LoRA adapter-A gradient" adapter.adapterAGradient (floatBytes aElements)
  requireBytes "LoRA adapter-B gradient" adapter.adapterBGradient (floatBytes bElements)
  requireBytes "LoRA adapter-A first moment" adapter.adapterAFirstMoment (floatBytes aElements)
  requireBytes "LoRA adapter-A second moment" adapter.adapterASecondMoment (floatBytes aElements)
  requireBytes "LoRA adapter-B first moment" adapter.adapterBFirstMoment (floatBytes bElements)
  requireBytes "LoRA adapter-B second moment" adapter.adapterBSecondMoment (floatBytes bElements)
  requireBytes "LoRA update mask" adapter.updateMask ((shape.adapterCount * 4).toUSize)
  unless adapter.weightDecay ≥ 0 &&
      adapter.weightDecay.toBits &&& 0x7f800000 != 0x7f800000 do
    throw <| IO.userError "Qwen3.6 LoRA weight decay must be nonnegative and finite"

/--
Allocate and initialize every buffer owned by one LoRA projection. Adapter A uses deterministic
Philox values, adapter B is zero so the initial projection exactly equals its frozen base, and all
gradients and AdamW moments start at zero.
-/
def allocate (descriptor : LoRA.Descriptor) (updateMask : Array UInt32) (seed : UInt64)
    (initScale weightDecay : Float32) (stream : @& Cuda.Stream) : IO AdapterF32 := do
  let descriptor ← match descriptor.check with
    | .ok descriptor => pure descriptor
    | .error message => throw <| IO.userError message
  unless updateMask.size == descriptor.adapterCount do
    throw <| IO.userError <|
      s!"Qwen3.6 LoRA update-mask count {updateMask.size} does not equal adapter count " ++
      s!"{descriptor.adapterCount}"
  unless updateMask.all fun value => value == 0 || value == 1 do
    throw <| IO.userError "Qwen3.6 LoRA update mask must contain only zero or one"
  unless initScale > 0 && initScale.toBits &&& 0x7f800000 != 0x7f800000 do
    throw <| IO.userError
      "Qwen3.6 LoRA adapter-A initialization scale must be positive and finite"
  unless weightDecay ≥ 0 && weightDecay.toBits &&& 0x7f800000 != 0x7f800000 do
    throw <| IO.userError "Qwen3.6 LoRA weight decay must be nonnegative and finite"
  let shape ← match descriptor.toShape with
    | .ok shape => pure shape
    | .error message => throw <| IO.userError message
  let aElements := descriptor.adapterCount * descriptor.rank * descriptor.inputFeatures
  let bElements := descriptor.adapterCount * descriptor.outputFeatures * descriptor.rank
  let rankElements := descriptor.rows * descriptor.rank
  let adapterA ← Cuda.Buffer.alloc Float32 aElements.toUSize
  let adapterB ← Cuda.Buffer.alloc Float32 bElements.toUSize
  let adapterIds ← Cuda.Buffer.alloc UInt32 descriptor.batchSize.toUSize
  let rankActivation ← Cuda.Buffer.alloc Float32 rankElements.toUSize
  let adapterAGradient ← Cuda.Buffer.alloc Float32 aElements.toUSize
  let adapterBGradient ← Cuda.Buffer.alloc Float32 bElements.toUSize
  let adapterAFirstMoment ← Cuda.Buffer.alloc Float32 aElements.toUSize
  let adapterASecondMoment ← Cuda.Buffer.alloc Float32 aElements.toUSize
  let adapterBFirstMoment ← Cuda.Buffer.alloc Float32 bElements.toUSize
  let adapterBSecondMoment ← Cuda.Buffer.alloc Float32 bElements.toUSize
  let updateMaskBuffer ← Cuda.Buffer.alloc UInt32 descriptor.adapterCount.toUSize
  adapterIds.copyFrom (packUInt32 descriptor.adapterIds) stream
  updateMaskBuffer.copyFrom (packUInt32 updateMask) stream
  LoRA.initializeAdapterF32 stream adapterA adapterB rankActivation adapterAGradient
    adapterBGradient adapterAFirstMoment adapterASecondMoment adapterBFirstMoment
    adapterBSecondMoment seed initScale shape
  let adapter : AdapterF32 := {
    adapterA, adapterB, adapterIds, rankActivation, adapterAGradient, adapterBGradient,
    adapterAFirstMoment, adapterASecondMoment, adapterBFirstMoment, adapterBSecondMoment,
    updateMask := updateMaskBuffer, weightDecay, shape
  }
  adapter.check shape.rows shape.inputFeatures shape.outputFeatures
  return adapter

/-- Copy all trainable, routing, masking, and AdamW state to host-owned checkpoint bytes. -/
def snapshot (adapter : @& AdapterF32) (stream : @& Cuda.Stream) : IO AdapterSnapshotF32 := do
  return {
    adapterA := ← adapter.adapterA.copyTo stream
    adapterB := ← adapter.adapterB.copyTo stream
    adapterIds := ← adapter.adapterIds.copyTo stream
    adapterAFirstMoment := ← adapter.adapterAFirstMoment.copyTo stream
    adapterASecondMoment := ← adapter.adapterASecondMoment.copyTo stream
    adapterBFirstMoment := ← adapter.adapterBFirstMoment.copyTo stream
    adapterBSecondMoment := ← adapter.adapterBSecondMoment.copyTo stream
    updateMask := ← adapter.updateMask.copyTo stream
  }

private def requireSnapshotBytes (label : String) (buffer : @& Cuda.Buffer α)
    (bytes : @& ByteArray) : IO Unit := do
  unless bytes.size.toUSize == (← buffer.byteSize) do
    throw <| IO.userError s!"Qwen3.6 LoRA snapshot {label} size mismatch"

/-- Validate snapshot allocation sizes without mutating adapter state. -/
def checkSnapshot (adapter : @& AdapterF32) (snapshot : @& AdapterSnapshotF32) : IO Unit := do
  for (label, buffer, bytes) in #[
      ("adapter A", adapter.adapterA, snapshot.adapterA),
      ("adapter B", adapter.adapterB, snapshot.adapterB),
      ("adapter-A first moment", adapter.adapterAFirstMoment, snapshot.adapterAFirstMoment),
      ("adapter-A second moment", adapter.adapterASecondMoment, snapshot.adapterASecondMoment),
      ("adapter-B first moment", adapter.adapterBFirstMoment, snapshot.adapterBFirstMoment),
      ("adapter-B second moment", adapter.adapterBSecondMoment, snapshot.adapterBSecondMoment)
    ] do
    requireSnapshotBytes label buffer bytes
  requireSnapshotBytes "adapter IDs" adapter.adapterIds snapshot.adapterIds
  requireSnapshotBytes "update mask" adapter.updateMask snapshot.updateMask

/-- Restore adapter parameters, routing, masking, and both AdamW moments on `stream`. -/
def restore (adapter : @& AdapterF32) (snapshot : @& AdapterSnapshotF32)
    (stream : @& Cuda.Stream) : IO Unit := do
  adapter.checkSnapshot snapshot
  adapter.adapterA.copyFrom snapshot.adapterA stream
  adapter.adapterB.copyFrom snapshot.adapterB stream
  adapter.adapterIds.copyFrom snapshot.adapterIds stream
  adapter.adapterAFirstMoment.copyFrom snapshot.adapterAFirstMoment stream
  adapter.adapterASecondMoment.copyFrom snapshot.adapterASecondMoment stream
  adapter.adapterBFirstMoment.copyFrom snapshot.adapterBFirstMoment stream
  adapter.adapterBSecondMoment.copyFrom snapshot.adapterBSecondMoment stream
  adapter.updateMask.copyFrom snapshot.updateMask stream

end AdapterF32

/-- Dense projection policy. The LoRA constructor owns a frozen base and mutable adapters. -/
inductive WeightsF32 where
  | base (weight : Cuda.Buffer Float32)
  | lora (baseWeight : Cuda.Buffer Float32) (adapter : AdapterF32)
  | frozenBF16 (baseWeight : Linear.FrozenBFloat16F32)
  | loraFrozenBF16 (baseWeight : Linear.FrozenBFloat16F32) (adapter : AdapterF32)

instance : Coe (Cuda.Buffer Float32) WeightsF32 where
  coe := .base

namespace WeightsF32

/-- Underlying Float32 dense matrix when this is an oracle/full-model projection. -/
def baseWeight? : WeightsF32 → Option (Cuda.Buffer Float32)
  | .base weight => some weight
  | .lora weight _ => some weight
  | .frozenBF16 _ => none
  | .loraFrozenBF16 _ _ => none

/-- Projection-local adapter state when the base matrix is frozen. -/
def adapter? : WeightsF32 → Option AdapterF32
  | .base _ => none
  | .lora _ adapter => some adapter
  | .frozenBF16 _ => none
  | .loraFrozenBF16 _ adapter => some adapter

end WeightsF32

/-- Submit one dense or mixed-LoRA projection to the shared stream/graph executor. -/
def submitForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input : @& Cuda.Buffer Float32) (weights : @& WeightsF32)
    (output : @& Cuda.Buffer Float32) (rows inputFeatures outputFeatures : UInt32) : IO Unit :=
  match weights with
  | .base weight =>
      Linear.submitForwardF32 executor label input weight output rows inputFeatures outputFeatures
  | .lora baseWeight adapter => do
      adapter.check rows inputFeatures outputFeatures
      unless (← baseWeight.byteSize) == (outputFeatures * inputFeatures * 4).toUSize do
        throw <| IO.userError "Qwen3.6 frozen LoRA base allocation size mismatch"
      LoRA.submitForwardF32 executor label input baseWeight adapter.adapterA adapter.adapterB
        adapter.adapterIds adapter.rankActivation output adapter.shape
  | .frozenBF16 baseWeight =>
      Linear.submitForwardFrozenBF16F32 executor label input baseWeight output rows inputFeatures
        outputFeatures
  | .loraFrozenBF16 baseWeight adapter => do
      adapter.check rows inputFeatures outputFeatures
      Linear.submitForwardFrozenBF16F32 executor (label ++ " frozen base") input baseWeight output
        rows inputFeatures outputFeatures
      LoRA.submitAddForwardF32 executor label input adapter.adapterA adapter.adapterB
        adapter.adapterIds adapter.rankActivation output adapter.shape

/-- Submit input and trainable-parameter VJPs. LoRA freezes the base and publishes only dA/dB. -/
def submitBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input : @& Cuda.Buffer Float32) (weights : @& WeightsF32)
    (outputGradient inputGradient baseWeightGradient : @& Cuda.Buffer Float32)
    (rows inputFeatures outputFeatures : UInt32) : IO Unit :=
  match weights with
  | .base weight => do
      Linear.submitBackwardInputF32 executor (label ++ " input VJP") outputGradient weight
        inputGradient rows inputFeatures outputFeatures
      Linear.submitBackwardWeightF32 executor (label ++ " weight VJP") input outputGradient
        baseWeightGradient rows inputFeatures outputFeatures
  | .lora baseWeight adapter => do
      adapter.check rows inputFeatures outputFeatures
      unless (← baseWeight.byteSize) == (outputFeatures * inputFeatures * 4).toUSize do
        throw <| IO.userError "Qwen3.6 frozen LoRA base allocation size mismatch"
      LoRA.submitBackwardF32 executor label input baseWeight adapter.adapterA adapter.adapterB
        adapter.adapterIds adapter.rankActivation outputGradient inputGradient
        adapter.adapterAGradient adapter.adapterBGradient adapter.shape
  | .frozenBF16 baseWeight =>
      Linear.submitBackwardInputFrozenBF16F32 executor (label ++ " input VJP") outputGradient
        baseWeight inputGradient rows inputFeatures outputFeatures
  | .loraFrozenBF16 baseWeight adapter => do
      adapter.check rows inputFeatures outputFeatures
      Linear.submitBackwardInputFrozenBF16F32 executor (label ++ " frozen-base input VJP")
        outputGradient baseWeight inputGradient rows inputFeatures outputFeatures
      LoRA.submitAddBackwardF32 executor label input adapter.adapterA adapter.adapterB
        adapter.adapterIds adapter.rankActivation outputGradient inputGradient
        adapter.adapterAGradient adapter.adapterBGradient adapter.shape

end Cuda.Qwen36.Projection
