/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda

/-!
# Qwen3.6 decode projection engine

This is the plain-Lean CUDA matrix-vector phase used to attach checkpoint projections to the
resident Qwen3.6 token worker. The validation problem is the model's exact largest recurrent
input projection: BF16 `[10240, 5120]` times one BF16 hidden vector. Forty-eight cooperative
blocks cover output rows with FP32 accumulation and a convergent 256-thread reduction.

The descriptor carries dimensions and offsets so the same resident body can execute QKV, Z, A,
B, DeltaNet output, full-attention, MLP, and LM-head projections without another kernel family.
-/

namespace Qwen36Projection

private def hiddenSizeNat : Nat := 5120
private def qkvSizeNat : Nat := 10240
private def blockCountNat : Nat := 48
private def elementCount : Nat := hiddenSizeNat * qkvSizeNat

private def hiddenSize : UInt32 := hiddenSizeNat.toUInt32
private def qkvSize : UInt32 := qkvSizeNat.toUInt32
private def blockCount : UInt32 := blockCountNat.toUInt32
private def zero : Float32 := Float32.ofBits 0
private def one : Float32 := Float32.ofBits 0x3f800000

structure ProjectionItem where
  rows : UInt32
  columns : UInt32
  inputOffset : UInt32
  weightOffset : UInt32
  outputOffset : UInt32
  descriptor : UInt32
  deriving Cuda.POD

@[always_inline]
private partial def initializeWeightsLoop (weight : Cuda.DevicePtr Cuda.BFloat16)
    (columns index stride count : UInt32) : Cuda.DeviceM Unit := do
  if index < count then
    let row := index / columns
    let column := index % columns
    let residue := (column * 7 + 3) % 17
    let magnitude := (residue.toFloat32 - Float32.ofBits 0x41000000) /
      Float32.ofBits 0x42800000
    let value := if row % 2 == 0 then magnitude else -magnitude
    Cuda.storeBFloat16 weight index.toUSize (Cuda.BFloat16.ofFloat32 value)
    initializeWeightsLoop weight columns (index + stride) stride count

@[cuda_kernel]
def initializeProjectionWeights (rows columns : UInt32)
    (weight : Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  initializeWeightsLoop weight columns (← Cuda.globalThreadIdxX)
    (← Cuda.globalThreadCountX) (rows * columns)

@[always_inline]
private partial def dotRow (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (inputOffset weightOffset row columns column stride : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < columns then
    let inputValue := (← Cuda.loadBFloat16 input (inputOffset + column).toUSize).toFloat32
    let weightValue := (← Cuda.loadBFloat16 weight
      (weightOffset + row * columns + column).toUSize).toFloat32
    dotRow input weight inputOffset weightOffset row columns (column + stride) stride
      (Cuda.fma weightValue inputValue accumulator)
  else
    return accumulator

@[always_inline, convergent]
private partial def runRows (item : ProjectionItem)
    (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (scratch : Cuda.Collective.BlockScratch 8) (row rowStride thread : UInt32) :
    Cuda.DeviceM Unit := do
  if row < item.rows then
    let localSum ← dotRow input weight item.inputOffset item.weightOffset row item.columns thread 256 zero
    let total ← Cuda.Collective.blockSum scratch localSum
    if thread == 0 then
      Cuda.storeBFloat16 output (item.outputOffset + row).toUSize
        (Cuda.BFloat16.ofFloat32 total)
    runRows item input weight output scratch (row + rowStride) rowStride thread

@[always_inline, convergent]
private def runProjection (item : ProjectionItem)
    (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runRows item input weight output scratch (← Cuda.blockIdxX) (← Cuda.gridDimX)
    (← Cuda.threadIdxX)

@[cuda_kernel]
def qwen36ProjectionStep (item : ProjectionItem)
    (input weight output : Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runProjection item input weight output shared

@[cuda_grid_persistent]
def qwen36Projection (item : ProjectionItem)
    (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runProjection item input weight output shared
  Cuda.gridSync
  if (← Cuda.blockIdxX) == 0 && (← Cuda.threadIdxX) == 0 then
    Cuda.atomicAddUInt32At_ counts item.descriptor 1

private structure DeviceBuffers where
  input : Cuda.Buffer Cuda.BFloat16
  weight : Cuda.Buffer Cuda.BFloat16
  separateOutput : Cuda.Buffer Cuda.BFloat16
  persistentOutput : Cuda.Buffer Cuda.BFloat16
  counts : Cuda.Buffer UInt32

private def pushUInt16 (bytes : ByteArray) (value : UInt16) : ByteArray :=
  (bytes.push value.toUInt8).push (value >>> 8).toUInt8

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def readUInt16 (bytes : ByteArray) (index : Nat) : UInt16 :=
  let offset := index * 2
  bytes[offset]!.toUInt16 ||| (bytes[offset + 1]!.toUInt16 <<< 8)

private def readUInt32 (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def readBFloat (bytes : ByteArray) (index : Nat) : Float32 :=
  let value : Cuda.BFloat16 := ⟨readUInt16 bytes index⟩
  value.toFloat32

private def zeroBytes (count : Nat) : ByteArray :=
  List.replicate count 0 |>.toByteArray

private def makeBFloatData (count : Nat) (value : Nat → Float32) : ByteArray :=
  (List.range count).foldl (init := ByteArray.empty) fun bytes index =>
    pushUInt16 bytes (Cuda.BFloat16.ofFloat32 (value index)).bits

private def signedValue (index modulus center divisor : Nat) : Float32 :=
  (Int32.ofInt (Int.ofNat (index % modulus) - Int.ofNat center)).toFloat32 /
    divisor.toUInt32.toFloat32

private def inputValue (column : Nat) : Float32 :=
  signedValue (column * 11 + 5) 31 15 32

private def baseWeightValue (column : Nat) : Float32 :=
  signedValue (column * 7 + 3) 17 8 64

private def initializeBuffers (stream : Cuda.Stream) : IO DeviceBuffers := do
  let input ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let weight ← Cuda.Buffer.alloc Cuda.BFloat16 elementCount.toUSize
  let separateOutput ← Cuda.Buffer.alloc Cuda.BFloat16 qkvSizeNat.toUSize
  let persistentOutput ← Cuda.Buffer.alloc Cuda.BFloat16 qkvSizeNat.toUSize
  let counts ← Cuda.Buffer.alloc UInt32 1
  input.copyFrom (makeBFloatData hiddenSizeNat inputValue) stream
  separateOutput.copyFrom (zeroBytes (qkvSizeNat * 2)) stream
  persistentOutput.copyFrom (zeroBytes (qkvSizeNat * 2)) stream
  counts.copyFrom (zeroBytes 4) stream
  return { input, weight, separateOutput, persistentOutput, counts }

private def projectionItem : ProjectionItem := {
  rows := qkvSize
  columns := hiddenSize
  inputOffset := 0
  weightOffset := 0
  outputOffset := 0
  descriptor := 0
}

private def initializerConfig : Cuda.LaunchConfig := {
  grid := { x := 256 }
  block := { x := 256 }
  blockArenaBytes := 4 * 1024
}

private def launchConfig : Cuda.LaunchConfig := {
  grid := { x := blockCount }
  block := { x := 256 }
  sharedMemoryBytes := (8 * 4).toUSize
  blockArenaBytes := 8 * 1024
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := blockCount }
  block := { x := 256 }
  sharedMemoryBytes := (8 * 4).toUSize
  blockArenaBytes := 8 * 1024
  queueCapacity := 1
}

private def waitKernel (label : String) (handle : Cuda.KernelHandle) : IO Unit := do
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"{label} failed: {repr status}"

private def hostOracle : Array Float32 := Id.run do
  let mut base := zero
  for column in [:hiddenSizeNat] do
    base := Cuda.fma (baseWeightValue column) (inputValue column) base
  return (List.range qkvSizeNat).map (fun row =>
    Cuda.BFloat16.ofFloat32 (if row % 2 == 0 then base else -base) |>.toFloat32) |>.toArray

private def requireClose (actual : ByteArray) (expected : Array Float32)
    (tolerance : Float32) : IO Unit := do
  let mut maximumError := zero
  let mut maximumIndex := 0
  for index in [:expected.size] do
    let error := Float32.abs (readBFloat actual index - expected[index]!)
    if error > maximumError then
      maximumError := error
      maximumIndex := index
  unless maximumError <= tolerance do
    throw <| IO.userError <| s!"projection differs from host Lean oracle at {maximumIndex}: " ++
      s!"actual={readBFloat actual maximumIndex}, expected={expected[maximumIndex]!}, " ++
      s!"error={maximumError}, tolerance={tolerance}"
  IO.println s!"Qwen3.6 [10240,5120] projection host-oracle max error: {maximumError}"

def main : IO Unit := do
  let stream ← Cuda.Stream.default
  let buffers ← initializeBuffers stream
  stream.synchronize
  waitKernel "Qwen3.6 projection weight initialization" <|
    ← initializeProjectionWeights.launch initializerConfig qkvSize hiddenSize buffers.weight
  waitKernel "Qwen3.6 separate projection" <|
    ← qwen36ProjectionStep.launch launchConfig projectionItem buffers.input buffers.weight
      buffers.separateOutput
  let handle ← qwen36Projection.start persistentConfig buffers.input buffers.weight
    buffers.persistentOutput buffers.counts
  qwen36Projection.enqueue handle projectionItem
  handle.shutdown
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"Qwen3.6 projection megakernel failed: {repr status}"
  let separate ← buffers.separateOutput.copyTo stream
  let persistent ← buffers.persistentOutput.copyTo stream
  unless separate == persistent do
    throw <| IO.userError "persistent and separate Qwen3.6 projection results differ"
  let counts ← buffers.counts.copyTo stream
  unless readUInt32 counts 0 == 1 do
    throw <| IO.userError "Qwen3.6 projection descriptor did not execute exactly once"
  requireClose persistent hostOracle (Float32.ofBits 0x3a800000)
  IO.println "Lean Qwen3.6 exact-shape decode projection megakernel ok"

end Qwen36Projection

def main : IO Unit := Qwen36Projection.main
