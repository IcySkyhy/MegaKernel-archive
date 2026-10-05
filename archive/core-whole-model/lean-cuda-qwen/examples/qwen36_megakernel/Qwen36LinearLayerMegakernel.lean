/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda

/-!
# Qwen3.6-27B linear-attention layer core

This executable extends the recurrent decode kernel across the exact Qwen3.6 causal-convolution,
gate, and gated-RMSNorm boundary. One cooperative-grid resident worker consumes token descriptors
and keeps both the 10,240-channel convolution cache and the 48 x 128 x 128 FP32 recurrent state
on device across tokens.

The inputs are the checkpoint projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, and
`in_proj_a`). The remaining `out_proj` matrix-vector product is intentionally a separate boundary:
this file proves the nonlinear/stateful part of a Qwen3.6 linear-attention layer before the generic
projection engine is attached.
-/

namespace Qwen36LinearLayer

private def tokenCount : Nat := 3
private def keyHeadCount : Nat := 16
private def valueHeadCount : Nat := 48
private def headDimensionNat : Nat := 128
private def keyElementCount : Nat := keyHeadCount * headDimensionNat
private def valueElementCount : Nat := valueHeadCount * headDimensionNat
private def convolutionDimensionNat : Nat := keyElementCount * 2 + valueElementCount
private def convolutionKernelSize : Nat := 4
private def stateElementsPerHeadNat : Nat := headDimensionNat * headDimensionNat
private def stateElementCount : Nat := valueHeadCount * stateElementsPerHeadNat

private def keyHeads : UInt32 := keyHeadCount.toUInt32
private def valueHeads : UInt32 := valueHeadCount.toUInt32
private def headDimension : UInt32 := headDimensionNat.toUInt32
private def keyElements : UInt32 := keyElementCount.toUInt32
private def valueElements : UInt32 := valueElementCount.toUInt32
private def convolutionDimension : UInt32 := convolutionDimensionNat.toUInt32
private def stateElementsPerHead : UInt32 := headDimension * headDimension
private def valueHeadsPerKeyHead : UInt32 := valueHeads / keyHeads

private def normalizationEpsilon : Float32 := Float32.ofBits 0x358637bd
private def queryScale : Float32 := Float32.ofBits 0x3db504f3
private def inverseHeadDimension : Float32 := Float32.ofBits 0x3c000000
private def one : Float32 := Float32.ofBits 0x3f800000
private def zero : Float32 := Float32.ofBits 0

private def reductionBytes : Nat := 8 * 4
private def vectorBytes : Nat := headDimensionNat * 4
private def queryOffset : Nat := reductionBytes
private def keyOffset : Nat := queryOffset + vectorBytes
private def deltaOffset : Nat := keyOffset + vectorBytes
private def sharedBytes : Nat := deltaOffset + vectorBytes

structure TokenItem where
  token : UInt32
  deriving Cuda.POD

@[struct] structure Inputs where
  projectedQkv : Cuda.DevicePtr Cuda.BFloat16
  projectedZ : Cuda.DevicePtr Cuda.BFloat16
  projectedB : Cuda.DevicePtr Cuda.BFloat16
  projectedA : Cuda.DevicePtr Cuda.BFloat16
  convolutionWeight : Cuda.DevicePtr Cuda.BFloat16
  aLog : Cuda.DevicePtr Float32
  dtBias : Cuda.DevicePtr Float32
  normWeight : Cuda.DevicePtr Cuda.BFloat16

@[struct] structure Buffers where
  convolutionState : Cuda.DevicePtr Cuda.BFloat16
  convolvedQkv : Cuda.DevicePtr Cuda.BFloat16
  recurrentState : Cuda.DevicePtr Float32
  recurrentOutput : Cuda.DevicePtr Cuda.BFloat16
  output : Cuda.DevicePtr Cuda.BFloat16

@[always_inline]
private def projectedIndex (token channel : UInt32) : UInt32 :=
  token * convolutionDimension + channel

@[always_inline]
private def valueIndex (token head channel : UInt32) : UInt32 :=
  (token * valueHeads + head) * headDimension + channel

@[always_inline]
private def compactIndex (compactHead channel : UInt32) : UInt32 :=
  compactHead * headDimension + channel

@[always_inline]
private def stateIndex (head row column : UInt32) : UInt32 :=
  (head * headDimension + row) * headDimension + column

@[always_inline]
private def loadBFloat (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) : Cuda.DeviceM Float32 := do
  return (← Cuda.loadBFloat16 pointer index.toUSize).toFloat32

@[always_inline]
private def storeBFloat (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) (value : Float32) : Cuda.DeviceM Unit :=
  Cuda.storeBFloat16 pointer index.toUSize (Cuda.BFloat16.ofFloat32 value)

@[always_inline]
private def deviceSigmoid (value : Float32) : Cuda.DeviceM Float32 := do
  return one / (one + (← Cuda.fastExp (-value)))

@[always_inline]
private def deviceSilu (value : Float32) : Cuda.DeviceM Float32 := do
  return value * (← deviceSigmoid value)

@[always_inline]
private def deviceSoftplus (value : Float32) : Cuda.DeviceM Float32 := do
  if value > zero then
    return value + (← Cuda.fastLog (one + (← Cuda.fastExp (-value))))
  else
    return (← Cuda.fastLog (one + (← Cuda.fastExp value)))

@[always_inline]
private def runConvolution (token block thread : UInt32) (inputs : Inputs)
    (buffers : Buffers) : Cuda.DeviceM Unit := do
  let channel := block * 256 + thread
  if channel < convolutionDimension then
    let stateBase := channel * convolutionKernelSize.toUInt32
    let old1 ← loadBFloat buffers.convolutionState (stateBase + 1)
    let old2 ← loadBFloat buffers.convolutionState (stateBase + 2)
    let old3 ← loadBFloat buffers.convolutionState (stateBase + 3)
    let current ← loadBFloat inputs.projectedQkv (projectedIndex token channel)
    let weight0 ← loadBFloat inputs.convolutionWeight stateBase
    let weight1 ← loadBFloat inputs.convolutionWeight (stateBase + 1)
    let weight2 ← loadBFloat inputs.convolutionWeight (stateBase + 2)
    let weight3 ← loadBFloat inputs.convolutionWeight (stateBase + 3)
    let accumulator := Cuda.fma weight0 old1 zero
    let accumulator := Cuda.fma weight1 old2 accumulator
    let accumulator := Cuda.fma weight2 old3 accumulator
    let accumulator := Cuda.fma weight3 current accumulator
    let activated ← deviceSilu accumulator
    storeBFloat buffers.convolvedQkv channel activated
    storeBFloat buffers.convolutionState stateBase old1
    storeBFloat buffers.convolutionState (stateBase + 1) old2
    storeBFloat buffers.convolutionState (stateBase + 2) old3
    storeBFloat buffers.convolutionState (stateBase + 3) current

@[always_inline]
private partial def decayState (state : Cuda.DevicePtr Float32)
    (head index stride : UInt32) (decay : Float32) : Cuda.DeviceM Unit := do
  if index < stateElementsPerHead then
    let address := head * stateElementsPerHead + index
    let previous ← Cuda.loadFloat32 state address.toUSize
    Cuda.storeFloat32 state address.toUSize (previous * decay)
    decayState state head (index + stride) stride decay

@[always_inline]
private partial def stateKeyDot (state keyShared : Cuda.DevicePtr Float32)
    (head column row : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < headDimension then
    let stateValue ← Cuda.loadFloat32 state (stateIndex head row column).toUSize
    let keyValue ← Cuda.loadFloat32 keyShared row.toUSize
    stateKeyDot state keyShared head column (row + 1)
      (Cuda.fma stateValue keyValue accumulator)
  else
    return accumulator

@[always_inline]
private partial def updateState (state keyShared deltaShared : Cuda.DevicePtr Float32)
    (head index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < stateElementsPerHead then
    let row := index / headDimension
    let column := index % headDimension
    let address := stateIndex head row column
    let previous ← Cuda.loadFloat32 state address.toUSize
    let keyValue ← Cuda.loadFloat32 keyShared row.toUSize
    let delta ← Cuda.loadFloat32 deltaShared column.toUSize
    Cuda.storeFloat32 state address.toUSize (Cuda.fma keyValue delta previous)
    updateState state keyShared deltaShared head (index + stride) stride

@[always_inline]
private partial def stateQueryDot (state queryShared : Cuda.DevicePtr Float32)
    (head column row : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < headDimension then
    let stateValue ← Cuda.loadFloat32 state (stateIndex head row column).toUSize
    let queryValue ← Cuda.loadFloat32 queryShared row.toUSize
    stateQueryDot state queryShared head column (row + 1)
      (Cuda.fma stateValue queryValue accumulator)
  else
    return accumulator

@[always_inline, convergent]
private def runRecurrentAndNorm (token head thread : UInt32) (inputs : Inputs)
    (buffers : Buffers) (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let reduction : Cuda.Collective.BlockScratch 8 := shared
  let queryShared ← Cuda.dynamicShared (α := Float32) queryOffset.toUSize
  let keyShared ← Cuda.dynamicShared (α := Float32) keyOffset.toUSize
  let deltaShared ← Cuda.dynamicShared (α := Float32) deltaOffset.toUSize
  let compactHead := head / valueHeadsPerKeyHead
  let queryValue ← if thread < headDimension then
      loadBFloat buffers.convolvedQkv (compactIndex compactHead thread)
    else
      pure zero
  let keyValue ← if thread < headDimension then
      loadBFloat buffers.convolvedQkv (keyElements + compactIndex compactHead thread)
    else
      pure zero
  let querySquareSum ← Cuda.Collective.blockSum reduction (queryValue * queryValue)
  let keySquareSum ← Cuda.Collective.blockSum reduction (keyValue * keyValue)
  let queryInverse ← Cuda.fastRsqrt (querySquareSum + normalizationEpsilon)
  let keyInverse ← Cuda.fastRsqrt (keySquareSum + normalizationEpsilon)
  if thread < headDimension then
    Cuda.storeFloat32 queryShared thread.toUSize (queryValue * queryInverse * queryScale)
    Cuda.storeFloat32 keyShared thread.toUSize (keyValue * keyInverse)
  Cuda.blockSync

  if thread == 0 then
    let projectedA ← loadBFloat inputs.projectedA (token * valueHeads + head)
    let aLog ← Cuda.loadFloat32 inputs.aLog head.toUSize
    let dtBias ← Cuda.loadFloat32 inputs.dtBias head.toUSize
    let rate ← Cuda.fastExp aLog
    let step ← deviceSoftplus (projectedA + dtBias)
    let decay ← Cuda.fastExp (-(rate * step))
    let projectedB ← loadBFloat inputs.projectedB (token * valueHeads + head)
    let beta ← deviceSigmoid projectedB
    Cuda.storeFloat32 reduction 0 decay
    Cuda.storeFloat32 reduction 1 beta
  Cuda.blockSync
  let decay ← Cuda.loadFloat32 reduction 0
  let beta ← Cuda.loadFloat32 reduction 1
  decayState buffers.recurrentState head thread 256 decay
  Cuda.blockSync
  if thread < headDimension then
    let memory ← stateKeyDot buffers.recurrentState keyShared head thread 0 zero
    let value ← loadBFloat buffers.convolvedQkv (keyElements * 2 + head * headDimension + thread)
    Cuda.storeFloat32 deltaShared thread.toUSize ((value - memory) * beta)
  Cuda.blockSync
  updateState buffers.recurrentState keyShared deltaShared head thread 256
  Cuda.blockSync
  if thread < headDimension then
    let result ← stateQueryDot buffers.recurrentState queryShared head thread 0 zero
    storeBFloat buffers.recurrentOutput (head * headDimension + thread) result
  Cuda.blockSync

  let recurrent ← if thread < headDimension then
      loadBFloat buffers.recurrentOutput (head * headDimension + thread)
    else
      pure zero
  let squareSum ← Cuda.Collective.blockSum reduction (recurrent * recurrent)
  let normInverse ← Cuda.fastRsqrt (squareSum * inverseHeadDimension + normalizationEpsilon)
  if thread < headDimension then
    let weight ← loadBFloat inputs.normWeight thread
    let z ← loadBFloat inputs.projectedZ (valueIndex token head thread)
    let gate ← deviceSilu z
    storeBFloat buffers.output (valueIndex token head thread)
      (recurrent * normInverse * weight * gate)

@[cuda_kernel]
def qwen36LinearPreprocess (token : UInt32)
    (projectedQkv projectedZ projectedB projectedA convolutionWeight :
      Cuda.DevicePtr Cuda.BFloat16)
    (aLog dtBias : Cuda.DevicePtr Float32) (normWeight : Cuda.DevicePtr Cuda.BFloat16)
    (convolutionState convolvedQkv recurrentOutput output : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  runConvolution token (← Cuda.blockIdxX) (← Cuda.threadIdxX) {
    projectedQkv, projectedZ, projectedB, projectedA, convolutionWeight, aLog, dtBias, normWeight
  } { convolutionState, convolvedQkv, recurrentState, recurrentOutput, output }

@[cuda_kernel]
def qwen36LinearRecurrentPost (token : UInt32)
    (projectedQkv projectedZ projectedB projectedA convolutionWeight :
      Cuda.DevicePtr Cuda.BFloat16)
    (aLog dtBias : Cuda.DevicePtr Float32) (normWeight : Cuda.DevicePtr Cuda.BFloat16)
    (convolutionState convolvedQkv recurrentOutput output : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runRecurrentAndNorm token (← Cuda.blockIdxX) (← Cuda.threadIdxX) {
    projectedQkv, projectedZ, projectedB, projectedA, convolutionWeight, aLog, dtBias, normWeight
  } { convolutionState, convolvedQkv, recurrentState, recurrentOutput, output } shared

@[cuda_grid_persistent]
def qwen36LinearLayerToken (item : TokenItem)
    (projectedQkv projectedZ projectedB projectedA convolutionWeight :
      Cuda.DevicePtr Cuda.BFloat16)
    (aLog dtBias : Cuda.DevicePtr Float32) (normWeight : Cuda.DevicePtr Cuda.BFloat16)
    (convolutionState convolvedQkv recurrentOutput output : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32) (counts : Cuda.DevicePtr UInt32) :
    Cuda.DeviceM Unit := do
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let shared ← Cuda.dynamicShared (α := Float32)
  let inputs : Inputs := {
    projectedQkv, projectedZ, projectedB, projectedA, convolutionWeight, aLog, dtBias, normWeight
  }
  let buffers : Buffers := {
    convolutionState, convolvedQkv, recurrentState, recurrentOutput, output
  }
  runConvolution item.token block thread inputs buffers
  Cuda.gridSync
  runRecurrentAndNorm item.token block thread inputs buffers shared
  Cuda.gridSync
  if block == 0 && thread == 0 then
    Cuda.atomicAddUInt32At_ counts item.token 1

private structure InputBuffers where
  projectedQkv : Cuda.Buffer Cuda.BFloat16
  projectedZ : Cuda.Buffer Cuda.BFloat16
  projectedB : Cuda.Buffer Cuda.BFloat16
  projectedA : Cuda.Buffer Cuda.BFloat16
  convolutionWeight : Cuda.Buffer Cuda.BFloat16
  aLog : Cuda.Buffer Float32
  dtBias : Cuda.Buffer Float32
  normWeight : Cuda.Buffer Cuda.BFloat16

private structure RunBuffers where
  convolutionState : Cuda.Buffer Cuda.BFloat16
  convolvedQkv : Cuda.Buffer Cuda.BFloat16
  recurrentState : Cuda.Buffer Float32
  recurrentOutput : Cuda.Buffer Cuda.BFloat16
  output : Cuda.Buffer Cuda.BFloat16
  counts : Cuda.Buffer UInt32

private structure Oracle where
  convolutionState : Array Float32
  recurrentState : Array Float32
  output : Array Float32

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

private def readFloat (bytes : ByteArray) (index : Nat) : Float32 :=
  Float32.ofBits (readUInt32 bytes index)

private def zeroBytes (count : Nat) : ByteArray :=
  List.replicate count 0 |>.toByteArray

private def makeBFloatData (count : Nat) (value : Nat → Float32) : ByteArray :=
  (List.range count).foldl (init := ByteArray.empty) fun bytes index =>
    pushUInt16 bytes (Cuda.BFloat16.ofFloat32 (value index)).bits

private def makeFloatData (count : Nat) (value : Nat → Float32) : ByteArray :=
  (List.range count).foldl (init := ByteArray.empty) fun bytes index =>
    pushUInt32 bytes (value index).toBits

private def signedValue (index modulus center divisor : Nat) : Float32 :=
  (Int32.ofInt (Int.ofNat (index % modulus) - Int.ofNat center)).toFloat32 /
    divisor.toUInt32.toFloat32

private def projectedQkvValue (token channel : Nat) : Float32 :=
  signedValue (token * 101 + channel * 17 + 3) 41 20 32

private def projectedZValue (token index : Nat) : Float32 :=
  signedValue (token * 73 + index * 13 + 5) 37 18 24

private def projectedBValue (token head : Nat) : Float32 :=
  signedValue (token * 29 + head * 7 + 11) 31 15 16

private def projectedAValue (token head : Nat) : Float32 :=
  signedValue (token * 23 + head * 5 + 9) 29 14 20

private def convolutionWeightValue (channel slot : Nat) : Float32 :=
  signedValue (channel * 11 + slot * 19 + 7) 23 11 48

private def initialConvolutionStateValue (channel slot : Nat) : Float32 :=
  signedValue (channel * 5 + slot * 31 + 13) 19 9 64

private def initialRecurrentStateValue (index : Nat) : Float32 :=
  signedValue (index * 19 + 7) 41 20 512

private def aLogValue (head : Nat) : Float32 :=
  Float32.log (Float32.ofBits 0x3f000000 +
    (head % 13 + 1).toUInt32.toFloat32 / Float32.ofBits 0x41000000)

private def dtBiasValue (head : Nat) : Float32 :=
  signedValue (head * 7 + 3) 17 8 12

private def normWeightValue (channel : Nat) : Float32 :=
  Float32.ofBits 0x3f800000 + signedValue (channel * 5 + 1) 13 6 64

private def rounded (value : Float32) : Float32 :=
  (Cuda.BFloat16.ofFloat32 value).toFloat32

private def hostSigmoid (value : Float32) : Float32 :=
  one / (one + Float32.exp (-value))

private def hostSilu (value : Float32) : Float32 :=
  value * hostSigmoid value

private def hostSoftplus (value : Float32) : Float32 :=
  if value > zero then
    value + Float32.log (one + Float32.exp (-value))
  else
    Float32.log (one + Float32.exp value)

private def initializeInputs (stream : Cuda.Stream) : IO InputBuffers := do
  let projectedQkv ← Cuda.Buffer.alloc Cuda.BFloat16
    (tokenCount * convolutionDimensionNat).toUSize
  let projectedZ ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * valueElementCount).toUSize
  let projectedB ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * valueHeadCount).toUSize
  let projectedA ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * valueHeadCount).toUSize
  let convolutionWeight ← Cuda.Buffer.alloc Cuda.BFloat16
    (convolutionDimensionNat * convolutionKernelSize).toUSize
  let aLog ← Cuda.Buffer.alloc Float32 valueHeadCount.toUSize
  let dtBias ← Cuda.Buffer.alloc Float32 valueHeadCount.toUSize
  let normWeight ← Cuda.Buffer.alloc Cuda.BFloat16 headDimensionNat.toUSize
  projectedQkv.copyFrom (makeBFloatData (tokenCount * convolutionDimensionNat) fun index =>
    projectedQkvValue (index / convolutionDimensionNat) (index % convolutionDimensionNat)) stream
  projectedZ.copyFrom (makeBFloatData (tokenCount * valueElementCount) fun index =>
    projectedZValue (index / valueElementCount) (index % valueElementCount)) stream
  projectedB.copyFrom (makeBFloatData (tokenCount * valueHeadCount) fun index =>
    projectedBValue (index / valueHeadCount) (index % valueHeadCount)) stream
  projectedA.copyFrom (makeBFloatData (tokenCount * valueHeadCount) fun index =>
    projectedAValue (index / valueHeadCount) (index % valueHeadCount)) stream
  convolutionWeight.copyFrom
    (makeBFloatData (convolutionDimensionNat * convolutionKernelSize) fun index =>
      convolutionWeightValue (index / convolutionKernelSize) (index % convolutionKernelSize)) stream
  aLog.copyFrom (makeFloatData valueHeadCount aLogValue) stream
  dtBias.copyFrom (makeFloatData valueHeadCount dtBiasValue) stream
  normWeight.copyFrom (makeBFloatData headDimensionNat normWeightValue) stream
  return {
    projectedQkv, projectedZ, projectedB, projectedA, convolutionWeight, aLog, dtBias, normWeight
  }

private def initializeRunBuffers (stream : Cuda.Stream) : IO RunBuffers := do
  let convolutionState ← Cuda.Buffer.alloc Cuda.BFloat16
    (convolutionDimensionNat * convolutionKernelSize).toUSize
  let convolvedQkv ← Cuda.Buffer.alloc Cuda.BFloat16 convolutionDimensionNat.toUSize
  let recurrentState ← Cuda.Buffer.alloc Float32 stateElementCount.toUSize
  let recurrentOutput ← Cuda.Buffer.alloc Cuda.BFloat16 valueElementCount.toUSize
  let output ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * valueElementCount).toUSize
  let counts ← Cuda.Buffer.alloc UInt32 tokenCount.toUSize
  convolutionState.copyFrom
    (makeBFloatData (convolutionDimensionNat * convolutionKernelSize) fun index =>
      initialConvolutionStateValue (index / convolutionKernelSize) (index % convolutionKernelSize))
    stream
  convolvedQkv.copyFrom (zeroBytes (convolutionDimensionNat * 2)) stream
  recurrentState.copyFrom (makeFloatData stateElementCount initialRecurrentStateValue) stream
  recurrentOutput.copyFrom (zeroBytes (valueElementCount * 2)) stream
  output.copyFrom (zeroBytes (tokenCount * valueElementCount * 2)) stream
  counts.copyFrom (zeroBytes (tokenCount * 4)) stream
  return { convolutionState, convolvedQkv, recurrentState, recurrentOutput, output, counts }

private def launchConfig : Cuda.LaunchConfig := {
  grid := { x := valueHeads }
  block := { x := 256 }
  sharedMemoryBytes := sharedBytes.toUSize
  blockArenaBytes := 16 * 1024
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := valueHeads }
  block := { x := 256 }
  sharedMemoryBytes := sharedBytes.toUSize
  blockArenaBytes := 16 * 1024
  queueCapacity := tokenCount.toUSize
}

private def waitKernel (label : String) (handle : Cuda.KernelHandle) : IO Unit := do
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"{label} failed: {repr status}"

private def runSeparate (inputs : InputBuffers) (buffers : RunBuffers) : IO Unit := do
  for token in [:tokenCount] do
    waitKernel "Qwen3.6 causal convolution" <| ← qwen36LinearPreprocess.launch launchConfig
      token.toUInt32 inputs.projectedQkv inputs.projectedZ inputs.projectedB inputs.projectedA
      inputs.convolutionWeight inputs.aLog inputs.dtBias inputs.normWeight buffers.convolutionState
      buffers.convolvedQkv buffers.recurrentOutput buffers.output buffers.recurrentState
    waitKernel "Qwen3.6 recurrent core and gated RMSNorm" <|
      ← qwen36LinearRecurrentPost.launch launchConfig token.toUInt32 inputs.projectedQkv
        inputs.projectedZ inputs.projectedB inputs.projectedA inputs.convolutionWeight inputs.aLog
        inputs.dtBias inputs.normWeight buffers.convolutionState buffers.convolvedQkv
        buffers.recurrentOutput buffers.output buffers.recurrentState

private def runPersistent (inputs : InputBuffers) (buffers : RunBuffers) : IO Unit := do
  let handle ← qwen36LinearLayerToken.start persistentConfig inputs.projectedQkv inputs.projectedZ
    inputs.projectedB inputs.projectedA inputs.convolutionWeight inputs.aLog inputs.dtBias
    inputs.normWeight buffers.convolutionState buffers.convolvedQkv buffers.recurrentOutput
    buffers.output buffers.recurrentState buffers.counts
  for token in [:tokenCount] do
    qwen36LinearLayerToken.enqueue handle { token := token.toUInt32 }
  handle.shutdown
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"Qwen3.6 linear-layer megakernel failed: {repr status}"

private def hostOracle : Oracle := Id.run do
  let mut convolutionState := Array.replicate
    (convolutionDimensionNat * convolutionKernelSize) zero
  let mut recurrentState := Array.replicate stateElementCount zero
  let mut output := Array.replicate (tokenCount * valueElementCount) zero
  for channel in [:convolutionDimensionNat] do
    for slot in [:convolutionKernelSize] do
      convolutionState := convolutionState.set! (channel * convolutionKernelSize + slot)
        (rounded (initialConvolutionStateValue channel slot))
  for index in [:stateElementCount] do
    recurrentState := recurrentState.set! index (initialRecurrentStateValue index)
  for token in [:tokenCount] do
    let mut convolved := Array.replicate convolutionDimensionNat zero
    for channel in [:convolutionDimensionNat] do
      let base := channel * convolutionKernelSize
      let old1 := convolutionState[base + 1]!
      let old2 := convolutionState[base + 2]!
      let old3 := convolutionState[base + 3]!
      let current := rounded (projectedQkvValue token channel)
      let accumulator := Cuda.fma (rounded (convolutionWeightValue channel 0)) old1 zero
      let accumulator := Cuda.fma (rounded (convolutionWeightValue channel 1)) old2 accumulator
      let accumulator := Cuda.fma (rounded (convolutionWeightValue channel 2)) old3 accumulator
      let accumulator := Cuda.fma (rounded (convolutionWeightValue channel 3)) current accumulator
      convolved := convolved.set! channel (rounded (hostSilu accumulator))
      convolutionState := convolutionState.set! base old1
      convolutionState := convolutionState.set! (base + 1) old2
      convolutionState := convolutionState.set! (base + 2) old3
      convolutionState := convolutionState.set! (base + 3) current
    for head in [:valueHeadCount] do
      let compactHead := head / (valueHeadCount / keyHeadCount)
      let mut querySquareSum := zero
      let mut keySquareSum := zero
      for channel in [:headDimensionNat] do
        let query := convolved[compactHead * headDimensionNat + channel]!
        let key := convolved[keyElementCount + compactHead * headDimensionNat + channel]!
        querySquareSum := querySquareSum + query * query
        keySquareSum := keySquareSum + key * key
      let queryInverse := one / Float32.sqrt (querySquareSum + normalizationEpsilon)
      let keyInverse := one / Float32.sqrt (keySquareSum + normalizationEpsilon)
      let query := (List.range headDimensionNat).map (fun channel =>
        convolved[compactHead * headDimensionNat + channel]! * queryInverse * queryScale) |>.toArray
      let key := (List.range headDimensionNat).map (fun channel =>
        convolved[keyElementCount + compactHead * headDimensionNat + channel]! * keyInverse) |>.toArray
      let rate := Float32.exp (aLogValue head)
      let step := hostSoftplus (rounded (projectedAValue token head) + dtBiasValue head)
      let decay := Float32.exp (-(rate * step))
      let beta := hostSigmoid (rounded (projectedBValue token head))
      let headBase := head * stateElementsPerHeadNat
      for index in [:stateElementsPerHeadNat] do
        recurrentState := recurrentState.set! (headBase + index)
          (recurrentState[headBase + index]! * decay)
      let mut delta := Array.replicate headDimensionNat zero
      for column in [:headDimensionNat] do
        let mut memory := zero
        for row in [:headDimensionNat] do
          memory := Cuda.fma recurrentState[headBase + row * headDimensionNat + column]!
            key[row]! memory
        let value := convolved[keyElementCount * 2 + head * headDimensionNat + column]!
        delta := delta.set! column ((value - memory) * beta)
      for row in [:headDimensionNat] do
        for column in [:headDimensionNat] do
          let index := headBase + row * headDimensionNat + column
          recurrentState := recurrentState.set! index
            (Cuda.fma key[row]! delta[column]! recurrentState[index]!)
      let mut recurrentOutput := Array.replicate headDimensionNat zero
      let mut squareSum := zero
      for column in [:headDimensionNat] do
        let mut result := zero
        for row in [:headDimensionNat] do
          result := Cuda.fma recurrentState[headBase + row * headDimensionNat + column]!
            query[row]! result
        let roundedResult := rounded result
        recurrentOutput := recurrentOutput.set! column roundedResult
        squareSum := squareSum + roundedResult * roundedResult
      let normInverse := one /
        Float32.sqrt (squareSum * inverseHeadDimension + normalizationEpsilon)
      for channel in [:headDimensionNat] do
        let z := rounded (projectedZValue token (head * headDimensionNat + channel))
        let result := recurrentOutput[channel]! * normInverse * rounded (normWeightValue channel) *
          hostSilu z
        output := output.set! ((token * valueHeadCount + head) * headDimensionNat + channel)
          (rounded result)
  return { convolutionState, recurrentState, output }

private def requireEqual (label : String) (actual expected : ByteArray) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError s!"persistent and separate-launch {label} differ"

private def requireFloatClose (label : String) (actual : ByteArray) (expected : Array Float32)
    (tolerance : Float32) : IO Unit := do
  let mut maximumError := zero
  let mut maximumIndex := 0
  for index in [:expected.size] do
    let error := Float32.abs (readFloat actual index - expected[index]!)
    if error > maximumError then
      maximumError := error
      maximumIndex := index
  unless maximumError <= tolerance do
    throw <| IO.userError <| s!"{label} differs from host Lean oracle at {maximumIndex}: " ++
      s!"actual={readFloat actual maximumIndex}, expected={expected[maximumIndex]!}, " ++
      s!"error={maximumError}, tolerance={tolerance}"
  IO.println s!"{label} host-oracle max error: {maximumError}"

private def requireBFloatClose (label : String) (actual : ByteArray) (expected : Array Float32)
    (tolerance : Float32) : IO Unit := do
  let mut maximumError := zero
  let mut maximumIndex := 0
  for index in [:expected.size] do
    let error := Float32.abs (readBFloat actual index - expected[index]!)
    if error > maximumError then
      maximumError := error
      maximumIndex := index
  unless maximumError <= tolerance do
    throw <| IO.userError <| s!"{label} differs from host Lean oracle at {maximumIndex}: " ++
      s!"actual={readBFloat actual maximumIndex}, expected={expected[maximumIndex]!}, " ++
      s!"error={maximumError}, tolerance={tolerance}"
  IO.println s!"{label} host-oracle max error: {maximumError}"

def main : IO Unit := do
  let stream ← Cuda.Stream.default
  let inputs ← initializeInputs stream
  let separate ← initializeRunBuffers stream
  let persistent ← initializeRunBuffers stream
  stream.synchronize
  runSeparate inputs separate
  runPersistent inputs persistent
  requireEqual "linear-layer convolution state" (← persistent.convolutionState.copyTo stream)
    (← separate.convolutionState.copyTo stream)
  requireEqual "linear-layer recurrent state" (← persistent.recurrentState.copyTo stream)
    (← separate.recurrentState.copyTo stream)
  requireEqual "linear-layer outputs" (← persistent.output.copyTo stream)
    (← separate.output.copyTo stream)
  let counts ← persistent.counts.copyTo stream
  for token in [:tokenCount] do
    unless readUInt32 counts token == 1 do
      throw <| IO.userError s!"Qwen3.6 linear-layer token {token} did not execute exactly once"
  let oracle := hostOracle
  requireBFloatClose "linear-layer convolution state" (← persistent.convolutionState.copyTo stream)
    oracle.convolutionState (Float32.ofBits 0x3b800000)
  requireFloatClose "linear-layer recurrent state" (← persistent.recurrentState.copyTo stream)
    oracle.recurrentState (Float32.ofBits 0x3c000000)
  requireBFloatClose "linear-layer output" (← persistent.output.copyTo stream) oracle.output
    (Float32.ofBits 0x3b800000)
  IO.println "Lean Qwen3.6 causal-conv/recurrent/gated-RMSNorm megakernel ok"

end Qwen36LinearLayer

def main : IO Unit := Qwen36LinearLayer.main
