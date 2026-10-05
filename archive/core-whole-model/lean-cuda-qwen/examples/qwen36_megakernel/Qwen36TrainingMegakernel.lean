/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda

/-!
# Qwen3.6-27B recurrent Gated DeltaNet training VJP

This executable differentiates one exact-geometry recurrent Qwen3.6 Gated DeltaNet step. A
cooperative-grid persistent worker consumes a forward descriptor and then a backward descriptor
without ending the launch. The backward phase returns gradients for compact query/key inputs,
value, log-decay, beta, and the previous recurrent state.

The persistent result must match ordinary forward/backward launches and an independent host Lean
analytic replay. This is a training component, not yet a full Qwen train step: projection VJPs,
causal-convolution backward, gated RMSNorm, MLP/full-attention backward, loss, and parameter
updates remain outside this slice.
-/

namespace Qwen36Training

private def tokenCount : Nat := 1
private def keyHeadCount : Nat := 16
private def valueHeadCount : Nat := 48
private def headDimensionNat : Nat := 128
private def stateElementsPerHeadNat : Nat := headDimensionNat * headDimensionNat
private def stateElementCount : Nat := valueHeadCount * stateElementsPerHeadNat
private def outputElementCount : Nat := tokenCount * valueHeadCount * headDimensionNat
private def compactElementCount : Nat := tokenCount * keyHeadCount * headDimensionNat

private def keyHeads : UInt32 := keyHeadCount.toUInt32
private def valueHeads : UInt32 := valueHeadCount.toUInt32
private def headDimension : UInt32 := headDimensionNat.toUInt32
private def valueHeadsPerKeyHead : UInt32 := valueHeads / keyHeads
private def stateElementsPerHead : UInt32 := headDimension * headDimension
private def stateElements : UInt32 := valueHeads * stateElementsPerHead

private def normalizationEpsilon : Float32 := Float32.ofBits 0x358637bd
private def queryScale : Float32 := Float32.ofBits 0x3db504f3

private def reductionBytes : Nat := 8 * 4
private def vectorBytes : Nat := headDimensionNat * 4
private def queryOffset : Nat := reductionBytes
private def keyOffset : Nat := queryOffset + vectorBytes
private def deltaOffset : Nat := keyOffset + vectorBytes
private def auxiliaryOffset : Nat := deltaOffset + vectorBytes
private def deltaGradientOffset : Nat := auxiliaryOffset + vectorBytes
private def queryGradientOffset : Nat := deltaGradientOffset + vectorBytes
private def keyGradientOffset : Nat := queryGradientOffset + vectorBytes
private def trainingSharedBytes : Nat := keyGradientOffset + vectorBytes

private def forwardPhase : UInt32 := 0
private def backwardPhase : UInt32 := 1

structure TrainItem where
  token : UInt32
  phase : UInt32
  deriving Cuda.POD

@[struct] structure Buffers where
  query : Cuda.DevicePtr Cuda.BFloat16
  key : Cuda.DevicePtr Cuda.BFloat16
  value : Cuda.DevicePtr Cuda.BFloat16
  decayLog : Cuda.DevicePtr Float32
  beta : Cuda.DevicePtr Float32
  stateCheckpoints : Cuda.DevicePtr Float32
  output : Cuda.DevicePtr Float32
  outputGradient : Cuda.DevicePtr Float32
  stateGradients : Cuda.DevicePtr Float32
  stateAdjoint : Cuda.DevicePtr Float32
  queryGradient : Cuda.DevicePtr Float32
  keyGradient : Cuda.DevicePtr Float32
  valueGradient : Cuda.DevicePtr Float32
  decayLogGradient : Cuda.DevicePtr Float32
  betaGradient : Cuda.DevicePtr Float32

@[always_inline]
private def compactIndex (token compactHead channel : UInt32) : UInt32 :=
  (token * keyHeads + compactHead) * headDimension + channel

@[always_inline]
private def valueIndex (token head channel : UInt32) : UInt32 :=
  (token * valueHeads + head) * headDimension + channel

@[always_inline]
private def stateIndex (head row column : UInt32) : UInt32 :=
  (head * headDimension + row) * headDimension + column

@[always_inline]
private def stateLinearIndex (head index : UInt32) : UInt32 :=
  head * stateElementsPerHead + index

@[always_inline]
private def checkpointIndex (checkpoint head index : UInt32) : UInt32 :=
  checkpoint * stateElements + stateLinearIndex head index

@[always_inline]
private def loadProjected (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) : Cuda.DeviceM Float32 := do
  return (← Cuda.loadBFloat16 pointer index.toUSize).toFloat32

@[always_inline]
private partial def copyCheckpoint (checkpoints : Cuda.DevicePtr Float32)
    (source destination head index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < stateElementsPerHead then
    let value ← Cuda.loadFloat32 checkpoints (checkpointIndex source head index).toUSize
    Cuda.storeFloat32 checkpoints (checkpointIndex destination head index).toUSize value
    copyCheckpoint checkpoints source destination head (index + stride) stride

@[always_inline]
private partial def decayCheckpoint (checkpoints : Cuda.DevicePtr Float32)
    (checkpoint head index stride : UInt32) (decay : Float32) : Cuda.DeviceM Unit := do
  if index < stateElementsPerHead then
    let address := checkpointIndex checkpoint head index
    let previous ← Cuda.loadFloat32 checkpoints address.toUSize
    Cuda.storeFloat32 checkpoints address.toUSize (previous * decay)
    decayCheckpoint checkpoints checkpoint head (index + stride) stride decay

@[always_inline]
private partial def checkpointKeyDot (checkpoints keyShared : Cuda.DevicePtr Float32)
    (checkpoint head column row : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < headDimension then
    let stateValue ← Cuda.loadFloat32 checkpoints
      (checkpointIndex checkpoint head (row * headDimension + column)).toUSize
    let keyValue ← Cuda.loadFloat32 keyShared row.toUSize
    checkpointKeyDot checkpoints keyShared checkpoint head column (row + 1)
      (Cuda.fma stateValue keyValue accumulator)
  else
    return accumulator

@[always_inline]
private partial def updateCheckpoint (checkpoints keyShared deltaShared : Cuda.DevicePtr Float32)
    (checkpoint head index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < stateElementsPerHead then
    let row := index / headDimension
    let column := index % headDimension
    let address := checkpointIndex checkpoint head index
    let previous ← Cuda.loadFloat32 checkpoints address.toUSize
    let keyValue ← Cuda.loadFloat32 keyShared row.toUSize
    let delta ← Cuda.loadFloat32 deltaShared column.toUSize
    Cuda.storeFloat32 checkpoints address.toUSize (Cuda.fma keyValue delta previous)
    updateCheckpoint checkpoints keyShared deltaShared checkpoint head (index + stride) stride

@[always_inline]
private partial def checkpointQueryDot (checkpoints queryShared : Cuda.DevicePtr Float32)
    (checkpoint head column row : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < headDimension then
    let stateValue ← Cuda.loadFloat32 checkpoints
      (checkpointIndex checkpoint head (row * headDimension + column)).toUSize
    let queryValue ← Cuda.loadFloat32 queryShared row.toUSize
    checkpointQueryDot checkpoints queryShared checkpoint head column (row + 1)
      (Cuda.fma stateValue queryValue accumulator)
  else
    return accumulator

@[always_inline, convergent]
private def normalizeQueryKey (token head thread : UInt32) (buffers : Buffers)
    (reduction : Cuda.Collective.BlockScratch 8)
    (queryShared keyShared : Cuda.DevicePtr Float32) :
    Cuda.DeviceM (Float32 × Float32) := do
  let compactHead := head / valueHeadsPerKeyHead
  let queryValue ← if thread < headDimension then
      loadProjected buffers.query (compactIndex token compactHead thread)
    else
      pure (Float32.ofBits 0)
  let keyValue ← if thread < headDimension then
      loadProjected buffers.key (compactIndex token compactHead thread)
    else
      pure (Float32.ofBits 0)
  let querySquareSum ← Cuda.Collective.blockSum reduction (queryValue * queryValue)
  let keySquareSum ← Cuda.Collective.blockSum reduction (keyValue * keyValue)
  let queryInverse ← Cuda.fastRsqrt (querySquareSum + normalizationEpsilon)
  let keyInverse ← Cuda.fastRsqrt (keySquareSum + normalizationEpsilon)
  if thread < headDimension then
    Cuda.storeFloat32 queryShared thread.toUSize (queryValue * queryInverse * queryScale)
    Cuda.storeFloat32 keyShared thread.toUSize (keyValue * keyInverse)
  Cuda.blockSync
  return (queryInverse, keyInverse)

@[always_inline, convergent]
private def runForward (token : UInt32) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let head ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let reduction : Cuda.Collective.BlockScratch 8 := shared
  let queryShared ← Cuda.dynamicShared (α := Float32) queryOffset.toUSize
  let keyShared ← Cuda.dynamicShared (α := Float32) keyOffset.toUSize
  let deltaShared ← Cuda.dynamicShared (α := Float32) deltaOffset.toUSize
  discard <| normalizeQueryKey token head thread buffers reduction queryShared keyShared
  copyCheckpoint buffers.stateCheckpoints token (token + 1) head thread 256
  Cuda.blockSync
  if thread == 0 then
    let decay ← Cuda.fastExp
      (← Cuda.loadFloat32 buffers.decayLog (token * valueHeads + head).toUSize)
    Cuda.storeFloat32 reduction 0 decay
  Cuda.blockSync
  let decay ← Cuda.loadFloat32 reduction 0
  decayCheckpoint buffers.stateCheckpoints (token + 1) head thread 256 decay
  Cuda.blockSync
  if thread < headDimension then
    let memory ← checkpointKeyDot buffers.stateCheckpoints keyShared (token + 1) head thread 0
      (Float32.ofBits 0)
    let value ← loadProjected buffers.value (valueIndex token head thread)
    let beta ← Cuda.loadFloat32 buffers.beta (token * valueHeads + head).toUSize
    Cuda.storeFloat32 deltaShared thread.toUSize ((value - memory) * beta)
  Cuda.blockSync
  updateCheckpoint buffers.stateCheckpoints keyShared deltaShared (token + 1) head thread 256
  Cuda.blockSync
  if thread < headDimension then
    let result ← checkpointQueryDot buffers.stateCheckpoints queryShared (token + 1) head thread 0
      (Float32.ofBits 0)
    Cuda.storeFloat32 buffers.output (valueIndex token head thread).toUSize result

@[always_inline]
private partial def buildStateAdjoint (token head index stride : UInt32) (buffers : Buffers)
    (queryShared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  if index < stateElementsPerHead then
    let row := index / headDimension
    let column := index % headDimension
    let stateGradient ← Cuda.loadFloat32 buffers.stateGradients
      (checkpointIndex (token + 1) head index).toUSize
    let query ← Cuda.loadFloat32 queryShared row.toUSize
    let outputGradient ← Cuda.loadFloat32 buffers.outputGradient
      (valueIndex token head column).toUSize
    Cuda.storeFloat32 buffers.stateAdjoint (stateLinearIndex head index).toUSize
      (Cuda.fma query outputGradient stateGradient)
    buildStateAdjoint token head (index + stride) stride buffers queryShared

@[always_inline]
private partial def deltaAdjointDot (head column row : UInt32) (buffers : Buffers)
    (keyShared : Cuda.DevicePtr Float32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < headDimension then
    let key ← Cuda.loadFloat32 keyShared row.toUSize
    let stateAdjoint ← Cuda.loadFloat32 buffers.stateAdjoint
      (stateIndex head row column).toUSize
    deltaAdjointDot head column (row + 1) buffers keyShared
      (Cuda.fma key stateAdjoint accumulator)
  else
    return accumulator

@[always_inline]
private partial def queryAdjointDot (token head row column : UInt32) (buffers : Buffers)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < headDimension then
    let state ← Cuda.loadFloat32 buffers.stateCheckpoints
      (checkpointIndex (token + 1) head (row * headDimension + column)).toUSize
    let outputGradient ← Cuda.loadFloat32 buffers.outputGradient
      (valueIndex token head column).toUSize
    queryAdjointDot token head row (column + 1) buffers
      (Cuda.fma state outputGradient accumulator)
  else
    return accumulator

@[always_inline]
private partial def keyAdjointDot (token head row column : UInt32) (buffers : Buffers)
    (deltaShared deltaGradientShared : Cuda.DevicePtr Float32) (decay beta : Float32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < headDimension then
    let delta ← Cuda.loadFloat32 deltaShared column.toUSize
    let stateAdjoint ← Cuda.loadFloat32 buffers.stateAdjoint
      (stateIndex head row column).toUSize
    let stateBefore ← Cuda.loadFloat32 buffers.stateCheckpoints
      (checkpointIndex token head (row * headDimension + column)).toUSize
    let deltaGradient ← Cuda.loadFloat32 deltaGradientShared column.toUSize
    let memoryGradient := -beta * deltaGradient
    let accumulator := Cuda.fma delta stateAdjoint accumulator
    keyAdjointDot token head row (column + 1) buffers deltaShared deltaGradientShared decay beta
      (Cuda.fma (stateBefore * decay) memoryGradient accumulator)
  else
    return accumulator

@[always_inline]
private partial def writePreviousStateGradient (token head index stride : UInt32)
    (buffers : Buffers) (keyShared deltaGradientShared : Cuda.DevicePtr Float32)
    (decay beta accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < stateElementsPerHead then
    let row := index / headDimension
    let column := index % headDimension
    let stateAdjoint ← Cuda.loadFloat32 buffers.stateAdjoint
      (stateLinearIndex head index).toUSize
    let key ← Cuda.loadFloat32 keyShared row.toUSize
    let deltaGradient ← Cuda.loadFloat32 deltaGradientShared column.toUSize
    let previousAdjoint := Cuda.fma key (-beta * deltaGradient) stateAdjoint
    Cuda.storeFloat32 buffers.stateGradients (checkpointIndex token head index).toUSize
      (previousAdjoint * decay)
    let stateBefore ← Cuda.loadFloat32 buffers.stateCheckpoints
      (checkpointIndex token head index).toUSize
    writePreviousStateGradient token head (index + stride) stride buffers keyShared
      deltaGradientShared decay beta (Cuda.fma stateBefore previousAdjoint accumulator)
  else
    return accumulator

@[always_inline, convergent]
private def runBackward (token : UInt32) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let head ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let compactHead := head / valueHeadsPerKeyHead
  let reduction : Cuda.Collective.BlockScratch 8 := shared
  let queryShared ← Cuda.dynamicShared (α := Float32) queryOffset.toUSize
  let keyShared ← Cuda.dynamicShared (α := Float32) keyOffset.toUSize
  let deltaShared ← Cuda.dynamicShared (α := Float32) deltaOffset.toUSize
  let auxiliaryShared ← Cuda.dynamicShared (α := Float32) auxiliaryOffset.toUSize
  let deltaGradientShared ← Cuda.dynamicShared (α := Float32) deltaGradientOffset.toUSize
  let queryGradientShared ← Cuda.dynamicShared (α := Float32) queryGradientOffset.toUSize
  let keyGradientShared ← Cuda.dynamicShared (α := Float32) keyGradientOffset.toUSize
  let inverses ← normalizeQueryKey token head thread buffers reduction queryShared keyShared
  let queryInverse := inverses.1
  let keyInverse := inverses.2
  if thread == 0 then
    let decay ← Cuda.fastExp
      (← Cuda.loadFloat32 buffers.decayLog (token * valueHeads + head).toUSize)
    Cuda.storeFloat32 reduction 0 decay
  Cuda.blockSync
  let decay ← Cuda.loadFloat32 reduction 0
  let beta ← Cuda.loadFloat32 buffers.beta (token * valueHeads + head).toUSize

  if thread < headDimension then
    let memory ← checkpointKeyDot buffers.stateCheckpoints keyShared token head thread 0
      (Float32.ofBits 0)
    let memory := memory * decay
    let value ← loadProjected buffers.value (valueIndex token head thread)
    let residual := value - memory
    Cuda.storeFloat32 auxiliaryShared thread.toUSize residual
    Cuda.storeFloat32 deltaShared thread.toUSize (residual * beta)
  Cuda.blockSync

  buildStateAdjoint token head thread 256 buffers queryShared
  Cuda.blockSync
  if thread < headDimension then
    let gradient ← deltaAdjointDot head thread 0 buffers keyShared (Float32.ofBits 0)
    Cuda.storeFloat32 deltaGradientShared thread.toUSize gradient
  Cuda.blockSync

  let betaContribution ← if thread < headDimension then
      pure ((← Cuda.loadFloat32 deltaGradientShared thread.toUSize) *
        (← Cuda.loadFloat32 auxiliaryShared thread.toUSize))
    else
      pure (Float32.ofBits 0)
  let betaGradient ← Cuda.Collective.blockSum reduction betaContribution
  if thread == 0 then
    Cuda.storeFloat32 buffers.betaGradient (token * valueHeads + head).toUSize betaGradient
  if thread < headDimension then
    let deltaGradient ← Cuda.loadFloat32 deltaGradientShared thread.toUSize
    Cuda.storeFloat32 buffers.valueGradient (valueIndex token head thread).toUSize
      (beta * deltaGradient)
    let queryGradient ← queryAdjointDot token head thread 0 buffers (Float32.ofBits 0)
    Cuda.storeFloat32 queryGradientShared thread.toUSize queryGradient
    let keyGradient ← keyAdjointDot token head thread 0 buffers deltaShared deltaGradientShared
      decay beta (Float32.ofBits 0)
    Cuda.storeFloat32 keyGradientShared thread.toUSize keyGradient
  Cuda.blockSync

  let stateContribution ← writePreviousStateGradient token head thread 256 buffers keyShared
    deltaGradientShared decay beta (Float32.ofBits 0)
  let decayGradient ← Cuda.Collective.blockSum reduction stateContribution
  if thread == 0 then
    Cuda.storeFloat32 buffers.decayLogGradient (token * valueHeads + head).toUSize
      (decay * decayGradient)

  let queryNormGradient ← if thread < headDimension then
      pure ((← Cuda.loadFloat32 queryGradientShared thread.toUSize) * queryScale)
    else
      pure (Float32.ofBits 0)
  let queryNorm ← if thread < headDimension then
      Cuda.loadFloat32 queryShared thread.toUSize
    else
      pure (Float32.ofBits 0)
  let queryDot ← Cuda.Collective.blockSum reduction (queryNormGradient * (queryNorm / queryScale))
  if thread < headDimension then
    let rawGradient := queryInverse *
      (queryNormGradient - (queryNorm / queryScale) * queryDot)
    discard <| Cuda.atomicAddFloat32
      (buffers.queryGradient + (compactIndex token compactHead thread).toUSize * 4) rawGradient

  let keyNormGradient ← if thread < headDimension then
      Cuda.loadFloat32 keyGradientShared thread.toUSize
    else
      pure (Float32.ofBits 0)
  let keyNorm ← if thread < headDimension then
      Cuda.loadFloat32 keyShared thread.toUSize
    else
      pure (Float32.ofBits 0)
  let keyDot ← Cuda.Collective.blockSum reduction (keyNormGradient * keyNorm)
  if thread < headDimension then
    let rawGradient := keyInverse * (keyNormGradient - keyNorm * keyDot)
    discard <| Cuda.atomicAddFloat32
      (buffers.keyGradient + (compactIndex token compactHead thread).toUSize * 4) rawGradient

@[cuda_kernel]
def qwen36TrainForwardStep (token : UInt32)
    (query key value : Cuda.DevicePtr Cuda.BFloat16)
    (decayLog beta stateCheckpoints output outputGradient stateGradients stateAdjoint
      queryGradient keyGradient valueGradient decayLogGradient betaGradient :
      Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runForward token {
    query, key, value, decayLog, beta, stateCheckpoints, output, outputGradient, stateGradients,
    stateAdjoint, queryGradient, keyGradient, valueGradient, decayLogGradient, betaGradient
  } shared

@[cuda_kernel]
def qwen36TrainBackwardStep (token : UInt32)
    (query key value : Cuda.DevicePtr Cuda.BFloat16)
    (decayLog beta stateCheckpoints output outputGradient stateGradients stateAdjoint
      queryGradient keyGradient valueGradient decayLogGradient betaGradient :
      Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runBackward token {
    query, key, value, decayLog, beta, stateCheckpoints, output, outputGradient, stateGradients,
    stateAdjoint, queryGradient, keyGradient, valueGradient, decayLogGradient, betaGradient
  } shared

@[cuda_grid_persistent]
def qwen36Train (item : TrainItem)
    (query key value : Cuda.DevicePtr Cuda.BFloat16)
    (decayLog beta stateCheckpoints output outputGradient stateGradients stateAdjoint
      queryGradient keyGradient valueGradient decayLogGradient betaGradient :
      Cuda.DevicePtr Float32) (counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  let buffers : Buffers := {
    query, key, value, decayLog, beta, stateCheckpoints, output, outputGradient, stateGradients,
    stateAdjoint, queryGradient, keyGradient, valueGradient, decayLogGradient, betaGradient
  }
  if item.phase == forwardPhase then
    runForward item.token buffers shared
  else
    runBackward item.token buffers shared
  Cuda.gridSync
  if (← Cuda.blockIdxX) == 0 && (← Cuda.threadIdxX) == 0 then
    Cuda.atomicAddUInt32At_ counts (item.phase * tokenCount.toUInt32 + item.token) 1

private structure InputBuffers where
  query : Cuda.Buffer Cuda.BFloat16
  key : Cuda.Buffer Cuda.BFloat16
  value : Cuda.Buffer Cuda.BFloat16
  decayLog : Cuda.Buffer Float32
  beta : Cuda.Buffer Float32

private structure RunBuffers where
  stateCheckpoints : Cuda.Buffer Float32
  output : Cuda.Buffer Float32
  outputGradient : Cuda.Buffer Float32
  stateGradients : Cuda.Buffer Float32
  stateAdjoint : Cuda.Buffer Float32
  queryGradient : Cuda.Buffer Float32
  keyGradient : Cuda.Buffer Float32
  valueGradient : Cuda.Buffer Float32
  decayLogGradient : Cuda.Buffer Float32
  betaGradient : Cuda.Buffer Float32
  counts : Cuda.Buffer UInt32

private def pushUInt16 (bytes : ByteArray) (value : UInt16) : ByteArray :=
  (bytes.push value.toUInt8).push (value >>> 8).toUInt8

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def readUInt32 (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def readFloat32 (bytes : ByteArray) (index : Nat) : Float32 :=
  Float32.ofBits (readUInt32 bytes index)

private def zeroBytes (count : Nat) : ByteArray :=
  List.replicate count 0 |>.toByteArray

private def makeBFloat16Data (count : Nat) (value : Nat → Float32) : ByteArray :=
  (List.range count).foldl (init := ByteArray.empty) fun bytes index =>
    pushUInt16 bytes (Cuda.BFloat16.ofFloat32 (value index)).bits

private def makeFloat32Data (count : Nat) (value : Nat → Float32) : ByteArray :=
  (List.range count).foldl (init := ByteArray.empty) fun bytes index =>
    pushUInt32 bytes (value index).toBits

private def signedValue (index modulus center divisor : Nat) : Float32 :=
  (Int32.ofInt (Int.ofNat (index % modulus) - Int.ofNat center)).toFloat32 /
    divisor.toUInt32.toFloat32

private def queryValue (head channel : Nat) : Float32 :=
  signedValue (head * 31 + channel * 7 + 3) 29 14 32

private def keyValue (head channel : Nat) : Float32 :=
  signedValue (head * 43 + channel * 11 + 5) 31 15 36

private def projectedValue (head channel : Nat) : Float32 :=
  signedValue (head * 17 + channel * 13 + 9) 37 18 40

private def decayLogValue (head : Nat) : Float32 :=
  -(head % 7 + 1).toUInt32.toFloat32 / Float32.ofBits 0x42c80000

private def betaValue (head : Nat) : Float32 :=
  Float32.ofBits 0x3e800000 +
    (head % 5).toUInt32.toFloat32 / Float32.ofBits 0x41200000

private def initialStateValue (index : Nat) : Float32 :=
  signedValue (index * 19 + 7) 41 20 512

private def terminalStateGradientValue (index : Nat) : Float32 :=
  signedValue (index * 23 + 11) 43 21 1024

private def outputGradientValue (index : Nat) : Float32 :=
  signedValue (index * 29 + 13) 47 23 128

private def roundedQueryValue (head channel : Nat) : Float32 :=
  (Cuda.BFloat16.ofFloat32 (queryValue head channel)).toFloat32

private def roundedKeyValue (head channel : Nat) : Float32 :=
  (Cuda.BFloat16.ofFloat32 (keyValue head channel)).toFloat32

private def roundedProjectedValue (head channel : Nat) : Float32 :=
  (Cuda.BFloat16.ofFloat32 (projectedValue head channel)).toFloat32

private def initializeInputs (stream : Cuda.Stream) : IO InputBuffers := do
  let query ← Cuda.Buffer.alloc Cuda.BFloat16 compactElementCount.toUSize
  let key ← Cuda.Buffer.alloc Cuda.BFloat16 compactElementCount.toUSize
  let value ← Cuda.Buffer.alloc Cuda.BFloat16 outputElementCount.toUSize
  let decayLog ← Cuda.Buffer.alloc Float32 valueHeadCount.toUSize
  let beta ← Cuda.Buffer.alloc Float32 valueHeadCount.toUSize
  query.copyFrom (makeBFloat16Data compactElementCount fun index =>
    queryValue (index / headDimensionNat) (index % headDimensionNat)) stream
  key.copyFrom (makeBFloat16Data compactElementCount fun index =>
    keyValue (index / headDimensionNat) (index % headDimensionNat)) stream
  value.copyFrom (makeBFloat16Data outputElementCount fun index =>
    projectedValue (index / headDimensionNat) (index % headDimensionNat)) stream
  decayLog.copyFrom (makeFloat32Data valueHeadCount decayLogValue) stream
  beta.copyFrom (makeFloat32Data valueHeadCount betaValue) stream
  return { query, key, value, decayLog, beta }

private def initializeRunBuffers (stream : Cuda.Stream) : IO RunBuffers := do
  let stateCheckpoints ← Cuda.Buffer.alloc Float32 (stateElementCount * 2).toUSize
  let output ← Cuda.Buffer.alloc Float32 outputElementCount.toUSize
  let outputGradient ← Cuda.Buffer.alloc Float32 outputElementCount.toUSize
  let stateGradients ← Cuda.Buffer.alloc Float32 (stateElementCount * 2).toUSize
  let stateAdjoint ← Cuda.Buffer.alloc Float32 stateElementCount.toUSize
  let queryGradient ← Cuda.Buffer.alloc Float32 compactElementCount.toUSize
  let keyGradient ← Cuda.Buffer.alloc Float32 compactElementCount.toUSize
  let valueGradient ← Cuda.Buffer.alloc Float32 outputElementCount.toUSize
  let decayLogGradient ← Cuda.Buffer.alloc Float32 valueHeadCount.toUSize
  let betaGradient ← Cuda.Buffer.alloc Float32 valueHeadCount.toUSize
  let counts ← Cuda.Buffer.alloc UInt32 2
  stateCheckpoints.copyFrom (makeFloat32Data (stateElementCount * 2) fun index =>
    if index < stateElementCount then initialStateValue index else Float32.ofBits 0) stream
  output.copyFrom (zeroBytes (outputElementCount * 4)) stream
  outputGradient.copyFrom (makeFloat32Data outputElementCount outputGradientValue) stream
  stateGradients.copyFrom (makeFloat32Data (stateElementCount * 2) fun index =>
    if index < stateElementCount then Float32.ofBits 0
    else terminalStateGradientValue (index - stateElementCount)) stream
  stateAdjoint.copyFrom (zeroBytes (stateElementCount * 4)) stream
  queryGradient.copyFrom (zeroBytes (compactElementCount * 4)) stream
  keyGradient.copyFrom (zeroBytes (compactElementCount * 4)) stream
  valueGradient.copyFrom (zeroBytes (outputElementCount * 4)) stream
  decayLogGradient.copyFrom (zeroBytes (valueHeadCount * 4)) stream
  betaGradient.copyFrom (zeroBytes (valueHeadCount * 4)) stream
  counts.copyFrom (zeroBytes 8) stream
  return {
    stateCheckpoints, output, outputGradient, stateGradients, stateAdjoint, queryGradient,
    keyGradient, valueGradient, decayLogGradient, betaGradient, counts
  }

private def launchConfig : Cuda.LaunchConfig := {
  grid := { x := valueHeads }
  block := { x := 256 }
  sharedMemoryBytes := trainingSharedBytes.toUSize
  blockArenaBytes := 16 * 1024
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := valueHeads }
  block := { x := 256 }
  sharedMemoryBytes := trainingSharedBytes.toUSize
  blockArenaBytes := 16 * 1024
  queueCapacity := 2
}

private def waitKernel (label : String) (handle : Cuda.KernelHandle) : IO Unit := do
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"{label} failed: {repr status}"

private def runSeparate (inputs : InputBuffers) (buffers : RunBuffers) : IO Unit := do
  waitKernel "Qwen3.6 recurrent training forward" <| ← qwen36TrainForwardStep.launch launchConfig 0
    inputs.query inputs.key inputs.value inputs.decayLog inputs.beta buffers.stateCheckpoints
    buffers.output buffers.outputGradient buffers.stateGradients buffers.stateAdjoint
    buffers.queryGradient buffers.keyGradient buffers.valueGradient buffers.decayLogGradient
    buffers.betaGradient
  waitKernel "Qwen3.6 recurrent training backward" <| ← qwen36TrainBackwardStep.launch launchConfig 0
    inputs.query inputs.key inputs.value inputs.decayLog inputs.beta buffers.stateCheckpoints
    buffers.output buffers.outputGradient buffers.stateGradients buffers.stateAdjoint
    buffers.queryGradient buffers.keyGradient buffers.valueGradient buffers.decayLogGradient
    buffers.betaGradient

private def runPersistent (inputs : InputBuffers) (buffers : RunBuffers) : IO Unit := do
  let handle ← qwen36Train.start persistentConfig inputs.query inputs.key inputs.value
    inputs.decayLog inputs.beta buffers.stateCheckpoints buffers.output buffers.outputGradient
    buffers.stateGradients buffers.stateAdjoint buffers.queryGradient buffers.keyGradient
    buffers.valueGradient buffers.decayLogGradient buffers.betaGradient buffers.counts
  qwen36Train.enqueue handle { token := 0, phase := forwardPhase }
  qwen36Train.enqueue handle { token := 0, phase := backwardPhase }
  handle.shutdown
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"Qwen3.6 recurrent training megakernel failed: {repr status}"

private structure Oracle where
  stateCheckpoints : Array Float32
  output : Array Float32
  stateGradients : Array Float32
  queryGradient : Array Float32
  keyGradient : Array Float32
  valueGradient : Array Float32
  decayLogGradient : Array Float32
  betaGradient : Array Float32

private def hostOracle : Oracle := Id.run do
  let mut checkpoints := Array.replicate (stateElementCount * 2) (Float32.ofBits 0)
  let mut stateGradients := Array.replicate (stateElementCount * 2) (Float32.ofBits 0)
  let mut output := Array.replicate outputElementCount (Float32.ofBits 0)
  let mut queryGradient := Array.replicate compactElementCount (Float32.ofBits 0)
  let mut keyGradient := Array.replicate compactElementCount (Float32.ofBits 0)
  let mut valueGradient := Array.replicate outputElementCount (Float32.ofBits 0)
  let mut decayLogGradient := Array.replicate valueHeadCount (Float32.ofBits 0)
  let mut betaGradient := Array.replicate valueHeadCount (Float32.ofBits 0)
  for index in [:stateElementCount] do
    checkpoints := checkpoints.set! index (initialStateValue index)
    stateGradients := stateGradients.set! (stateElementCount + index)
      (terminalStateGradientValue index)
  for head in [:valueHeadCount] do
    let compactHead := head / (valueHeadCount / keyHeadCount)
    let mut querySquareSum := Float32.ofBits 0
    let mut keySquareSum := Float32.ofBits 0
    for channel in [:headDimensionNat] do
      let query := roundedQueryValue compactHead channel
      let key := roundedKeyValue compactHead channel
      querySquareSum := querySquareSum + query * query
      keySquareSum := keySquareSum + key * key
    let queryInverse := Float32.ofBits 0x3f800000 /
      Float32.sqrt (querySquareSum + normalizationEpsilon)
    let keyInverse := Float32.ofBits 0x3f800000 /
      Float32.sqrt (keySquareSum + normalizationEpsilon)
    let queryNormalized := (List.range headDimensionNat).map (fun channel =>
      roundedQueryValue compactHead channel * queryInverse) |>.toArray
    let keyNormalized := (List.range headDimensionNat).map (fun channel =>
      roundedKeyValue compactHead channel * keyInverse) |>.toArray
    let queryScaled := queryNormalized.map fun value => value * queryScale
    let decay := Float32.exp (decayLogValue head)
    let beta := betaValue head
    let headBase := head * stateElementsPerHeadNat
    let nextBase := stateElementCount + headBase
    for index in [:stateElementsPerHeadNat] do
      checkpoints := checkpoints.set! (nextBase + index)
        (checkpoints[headBase + index]! * decay)
    let mut residual := Array.replicate headDimensionNat (Float32.ofBits 0)
    let mut delta := Array.replicate headDimensionNat (Float32.ofBits 0)
    for column in [:headDimensionNat] do
      let mut memory := Float32.ofBits 0
      for row in [:headDimensionNat] do
        memory := Cuda.fma checkpoints[nextBase + row * headDimensionNat + column]!
          keyNormalized[row]! memory
      let valueResidual := roundedProjectedValue head column - memory
      residual := residual.set! column valueResidual
      delta := delta.set! column (valueResidual * beta)
    for row in [:headDimensionNat] do
      for column in [:headDimensionNat] do
        let index := nextBase + row * headDimensionNat + column
        checkpoints := checkpoints.set! index
          (Cuda.fma keyNormalized[row]! delta[column]! checkpoints[index]!)
    for column in [:headDimensionNat] do
      let mut result := Float32.ofBits 0
      for row in [:headDimensionNat] do
        result := Cuda.fma checkpoints[nextBase + row * headDimensionNat + column]!
          queryScaled[row]! result
      output := output.set! (head * headDimensionNat + column) result

    let mut stateAdjoint := Array.replicate stateElementsPerHeadNat (Float32.ofBits 0)
    for row in [:headDimensionNat] do
      for column in [:headDimensionNat] do
        let index := row * headDimensionNat + column
        stateAdjoint := stateAdjoint.set! index
          (terminalStateGradientValue (headBase + index) +
            queryScaled[row]! * outputGradientValue (head * headDimensionNat + column))
    let mut deltaGradient := Array.replicate headDimensionNat (Float32.ofBits 0)
    for column in [:headDimensionNat] do
      let mut gradient := Float32.ofBits 0
      for row in [:headDimensionNat] do
        gradient := Cuda.fma keyNormalized[row]!
          stateAdjoint[row * headDimensionNat + column]! gradient
      deltaGradient := deltaGradient.set! column gradient
      valueGradient := valueGradient.set! (head * headDimensionNat + column) (beta * gradient)
      betaGradient := betaGradient.set! head
        (betaGradient[head]! + gradient * residual[column]!)
    let mut queryScaledGradient := Array.replicate headDimensionNat (Float32.ofBits 0)
    let mut keyNormalizedGradient := Array.replicate headDimensionNat (Float32.ofBits 0)
    for row in [:headDimensionNat] do
      let mut queryGradientValue := Float32.ofBits 0
      let mut keyGradientValue := Float32.ofBits 0
      for column in [:headDimensionNat] do
        queryGradientValue := Cuda.fma checkpoints[nextBase + row * headDimensionNat + column]!
          (outputGradientValue (head * headDimensionNat + column)) queryGradientValue
        keyGradientValue := Cuda.fma delta[column]!
          stateAdjoint[row * headDimensionNat + column]! keyGradientValue
        keyGradientValue := Cuda.fma (checkpoints[headBase + row * headDimensionNat + column]! * decay)
          (-beta * deltaGradient[column]!) keyGradientValue
      queryScaledGradient := queryScaledGradient.set! row queryGradientValue
      keyNormalizedGradient := keyNormalizedGradient.set! row keyGradientValue
    let mut decayGradient := Float32.ofBits 0
    for row in [:headDimensionNat] do
      for column in [:headDimensionNat] do
        let index := row * headDimensionNat + column
        let previousAdjoint := stateAdjoint[index]! +
          keyNormalized[row]! * (-beta * deltaGradient[column]!)
        stateGradients := stateGradients.set! (headBase + index) (previousAdjoint * decay)
        decayGradient := Cuda.fma checkpoints[headBase + index]! previousAdjoint decayGradient
    decayLogGradient := decayLogGradient.set! head (decay * decayGradient)

    let queryNormGradient := queryScaledGradient.map fun value => value * queryScale
    let mut queryDot := Float32.ofBits 0
    let mut keyDot := Float32.ofBits 0
    for channel in [:headDimensionNat] do
      queryDot := queryDot + queryNormGradient[channel]! * queryNormalized[channel]!
      keyDot := keyDot + keyNormalizedGradient[channel]! * keyNormalized[channel]!
    for channel in [:headDimensionNat] do
      let compactIndex := compactHead * headDimensionNat + channel
      let rawQueryGradient := queryInverse *
        (queryNormGradient[channel]! - queryNormalized[channel]! * queryDot)
      let rawKeyGradient := keyInverse *
        (keyNormalizedGradient[channel]! - keyNormalized[channel]! * keyDot)
      queryGradient := queryGradient.set! compactIndex
        (queryGradient[compactIndex]! + rawQueryGradient)
      keyGradient := keyGradient.set! compactIndex
        (keyGradient[compactIndex]! + rawKeyGradient)
  return {
    stateCheckpoints := checkpoints
    output
    stateGradients
    queryGradient
    keyGradient
    valueGradient
    decayLogGradient
    betaGradient
  }

private def requireEqual (label : String) (actual expected : ByteArray) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError s!"persistent and separate-launch {label} differ"

private def requireRouteClose (label : String) (actual expected : ByteArray) (count : Nat)
    (tolerance : Float32) : IO Unit := do
  let mut maximumError := Float32.ofBits 0
  let mut maximumIndex := 0
  for index in [:count] do
    let error := Float32.abs (readFloat32 actual index - readFloat32 expected index)
    if error > maximumError then
      maximumError := error
      maximumIndex := index
  unless maximumError <= tolerance do
    throw <| IO.userError <| s!"persistent and separate-launch {label} differ at " ++
      s!"{maximumIndex}: persistent={readFloat32 actual maximumIndex}, " ++
      s!"separate={readFloat32 expected maximumIndex}, error={maximumError}, " ++
      s!"tolerance={tolerance}"
  IO.println s!"{label} route max error: {maximumError}"

private def requireClose (label : String) (actual : ByteArray) (expected : Array Float32)
    (tolerance : Float32) : IO Unit := do
  let mut maximumError := Float32.ofBits 0
  let mut maximumIndex := 0
  for index in [:expected.size] do
    let error := Float32.abs (readFloat32 actual index - expected[index]!)
    if error > maximumError then
      maximumError := error
      maximumIndex := index
  unless maximumError <= tolerance do
    throw <| IO.userError <| s!"{label} differs from host Lean oracle at {maximumIndex}: " ++
      s!"actual={readFloat32 actual maximumIndex}, expected={expected[maximumIndex]!}, " ++
      s!"error={maximumError}, tolerance={tolerance}"
  IO.println s!"{label} host-oracle max error: {maximumError}"

private def compareRoutes (persistent separate : RunBuffers) (stream : Cuda.Stream) : IO Unit := do
  requireEqual "training state checkpoints" (← persistent.stateCheckpoints.copyTo stream)
    (← separate.stateCheckpoints.copyTo stream)
  requireEqual "training outputs" (← persistent.output.copyTo stream)
    (← separate.output.copyTo stream)
  requireEqual "training state gradients" (← persistent.stateGradients.copyTo stream)
    (← separate.stateGradients.copyTo stream)
  -- Three value heads atomically accumulate into each compact Q/K head. The routes may choose a
  -- different legal FP32 addition order, so compare those two reductions numerically.
  requireRouteClose "training query gradients" (← persistent.queryGradient.copyTo stream)
    (← separate.queryGradient.copyTo stream) compactElementCount (Float32.ofBits 0x358637bd)
  requireRouteClose "training key gradients" (← persistent.keyGradient.copyTo stream)
    (← separate.keyGradient.copyTo stream) compactElementCount (Float32.ofBits 0x358637bd)
  requireEqual "training value gradients" (← persistent.valueGradient.copyTo stream)
    (← separate.valueGradient.copyTo stream)
  requireEqual "training decay gradients" (← persistent.decayLogGradient.copyTo stream)
    (← separate.decayLogGradient.copyTo stream)
  requireEqual "training beta gradients" (← persistent.betaGradient.copyTo stream)
    (← separate.betaGradient.copyTo stream)

def main : IO Unit := do
  let stream ← Cuda.Stream.default
  let inputs ← initializeInputs stream
  let separate ← initializeRunBuffers stream
  let persistent ← initializeRunBuffers stream
  stream.synchronize
  runSeparate inputs separate
  runPersistent inputs persistent
  compareRoutes persistent separate stream
  let counts ← persistent.counts.copyTo stream
  unless readUInt32 counts 0 == 1 && readUInt32 counts 1 == 1 do
    throw <| IO.userError "Qwen3.6 training phases did not each execute exactly once"
  let oracle := hostOracle
  requireClose "training state checkpoints" (← persistent.stateCheckpoints.copyTo stream)
    oracle.stateCheckpoints (Float32.ofBits 0x3a83126f)
  requireClose "training output" (← persistent.output.copyTo stream) oracle.output
    (Float32.ofBits 0x3b03126f)
  requireClose "training state gradient" (← persistent.stateGradients.copyTo stream)
    oracle.stateGradients (Float32.ofBits 0x3b83126f)
  requireClose "training query gradient" (← persistent.queryGradient.copyTo stream)
    oracle.queryGradient (Float32.ofBits 0x3c03126f)
  requireClose "training key gradient" (← persistent.keyGradient.copyTo stream)
    oracle.keyGradient (Float32.ofBits 0x3c03126f)
  requireClose "training value gradient" (← persistent.valueGradient.copyTo stream)
    oracle.valueGradient (Float32.ofBits 0x3b83126f)
  requireClose "training decay gradient" (← persistent.decayLogGradient.copyTo stream)
    oracle.decayLogGradient (Float32.ofBits 0x3c83126f)
  requireClose "training beta gradient" (← persistent.betaGradient.copyTo stream)
    oracle.betaGradient (Float32.ofBits 0x3c03126f)
  IO.println "Lean Qwen3.6 recurrent forward/backward megakernel VJP ok"

end Qwen36Training

def main : IO Unit := Qwen36Training.main
