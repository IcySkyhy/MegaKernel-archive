/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda
import Qwen36Architecture

/-!
# Qwen3.6-27B real-checkpoint inference megakernel

This is the model-wide, plain-Lean CUDA decode path. The published safetensors shards are copied
without repacking, and queued address-free descriptors keep their original shard-relative offsets.
One cooperative-grid persistent launch executes embedding lookup, all 64 decoder layers, final
RMSNorm, the full LM head, stable cross-entropy, and greedy sampling for the first token position.

The first-position restriction keeps RoPE equal to the identity and full-attention softmax equal to
one while the real-weight, all-layer boundary is established. Recurrent and convolution states plus
full-attention KV caches are nevertheless materialized and updated for the next decode extension.
-/

namespace Qwen36Model

open Qwen36.Architecture
open Qwen36.Checkpoint

private def blockCountNat : Nat := 48
private def descriptorCountNat : Nat := layerCount + 2
private def linearStateElementsPerLayerNat : Nat :=
  linearHeadCount * linearHeadDimension * linearHeadDimension
private def convolutionElementsPerLayerNat : Nat := linearQkvSize * 4
private def fullCacheElementsPerLayerNat : Nat := fullKeyValueProjectionSize

private def hiddenSizeU32 : UInt32 := hiddenSize.toUInt32
private def intermediateSizeU32 : UInt32 := intermediateSize.toUInt32
private def vocabularySizeU32 : UInt32 := vocabularySize.toUInt32
private def blockCount : UInt32 := blockCountNat.toUInt32
private def linearQkvSizeU32 : UInt32 := linearQkvSize.toUInt32
private def linearValueSizeU32 : UInt32 := linearValueSize.toUInt32
private def linearHeads : UInt32 := linearHeadCount.toUInt32
private def linearDimension : UInt32 := linearHeadDimension.toUInt32
private def linearKeyElements : UInt32 := (16 * linearHeadDimension).toUInt32
private def linearStateElementsPerHead : UInt32 := linearDimension * linearDimension
private def fullQueryHeads : UInt32 := 24
private def fullKeyValueHeads : UInt32 := 4
private def fullDimension : UInt32 := fullHeadDimension.toUInt32
private def fullGroups : UInt32 := fullQueryHeads / fullKeyValueHeads

private def zero : Float32 := Float32.ofBits 0
private def one : Float32 := Float32.ofBits 0x3f800000
private def negativeInfinity : Float32 := Float32.ofBits 0xff800000
private def inverseHiddenSize : Float32 := Float32.ofBits 0x394ccccd
private def inverseLinearDimension : Float32 := Float32.ofBits 0x3c000000
private def inverseFullDimension : Float32 := Float32.ofBits 0x3b800000
private def rmsEpsilon : Float32 := Float32.ofBits 0x358637bd
private def queryScale : Float32 := Float32.ofBits 0x3db504f3
private def log2e : Float32 := Float32.ofBits 0x3fb8aa3b

private def reductionBytes : Nat := 8 * 4
private def vectorBytes : Nat := linearHeadDimension * 4
private def queryOffset : Nat := reductionBytes
private def keyOffset : Nat := queryOffset + vectorBytes
private def deltaOffset : Nat := keyOffset + vectorBytes
private def sharedBytes : Nat := deltaOffset + vectorBytes

private def kindEmbedding : UInt32 := 0
private def kindLinear : UInt32 := 1
private def kindFull : UInt32 := 2
private def kindFinal : UInt32 := 3

@[struct] structure WeightRef where
  shard : UInt32
  reserved : UInt32 := 0
  byteOffset : UInt64
  deriving Repr, Inhabited, BEq, Cuda.POD

@[struct] structure ModelItem where
  kind : UInt32
  stateSlot : UInt32
  token : UInt32
  position : UInt32
  target : UInt32
  descriptor : UInt32
  inputNorm : WeightRef
  postNorm : WeightRef
  mlpGate : WeightRef
  mlpUp : WeightRef
  mlpDown : WeightRef
  attention0 : WeightRef
  attention1 : WeightRef
  attention2 : WeightRef
  attention3 : WeightRef
  attention4 : WeightRef
  attention5 : WeightRef
  attention6 : WeightRef
  attention7 : WeightRef
  attention8 : WeightRef
  deriving Repr, Inhabited, BEq, Cuda.POD

@[struct] private structure Shards where
  s0 : Cuda.DevicePtr UInt8
  s1 : Cuda.DevicePtr UInt8
  s2 : Cuda.DevicePtr UInt8
  s3 : Cuda.DevicePtr UInt8
  s4 : Cuda.DevicePtr UInt8
  s5 : Cuda.DevicePtr UInt8
  s6 : Cuda.DevicePtr UInt8
  s7 : Cuda.DevicePtr UInt8
  s8 : Cuda.DevicePtr UInt8
  s9 : Cuda.DevicePtr UInt8
  s10 : Cuda.DevicePtr UInt8
  s11 : Cuda.DevicePtr UInt8
  s12 : Cuda.DevicePtr UInt8
  s13 : Cuda.DevicePtr UInt8
  s14 : Cuda.DevicePtr UInt8

@[struct] private structure Buffers where
  hidden : Cuda.DevicePtr Cuda.BFloat16
  normalized : Cuda.DevicePtr Cuda.BFloat16
  workspace0 : Cuda.DevicePtr Cuda.BFloat16
  workspace1 : Cuda.DevicePtr Cuda.BFloat16
  workspace2 : Cuda.DevicePtr Cuda.BFloat16
  attentionOutput : Cuda.DevicePtr Cuda.BFloat16
  projectedOutput : Cuda.DevicePtr Cuda.BFloat16
  convolutionState : Cuda.DevicePtr Cuda.BFloat16
  recurrentState : Cuda.DevicePtr Float32
  keyCache : Cuda.DevicePtr Cuda.BFloat16
  valueCache : Cuda.DevicePtr Cuda.BFloat16
  logits : Cuda.DevicePtr Float32
  gradient : Cuda.DevicePtr Cuda.BFloat16
  partialMaximum : Cuda.DevicePtr Float32
  partialSum : Cuda.DevicePtr Float32
  loss : Cuda.DevicePtr Float32
  sampled : Cuda.DevicePtr UInt32
  counts : Cuda.DevicePtr UInt32

@[always_inline]
private def Shards.get (shards : Shards) (index : UInt32) : Cuda.DevicePtr UInt8 :=
  if index == 0 then shards.s0
  else if index == 1 then shards.s1
  else if index == 2 then shards.s2
  else if index == 3 then shards.s3
  else if index == 4 then shards.s4
  else if index == 5 then shards.s5
  else if index == 6 then shards.s6
  else if index == 7 then shards.s7
  else if index == 8 then shards.s8
  else if index == 9 then shards.s9
  else if index == 10 then shards.s10
  else if index == 11 then shards.s11
  else if index == 12 then shards.s12
  else if index == 13 then shards.s13
  else shards.s14

@[always_inline]
private def weightPointer (shards : Shards) (weight : WeightRef) :
    Cuda.DevicePtr Cuda.BFloat16 :=
  shards.get weight.shard + weight.byteOffset.toUSize

@[always_inline]
private def bfloatOffset (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (elements : UInt32) : Cuda.DevicePtr Cuda.BFloat16 :=
  pointer + elements.toUSize * 2

@[always_inline]
private def floatOffset (pointer : Cuda.DevicePtr Float32)
    (elements : UInt32) : Cuda.DevicePtr Float32 :=
  pointer + elements.toUSize * 4

@[always_inline]
private def loadBFloat (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) : Cuda.DeviceM Float32 := do
  return (← Cuda.loadBFloat16 pointer index.toUSize).toFloat32

@[always_inline]
private def storeBFloat (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) (value : Float32) : Cuda.DeviceM Unit :=
  Cuda.storeBFloat16 pointer index.toUSize (Cuda.BFloat16.ofFloat32 value)

@[always_inline]
private def rounded (value : Float32) : Float32 :=
  (Cuda.BFloat16.ofFloat32 value).toFloat32

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
private partial def sumPartials (partials : Cuda.DevicePtr Float32)
    (index : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < blockCount then
    sumPartials partials (index + 1) (accumulator + (← Cuda.loadFloat32 partials index.toUSize))
  else
    return accumulator

@[always_inline]
private partial def maximumPartials (partials : Cuda.DevicePtr Float32)
    (index : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < blockCount then
    maximumPartials partials (index + 1)
      (max accumulator (← Cuda.loadFloat32 partials index.toUSize))
  else
    return accumulator

@[always_inline]
private partial def squareSumLoop (source : Cuda.DevicePtr Cuda.BFloat16)
    (count index stride : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < count then
    let value ← loadBFloat source index
    squareSumLoop source count (index + stride) stride (Cuda.fma value value accumulator)
  else
    return accumulator

@[always_inline]
private partial def normalizeLoop (source weight output : Cuda.DevicePtr Cuda.BFloat16)
    (count index stride : UInt32) (inverse : Float32) : Cuda.DeviceM Unit := do
  if index < count then
    let value ← loadBFloat source index
    let scale ← loadBFloat weight index
    storeBFloat output index (value * inverse * (one + scale))
    normalizeLoop source weight output count (index + stride) stride inverse

@[always_inline, convergent]
private def runRmsNorm (source weight output : Cuda.DevicePtr Cuda.BFloat16)
    (buffers : Buffers) (scratch : Cuda.Collective.BlockScratch 8)
    (block thread globalThread globalThreads : UInt32) : Cuda.DeviceM Unit := do
  let localSquares ← squareSumLoop source hiddenSizeU32 globalThread globalThreads zero
  let blockSquares ← Cuda.Collective.blockSum scratch localSquares
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialSum block.toUSize blockSquares
  Cuda.gridSync
  let squareSum ← sumPartials buffers.partialSum 0 zero
  let inverse ← Cuda.fastRsqrt (squareSum * inverseHiddenSize + rmsEpsilon)
  normalizeLoop source weight output hiddenSizeU32 globalThread globalThreads inverse
  Cuda.gridSync

@[always_inline]
private partial def dotProjectionRow (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columns column : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < columns then
    let inputValue ← loadBFloat input column
    let weightValue ← loadBFloat weight (row * columns + column)
    dotProjectionRow input weight row columns (column + 256)
      (Cuda.fma weightValue inputValue accumulator)
  else
    return accumulator

@[always_inline, convergent]
private partial def projectRowsBFloat (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSum ← dotProjectionRow input weight row columns thread zero
    let total ← Cuda.Collective.blockSum scratch localSum
    if thread == 0 then
      storeBFloat output row total
    projectRowsBFloat input weight output rows columns (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowsFloat (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (output : Cuda.DevicePtr Float32) (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSum ← dotProjectionRow input weight row columns thread zero
    let total ← Cuda.Collective.blockSum scratch localSum
    if thread == 0 then
      Cuda.storeFloat32 output row.toUSize total
    projectRowsFloat input weight output rows columns (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private def runProjection (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns block thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8) :
    Cuda.DeviceM Unit := do
  projectRowsBFloat input weight output rows columns block blockCount thread scratch
  Cuda.gridSync

@[always_inline]
private partial def residualLoop (hidden update : Cuda.DevicePtr Cuda.BFloat16)
    (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < hiddenSizeU32 then
    storeBFloat hidden index ((← loadBFloat hidden index) + (← loadBFloat update index))
    residualLoop hidden update (index + stride) stride

@[always_inline]
private partial def activationLoop (gate up output : Cuda.DevicePtr Cuda.BFloat16)
    (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < intermediateSizeU32 then
    let gateValue ← deviceSilu (← loadBFloat gate index)
    let activatedGate := rounded gateValue
    storeBFloat output index (activatedGate * (← loadBFloat up index))
    activationLoop gate up output (index + stride) stride

@[always_inline]
private def linearStateIndex (head row column : UInt32) : UInt32 :=
  (head * linearDimension + row) * linearDimension + column

@[always_inline]
private def linearCompactIndex (head channel : UInt32) : UInt32 :=
  head * linearDimension + channel

@[always_inline]
private def runLinearConvolution (item : ModelItem) (convolutionWeight : Cuda.DevicePtr Cuda.BFloat16)
    (buffers : Buffers) (block thread : UInt32) : Cuda.DeviceM Unit := do
  let channel := block * 256 + thread
  if channel < linearQkvSizeU32 then
    let layerBase := item.stateSlot * convolutionElementsPerLayerNat.toUInt32
    let stateBase := layerBase + channel * 4
    let old1 ← loadBFloat buffers.convolutionState (stateBase + 1)
    let old2 ← loadBFloat buffers.convolutionState (stateBase + 2)
    let old3 ← loadBFloat buffers.convolutionState (stateBase + 3)
    let current ← loadBFloat buffers.workspace0 channel
    let weightBase := channel * 4
    let accumulator := Cuda.fma (← loadBFloat convolutionWeight weightBase) old1 zero
    let accumulator := Cuda.fma (← loadBFloat convolutionWeight (weightBase + 1)) old2 accumulator
    let accumulator := Cuda.fma (← loadBFloat convolutionWeight (weightBase + 2)) old3 accumulator
    let accumulator := Cuda.fma (← loadBFloat convolutionWeight (weightBase + 3)) current accumulator
    storeBFloat (bfloatOffset buffers.workspace2 (linearHeads * 2)) channel
      (← deviceSilu accumulator)
    storeBFloat buffers.convolutionState stateBase old1
    storeBFloat buffers.convolutionState (stateBase + 1) old2
    storeBFloat buffers.convolutionState (stateBase + 2) old3
    storeBFloat buffers.convolutionState (stateBase + 3) current

@[always_inline]
private partial def decayState (state : Cuda.DevicePtr Float32)
    (head index : UInt32) (decay : Float32) : Cuda.DeviceM Unit := do
  if index < linearStateElementsPerHead then
    let address := head * linearStateElementsPerHead + index
    Cuda.storeFloat32 state address.toUSize ((← Cuda.loadFloat32 state address.toUSize) * decay)
    decayState state head (index + 256) decay

@[always_inline]
private partial def stateKeyDot (state keyShared : Cuda.DevicePtr Float32)
    (head column row : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < linearDimension then
    let stateValue ← Cuda.loadFloat32 state (linearStateIndex head row column).toUSize
    let keyValue ← Cuda.loadFloat32 keyShared row.toUSize
    stateKeyDot state keyShared head column (row + 1)
      (Cuda.fma stateValue keyValue accumulator)
  else
    return accumulator

@[always_inline]
private partial def updateState (state keyShared deltaShared : Cuda.DevicePtr Float32)
    (head index : UInt32) : Cuda.DeviceM Unit := do
  if index < linearStateElementsPerHead then
    let row := index / linearDimension
    let column := index % linearDimension
    let address := linearStateIndex head row column
    let previous ← Cuda.loadFloat32 state address.toUSize
    let keyValue ← Cuda.loadFloat32 keyShared row.toUSize
    let delta ← Cuda.loadFloat32 deltaShared column.toUSize
    Cuda.storeFloat32 state address.toUSize (Cuda.fma keyValue delta previous)
    updateState state keyShared deltaShared head (index + 256)

@[always_inline]
private partial def stateQueryDot (state queryShared : Cuda.DevicePtr Float32)
    (head column row : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < linearDimension then
    let stateValue ← Cuda.loadFloat32 state (linearStateIndex head row column).toUSize
    let queryValue ← Cuda.loadFloat32 queryShared row.toUSize
    stateQueryDot state queryShared head column (row + 1)
      (Cuda.fma stateValue queryValue accumulator)
  else
    return accumulator

@[always_inline, convergent]
private def runLinearRecurrent (item : ModelItem) (shards : Shards) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (head thread : UInt32) : Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  let queryShared ← Cuda.dynamicShared (α := Float32) queryOffset.toUSize
  let keyShared ← Cuda.dynamicShared (α := Float32) keyOffset.toUSize
  let deltaShared ← Cuda.dynamicShared (α := Float32) deltaOffset.toUSize
  let convolved := bfloatOffset buffers.workspace2 (linearHeads * 2)
  let projectedA := buffers.workspace2
  let projectedB := bfloatOffset buffers.workspace2 linearHeads
  let state := floatOffset buffers.recurrentState
    (item.stateSlot * linearStateElementsPerLayerNat.toUInt32)
  let compactHead := head / 3
  let queryValue ← if thread < linearDimension then
      loadBFloat convolved (linearCompactIndex compactHead thread)
    else pure zero
  let keyValue ← if thread < linearDimension then
      loadBFloat convolved (linearKeyElements + linearCompactIndex compactHead thread)
    else pure zero
  let querySquareSum ← Cuda.Collective.blockSum scratch (queryValue * queryValue)
  let keySquareSum ← Cuda.Collective.blockSum scratch (keyValue * keyValue)
  let queryInverse ← Cuda.fastRsqrt (querySquareSum + rmsEpsilon)
  let keyInverse ← Cuda.fastRsqrt (keySquareSum + rmsEpsilon)
  if thread < linearDimension then
    Cuda.storeFloat32 queryShared thread.toUSize (queryValue * queryInverse * queryScale)
    Cuda.storeFloat32 keyShared thread.toUSize (keyValue * keyInverse)
  Cuda.blockSync
  if thread == 0 then
    let aLog ← loadBFloat (weightPointer shards item.attention0) head
    let dtBias ← loadBFloat (weightPointer shards item.attention2) head
    let rate ← Cuda.fastExp aLog
    let step ← deviceSoftplus ((← loadBFloat projectedA head) + dtBias)
    let decay ← Cuda.fastExp (-(rate * step))
    let beta := rounded (← deviceSigmoid (← loadBFloat projectedB head))
    Cuda.storeFloat32 scratch 0 decay
    Cuda.storeFloat32 scratch 1 beta
  Cuda.blockSync
  let decay ← Cuda.loadFloat32 scratch 0
  let beta ← Cuda.loadFloat32 scratch 1
  decayState state head thread decay
  Cuda.blockSync
  if thread < linearDimension then
    let memory ← stateKeyDot state keyShared head thread 0 zero
    let value ← loadBFloat convolved (linearKeyElements * 2 + head * linearDimension + thread)
    Cuda.storeFloat32 deltaShared thread.toUSize ((value - memory) * beta)
  Cuda.blockSync
  updateState state keyShared deltaShared head thread
  Cuda.blockSync
  if thread < linearDimension then
    let result ← stateQueryDot state queryShared head thread 0 zero
    storeBFloat buffers.attentionOutput (head * linearDimension + thread) result
  Cuda.blockSync
  let recurrent ← if thread < linearDimension then
      loadBFloat buffers.attentionOutput (head * linearDimension + thread)
    else pure zero
  let squareSum ← Cuda.Collective.blockSum scratch (recurrent * recurrent)
  let inverse ← Cuda.fastRsqrt (squareSum * inverseLinearDimension + rmsEpsilon)
  if thread < linearDimension then
    let weight ← loadBFloat (weightPointer shards item.attention7) thread
    let normalizedWeighted := rounded (recurrent * inverse * weight)
    let gate ← deviceSilu (← loadBFloat buffers.workspace1 (head * linearDimension + thread))
    storeBFloat buffers.attentionOutput (head * linearDimension + thread) (normalizedWeighted * gate)

@[always_inline, convergent]
private def runLinearLayer (item : ModelItem) (shards : Shards) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runProjection buffers.normalized (weightPointer shards item.attention5) buffers.workspace0
    linearQkvSizeU32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (weightPointer shards item.attention6) buffers.workspace1
    linearValueSizeU32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (weightPointer shards item.attention3) buffers.workspace2
    linearHeads hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (weightPointer shards item.attention4)
    (bfloatOffset buffers.workspace2 linearHeads) linearHeads hiddenSizeU32 block thread scratch
  runLinearConvolution item (weightPointer shards item.attention1) buffers block thread
  Cuda.gridSync
  runLinearRecurrent item shards buffers shared block thread
  Cuda.gridSync
  runProjection buffers.attentionOutput (weightPointer shards item.attention8)
    buffers.projectedOutput hiddenSizeU32 linearValueSizeU32 block thread scratch
  residualLoop buffers.hidden buffers.projectedOutput globalThread globalThreads
  Cuda.gridSync

@[always_inline, convergent]
private def runFullCore (item : ModelItem) (shards : Shards) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (head thread : UInt32) : Cuda.DeviceM Unit := do
  if head < fullQueryHeads then
    let scratch : Cuda.Collective.BlockScratch 8 := shared
    let queryValue ← loadBFloat buffers.workspace0 (head * 512 + thread)
    let querySquares ← Cuda.Collective.blockSum scratch (queryValue * queryValue)
    let queryInverse ← Cuda.fastRsqrt (querySquares * inverseFullDimension + rmsEpsilon)
    let queryWeight ← loadBFloat (weightPointer shards item.attention0) thread
    let _normalizedQuery := queryValue * queryInverse * (one + queryWeight)
    if head < fullKeyValueHeads then
      let keyValue ← loadBFloat buffers.workspace1 (head * fullDimension + thread)
      let keySquares ← Cuda.Collective.blockSum scratch (keyValue * keyValue)
      let keyInverse ← Cuda.fastRsqrt (keySquares * inverseFullDimension + rmsEpsilon)
      let keyWeight ← loadBFloat (weightPointer shards item.attention1) thread
      let cacheIndex := item.stateSlot * fullKeyValueProjectionSize.toUInt32 +
        head * fullDimension + thread
      storeBFloat buffers.keyCache cacheIndex (keyValue * keyInverse * (one + keyWeight))
      let value ← loadBFloat buffers.workspace2 (head * fullDimension + thread)
      Cuda.storeBFloat16 buffers.valueCache cacheIndex.toUSize (Cuda.BFloat16.ofFloat32 value)
    let valueHead := head / fullGroups
    let value ← loadBFloat buffers.workspace2 (valueHead * fullDimension + thread)
    let gateInput ← loadBFloat buffers.workspace0 (head * 512 + fullDimension + thread)
    let gate := rounded (← deviceSigmoid gateInput)
    storeBFloat buffers.attentionOutput (head * fullDimension + thread) (value * gate)

@[always_inline, convergent]
private def runFullLayer (item : ModelItem) (shards : Shards) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runProjection buffers.normalized (weightPointer shards item.attention2) buffers.workspace0
    fullQueryProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (weightPointer shards item.attention3) buffers.workspace1
    fullKeyValueProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (weightPointer shards item.attention4) buffers.workspace2
    fullKeyValueProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runFullCore item shards buffers shared block thread
  Cuda.gridSync
  runProjection buffers.attentionOutput (weightPointer shards item.attention5)
    buffers.projectedOutput hiddenSizeU32 fullOutputSize.toUInt32 block thread scratch
  residualLoop buffers.hidden buffers.projectedOutput globalThread globalThreads
  Cuda.gridSync

@[always_inline, convergent]
private def runMLP (item : ModelItem) (shards : Shards) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runRmsNorm buffers.hidden (weightPointer shards item.postNorm) buffers.normalized buffers scratch
    block thread globalThread globalThreads
  runProjection buffers.normalized (weightPointer shards item.mlpGate) buffers.workspace0
    intermediateSizeU32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (weightPointer shards item.mlpUp) buffers.workspace1
    intermediateSizeU32 hiddenSizeU32 block thread scratch
  activationLoop buffers.workspace0 buffers.workspace1 buffers.workspace2 globalThread globalThreads
  Cuda.gridSync
  runProjection buffers.workspace2 (weightPointer shards item.mlpDown) buffers.projectedOutput
    hiddenSizeU32 intermediateSizeU32 block thread scratch
  residualLoop buffers.hidden buffers.projectedOutput globalThread globalThreads
  Cuda.gridSync

@[always_inline]
private partial def embeddingLoop (embedding hidden : Cuda.DevicePtr Cuda.BFloat16)
    (token index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < hiddenSizeU32 then
    let value ← Cuda.loadBFloat16 embedding (token * hiddenSizeU32 + index).toUSize
    Cuda.storeBFloat16 hidden index.toUSize value
    embeddingLoop embedding hidden token (index + stride) stride

@[always_inline]
private partial def localMaximumLoop (logits : Cuda.DevicePtr Float32)
    (index stride : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < vocabularySizeU32 then
    localMaximumLoop logits (index + stride) stride
      (max accumulator (← Cuda.loadFloat32 logits index.toUSize))
  else return accumulator

@[always_inline]
private partial def localExponentialSumLoop (logits : Cuda.DevicePtr Float32)
    (maximum : Float32) (index stride : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if index < vocabularySizeU32 then
    let value ← Cuda.loadFloat32 logits index.toUSize
    let exponential ← Cuda.fastExp2Fma ((value - maximum) * log2e)
    localExponentialSumLoop logits maximum (index + stride) stride (accumulator + exponential)
  else return accumulator

@[always_inline]
private partial def gradientAndSampleLoop (item : ModelItem) (buffers : Buffers)
    (maximum inverseDenominator : Float32) (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < vocabularySizeU32 then
    let value ← Cuda.loadFloat32 buffers.logits index.toUSize
    let exponential ← Cuda.fastExp2Fma ((value - maximum) * log2e)
    let target := if index == item.target then one else zero
    storeBFloat buffers.gradient index (exponential * inverseDenominator - target)
    if value == maximum then
      discard <| Cuda.atomicMinUInt32 buffers.sampled index
    gradientAndSampleLoop item buffers maximum inverseDenominator (index + stride) stride

@[always_inline, convergent]
private def runFinal (item : ModelItem) (shards : Shards) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runRmsNorm buffers.hidden (weightPointer shards item.inputNorm) buffers.normalized buffers scratch
    block thread globalThread globalThreads
  projectRowsFloat buffers.normalized (weightPointer shards item.attention0) buffers.logits
    vocabularySizeU32 hiddenSizeU32 block blockCount thread scratch
  Cuda.gridSync
  if block == 0 && thread == 0 then
    Cuda.storeUInt32 buffers.sampled 0 0xffffffff
  Cuda.gridSync
  let localMaximum ← localMaximumLoop buffers.logits globalThread globalThreads negativeInfinity
  let blockMaximum ← Cuda.Collective.blockMax scratch localMaximum
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialMaximum block.toUSize blockMaximum
  Cuda.gridSync
  let maximum ← maximumPartials buffers.partialMaximum 0 negativeInfinity
  let localSum ← localExponentialSumLoop buffers.logits maximum globalThread globalThreads zero
  let blockSum ← Cuda.Collective.blockSum scratch localSum
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialSum block.toUSize blockSum
  Cuda.gridSync
  let denominator ← sumPartials buffers.partialSum 0 zero
  let inverseDenominator ← Cuda.fastDivide one denominator
  gradientAndSampleLoop item buffers maximum inverseDenominator globalThread globalThreads
  if block == 0 && thread == 0 then
    let targetLogit ← Cuda.loadFloat32 buffers.logits item.target.toUSize
    Cuda.storeFloat32 buffers.loss 0 (maximum + (← Cuda.fastLog denominator) - targetLogit)
  Cuda.gridSync

@[cuda_grid_persistent]
def qwen36Model (item : ModelItem)
    (s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 : Cuda.DevicePtr UInt8)
    (hidden normalized workspace0 workspace1 workspace2 attentionOutput projectedOutput
      convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (keyCache valueCache : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let globalThread ← Cuda.globalThreadIdxX
  let globalThreads ← Cuda.globalThreadCountX
  let shared ← Cuda.dynamicShared (α := Float32)
  let shards : Shards := { s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14 }
  let buffers : Buffers := {
    hidden, normalized, workspace0, workspace1, workspace2, attentionOutput, projectedOutput,
    convolutionState, recurrentState, keyCache, valueCache, logits, gradient, partialMaximum,
    partialSum, loss, sampled, counts
  }
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  if item.kind == kindEmbedding then
    embeddingLoop (weightPointer shards item.inputNorm) buffers.hidden item.token
      globalThread globalThreads
    Cuda.gridSync
  else if item.kind == kindLinear then
    runRmsNorm buffers.hidden (weightPointer shards item.inputNorm) buffers.normalized buffers scratch
      block thread globalThread globalThreads
    runLinearLayer item shards buffers shared block thread globalThread globalThreads
    runMLP item shards buffers shared block thread globalThread globalThreads
  else if item.kind == kindFull then
    if item.position != 0 then
      Cuda.panic 0x3601 item.position.toUInt64
    else
      runRmsNorm buffers.hidden (weightPointer shards item.inputNorm) buffers.normalized buffers scratch
        block thread globalThread globalThreads
      runFullLayer item shards buffers shared block thread globalThread globalThreads
      runMLP item shards buffers shared block thread globalThread globalThreads
  else if item.kind == kindFinal then
    runFinal item shards buffers shared block thread globalThread globalThreads
  else
    Cuda.panic 0x3602 item.kind.toUInt64
  if block == 0 && thread == 0 then
    Cuda.atomicAddUInt32At_ buffers.counts item.descriptor 1

@[always_inline]
private partial def zeroFloatLoop (pointer : Cuda.DevicePtr Float32)
    (count index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < count then
    Cuda.storeFloat32 pointer index.toUSize zero
    zeroFloatLoop pointer count (index + stride) stride

@[always_inline]
private partial def zeroBFloatLoop (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (count index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < count then
    Cuda.storeBFloat16 pointer index.toUSize (Cuda.BFloat16.ofFloat32 zero)
    zeroBFloatLoop pointer count (index + stride) stride

@[always_inline]
private partial def zeroUIntLoop (pointer : Cuda.DevicePtr UInt32)
    (count index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < count then
    Cuda.storeUInt32 pointer index.toUSize 0
    zeroUIntLoop pointer count (index + stride) stride

@[cuda_kernel]
def initializeModelState (convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (keyCache valueCache : Cuda.DevicePtr Cuda.BFloat16)
    (counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let index ← Cuda.globalThreadIdxX
  let stride ← Cuda.globalThreadCountX
  zeroBFloatLoop convolutionState
    (linearLayerCount * convolutionElementsPerLayerNat).toUInt32 index stride
  zeroFloatLoop recurrentState
    (linearLayerCount * linearStateElementsPerLayerNat).toUInt32 index stride
  zeroBFloatLoop keyCache
    (fullAttentionLayerCount * fullCacheElementsPerLayerNat).toUInt32 index stride
  zeroBFloatLoop valueCache
    (fullAttentionLayerCount * fullCacheElementsPerLayerNat).toUInt32 index stride
  zeroUIntLoop counts descriptorCountNat.toUInt32 index stride

private structure LoadedShards where
  s0 : Cuda.Buffer UInt8
  s1 : Cuda.Buffer UInt8
  s2 : Cuda.Buffer UInt8
  s3 : Cuda.Buffer UInt8
  s4 : Cuda.Buffer UInt8
  s5 : Cuda.Buffer UInt8
  s6 : Cuda.Buffer UInt8
  s7 : Cuda.Buffer UInt8
  s8 : Cuda.Buffer UInt8
  s9 : Cuda.Buffer UInt8
  s10 : Cuda.Buffer UInt8
  s11 : Cuda.Buffer UInt8
  s12 : Cuda.Buffer UInt8
  s13 : Cuda.Buffer UInt8
  s14 : Cuda.Buffer UInt8

private structure ModelBuffers where
  hidden : Cuda.Buffer Cuda.BFloat16
  normalized : Cuda.Buffer Cuda.BFloat16
  workspace0 : Cuda.Buffer Cuda.BFloat16
  workspace1 : Cuda.Buffer Cuda.BFloat16
  workspace2 : Cuda.Buffer Cuda.BFloat16
  attentionOutput : Cuda.Buffer Cuda.BFloat16
  projectedOutput : Cuda.Buffer Cuda.BFloat16
  convolutionState : Cuda.Buffer Cuda.BFloat16
  recurrentState : Cuda.Buffer Float32
  keyCache : Cuda.Buffer Cuda.BFloat16
  valueCache : Cuda.Buffer Cuda.BFloat16
  logits : Cuda.Buffer Float32
  gradient : Cuda.Buffer Cuda.BFloat16
  partialMaximum : Cuda.Buffer Float32
  partialSum : Cuda.Buffer Float32
  loss : Cuda.Buffer Float32
  sampled : Cuda.Buffer UInt32
  counts : Cuda.Buffer UInt32

private def loadShard (manifest : Manifest) (stream : Cuda.Stream) (index : Nat) :
    IO (Cuda.Buffer UInt8) := do
  let some shard := manifest.shards[index]?
    | throw <| IO.userError s!"missing checkpoint shard {index}"
  IO.println s!"loading Qwen3.6 shard {index + 1}/{manifest.shards.size}: {shard.filename}"
  let buffer ← Cuda.Buffer.alloc UInt8 shard.header.dataBytes.toUSize
  forEachShardPayloadChunk (manifest.root / shard.filename) shard.header fun offset chunk =>
    buffer.copyFromAt chunk offset.toUSize stream
  stream.synchronize
  return buffer

private def loadShards (manifest : Manifest) (stream : Cuda.Stream) : IO LoadedShards := do
  let s0 ← loadShard manifest stream 0
  let s1 ← loadShard manifest stream 1
  let s2 ← loadShard manifest stream 2
  let s3 ← loadShard manifest stream 3
  let s4 ← loadShard manifest stream 4
  let s5 ← loadShard manifest stream 5
  let s6 ← loadShard manifest stream 6
  let s7 ← loadShard manifest stream 7
  let s8 ← loadShard manifest stream 8
  let s9 ← loadShard manifest stream 9
  let s10 ← loadShard manifest stream 10
  let s11 ← loadShard manifest stream 11
  let s12 ← loadShard manifest stream 12
  let s13 ← loadShard manifest stream 13
  let s14 ← loadShard manifest stream 14
  return { s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14 }

private def allocateBuffers : IO ModelBuffers := do
  let hidden ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSize.toUSize
  let normalized ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSize.toUSize
  let workspace0 ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSize.toUSize
  let workspace1 ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSize.toUSize
  let workspace2 ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSize.toUSize
  let attentionOutput ← Cuda.Buffer.alloc Cuda.BFloat16 linearValueSize.toUSize
  let projectedOutput ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSize.toUSize
  let convolutionState ← Cuda.Buffer.alloc Cuda.BFloat16
    (linearLayerCount * convolutionElementsPerLayerNat).toUSize
  let recurrentState ← Cuda.Buffer.alloc Float32
    (linearLayerCount * linearStateElementsPerLayerNat).toUSize
  let keyCache ← Cuda.Buffer.alloc Cuda.BFloat16
    (fullAttentionLayerCount * fullCacheElementsPerLayerNat).toUSize
  let valueCache ← Cuda.Buffer.alloc Cuda.BFloat16
    (fullAttentionLayerCount * fullCacheElementsPerLayerNat).toUSize
  let logits ← Cuda.Buffer.alloc Float32 vocabularySize.toUSize
  let gradient ← Cuda.Buffer.alloc Cuda.BFloat16 vocabularySize.toUSize
  let partialMaximum ← Cuda.Buffer.alloc Float32 blockCountNat.toUSize
  let partialSum ← Cuda.Buffer.alloc Float32 blockCountNat.toUSize
  let loss ← Cuda.Buffer.alloc Float32 1
  let sampled ← Cuda.Buffer.alloc UInt32 1
  let counts ← Cuda.Buffer.alloc UInt32 descriptorCountNat.toUSize
  return {
    hidden, normalized, workspace0, workspace1, workspace2, attentionOutput, projectedOutput,
    convolutionState, recurrentState, keyCache, valueCache, logits, gradient, partialMaximum,
    partialSum, loss, sampled, counts
  }

private def toWeightRef (tensor : TensorRef) : WeightRef := {
  shard := tensor.shard
  byteOffset := tensor.byteOffset
}

private def zeroWeight : WeightRef := { shard := 0, byteOffset := 0 }

private def blankItem : ModelItem := {
  kind := 0, stateSlot := 0, token := 0, position := 0, target := 0, descriptor := 0,
  inputNorm := zeroWeight, postNorm := zeroWeight, mlpGate := zeroWeight,
  mlpUp := zeroWeight, mlpDown := zeroWeight, attention0 := zeroWeight,
  attention1 := zeroWeight, attention2 := zeroWeight, attention3 := zeroWeight,
  attention4 := zeroWeight, attention5 := zeroWeight, attention6 := zeroWeight,
  attention7 := zeroWeight, attention8 := zeroWeight
}

private def embeddingItem (layout : TextLayout) (token : UInt32) : ModelItem := {
  blankItem with
  kind := kindEmbedding
  token
  inputNorm := toWeightRef layout.embedding
}

private def layerItem (layer : LayerWeights) (slot descriptor token target : UInt32) : ModelItem :=
  let common := layer.common
  let base : ModelItem := {
    blankItem with
    stateSlot := slot
    token
    target
    descriptor
    inputNorm := toWeightRef common.inputNorm
    postNorm := toWeightRef common.postAttentionNorm
    mlpGate := toWeightRef common.mlpGate
    mlpUp := toWeightRef common.mlpUp
    mlpDown := toWeightRef common.mlpDown
  }
  match layer.attention with
  | .linear weights => {
      base with
      kind := kindLinear
      attention0 := toWeightRef weights.aLog
      attention1 := toWeightRef weights.convolution
      attention2 := toWeightRef weights.dtBias
      attention3 := toWeightRef weights.projectionA
      attention4 := toWeightRef weights.projectionB
      attention5 := toWeightRef weights.projectionQkv
      attention6 := toWeightRef weights.projectionZ
      attention7 := toWeightRef weights.norm
      attention8 := toWeightRef weights.outputProjection
    }
  | .full weights => {
      base with
      kind := kindFull
      attention0 := toWeightRef weights.queryNorm
      attention1 := toWeightRef weights.keyNorm
      attention2 := toWeightRef weights.queryProjection
      attention3 := toWeightRef weights.keyProjection
      attention4 := toWeightRef weights.valueProjection
      attention5 := toWeightRef weights.outputProjection
    }

private def finalItem (layout : TextLayout) (target : UInt32) : ModelItem := {
  blankItem with
  kind := kindFinal
  target
  descriptor := (layerCount + 1).toUInt32
  inputNorm := toWeightRef layout.finalNorm
  attention0 := toWeightRef layout.lmHead
}

private def initializerConfig : Cuda.LaunchConfig := {
  grid := { x := 256 }
  block := { x := 256 }
  blockArenaBytes := 4 * 1024
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := blockCount }
  block := { x := 256 }
  sharedMemoryBytes := sharedBytes.toUSize
  blockArenaBytes := 64 * 1024
  queueCapacity := descriptorCountNat.toUSize
}

private def readUInt32 (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def runOnce (shards : LoadedShards) (layout : TextLayout) (buffers : ModelBuffers)
    (token target : UInt32) (stream : Cuda.Stream) : IO (ByteArray × ByteArray × ByteArray) := do
  (← initializeModelState.launch initializerConfig buffers.convolutionState buffers.recurrentState
    buffers.keyCache buffers.valueCache buffers.counts).waitChecked "Qwen3.6 state initialization"
  let handle ← qwen36Model.start persistentConfig
    shards.s0 shards.s1 shards.s2 shards.s3 shards.s4 shards.s5 shards.s6 shards.s7
    shards.s8 shards.s9 shards.s10 shards.s11 shards.s12 shards.s13 shards.s14
    buffers.hidden buffers.normalized buffers.workspace0 buffers.workspace1 buffers.workspace2
    buffers.attentionOutput buffers.projectedOutput buffers.convolutionState buffers.recurrentState
    buffers.keyCache buffers.valueCache buffers.logits buffers.gradient buffers.partialMaximum
    buffers.partialSum buffers.loss buffers.sampled buffers.counts
  qwen36Model.enqueue handle (embeddingItem layout token)
  let mut linearSlot : UInt32 := 0
  let mut fullSlot : UInt32 := 0
  for index in [:layout.layers.size] do
    let some layer := layout.layers[index]?
      | throw <| IO.userError s!"missing resolved Qwen3.6 layer {index}"
    let slot := match layer.attention with
      | .linear _ => linearSlot
      | .full _ => fullSlot
    qwen36Model.enqueue handle (layerItem layer slot (index + 1).toUInt32 token target)
    match layer.attention with
    | .linear _ => linearSlot := linearSlot + 1
    | .full _ => fullSlot := fullSlot + 1
  qwen36Model.enqueue handle (finalItem layout target)
  handle.shutdown
  handle.waitChecked "Qwen3.6 model megakernel"
  let counts ← buffers.counts.copyTo stream
  for descriptor in [:descriptorCountNat] do
    unless readUInt32 counts descriptor == 1 do
      throw <| IO.userError
        s!"Qwen3.6 descriptor {descriptor} executed {readUInt32 counts descriptor} times"
  let logits ← buffers.logits.copyTo stream
  let loss ← buffers.loss.copyTo stream
  let sampled ← buffers.sampled.copyTo stream
  return (logits, loss, sampled)

def main : IO Unit := do
  let some modelDirectory ← IO.getEnv "QWEN36_MODEL_DIR"
    | throw <| IO.userError "set QWEN36_MODEL_DIR to the Qwen/Qwen3.6-27B snapshot"
  let token := (← IO.getEnv "QWEN36_TOKEN_ID").bind (·.toNat?) |>.getD 1234
  let target := (← IO.getEnv "QWEN36_TARGET_ID").bind (·.toNat?) |>.getD 1234
  if token >= vocabularySize || target >= vocabularySize then
    throw <| IO.userError "Qwen3.6 token and target IDs must be below 248320"
  let manifest ← readManifest modelDirectory
  let layout ← readTextLayout manifest
  let stream ← Cuda.Stream.default
  let shards ← loadShards manifest stream
  let buffers ← allocateBuffers
  let start ← IO.monoMsNow
  let first ← runOnce shards layout buffers token.toUInt32 target.toUInt32 stream
  let elapsed ← IO.monoMsNow
  let second ← runOnce shards layout buffers token.toUInt32 target.toUInt32 stream
  unless first == second do
    throw <| IO.userError "repeated Qwen3.6 model megakernel runs differ"
  let (_, loss, sampled) := first
  IO.println s!"Qwen3.6 first-position token={token}, sampled={readUInt32 sampled 0}, loss_bits={readUInt32 loss 0}, elapsed_ms={elapsed - start}"
  IO.println "Lean Qwen3.6 real-checkpoint 64-layer inference megakernel ok"

end Qwen36Model

def main : IO Unit := Qwen36Model.main
