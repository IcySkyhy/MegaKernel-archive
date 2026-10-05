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
# Qwen3.6 recurrent Gated DeltaNet

This module implements the recurrent form shared by decode and the numerical prefill baseline.
Query/key inputs use the compact key-head layout and are repeated across value heads on device.
The FP32 state has row-major shape `[valueHeads, keyWidth, valueWidth]`.

The recurrent-prefill kernel is the correctness baseline for the production chunk-64 body. It is
not the final chunked schedule: `cuda_qwen36_deltanet` requires both routes to agree with the
reference fixtures before the chunked route can replace it in model training.
-/

namespace Cuda.Qwen36.DeltaNet

open Cuda.Qwen36.Primitives

@[struct] structure Buffers where
  query : Cuda.DevicePtr Float32
  key : Cuda.DevicePtr Float32
  value : Cuda.DevicePtr Float32
  decayLog : Cuda.DevicePtr Float32
  beta : Cuda.DevicePtr Float32
  output : Cuda.DevicePtr Float32
  state : Cuda.DevicePtr Float32

@[struct] structure Shape where
  keyHeads : UInt32
  valueHeads : UInt32
  keyWidth : UInt32
  valueWidth : UInt32

@[struct] structure Scratch where
  query : Cuda.DevicePtr Float32
  key : Cuda.DevicePtr Float32
  delta : Cuda.DevicePtr Float32

@[struct] structure GateParameterGradients where
  aLog : Float32
  dtBias : Float32
  deriving Nonempty

@[struct] structure ReverseBuffers where
  query : Cuda.DevicePtr Float32
  key : Cuda.DevicePtr Float32
  value : Cuda.DevicePtr Float32
  decayLog : Cuda.DevicePtr Float32
  beta : Cuda.DevicePtr Float32
  stateHistory : Cuda.DevicePtr Float32
  outputGradient : Cuda.DevicePtr Float32
  queryGradient : Cuda.DevicePtr Float32
  keyGradient : Cuda.DevicePtr Float32
  valueGradient : Cuda.DevicePtr Float32
  decayLogGradient : Cuda.DevicePtr Float32
  betaGradient : Cuda.DevicePtr Float32
  stateGradient : Cuda.DevicePtr Float32

@[struct] structure ReverseScratch where
  delta : Cuda.DevicePtr Float32
  deltaGradient : Cuda.DevicePtr Float32
  memoryGradient : Cuda.DevicePtr Float32

@[struct] structure BatchedForwardBuffers where
  query : Cuda.DevicePtr Float32
  key : Cuda.DevicePtr Float32
  value : Cuda.DevicePtr Float32
  decayLog : Cuda.DevicePtr Float32
  beta : Cuda.DevicePtr Float32
  stateInput : Cuda.DevicePtr Float32
  output : Cuda.DevicePtr Float32
  stateOutput : Cuda.DevicePtr Float32
  stateHistory : Cuda.DevicePtr Float32

@[struct] structure BatchedReverseBuffers where
  query : Cuda.DevicePtr Float32
  key : Cuda.DevicePtr Float32
  value : Cuda.DevicePtr Float32
  decayLog : Cuda.DevicePtr Float32
  beta : Cuda.DevicePtr Float32
  stateHistory : Cuda.DevicePtr Float32
  outputGradient : Cuda.DevicePtr Float32
  queryGradient : Cuda.DevicePtr Float32
  keyGradient : Cuda.DevicePtr Float32
  valueGradient : Cuda.DevicePtr Float32
  decayLogGradient : Cuda.DevicePtr Float32
  betaGradient : Cuda.DevicePtr Float32
  stateGradient : Cuda.DevicePtr Float32

@[struct] structure BatchedShape where
  batchSize : UInt32
  sequenceLength : UInt32
  recurrent : Shape

namespace Internal

@[always_inline]
def compactIndex (token head channel keyHeads keyWidth : UInt32) : UInt32 :=
  (token * keyHeads + head) * keyWidth + channel

@[always_inline]
def valueIndex (token head channel valueHeads valueWidth : UInt32) : UInt32 :=
  (token * valueHeads + head) * valueWidth + channel

@[always_inline]
def stateIndex (head row column keyWidth valueWidth : UInt32) : UInt32 :=
  (head * keyWidth + row) * valueWidth + column

@[always_inline]
def historyIndex (token head row column valueHeads keyWidth valueWidth : UInt32) : UInt32 :=
  ((token * valueHeads + head) * keyWidth + row) * valueWidth + column

@[always_inline]
def qkvWidth (shape : Shape) : UInt32 :=
  2 * shape.keyHeads * shape.keyWidth + shape.valueHeads * shape.valueWidth

@[always_inline]
def repeatedIndex (token head channel heads width : UInt32) : UInt32 :=
  (token * heads + head) * width + channel

@[always_inline]
partial def copyStateHeadToHistory (source history : Cuda.DevicePtr Float32)
    (token head valueHeads keyWidth valueWidth index laneStride : UInt32) : Cuda.DeviceM Unit := do
  let count := keyWidth * valueWidth
  if index < count then
    let row := index / valueWidth
    let column := index % valueWidth
    let sourceIndex := stateIndex head row column keyWidth valueWidth
    let destination := historyIndex token head row column valueHeads keyWidth valueWidth
    Cuda.storeFloat32 history destination.toUSize
      (← Cuda.loadFloat32 source sourceIndex.toUSize)
    copyStateHeadToHistory source history token head valueHeads keyWidth valueWidth
      (index + laneStride) laneStride

@[always_inline]
partial def copyStateHead (source destination : Cuda.DevicePtr Float32)
    (head keyWidth valueWidth index laneStride : UInt32) : Cuda.DeviceM Unit := do
  let count := keyWidth * valueWidth
  if index < count then
    let row := index / valueWidth
    let column := index % valueWidth
    let address := stateIndex head row column keyWidth valueWidth
    Cuda.storeFloat32 destination address.toUSize
      (← Cuda.loadFloat32 source address.toUSize)
    copyStateHead source destination head keyWidth valueWidth (index + laneStride) laneStride

@[always_inline]
partial def squareSumCompact (input : Cuda.DevicePtr Float32)
    (token head heads width channel laneStride : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if channel < width then
    let value ← Cuda.loadFloat32 input (compactIndex token head channel heads width).toUSize
    squareSumCompact input token head heads width (channel + laneStride) laneStride
      (Cuda.fma value value accumulator)
  else
    return accumulator

@[always_inline]
partial def publishNormalizedCompact (input output : Cuda.DevicePtr Float32)
    (token head heads width channel laneStride : UInt32) (inverse : Float32) :
    Cuda.DeviceM Unit := do
  if channel < width then
    let value ← Cuda.loadFloat32 input (compactIndex token head channel heads width).toUSize
    Cuda.storeFloat32 output channel.toUSize (value * inverse)
    publishNormalizedCompact input output token head heads width (channel + laneStride) laneStride
      inverse

@[always_inline]
partial def decayStateHead (state : Cuda.DevicePtr Float32)
    (head keyWidth valueWidth index laneStride : UInt32) (decay : Float32) :
    Cuda.DeviceM Unit := do
  let count := keyWidth * valueWidth
  if index < count then
    let row := index / valueWidth
    let column := index % valueWidth
    let address := stateIndex head row column keyWidth valueWidth
    let previous ← Cuda.loadFloat32 state address.toUSize
    Cuda.storeFloat32 state address.toUSize (previous * decay)
    decayStateHead state head keyWidth valueWidth (index + laneStride) laneStride decay

@[always_inline]
partial def stateKeyDot (state key : Cuda.DevicePtr Float32)
    (head column keyWidth valueWidth row : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if row < keyWidth then
    let stateValue ← Cuda.loadFloat32 state
      (stateIndex head row column keyWidth valueWidth).toUSize
    let keyValue ← Cuda.loadFloat32 key row.toUSize
    stateKeyDot state key head column keyWidth valueWidth (row + 1)
      (Cuda.fma stateValue keyValue accumulator)
  else
    return accumulator

@[always_inline]
partial def publishDelta (value beta : Cuda.DevicePtr Float32)
    (state key delta : Cuda.DevicePtr Float32)
    (token head valueHeads keyWidth valueWidth column laneStride : UInt32) :
    Cuda.DeviceM Unit := do
  if column < valueWidth then
    let memory ← stateKeyDot state key head column keyWidth valueWidth 0 zeroF32
    let projected ← Cuda.loadFloat32 value
      (valueIndex token head column valueHeads valueWidth).toUSize
    let coefficient ← Cuda.loadFloat32 beta (token * valueHeads + head).toUSize
    Cuda.storeFloat32 delta column.toUSize ((projected - memory) * coefficient)
    publishDelta value beta state key delta token head valueHeads keyWidth valueWidth
      (column + laneStride) laneStride

@[always_inline]
partial def updateStateHead (state key delta : Cuda.DevicePtr Float32)
    (head keyWidth valueWidth index laneStride : UInt32) : Cuda.DeviceM Unit := do
  let count := keyWidth * valueWidth
  if index < count then
    let row := index / valueWidth
    let column := index % valueWidth
    let address := stateIndex head row column keyWidth valueWidth
    let previous ← Cuda.loadFloat32 state address.toUSize
    let keyValue ← Cuda.loadFloat32 key row.toUSize
    let deltaValue ← Cuda.loadFloat32 delta column.toUSize
    Cuda.storeFloat32 state address.toUSize (Cuda.fma keyValue deltaValue previous)
    updateStateHead state key delta head keyWidth valueWidth (index + laneStride) laneStride

@[always_inline]
partial def stateQueryDot (state query : Cuda.DevicePtr Float32)
    (head column keyWidth valueWidth row : UInt32) (queryScale accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if row < keyWidth then
    let stateValue ← Cuda.loadFloat32 state
      (stateIndex head row column keyWidth valueWidth).toUSize
    let queryValue ← Cuda.loadFloat32 query row.toUSize
    stateQueryDot state query head column keyWidth valueWidth (row + 1) queryScale
      (Cuda.fma stateValue (queryValue * queryScale) accumulator)
  else
    return accumulator

@[always_inline]
partial def publishOutput (output state query : Cuda.DevicePtr Float32)
    (token head valueHeads keyWidth valueWidth column laneStride : UInt32)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  if column < valueWidth then
    let result ← stateQueryDot state query head column keyWidth valueWidth 0 queryScale zeroF32
    Cuda.storeFloat32 output
      (valueIndex token head column valueHeads valueWidth).toUSize result
    publishOutput output state query token head valueHeads keyWidth valueWidth
      (column + laneStride) laneStride queryScale

end Internal

/-- Stable exact Float32 softplus. -/
@[expose, cuda_device, always_inline]
def softplusF32 (value : Float32) : Float32 :=
  if value > zeroF32 then
    value + Float32.log (oneF32 + Float32.exp (-value))
  else
    Float32.log (oneF32 + Float32.exp value)

/-- Split post-convolution QKV storage into compact q/k and value-head tensors. -/
@[expose, cuda_device, always_inline]
def splitQkvElementF32 (qkv query key value : Cuda.DevicePtr Float32) (tokens : UInt32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  let keyElements := shape.keyHeads * shape.keyWidth
  let valueElements := shape.valueHeads * shape.valueWidth
  let qkvElements := tokens * (2 * keyElements + valueElements)
  if linear < qkvElements then
    let token := linear / (2 * keyElements + valueElements)
    let channel := linear % (2 * keyElements + valueElements)
    let projected ← Cuda.loadFloat32 qkv linear.toUSize
    if channel < keyElements then
      Cuda.storeFloat32 query (token * keyElements + channel).toUSize projected
    else if channel < 2 * keyElements then
      Cuda.storeFloat32 key (token * keyElements + channel - keyElements).toUSize projected
    else
      Cuda.storeFloat32 value (token * valueElements + channel - 2 * keyElements).toUSize
        projected

/-- Repeat compact q/k heads across the value-head groups. -/
@[expose, cuda_device, always_inline]
def repeatCompactElementF32 (compact repeated : Cuda.DevicePtr Float32) (tokens : UInt32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < tokens * shape.valueHeads * shape.keyWidth then
    let token := linear / (shape.valueHeads * shape.keyWidth)
    let rest := linear % (shape.valueHeads * shape.keyWidth)
    let valueHead := rest / shape.keyWidth
    let channel := rest % shape.keyWidth
    let compactHead := valueHead / (shape.valueHeads / shape.keyHeads)
    let source := Internal.compactIndex token compactHead channel shape.keyHeads shape.keyWidth
    Cuda.storeFloat32 repeated linear.toUSize (← Cuda.loadFloat32 compact source.toUSize)

namespace Internal

@[always_inline]
partial def repeatedGradientSum (gradient : Cuda.DevicePtr Float32) (token compactHead channel
    valueHeads keyHeads keyWidth repeatedHead endHead : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if repeatedHead < endHead then
    let index := repeatedIndex token repeatedHead channel valueHeads keyWidth
    let value ← Cuda.loadFloat32 gradient index.toUSize
    repeatedGradientSum gradient token compactHead channel valueHeads keyHeads keyWidth
      (repeatedHead + 1) endHead (accumulator + value)
  else
    return accumulator

@[always_inline]
partial def gateParameterGradients (a decayLogGradient : Cuda.DevicePtr Float32)
    (aLog dtBias : Cuda.DevicePtr Float32) (tokens valueHeads head token : UInt32)
    (aLogGradient dtBiasGradient : Float32) : Cuda.DeviceM GateParameterGradients := do
  if token < tokens then
    let index := token * valueHeads + head
    let aValue ← Cuda.loadFloat32 a index.toUSize
    let aLogValue ← Cuda.loadFloat32 aLog head.toUSize
    let dtValue ← Cuda.loadFloat32 dtBias head.toUSize
    let dg ← Cuda.loadFloat32 decayLogGradient index.toUSize
    let decayLog := -(Float32.exp aLogValue * softplusF32 (aValue + dtValue))
    let da := dg * (-(Float32.exp aLogValue)) * sigmoidF32 (aValue + dtValue)
    gateParameterGradients a decayLogGradient aLog dtBias tokens valueHeads head (token + 1)
      (aLogGradient + dg * decayLog) (dtBiasGradient + da)
  else
    return { aLog := aLogGradient, dtBias := dtBiasGradient }

end Internal

/-- Sum the repeated-head VJP back into compact q/k storage. -/
@[expose, cuda_device, always_inline]
def reduceRepeatedGradientElementF32 (repeatedGradient compactGradient :
    Cuda.DevicePtr Float32) (tokens : UInt32) (shape : Shape) (linear : UInt32) :
    Cuda.DeviceM Unit := do
  if linear < tokens * shape.keyHeads * shape.keyWidth then
    let token := linear / (shape.keyHeads * shape.keyWidth)
    let rest := linear % (shape.keyHeads * shape.keyWidth)
    let compactHead := rest / shape.keyWidth
    let channel := rest % shape.keyWidth
    let group := shape.valueHeads / shape.keyHeads
    let firstHead := compactHead * group
    let gradient ← Internal.repeatedGradientSum repeatedGradient token compactHead channel
      shape.valueHeads shape.keyHeads shape.keyWidth firstHead (firstHead + group) zeroF32
    Cuda.storeFloat32 compactGradient linear.toUSize gradient

/-- Concatenate compact q/k and value VJPs into post-convolution QKV layout. -/
@[expose, cuda_device, always_inline]
def concatenateQkvGradientElementF32 (queryGradient keyGradient valueGradient qkvGradient :
    Cuda.DevicePtr Float32) (tokens : UInt32) (shape : Shape) (linear : UInt32) :
    Cuda.DeviceM Unit := do
  let keyElements := shape.keyHeads * shape.keyWidth
  let valueElements := shape.valueHeads * shape.valueWidth
  let rowWidth := 2 * keyElements + valueElements
  if linear < tokens * rowWidth then
    let token := linear / rowWidth
    let channel := linear % rowWidth
    let gradient ← if channel < keyElements then
        Cuda.loadFloat32 queryGradient (token * keyElements + channel).toUSize
      else if channel < 2 * keyElements then
        Cuda.loadFloat32 keyGradient (token * keyElements + channel - keyElements).toUSize
      else
        Cuda.loadFloat32 valueGradient (token * valueElements + channel - 2 * keyElements).toUSize
    Cuda.storeFloat32 qkvGradient linear.toUSize gradient

/-- Compute `beta = sigmoid(b)` and the log decay `g`. -/
@[expose, cuda_device, always_inline]
def gateForwardElementF32 (a b aLog dtBias decayLog beta : Cuda.DevicePtr Float32)
    (tokens valueHeads linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < tokens * valueHeads then
    let head := linear % valueHeads
    let aValue ← Cuda.loadFloat32 a linear.toUSize
    let bValue ← Cuda.loadFloat32 b linear.toUSize
    let aLogValue ← Cuda.loadFloat32 aLog head.toUSize
    let dtValue ← Cuda.loadFloat32 dtBias head.toUSize
    Cuda.storeFloat32 beta linear.toUSize (sigmoidF32 bValue)
    Cuda.storeFloat32 decayLog linear.toUSize
      (-(Float32.exp aLogValue * softplusF32 (aValue + dtValue)))

/-- VJP from recurrent `d(g), d(beta)` to projection outputs `a,b`. -/
@[expose, cuda_device, always_inline]
def gateBackwardElementF32 (a b aLog dtBias decayLogGradient betaGradient aGradient bGradient :
    Cuda.DevicePtr Float32) (tokens valueHeads linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < tokens * valueHeads then
    let head := linear % valueHeads
    let aValue ← Cuda.loadFloat32 a linear.toUSize
    let bValue ← Cuda.loadFloat32 b linear.toUSize
    let aLogValue ← Cuda.loadFloat32 aLog head.toUSize
    let dtValue ← Cuda.loadFloat32 dtBias head.toUSize
    let dg ← Cuda.loadFloat32 decayLogGradient linear.toUSize
    let dbeta ← Cuda.loadFloat32 betaGradient linear.toUSize
    let betaValue := sigmoidF32 bValue
    Cuda.storeFloat32 aGradient linear.toUSize
      (dg * (-(Float32.exp aLogValue)) * sigmoidF32 (aValue + dtValue))
    Cuda.storeFloat32 bGradient linear.toUSize
      (dbeta * betaValue * (oneF32 - betaValue))

/-- Reduce `A_log` and `dt_bias` parameter gradients for one value head. -/
@[expose, cuda_device, always_inline]
def gateParameterGradientElementF32 (a decayLogGradient aLog dtBias aLogGradient
    dtBiasGradient : Cuda.DevicePtr Float32) (tokens valueHeads head : UInt32) :
    Cuda.DeviceM Unit := do
  if head < valueHeads then
    let gradients ← Internal.gateParameterGradients a decayLogGradient aLog dtBias tokens
      valueHeads head 0 zeroF32 zeroF32
    Cuda.storeFloat32 aLogGradient head.toUSize gradients.aLog
    Cuda.storeFloat32 dtBiasGradient head.toUSize gradients.dtBias

/-- Execute one recurrent token when q/k are already repeated and L2-normalized. -/
@[expose, cuda_device, always_inline, convergent]
def recurrentNormalizedTokenF32 (buffers : Buffers) (history delta : Cuda.DevicePtr Float32)
    (token head : UInt32) (shape : Shape) (queryScale : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let queryOffset := Internal.repeatedIndex token head 0 shape.valueHeads shape.keyWidth
  let keyOffset := Internal.repeatedIndex token head 0 shape.valueHeads shape.keyWidth
  let queryHead := buffers.query + (queryOffset * 4).toUSize
  let keyHead := buffers.key + (keyOffset * 4).toUSize
  let decay := Float32.exp
    (← Cuda.loadFloat32 buffers.decayLog (token * shape.valueHeads + head).toUSize)
  Internal.decayStateHead buffers.state head shape.keyWidth shape.valueWidth lane 32 decay
  Cuda.blockSync
  Internal.publishDelta buffers.value buffers.beta buffers.state keyHead delta token head
    shape.valueHeads shape.keyWidth shape.valueWidth lane 32
  Cuda.blockSync
  Internal.updateStateHead buffers.state keyHead delta head shape.keyWidth shape.valueWidth lane 32
  Cuda.blockSync
  Internal.publishOutput buffers.output buffers.state queryHead token head shape.valueHeads
    shape.keyWidth shape.valueWidth lane 32 queryScale
  Cuda.blockSync
  Internal.copyStateHeadToHistory buffers.state history (token + 1) head shape.valueHeads
    shape.keyWidth shape.valueWidth lane 32
  Cuda.blockSync

namespace Internal

@[always_inline]
partial def recurrentNormalizedTokensF32 (buffers : Buffers) (history delta :
    Cuda.DevicePtr Float32) (token tokens head : UInt32) (shape : Shape)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  if token < tokens then
    recurrentNormalizedTokenF32 buffers history delta token head shape queryScale
    recurrentNormalizedTokensF32 buffers history delta (token + 1) tokens head shape queryScale

end Internal

namespace Internal

@[always_inline]
partial def clearStateGradient (stateGradient : Cuda.DevicePtr Float32) (head : UInt32)
    (shape : Shape) (index laneStride : UInt32) : Cuda.DeviceM Unit := do
  if index < shape.keyWidth * shape.valueWidth then
    let row := index / shape.valueWidth
    let column := index % shape.valueWidth
    Cuda.storeFloat32 stateGradient
      (stateIndex head row column shape.keyWidth shape.valueWidth).toUSize zeroF32
    clearStateGradient stateGradient head shape (index + laneStride) laneStride

@[always_inline]
partial def historyKeyDot (history key : Cuda.DevicePtr Float32) (token head column row :
    UInt32) (shape : Shape) (decay accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < shape.keyWidth then
    let stateValue ← Cuda.loadFloat32 history
      (historyIndex token head row column shape.valueHeads shape.keyWidth shape.valueWidth).toUSize
    let keyValue ← Cuda.loadFloat32 key
      (repeatedIndex token head row shape.valueHeads shape.keyWidth).toUSize
    historyKeyDot history key token head column (row + 1) shape decay
      (Cuda.fma (stateValue * decay) keyValue accumulator)
  else
    return accumulator

@[always_inline]
partial def recomputeDelta (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head column laneStride : UInt32) (shape : Shape) (decay : Float32) :
    Cuda.DeviceM Unit := do
  if column < shape.valueWidth then
    let memory ← historyKeyDot buffers.stateHistory buffers.key token head column 0 shape decay zeroF32
    let value ← Cuda.loadFloat32 buffers.value
      (valueIndex token head column shape.valueHeads shape.valueWidth).toUSize
    let beta ← Cuda.loadFloat32 buffers.beta (token * shape.valueHeads + head).toUSize
    Cuda.storeFloat32 scratch.delta column.toUSize ((value - memory) * beta)
    recomputeDelta buffers scratch token head (column + laneStride) laneStride shape decay

@[always_inline]
partial def addOutputStateGradient (buffers : ReverseBuffers) (token head index laneStride :
    UInt32) (shape : Shape) (queryScale : Float32) : Cuda.DeviceM Unit := do
  if index < shape.keyWidth * shape.valueWidth then
    let row := index / shape.valueWidth
    let column := index % shape.valueWidth
    let address := stateIndex head row column shape.keyWidth shape.valueWidth
    let previous ← Cuda.loadFloat32 buffers.stateGradient address.toUSize
    let query ← Cuda.loadFloat32 buffers.query
      (repeatedIndex token head row shape.valueHeads shape.keyWidth).toUSize
    let gradient ← Cuda.loadFloat32 buffers.outputGradient
      (valueIndex token head column shape.valueHeads shape.valueWidth).toUSize
    Cuda.storeFloat32 buffers.stateGradient address.toUSize
      (Cuda.fma (query * queryScale) gradient previous)
    addOutputStateGradient buffers token head (index + laneStride) laneStride shape queryScale

@[always_inline]
partial def queryGradientDot (buffers : ReverseBuffers) (token head row column : UInt32)
    (shape : Shape) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < shape.valueWidth then
    let stateValue ← Cuda.loadFloat32 buffers.stateHistory
      (historyIndex (token + 1) head row column shape.valueHeads shape.keyWidth
        shape.valueWidth).toUSize
    let gradient ← Cuda.loadFloat32 buffers.outputGradient
      (valueIndex token head column shape.valueHeads shape.valueWidth).toUSize
    queryGradientDot buffers token head row (column + 1) shape
      (Cuda.fma stateValue gradient accumulator)
  else
    return accumulator

@[always_inline]
partial def publishQueryGradient (buffers : ReverseBuffers) (token head row laneStride : UInt32)
    (shape : Shape) (queryScale : Float32) : Cuda.DeviceM Unit := do
  if row < shape.keyWidth then
    let gradient ← queryGradientDot buffers token head row 0 shape zeroF32
    Cuda.storeFloat32 buffers.queryGradient
      (repeatedIndex token head row shape.valueHeads shape.keyWidth).toUSize
      (gradient * queryScale)
    publishQueryGradient buffers token head (row + laneStride) laneStride shape queryScale

@[always_inline]
partial def keyUpdateGradientDot (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (head row column : UInt32) (shape : Shape) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if column < shape.valueWidth then
    let stateGradient ← Cuda.loadFloat32 buffers.stateGradient
      (stateIndex head row column shape.keyWidth shape.valueWidth).toUSize
    let delta ← Cuda.loadFloat32 scratch.delta column.toUSize
    keyUpdateGradientDot buffers scratch head row (column + 1) shape
      (Cuda.fma stateGradient delta accumulator)
  else
    return accumulator

@[always_inline]
partial def publishKeyUpdateGradient (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head row laneStride : UInt32) (shape : Shape) : Cuda.DeviceM Unit := do
  if row < shape.keyWidth then
    let gradient ← keyUpdateGradientDot buffers scratch head row 0 shape zeroF32
    Cuda.storeFloat32 buffers.keyGradient
      (repeatedIndex token head row shape.valueHeads shape.keyWidth).toUSize gradient
    publishKeyUpdateGradient buffers scratch token head (row + laneStride) laneStride shape

@[always_inline]
partial def deltaGradientDot (buffers : ReverseBuffers) (token head column row : UInt32)
    (shape : Shape) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < shape.keyWidth then
    let stateGradient ← Cuda.loadFloat32 buffers.stateGradient
      (stateIndex head row column shape.keyWidth shape.valueWidth).toUSize
    let key ← Cuda.loadFloat32 buffers.key
      (repeatedIndex token head row shape.valueHeads shape.keyWidth).toUSize
    deltaGradientDot buffers token head column (row + 1) shape
      (Cuda.fma stateGradient key accumulator)
  else
    return accumulator

@[always_inline]
partial def publishDeltaGradient (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head column laneStride : UInt32) (shape : Shape) : Cuda.DeviceM Unit := do
  if column < shape.valueWidth then
    let gradient ← deltaGradientDot buffers token head column 0 shape zeroF32
    Cuda.storeFloat32 scratch.deltaGradient column.toUSize gradient
    publishDeltaGradient buffers scratch token head (column + laneStride) laneStride shape

@[always_inline]
partial def publishValueMemoryGradients (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head column laneStride : UInt32) (shape : Shape) (decay beta accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if column < shape.valueWidth then
    let deltaGradient ← Cuda.loadFloat32 scratch.deltaGradient column.toUSize
    let memory ← historyKeyDot buffers.stateHistory buffers.key token head column 0 shape decay zeroF32
    let value ← Cuda.loadFloat32 buffers.value
      (valueIndex token head column shape.valueHeads shape.valueWidth).toUSize
    let gradient := deltaGradient * beta
    Cuda.storeFloat32 buffers.valueGradient
      (valueIndex token head column shape.valueHeads shape.valueWidth).toUSize gradient
    Cuda.storeFloat32 scratch.memoryGradient column.toUSize (-gradient)
    publishValueMemoryGradients buffers scratch token head (column + laneStride) laneStride shape
      decay beta (Cuda.fma deltaGradient (value - memory) accumulator)
  else
    return accumulator

@[always_inline]
partial def addMemoryStateGradient (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head index laneStride : UInt32) (shape : Shape) : Cuda.DeviceM Unit := do
  if index < shape.keyWidth * shape.valueWidth then
    let row := index / shape.valueWidth
    let column := index % shape.valueWidth
    let address := stateIndex head row column shape.keyWidth shape.valueWidth
    let previous ← Cuda.loadFloat32 buffers.stateGradient address.toUSize
    let key ← Cuda.loadFloat32 buffers.key
      (repeatedIndex token head row shape.valueHeads shape.keyWidth).toUSize
    let memoryGradient ← Cuda.loadFloat32 scratch.memoryGradient column.toUSize
    Cuda.storeFloat32 buffers.stateGradient address.toUSize
      (Cuda.fma key memoryGradient previous)
    addMemoryStateGradient buffers scratch token head (index + laneStride) laneStride shape

@[always_inline]
partial def keyMemoryGradientDot (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head row column : UInt32) (shape : Shape) (decay accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if column < shape.valueWidth then
    let stateValue ← Cuda.loadFloat32 buffers.stateHistory
      (historyIndex token head row column shape.valueHeads shape.keyWidth shape.valueWidth).toUSize
    let memoryGradient ← Cuda.loadFloat32 scratch.memoryGradient column.toUSize
    keyMemoryGradientDot buffers scratch token head row (column + 1) shape decay
      (Cuda.fma (stateValue * decay) memoryGradient accumulator)
  else
    return accumulator

@[always_inline]
partial def addKeyMemoryGradient (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head row laneStride : UInt32) (shape : Shape) (decay : Float32) :
    Cuda.DeviceM Unit := do
  if row < shape.keyWidth then
    let index := repeatedIndex token head row shape.valueHeads shape.keyWidth
    let previous ← Cuda.loadFloat32 buffers.keyGradient index.toUSize
    let gradient ← keyMemoryGradientDot buffers scratch token head row 0 shape decay zeroF32
    Cuda.storeFloat32 buffers.keyGradient index.toUSize (previous + gradient)
    addKeyMemoryGradient buffers scratch token head (row + laneStride) laneStride shape decay

@[always_inline]
partial def decayGradientPartial (buffers : ReverseBuffers) (token head index laneStride : UInt32)
    (shape : Shape) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < shape.keyWidth * shape.valueWidth then
    let row := index / shape.valueWidth
    let column := index % shape.valueWidth
    let stateGradient ← Cuda.loadFloat32 buffers.stateGradient
      (stateIndex head row column shape.keyWidth shape.valueWidth).toUSize
    let stateValue ← Cuda.loadFloat32 buffers.stateHistory
      (historyIndex token head row column shape.valueHeads shape.keyWidth shape.valueWidth).toUSize
    decayGradientPartial buffers token head (index + laneStride) laneStride shape
      (Cuda.fma stateGradient stateValue accumulator)
  else
    return accumulator

@[always_inline]
partial def scaleStateGradient (stateGradient : Cuda.DevicePtr Float32) (head index laneStride :
    UInt32) (shape : Shape) (decay : Float32) : Cuda.DeviceM Unit := do
  if index < shape.keyWidth * shape.valueWidth then
    let row := index / shape.valueWidth
    let column := index % shape.valueWidth
    let address := stateIndex head row column shape.keyWidth shape.valueWidth
    let gradient ← Cuda.loadFloat32 stateGradient address.toUSize
    Cuda.storeFloat32 stateGradient address.toUSize (gradient * decay)
    scaleStateGradient stateGradient head (index + laneStride) laneStride shape decay

end Internal

/-- Reverse one normalized recurrent token for one value head. -/
@[expose, cuda_device, always_inline, convergent]
def recurrentNormalizedTokenBackwardF32 (buffers : ReverseBuffers) (scratch : ReverseScratch)
    (token head : UInt32) (shape : Shape) (queryScale : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let decay := Float32.exp
    (← Cuda.loadFloat32 buffers.decayLog (token * shape.valueHeads + head).toUSize)
  let beta ← Cuda.loadFloat32 buffers.beta (token * shape.valueHeads + head).toUSize
  Internal.recomputeDelta buffers scratch token head lane 32 shape decay
  Cuda.blockSync
  Internal.addOutputStateGradient buffers token head lane 32 shape queryScale
  Cuda.blockSync
  Internal.publishQueryGradient buffers token head lane 32 shape queryScale
  Internal.publishKeyUpdateGradient buffers scratch token head lane 32 shape
  Cuda.blockSync
  Internal.publishDeltaGradient buffers scratch token head lane 32 shape
  Cuda.blockSync
  let localBetaGradient ← Internal.publishValueMemoryGradients buffers scratch token head lane 32
    shape decay beta zeroF32
  let betaGradient ← warpSumBroadcast localBetaGradient
  if lane == 0 then
    Cuda.storeFloat32 buffers.betaGradient (token * shape.valueHeads + head).toUSize betaGradient
  Cuda.blockSync
  Internal.addMemoryStateGradient buffers scratch token head lane 32 shape
  Cuda.blockSync
  Internal.addKeyMemoryGradient buffers scratch token head lane 32 shape decay
  let localDecayGradient ← Internal.decayGradientPartial buffers token head lane 32 shape zeroF32
  let decayGradient ← warpSumBroadcast localDecayGradient
  if lane == 0 then
    Cuda.storeFloat32 buffers.decayLogGradient (token * shape.valueHeads + head).toUSize
      (decayGradient * decay)
  Internal.scaleStateGradient buffers.stateGradient head lane 32 shape decay
  Cuda.blockSync

namespace Internal

@[always_inline]
partial def recurrentNormalizedTokensBackwardF32 (buffers : ReverseBuffers)
    (scratch : ReverseScratch) (remaining head : UInt32) (shape : Shape)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  if remaining != 0 then
    recurrentNormalizedTokenBackwardF32 buffers scratch (remaining - 1) head shape queryScale
    recurrentNormalizedTokensBackwardF32 buffers scratch (remaining - 1) head shape queryScale

end Internal

/--
Execute one recurrent delta-rule token for one value head.

All 32 lanes of the calling block participate. `query` and `key` are compact raw post-conv
projections; this body performs the per-head L2 normalization before applying the recurrent rule.
-/
@[expose, cuda_device, always_inline, convergent]
def recurrentTokenF32 (buffers : Buffers) (scratch : Scratch) (token head : UInt32) (shape : Shape)
    (epsilon queryScale : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let compactHead := head / (shape.valueHeads / shape.keyHeads)
  let qSquares ← warpSumBroadcast
    (← Internal.squareSumCompact buffers.query token compactHead shape.keyHeads shape.keyWidth lane
      32 zeroF32)
  let kSquares ← warpSumBroadcast
    (← Internal.squareSumCompact buffers.key token compactHead shape.keyHeads shape.keyWidth lane
      32 zeroF32)
  let qInverse := oneF32 / Float32.sqrt (qSquares + epsilon)
  let kInverse := oneF32 / Float32.sqrt (kSquares + epsilon)
  Internal.publishNormalizedCompact buffers.query scratch.query token compactHead shape.keyHeads
    shape.keyWidth lane 32 qInverse
  Internal.publishNormalizedCompact buffers.key scratch.key token compactHead shape.keyHeads
    shape.keyWidth lane 32 kInverse
  Cuda.blockSync
  let decay := Float32.exp
    (← Cuda.loadFloat32 buffers.decayLog (token * shape.valueHeads + head).toUSize)
  Internal.decayStateHead buffers.state head shape.keyWidth shape.valueWidth lane 32 decay
  Cuda.blockSync
  Internal.publishDelta buffers.value buffers.beta buffers.state scratch.key scratch.delta token
    head shape.valueHeads shape.keyWidth shape.valueWidth lane 32
  Cuda.blockSync
  Internal.updateStateHead buffers.state scratch.key scratch.delta head shape.keyWidth
    shape.valueWidth lane 32
  Cuda.blockSync
  Internal.publishOutput buffers.output buffers.state scratch.query token head shape.valueHeads
    shape.keyWidth shape.valueWidth lane 32 queryScale
  Cuda.blockSync

namespace Internal

@[always_inline]
partial def recurrentTokensF32 (buffers : Buffers) (scratch : Scratch)
    (token tokens head : UInt32) (shape : Shape)
    (epsilon queryScale : Float32) : Cuda.DeviceM Unit := do
  if token < tokens then
    recurrentTokenF32 buffers scratch token head shape epsilon queryScale
    recurrentTokensF32 buffers scratch (token + 1) tokens head shape epsilon queryScale

end Internal

/-- Single-token recurrent decode with explicit input and output state buffers. -/
@[cuda_kernel]
def recurrentDecodeF32Kernel (query key value decayLog beta stateInput output stateOutput :
    Cuda.DevicePtr Float32) (keyHeads valueHeads keyWidth valueWidth : UInt32)
    (epsilon queryScale : Float32) : Cuda.DeviceM Unit := do
  let head ← Cuda.blockIdxX
  let lane ← Cuda.laneId
  if head < valueHeads then
    let queryShared ← Cuda.dynamicShared (α := Float32)
    let keyShared ← Cuda.dynamicShared (α := Float32) (keyWidth * 4).toUSize
    let deltaShared ← Cuda.dynamicShared (α := Float32) (keyWidth * 8).toUSize
    Internal.copyStateHead stateInput stateOutput head keyWidth valueWidth lane 32
    Cuda.blockSync
    let buffers : Buffers := { query, key, value, decayLog, beta, output, state := stateOutput }
    let shape : Shape := { keyHeads, valueHeads, keyWidth, valueWidth }
    let scratch : Scratch := { query := queryShared, key := keyShared, delta := deltaShared }
    recurrentTokenF32 buffers scratch 0 head shape epsilon queryScale

/--
Multi-token recurrent prefill baseline.

This follows the same mathematical recurrence as the HF chunked rule and is retained as the
independent device baseline for the optimized chunk-64 implementation.
-/
@[cuda_kernel]
def recurrentPrefillF32Kernel (query key value decayLog beta stateInput output stateOutput :
    Cuda.DevicePtr Float32) (tokens keyHeads valueHeads keyWidth valueWidth : UInt32)
    (epsilon queryScale : Float32) : Cuda.DeviceM Unit := do
  let head ← Cuda.blockIdxX
  let lane ← Cuda.laneId
  if head < valueHeads then
    let queryShared ← Cuda.dynamicShared (α := Float32)
    let keyShared ← Cuda.dynamicShared (α := Float32) (keyWidth * 4).toUSize
    let deltaShared ← Cuda.dynamicShared (α := Float32) (keyWidth * 8).toUSize
    Internal.copyStateHead stateInput stateOutput head keyWidth valueWidth lane 32
    Cuda.blockSync
    let buffers : Buffers := { query, key, value, decayLog, beta, output, state := stateOutput }
    let shape : Shape := { keyHeads, valueHeads, keyWidth, valueWidth }
    let scratch : Scratch := { query := queryShared, key := keyShared, delta := deltaShared }
    Internal.recurrentTokensF32 buffers scratch 0 tokens head shape epsilon queryScale

@[cuda_kernel]
def splitQkvF32Kernel (qkv query key value : Cuda.DevicePtr Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : Cuda.DeviceM Unit := do
  splitQkvElementF32 qkv query key value tokens { keyHeads, valueHeads, keyWidth, valueWidth }
    (← elementIndex)

@[cuda_kernel]
def repeatCompactF32Kernel (compact repeated : Cuda.DevicePtr Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : Cuda.DeviceM Unit := do
  repeatCompactElementF32 compact repeated tokens { keyHeads, valueHeads, keyWidth, valueWidth }
    (← elementIndex)

@[cuda_kernel]
def reduceRepeatedGradientF32Kernel (repeatedGradient compactGradient :
    Cuda.DevicePtr Float32) (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) :
    Cuda.DeviceM Unit := do
  reduceRepeatedGradientElementF32 repeatedGradient compactGradient tokens
    { keyHeads, valueHeads, keyWidth, valueWidth } (← elementIndex)

@[cuda_kernel]
def concatenateQkvGradientF32Kernel (queryGradient keyGradient valueGradient qkvGradient :
    Cuda.DevicePtr Float32) (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) :
    Cuda.DeviceM Unit := do
  concatenateQkvGradientElementF32 queryGradient keyGradient valueGradient qkvGradient tokens
    { keyHeads, valueHeads, keyWidth, valueWidth } (← elementIndex)

@[cuda_kernel]
def gateForwardF32Kernel (a b aLog dtBias decayLog beta : Cuda.DevicePtr Float32)
    (tokens valueHeads : UInt32) : Cuda.DeviceM Unit := do
  gateForwardElementF32 a b aLog dtBias decayLog beta tokens valueHeads (← elementIndex)

@[cuda_kernel]
def gateBackwardF32Kernel (a b aLog dtBias decayLogGradient betaGradient aGradient bGradient :
    Cuda.DevicePtr Float32) (tokens valueHeads : UInt32) : Cuda.DeviceM Unit := do
  gateBackwardElementF32 a b aLog dtBias decayLogGradient betaGradient aGradient bGradient
    tokens valueHeads (← elementIndex)

@[cuda_kernel]
def gateParameterGradientF32Kernel (a decayLogGradient aLog dtBias aLogGradient dtBiasGradient :
    Cuda.DevicePtr Float32) (tokens valueHeads : UInt32) : Cuda.DeviceM Unit := do
  gateParameterGradientElementF32 a decayLogGradient aLog dtBias aLogGradient dtBiasGradient
    tokens valueHeads (← elementIndex)

/-- Recurrent prefill over already-normalized repeated q/k, retaining all state snapshots. -/
@[cuda_kernel]
def recurrentNormalizedPrefillF32Kernel (query key value decayLog beta stateInput output stateOutput
    stateHistory : Cuda.DevicePtr Float32) (tokens keyHeads valueHeads keyWidth valueWidth : UInt32)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  let head ← Cuda.blockIdxX
  let lane ← Cuda.laneId
  if head < valueHeads then
    let delta ← Cuda.dynamicShared (α := Float32)
    Internal.copyStateHead stateInput stateOutput head keyWidth valueWidth lane 32
    Cuda.blockSync
    Internal.copyStateHeadToHistory stateOutput stateHistory 0 head valueHeads keyWidth valueWidth
      lane 32
    Cuda.blockSync
    let buffers : Buffers := { query, key, value, decayLog, beta, output, state := stateOutput }
    let shape : Shape := { keyHeads, valueHeads, keyWidth, valueWidth }
    Internal.recurrentNormalizedTokensF32 buffers stateHistory delta 0 tokens head shape queryScale

/-- Exact reverse recurrence over saved state history. -/
@[cuda_kernel]
def recurrentNormalizedBackwardF32Kernel (query key value decayLog beta stateHistory outputGradient
    queryGradient keyGradient valueGradient decayLogGradient betaGradient stateGradient :
    Cuda.DevicePtr Float32) (tokens keyHeads valueHeads keyWidth valueWidth : UInt32)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  let head ← Cuda.blockIdxX
  let lane ← Cuda.laneId
  if head < valueHeads then
    let delta ← Cuda.dynamicShared (α := Float32)
    let deltaGradient ← Cuda.dynamicShared (α := Float32) (valueWidth * 4).toUSize
    let memoryGradient ← Cuda.dynamicShared (α := Float32) (valueWidth * 8).toUSize
    let shape : Shape := { keyHeads, valueHeads, keyWidth, valueWidth }
    let buffers : ReverseBuffers := {
      query, key, value, decayLog, beta, stateHistory, outputGradient, queryGradient, keyGradient,
      valueGradient, decayLogGradient, betaGradient, stateGradient
    }
    let scratch : ReverseScratch := { delta, deltaGradient, memoryGradient }
    Internal.clearStateGradient stateGradient head shape lane 32
    Cuda.blockSync
    Internal.recurrentNormalizedTokensBackwardF32 buffers scratch tokens head shape queryScale

/-- Batch-isolated normalized recurrence with distinct state/history for every sequence. -/
@[expose, cuda_device, always_inline, convergent]
def recurrentNormalizedBatchHeadF32 (batchBuffers : BatchedForwardBuffers)
    (shared : Cuda.DevicePtr Float32) (batchShape : BatchedShape) (linearHead : UInt32)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  let { batchSize, sequenceLength, recurrent := shape } := batchShape
  let { keyHeads, valueHeads, keyWidth, valueWidth } := shape
  if linearHead < batchSize * valueHeads then
    let lane ← Cuda.laneId
    let batch := linearHead / valueHeads
    let head := linearHead % valueHeads
    let keyElements := sequenceLength * valueHeads * keyWidth
    let valueElements := sequenceLength * valueHeads * valueWidth
    let gateElements := sequenceLength * valueHeads
    let stateElements := valueHeads * keyWidth * valueWidth
    let queryBatch : Cuda.DevicePtr Float32 := batchBuffers.query +
      (batch * keyElements * 4).toUSize
    let keyBatch : Cuda.DevicePtr Float32 := batchBuffers.key + (batch * keyElements * 4).toUSize
    let valueBatch : Cuda.DevicePtr Float32 := batchBuffers.value +
      (batch * valueElements * 4).toUSize
    let decayBatch : Cuda.DevicePtr Float32 := batchBuffers.decayLog +
      (batch * gateElements * 4).toUSize
    let betaBatch : Cuda.DevicePtr Float32 := batchBuffers.beta +
      (batch * gateElements * 4).toUSize
    let stateInputBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.stateInput + (batch * stateElements * 4).toUSize
    let outputBatch : Cuda.DevicePtr Float32 := batchBuffers.output +
      (batch * valueElements * 4).toUSize
    let stateOutputBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.stateOutput + (batch * stateElements * 4).toUSize
    let historyElements := (sequenceLength + 1) * stateElements
    let historyBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.stateHistory + (batch * historyElements * 4).toUSize
    Internal.copyStateHead stateInputBatch stateOutputBatch head keyWidth valueWidth lane 32
    Cuda.blockSync
    Internal.copyStateHeadToHistory stateOutputBatch historyBatch 0 head valueHeads keyWidth
      valueWidth lane 32
    Cuda.blockSync
    let buffers : Buffers := {
      query := queryBatch, key := keyBatch, value := valueBatch, decayLog := decayBatch,
      beta := betaBatch, output := outputBatch, state := stateOutputBatch
    }
    Internal.recurrentNormalizedTokensF32 buffers historyBatch shared 0 sequenceLength head shape
      queryScale

@[cuda_kernel]
def recurrentNormalizedBatchedPrefillF32Kernel (query key value decayLog beta stateInput output
    stateOutput stateHistory : Cuda.DevicePtr Float32)
    (batchSize sequenceLength keyHeads valueHeads keyWidth valueWidth : UInt32)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  let linearHead ← Cuda.blockIdxX
  let shared ← Cuda.dynamicShared (α := Float32)
  recurrentNormalizedBatchHeadF32 {
    query, key, value, decayLog, beta, stateInput, output, stateOutput, stateHistory
  } shared { batchSize, sequenceLength, recurrent := { keyHeads, valueHeads, keyWidth, valueWidth } }
    linearHead queryScale

/-- Exact reverse recurrence with independent sequence histories and state gradients. -/
@[expose, cuda_device, always_inline, convergent]
def recurrentNormalizedBatchHeadBackwardF32 (batchBuffers : BatchedReverseBuffers)
    (shared : Cuda.DevicePtr Float32) (batchShape : BatchedShape) (linearHead : UInt32)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  let { batchSize, sequenceLength, recurrent := shape } := batchShape
  let { keyHeads, valueHeads, keyWidth, valueWidth } := shape
  if linearHead < batchSize * valueHeads then
    let lane ← Cuda.laneId
    let batch := linearHead / valueHeads
    let head := linearHead % valueHeads
    let keyElements := sequenceLength * valueHeads * keyWidth
    let valueElements := sequenceLength * valueHeads * valueWidth
    let gateElements := sequenceLength * valueHeads
    let stateElements := valueHeads * keyWidth * valueWidth
    let historyElements := (sequenceLength + 1) * stateElements
    let queryBatch : Cuda.DevicePtr Float32 := batchBuffers.query +
      (batch * keyElements * 4).toUSize
    let keyBatch : Cuda.DevicePtr Float32 := batchBuffers.key + (batch * keyElements * 4).toUSize
    let valueBatch : Cuda.DevicePtr Float32 := batchBuffers.value +
      (batch * valueElements * 4).toUSize
    let decayBatch : Cuda.DevicePtr Float32 := batchBuffers.decayLog +
      (batch * gateElements * 4).toUSize
    let betaBatch : Cuda.DevicePtr Float32 := batchBuffers.beta +
      (batch * gateElements * 4).toUSize
    let historyBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.stateHistory + (batch * historyElements * 4).toUSize
    let outputGradientBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.outputGradient + (batch * valueElements * 4).toUSize
    let queryGradientBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.queryGradient + (batch * keyElements * 4).toUSize
    let keyGradientBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.keyGradient + (batch * keyElements * 4).toUSize
    let valueGradientBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.valueGradient + (batch * valueElements * 4).toUSize
    let decayGradientBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.decayLogGradient + (batch * gateElements * 4).toUSize
    let betaGradientBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.betaGradient + (batch * gateElements * 4).toUSize
    let stateGradientBatch : Cuda.DevicePtr Float32 :=
      batchBuffers.stateGradient + (batch * stateElements * 4).toUSize
    let delta := shared
    let deltaGradient := shared + (valueWidth * 4).toUSize
    let memoryGradient := shared + (valueWidth * 8).toUSize
    let buffers : ReverseBuffers := {
      query := queryBatch, key := keyBatch, value := valueBatch, decayLog := decayBatch,
      beta := betaBatch, stateHistory := historyBatch, outputGradient := outputGradientBatch,
      queryGradient := queryGradientBatch, keyGradient := keyGradientBatch,
      valueGradient := valueGradientBatch, decayLogGradient := decayGradientBatch,
      betaGradient := betaGradientBatch, stateGradient := stateGradientBatch
    }
    let scratch : ReverseScratch := { delta, deltaGradient, memoryGradient }
    Internal.clearStateGradient stateGradientBatch head shape lane 32
    Cuda.blockSync
    Internal.recurrentNormalizedTokensBackwardF32 buffers scratch sequenceLength head shape
      queryScale

@[cuda_kernel]
def recurrentNormalizedBatchedBackwardF32Kernel (query key value decayLog beta stateHistory
    outputGradient queryGradient keyGradient valueGradient decayLogGradient betaGradient
    stateGradient : Cuda.DevicePtr Float32)
    (batchSize sequenceLength keyHeads valueHeads keyWidth valueWidth : UInt32)
    (queryScale : Float32) : Cuda.DeviceM Unit := do
  let linearHead ← Cuda.blockIdxX
  let shared ← Cuda.dynamicShared (α := Float32)
  recurrentNormalizedBatchHeadBackwardF32 {
    query, key, value, decayLog, beta, stateHistory, outputGradient, queryGradient, keyGradient,
    valueGradient, decayLogGradient, betaGradient, stateGradient
  } shared { batchSize, sequenceLength, recurrent := { keyHeads, valueHeads, keyWidth, valueWidth } }
    linearHead queryScale

/-- Launch geometry for one-warp-per-value-head recurrent kernels. -/
def recurrentConfig (valueHeads keyWidth valueWidth : UInt32) : Cuda.LaunchConfig := {
  grid := { x := valueHeads }
  block := { x := 32 }
  sharedMemoryBytes := ((keyWidth * 2 + valueWidth) * 4).toUSize
  blockArenaBytes := 0
}

/-- Launch one recurrent decode token. -/
def recurrentDecodeF32 (stream : @& Cuda.Stream)
    (query key value decayLog beta stateInput output stateOutput : @& Cuda.Buffer Float32)
    (keyHeads valueHeads keyWidth valueWidth : UInt32) (epsilon queryScale : Float32) :
    IO Cuda.KernelHandle :=
  recurrentDecodeF32Kernel.launchOn stream (recurrentConfig valueHeads keyWidth valueWidth)
    query key value decayLog beta stateInput output stateOutput keyHeads valueHeads keyWidth
    valueWidth epsilon queryScale

/-- Launch the recurrent multi-token prefill baseline. -/
def recurrentPrefillF32 (stream : @& Cuda.Stream)
    (query key value decayLog beta stateInput output stateOutput : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) (epsilon queryScale : Float32) :
    IO Cuda.KernelHandle :=
  recurrentPrefillF32Kernel.launchOn stream (recurrentConfig valueHeads keyWidth valueWidth)
    query key value decayLog beta stateInput output stateOutput tokens keyHeads valueHeads keyWidth
    valueWidth epsilon queryScale

def splitQkvF32 (stream : @& Cuda.Stream) (qkv query key value : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Cuda.KernelHandle :=
  splitQkvF32Kernel.launchOn stream
    (elementConfig (tokens * (2 * keyHeads * keyWidth + valueHeads * valueWidth))) qkv query key
    value tokens keyHeads valueHeads keyWidth valueWidth

def repeatCompactF32 (stream : @& Cuda.Stream) (compact repeated : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Cuda.KernelHandle :=
  repeatCompactF32Kernel.launchOn stream (elementConfig (tokens * valueHeads * keyWidth)) compact
    repeated tokens keyHeads valueHeads keyWidth valueWidth

def reduceRepeatedGradientF32 (stream : @& Cuda.Stream)
    (repeatedGradient compactGradient : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Cuda.KernelHandle :=
  reduceRepeatedGradientF32Kernel.launchOn stream
    (elementConfig (tokens * keyHeads * keyWidth)) repeatedGradient compactGradient tokens keyHeads
    valueHeads keyWidth valueWidth

def concatenateQkvGradientF32 (stream : @& Cuda.Stream)
    (queryGradient keyGradient valueGradient qkvGradient : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Cuda.KernelHandle :=
  concatenateQkvGradientF32Kernel.launchOn stream
    (elementConfig (tokens * (2 * keyHeads * keyWidth + valueHeads * valueWidth))) queryGradient
    keyGradient valueGradient qkvGradient tokens keyHeads valueHeads keyWidth valueWidth

def gateForwardF32 (stream : @& Cuda.Stream) (a b aLog dtBias decayLog beta :
    @& Cuda.Buffer Float32) (tokens valueHeads : UInt32) : IO Cuda.KernelHandle :=
  gateForwardF32Kernel.launchOn stream (elementConfig (tokens * valueHeads)) a b aLog dtBias
    decayLog beta tokens valueHeads

def gateBackwardF32 (stream : @& Cuda.Stream)
    (a b aLog dtBias decayLogGradient betaGradient aGradient bGradient :
      @& Cuda.Buffer Float32)
    (tokens valueHeads : UInt32) : IO Cuda.KernelHandle :=
  gateBackwardF32Kernel.launchOn stream (elementConfig (tokens * valueHeads)) a b aLog dtBias
    decayLogGradient betaGradient aGradient bGradient tokens valueHeads

def gateParameterGradientF32 (stream : @& Cuda.Stream)
    (a decayLogGradient aLog dtBias aLogGradient dtBiasGradient : @& Cuda.Buffer Float32)
    (tokens valueHeads : UInt32) : IO Cuda.KernelHandle :=
  gateParameterGradientF32Kernel.launchOn stream (elementConfig valueHeads) a decayLogGradient
    aLog dtBias aLogGradient dtBiasGradient tokens valueHeads

def recurrentNormalizedPrefillF32 (stream : @& Cuda.Stream)
    (query key value decayLog beta stateInput output stateOutput stateHistory :
      @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) (queryScale : Float32) :
    IO Cuda.KernelHandle :=
  recurrentNormalizedPrefillF32Kernel.launchOn stream {
    grid := { x := valueHeads }
    block := { x := 32 }
    sharedMemoryBytes := (valueWidth * 4).toUSize
    blockArenaBytes := 0
  } query key value decayLog beta stateInput output stateOutput stateHistory tokens keyHeads
    valueHeads keyWidth valueWidth queryScale

def recurrentNormalizedBackwardF32 (stream : @& Cuda.Stream)
    (query key value decayLog beta stateHistory outputGradient queryGradient keyGradient
      valueGradient decayLogGradient betaGradient stateGradient : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) (queryScale : Float32) :
    IO Cuda.KernelHandle :=
  recurrentNormalizedBackwardF32Kernel.launchOn stream {
    grid := { x := valueHeads }
    block := { x := 32 }
    sharedMemoryBytes := (valueWidth * 12).toUSize
    blockArenaBytes := 0
  } query key value decayLog beta stateHistory outputGradient queryGradient keyGradient
    valueGradient decayLogGradient betaGradient stateGradient tokens keyHeads valueHeads keyWidth
    valueWidth queryScale

/-! ## Shared sequential/device-graph submissions -/

def submitSplitQkvF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (qkv query key value : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * (2 * keyHeads * keyWidth + valueHeads * valueWidth))
  executor.submit label
    (fun stream => splitQkvF32Kernel.launchOn stream launch qkv query key value tokens keyHeads
      valueHeads keyWidth valueWidth)
    (fun builder => splitQkvF32Kernel.addToGraph builder launch qkv query key value tokens keyHeads
      valueHeads keyWidth valueWidth)

def submitRepeatCompactF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (compact repeated : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * valueHeads * keyWidth)
  executor.submit label
    (fun stream => repeatCompactF32Kernel.launchOn stream launch compact repeated tokens keyHeads
      valueHeads keyWidth valueWidth)
    (fun builder => repeatCompactF32Kernel.addToGraph builder launch compact repeated tokens keyHeads
      valueHeads keyWidth valueWidth)

def submitReduceRepeatedGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (repeatedGradient compactGradient : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * keyHeads * keyWidth)
  executor.submit label
    (fun stream => reduceRepeatedGradientF32Kernel.launchOn stream launch repeatedGradient
      compactGradient tokens keyHeads valueHeads keyWidth valueWidth)
    (fun builder => reduceRepeatedGradientF32Kernel.addToGraph builder launch repeatedGradient
      compactGradient tokens keyHeads valueHeads keyWidth valueWidth)

def submitConcatenateQkvGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (queryGradient keyGradient valueGradient qkvGradient : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * (2 * keyHeads * keyWidth + valueHeads * valueWidth))
  executor.submit label
    (fun stream => concatenateQkvGradientF32Kernel.launchOn stream launch queryGradient
      keyGradient valueGradient qkvGradient tokens keyHeads valueHeads keyWidth valueWidth)
    (fun builder => concatenateQkvGradientF32Kernel.addToGraph builder launch queryGradient
      keyGradient valueGradient qkvGradient tokens keyHeads valueHeads keyWidth valueWidth)

def submitGateForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (a b aLog dtBias decayLog beta : @& Cuda.Buffer Float32)
    (tokens valueHeads : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * valueHeads)
  executor.submit label
    (fun stream => gateForwardF32Kernel.launchOn stream launch a b aLog dtBias decayLog beta tokens
      valueHeads)
    (fun builder => gateForwardF32Kernel.addToGraph builder launch a b aLog dtBias decayLog beta
      tokens valueHeads)

def submitGateBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (a b aLog dtBias decayLogGradient betaGradient aGradient bGradient :
      @& Cuda.Buffer Float32) (tokens valueHeads : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * valueHeads)
  executor.submit label
    (fun stream => gateBackwardF32Kernel.launchOn stream launch a b aLog dtBias decayLogGradient
      betaGradient aGradient bGradient tokens valueHeads)
    (fun builder => gateBackwardF32Kernel.addToGraph builder launch a b aLog dtBias
      decayLogGradient betaGradient aGradient bGradient tokens valueHeads)

def submitGateParameterGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (a decayLogGradient aLog dtBias aLogGradient dtBiasGradient : @& Cuda.Buffer Float32)
    (tokens valueHeads : UInt32) : IO Unit :=
  let launch := elementConfig valueHeads
  executor.submit label
    (fun stream => gateParameterGradientF32Kernel.launchOn stream launch a decayLogGradient aLog
      dtBias aLogGradient dtBiasGradient tokens valueHeads)
    (fun builder => gateParameterGradientF32Kernel.addToGraph builder launch a decayLogGradient aLog
      dtBias aLogGradient dtBiasGradient tokens valueHeads)

def submitRecurrentNormalizedPrefillF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (query key value decayLog beta stateInput output stateOutput stateHistory :
      @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) (queryScale : Float32) : IO Unit :=
  let launch : Cuda.LaunchConfig := {
    grid := { x := valueHeads }
    block := { x := 32 }
    sharedMemoryBytes := (valueWidth * 4).toUSize
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => recurrentNormalizedPrefillF32Kernel.launchOn stream launch query key value
      decayLog beta stateInput output stateOutput stateHistory tokens keyHeads valueHeads keyWidth
      valueWidth queryScale)
    (fun builder => recurrentNormalizedPrefillF32Kernel.addToGraph builder launch query key value
      decayLog beta stateInput output stateOutput stateHistory tokens keyHeads valueHeads keyWidth
      valueWidth queryScale)

def submitRecurrentNormalizedBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (query key value decayLog beta stateHistory outputGradient queryGradient keyGradient
      valueGradient decayLogGradient betaGradient stateGradient : @& Cuda.Buffer Float32)
    (tokens keyHeads valueHeads keyWidth valueWidth : UInt32) (queryScale : Float32) : IO Unit :=
  let launch : Cuda.LaunchConfig := {
    grid := { x := valueHeads }
    block := { x := 32 }
    sharedMemoryBytes := (valueWidth * 12).toUSize
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => recurrentNormalizedBackwardF32Kernel.launchOn stream launch query key value
      decayLog beta stateHistory outputGradient queryGradient keyGradient valueGradient
      decayLogGradient betaGradient stateGradient tokens keyHeads valueHeads keyWidth valueWidth
      queryScale)
    (fun builder => recurrentNormalizedBackwardF32Kernel.addToGraph builder launch query key value
      decayLog beta stateHistory outputGradient queryGradient keyGradient valueGradient
      decayLogGradient betaGradient stateGradient tokens keyHeads valueHeads keyWidth valueWidth
      queryScale)

def submitRecurrentNormalizedBatchedPrefillF32 (executor : @& Cuda.Qwen36.Executor)
    (label : String) (query key value decayLog beta stateInput output stateOutput stateHistory :
      @& Cuda.Buffer Float32) (layout : Cuda.Qwen36.SequenceLayout)
    (keyHeads valueHeads keyWidth valueWidth : UInt32) (queryScale : Float32) : IO Unit :=
  let launch : Cuda.LaunchConfig := {
    grid := { x := layout.batchSize * valueHeads }
    block := { x := 32 }
    sharedMemoryBytes := (valueWidth * 4).toUSize
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => recurrentNormalizedBatchedPrefillF32Kernel.launchOn stream launch query key value
      decayLog beta stateInput output stateOutput stateHistory layout.batchSize
      layout.sequenceLength keyHeads valueHeads keyWidth valueWidth queryScale)
    (fun builder => recurrentNormalizedBatchedPrefillF32Kernel.addToGraph builder launch query key
      value decayLog beta stateInput output stateOutput stateHistory layout.batchSize
      layout.sequenceLength keyHeads valueHeads keyWidth valueWidth queryScale)

def submitRecurrentNormalizedBatchedBackwardF32 (executor : @& Cuda.Qwen36.Executor)
    (label : String) (query key value decayLog beta stateHistory outputGradient queryGradient
      keyGradient valueGradient decayLogGradient betaGradient stateGradient :
      @& Cuda.Buffer Float32) (layout : Cuda.Qwen36.SequenceLayout)
    (keyHeads valueHeads keyWidth valueWidth : UInt32) (queryScale : Float32) : IO Unit :=
  let launch : Cuda.LaunchConfig := {
    grid := { x := layout.batchSize * valueHeads }
    block := { x := 32 }
    sharedMemoryBytes := (valueWidth * 12).toUSize
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => recurrentNormalizedBatchedBackwardF32Kernel.launchOn stream launch query key
      value decayLog beta stateHistory outputGradient queryGradient keyGradient valueGradient
      decayLogGradient betaGradient stateGradient layout.batchSize layout.sequenceLength keyHeads
      valueHeads keyWidth valueWidth queryScale)
    (fun builder => recurrentNormalizedBatchedBackwardF32Kernel.addToGraph builder launch query key
      value decayLog beta stateHistory outputGradient queryGradient keyGradient valueGradient
      decayLogGradient betaGradient stateGradient layout.batchSize layout.sequenceLength keyHeads
      valueHeads keyWidth valueWidth queryScale)

/-! ## Complete DeltaNet-stage schedule -/

structure StageWeightsF32 where
  inputNorm : Cuda.Buffer Float32
  queryKeyValue : Projection.WeightsF32
  z : Projection.WeightsF32
  b : Projection.WeightsF32
  a : Projection.WeightsF32
  convolution : Cuda.Buffer Float32
  aLog : Cuda.Buffer Float32
  dtBias : Cuda.Buffer Float32
  gatedNorm : Cuda.Buffer Float32
  output : Projection.WeightsF32

structure StageForwardF32 where
  input : Cuda.Buffer Float32
  normalizedInput : Cuda.Buffer Float32
  inputInverseRms : Cuda.Buffer Float32
  queryKeyValuePreConv : Cuda.Buffer Float32
  z : Cuda.Buffer Float32
  b : Cuda.Buffer Float32
  a : Cuda.Buffer Float32
  queryKeyValuePostConv : Cuda.Buffer Float32
  queryCompact : Cuda.Buffer Float32
  keyCompact : Cuda.Buffer Float32
  value : Cuda.Buffer Float32
  queryRepeated : Cuda.Buffer Float32
  keyRepeated : Cuda.Buffer Float32
  queryNormed : Cuda.Buffer Float32
  keyNormed : Cuda.Buffer Float32
  decayLog : Cuda.Buffer Float32
  beta : Cuda.Buffer Float32
  stateInput : Cuda.Buffer Float32
  stateOutput : Cuda.Buffer Float32
  stateHistory : Cuda.Buffer Float32
  deltaOutput : Cuda.Buffer Float32
  gatedInverseRms : Cuda.Buffer Float32
  gatedOutput : Cuda.Buffer Float32
  output : Cuda.Buffer Float32

structure StageBackwardF32 where
  outputGradient : Cuda.Buffer Float32
  gatedOutputGradient : Cuda.Buffer Float32
  deltaOutputGradient : Cuda.Buffer Float32
  zGradient : Cuda.Buffer Float32
  gatedNormWeightGradient : Cuda.Buffer Float32
  queryNormGradient : Cuda.Buffer Float32
  keyNormGradient : Cuda.Buffer Float32
  valueGradient : Cuda.Buffer Float32
  decayLogGradient : Cuda.Buffer Float32
  betaGradient : Cuda.Buffer Float32
  stateGradient : Cuda.Buffer Float32
  queryRepeatedGradient : Cuda.Buffer Float32
  keyRepeatedGradient : Cuda.Buffer Float32
  queryCompactGradient : Cuda.Buffer Float32
  keyCompactGradient : Cuda.Buffer Float32
  queryKeyValuePostConvGradient : Cuda.Buffer Float32
  queryKeyValuePreConvGradient : Cuda.Buffer Float32
  convolutionWeightGradient : Cuda.Buffer Float32
  aGradient : Cuda.Buffer Float32
  bGradient : Cuda.Buffer Float32
  aLogGradient : Cuda.Buffer Float32
  dtBiasGradient : Cuda.Buffer Float32
  queryKeyValueInputGradient : Cuda.Buffer Float32
  zInputGradient : Cuda.Buffer Float32
  bInputGradient : Cuda.Buffer Float32
  aInputGradient : Cuda.Buffer Float32
  queryKeyValueZInputGradient : Cuda.Buffer Float32
  queryKeyValueZBInputGradient : Cuda.Buffer Float32
  normalizedInputGradient : Cuda.Buffer Float32
  inputGradient : Cuda.Buffer Float32
  inputNormWeightGradient : Cuda.Buffer Float32
  queryKeyValueWeightGradient : Cuda.Buffer Float32
  zWeightGradient : Cuda.Buffer Float32
  bWeightGradient : Cuda.Buffer Float32
  aWeightGradient : Cuda.Buffer Float32
  outputWeightGradient : Cuda.Buffer Float32

/-- Complete batch-isolated numerical Float32 DeltaNet forward schedule. -/
def submitBatchedForwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32)
    (buffers : @& StageForwardF32) (layout : Cuda.Qwen36.SequenceLayout)
    (hidden keyHeads valueHeads keyWidth valueWidth : UInt32)
    (epsilon queryScale : Float32) : IO Unit := do
  let tokens := layout.tokens
  let convolutionWidth := 2 * keyHeads * keyWidth + valueHeads * valueWidth
  submitRmsNormForwardF32 executor "Qwen3.6 DeltaNet input RMSNorm" buffers.input
    weights.inputNorm buffers.normalizedInput buffers.inputInverseRms tokens hidden epsilon
  Projection.submitForwardF32 executor "Qwen3.6 DeltaNet QKV projection" buffers.normalizedInput
    weights.queryKeyValue buffers.queryKeyValuePreConv tokens hidden convolutionWidth
  Projection.submitForwardF32 executor "Qwen3.6 DeltaNet z projection" buffers.normalizedInput
    weights.z buffers.z tokens hidden (valueHeads * valueWidth)
  Projection.submitForwardF32 executor "Qwen3.6 DeltaNet b projection" buffers.normalizedInput
    weights.b buffers.b tokens hidden valueHeads
  Projection.submitForwardF32 executor "Qwen3.6 DeltaNet a projection" buffers.normalizedInput
    weights.a buffers.a tokens hidden valueHeads
  submitConv1dSiLUBatchedForwardF32 executor "Qwen3.6 DeltaNet causal convolution"
    buffers.queryKeyValuePreConv weights.convolution buffers.queryKeyValuePostConv layout
    convolutionWidth
  submitSplitQkvF32 executor "Qwen3.6 DeltaNet QKV split" buffers.queryKeyValuePostConv
    buffers.queryCompact buffers.keyCompact buffers.value tokens keyHeads valueHeads keyWidth
    valueWidth
  submitRepeatCompactF32 executor "Qwen3.6 DeltaNet query repeat" buffers.queryCompact
    buffers.queryRepeated tokens keyHeads valueHeads keyWidth valueWidth
  submitRepeatCompactF32 executor "Qwen3.6 DeltaNet key repeat" buffers.keyCompact
    buffers.keyRepeated tokens keyHeads valueHeads keyWidth valueWidth
  submitL2normForwardF32 executor "Qwen3.6 DeltaNet query L2 norm" buffers.queryRepeated
    buffers.queryNormed (tokens * valueHeads) keyWidth epsilon
  submitL2normForwardF32 executor "Qwen3.6 DeltaNet key L2 norm" buffers.keyRepeated
    buffers.keyNormed (tokens * valueHeads) keyWidth epsilon
  submitGateForwardF32 executor "Qwen3.6 DeltaNet beta/decay" buffers.a buffers.b weights.aLog
    weights.dtBias buffers.decayLog buffers.beta tokens valueHeads
  submitRecurrentNormalizedBatchedPrefillF32 executor "Qwen3.6 DeltaNet normalized recurrence"
    buffers.queryNormed buffers.keyNormed buffers.value buffers.decayLog buffers.beta
    buffers.stateInput buffers.deltaOutput buffers.stateOutput buffers.stateHistory layout keyHeads
    valueHeads keyWidth valueWidth queryScale
  submitWeightedGatedRmsNormForwardF32 executor "Qwen3.6 DeltaNet weighted gated RMSNorm"
    buffers.deltaOutput buffers.z weights.gatedNorm buffers.gatedOutput buffers.gatedInverseRms
    (tokens * valueHeads) valueWidth epsilon
  Projection.submitForwardF32 executor "Qwen3.6 DeltaNet output projection" buffers.gatedOutput
    weights.output buffers.output tokens (valueHeads * valueWidth) hidden

/-- Single-sequence convenience schedule containing all rows. -/
def submitForwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32) (buffers : @& StageForwardF32)
    (tokens hidden keyHeads valueHeads keyWidth valueWidth : UInt32)
    (epsilon queryScale : Float32) : IO Unit :=
  submitBatchedForwardStageF32 executor weights buffers (.single tokens) hidden keyHeads valueHeads
    keyWidth valueWidth epsilon queryScale

/-- Sequential oracle route for the common DeltaNet forward schedule. -/
def forwardStageF32 (stream : @& Cuda.Stream) (weights : @& StageWeightsF32)
    (buffers : @& StageForwardF32) (tokens hidden keyHeads valueHeads keyWidth valueWidth :
      UInt32) (epsilon queryScale : Float32) : IO Unit :=
  submitForwardStageF32 (.sequential stream) weights buffers tokens hidden keyHeads valueHeads
    keyWidth valueWidth epsilon queryScale

/-- Complete batch-isolated numerical Float32 DeltaNet reverse schedule. -/
def submitBatchedBackwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32)
    (forward : @& StageForwardF32) (backward : @& StageBackwardF32)
    (layout : Cuda.Qwen36.SequenceLayout) (hidden keyHeads valueHeads keyWidth valueWidth : UInt32)
    (epsilon queryScale : Float32) : IO Unit := do
  let tokens := layout.tokens
  let convolutionWidth := 2 * keyHeads * keyWidth + valueHeads * valueWidth
  Projection.submitBackwardF32 executor "Qwen3.6 DeltaNet output projection"
    forward.gatedOutput weights.output backward.outputGradient backward.gatedOutputGradient
    backward.outputWeightGradient tokens (valueHeads * valueWidth) hidden
  submitWeightedGatedRmsNormBackwardF32 executor
    "Qwen3.6 DeltaNet gated RMSNorm input/gate VJP" forward.deltaOutput forward.z
    weights.gatedNorm backward.gatedOutputGradient backward.deltaOutputGradient backward.zGradient
    forward.gatedInverseRms (tokens * valueHeads) valueWidth
  submitWeightedGatedRmsNormBackwardWeightF32 executor
    "Qwen3.6 DeltaNet gated RMSNorm weight VJP" forward.deltaOutput forward.z
    backward.gatedOutputGradient forward.gatedInverseRms backward.gatedNormWeightGradient
    (tokens * valueHeads) valueWidth
  submitRecurrentNormalizedBatchedBackwardF32 executor "Qwen3.6 DeltaNet reverse recurrence"
    forward.queryNormed forward.keyNormed forward.value forward.decayLog forward.beta
    forward.stateHistory backward.deltaOutputGradient backward.queryNormGradient
    backward.keyNormGradient backward.valueGradient backward.decayLogGradient
    backward.betaGradient backward.stateGradient layout keyHeads valueHeads keyWidth valueWidth
    queryScale
  submitL2normBackwardF32 executor "Qwen3.6 DeltaNet query L2 VJP" forward.queryRepeated
    backward.queryNormGradient backward.queryRepeatedGradient (tokens * valueHeads) keyWidth epsilon
  submitL2normBackwardF32 executor "Qwen3.6 DeltaNet key L2 VJP" forward.keyRepeated
    backward.keyNormGradient backward.keyRepeatedGradient (tokens * valueHeads) keyWidth epsilon
  submitReduceRepeatedGradientF32 executor "Qwen3.6 DeltaNet query repeat VJP"
    backward.queryRepeatedGradient backward.queryCompactGradient tokens keyHeads valueHeads keyWidth
    valueWidth
  submitReduceRepeatedGradientF32 executor "Qwen3.6 DeltaNet key repeat VJP"
    backward.keyRepeatedGradient backward.keyCompactGradient tokens keyHeads valueHeads keyWidth
    valueWidth
  submitConcatenateQkvGradientF32 executor "Qwen3.6 DeltaNet QKV concatenate VJP"
    backward.queryCompactGradient backward.keyCompactGradient backward.valueGradient
    backward.queryKeyValuePostConvGradient tokens keyHeads valueHeads keyWidth valueWidth
  submitConv1dSiLUBatchedBackwardInputF32 executor "Qwen3.6 DeltaNet convolution input VJP"
    forward.queryKeyValuePreConv weights.convolution backward.queryKeyValuePostConvGradient
    backward.queryKeyValuePreConvGradient layout convolutionWidth
  submitConv1dSiLUBatchedBackwardWeightF32 executor "Qwen3.6 DeltaNet convolution weight VJP"
    forward.queryKeyValuePreConv weights.convolution backward.queryKeyValuePostConvGradient
    backward.convolutionWeightGradient layout convolutionWidth
  submitGateBackwardF32 executor "Qwen3.6 DeltaNet beta/decay projection VJP" forward.a forward.b
    weights.aLog weights.dtBias backward.decayLogGradient backward.betaGradient backward.aGradient
    backward.bGradient tokens valueHeads
  submitGateParameterGradientF32 executor "Qwen3.6 DeltaNet decay parameter VJP" forward.a
    backward.decayLogGradient weights.aLog weights.dtBias backward.aLogGradient
    backward.dtBiasGradient tokens valueHeads
  Projection.submitBackwardF32 executor "Qwen3.6 DeltaNet QKV projection"
    forward.normalizedInput weights.queryKeyValue backward.queryKeyValuePreConvGradient
    backward.queryKeyValueInputGradient backward.queryKeyValueWeightGradient tokens hidden
    convolutionWidth
  Projection.submitBackwardF32 executor "Qwen3.6 DeltaNet z projection" forward.normalizedInput
    weights.z backward.zGradient backward.zInputGradient backward.zWeightGradient tokens hidden
    (valueHeads * valueWidth)
  Projection.submitBackwardF32 executor "Qwen3.6 DeltaNet b projection" forward.normalizedInput
    weights.b backward.bGradient backward.bInputGradient backward.bWeightGradient tokens hidden
    valueHeads
  Projection.submitBackwardF32 executor "Qwen3.6 DeltaNet a projection" forward.normalizedInput
    weights.a backward.aGradient backward.aInputGradient backward.aWeightGradient tokens hidden
    valueHeads
  Linear.submitAddF32 executor "Qwen3.6 DeltaNet QKV/z input-gradient sum"
    backward.queryKeyValueInputGradient backward.zInputGradient
    backward.queryKeyValueZInputGradient (tokens * hidden)
  Linear.submitAddF32 executor "Qwen3.6 DeltaNet QKV/z/b input-gradient sum"
    backward.queryKeyValueZInputGradient backward.bInputGradient
    backward.queryKeyValueZBInputGradient (tokens * hidden)
  Linear.submitAddF32 executor "Qwen3.6 DeltaNet projection input-gradient sum"
    backward.queryKeyValueZBInputGradient backward.aInputGradient backward.normalizedInputGradient
    (tokens * hidden)
  submitRmsNormBackwardInputF32 executor "Qwen3.6 DeltaNet input RMSNorm VJP" forward.input
    weights.inputNorm backward.normalizedInputGradient backward.inputGradient
    forward.inputInverseRms tokens hidden
  submitRmsNormBackwardWeightF32 executor "Qwen3.6 DeltaNet input RMSNorm weight VJP"
    forward.input backward.normalizedInputGradient forward.inputInverseRms
    backward.inputNormWeightGradient tokens hidden

/-- Single-sequence convenience reverse schedule containing all rows. -/
def submitBackwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32) (forward : @& StageForwardF32)
    (backward : @& StageBackwardF32)
    (tokens hidden keyHeads valueHeads keyWidth valueWidth : UInt32)
    (epsilon queryScale : Float32) : IO Unit :=
  submitBatchedBackwardStageF32 executor weights forward backward (.single tokens) hidden keyHeads
    valueHeads keyWidth valueWidth epsilon queryScale

/-- Sequential oracle route for the common DeltaNet reverse schedule. -/
def backwardStageF32 (stream : @& Cuda.Stream) (weights : @& StageWeightsF32)
    (forward : @& StageForwardF32) (backward : @& StageBackwardF32)
    (tokens hidden keyHeads valueHeads keyWidth valueWidth : UInt32) (epsilon queryScale : Float32) :
    IO Unit :=
  submitBackwardStageF32 (.sequential stream) weights forward backward tokens hidden keyHeads
    valueHeads keyWidth valueWidth epsilon queryScale

end Cuda.Qwen36.DeltaNet
