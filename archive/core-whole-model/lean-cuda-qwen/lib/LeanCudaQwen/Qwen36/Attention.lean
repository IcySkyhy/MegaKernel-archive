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
# Qwen3.6 gated grouped-query attention

The FP32 causal kernel here is the numerical baseline shared by prefill tests and the optimized
BF16/WMMA implementation. Inputs are already q/k-normalized and RoPE-rotated; the primitive layer
owns those transformations. One warp computes one query row, publishes head-major causal
probabilities, and applies the Qwen3.6 sigmoid output gate.
-/

namespace Cuda.Qwen36.Attention

open Cuda.Qwen36.Primitives

@[struct] structure Buffers where
  query : Cuda.DevicePtr Float32
  key : Cuda.DevicePtr Float32
  value : Cuda.DevicePtr Float32
  gate : Cuda.DevicePtr Float32
  outputPreGate : Cuda.DevicePtr Float32
  outputPostGate : Cuda.DevicePtr Float32
  probabilities : Cuda.DevicePtr Float32

@[struct] structure Shape where
  tokens : UInt32
  queryHeads : UInt32
  keyValueHeads : UInt32
  width : UInt32

@[expose, cuda_device, always_inline]
def negativeInfinity : Float32 := Float32.ofBits 0xff800000

@[expose, cuda_device, always_inline, convergent]
def warpMaxBroadcast (value : Float32) : Cuda.DeviceM Float32 := do
  let maximum ← Cuda.Collective.warpMax value
  return Float32.ofBits (← Cuda.shuffleUInt32 maximum.toBits 0)

namespace Internal

@[always_inline]
def queryIndex (token head channel : UInt32) (shape : Shape) : UInt32 :=
  (token * shape.queryHeads + head) * shape.width + channel

@[always_inline]
def keyValueIndex (token head channel : UInt32) (shape : Shape) : UInt32 :=
  (token * shape.keyValueHeads + head) * shape.width + channel

@[always_inline]
def probabilityIndex (head queryToken keyToken : UInt32) (shape : Shape) : UInt32 :=
  (head * shape.tokens + queryToken) * shape.tokens + keyToken

@[always_inline]
partial def scoreDot (buffers : Buffers) (shape : Shape)
    (queryToken queryHead keyToken keyValueHead channel : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if channel < shape.width then
    let queryValue ← Cuda.loadFloat32 buffers.query
      (queryIndex queryToken queryHead channel shape).toUSize
    let keyValue ← Cuda.loadFloat32 buffers.key
      (keyValueIndex keyToken keyValueHead channel shape).toUSize
    scoreDot buffers shape queryToken queryHead keyToken keyValueHead (channel + 1)
      (Cuda.fma queryValue keyValue accumulator)
  else
    return accumulator

@[always_inline]
partial def publishScores (buffers : Buffers) (scores : Cuda.DevicePtr Float32)
    (shape : Shape) (queryToken queryHead keyValueHead keyToken laneStride : UInt32)
    (scale localMaximum : Float32) : Cuda.DeviceM Float32 := do
  if keyToken <= queryToken then
    let dot ← scoreDot buffers shape queryToken queryHead keyToken keyValueHead 0 zeroF32
    let score := dot * scale
    Cuda.storeFloat32 scores keyToken.toUSize score
    publishScores buffers scores shape queryToken queryHead keyValueHead
      (keyToken + laneStride) laneStride scale (max localMaximum score)
  else
    return localMaximum

@[always_inline]
partial def publishExponentials (scores : Cuda.DevicePtr Float32)
    (queryToken keyToken laneStride : UInt32) (maximum localSum : Float32) :
    Cuda.DeviceM Float32 := do
  if keyToken <= queryToken then
    let exponential := Float32.exp ((← Cuda.loadFloat32 scores keyToken.toUSize) - maximum)
    Cuda.storeFloat32 scores keyToken.toUSize exponential
    publishExponentials scores queryToken (keyToken + laneStride) laneStride maximum
      (localSum + exponential)
  else
    return localSum

@[always_inline]
partial def publishProbabilities (buffers : Buffers) (scores : Cuda.DevicePtr Float32)
    (shape : Shape) (queryToken queryHead keyToken laneStride : UInt32)
    (inverseDenominator : Float32) : Cuda.DeviceM Unit := do
  if keyToken < shape.tokens then
    let probability ← if keyToken <= queryToken then
        pure ((← Cuda.loadFloat32 scores keyToken.toUSize) * inverseDenominator)
      else
        pure zeroF32
    Cuda.storeFloat32 scores keyToken.toUSize probability
    Cuda.storeFloat32 buffers.probabilities
      (probabilityIndex queryHead queryToken keyToken shape).toUSize probability
    publishProbabilities buffers scores shape queryToken queryHead (keyToken + laneStride)
      laneStride inverseDenominator

@[always_inline]
partial def valueDot (buffers : Buffers) (scores : Cuda.DevicePtr Float32)
    (shape : Shape) (queryToken keyValueHead channel keyToken : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if keyToken <= queryToken then
    let probability ← Cuda.loadFloat32 scores keyToken.toUSize
    let projectedValue ← Cuda.loadFloat32 buffers.value
      (keyValueIndex keyToken keyValueHead channel shape).toUSize
    valueDot buffers scores shape queryToken keyValueHead channel (keyToken + 1)
      (Cuda.fma probability projectedValue accumulator)
  else
    return accumulator

@[always_inline]
partial def publishOutput (buffers : Buffers) (scores : Cuda.DevicePtr Float32)
    (shape : Shape) (queryToken queryHead keyValueHead channel laneStride : UInt32) :
    Cuda.DeviceM Unit := do
  if channel < shape.width then
    let output ← valueDot buffers scores shape queryToken keyValueHead channel 0 zeroF32
    let index := queryIndex queryToken queryHead channel shape
    let gateValue ← Cuda.loadFloat32 buffers.gate index.toUSize
    Cuda.storeFloat32 buffers.outputPreGate index.toUSize output
    Cuda.storeFloat32 buffers.outputPostGate index.toUSize (output * sigmoidF32 gateValue)
    publishOutput buffers scores shape queryToken queryHead keyValueHead
      (channel + laneStride) laneStride

end Internal

namespace Internal

@[always_inline]
def queryGateIndex (token head channel : UInt32) (shape : Shape) : UInt32 :=
  (token * shape.queryHeads + head) * (2 * shape.width) + channel

@[always_inline]
partial def probabilityGradientDot (outputGradient value : Cuda.DevicePtr Float32)
    (shape : Shape) (queryToken queryHead keyToken keyValueHead channel : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if channel < shape.width then
    let gradient ← Cuda.loadFloat32 outputGradient
      (queryIndex queryToken queryHead channel shape).toUSize
    let valueElement ← Cuda.loadFloat32 value
      (keyValueIndex keyToken keyValueHead channel shape).toUSize
    probabilityGradientDot outputGradient value shape queryToken queryHead keyToken keyValueHead
      (channel + 1) (Cuda.fma gradient valueElement accumulator)
  else
    return accumulator

@[always_inline]
partial def probabilityScoreDot (probabilities probabilityGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (queryHead queryToken keyToken : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if keyToken < shape.tokens then
    let index := probabilityIndex queryHead queryToken keyToken shape
    let probability ← Cuda.loadFloat32 probabilities index.toUSize
    let gradient ← Cuda.loadFloat32 probabilityGradient index.toUSize
    probabilityScoreDot probabilities probabilityGradient shape queryHead queryToken
      (keyToken + 1) (Cuda.fma probability gradient accumulator)
  else
    return accumulator

@[always_inline]
partial def queryGradientDot (scoreGradient key : Cuda.DevicePtr Float32) (shape : Shape)
    (queryToken queryHead keyValueHead channel keyToken : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if keyToken < shape.tokens then
    let gradient ← Cuda.loadFloat32 scoreGradient
      (probabilityIndex queryHead queryToken keyToken shape).toUSize
    let keyValue ← Cuda.loadFloat32 key
      (keyValueIndex keyToken keyValueHead channel shape).toUSize
    queryGradientDot scoreGradient key shape queryToken queryHead keyValueHead channel
      (keyToken + 1) (Cuda.fma gradient keyValue accumulator)
  else
    return accumulator

@[always_inline]
partial def keyGradientTokens (scoreGradient query : Cuda.DevicePtr Float32) (shape : Shape)
    (keyToken keyValueHead channel queryHead queryToken : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if queryToken < shape.tokens then
    let gradient ← Cuda.loadFloat32 scoreGradient
      (probabilityIndex queryHead queryToken keyToken shape).toUSize
    let queryValue ← Cuda.loadFloat32 query
      (queryIndex queryToken queryHead channel shape).toUSize
    keyGradientTokens scoreGradient query shape keyToken keyValueHead channel queryHead
      (queryToken + 1) (Cuda.fma gradient queryValue accumulator)
  else
    return accumulator

@[always_inline]
partial def keyGradientHeads (scoreGradient query : Cuda.DevicePtr Float32) (shape : Shape)
    (keyToken keyValueHead channel queryHead endHead : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if queryHead < endHead then
    let accumulator ← keyGradientTokens scoreGradient query shape keyToken keyValueHead channel
      queryHead 0 accumulator
    keyGradientHeads scoreGradient query shape keyToken keyValueHead channel (queryHead + 1)
      endHead accumulator
  else
    return accumulator

@[always_inline]
partial def valueGradientTokens (probabilities outputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (keyToken keyValueHead channel queryHead queryToken : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if queryToken < shape.tokens then
    let probability ← Cuda.loadFloat32 probabilities
      (probabilityIndex queryHead queryToken keyToken shape).toUSize
    let gradient ← Cuda.loadFloat32 outputGradient
      (queryIndex queryToken queryHead channel shape).toUSize
    valueGradientTokens probabilities outputGradient shape keyToken keyValueHead channel queryHead
      (queryToken + 1) (Cuda.fma probability gradient accumulator)
  else
    return accumulator

@[always_inline]
partial def valueGradientHeads (probabilities outputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (keyToken keyValueHead channel queryHead endHead : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if queryHead < endHead then
    let accumulator ← valueGradientTokens probabilities outputGradient shape keyToken keyValueHead
      channel queryHead 0 accumulator
    valueGradientHeads probabilities outputGradient shape keyToken keyValueHead channel
      (queryHead + 1) endHead accumulator
  else
    return accumulator

end Internal

/-- Split the doubled Qwen query projection into compact query and gate tensors. -/
@[expose, cuda_device, always_inline]
def splitQueryGateElementF32 (queryGate query gate : Cuda.DevicePtr Float32) (shape : Shape)
    (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.tokens * shape.queryHeads * shape.width then
    let token := linear / (shape.queryHeads * shape.width)
    let rest := linear % (shape.queryHeads * shape.width)
    let head := rest / shape.width
    let channel := rest % shape.width
    let base := Internal.queryGateIndex token head channel shape
    let queryValue ← Cuda.loadFloat32 queryGate base.toUSize
    let gateValue ← Cuda.loadFloat32 queryGate (base + shape.width).toUSize
    Cuda.storeFloat32 query linear.toUSize queryValue
    Cuda.storeFloat32 gate linear.toUSize gateValue

/-- Adjoint of `splitQueryGateElementF32`. -/
@[expose, cuda_device, always_inline]
def mergeQueryGateGradientElementF32 (queryGradient gateGradient queryGateGradient :
    Cuda.DevicePtr Float32) (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.tokens * shape.queryHeads * shape.width then
    let token := linear / (shape.queryHeads * shape.width)
    let rest := linear % (shape.queryHeads * shape.width)
    let head := rest / shape.width
    let channel := rest % shape.width
    let base := Internal.queryGateIndex token head channel shape
    let queryValue ← Cuda.loadFloat32 queryGradient linear.toUSize
    let gateValue ← Cuda.loadFloat32 gateGradient linear.toUSize
    Cuda.storeFloat32 queryGateGradient base.toUSize queryValue
    Cuda.storeFloat32 queryGateGradient (base + shape.width).toUSize gateValue

/-- Sigmoid-gate VJP from the projected attention output. -/
@[expose, cuda_device, always_inline]
def gateBackwardElementF32 (preGate gate outputGradient preGateGradient gateGradient :
    Cuda.DevicePtr Float32) (count linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < count then
    let pre ← Cuda.loadFloat32 preGate linear.toUSize
    let gateValue ← Cuda.loadFloat32 gate linear.toUSize
    let gradient ← Cuda.loadFloat32 outputGradient linear.toUSize
    let sigmoid := sigmoidF32 gateValue
    Cuda.storeFloat32 preGateGradient linear.toUSize (gradient * sigmoid)
    Cuda.storeFloat32 gateGradient linear.toUSize
      (gradient * pre * sigmoid * (oneF32 - sigmoid))

/-- Publish one `dL/d(probability)` element. -/
@[expose, cuda_device, always_inline]
def probabilityGradientElementF32 (outputGradient value probabilityGradient :
    Cuda.DevicePtr Float32) (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  let count := shape.queryHeads * shape.tokens * shape.tokens
  if linear < count then
    let queryHead := linear / (shape.tokens * shape.tokens)
    let rest := linear % (shape.tokens * shape.tokens)
    let queryToken := rest / shape.tokens
    let keyToken := rest % shape.tokens
    let keyValueHead := queryHead / (shape.queryHeads / shape.keyValueHeads)
    let gradient ← Internal.probabilityGradientDot outputGradient value shape queryToken queryHead
      keyToken keyValueHead 0 zeroF32
    Cuda.storeFloat32 probabilityGradient linear.toUSize gradient

/-- Softmax VJP, including the attention scale. -/
@[expose, cuda_device, always_inline]
def scoreGradientElementF32 (probabilities probabilityGradient scoreGradient :
    Cuda.DevicePtr Float32) (shape : Shape) (scale : Float32) (linear : UInt32) :
    Cuda.DeviceM Unit := do
  let count := shape.queryHeads * shape.tokens * shape.tokens
  if linear < count then
    let queryHead := linear / (shape.tokens * shape.tokens)
    let rest := linear % (shape.tokens * shape.tokens)
    let queryToken := rest / shape.tokens
    let dot ← Internal.probabilityScoreDot probabilities probabilityGradient shape queryHead
      queryToken 0 zeroF32
    let probability ← Cuda.loadFloat32 probabilities linear.toUSize
    let gradient ← Cuda.loadFloat32 probabilityGradient linear.toUSize
    Cuda.storeFloat32 scoreGradient linear.toUSize (probability * (gradient - dot) * scale)

/-- Query VJP from scaled score gradients. -/
@[expose, cuda_device, always_inline]
def queryGradientElementF32 (scoreGradient key queryGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.tokens * shape.queryHeads * shape.width then
    let queryToken := linear / (shape.queryHeads * shape.width)
    let rest := linear % (shape.queryHeads * shape.width)
    let queryHead := rest / shape.width
    let channel := rest % shape.width
    let keyValueHead := queryHead / (shape.queryHeads / shape.keyValueHeads)
    let gradient ← Internal.queryGradientDot scoreGradient key shape queryToken queryHead
      keyValueHead channel 0 zeroF32
    Cuda.storeFloat32 queryGradient linear.toUSize gradient

/-- Key and value VJPs, reducing all query heads in one GQA group. -/
@[expose, cuda_device, always_inline]
def keyValueGradientElementF32 (scoreGradient probabilities query outputGradient keyGradient
    valueGradient : Cuda.DevicePtr Float32) (shape : Shape) (linear : UInt32) :
    Cuda.DeviceM Unit := do
  if linear < shape.tokens * shape.keyValueHeads * shape.width then
    let keyToken := linear / (shape.keyValueHeads * shape.width)
    let rest := linear % (shape.keyValueHeads * shape.width)
    let keyValueHead := rest / shape.width
    let channel := rest % shape.width
    let group := shape.queryHeads / shape.keyValueHeads
    let firstHead := keyValueHead * group
    let endHead := firstHead + group
    let dk ← Internal.keyGradientHeads scoreGradient query shape keyToken keyValueHead channel
      firstHead endHead zeroF32
    let dv ← Internal.valueGradientHeads probabilities outputGradient shape keyToken keyValueHead
      channel firstHead endHead zeroF32
    Cuda.storeFloat32 keyGradient linear.toUSize dk
    Cuda.storeFloat32 valueGradient linear.toUSize dv

/-- Compute one causal GQA row and its sigmoid-gated output with one convergent warp. -/
@[expose, cuda_device, always_inline, convergent]
def attentionRowF32 (buffers : Buffers) (scores : Cuda.DevicePtr Float32) (shape : Shape)
    (queryToken queryHead : UInt32) (scale : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let keyValueHead := queryHead / (shape.queryHeads / shape.keyValueHeads)
  let localMaximum ← Internal.publishScores buffers scores shape queryToken queryHead keyValueHead
    lane 32 scale negativeInfinity
  let maximum ← warpMaxBroadcast localMaximum
  Cuda.blockSync
  let localSum ← Internal.publishExponentials scores queryToken lane 32 maximum zeroF32
  let denominator ← warpSumBroadcast localSum
  let inverseDenominator := oneF32 / denominator
  Internal.publishProbabilities buffers scores shape queryToken queryHead lane 32 inverseDenominator
  Cuda.blockSync
  Internal.publishOutput buffers scores shape queryToken queryHead keyValueHead lane 32

/-- Numerical FP32 causal-attention baseline, one warp per `(token, queryHead)` row. -/
@[cuda_kernel]
def causalForwardF32Kernel (query key value gate outputPreGate outputPostGate probabilities :
    Cuda.DevicePtr Float32) (tokens queryHeads keyValueHeads width : UInt32) (scale : Float32) :
    Cuda.DeviceM Unit := do
  let row ← Cuda.blockIdxX
  let shape : Shape := { tokens, queryHeads, keyValueHeads, width }
  if row < tokens * queryHeads then
    let scores ← Cuda.dynamicShared (α := Float32)
    let buffers : Buffers := {
      query, key, value, gate, outputPreGate, outputPostGate, probabilities
    }
    attentionRowF32 buffers scores shape (row / queryHeads) (row % queryHeads) scale

@[cuda_kernel]
def splitQueryGateF32Kernel (queryGate query gate : Cuda.DevicePtr Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : Cuda.DeviceM Unit := do
  splitQueryGateElementF32 queryGate query gate { tokens, queryHeads, keyValueHeads, width }
    (← elementIndex)

@[cuda_kernel]
def mergeQueryGateGradientF32Kernel (queryGradient gateGradient queryGateGradient :
    Cuda.DevicePtr Float32) (tokens queryHeads keyValueHeads width : UInt32) :
    Cuda.DeviceM Unit := do
  mergeQueryGateGradientElementF32 queryGradient gateGradient queryGateGradient
    { tokens, queryHeads, keyValueHeads, width } (← elementIndex)

@[cuda_kernel]
def gateBackwardF32Kernel (preGate gate outputGradient preGateGradient gateGradient :
    Cuda.DevicePtr Float32) (count : UInt32) : Cuda.DeviceM Unit := do
  gateBackwardElementF32 preGate gate outputGradient preGateGradient gateGradient count
    (← elementIndex)

@[cuda_kernel]
def probabilityGradientF32Kernel (outputGradient value probabilityGradient :
    Cuda.DevicePtr Float32) (tokens queryHeads keyValueHeads width : UInt32) :
    Cuda.DeviceM Unit := do
  probabilityGradientElementF32 outputGradient value probabilityGradient
    { tokens, queryHeads, keyValueHeads, width } (← elementIndex)

@[cuda_kernel]
def scoreGradientF32Kernel (probabilities probabilityGradient scoreGradient :
    Cuda.DevicePtr Float32) (tokens queryHeads keyValueHeads width : UInt32) (scale : Float32) :
    Cuda.DeviceM Unit := do
  scoreGradientElementF32 probabilities probabilityGradient scoreGradient
    { tokens, queryHeads, keyValueHeads, width } scale (← elementIndex)

@[cuda_kernel]
def queryGradientF32Kernel (scoreGradient key queryGradient : Cuda.DevicePtr Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : Cuda.DeviceM Unit := do
  queryGradientElementF32 scoreGradient key queryGradient
    { tokens, queryHeads, keyValueHeads, width } (← elementIndex)

@[cuda_kernel]
def keyValueGradientF32Kernel (scoreGradient probabilities query outputGradient keyGradient
    valueGradient : Cuda.DevicePtr Float32) (tokens queryHeads keyValueHeads width : UInt32) :
    Cuda.DeviceM Unit := do
  keyValueGradientElementF32 scoreGradient probabilities query outputGradient keyGradient
    valueGradient { tokens, queryHeads, keyValueHeads, width } (← elementIndex)

/-- Batch-isolated causal GQA. Each block sees one sequence-local attention row and probability
matrix, so neither the causal mask nor the softmax can cross a batch boundary. -/
@[cuda_kernel]
def causalBatchedForwardF32Kernel (query key value gate outputPreGate outputPostGate probabilities :
    Cuda.DevicePtr Float32) (batchSize sequenceLength queryHeads keyValueHeads width : UInt32)
    (scale : Float32) : Cuda.DeviceM Unit := do
  let linearRow ← Cuda.blockIdxX
  let rowsPerBatch := sequenceLength * queryHeads
  if linearRow < batchSize * rowsPerBatch then
    let batch := linearRow / rowsPerBatch
    let row := linearRow % rowsPerBatch
    let queryElements := sequenceLength * queryHeads * width
    let keyValueElements := sequenceLength * keyValueHeads * width
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    let queryBatch : Cuda.DevicePtr Float32 := query + (batch * queryElements * 4).toUSize
    let keyBatch : Cuda.DevicePtr Float32 := key + (batch * keyValueElements * 4).toUSize
    let valueBatch : Cuda.DevicePtr Float32 := value + (batch * keyValueElements * 4).toUSize
    let gateBatch : Cuda.DevicePtr Float32 := gate + (batch * queryElements * 4).toUSize
    let preGateBatch : Cuda.DevicePtr Float32 :=
      outputPreGate + (batch * queryElements * 4).toUSize
    let postGateBatch : Cuda.DevicePtr Float32 :=
      outputPostGate + (batch * queryElements * 4).toUSize
    let probabilitiesBatch : Cuda.DevicePtr Float32 :=
      probabilities + (batch * probabilityElements * 4).toUSize
    let scores ← Cuda.dynamicShared (α := Float32)
    let buffers : Buffers := {
      query := queryBatch, key := keyBatch, value := valueBatch, gate := gateBatch,
      outputPreGate := preGateBatch, outputPostGate := postGateBatch,
      probabilities := probabilitiesBatch
    }
    let shape : Shape := { tokens := sequenceLength, queryHeads, keyValueHeads, width }
    attentionRowF32 buffers scores shape (row / queryHeads) (row % queryHeads) scale

/-- Batch-isolated dense probability VJP. -/
@[cuda_kernel]
def probabilityBatchedGradientF32Kernel (outputGradient value probabilityGradient :
    Cuda.DevicePtr Float32) (batchSize sequenceLength queryHeads keyValueHeads width : UInt32) :
    Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let probabilityElements := queryHeads * sequenceLength * sequenceLength
  if linear < batchSize * probabilityElements then
    let batch := linear / probabilityElements
    let localIndex := linear % probabilityElements
    let queryElements := sequenceLength * queryHeads * width
    let keyValueElements := sequenceLength * keyValueHeads * width
    let outputGradientBatch : Cuda.DevicePtr Float32 :=
      outputGradient + (batch * queryElements * 4).toUSize
    let valueBatch : Cuda.DevicePtr Float32 := value + (batch * keyValueElements * 4).toUSize
    let probabilityGradientBatch : Cuda.DevicePtr Float32 :=
      probabilityGradient + (batch * probabilityElements * 4).toUSize
    probabilityGradientElementF32 outputGradientBatch valueBatch probabilityGradientBatch
      { tokens := sequenceLength, queryHeads, keyValueHeads, width } localIndex

/-- Batch-isolated softmax-score VJP. -/
@[cuda_kernel]
def scoreBatchedGradientF32Kernel (probabilities probabilityGradient scoreGradient :
    Cuda.DevicePtr Float32) (batchSize sequenceLength queryHeads keyValueHeads width : UInt32)
    (scale : Float32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let probabilityElements := queryHeads * sequenceLength * sequenceLength
  if linear < batchSize * probabilityElements then
    let batch := linear / probabilityElements
    let localIndex := linear % probabilityElements
    let probabilitiesBatch : Cuda.DevicePtr Float32 :=
      probabilities + (batch * probabilityElements * 4).toUSize
    let probabilityGradientBatch : Cuda.DevicePtr Float32 :=
      probabilityGradient + (batch * probabilityElements * 4).toUSize
    let scoreGradientBatch : Cuda.DevicePtr Float32 :=
      scoreGradient + (batch * probabilityElements * 4).toUSize
    scoreGradientElementF32 probabilitiesBatch probabilityGradientBatch scoreGradientBatch
      { tokens := sequenceLength, queryHeads, keyValueHeads, width } scale localIndex

/-- Batch-isolated query VJP. -/
@[cuda_kernel]
def queryBatchedGradientF32Kernel (scoreGradient key queryGradient : Cuda.DevicePtr Float32)
    (batchSize sequenceLength queryHeads keyValueHeads width : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let queryElements := sequenceLength * queryHeads * width
  if linear < batchSize * queryElements then
    let batch := linear / queryElements
    let localIndex := linear % queryElements
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    let keyValueElements := sequenceLength * keyValueHeads * width
    let scoreGradientBatch : Cuda.DevicePtr Float32 :=
      scoreGradient + (batch * probabilityElements * 4).toUSize
    let keyBatch : Cuda.DevicePtr Float32 := key + (batch * keyValueElements * 4).toUSize
    let queryGradientBatch : Cuda.DevicePtr Float32 :=
      queryGradient + (batch * queryElements * 4).toUSize
    queryGradientElementF32 scoreGradientBatch keyBatch queryGradientBatch
      { tokens := sequenceLength, queryHeads, keyValueHeads, width } localIndex

/-- Batch-isolated grouped key/value VJPs. -/
@[cuda_kernel]
def keyValueBatchedGradientF32Kernel (scoreGradient probabilities query outputGradient keyGradient
    valueGradient : Cuda.DevicePtr Float32)
    (batchSize sequenceLength queryHeads keyValueHeads width : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let keyValueElements := sequenceLength * keyValueHeads * width
  if linear < batchSize * keyValueElements then
    let batch := linear / keyValueElements
    let localIndex := linear % keyValueElements
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    let queryElements := sequenceLength * queryHeads * width
    let scoreGradientBatch : Cuda.DevicePtr Float32 :=
      scoreGradient + (batch * probabilityElements * 4).toUSize
    let probabilitiesBatch : Cuda.DevicePtr Float32 :=
      probabilities + (batch * probabilityElements * 4).toUSize
    let queryBatch : Cuda.DevicePtr Float32 := query + (batch * queryElements * 4).toUSize
    let outputGradientBatch : Cuda.DevicePtr Float32 :=
      outputGradient + (batch * queryElements * 4).toUSize
    let keyGradientBatch : Cuda.DevicePtr Float32 :=
      keyGradient + (batch * keyValueElements * 4).toUSize
    let valueGradientBatch : Cuda.DevicePtr Float32 :=
      valueGradient + (batch * keyValueElements * 4).toUSize
    keyValueGradientElementF32 scoreGradientBatch probabilitiesBatch queryBatch
      outputGradientBatch keyGradientBatch valueGradientBatch
      { tokens := sequenceLength, queryHeads, keyValueHeads, width } localIndex

/-- Launch configuration for the FP32 attention baseline. -/
def causalConfig (tokens queryHeads : UInt32) : Cuda.LaunchConfig := {
  grid := { x := tokens * queryHeads }
  block := { x := 32 }
  sharedMemoryBytes := (tokens * 4).toUSize
  blockArenaBytes := 0
}

/-- Launch the numerical FP32 causal grouped-query attention baseline. -/
def causalForwardF32 (stream : @& Cuda.Stream)
    (query key value gate outputPreGate outputPostGate probabilities : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) (scale : Float32) : IO Cuda.KernelHandle :=
  causalForwardF32Kernel.launchOn stream (causalConfig tokens queryHeads)
    query key value gate outputPreGate outputPostGate probabilities tokens queryHeads keyValueHeads
    width scale

/-- Split `[tokens, queryHeads, 2 * width]` query/gate projections. -/
def splitQueryGateF32 (stream : @& Cuda.Stream) (queryGate query gate :
    @& Cuda.Buffer Float32) (tokens queryHeads keyValueHeads width : UInt32) :
    IO Cuda.KernelHandle :=
  splitQueryGateF32Kernel.launchOn stream (elementConfig (tokens * queryHeads * width)) queryGate
    query gate tokens queryHeads keyValueHeads width

/-- Merge query and gate VJPs into the doubled query-projection layout. -/
def mergeQueryGateGradientF32 (stream : @& Cuda.Stream)
    (queryGradient gateGradient queryGateGradient : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Cuda.KernelHandle :=
  mergeQueryGateGradientF32Kernel.launchOn stream
    (elementConfig (tokens * queryHeads * width)) queryGradient gateGradient queryGateGradient
    tokens queryHeads keyValueHeads width

/-- VJP of the sigmoid attention-output gate. -/
def gateBackwardF32 (stream : @& Cuda.Stream)
    (preGate gate outputGradient preGateGradient gateGradient : @& Cuda.Buffer Float32)
    (count : UInt32) : IO Cuda.KernelHandle :=
  gateBackwardF32Kernel.launchOn stream (elementConfig count) preGate gate outputGradient
    preGateGradient gateGradient count

/-- Publish the dense probability gradient. -/
def probabilityGradientF32 (stream : @& Cuda.Stream)
    (outputGradient value probabilityGradient : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Cuda.KernelHandle :=
  probabilityGradientF32Kernel.launchOn stream
    (elementConfig (queryHeads * tokens * tokens)) outputGradient value probabilityGradient
    tokens queryHeads keyValueHeads width

/-- Softmax-score VJP including the attention scale. -/
def scoreGradientF32 (stream : @& Cuda.Stream)
    (probabilities probabilityGradient scoreGradient : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) (scale : Float32) : IO Cuda.KernelHandle :=
  scoreGradientF32Kernel.launchOn stream (elementConfig (queryHeads * tokens * tokens))
    probabilities probabilityGradient scoreGradient tokens queryHeads keyValueHeads width scale

/-- Query VJP from scaled score gradients. -/
def queryGradientF32 (stream : @& Cuda.Stream) (scoreGradient key queryGradient :
    @& Cuda.Buffer Float32) (tokens queryHeads keyValueHeads width : UInt32) :
    IO Cuda.KernelHandle :=
  queryGradientF32Kernel.launchOn stream (elementConfig (tokens * queryHeads * width))
    scoreGradient key queryGradient tokens queryHeads keyValueHeads width

/-- GQA key/value VJPs. -/
def keyValueGradientF32 (stream : @& Cuda.Stream)
    (scoreGradient probabilities query outputGradient keyGradient valueGradient :
      @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Cuda.KernelHandle :=
  keyValueGradientF32Kernel.launchOn stream (elementConfig (tokens * keyValueHeads * width))
    scoreGradient probabilities query outputGradient keyGradient valueGradient tokens queryHeads
    keyValueHeads width

/-! ## Shared sequential/device-graph submissions -/

def submitCausalForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (query key value gate outputPreGate outputPostGate probabilities : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) (scale : Float32) : IO Unit :=
  let launch := causalConfig tokens queryHeads
  executor.submit label
    (fun stream => causalForwardF32Kernel.launchOn stream launch query key value gate outputPreGate
      outputPostGate probabilities tokens queryHeads keyValueHeads width scale)
    (fun builder => causalForwardF32Kernel.addToGraph builder launch query key value gate
      outputPreGate outputPostGate probabilities tokens queryHeads keyValueHeads width scale)

def submitSplitQueryGateF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (queryGate query gate : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * queryHeads * width)
  executor.submit label
    (fun stream => splitQueryGateF32Kernel.launchOn stream launch queryGate query gate tokens
      queryHeads keyValueHeads width)
    (fun builder => splitQueryGateF32Kernel.addToGraph builder launch queryGate query gate tokens
      queryHeads keyValueHeads width)

def submitMergeQueryGateGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (queryGradient gateGradient queryGateGradient : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * queryHeads * width)
  executor.submit label
    (fun stream => mergeQueryGateGradientF32Kernel.launchOn stream launch queryGradient
      gateGradient queryGateGradient tokens queryHeads keyValueHeads width)
    (fun builder => mergeQueryGateGradientF32Kernel.addToGraph builder launch queryGradient
      gateGradient queryGateGradient tokens queryHeads keyValueHeads width)

def submitGateBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (preGate gate outputGradient preGateGradient gateGradient : @& Cuda.Buffer Float32)
    (count : UInt32) : IO Unit :=
  let launch := elementConfig count
  executor.submit label
    (fun stream => gateBackwardF32Kernel.launchOn stream launch preGate gate outputGradient
      preGateGradient gateGradient count)
    (fun builder => gateBackwardF32Kernel.addToGraph builder launch preGate gate outputGradient
      preGateGradient gateGradient count)

def submitProbabilityGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (outputGradient value probabilityGradient : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let launch := elementConfig (queryHeads * tokens * tokens)
  executor.submit label
    (fun stream => probabilityGradientF32Kernel.launchOn stream launch outputGradient value
      probabilityGradient tokens queryHeads keyValueHeads width)
    (fun builder => probabilityGradientF32Kernel.addToGraph builder launch outputGradient value
      probabilityGradient tokens queryHeads keyValueHeads width)

def submitScoreGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (probabilities probabilityGradient scoreGradient : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) (scale : Float32) : IO Unit :=
  let launch := elementConfig (queryHeads * tokens * tokens)
  executor.submit label
    (fun stream => scoreGradientF32Kernel.launchOn stream launch probabilities probabilityGradient
      scoreGradient tokens queryHeads keyValueHeads width scale)
    (fun builder => scoreGradientF32Kernel.addToGraph builder launch probabilities
      probabilityGradient scoreGradient tokens queryHeads keyValueHeads width scale)

def submitQueryGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (scoreGradient key queryGradient : @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * queryHeads * width)
  executor.submit label
    (fun stream => queryGradientF32Kernel.launchOn stream launch scoreGradient key queryGradient
      tokens queryHeads keyValueHeads width)
    (fun builder => queryGradientF32Kernel.addToGraph builder launch scoreGradient key queryGradient
      tokens queryHeads keyValueHeads width)

def submitKeyValueGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (scoreGradient probabilities query outputGradient keyGradient valueGradient :
      @& Cuda.Buffer Float32)
    (tokens queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * keyValueHeads * width)
  executor.submit label
    (fun stream => keyValueGradientF32Kernel.launchOn stream launch scoreGradient probabilities
      query outputGradient keyGradient valueGradient tokens queryHeads keyValueHeads width)
    (fun builder => keyValueGradientF32Kernel.addToGraph builder launch scoreGradient probabilities
      query outputGradient keyGradient valueGradient tokens queryHeads keyValueHeads width)

def submitCausalBatchedForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (query key value gate outputPreGate outputPostGate probabilities : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (queryHeads keyValueHeads width : UInt32)
    (scale : Float32) : IO Unit :=
  let launch : Cuda.LaunchConfig := {
    grid := { x := layout.tokens * queryHeads }
    block := { x := 32 }
    sharedMemoryBytes := (layout.sequenceLength * 4).toUSize
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => causalBatchedForwardF32Kernel.launchOn stream launch query key value gate
      outputPreGate outputPostGate probabilities layout.batchSize layout.sequenceLength queryHeads
      keyValueHeads width scale)
    (fun builder => causalBatchedForwardF32Kernel.addToGraph builder launch query key value gate
      outputPreGate outputPostGate probabilities layout.batchSize layout.sequenceLength queryHeads
      keyValueHeads width scale)

def submitProbabilityBatchedGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (outputGradient value probabilityGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let count := layout.batchSize * queryHeads * layout.sequenceLength * layout.sequenceLength
  let launch := elementConfig count
  executor.submit label
    (fun stream => probabilityBatchedGradientF32Kernel.launchOn stream launch outputGradient value
      probabilityGradient layout.batchSize layout.sequenceLength queryHeads keyValueHeads width)
    (fun builder => probabilityBatchedGradientF32Kernel.addToGraph builder launch outputGradient
      value probabilityGradient layout.batchSize layout.sequenceLength queryHeads keyValueHeads
      width)

def submitScoreBatchedGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (probabilities probabilityGradient scoreGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (queryHeads keyValueHeads width : UInt32)
    (scale : Float32) : IO Unit :=
  let count := layout.batchSize * queryHeads * layout.sequenceLength * layout.sequenceLength
  let launch := elementConfig count
  executor.submit label
    (fun stream => scoreBatchedGradientF32Kernel.launchOn stream launch probabilities
      probabilityGradient scoreGradient layout.batchSize layout.sequenceLength queryHeads
      keyValueHeads width scale)
    (fun builder => scoreBatchedGradientF32Kernel.addToGraph builder launch probabilities
      probabilityGradient scoreGradient layout.batchSize layout.sequenceLength queryHeads
      keyValueHeads width scale)

def submitQueryBatchedGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (scoreGradient key queryGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let launch := elementConfig (layout.tokens * queryHeads * width)
  executor.submit label
    (fun stream => queryBatchedGradientF32Kernel.launchOn stream launch scoreGradient key
      queryGradient layout.batchSize layout.sequenceLength queryHeads keyValueHeads width)
    (fun builder => queryBatchedGradientF32Kernel.addToGraph builder launch scoreGradient key
      queryGradient layout.batchSize layout.sequenceLength queryHeads keyValueHeads width)

def submitKeyValueBatchedGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (scoreGradient probabilities query outputGradient keyGradient valueGradient :
      @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (queryHeads keyValueHeads width : UInt32) : IO Unit :=
  let launch := elementConfig (layout.tokens * keyValueHeads * width)
  executor.submit label
    (fun stream => keyValueBatchedGradientF32Kernel.launchOn stream launch scoreGradient
      probabilities query outputGradient keyGradient valueGradient layout.batchSize
      layout.sequenceLength queryHeads keyValueHeads width)
    (fun builder => keyValueBatchedGradientF32Kernel.addToGraph builder launch scoreGradient
      probabilities query outputGradient keyGradient valueGradient layout.batchSize
      layout.sequenceLength queryHeads keyValueHeads width)

/-! ## Complete attention-stage schedule -/

/-- Immutable Float32 weights for one full-attention stage. -/
structure StageWeightsF32 where
  inputNorm : Cuda.Buffer Float32
  query : Projection.WeightsF32
  key : Projection.WeightsF32
  value : Projection.WeightsF32
  queryNorm : Cuda.Buffer Float32
  keyNorm : Cuda.Buffer Float32
  output : Projection.WeightsF32
  inverseFrequency : Cuda.Buffer Float32

/-- Saved forward values for one full-attention stage. -/
structure StageForwardF32 where
  input : Cuda.Buffer Float32
  normalizedInput : Cuda.Buffer Float32
  inputInverseRms : Cuda.Buffer Float32
  queryGateProjection : Cuda.Buffer Float32
  queryPreNorm : Cuda.Buffer Float32
  gate : Cuda.Buffer Float32
  keyPreNorm : Cuda.Buffer Float32
  value : Cuda.Buffer Float32
  queryNormed : Cuda.Buffer Float32
  queryInverseRms : Cuda.Buffer Float32
  keyNormed : Cuda.Buffer Float32
  keyInverseRms : Cuda.Buffer Float32
  queryRope : Cuda.Buffer Float32
  keyRope : Cuda.Buffer Float32
  probabilities : Cuda.Buffer Float32
  preGate : Cuda.Buffer Float32
  postGate : Cuda.Buffer Float32
  output : Cuda.Buffer Float32

/-- Saved gradients and scratch for the complete attention VJP. -/
structure StageBackwardF32 where
  outputGradient : Cuda.Buffer Float32
  postGateGradient : Cuda.Buffer Float32
  preGateGradient : Cuda.Buffer Float32
  gateGradient : Cuda.Buffer Float32
  probabilityGradient : Cuda.Buffer Float32
  scoreGradient : Cuda.Buffer Float32
  queryRopeGradient : Cuda.Buffer Float32
  keyRopeGradient : Cuda.Buffer Float32
  valueGradient : Cuda.Buffer Float32
  queryNormGradient : Cuda.Buffer Float32
  keyNormGradient : Cuda.Buffer Float32
  queryPreNormGradient : Cuda.Buffer Float32
  keyPreNormGradient : Cuda.Buffer Float32
  queryNormWeightGradient : Cuda.Buffer Float32
  keyNormWeightGradient : Cuda.Buffer Float32
  queryGateProjectionGradient : Cuda.Buffer Float32
  queryInputGradient : Cuda.Buffer Float32
  keyInputGradient : Cuda.Buffer Float32
  valueInputGradient : Cuda.Buffer Float32
  queryKeyInputGradient : Cuda.Buffer Float32
  normalizedInputGradient : Cuda.Buffer Float32
  inputGradient : Cuda.Buffer Float32
  inputNormWeightGradient : Cuda.Buffer Float32
  queryWeightGradient : Cuda.Buffer Float32
  keyWeightGradient : Cuda.Buffer Float32
  valueWeightGradient : Cuda.Buffer Float32
  outputWeightGradient : Cuda.Buffer Float32

/-- Complete batch-isolated numerical Float32 full-attention schedule. -/
def submitBatchedForwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32) (buffers : @& StageForwardF32)
    (layout : Cuda.Qwen36.SequenceLayout)
    (hidden queryHeads keyValueHeads width rotaryHalf : UInt32)
    (epsilon scale : Float32) : IO Unit := do
  let tokens := layout.tokens
  let queryElements := tokens * queryHeads * width
  let keyValueElements := tokens * keyValueHeads * width
  submitRmsNormForwardF32 executor "Qwen3.6 attention input RMSNorm" buffers.input
    weights.inputNorm buffers.normalizedInput buffers.inputInverseRms tokens hidden epsilon
  Projection.submitForwardF32 executor "Qwen3.6 attention query/gate projection"
    buffers.normalizedInput weights.query buffers.queryGateProjection tokens hidden
    (2 * queryHeads * width)
  submitSplitQueryGateF32 executor "Qwen3.6 attention query/gate split"
    buffers.queryGateProjection buffers.queryPreNorm buffers.gate tokens queryHeads keyValueHeads
    width
  Projection.submitForwardF32 executor "Qwen3.6 attention key projection" buffers.normalizedInput
    weights.key buffers.keyPreNorm tokens hidden (keyValueHeads * width)
  Projection.submitForwardF32 executor "Qwen3.6 attention value projection" buffers.normalizedInput
    weights.value buffers.value tokens hidden (keyValueHeads * width)
  submitRmsNormForwardF32 executor "Qwen3.6 attention query RMSNorm" buffers.queryPreNorm
    weights.queryNorm buffers.queryNormed buffers.queryInverseRms (tokens * queryHeads) width
    epsilon
  submitRmsNormForwardF32 executor "Qwen3.6 attention key RMSNorm" buffers.keyPreNorm
    weights.keyNorm buffers.keyNormed buffers.keyInverseRms (tokens * keyValueHeads) width epsilon
  submitRopeBatchedForwardF32 executor "Qwen3.6 attention query RoPE" buffers.queryNormed
    weights.inverseFrequency buffers.queryRope layout queryHeads width rotaryHalf
  submitRopeBatchedForwardF32 executor "Qwen3.6 attention key RoPE" buffers.keyNormed
    weights.inverseFrequency buffers.keyRope layout keyValueHeads width rotaryHalf
  submitCausalBatchedForwardF32 executor "Qwen3.6 causal grouped-query attention"
    buffers.queryRope buffers.keyRope buffers.value buffers.gate buffers.preGate buffers.postGate
    buffers.probabilities layout queryHeads keyValueHeads width scale
  Projection.submitForwardF32 executor "Qwen3.6 attention output projection" buffers.postGate
    weights.output buffers.output tokens (queryHeads * width) hidden
  unless queryElements == tokens * queryHeads * width &&
      keyValueElements == tokens * keyValueHeads * width do
    throw <| IO.userError "unreachable Qwen3.6 attention element-count mismatch"

/-- Single-sequence convenience schedule containing all rows. -/
def submitForwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32)
    (buffers : @& StageForwardF32) (tokens hidden queryHeads keyValueHeads width rotaryHalf :
      UInt32) (epsilon scale : Float32) : IO Unit :=
  submitBatchedForwardStageF32 executor weights buffers (.single tokens) hidden queryHeads
    keyValueHeads width rotaryHalf epsilon scale

/-- Sequential oracle route for the common full-attention schedule. -/
def forwardStageF32 (stream : @& Cuda.Stream) (weights : @& StageWeightsF32)
    (buffers : @& StageForwardF32) (tokens hidden queryHeads keyValueHeads width rotaryHalf :
      UInt32) (epsilon scale : Float32) : IO Unit :=
  submitForwardStageF32 (.sequential stream) weights buffers tokens hidden queryHeads keyValueHeads
    width rotaryHalf epsilon scale

/-- Complete batch-isolated numerical Float32 full-attention reverse schedule. -/
def submitBatchedBackwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32)
    (forward : @& StageForwardF32) (backward : @& StageBackwardF32)
    (layout : Cuda.Qwen36.SequenceLayout)
    (hidden queryHeads keyValueHeads width rotaryHalf : UInt32) (scale : Float32) : IO Unit := do
  let tokens := layout.tokens
  let queryElements := tokens * queryHeads * width
  Projection.submitBackwardF32 executor "Qwen3.6 attention output projection" forward.postGate
    weights.output backward.outputGradient backward.postGateGradient backward.outputWeightGradient
    tokens (queryHeads * width) hidden
  submitGateBackwardF32 executor "Qwen3.6 attention sigmoid-gate VJP" forward.preGate
    forward.gate backward.postGateGradient backward.preGateGradient backward.gateGradient
    queryElements
  submitProbabilityBatchedGradientF32 executor "Qwen3.6 attention probability VJP"
    backward.preGateGradient forward.value backward.probabilityGradient layout queryHeads
    keyValueHeads width
  submitScoreBatchedGradientF32 executor "Qwen3.6 attention score VJP" forward.probabilities
    backward.probabilityGradient backward.scoreGradient layout queryHeads keyValueHeads width scale
  submitQueryBatchedGradientF32 executor "Qwen3.6 attention query VJP" backward.scoreGradient
    forward.keyRope backward.queryRopeGradient layout queryHeads keyValueHeads width
  submitKeyValueBatchedGradientF32 executor "Qwen3.6 attention key/value VJP"
    backward.scoreGradient forward.probabilities forward.queryRope backward.preGateGradient
    backward.keyRopeGradient backward.valueGradient layout queryHeads keyValueHeads width
  submitRopeBatchedBackwardF32 executor "Qwen3.6 attention query RoPE VJP"
    backward.queryRopeGradient weights.inverseFrequency backward.queryNormGradient layout queryHeads
    width rotaryHalf
  submitRopeBatchedBackwardF32 executor "Qwen3.6 attention key RoPE VJP"
    backward.keyRopeGradient weights.inverseFrequency backward.keyNormGradient layout keyValueHeads
    width rotaryHalf
  submitRmsNormBackwardInputF32 executor "Qwen3.6 attention query RMSNorm input VJP"
    forward.queryPreNorm weights.queryNorm backward.queryNormGradient
    backward.queryPreNormGradient forward.queryInverseRms (tokens * queryHeads) width
  submitRmsNormBackwardWeightF32 executor "Qwen3.6 attention query RMSNorm weight VJP"
    forward.queryPreNorm backward.queryNormGradient forward.queryInverseRms
    backward.queryNormWeightGradient (tokens * queryHeads) width
  submitRmsNormBackwardInputF32 executor "Qwen3.6 attention key RMSNorm input VJP"
    forward.keyPreNorm weights.keyNorm backward.keyNormGradient backward.keyPreNormGradient
    forward.keyInverseRms (tokens * keyValueHeads) width
  submitRmsNormBackwardWeightF32 executor "Qwen3.6 attention key RMSNorm weight VJP"
    forward.keyPreNorm backward.keyNormGradient forward.keyInverseRms
    backward.keyNormWeightGradient (tokens * keyValueHeads) width
  submitMergeQueryGateGradientF32 executor "Qwen3.6 attention query/gate merge VJP"
    backward.queryPreNormGradient backward.gateGradient backward.queryGateProjectionGradient tokens
    queryHeads keyValueHeads width
  Projection.submitBackwardF32 executor "Qwen3.6 attention query projection"
    forward.normalizedInput weights.query backward.queryGateProjectionGradient
    backward.queryInputGradient backward.queryWeightGradient tokens hidden
    (2 * queryHeads * width)
  Projection.submitBackwardF32 executor "Qwen3.6 attention key projection" forward.normalizedInput
    weights.key backward.keyPreNormGradient backward.keyInputGradient backward.keyWeightGradient
    tokens hidden (keyValueHeads * width)
  Projection.submitBackwardF32 executor "Qwen3.6 attention value projection"
    forward.normalizedInput weights.value backward.valueGradient backward.valueInputGradient
    backward.valueWeightGradient tokens hidden (keyValueHeads * width)
  Linear.submitAddF32 executor "Qwen3.6 attention query/key input-gradient sum"
    backward.queryInputGradient backward.keyInputGradient backward.queryKeyInputGradient
    (tokens * hidden)
  Linear.submitAddF32 executor "Qwen3.6 attention projection input-gradient sum"
    backward.queryKeyInputGradient backward.valueInputGradient backward.normalizedInputGradient
    (tokens * hidden)
  submitRmsNormBackwardInputF32 executor "Qwen3.6 attention input RMSNorm VJP" forward.input
    weights.inputNorm backward.normalizedInputGradient backward.inputGradient
    forward.inputInverseRms tokens hidden
  submitRmsNormBackwardWeightF32 executor "Qwen3.6 attention input RMSNorm weight VJP"
    forward.input backward.normalizedInputGradient forward.inputInverseRms
    backward.inputNormWeightGradient tokens hidden

/-- Single-sequence convenience reverse schedule containing all rows. -/
def submitBackwardStageF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& StageWeightsF32)
    (forward : @& StageForwardF32) (backward : @& StageBackwardF32)
    (tokens hidden queryHeads keyValueHeads width rotaryHalf : UInt32) (scale : Float32) : IO Unit :=
  submitBatchedBackwardStageF32 executor weights forward backward (.single tokens) hidden
    queryHeads keyValueHeads width rotaryHalf scale

/-- Sequential oracle route for the common full-attention reverse schedule. -/
def backwardStageF32 (stream : @& Cuda.Stream) (weights : @& StageWeightsF32)
    (forward : @& StageForwardF32) (backward : @& StageBackwardF32)
    (tokens hidden queryHeads keyValueHeads width rotaryHalf : UInt32) (scale : Float32) : IO Unit :=
  submitBackwardStageF32 (.sequential stream) weights forward backward tokens hidden queryHeads
    keyValueHeads width rotaryHalf scale

end Cuda.Qwen36.Attention
