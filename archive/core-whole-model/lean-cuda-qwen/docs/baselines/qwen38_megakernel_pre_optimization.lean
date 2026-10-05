/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Foundation
public import LeanCudaQwen.Qwen36.Config
public import LeanCudaQwen.Qwen36.Primitives
public import LeanCudaQwen.Qwen36.SafeTensors

public section

/-!
# Qwen3.8-27B persistent multi-token inference megakernel

This is the model-wide, plain-Lean CUDA decode path. Published text tensors are streamed once into
independently aligned CUDA buffers, and queued address-free descriptors resolve through an owning
pointer table.
One cooperative-grid persistent launch executes embedding lookup, all 64 decoder layers, final
RMSNorm, the full LM head, stable cross-entropy, and greedy sampling for each queued token. DeltaNet
convolution/recurrent state and full-attention KV caches remain device-resident across positions;
full attention applies the checkpoint's 64-channel partial RoPE and causal GQA softmax over all
cached positions. Returned logits, losses, and samples are position-major.
-/

namespace Cuda.Qwen36.Megakernel

open Cuda.Qwen36.SafeTensors

/-- One greedy token published directly by the persistent grid to a mapped host queue. -/
@[struct] structure TokenEvent where
  position : UInt32
  token : UInt32
  deriving Repr, Inhabited, BEq, Cuda.POD

namespace TokenEvent

private def appendUInt32LE (value : UInt32) (bytes : ByteArray) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def readUInt32LE (bytes : ByteArray) (offset : Nat) : UInt32 :=
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

/-- Stable little-endian host representation used by the mapped token queue. -/
def toBytes (event : TokenEvent) : ByteArray :=
  ByteArray.emptyWithCapacity 8
    |> appendUInt32LE event.position
    |> appendUInt32LE event.token

/-- Decode one mapped token-queue payload. -/
def ofBytes (bytes : ByteArray) : Except String TokenEvent := do
  unless bytes.size == 8 do
    throw "a Qwen3.8 token event must occupy 8 bytes"
  return { position := readUInt32LE bytes 0, token := readUInt32LE bytes 4 }

end TokenEvent

instance : Cuda.HostIO.Codec TokenEvent where
  encode := TokenEvent.toBytes
  decode := TokenEvent.ofBytes

private def hiddenSize : Nat := 5120
private def intermediateSize : Nat := 17408
private def vocabularySize : Nat := 248320
private def layerCount : Nat := 64
private def linearLayerCount : Nat := 48
private def fullAttentionLayerCount : Nat := 16
private def linearQkvSize : Nat := 10240
private def linearValueSize : Nat := 6144
private def linearHeadCount : Nat := 48
private def linearHeadDimension : Nat := 128
private def fullQueryProjectionSize : Nat := 12288
private def fullKeyValueProjectionSize : Nat := 1024
private def fullOutputSize : Nat := 6144
private def fullHeadDimension : Nat := 256
private def rotaryDimensionNat : Nat := 64
private def rotaryHalfNat : Nat := rotaryDimensionNat / 2

/-- The bounded causal context owned by one persistent launch. -/
def maxSequenceTokens : Nat := 256

private inductive LayerKind where
  | linearAttention
  | fullAttention
  deriving Repr, BEq, DecidableEq

private def layerKind (layer : Nat) : LayerKind :=
  if (layer + 1) % 4 == 0 then .fullAttention else .linearAttention

private structure TensorRef where
  name : String
  shard : UInt32
  byteOffset : UInt64
  byteCount : UInt64
  deriving Repr, BEq

private structure CommonLayerWeights where
  inputNorm : TensorRef
  postAttentionNorm : TensorRef
  mlpGate : TensorRef
  mlpUp : TensorRef
  mlpDown : TensorRef
  deriving Repr, BEq

private structure LinearAttentionWeights where
  aLog : TensorRef
  convolution : TensorRef
  dtBias : TensorRef
  projectionA : TensorRef
  projectionB : TensorRef
  projectionQkv : TensorRef
  projectionZ : TensorRef
  norm : TensorRef
  outputProjection : TensorRef
  deriving Repr, BEq

private structure FullAttentionWeights where
  queryNorm : TensorRef
  keyNorm : TensorRef
  queryProjection : TensorRef
  keyProjection : TensorRef
  valueProjection : TensorRef
  outputProjection : TensorRef
  deriving Repr, BEq

private inductive AttentionWeights where
  | linear (weights : LinearAttentionWeights)
  | full (weights : FullAttentionWeights)
  deriving Repr, BEq

private structure LayerWeights where
  index : Nat
  common : CommonLayerWeights
  attention : AttentionWeights
  deriving Repr, BEq

private structure TextLayout where
  embedding : TensorRef
  layers : Array LayerWeights
  finalNorm : TensorRef
  lmHead : TensorRef
  deriving Repr, BEq

private def gridBlockCountNat : Nat := 288
private def blockThreadCountNat : Nat := 256

/-- Compile-time dense-projection load grouping. Every step consists of one or more naturally
aligned four-BF16 transactions; keeping the alternatives here makes performance candidates
reproducible without rewriting the projection loops. -/
private inductive ProjectionLoadWidth where
  | bf16x4
  | bf16x8
  | bf16x16
  deriving Repr, BEq

private def projectionLoadWidth : ProjectionLoadWidth := .bf16x16

private def projectionQuadsPerStepNat : Nat :=
  match projectionLoadWidth with
  | .bf16x4 => 1
  | .bf16x8 => 2
  | .bf16x16 => 4

private def projectionValuesPerLoadNat : Nat := 4 * projectionQuadsPerStepNat
private def weightTensorAlignmentNat : Nat := 128
private def lmHeadScaleBlockElementsNat : Nat := 32
private def lmHeadElementsNat : Nat := vocabularySize * hiddenSize
private def lmHeadScaleBlockCountNat : Nat := lmHeadElementsNat / lmHeadScaleBlockElementsNat
private def useGreedyLmHeadMXFP8 : Bool := true
private def useAlignedTextWeights : Bool := true

/-- Benchmark route derived from the compile-time launch and projection configuration. -/
def benchmarkRoute : String :=
  let lmHeadRoute := if useGreedyLmHeadMXFP8 then "mxfp8b32_dot4" else "bf16"
  let weightRoute :=
    if useAlignedTextWeights then s!"aligned{weightTensorAlignmentNat}" else "raw_shards"
  s!"persistent_megakernel_grid{gridBlockCountNat}_block{blockThreadCountNat}_" ++
    s!"bf16x{projectionValuesPerLoadNat}_" ++
    s!"greedy_lmhead_{lmHeadRoute}_paired_swiglu_rowpair2_fused_residual_" ++
    s!"resolved_weight_ptrs_{weightRoute}"
private def descriptorsPerTokenNat : Nat := layerCount + 2
private def linearStateElementsPerLayerNat : Nat :=
  linearHeadCount * linearHeadDimension * linearHeadDimension
private def convolutionElementsPerLayerNat : Nat := linearQkvSize * 4
private def fullCacheElementsPerLayerNat : Nat :=
  maxSequenceTokens * fullKeyValueProjectionSize

private def hiddenSizeU32 : UInt32 := hiddenSize.toUInt32
private def intermediateSizeU32 : UInt32 := intermediateSize.toUInt32
private def vocabularySizeU32 : UInt32 := vocabularySize.toUInt32
private def gridBlockCount : UInt32 := gridBlockCountNat.toUInt32
private def blockThreadCount : UInt32 := blockThreadCountNat.toUInt32
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
private def rotaryDimension : UInt32 := rotaryDimensionNat.toUInt32
private def rotaryHalf : UInt32 := rotaryHalfNat.toUInt32
private def maxSequenceTokensU32 : UInt32 := maxSequenceTokens.toUInt32

private def zero : Float32 := Float32.ofBits 0
private def one : Float32 := Float32.ofBits 0x3f800000
private def negativeInfinity : Float32 := Float32.ofBits 0xff800000
private def inverseHiddenSize : Float32 := Float32.ofBits 0x394ccccd
private def inverseLinearDimension : Float32 := Float32.ofBits 0x3c000000
private def inverseFullDimension : Float32 := Float32.ofBits 0x3b800000
private def rmsEpsilon : Float32 := Float32.ofBits 0x358637bd
private def queryScale : Float32 := Float32.ofBits 0x3db504f3
private def attentionScale : Float32 := Float32.ofBits 0x3d800000
private def ropeTheta : Float32 := Float32.ofBits 0x4b189680
private def log2e : Float32 := Float32.ofBits 0x3fb8aa3b

private def reductionBytes : Nat := 8 * 4
private def vectorBytes : Nat := linearHeadDimension * 4
private def queryOffset : Nat := reductionBytes
private def keyOffset : Nat := queryOffset + vectorBytes
private def deltaOffset : Nat := keyOffset + vectorBytes
private def fullVectorOffset : Nat := reductionBytes
private def fullScoresOffset : Nat := fullVectorOffset + fullHeadDimension * 4
private def sharedBytes : Nat := fullScoresOffset + maxSequenceTokens * 4

private def kindEmbedding : UInt32 := 0
private def kindLinear : UInt32 := 1
private def kindFull : UInt32 := 2
private def kindFinal : UInt32 := 3
private def kindGreedyFinal : UInt32 := 4
private def weightFieldCount : UInt32 := 14
private def weightInputNorm : UInt32 := 0
private def weightPostNorm : UInt32 := 1
private def weightMlpGate : UInt32 := 2
private def weightMlpUp : UInt32 := 3
private def weightMlpDown : UInt32 := 4
private def weightAttention0 : UInt32 := 5
private def weightAttention1 : UInt32 := 6
private def weightAttention2 : UInt32 := 7
private def weightAttention3 : UInt32 := 8
private def weightAttention4 : UInt32 := 9
private def weightAttention5 : UInt32 := 10
private def weightAttention6 : UInt32 := 11
private def weightAttention7 : UInt32 := 12
private def weightAttention8 : UInt32 := 13

@[struct] structure WeightRef where
  shard : UInt32
  byteOffset : UInt64
  byteCount : UInt64
  deriving Repr, Inhabited, BEq

@[struct] structure ModelItem where
  kind : UInt32
  stateSlot : UInt32
  weightSlot : UInt32
  token : UInt32
  position : UInt32
  target : UInt32
  descriptor : UInt32
  deriving Repr, Inhabited, BEq, Cuda.POD

private structure ModelWeights where
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
  deriving Repr, Inhabited, BEq

@[struct] private structure Buffers where
  hidden : Cuda.DevicePtr Cuda.BFloat16
  normalized : Cuda.DevicePtr Cuda.BFloat16
  finalHidden : Cuda.DevicePtr Cuda.BFloat16
  workspace0 : Cuda.DevicePtr Cuda.BFloat16
  workspace1 : Cuda.DevicePtr Cuda.BFloat16
  workspace2 : Cuda.DevicePtr Cuda.BFloat16
  attentionOutput : Cuda.DevicePtr Cuda.BFloat16
  lmHeadData : Cuda.DevicePtr Cuda.Float8E4M3
  lmHeadScales : Cuda.DevicePtr Cuda.ScaleUE8M0
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
  tokenEvents : Cuda.DevicePtr TokenEvent

@[always_inline]
private def weightPointer (weights : Cuda.DevicePtrTable UInt8)
    (item : ModelItem) (field : UInt32) :
    Cuda.DeviceM (Cuda.DevicePtr Cuda.BFloat16) := do
  let record := item.weightSlot * weightFieldCount + field
  return ← weights.get record.toUSize

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
private def storeResidualBFloat (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) (update : Float32) : Cuda.DeviceM Unit := do
  storeBFloat pointer index ((← loadBFloat pointer index) + rounded update)

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
  if index < gridBlockCount then
    sumPartials partials (index + 1) (accumulator + (← Cuda.loadFloat32 partials index.toUSize))
  else
    return accumulator

@[always_inline]
private partial def maximumPartials (partials : Cuda.DevicePtr Float32)
    (index : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < gridBlockCount then
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
  let blockSquares ← Cuda.Collective.blockSumLeader scratch localSquares
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialSum block.toUSize blockSquares
  Cuda.gridSync
  let squareSum ← sumPartials buffers.partialSum 0 zero
  let inverse ← Cuda.fastRsqrt (squareSum * inverseHiddenSize + rmsEpsilon)
  normalizeLoop source weight output hiddenSizeU32 globalThread globalThreads inverse
  Cuda.gridSync

@[struct]
private structure ProjectionPair where
  first : Float32
  second : Float32
  deriving Nonempty

@[always_inline]
private def dotProjectionQuad
    (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (inputQuad weightQuad : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  let inputValues ← Cuda.loadBFloat16x4 input inputQuad.toUSize
  let weightValues ← Cuda.loadReadOnlyBFloat16x4 weight weightQuad.toUSize
  let accumulator := Cuda.fma weightValues.x0.toFloat32 inputValues.x0.toFloat32 accumulator
  let accumulator := Cuda.fma weightValues.x1.toFloat32 inputValues.x1.toFloat32 accumulator
  let accumulator := Cuda.fma weightValues.x2.toFloat32 inputValues.x2.toFloat32 accumulator
  return Cuda.fma weightValues.x3.toFloat32 inputValues.x3.toFloat32 accumulator

@[always_inline]
private def dotProjectionPairQuad
    (input firstWeight secondWeight : Cuda.DevicePtr Cuda.BFloat16)
    (inputQuad weightQuad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let inputValues ← Cuda.loadBFloat16x4 input inputQuad.toUSize
  let firstValues ← Cuda.loadReadOnlyBFloat16x4 firstWeight weightQuad.toUSize
  let secondValues ← Cuda.loadReadOnlyBFloat16x4 secondWeight weightQuad.toUSize
  let firstAccumulator :=
    Cuda.fma firstValues.x0.toFloat32 inputValues.x0.toFloat32 firstAccumulator
  let firstAccumulator :=
    Cuda.fma firstValues.x1.toFloat32 inputValues.x1.toFloat32 firstAccumulator
  let firstAccumulator :=
    Cuda.fma firstValues.x2.toFloat32 inputValues.x2.toFloat32 firstAccumulator
  let firstAccumulator :=
    Cuda.fma firstValues.x3.toFloat32 inputValues.x3.toFloat32 firstAccumulator
  let secondAccumulator :=
    Cuda.fma secondValues.x0.toFloat32 inputValues.x0.toFloat32 secondAccumulator
  let secondAccumulator :=
    Cuda.fma secondValues.x1.toFloat32 inputValues.x1.toFloat32 secondAccumulator
  let secondAccumulator :=
    Cuda.fma secondValues.x2.toFloat32 inputValues.x2.toFloat32 secondAccumulator
  let secondAccumulator :=
    Cuda.fma secondValues.x3.toFloat32 inputValues.x3.toFloat32 secondAccumulator
  return { first := firstAccumulator, second := secondAccumulator }

@[always_inline]
private def dotProjectionGroup (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnGroups group : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  let quadsPerStep := projectionQuadsPerStepNat.toUInt32
  let inputQuad := group * quadsPerStep
  let weightQuad := row * columnGroups * quadsPerStep + inputQuad
  let accumulator ← dotProjectionQuad input weight inputQuad weightQuad accumulator
  match projectionLoadWidth with
  | .bf16x4 => return accumulator
  | .bf16x8 =>
    dotProjectionQuad input weight (inputQuad + 1) (weightQuad + 1) accumulator
  | .bf16x16 => do
    let accumulator ←
      dotProjectionQuad input weight (inputQuad + 1) (weightQuad + 1) accumulator
    let accumulator ←
      dotProjectionQuad input weight (inputQuad + 2) (weightQuad + 2) accumulator
    dotProjectionQuad input weight (inputQuad + 3) (weightQuad + 3) accumulator

@[always_inline]
private def dotProjectionPairGroup
    (input firstWeight secondWeight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnGroups group : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let quadsPerStep := projectionQuadsPerStepNat.toUInt32
  let inputQuad := group * quadsPerStep
  let weightQuad := row * columnGroups * quadsPerStep + inputQuad
  let accumulators ← dotProjectionPairQuad input firstWeight secondWeight inputQuad weightQuad
    firstAccumulator secondAccumulator
  match projectionLoadWidth with
  | .bf16x4 => return accumulators
  | .bf16x8 =>
    dotProjectionPairQuad input firstWeight secondWeight (inputQuad + 1) (weightQuad + 1)
      accumulators.first accumulators.second
  | .bf16x16 => do
    let accumulators ←
      dotProjectionPairQuad input firstWeight secondWeight (inputQuad + 1) (weightQuad + 1)
        accumulators.first accumulators.second
    let accumulators ←
      dotProjectionPairQuad input firstWeight secondWeight (inputQuad + 2) (weightQuad + 2)
        accumulators.first accumulators.second
    dotProjectionPairQuad input firstWeight secondWeight (inputQuad + 3) (weightQuad + 3)
      accumulators.first accumulators.second

@[always_inline]
private partial def dotProjectionRow (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnGroups group : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if group < columnGroups then
    let accumulator ← dotProjectionGroup input weight row columnGroups group accumulator
    dotProjectionRow input weight row columnGroups (group + blockThreadCount) accumulator
  else
    return accumulator

@[always_inline]
private partial def dotProjectionRowPair
    (input firstWeight secondWeight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnGroups group : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  if group < columnGroups then
    let accumulators ← dotProjectionPairGroup input firstWeight secondWeight
      row columnGroups group firstAccumulator secondAccumulator
    dotProjectionRowPair input firstWeight secondWeight row columnGroups
      (group + blockThreadCount) accumulators.first accumulators.second
  else
    return { first := firstAccumulator, second := secondAccumulator }

@[always_inline, convergent]
private partial def projectRowsSwiGLU
    (input gateWeight upWeight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSums ← dotProjectionRowPair input gateWeight upWeight row
      (columns / projectionValuesPerLoadNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      let gateValue ← deviceSilu (rounded totals.first)
      storeBFloat output row (rounded gateValue * rounded totals.second)
    projectRowsSwiGLU input gateWeight upWeight output rows columns
      (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowsBFloat (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSum ← dotProjectionRow input weight row
      (columns / projectionValuesPerLoadNat.toUInt32) thread zero
    let total ← Cuda.Collective.blockSumLeader scratch localSum
    if thread == 0 then
      storeBFloat output row total
    projectRowsBFloat input weight output rows columns (row + rowStride) rowStride thread scratch

/-- Project two adjacent output rows per block iteration, reusing each packed input load and the
fused pair reduction. The persistent Qwen projection shapes routed here all have an even row
count; `row` and `rowStride` therefore advance in units of two rows. -/
@[always_inline, convergent]
private partial def projectRowPairsBFloat
    (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let secondWeight := bfloatOffset weight columns
    let localSums ← dotProjectionRowPair input weight secondWeight row
      (columns / projectionValuesPerLoadNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      storeBFloat output row totals.first
      storeBFloat output (row + 1) totals.second
    projectRowPairsBFloat input weight output rows columns
      (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowPairsResidualBFloat
    (input weight residual : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let secondWeight := bfloatOffset weight columns
    let localSums ← dotProjectionRowPair input weight secondWeight row
      (columns / projectionValuesPerLoadNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      storeResidualBFloat residual row totals.first
      storeResidualBFloat residual (row + 1) totals.second
    projectRowPairsResidualBFloat input weight residual rows columns
      (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowsFloat (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (output : Cuda.DevicePtr Float32) (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSum ← dotProjectionRow input weight row
      (columns / projectionValuesPerLoadNat.toUInt32) thread zero
    let total ← Cuda.Collective.blockSumLeader scratch localSum
    if thread == 0 then
      Cuda.storeFloat32 output row.toUSize total
    projectRowsFloat input weight output rows columns (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowPairsFloat (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (output : Cuda.DevicePtr Float32) (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let secondWeight := bfloatOffset weight columns
    let localSums ← dotProjectionRowPair input weight secondWeight row
      (columns / projectionValuesPerLoadNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      Cuda.storeFloat32 output row.toUSize totals.first
      Cuda.storeFloat32 output (row + 1).toUSize totals.second
    projectRowPairsFloat input weight output rows columns
      (row + rowStride) rowStride thread scratch

@[always_inline]
private partial def dotProjectionRowMXFP8 (input : Cuda.DevicePtr Cuda.BFloat16)
    (weight : Cuda.DevicePtr Cuda.Float8E4M3) (scales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (row columnQuads quad : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if quad < columnQuads then
    let inputValues ← Cuda.loadBFloat16x4 input quad.toUSize
    let weightQuad := row * columnQuads + quad
    let weights ← Cuda.loadReadOnlyFloat8E4M3x4 weight weightQuad.toUSize
    let scale ← Cuda.loadReadOnlyScaleUE8M0 scales (weightQuad / 8).toUSize
    let accumulator :=
      weights.fmaDotBFloat16x4 scale inputValues accumulator
    dotProjectionRowMXFP8 input weight scales row columnQuads
      (quad + blockThreadCount) accumulator
  else
    return accumulator

@[always_inline]
private partial def dotProjectionRowPairMXFP8 (input : Cuda.DevicePtr Cuda.BFloat16)
    (firstWeight secondWeight : Cuda.DevicePtr Cuda.Float8E4M3)
    (firstScales secondScales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (row columnQuads quad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  if quad < columnQuads then
    let inputValues ← Cuda.loadBFloat16x4 input quad.toUSize
    let weightQuad := row * columnQuads + quad
    let firstWeights ← Cuda.loadReadOnlyFloat8E4M3x4 firstWeight weightQuad.toUSize
    let secondWeights ← Cuda.loadReadOnlyFloat8E4M3x4 secondWeight weightQuad.toUSize
    let firstScale ← Cuda.loadReadOnlyScaleUE8M0 firstScales (weightQuad / 8).toUSize
    let secondScale ← Cuda.loadReadOnlyScaleUE8M0 secondScales (weightQuad / 8).toUSize
    let firstAccumulator :=
      firstWeights.fmaDotBFloat16x4 firstScale inputValues firstAccumulator
    let secondAccumulator :=
      secondWeights.fmaDotBFloat16x4 secondScale inputValues secondAccumulator
    dotProjectionRowPairMXFP8 input firstWeight secondWeight firstScales secondScales
      row columnQuads (quad + blockThreadCount) firstAccumulator secondAccumulator
  else
    return { first := firstAccumulator, second := secondAccumulator }

@[always_inline, convergent]
private partial def projectRowsFloatMXFP8 (input : Cuda.DevicePtr Cuda.BFloat16)
    (weight : Cuda.DevicePtr Cuda.Float8E4M3) (scales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (output : Cuda.DevicePtr Float32) (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSum ← dotProjectionRowMXFP8 input weight scales row (columns / 4) thread zero
    let total ← Cuda.Collective.blockSumLeader scratch localSum
    if thread == 0 then
      Cuda.storeFloat32 output row.toUSize total
    projectRowsFloatMXFP8 input weight scales output rows columns
      (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowPairsFloatMXFP8 (input : Cuda.DevicePtr Cuda.BFloat16)
    (weight : Cuda.DevicePtr Cuda.Float8E4M3) (scales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (output : Cuda.DevicePtr Float32) (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let columnQuads := columns / 4
    let secondWeight := weight + columns.toUSize
    let secondScales := scales + (columnQuads / 8).toUSize
    let localSums ← dotProjectionRowPairMXFP8 input weight secondWeight scales secondScales
      row columnQuads thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      Cuda.storeFloat32 output row.toUSize totals.first
      Cuda.storeFloat32 output (row + 1).toUSize totals.second
    projectRowPairsFloatMXFP8 input weight scales output rows columns
      (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private def runProjection (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns block thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8) :
    Cuda.DeviceM Unit := do
  if rows < gridBlockCount * 2 then
    projectRowsBFloat input weight output rows columns block gridBlockCount thread scratch
  else
    projectRowPairsBFloat input weight output rows columns (block * 2) (gridBlockCount * 2)
      thread scratch
  Cuda.gridSync


@[always_inline, convergent]
private def runResidualProjection (input weight residual : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns block thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8) :
    Cuda.DeviceM Unit := do
  projectRowPairsResidualBFloat input weight residual rows columns
    (block * 2) (gridBlockCount * 2) thread scratch
  Cuda.gridSync
@[always_inline, convergent]
private def runSwiGLUProjection
    (input gateWeight upWeight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns block thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8) :
    Cuda.DeviceM Unit := do
  projectRowsSwiGLU input gateWeight upWeight output rows columns
    block gridBlockCount thread scratch
  Cuda.gridSync

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
  let channel := block * blockThreadCount + thread
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
    decayState state head (index + blockThreadCount) decay

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
    updateState state keyShared deltaShared head (index + blockThreadCount)

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
private def runLinearRecurrent (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
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
    let aLog ← loadBFloat (← weightPointer weights item weightAttention0) head
    let dtBias ← loadBFloat (← weightPointer weights item weightAttention2) head
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
    let weight ← loadBFloat (← weightPointer weights item weightAttention7) thread
    let normalizedWeighted := rounded (recurrent * inverse * weight)
    let gate ← deviceSilu (← loadBFloat buffers.workspace1 (head * linearDimension + thread))
    storeBFloat buffers.attentionOutput (head * linearDimension + thread) (normalizedWeighted * gate)

@[always_inline, convergent]
private def runLinearLayer (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runProjection buffers.normalized (← weightPointer weights item weightAttention5) buffers.workspace0
    linearQkvSizeU32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (← weightPointer weights item weightAttention6) buffers.workspace1
    linearValueSizeU32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (← weightPointer weights item weightAttention3) buffers.workspace2
    linearHeads hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (← weightPointer weights item weightAttention4)
    (bfloatOffset buffers.workspace2 linearHeads) linearHeads hiddenSizeU32 block thread scratch
  runLinearConvolution item (← weightPointer weights item weightAttention1) buffers block thread
  Cuda.gridSync
  if block < linearHeads then
    runLinearRecurrent item weights buffers shared block thread
  Cuda.gridSync
  runResidualProjection buffers.attentionOutput (← weightPointer weights item weightAttention8)
    buffers.hidden hiddenSizeU32 linearValueSizeU32 block thread scratch

@[always_inline]
private def fullCacheIndex (item : ModelItem) (position head channel : UInt32) : UInt32 :=
  ((item.stateSlot * maxSequenceTokensU32 + position) *
    fullKeyValueProjectionSize.toUInt32) + head * fullDimension + channel

@[always_inline, convergent]
private def normalizeFullVector (source weight : Cuda.DevicePtr Cuda.BFloat16)
    (sourceBase thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8)
    (vector : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let value ← loadBFloat source (sourceBase + thread)
  let squareSum ← Cuda.Collective.blockSum scratch (value * value)
  let inverse ← Cuda.fastRsqrt (squareSum * inverseFullDimension + rmsEpsilon)
  let normWeight ← loadBFloat weight thread
  Cuda.storeFloat32 vector thread.toUSize (rounded (value * inverse * (one + normWeight)))
  Cuda.blockSync

@[always_inline]
private def applyPartialRope (position thread : UInt32) (vector : Cuda.DevicePtr Float32) :
    Cuda.DeviceM Float32 := do
  let value ← Cuda.loadFloat32 vector thread.toUSize
  if thread < rotaryDimension then
    let frequency := thread % rotaryHalf
    let partnerIndex := if thread < rotaryHalf then thread + rotaryHalf else thread - rotaryHalf
    let partner ← Cuda.loadFloat32 vector partnerIndex.toUSize
    let inverseFrequency := Cuda.Qwen36.Primitives.ropeInvFreqAt
      ropeTheta frequency rotaryDimension
    let angle := position.toFloat32 * inverseFrequency
    let cosine := rounded (Float32.cos angle)
    let sine := rounded (Float32.sin angle)
    if thread < rotaryHalf then
      return rounded (rounded (value * cosine) - rounded (partner * sine))
    else
      return rounded (rounded (partner * sine) + rounded (value * cosine))
  else
    return value

@[always_inline, convergent]
private def runFullPrepare (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (head thread : UInt32) : Cuda.DeviceM Unit := do
  if head < fullQueryHeads then
    let scratch : Cuda.Collective.BlockScratch 8 := shared
    let vector ← Cuda.dynamicShared (α := Float32) fullVectorOffset.toUSize
    normalizeFullVector buffers.workspace0 (← weightPointer weights item weightAttention0)
      (head * (2 * fullDimension)) thread scratch vector
    let query ← applyPartialRope item.position thread vector
    storeBFloat buffers.attentionOutput (head * fullDimension + thread) query
    Cuda.blockSync
    if head < fullKeyValueHeads then
      normalizeFullVector buffers.workspace1 (← weightPointer weights item weightAttention1)
        (head * fullDimension) thread scratch vector
      let key ← applyPartialRope item.position thread vector
      let cacheIndex := fullCacheIndex item item.position head thread
      storeBFloat buffers.keyCache cacheIndex key
      let value ← loadBFloat buffers.workspace2 (head * fullDimension + thread)
      storeBFloat buffers.valueCache cacheIndex value

@[always_inline]
private partial def fullScoreDot (item : ModelItem) (buffers : Buffers)
    (queryHead keyValueHead position channel : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if channel < fullDimension then
    let query ← loadBFloat buffers.attentionOutput (queryHead * fullDimension + channel)
    let key ← loadBFloat buffers.keyCache (fullCacheIndex item position keyValueHead channel)
    fullScoreDot item buffers queryHead keyValueHead position
      (channel + blockThreadCount)
      (Cuda.fma query key accumulator)
  else
    return accumulator

@[always_inline, convergent]
private partial def writeFullScores (item : ModelItem) (buffers : Buffers)
    (queryHead keyValueHead position thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) (scores : Cuda.DevicePtr Float32) :
    Cuda.DeviceM Unit := do
  if position <= item.position then
    let localScore ← fullScoreDot item buffers queryHead keyValueHead position thread zero
    let score ← Cuda.Collective.blockSumLeader scratch localScore
    if thread == 0 then
      Cuda.storeFloat32 scores position.toUSize (score * attentionScale)
    writeFullScores item buffers queryHead keyValueHead (position + 1) thread scratch scores

@[always_inline]
private partial def fullScoreMaximum (scores : Cuda.DevicePtr Float32)
    (limit index : UInt32) (maximum : Float32) : Cuda.DeviceM Float32 := do
  if index < limit then
    let value ← Cuda.loadFloat32 scores index.toUSize
    fullScoreMaximum scores limit (index + 1) (max maximum value)
  else
    return maximum

@[always_inline]
private partial def fullExponentialSum (scores : Cuda.DevicePtr Float32)
    (limit index : UInt32) (maximum accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < limit then
    let score ← Cuda.loadFloat32 scores index.toUSize
    let exponential ← Cuda.fastExp (score - maximum)
    Cuda.storeFloat32 scores index.toUSize exponential
    fullExponentialSum scores limit (index + 1) maximum (accumulator + exponential)
  else
    return accumulator

@[always_inline]
private partial def normalizeFullProbabilities (scores : Cuda.DevicePtr Float32)
    (limit index : UInt32) (inverse : Float32) : Cuda.DeviceM Unit := do
  if index < limit then
    let exponential ← Cuda.loadFloat32 scores index.toUSize
    Cuda.storeFloat32 scores index.toUSize (rounded (exponential * inverse))
    normalizeFullProbabilities scores limit (index + 1) inverse

@[always_inline]
private partial def weightedFullValue (item : ModelItem) (buffers : Buffers)
    (keyValueHead channel position limit : UInt32) (scores : Cuda.DevicePtr Float32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if position < limit then
    let probability ← Cuda.loadFloat32 scores position.toUSize
    let value ← loadBFloat buffers.valueCache
      (fullCacheIndex item position keyValueHead channel)
    weightedFullValue item buffers keyValueHead channel (position + 1) limit scores
      (Cuda.fma probability value accumulator)
  else
    return accumulator

@[always_inline, convergent]
private def runFullAttention (item : ModelItem) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (head thread : UInt32) : Cuda.DeviceM Unit := do
  if head < fullQueryHeads then
    let scratch : Cuda.Collective.BlockScratch 8 := shared
    let scores ← Cuda.dynamicShared (α := Float32) fullScoresOffset.toUSize
    let keyValueHead := head / fullGroups
    writeFullScores item buffers head keyValueHead 0 thread scratch scores
    if thread == 0 then
      let limit := item.position + 1
      let first ← Cuda.loadFloat32 scores 0
      let maximum ← fullScoreMaximum scores limit 1 first
      let denominator ← fullExponentialSum scores limit 0 maximum zero
      normalizeFullProbabilities scores limit 0 (one / denominator)
    Cuda.blockSync
    let attention ← weightedFullValue item buffers keyValueHead thread 0
      (item.position + 1) scores zero
    let gateInput ← loadBFloat buffers.workspace0
      (head * (2 * fullDimension) + fullDimension + thread)
    let gate := rounded (← deviceSigmoid gateInput)
    storeBFloat buffers.attentionOutput (head * fullDimension + thread)
      (rounded attention * gate)

@[always_inline, convergent]
private def runFullLayer (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runProjection buffers.normalized (← weightPointer weights item weightAttention2) buffers.workspace0
    fullQueryProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (← weightPointer weights item weightAttention3) buffers.workspace1
    fullKeyValueProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runProjection buffers.normalized (← weightPointer weights item weightAttention4) buffers.workspace2
    fullKeyValueProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runFullPrepare item weights buffers shared block thread
  Cuda.gridSync
  runFullAttention item buffers shared block thread
  Cuda.gridSync
  runResidualProjection buffers.attentionOutput (← weightPointer weights item weightAttention5)
    buffers.hidden hiddenSizeU32 fullOutputSize.toUInt32 block thread scratch

@[always_inline, convergent]
private def runMLP (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runRmsNorm buffers.hidden (← weightPointer weights item weightPostNorm) buffers.normalized buffers scratch
    block thread globalThread globalThreads
  runSwiGLUProjection buffers.normalized (← weightPointer weights item weightMlpGate)
    (← weightPointer weights item weightMlpUp) buffers.workspace2 intermediateSizeU32 hiddenSizeU32
    block thread scratch
  runResidualProjection buffers.workspace2 (← weightPointer weights item weightMlpDown)
    buffers.hidden hiddenSizeU32 intermediateSizeU32 block thread scratch

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
private partial def sampleMaximumLoop (logits : Cuda.DevicePtr Float32)
    (sampled : Cuda.DevicePtr UInt32) (maximum : Float32)
    (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < vocabularySizeU32 then
    if (← Cuda.loadFloat32 logits index.toUSize) == maximum then
      discard <| Cuda.atomicMinUInt32 sampled index
    sampleMaximumLoop logits sampled maximum (index + stride) stride

@[always_inline]
private partial def gradientAndSampleLoop (item : ModelItem)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (sampled : Cuda.DevicePtr UInt32)
    (maximum inverseDenominator : Float32) (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < vocabularySizeU32 then
    let value ← Cuda.loadFloat32 logits index.toUSize
    let exponential ← Cuda.fastExp2Fma ((value - maximum) * log2e)
    let target := if index == item.target then one else zero
    storeBFloat gradient index (exponential * inverseDenominator - target)
    if value == maximum then
      discard <| Cuda.atomicMinUInt32 sampled index
    gradientAndSampleLoop item logits gradient sampled maximum inverseDenominator
      (index + stride) stride

@[always_inline, convergent]
private def runFinal (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (outputQueue : Cuda.Mailbox.Device TokenEvent)
    (shared : Cuda.DevicePtr Float32) (block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  let logits := floatOffset buffers.logits (item.position * vocabularySizeU32)
  let gradient := bfloatOffset buffers.gradient (item.position * vocabularySizeU32)
  let sampled := buffers.sampled + item.position.toUSize * 4
  let finalHidden := bfloatOffset buffers.finalHidden (item.position * hiddenSizeU32)
  runRmsNorm buffers.hidden (← weightPointer weights item weightInputNorm) finalHidden buffers scratch
    block thread globalThread globalThreads
  if useGreedyLmHeadMXFP8 && item.kind == kindGreedyFinal then
    projectRowPairsFloatMXFP8 finalHidden buffers.lmHeadData buffers.lmHeadScales logits
      vocabularySizeU32 hiddenSizeU32 (block * 2) (gridBlockCount * 2) thread scratch
  else
    projectRowPairsFloat finalHidden (← weightPointer weights item weightAttention0) logits
      vocabularySizeU32 hiddenSizeU32 (block * 2) (gridBlockCount * 2) thread scratch
  Cuda.gridSync
  if block == 0 && thread == 0 then
    Cuda.storeUInt32 sampled 0 0xffffffff
  Cuda.gridSync
  let localMaximum ← localMaximumLoop logits globalThread globalThreads negativeInfinity
  let blockMaximum ← Cuda.Collective.blockMaxLeader scratch localMaximum
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialMaximum block.toUSize blockMaximum
  Cuda.gridSync
  let maximum ← maximumPartials buffers.partialMaximum 0 negativeInfinity
  if item.kind == kindFinal then
    let localSum ← localExponentialSumLoop logits maximum globalThread globalThreads zero
    let blockSum ← Cuda.Collective.blockSumLeader scratch localSum
    if thread == 0 then
      Cuda.storeFloat32 buffers.partialSum block.toUSize blockSum
    Cuda.gridSync
    let denominator ← sumPartials buffers.partialSum 0 zero
    let inverseDenominator ← Cuda.fastDivide one denominator
    gradientAndSampleLoop item logits gradient sampled maximum inverseDenominator
      globalThread globalThreads
    if block == 0 && thread == 0 then
      let targetLogit ← Cuda.loadFloat32 logits item.target.toUSize
      Cuda.storeFloat32 buffers.loss item.position.toUSize
        (maximum + (← Cuda.fastLog denominator) - targetLogit)
  else
    sampleMaximumLoop logits sampled maximum globalThread globalThreads
  Cuda.gridSync
  if block == 0 && thread == 0 then
    let event := buffers.tokenEvents + item.position.toUSize * 8
    Cuda.storeUInt32 (event : Cuda.DevicePtr UInt32) 0 item.position
    Cuda.storeUInt32 (event : Cuda.DevicePtr UInt32) 1 (← Cuda.loadUInt32 sampled 0)
    let result ← Cuda.Mailbox.Device.trySendFrom outputQueue event
    unless result.isSent do
      Cuda.panic 0x3603 result.code.toUInt64

@[cuda_grid_persistent]
def qwen36Model (item : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8)
    (lmHeadData : Cuda.DevicePtr Cuda.Float8E4M3)
    (lmHeadScales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput
      convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (keyCache valueCache : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled counts : Cuda.DevicePtr UInt32) (tokenEvents : Cuda.DevicePtr TokenEvent)
    (outputStorage : Cuda.DeviceSlice UInt8) (outputLayout : Cuda.Mailbox.Layout) :
    Cuda.DeviceM Unit := do
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let globalThread ← Cuda.globalThreadIdxX
  let globalThreads ← Cuda.globalThreadCountX
  let shared ← Cuda.dynamicShared (α := Float32)
  let buffers : Buffers := {
    hidden, normalized, finalHidden, workspace0, workspace1, workspace2, attentionOutput,
    lmHeadData, lmHeadScales, convolutionState, recurrentState,
    keyCache, valueCache, logits, gradient,
    partialMaximum,
    partialSum, loss, sampled, counts, tokenEvents
  }
  let outputQueue : Cuda.Mailbox.Device TokenEvent :=
    Cuda.Mailbox.attachSystem outputStorage outputLayout
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  if item.position >= maxSequenceTokensU32 then
    Cuda.panic 0x3601 item.position.toUInt64
  else if item.kind == kindEmbedding then
    embeddingLoop (← weightPointer weights item weightInputNorm) buffers.hidden item.token
      globalThread globalThreads
    Cuda.gridSync
  else if item.kind == kindLinear then
    runRmsNorm buffers.hidden (← weightPointer weights item weightInputNorm) buffers.normalized buffers scratch
      block thread globalThread globalThreads
    runLinearLayer item weights buffers shared block thread
    runMLP item weights buffers shared block thread globalThread globalThreads
  else if item.kind == kindFull then
    runRmsNorm buffers.hidden (← weightPointer weights item weightInputNorm) buffers.normalized buffers scratch
      block thread globalThread globalThreads
    runFullLayer item weights buffers shared block thread
    runMLP item weights buffers shared block thread globalThread globalThreads
  else if item.kind == kindFinal || item.kind == kindGreedyFinal then
    runFinal item weights buffers outputQueue shared block thread globalThread globalThreads
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
    (counts : Cuda.DevicePtr UInt32) (descriptorCount : UInt32) : Cuda.DeviceM Unit := do
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
  zeroUIntLoop counts descriptorCount index stride

@[cuda_kernel]
def quantizeLmHeadMXFP8 (sourceBytes : Cuda.DevicePtr UInt8) (sourceByteOffset : UInt64)
    (data : Cuda.DevicePtr Cuda.Float8E4M3) (scales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (blockCount : UInt32) : Cuda.DeviceM Unit := do
  let block ← Primitives.elementIndex
  if block < blockCount then
    let source : Cuda.DevicePtr Cuda.BFloat16 := sourceBytes + sourceByteOffset.toUSize
    let sourceBlock := Cuda.MXFP8.BFloat16Block32.select source block
    let destination := Cuda.MXFP8.E4M3Block32.select data block
    let scale ← Cuda.MXFP8.quantizeBlock32 sourceBlock destination
    Cuda.storeScaleUE8M0 scales block.toUSize scale

@[cuda_kernel]
def copyTrainingBF16Kernel (source destination : Cuda.DevicePtr Cuda.BFloat16)
    (destinationOffset count : UInt32) : Cuda.DeviceM Unit := do
  let index ← Primitives.elementIndex
  if index < count then
    Cuda.storeBFloat16 destination (destinationOffset + index).toUSize
      (← Cuda.loadBFloat16 source index.toUSize)

@[cuda_kernel]
def copyTrainingF32Kernel (source destination : Cuda.DevicePtr Float32)
    (destinationOffset count : UInt32) : Cuda.DeviceM Unit := do
  let index ← Primitives.elementIndex
  if index < count then
    Cuda.storeFloat32 destination (destinationOffset + index).toUSize
      (← Cuda.loadFloat32 source index.toUSize)

private structure WeightRelocation where
  originalShard : UInt32
  originalOffset : UInt64
  byteCount : UInt64
  packedShard : UInt32
  packedOffset : UInt64

private structure LoadedWeights where
  buffers : Array (Cuda.Buffer UInt8)
  relocated : Array WeightRelocation

private structure QuantizedLmHead where
  data : Cuda.Buffer Cuda.Float8E4M3
  scales : Cuda.Buffer Cuda.ScaleUE8M0

private def LoadedWeights.getBuffer (weights : @& LoadedWeights) (index : UInt32) :
    IO (Cuda.Buffer UInt8) := do
  let some buffer := weights.buffers[index.toNat]?
    | throw <| IO.userError
        s!"invalid Qwen checkpoint weight-buffer index {index}; loaded {weights.buffers.size} buffers"
  return buffer

private def LoadedWeights.resolve (weights : @& LoadedWeights) (originalShard : UInt32)
    (originalOffset byteCount : UInt64) : IO (Cuda.Buffer UInt8 × UInt64) := do
  if weights.relocated.isEmpty || byteCount == 0 then
    return (← weights.getBuffer originalShard, originalOffset)
  let some relocated := weights.relocated.findSome? fun relocation =>
      if relocation.originalShard == originalShard &&
          relocation.originalOffset == originalOffset then
        some relocation
      else
        none
    | throw <| IO.userError
        s!"packed Qwen checkpoint address is missing: shard={originalShard}, offset={originalOffset}"
  unless relocated.byteCount == byteCount do
    throw <| IO.userError
      s!"packed Qwen checkpoint address has {relocated.byteCount} bytes, expected {byteCount}"
  return (← weights.getBuffer relocated.packedShard, relocated.packedOffset)

/-- Checked host-side ownership layout for one bounded persistent inference sequence. -/
structure PersistentLayout where
  tokenCount : Nat
  descriptorCount : Nat
  convolutionElements : Nat
  recurrentElements : Nat
  keyValueCacheElements : Nat
  logitsElements : Nat
  deriving Repr, BEq

/-- Derive every state/output extent from a token count, rejecting empty or over-context runs. -/
def PersistentLayout.forTokens (tokenCount : Nat) : Except String PersistentLayout := do
  if tokenCount == 0 then
    throw "persistent Qwen3.8 inference requires at least one token"
  if tokenCount > maxSequenceTokens then
    throw s!"persistent Qwen3.8 inference supports at most {maxSequenceTokens} tokens, got {tokenCount}"
  return {
    tokenCount
    descriptorCount := tokenCount * descriptorsPerTokenNat
    convolutionElements := linearLayerCount * convolutionElementsPerLayerNat
    recurrentElements := linearLayerCount * linearStateElementsPerLayerNat
    keyValueCacheElements := fullAttentionLayerCount * fullCacheElementsPerLayerNat
    logitsElements := tokenCount * vocabularySize
  }

private structure ModelBuffers where
  layout : PersistentLayout
  hidden : Cuda.Buffer Cuda.BFloat16
  normalized : Cuda.Buffer Cuda.BFloat16
  finalHidden : Cuda.Buffer Cuda.BFloat16
  workspace0 : Cuda.Buffer Cuda.BFloat16
  workspace1 : Cuda.Buffer Cuda.BFloat16
  workspace2 : Cuda.Buffer Cuda.BFloat16
  attentionOutput : Cuda.Buffer Cuda.BFloat16
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
  tokenEvents : Cuda.Buffer TokenEvent

private def requireTensor (manifest : PayloadManifest) (name : String)
    (shape : Array Nat) : IO TensorRef := do
  let some resolved := manifest.findTensor? name
    | throw <| IO.userError s!"checkpoint tensor is missing: {name}"
  let tensor := resolved.tensor
  unless tensor.dtype == .bf16 do
    throw <| IO.userError s!"checkpoint tensor '{name}' has dtype {tensor.dtype}, expected BF16"
  unless tensor.shape == shape do
    throw <| IO.userError
      s!"checkpoint tensor '{name}' has shape {tensor.shape}, expected {shape}"
  return {
    name
    shard := resolved.shard
    byteOffset := resolved.byteOffset
    byteCount := tensor.byteSize.toUInt64
  }

private def layerPrefix (layer : Nat) : String :=
  s!"model.language_model.layers.{layer}"

private def readCommonWeights (manifest : PayloadManifest) (layer : Nat) :
    IO CommonLayerWeights := do
  let baseName := layerPrefix layer
  let inputNorm ← requireTensor manifest s!"{baseName}.input_layernorm.weight" #[hiddenSize]
  let postAttentionNorm ← requireTensor manifest
    s!"{baseName}.post_attention_layernorm.weight" #[hiddenSize]
  let mlpGate ← requireTensor manifest s!"{baseName}.mlp.gate_proj.weight"
    #[intermediateSize, hiddenSize]
  let mlpUp ← requireTensor manifest s!"{baseName}.mlp.up_proj.weight"
    #[intermediateSize, hiddenSize]
  let mlpDown ← requireTensor manifest s!"{baseName}.mlp.down_proj.weight"
    #[hiddenSize, intermediateSize]
  return { inputNorm, postAttentionNorm, mlpGate, mlpUp, mlpDown }

private def readLinearWeights (manifest : PayloadManifest) (layer : Nat) :
    IO LinearAttentionWeights := do
  let baseName := s!"{layerPrefix layer}.linear_attn"
  let aLog ← requireTensor manifest s!"{baseName}.A_log" #[linearHeadCount]
  let convolution ← requireTensor manifest s!"{baseName}.conv1d.weight"
    #[linearQkvSize, 1, 4]
  let dtBias ← requireTensor manifest s!"{baseName}.dt_bias" #[linearHeadCount]
  let projectionA ← requireTensor manifest s!"{baseName}.in_proj_a.weight"
    #[linearHeadCount, hiddenSize]
  let projectionB ← requireTensor manifest s!"{baseName}.in_proj_b.weight"
    #[linearHeadCount, hiddenSize]
  let projectionQkv ← requireTensor manifest s!"{baseName}.in_proj_qkv.weight"
    #[linearQkvSize, hiddenSize]
  let projectionZ ← requireTensor manifest s!"{baseName}.in_proj_z.weight"
    #[linearValueSize, hiddenSize]
  let norm ← requireTensor manifest s!"{baseName}.norm.weight" #[linearHeadDimension]
  let outputProjection ← requireTensor manifest s!"{baseName}.out_proj.weight"
    #[hiddenSize, linearValueSize]
  return {
    aLog, convolution, dtBias, projectionA, projectionB, projectionQkv, projectionZ, norm,
    outputProjection
  }

private def readFullWeights (manifest : PayloadManifest) (layer : Nat) :
    IO FullAttentionWeights := do
  let baseName := s!"{layerPrefix layer}.self_attn"
  let queryNorm ← requireTensor manifest s!"{baseName}.q_norm.weight" #[fullHeadDimension]
  let keyNorm ← requireTensor manifest s!"{baseName}.k_norm.weight" #[fullHeadDimension]
  let queryProjection ← requireTensor manifest s!"{baseName}.q_proj.weight"
    #[fullQueryProjectionSize, hiddenSize]
  let keyProjection ← requireTensor manifest s!"{baseName}.k_proj.weight"
    #[fullKeyValueProjectionSize, hiddenSize]
  let valueProjection ← requireTensor manifest s!"{baseName}.v_proj.weight"
    #[fullKeyValueProjectionSize, hiddenSize]
  let outputProjection ← requireTensor manifest s!"{baseName}.o_proj.weight"
    #[hiddenSize, fullOutputSize]
  return {
    queryNorm, keyNorm, queryProjection, keyProjection, valueProjection, outputProjection
  }

private def readTextLayout (manifest : PayloadManifest) : IO TextLayout := do
  let report := checkComplete { numLayers := layerCount, fullAttentionPeriod := 4 }
    manifest.schema
  unless report.isClean do
    throw <| IO.userError
      s!"Qwen3.8 text checkpoint schema mismatch: missing={report.missing}, unexpected={report.unexpected}"
  let embedding ← requireTensor manifest "model.language_model.embed_tokens.weight"
    #[vocabularySize, hiddenSize]
  let mut layers := #[]
  for layer in [:layerCount] do
    let common ← readCommonWeights manifest layer
    let attention ← match layerKind layer with
      | .linearAttention => pure (.linear (← readLinearWeights manifest layer))
      | .fullAttention => pure (.full (← readFullWeights manifest layer))
    layers := layers.push { index := layer, common, attention }
  let finalNorm ← requireTensor manifest "model.language_model.norm.weight" #[hiddenSize]
  let lmHead ← requireTensor manifest "lm_head.weight" #[vocabularySize, hiddenSize]
  return { embedding, layers, finalNorm, lmHead }

private def TextLayout.tensorRefs (layout : TextLayout) : Array TensorRef := Id.run do
  let mut tensors := #[layout.embedding]
  for layer in layout.layers do
    let common := layer.common
    tensors := tensors ++ #[
      common.inputNorm, common.postAttentionNorm, common.mlpGate, common.mlpUp, common.mlpDown
    ]
    match layer.attention with
    | .linear weights =>
      tensors := tensors ++ #[
        weights.aLog, weights.convolution, weights.dtBias, weights.projectionA,
        weights.projectionB, weights.projectionQkv, weights.projectionZ, weights.norm,
        weights.outputProjection
      ]
    | .full weights =>
      tensors := tensors ++ #[
        weights.queryNorm, weights.keyNorm, weights.queryProjection, weights.keyProjection,
        weights.valueProjection, weights.outputProjection
      ]
  tensors := tensors.push layout.finalNorm
  tensors := tensors.push layout.lmHead
  return tensors

private def loadRawWeights (manifest : PayloadManifest) (stream : Cuda.Stream) :
    IO LoadedWeights := do
  if manifest.shards.isEmpty then
    throw <| IO.userError "Qwen checkpoint contains no weight shards"
  let mut buffers : Array (Cuda.Buffer UInt8) := #[]
  for index in [:manifest.shards.size] do
    IO.println s!"loading Qwen shard {index + 1}/{manifest.shards.size}: {manifest.shards[index]!.filename}"
    buffers := buffers.push (← manifest.loadShard index stream).2
  return { buffers, relocated := #[] }

private def loadAlignedWeights (manifest : PayloadManifest) (layout : TextLayout)
    (stream : Cuda.Stream) : IO LoadedWeights := do
  let tensors := layout.tensorRefs
  IO.println s!"packing {tensors.size} Qwen3.8 text tensors at {weightTensorAlignmentNat}-byte alignment"
  let packed ← manifest.loadPacked (tensors.map (·.name)) stream weightTensorAlignmentNat
  let mut relocated : Array WeightRelocation := #[]
  for tensor in tensors do
    let some packedTensor := packed.findTensor? tensor.name
      | throw <| IO.userError s!"packed Qwen checkpoint tensor is missing: {tensor.name}"
    relocated := relocated.push {
      originalShard := tensor.shard
      originalOffset := tensor.byteOffset
      byteCount := tensor.byteCount
      packedShard := packedTensor.shard
      packedOffset := packedTensor.byteOffset
    }
  IO.println s!"packed Qwen3.8 text weights: {packed.byteSize} bytes in {packed.buffers.size} buffers"
  return { buffers := packed.buffers, relocated }

private def loadWeights (manifest : PayloadManifest) (layout : TextLayout)
    (stream : Cuda.Stream) : IO LoadedWeights :=
  if useAlignedTextWeights then
    loadAlignedWeights manifest layout stream
  else
    loadRawWeights manifest stream

private def quantizeLmHead (weights : @& LoadedWeights) (layout : @& TextLayout)
    (stream : @& Cuda.Stream) : IO QuantizedLmHead := do
  unless layout.lmHead.byteCount == (lmHeadElementsNat * 2).toUInt64 do
    throw <| IO.userError
      s!"Qwen3.8 LM head has {layout.lmHead.byteCount} bytes, expected {lmHeadElementsNat * 2}"
  unless lmHeadElementsNat % lmHeadScaleBlockElementsNat == 0 do
    throw <| IO.userError "Qwen3.8 LM head is not divisible into 32-value MXFP8 blocks"
  let (source, sourceOffset) ← weights.resolve layout.lmHead.shard
    layout.lmHead.byteOffset layout.lmHead.byteCount
  let data ← Cuda.Buffer.alloc Cuda.Float8E4M3 lmHeadElementsNat.toUSize
  let scales ← Cuda.Buffer.alloc Cuda.ScaleUE8M0 lmHeadScaleBlockCountNat.toUSize
  IO.println s!"quantizing Qwen3.8 greedy LM head: {lmHeadScaleBlockCountNat} MXFP8 blocks"
  (← quantizeLmHeadMXFP8.launchOn stream
    (Primitives.elementConfig lmHeadScaleBlockCountNat.toUInt32)
    source sourceOffset data scales lmHeadScaleBlockCountNat.toUInt32).waitChecked
    "Qwen3.8 greedy LM-head MXFP8 quantization"
  return { data, scales }

private def allocateBuffers (layout : PersistentLayout) : IO ModelBuffers := do
  let hidden ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSize.toUSize
  let normalized ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSize.toUSize
  let finalHidden ← Cuda.Buffer.alloc Cuda.BFloat16
    (layout.tokenCount * hiddenSize).toUSize
  let workspace0 ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSize.toUSize
  let workspace1 ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSize.toUSize
  let workspace2 ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSize.toUSize
  let attentionOutput ← Cuda.Buffer.alloc Cuda.BFloat16 linearValueSize.toUSize
  let convolutionState ← Cuda.Buffer.alloc Cuda.BFloat16
    layout.convolutionElements.toUSize
  let recurrentState ← Cuda.Buffer.alloc Float32
    layout.recurrentElements.toUSize
  let keyCache ← Cuda.Buffer.alloc Cuda.BFloat16
    layout.keyValueCacheElements.toUSize
  let valueCache ← Cuda.Buffer.alloc Cuda.BFloat16
    layout.keyValueCacheElements.toUSize
  let logits ← Cuda.Buffer.alloc Float32 layout.logitsElements.toUSize
  let gradient ← Cuda.Buffer.alloc Cuda.BFloat16 layout.logitsElements.toUSize
  let partialMaximum ← Cuda.Buffer.alloc Float32 gridBlockCountNat.toUSize
  let partialSum ← Cuda.Buffer.alloc Float32 gridBlockCountNat.toUSize
  let loss ← Cuda.Buffer.alloc Float32 layout.tokenCount.toUSize
  let sampled ← Cuda.Buffer.alloc UInt32 layout.tokenCount.toUSize
  let counts ← Cuda.Buffer.alloc UInt32 layout.descriptorCount.toUSize
  let tokenEvents ← Cuda.Buffer.alloc TokenEvent layout.tokenCount.toUSize
  return {
    layout, hidden, normalized, finalHidden, workspace0, workspace1, workspace2, attentionOutput,
    convolutionState, recurrentState, keyCache, valueCache, logits, gradient,
    partialMaximum,
    partialSum, loss, sampled, counts, tokenEvents
  }

private def toWeightRef (tensor : TensorRef) : WeightRef := {
  shard := tensor.shard
  byteOffset := tensor.byteOffset
  byteCount := tensor.byteCount
}

private def zeroWeight : WeightRef := { shard := 0, byteOffset := 0, byteCount := 0 }

private def blankWeights : ModelWeights := {
  inputNorm := zeroWeight, postNorm := zeroWeight, mlpGate := zeroWeight,
  mlpUp := zeroWeight, mlpDown := zeroWeight, attention0 := zeroWeight,
  attention1 := zeroWeight, attention2 := zeroWeight, attention3 := zeroWeight,
  attention4 := zeroWeight, attention5 := zeroWeight, attention6 := zeroWeight,
  attention7 := zeroWeight, attention8 := zeroWeight
}

private def embeddingWeights (layout : TextLayout) : ModelWeights := {
  blankWeights with
  inputNorm := toWeightRef layout.embedding
}

private def layerWeights (layer : LayerWeights) : ModelWeights :=
  let common := layer.common
  let base : ModelWeights := {
    blankWeights with
    inputNorm := toWeightRef common.inputNorm
    postNorm := toWeightRef common.postAttentionNorm
    mlpGate := toWeightRef common.mlpGate
    mlpUp := toWeightRef common.mlpUp
    mlpDown := toWeightRef common.mlpDown
  }
  match layer.attention with
  | .linear weights => {
      base with
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
      attention0 := toWeightRef weights.queryNorm
      attention1 := toWeightRef weights.keyNorm
      attention2 := toWeightRef weights.queryProjection
      attention3 := toWeightRef weights.keyProjection
      attention4 := toWeightRef weights.valueProjection
      attention5 := toWeightRef weights.outputProjection
    }

private def finalWeights (layout : TextLayout) : ModelWeights := {
  blankWeights with
  inputNorm := toWeightRef layout.finalNorm
  attention0 := toWeightRef layout.lmHead
}

private def blankItem : ModelItem := {
  kind := 0
  stateSlot := 0
  weightSlot := 0
  token := 0
  position := 0
  target := 0
  descriptor := 0
}

private def embeddingItem (token position descriptor : UInt32) : ModelItem := {
  blankItem with
  kind := kindEmbedding
  token
  position
  descriptor
}

private def layerItem (layer : LayerWeights)
    (slot position descriptor token target : UInt32) : ModelItem := {
  blankItem with
  kind := match layer.attention with
    | .linear _ => kindLinear
    | .full _ => kindFull
  stateSlot := slot
  weightSlot := (layer.index + 1).toUInt32
  token
  position
  target
  descriptor
}

private def finalItem (position target descriptor : UInt32)
    (greedyOnly : Bool) : ModelItem := {
  blankItem with
  kind := if greedyOnly then kindGreedyFinal else kindFinal
  weightSlot := (layerCount + 1).toUInt32
  position
  target
  descriptor
}

private def ModelWeights.toArray (weights : ModelWeights) : Array WeightRef :=
  #[weights.inputNorm, weights.postNorm, weights.mlpGate, weights.mlpUp, weights.mlpDown,
    weights.attention0, weights.attention1, weights.attention2, weights.attention3,
    weights.attention4, weights.attention5, weights.attention6, weights.attention7,
    weights.attention8]

private def resolveWeightTable (weights : @& LoadedWeights) (layout : @& TextLayout) :
    IO (Cuda.BufferTable UInt8) := do
  let mut records := #[embeddingWeights layout]
  for layer in layout.layers do
    records := records.push (layerWeights layer)
  records := records.push (finalWeights layout)
  let mut owners : Array (Cuda.Buffer UInt8) := #[]
  let mut byteOffsets : Array USize := #[]
  for record in records do
    for weight in record.toArray do
      let (owner, byteOffset) ← weights.resolve weight.shard
        weight.byteOffset weight.byteCount
      owners := owners.push owner
      byteOffsets := byteOffsets.push byteOffset.toUSize
  unless owners.size == records.size * weightFieldCount.toNat do
    throw <| IO.userError "Qwen3.8 resolved weight table has an inconsistent field count"
  Cuda.BufferTable.createAtByteOffsets owners byteOffsets

private def initializerConfig : Cuda.LaunchConfig := {
  grid := { x := 256 }
  block := { x := 256 }
  blockArenaBytes := 4 * 1024
}

private def persistentConfig (layout : PersistentLayout) : Cuda.GridPersistentConfig := {
  grid := { x := gridBlockCount }
  block := { x := blockThreadCount }
  sharedMemoryBytes := sharedBytes.toUSize
  blockArenaBytes := 64 * 1024
  queueCapacity := layout.descriptorCount.toUSize
}

private def readUInt32 (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

/-- Position-major outputs collected after a streamed persistent launch finishes. -/
structure InferenceResult where
  logits : ByteArray
  loss : ByteArray
  sampled : ByteArray
  deriving BEq

/-- Device-resident frozen-base outputs retained for a real-checkpoint training head. -/
structure TrainingBatch where
  rows : UInt32
  inputFeatures : UInt32
  outputFeatures : UInt32
  finalHidden : Cuda.Buffer Cuda.BFloat16
  logits : Cuda.Buffer Float32

/--
A single-consumer host view of a live Qwen3.8 persistent launch.

`enqueue` submits one causal position, `nextToken` waits only for that position's mapped token
publication, and `finish` closes a complete launch without copying the full logits tensor.
`finishEarly` cleanly closes a partially consumed launch, for example after an end token.
`collect` additionally returns all position-major outputs from a complete launch.
-/
structure InferenceStream where
  expectedPositions : Nat
  enqueue : UInt32 → UInt32 → IO Unit
  enqueueGreedy : UInt32 → IO Unit
  nextToken : IO TokenEvent
  finish : IO Unit
  finishEarly : IO Unit
  collect : IO InferenceResult
  abort : IO Unit

private structure InferenceStreamState where
  submitted : Nat := 0
  received : Nat := 0
  finalized : Bool := false
  aborted : Bool := false

private def enqueuePosition (handle : @& Cuda.GridPersistentHandle ModelItem)
    (layout : @& TextLayout) (position : Nat) (token target : UInt32)
    (greedyOnly : Bool) : IO Unit := do
  let descriptorBase := position * descriptorsPerTokenNat
  qwen36Model.enqueue handle (embeddingItem token position.toUInt32
    descriptorBase.toUInt32)
  let mut linearSlot : UInt32 := 0
  let mut fullSlot : UInt32 := 0
  for index in [:layout.layers.size] do
    let some layer := layout.layers[index]?
      | throw <| IO.userError s!"missing resolved Qwen3.8 layer {index}"
    let slot := match layer.attention with
      | .linear _ => linearSlot
      | .full _ => fullSlot
    qwen36Model.enqueue handle (layerItem layer slot position.toUInt32
      (descriptorBase + index + 1).toUInt32 token target)
    match layer.attention with
    | .linear _ => linearSlot := linearSlot + 1
    | .full _ => fullSlot := fullSlot + 1
  qwen36Model.enqueue handle (finalItem position.toUInt32 target
    (descriptorBase + layerCount + 1).toUInt32 greedyOnly)

private partial def awaitTokenEvent (queue : @& Cuda.HostIO.Queue TokenEvent)
    (handle : @& Cuda.GridPersistentHandle ModelItem) : IO TokenEvent := do
  if let some event ← queue.tryReceive then
    return event
  let status ← handle.query
  if status.state == .failed then
    handle.waitChecked "Qwen3.8 streaming model megakernel"
    throw <| IO.userError "Qwen3.8 streaming model megakernel failed without a diagnostic"
  else if status.state == .completed then
    throw <| IO.userError "Qwen3.8 streaming model ended before publishing its next token"
  else
    IO.sleep 1
    awaitTokenEvent queue handle

private def validateDescriptorCounts (buffers : @& ModelBuffers) (positions : Nat)
    (stream : @& Cuda.Stream) : IO Unit := do
  let counts ← buffers.counts.copyTo stream
  for descriptor in [:positions * descriptorsPerTokenNat] do
    unless readUInt32 counts descriptor == 1 do
      throw <| IO.userError
        s!"Qwen3.8 descriptor {descriptor} executed {readUInt32 counts descriptor} times"

private def startLoadedStream (layout : @& TextLayout)
    (weights : @& Cuda.BufferTable UInt8) (lmHead : @& QuantizedLmHead)
    (buffers : @& ModelBuffers) (stream : @& Cuda.Stream) : IO InferenceStream := do
  let persistentLayout := buffers.layout
  (← initializeModelState.launchOn stream initializerConfig buffers.convolutionState
    buffers.recurrentState buffers.keyCache buffers.valueCache buffers.counts
    persistentLayout.descriptorCount.toUInt32).waitChecked "Qwen3.8 state initialization"
  let outputQueue ← Cuda.HostIO.Queue.alloc TokenEvent maxSequenceTokens.toUSize
  let handle ← qwen36Model.startOn stream (persistentConfig persistentLayout)
    weights
    lmHead.data lmHead.scales
    buffers.hidden buffers.normalized buffers.finalHidden buffers.workspace0 buffers.workspace1
    buffers.workspace2 buffers.attentionOutput buffers.convolutionState
    buffers.recurrentState
    buffers.keyCache buffers.valueCache buffers.logits buffers.gradient buffers.partialMaximum
    buffers.partialSum buffers.loss buffers.sampled buffers.counts buffers.tokenEvents
    outputQueue.buffer outputQueue.layout
  let state ← IO.mkRef ({} : InferenceStreamState)
  let close (requireComplete : Bool) : IO Unit := do
    let current ← state.get
    if current.finalized then
      return
    if current.aborted then
      throw <| IO.userError "Qwen3.8 inference stream was aborted"
    unless current.received == current.submitted do
      throw <| IO.userError <|
        s!"Qwen3.8 inference stream submitted {current.submitted} positions but received " ++
        s!"{current.received} tokens"
    if requireComplete && current.submitted != persistentLayout.tokenCount then
      throw <| IO.userError <|
        s!"Qwen3.8 inference stream expected {persistentLayout.tokenCount} positions, " ++
        s!"submitted {current.submitted}"
    handle.shutdown
    handle.waitChecked "Qwen3.8 model megakernel"
    validateDescriptorCounts buffers current.submitted stream
    state.set { current with finalized := true }
  let finalize := close true
  let abort : IO Unit := do
    let current ← state.get
    unless current.finalized || current.aborted do
      state.set { current with aborted := true }
      try
        handle.shutdown
        discard <| handle.wait
      catch _ =>
        pure ()
  let enqueue (greedyOnly : Bool) (token target : UInt32) : IO Unit := do
    let current ← state.get
    if current.finalized || current.aborted then
      throw <| IO.userError "cannot enqueue into a closed Qwen3.8 inference stream"
    if current.submitted >= persistentLayout.tokenCount then
      throw <| IO.userError "Qwen3.8 inference stream received too many positions"
    if token >= vocabularySizeU32 || (!greedyOnly && target >= vocabularySizeU32) then
      throw <| IO.userError "Qwen3.8 token and target IDs must be below 248320"
    enqueuePosition handle layout current.submitted token target greedyOnly
    state.set { current with submitted := current.submitted + 1 }
  return {
    expectedPositions := persistentLayout.tokenCount
    enqueue := enqueue false
    enqueueGreedy := fun token => enqueue true token 0
    nextToken := do
      let current ← state.get
      if current.finalized || current.aborted then
        throw <| IO.userError "cannot receive from a closed Qwen3.8 inference stream"
      unless current.received < current.submitted do
        throw <| IO.userError "Qwen3.8 inference stream has no submitted token to receive"
      let event ← awaitTokenEvent outputQueue handle
      unless event.position.toNat == current.received do
        throw <| IO.userError <|
          s!"Qwen3.8 token stream expected position {current.received}, got {event.position}"
      state.set { current with received := current.received + 1 }
      return event
    finish := finalize
    finishEarly := close false
    collect := do
      finalize
      return {
        logits := ← buffers.logits.copyTo stream
        loss := ← buffers.loss.copyTo stream
        sampled := ← buffers.sampled.copyTo stream
      }
    abort
  }

private def runOnce (layout : TextLayout) (weights : Cuda.BufferTable UInt8)
    (lmHead : QuantizedLmHead)
    (buffers : ModelBuffers)
    (tokens targets : Array UInt32) (stream : Cuda.Stream) : IO InferenceResult := do
  unless !tokens.isEmpty && tokens.size == targets.size && tokens.size <= maxSequenceTokens do
    throw <| IO.userError
      s!"Qwen3.8 token/target arrays must have equal positive length at most {maxSequenceTokens}"
  let persistentLayout ← match PersistentLayout.forTokens tokens.size with
    | .ok persistentLayout => pure persistentLayout
    | .error error => throw <| IO.userError error
  unless buffers.layout == persistentLayout do
    throw <| IO.userError "Qwen3.8 persistent buffers were allocated for a different sequence layout"
  let inference ← startLoadedStream layout weights lmHead buffers stream
  try
    for position in [:tokens.size] do
      inference.enqueue tokens[position]! targets[position]!
    for _ in [:tokens.size] do
      discard <| inference.nextToken
    inference.collect
  catch error =>
    inference.abort
    throw error

private def runTrainingForward (layout : TextLayout)
    (weights : Cuda.BufferTable UInt8) (lmHead : QuantizedLmHead)
    (tokens targets : Array UInt32) (stream : Cuda.Stream) : IO TrainingBatch := do
  unless !tokens.isEmpty && tokens.size == targets.size && tokens.size <= maxSequenceTokens do
    throw <| IO.userError
      s!"Qwen3.8 token/target arrays must have equal positive length at most {maxSequenceTokens}"
  let persistentLayout ← match PersistentLayout.forTokens tokens.size with
    | .ok persistentLayout => pure persistentLayout
    | .error error => throw <| IO.userError error
  let buffers ← allocateBuffers persistentLayout
  let inference ← startLoadedStream layout weights lmHead buffers stream
  try
    for position in [:tokens.size] do
      inference.enqueue tokens[position]! targets[position]!
    for _ in [:tokens.size] do
      discard <| inference.nextToken
    inference.finish
    return {
      rows := tokens.size.toUInt32
      inputFeatures := hiddenSizeU32
      outputFeatures := vocabularySizeU32
      finalHidden := buffers.finalHidden
      logits := buffers.logits
    }
  catch error =>
    inference.abort
    throw error

/--
Collect independent causal sequences into one row-major training tensor. This correctness bridge
shares one resolved weight table but currently runs the frozen-base sequence streams serially; a native
batched base kernel can replace it without changing the mixed-adapter training interface.
-/
private def runTrainingBatchForward (layout : TextLayout)
    (weights : Cuda.BufferTable UInt8) (lmHead : QuantizedLmHead)
    (tokens targets : Array (Array UInt32)) (stream : Cuda.Stream) : IO TrainingBatch := do
  unless !tokens.isEmpty && tokens.size == targets.size do
    throw <| IO.userError
      "Qwen3.8 training batch must contain equally many token and target sequences"
  let sequenceLength := tokens[0]!.size
  unless sequenceLength > 0 && sequenceLength ≤ maxSequenceTokens do
    throw <| IO.userError <|
      s!"Qwen3.8 training sequence length must lie in [1, {maxSequenceTokens}]"
  for batch in [:tokens.size] do
    unless tokens[batch]!.size == sequenceLength && targets[batch]!.size == sequenceLength do
      throw <| IO.userError
        "Qwen3.8 training batch requires a fixed positive sequence length"
  let rows := tokens.size * sequenceLength
  let hiddenElements := rows * hiddenSize
  let logitElements := rows * vocabularySize
  unless hiddenElements ≤ 0xffffffff && logitElements ≤ 0xffffffff do
    throw <| IO.userError "Qwen3.8 training batch exceeds UInt32 tensor indexing"
  let finalHidden ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenElements.toUSize
  let logits ← Cuda.Buffer.alloc Float32 logitElements.toUSize
  for batch in [:tokens.size] do
    let result ← runTrainingForward layout weights lmHead tokens[batch]! targets[batch]! stream
    let hiddenCount := sequenceLength * hiddenSize
    let logitCount := sequenceLength * vocabularySize
    (← copyTrainingBF16Kernel.launchOn stream
      (Primitives.elementConfig hiddenCount.toUInt32) result.finalHidden finalHidden
      (batch * hiddenCount).toUInt32 hiddenCount.toUInt32).waitChecked
      "Qwen3.8 batched final-hidden copy"
    (← copyTrainingF32Kernel.launchOn stream
      (Primitives.elementConfig logitCount.toUInt32) result.logits logits
      (batch * logitCount).toUInt32 logitCount.toUInt32).waitChecked
      "Qwen3.8 batched logit copy"
  return {
    rows := rows.toUInt32
    inputFeatures := hiddenSizeU32
    outputFeatures := vocabularySizeU32
    finalHidden
    logits
  }

/-- A loaded checkpoint that can start multiple sequential streaming sessions without reloading. -/
structure Checkpoint where
  start : Nat → IO InferenceStream

/-- A shared real checkpoint with both inference and device-resident frozen-base training views. -/
structure TrainableCheckpoint where
  checkpoint : Checkpoint
  stream : Cuda.Stream
  inputFeatures : UInt32
  outputFeatures : UInt32
  forward : Array UInt32 → Array UInt32 → IO TrainingBatch
  forwardBatch : Array (Array UInt32) → Array (Array UInt32) → IO TrainingBatch

private structure LoadedCheckpoint where
  layout : TextLayout
  weights : Cuda.BufferTable UInt8
  lmHead : QuantizedLmHead
  stream : Cuda.Stream

private def loadCheckpointData (modelDirectory : System.FilePath) : IO LoadedCheckpoint := do
  let schema ← introspect modelDirectory
  unless schema.isDirectory do
    throw <| IO.userError "Qwen3.8-27B must be loaded from its sharded checkpoint directory"
  let manifest ← schema.payloadManifest
  let layout ← readTextLayout manifest
  let stream ← Cuda.Stream.default
  let loadedWeights ← loadWeights manifest layout stream
  let weights ← resolveWeightTable loadedWeights layout
  let lmHead ← quantizeLmHead loadedWeights layout stream
  return { layout, weights, lmHead, stream }

private def checkpointOfLoaded (loaded : LoadedCheckpoint) : Checkpoint := {
  start := fun positions => do
    let persistentLayout ← match PersistentLayout.forTokens positions with
      | .ok layout => pure layout
      | .error error => throw <| IO.userError error
    let buffers ← allocateBuffers persistentLayout
    startLoadedStream loaded.layout loaded.weights loaded.lmHead buffers loaded.stream
}

/-- Load and retain all published checkpoint shards for subsequent streaming sessions. -/
def loadCheckpoint (modelDirectory : System.FilePath) : IO Checkpoint := do
  return checkpointOfLoaded (← loadCheckpointData modelDirectory)

/-- Load one immutable shard set and expose both inference and retained training activations. -/
def loadTrainableCheckpoint (modelDirectory : System.FilePath) : IO TrainableCheckpoint := do
  let loaded ← loadCheckpointData modelDirectory
  return {
    checkpoint := checkpointOfLoaded loaded
    stream := loaded.stream
    inputFeatures := hiddenSizeU32
    outputFeatures := vocabularySizeU32
    forward := fun tokens targets =>
      runTrainingForward loaded.layout loaded.weights loaded.lmHead tokens targets loaded.stream
    forwardBatch := fun tokens targets =>
      runTrainingBatchForward loaded.layout loaded.weights loaded.lmHead tokens targets loaded.stream
  }

/-- Run explicit input/target positions while invoking `onToken` as each device token is ready. -/
def Checkpoint.runStreaming (checkpoint : @& Checkpoint) (tokens targets : Array UInt32)
    (onToken : TokenEvent → IO Unit) : IO InferenceResult := do
  unless !tokens.isEmpty && tokens.size == targets.size && tokens.size <= maxSequenceTokens do
    throw <| IO.userError
      s!"Qwen3.8 token/target arrays must have equal positive length at most {maxSequenceTokens}"
  unless tokens.all (· < vocabularySizeU32) && targets.all (· < vocabularySizeU32) do
    throw <| IO.userError "Qwen3.8 token and target IDs must be below 248320"
  let inference ← checkpoint.start tokens.size
  try
    for position in [:tokens.size] do
      inference.enqueue tokens[position]! targets[position]!
    for _ in [:tokens.size] do
      onToken (← inference.nextToken)
    inference.collect
  catch error =>
    inference.abort
    throw error

/-- Controls bounded greedy generation and clean early completion at any configured token. -/
structure GenerationConfig where
  maxNewTokens : Nat
  stopTokens : Array UInt32 := #[]
  deriving Repr, BEq

/-- Whether a generated token cleanly terminates this request. -/
def GenerationConfig.shouldStop (config : @& GenerationConfig) (token : UInt32) : Bool :=
  config.stopTokens.contains token

private partial def generateRemaining (inference : @& InferenceStream)
    (stopTokens : @& Array UInt32) (remaining : Nat) (latest : TokenEvent)
    (generated : Array UInt32) (onToken : @& TokenEvent → IO Unit) : IO (Array UInt32) := do
  if remaining == 0 then
    inference.finish
    return generated
  inference.enqueueGreedy latest.token
  let latest ← inference.nextToken
  let generated := generated.push latest.token
  onToken latest
  if stopTokens.contains latest.token then
    inference.finishEarly
    return generated
  generateRemaining inference stopTokens (remaining - 1) latest generated onToken

/--
Autoregressively feed each sampled token back into the same live persistent launch and publish
generated tokens through `onToken`. The prompt is consumed causally; only generated tokens are
reported to the callback and returned. A configured stop token finishes the partially filled
launch normally after publishing that token.
-/
def Checkpoint.generateStreamingWith (checkpoint : @& Checkpoint) (prompt : Array UInt32)
    (config : GenerationConfig) (onToken : TokenEvent → IO Unit) : IO (Array UInt32) := do
  if prompt.isEmpty then
    throw <| IO.userError "Qwen3.8 streaming generation requires a nonempty prompt"
  unless prompt.all (· < vocabularySizeU32) do
    throw <| IO.userError "Qwen3.8 prompt token IDs must be below 248320"
  unless config.stopTokens.all (· < vocabularySizeU32) do
    throw <| IO.userError "Qwen3.8 stop token IDs must be below 248320"
  if config.maxNewTokens == 0 then
    return #[]
  let positions := prompt.size + config.maxNewTokens - 1
  if positions > maxSequenceTokens then
    throw <| IO.userError
      s!"Qwen3.8 prompt plus generation requires {positions} positions, maximum is {maxSequenceTokens}"
  let inference ← checkpoint.start positions
  try
    let mut latest : TokenEvent := default
    for position in [:prompt.size] do
      inference.enqueueGreedy prompt[position]!
      latest := ← inference.nextToken
    let mut generated := #[latest.token]
    onToken latest
    if config.shouldStop latest.token then
      inference.finishEarly
      return generated
    generateRemaining inference config.stopTokens (config.maxNewTokens - 1) latest generated onToken
  catch error =>
    inference.abort
    throw error

/-- Generate a fixed maximum number of tokens without an early-stop token. -/
def Checkpoint.generateStreaming (checkpoint : @& Checkpoint) (prompt : Array UInt32)
    (maxNewTokens : Nat) (onToken : TokenEvent → IO Unit) : IO (Array UInt32) :=
  checkpoint.generateStreamingWith prompt { maxNewTokens } onToken

/-- Load a checkpoint and stream explicit-position greedy tokens to `onToken`. -/
def runCheckpointStreaming (modelDirectory : System.FilePath) (tokens targets : Array UInt32)
    (onToken : TokenEvent → IO Unit) : IO (ByteArray × ByteArray × ByteArray) := do
  let checkpoint ← loadCheckpoint modelDirectory
  let result ← checkpoint.runStreaming tokens targets onToken
  return (result.logits, result.loss, result.sampled)

/-- Load the published checkpoint once and return position-major Float32 logits plus loss/sample
buffers for an explicit token-ID sequence. -/
def runCheckpoint (modelDirectory : System.FilePath) (tokens targets : Array UInt32) :
    IO (ByteArray × ByteArray × ByteArray) := do
  runCheckpointStreaming modelDirectory tokens targets fun _ => pure ()

private def parseIds (name raw : String) : IO (Array UInt32) := do
  let mut values := #[]
  for part in raw.splitOn "," do
    let some value := part.toNat?
      | throw <| IO.userError s!"{name} must be a comma-separated list of token IDs"
    if value >= vocabularySize then
      throw <| IO.userError s!"{name} token ID {value} is outside the Qwen3.8 vocabulary"
    values := values.push value.toUInt32
  if values.isEmpty then
    throw <| IO.userError s!"{name} must contain at least one token ID"
  return values

/-- Assert the checked persistent queue/state layout without relying on console output. -/
def checkPersistentLayout : IO Unit := do
  match PersistentLayout.forTokens 0 with
  | .ok _ => throw <| IO.userError "Qwen3.8 persistent layout accepted an empty sequence"
  | .error _ => pure ()
  match PersistentLayout.forTokens (maxSequenceTokens + 1) with
  | .ok _ => throw <| IO.userError "Qwen3.8 persistent layout accepted an over-context sequence"
  | .error _ => pure ()
  let layout ← match PersistentLayout.forTokens 2 with
    | .ok layout => pure layout
    | .error error => throw <| IO.userError error
  unless layout.descriptorCount == 2 * descriptorsPerTokenNat &&
      layout.logitsElements == 2 * vocabularySize &&
      layout.keyValueCacheElements == fullAttentionLayerCount * maxSequenceTokens *
        fullKeyValueProjectionSize do
    throw <| IO.userError "Qwen3.8 persistent layout derived inconsistent state extents"

def main : IO Unit := do
  checkPersistentLayout
  let some modelDirectory ← IO.getEnv "QWEN_MODEL_DIR" | do
    IO.println "Skipping real Qwen3.8 persistent inference: QWEN_MODEL_DIR is not set"
    return
  let tokens ← parseIds "QWEN36_TOKEN_IDS"
    ((← IO.getEnv "QWEN36_TOKEN_IDS").getD "1234,5678")
  let targets ← parseIds "QWEN36_TARGET_IDS"
    ((← IO.getEnv "QWEN36_TARGET_IDS").getD "1234,5678")
  unless tokens.size == targets.size do
    throw <| IO.userError "QWEN36_TOKEN_IDS and QWEN36_TARGET_IDS must have equal length"
  unless tokens.size <= maxSequenceTokens do
    throw <| IO.userError s!"Qwen3.8 sequence exceeds persistent context {maxSequenceTokens}"
  let schema ← introspect modelDirectory
  unless schema.isDirectory do
    throw <| IO.userError "Qwen3.8-27B must be loaded from its sharded checkpoint directory"
  let manifest ← schema.payloadManifest
  let layout ← readTextLayout manifest
  let stream ← Cuda.Stream.default
  let loadedWeights ← loadWeights manifest layout stream
  let weights ← resolveWeightTable loadedWeights layout
  let lmHead ← quantizeLmHead loadedWeights layout stream
  let persistentLayout ← match PersistentLayout.forTokens tokens.size with
    | .ok layout => pure layout
    | .error error => throw <| IO.userError error
  let buffers ← allocateBuffers persistentLayout
  let start ← IO.monoMsNow
  let first ← runOnce layout weights lmHead buffers tokens targets stream
  let elapsed := (← IO.monoMsNow) - start
  if (← IO.getEnv "QWEN36_REPEAT_DETERMINISM") == some "1" then
    let second ← runOnce layout weights lmHead buffers tokens targets stream
    unless first == second do
      throw <| IO.userError "repeated Qwen3.8 multi-token megakernel runs differ"
  if let some outputPath ← IO.getEnv "QWEN36_LOGITS_OUTPUT" then
    IO.FS.writeBinFile outputPath first.logits
  for position in [:tokens.size] do
    IO.println s!"Qwen3.8 position={position}, token={tokens[position]!}, sampled={readUInt32 first.sampled position}, loss_bits={readUInt32 first.loss position}"
  IO.println s!"Lean Qwen3.8 real-checkpoint {tokens.size}-token persistent inference ok; elapsed_ms={elapsed}"

end Cuda.Qwen36.Megakernel
