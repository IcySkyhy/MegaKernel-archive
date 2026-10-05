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
import Init.Data.String.Legacy
import Init.Data.String.Search

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
cached positions. Training retains position-major outputs; autoregressive inference recycles one
output row while retaining only its request-sized KV cache.
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

/-- The causal context published by Qwen3.8-27B. -/
def maxSequenceTokens : Nat := 262144

/-- Retained training outputs remain deliberately bounded independently of inference context. -/
def maxTrainingSequenceTokens : Nat := 256

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
private def prefillTileRowsNat : Nat := 2

/-- Compile-time dense-projection branch width. Every branch consists of one or more naturally
aligned four-BF16 transactions; keeping the alternatives here makes performance candidates
reproducible without rewriting the projection loops. -/
private inductive ProjectionLoadWidth where
  | bf16x4
  | bf16x8
  | bf16x16
  deriving Repr, BEq

private def projectionLoadWidth : ProjectionLoadWidth := .bf16x16

/-- Whether one thread consumes adjacent transactions or unrolls across block-stride,
fully-coalesced transactions. -/
private inductive ProjectionAccessOrder where
  | adjacent
  | blockStrided
  deriving Repr, BEq

private def projectionAccessOrder : ProjectionAccessOrder := .blockStrided

/-- Recurrent-state decode schedule. The phased form retains the original row-major state and
materializes every algebraic phase in global memory. The resident form transposes the private
state layout so one warp owns a contiguous column and keeps its four rows live across decay,
memory lookup, rank-one update, and query projection. -/
private inductive RecurrentStateSchedule where
  | phasedRowMajor
  | warpResidentColumnMajor
  deriving Repr, BEq

private def recurrentStateSchedule : RecurrentStateSchedule := .warpResidentColumnMajor

/-- Number of persistent CTAs assigned to each recurrent value head. The single-CTA form is the
direct state-residency baseline; the full-grid form uses every already-resident CTA and coordinates
one normalization owner per head through device-scope release/acquire arrivals. -/
private inductive RecurrentGridSchedule where
  | oneBlockPerHead
  | fullResidentGrid
  deriving Repr, BEq

private def recurrentGridSchedule : RecurrentGridSchedule := .fullResidentGrid
private def recurrentGroupCountNat : Nat := gridBlockCountNat / linearHeadCount
private def recurrentWarpsPerBlockNat : Nat := blockThreadCountNat / 32

/-- Hidden-state RMSNorm reduction schedule. The legacy path spreads 5,120 inputs over the full
grid, then has every persistent thread repeat the 288-partial final reduction. The owner path
lets one CTA consume the small vector exactly once and publishes one inverse scale at the existing
grid barrier. -/
private inductive RmsNormReductionSchedule where
  | repeatedGridPartials
  | singleBlockOwner
  deriving Repr, BEq

private def rmsNormReductionSchedule : RmsNormReductionSchedule := .singleBlockOwner

/-- Final vocabulary-maximum reduction schedule. The legacy path has every persistent thread
rescan all 288 block partials. The owner path reduces those partials once in CTA zero, then
publishes the scalar at a grid barrier. -/
private inductive FinalMaximumReductionSchedule where
  | repeatedGridPartials
  | singleBlockOwner
  deriving Repr, BEq

private def finalMaximumReductionSchedule : FinalMaximumReductionSchedule := .singleBlockOwner

/-- Greedy sampling schedule. The legacy path rescans every logit with the full grid. The
block-owned path uses one thread per CTA to nominate the best row that CTA just projected, refines
those 288 candidates against the original BF16 head, then reduces the exact indexed maxima. -/
private inductive GreedySamplingSchedule where
  | globalLogitRescan
  | blockOwnedRows
  deriving Repr, BEq

private def greedySamplingSchedule : GreedySamplingSchedule := .blockOwnedRows

private def projectionQuadsPerStepNat : Nat :=
  match projectionLoadWidth with
  | .bf16x4 => 1
  | .bf16x8 => 2
  | .bf16x16 => 4

private def projectionValuesPerBranchNat : Nat := 4 * projectionQuadsPerStepNat
private def projectionValuesPerTransactionNat : Nat := 4
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
  let projectionOrder :=
    match projectionAccessOrder with
    | .adjacent => "adjacent"
    | .blockStrided => s!"strided{blockThreadCountNat}"
  let recurrentRoute :=
    match recurrentStateSchedule with
    | .phasedRowMajor => "statephased_rowmajor"
    | .warpResidentColumnMajor => "statewarp_resident_colmajor"
  let recurrentGridRoute :=
    match recurrentGridSchedule with
    | .oneBlockPerHead => "stategrid1"
    | .fullResidentGrid =>
      s!"stategrid{recurrentGroupCountNat}x{recurrentWarpsPerBlockNat}warps"
  let rmsNormRoute :=
    match rmsNormReductionSchedule with
    | .repeatedGridPartials => "rms_gridpartials_repeated"
    | .singleBlockOwner => "rms_block1_owner"
  let finalMaximumRoute :=
    match finalMaximumReductionSchedule with
    | .repeatedGridPartials => "max_gridpartials_repeated"
    | .singleBlockOwner => "max_block1_owner"
  let greedySamplingRoute :=
    match greedySamplingSchedule with
    | .globalLogitRescan => "sample_global_rescan"
    | .blockOwnedRows => "sample_blockrows_bf16refine"
  s!"persistent_megakernel_grid{gridBlockCountNat}_block{blockThreadCountNat}_" ++
    s!"bf16x{projectionValuesPerBranchNat}_{projectionOrder}_" ++
    s!"{recurrentRoute}_{recurrentGridRoute}_attn_online_softmax_" ++
    s!"{rmsNormRoute}_{finalMaximumRoute}_{greedySamplingRoute}_" ++
    s!"greedy_lmhead_{lmHeadRoute}_paired_swiglu_rowpair2_fused_residual_" ++
    s!"integrated_prefill_prefix_cache_tile{prefillTileRowsNat}_resolved_weight_ptrs_{weightRoute}"
private def descriptorsPerTokenNat : Nat := layerCount + 2
private def linearStateElementsPerLayerNat : Nat :=
  linearHeadCount * linearHeadDimension * linearHeadDimension
private def convolutionElementsPerLayerNat : Nat := linearQkvSize * 4

private def hiddenSizeU32 : UInt32 := hiddenSize.toUInt32
private def intermediateSizeU32 : UInt32 := intermediateSize.toUInt32
private def vocabularySizeU32 : UInt32 := vocabularySize.toUInt32
private def gridBlockCount : UInt32 := gridBlockCountNat.toUInt32
private def blockThreadCount : UInt32 := blockThreadCountNat.toUInt32
private def recurrentGroupCount : UInt32 := recurrentGroupCountNat.toUInt32
private def recurrentWarpsPerBlock : UInt32 := recurrentWarpsPerBlockNat.toUInt32
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
private def maxTrainingSequenceTokensU32 : UInt32 := maxTrainingSequenceTokens.toUInt32

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
private def sharedBytes : Nat := fullScoresOffset

private def kindEmbedding : UInt32 := 0
private def kindLinear : UInt32 := 1
private def kindFull : UInt32 := 2
private def kindFinal : UInt32 := 3
private def kindGreedyFinal : UInt32 := 4
private def kindReferenceFinal : UInt32 := 5
private def kindPrefill : UInt32 := 6
private def kindTrainingBatch2 : UInt32 := 7
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
  count : UInt32
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
  contextCapacity : UInt32
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
  promptTokens : Cuda.DevicePtr UInt32
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
private def prefillRowBuffers (buffers : Buffers) (row : UInt32) : Buffers := {
  buffers with
  hidden := bfloatOffset buffers.hidden (row * hiddenSizeU32)
  normalized := bfloatOffset buffers.normalized (row * hiddenSizeU32)
  workspace0 := bfloatOffset buffers.workspace0 (row * intermediateSizeU32)
  workspace1 := bfloatOffset buffers.workspace1 (row * intermediateSizeU32)
  workspace2 := bfloatOffset buffers.workspace2 (row * intermediateSizeU32)
  attentionOutput := bfloatOffset buffers.attentionOutput (row * linearValueSizeU32)
}
@[always_inline]
private def trainingBatchRowBuffers (buffers : Buffers) (row : UInt32) : Buffers :=
  let rowBuffers := prefillRowBuffers buffers row
  {
    rowBuffers with
    convolutionState := bfloatOffset buffers.convolutionState
      (row * (linearLayerCount * convolutionElementsPerLayerNat).toUInt32)
    recurrentState := floatOffset buffers.recurrentState
      (row * (linearLayerCount * linearStateElementsPerLayerNat).toUInt32)
    keyCache := bfloatOffset buffers.keyCache
      (row * fullAttentionLayerCount.toUInt32 * buffers.contextCapacity *
        fullKeyValueProjectionSize.toUInt32)
    valueCache := bfloatOffset buffers.valueCache
      (row * fullAttentionLayerCount.toUInt32 * buffers.contextCapacity *
        fullKeyValueProjectionSize.toUInt32)
  }

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
private partial def maximumPartialsStrided (partials : Cuda.DevicePtr Float32)
    (index stride : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < gridBlockCount then
    maximumPartialsStrided partials (index + stride) stride
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
  match rmsNormReductionSchedule with
  | .repeatedGridPartials =>
    let localSquares ← squareSumLoop source hiddenSizeU32 globalThread globalThreads zero
    let blockSquares ← Cuda.Collective.blockSumLeader scratch localSquares
    if thread == 0 then
      Cuda.storeFloat32 buffers.partialSum block.toUSize blockSquares
    Cuda.gridSync
    let squareSum ← sumPartials buffers.partialSum 0 zero
    let inverse ← Cuda.fastRsqrt (squareSum * inverseHiddenSize + rmsEpsilon)
    normalizeLoop source weight output hiddenSizeU32 globalThread globalThreads inverse
    Cuda.gridSync
  | .singleBlockOwner =>
    if block == 0 then
      let localSquares ← squareSumLoop source hiddenSizeU32 thread blockThreadCount zero
      let squareSum ← Cuda.Collective.blockSumLeader scratch localSquares
      if thread == 0 then
        let inverse ← Cuda.fastRsqrt (squareSum * inverseHiddenSize + rmsEpsilon)
        Cuda.storeFloat32 buffers.partialSum 0 inverse
    Cuda.gridSync
    let inverse ← Cuda.loadFloat32 buffers.partialSum 0
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
private def dotProjectionAdjacentGroup (input weight : Cuda.DevicePtr Cuda.BFloat16)
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
private def dotProjectionAdjacentPairGroup
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
private def dotProjectionStridedGroup (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  let weightQuad := row * columnQuads + quad
  let accumulator ← dotProjectionQuad input weight quad weightQuad accumulator
  match projectionLoadWidth with
  | .bf16x4 => return accumulator
  | .bf16x8 =>
    dotProjectionQuad input weight (quad + blockThreadCount)
      (weightQuad + blockThreadCount) accumulator
  | .bf16x16 => do
    let accumulator ← dotProjectionQuad input weight (quad + blockThreadCount)
      (weightQuad + blockThreadCount) accumulator
    let accumulator ← dotProjectionQuad input weight (quad + 2 * blockThreadCount)
      (weightQuad + 2 * blockThreadCount) accumulator
    dotProjectionQuad input weight (quad + 3 * blockThreadCount)
      (weightQuad + 3 * blockThreadCount) accumulator

@[always_inline]
private def dotProjectionStridedPairGroup
    (input firstWeight secondWeight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let weightQuad := row * columnQuads + quad
  let accumulators ← dotProjectionPairQuad input firstWeight secondWeight quad weightQuad
    firstAccumulator secondAccumulator
  match projectionLoadWidth with
  | .bf16x4 => return accumulators
  | .bf16x8 =>
    dotProjectionPairQuad input firstWeight secondWeight (quad + blockThreadCount)
      (weightQuad + blockThreadCount) accumulators.first accumulators.second
  | .bf16x16 => do
    let accumulators ← dotProjectionPairQuad input firstWeight secondWeight
      (quad + blockThreadCount) (weightQuad + blockThreadCount)
      accumulators.first accumulators.second
    let accumulators ← dotProjectionPairQuad input firstWeight secondWeight
      (quad + 2 * blockThreadCount) (weightQuad + 2 * blockThreadCount)
      accumulators.first accumulators.second
    dotProjectionPairQuad input firstWeight secondWeight
      (quad + 3 * blockThreadCount) (weightQuad + 3 * blockThreadCount)
      accumulators.first accumulators.second

@[always_inline]
private partial def dotProjectionRowAdjacent (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads group : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  let columnGroups := columnQuads / projectionQuadsPerStepNat.toUInt32
  if group < columnGroups then
    let accumulator ←
      dotProjectionAdjacentGroup input weight row columnGroups group accumulator
    dotProjectionRowAdjacent input weight row columnQuads
      (group + blockThreadCount) accumulator
  else
    return accumulator

@[always_inline]
private partial def dotProjectionRowPairAdjacent
    (input firstWeight secondWeight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads group : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let columnGroups := columnQuads / projectionQuadsPerStepNat.toUInt32
  if group < columnGroups then
    let accumulators ← dotProjectionAdjacentPairGroup input firstWeight secondWeight
      row columnGroups group firstAccumulator secondAccumulator
    dotProjectionRowPairAdjacent input firstWeight secondWeight row columnQuads
      (group + blockThreadCount) accumulators.first accumulators.second
  else
    return { first := firstAccumulator, second := secondAccumulator }

@[always_inline]
private partial def dotProjectionRowStrided (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if quad < columnQuads then
    let finalQuad := quad +
      (projectionQuadsPerStepNat - 1).toUInt32 * blockThreadCount
    if finalQuad < columnQuads then
      let accumulator ← dotProjectionStridedGroup input weight row columnQuads quad accumulator
      dotProjectionRowStrided input weight row columnQuads
        (quad + projectionQuadsPerStepNat.toUInt32 * blockThreadCount) accumulator
    else
      let weightQuad := row * columnQuads + quad
      let accumulator ← dotProjectionQuad input weight quad weightQuad accumulator
      dotProjectionRowStrided input weight row columnQuads
        (quad + blockThreadCount) accumulator
  else
    return accumulator

@[always_inline]
private partial def dotProjectionRowPairStrided
    (input firstWeight secondWeight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  if quad < columnQuads then
    let finalQuad := quad +
      (projectionQuadsPerStepNat - 1).toUInt32 * blockThreadCount
    if finalQuad < columnQuads then
      let accumulators ← dotProjectionStridedPairGroup input firstWeight secondWeight
        row columnQuads quad firstAccumulator secondAccumulator
      dotProjectionRowPairStrided input firstWeight secondWeight row columnQuads
        (quad + projectionQuadsPerStepNat.toUInt32 * blockThreadCount)
        accumulators.first accumulators.second
    else
      let weightQuad := row * columnQuads + quad
      let accumulators ← dotProjectionPairQuad input firstWeight secondWeight quad weightQuad
        firstAccumulator secondAccumulator
      dotProjectionRowPairStrided input firstWeight secondWeight row columnQuads
        (quad + blockThreadCount) accumulators.first accumulators.second
  else
    return { first := firstAccumulator, second := secondAccumulator }

@[always_inline]
private def dotProjectionRow (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 :=
  match projectionAccessOrder with
  | .adjacent => dotProjectionRowAdjacent input weight row columnQuads quad accumulator
  | .blockStrided => dotProjectionRowStrided input weight row columnQuads quad accumulator

@[always_inline]
private def dotProjectionRowPair
    (input firstWeight secondWeight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair :=
  match projectionAccessOrder with
  | .adjacent =>
    dotProjectionRowPairAdjacent input firstWeight secondWeight row columnQuads quad
      firstAccumulator secondAccumulator
  | .blockStrided =>
    dotProjectionRowPairStrided input firstWeight secondWeight row columnQuads quad
      firstAccumulator secondAccumulator

/-! A two-row prefill dot product keeps one weight transaction live while applying it to two
activation rows. Each row preserves the decode dot-product accumulation order exactly. -/

@[always_inline]
private def dotProjectionBatch2Quad
    (firstInput secondInput weight : Cuda.DevicePtr Cuda.BFloat16)
    (inputQuad weightQuad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let firstValues ← Cuda.loadBFloat16x4 firstInput inputQuad.toUSize
  let secondValues ← Cuda.loadBFloat16x4 secondInput inputQuad.toUSize
  let weightValues ← Cuda.loadReadOnlyBFloat16x4 weight weightQuad.toUSize
  let firstAccumulator :=
    Cuda.fma weightValues.x0.toFloat32 firstValues.x0.toFloat32 firstAccumulator
  let firstAccumulator :=
    Cuda.fma weightValues.x1.toFloat32 firstValues.x1.toFloat32 firstAccumulator
  let firstAccumulator :=
    Cuda.fma weightValues.x2.toFloat32 firstValues.x2.toFloat32 firstAccumulator
  let firstAccumulator :=
    Cuda.fma weightValues.x3.toFloat32 firstValues.x3.toFloat32 firstAccumulator
  let secondAccumulator :=
    Cuda.fma weightValues.x0.toFloat32 secondValues.x0.toFloat32 secondAccumulator
  let secondAccumulator :=
    Cuda.fma weightValues.x1.toFloat32 secondValues.x1.toFloat32 secondAccumulator
  let secondAccumulator :=
    Cuda.fma weightValues.x2.toFloat32 secondValues.x2.toFloat32 secondAccumulator
  let secondAccumulator :=
    Cuda.fma weightValues.x3.toFloat32 secondValues.x3.toFloat32 secondAccumulator
  return { first := firstAccumulator, second := secondAccumulator }

@[always_inline]
private def dotProjectionBatch2AdjacentGroup
    (firstInput secondInput weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnGroups group : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let quadsPerStep := projectionQuadsPerStepNat.toUInt32
  let inputQuad := group * quadsPerStep
  let weightQuad := row * columnGroups * quadsPerStep + inputQuad
  let accumulators ← dotProjectionBatch2Quad firstInput secondInput weight inputQuad weightQuad
    firstAccumulator secondAccumulator
  match projectionLoadWidth with
  | .bf16x4 => return accumulators
  | .bf16x8 =>
    dotProjectionBatch2Quad firstInput secondInput weight (inputQuad + 1) (weightQuad + 1)
      accumulators.first accumulators.second
  | .bf16x16 => do
    let accumulators ← dotProjectionBatch2Quad firstInput secondInput weight
      (inputQuad + 1) (weightQuad + 1) accumulators.first accumulators.second
    let accumulators ← dotProjectionBatch2Quad firstInput secondInput weight
      (inputQuad + 2) (weightQuad + 2) accumulators.first accumulators.second
    dotProjectionBatch2Quad firstInput secondInput weight
      (inputQuad + 3) (weightQuad + 3) accumulators.first accumulators.second

@[always_inline]
private def dotProjectionBatch2StridedGroup
    (firstInput secondInput weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let weightQuad := row * columnQuads + quad
  let accumulators ← dotProjectionBatch2Quad firstInput secondInput weight quad weightQuad
    firstAccumulator secondAccumulator
  match projectionLoadWidth with
  | .bf16x4 => return accumulators
  | .bf16x8 =>
    dotProjectionBatch2Quad firstInput secondInput weight
      (quad + blockThreadCount) (weightQuad + blockThreadCount)
      accumulators.first accumulators.second
  | .bf16x16 => do
    let accumulators ← dotProjectionBatch2Quad firstInput secondInput weight
      (quad + blockThreadCount) (weightQuad + blockThreadCount)
      accumulators.first accumulators.second
    let accumulators ← dotProjectionBatch2Quad firstInput secondInput weight
      (quad + 2 * blockThreadCount) (weightQuad + 2 * blockThreadCount)
      accumulators.first accumulators.second
    dotProjectionBatch2Quad firstInput secondInput weight
      (quad + 3 * blockThreadCount) (weightQuad + 3 * blockThreadCount)
      accumulators.first accumulators.second

@[always_inline]
private partial def dotProjectionBatch2RowAdjacent
    (firstInput secondInput weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads group : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  let columnGroups := columnQuads / projectionQuadsPerStepNat.toUInt32
  if group < columnGroups then
    let accumulators ← dotProjectionBatch2AdjacentGroup firstInput secondInput weight
      row columnGroups group firstAccumulator secondAccumulator
    dotProjectionBatch2RowAdjacent firstInput secondInput weight row columnQuads
      (group + blockThreadCount) accumulators.first accumulators.second
  else
    return { first := firstAccumulator, second := secondAccumulator }

@[always_inline]
private partial def dotProjectionBatch2RowStrided
    (firstInput secondInput weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair := do
  if quad < columnQuads then
    let finalQuad := quad +
      (projectionQuadsPerStepNat - 1).toUInt32 * blockThreadCount
    if finalQuad < columnQuads then
      let accumulators ← dotProjectionBatch2StridedGroup firstInput secondInput weight
        row columnQuads quad firstAccumulator secondAccumulator
      dotProjectionBatch2RowStrided firstInput secondInput weight row columnQuads
        (quad + projectionQuadsPerStepNat.toUInt32 * blockThreadCount)
        accumulators.first accumulators.second
    else
      let weightQuad := row * columnQuads + quad
      let accumulators ← dotProjectionBatch2Quad firstInput secondInput weight quad weightQuad
        firstAccumulator secondAccumulator
      dotProjectionBatch2RowStrided firstInput secondInput weight row columnQuads
        (quad + blockThreadCount) accumulators.first accumulators.second
  else
    return { first := firstAccumulator, second := secondAccumulator }

@[always_inline]
private def dotProjectionBatch2Row
    (firstInput secondInput weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columnQuads quad : UInt32) (firstAccumulator secondAccumulator : Float32) :
    Cuda.DeviceM ProjectionPair :=
  match projectionAccessOrder with
  | .adjacent =>
    dotProjectionBatch2RowAdjacent firstInput secondInput weight row columnQuads quad
      firstAccumulator secondAccumulator
  | .blockStrided =>
    dotProjectionBatch2RowStrided firstInput secondInput weight row columnQuads quad
      firstAccumulator secondAccumulator

@[always_inline, convergent]
private partial def projectRowsBatch2BFloat
    (firstInput secondInput weight firstOutput secondOutput :
      Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSums ← dotProjectionBatch2Row firstInput secondInput weight row
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      storeBFloat firstOutput row totals.first
      storeBFloat secondOutput row totals.second
    projectRowsBatch2BFloat firstInput secondInput weight firstOutput secondOutput
      rows columns (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowsBatch2Float
    (firstInput secondInput weight : Cuda.DevicePtr Cuda.BFloat16)
    (firstOutput secondOutput : Cuda.DevicePtr Float32)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSums ← dotProjectionBatch2Row firstInput secondInput weight row
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      Cuda.storeFloat32 firstOutput row.toUSize totals.first
      Cuda.storeFloat32 secondOutput row.toUSize totals.second
    projectRowsBatch2Float firstInput secondInput weight firstOutput secondOutput
      rows columns (row + rowStride) rowStride thread scratch
@[always_inline, convergent]
private partial def projectRowsBatch2ResidualBFloat
    (firstInput secondInput weight firstResidual secondResidual :
      Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSums ← dotProjectionBatch2Row firstInput secondInput weight row
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let totals ← Cuda.Collective.blockSumPairLeader pairScratch
      localSums.first localSums.second
    if thread == 0 then
      storeResidualBFloat firstResidual row totals.first
      storeResidualBFloat secondResidual row totals.second
    projectRowsBatch2ResidualBFloat firstInput secondInput weight firstResidual secondResidual
      rows columns (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowsBatch2SwiGLU
    (firstInput secondInput gateWeight upWeight firstOutput secondOutput :
      Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localGates ← dotProjectionBatch2Row firstInput secondInput gateWeight row
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
    let pairScratch : Cuda.Collective.BlockPairScratch 8 := scratch
    let gates ← Cuda.Collective.blockSumPairLeader pairScratch
      localGates.first localGates.second
    let localUps ← dotProjectionBatch2Row firstInput secondInput upWeight row
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
    let ups ← Cuda.Collective.blockSumPairLeader pairScratch localUps.first localUps.second
    if thread == 0 then
      let firstGate ← deviceSilu (rounded gates.first)
      let secondGate ← deviceSilu (rounded gates.second)
      storeBFloat firstOutput row (rounded firstGate * rounded ups.first)
      storeBFloat secondOutput row (rounded secondGate * rounded ups.second)
    projectRowsBatch2SwiGLU firstInput secondInput gateWeight upWeight firstOutput secondOutput
      rows columns (row + rowStride) rowStride thread scratch

@[always_inline, convergent]
private partial def projectRowsSwiGLU
    (input gateWeight upWeight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns row rowStride thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSums ← dotProjectionRowPair input gateWeight upWeight row
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
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
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero
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
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
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
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
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
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero
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
      (columns / projectionValuesPerTransactionNat.toUInt32) thread zero zero
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

@[always_inline, convergent]
private def runProjectionBatch2
    (firstInput secondInput weight firstOutput secondOutput :
      Cuda.DevicePtr Cuda.BFloat16)
    (rows columns block thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8) :
    Cuda.DeviceM Unit := do
  projectRowsBatch2BFloat firstInput secondInput weight firstOutput secondOutput
    rows columns block gridBlockCount thread scratch
  Cuda.gridSync

@[always_inline, convergent]
private def runResidualProjectionBatch2
    (firstInput secondInput weight firstResidual secondResidual :
      Cuda.DevicePtr Cuda.BFloat16)
    (rows columns block thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8) :
    Cuda.DeviceM Unit := do
  projectRowsBatch2ResidualBFloat firstInput secondInput weight firstResidual secondResidual
    rows columns block gridBlockCount thread scratch
  Cuda.gridSync

@[always_inline, convergent]
private def runSwiGLUProjectionBatch2
    (firstInput secondInput gateWeight upWeight firstOutput secondOutput :
      Cuda.DevicePtr Cuda.BFloat16)
    (rows columns block thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8) :
    Cuda.DeviceM Unit := do
  projectRowsBatch2SwiGLU firstInput secondInput gateWeight upWeight firstOutput secondOutput
    rows columns block gridBlockCount thread scratch
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
private def linearStateRowMajorIndex (head row column : UInt32) : UInt32 :=
  (head * linearDimension + row) * linearDimension + column

@[always_inline]
private def linearStateColumnMajorIndex (head column row : UInt32) : UInt32 :=
  (head * linearDimension + column) * linearDimension + row

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
    let stateValue ← Cuda.loadFloat32 state (linearStateRowMajorIndex head row column).toUSize
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
    let address := linearStateRowMajorIndex head row column
    let previous ← Cuda.loadFloat32 state address.toUSize
    let keyValue ← Cuda.loadFloat32 keyShared row.toUSize
    let delta ← Cuda.loadFloat32 deltaShared column.toUSize
    Cuda.storeFloat32 state address.toUSize (Cuda.fma keyValue delta previous)
    updateState state keyShared deltaShared head (index + blockThreadCount)

@[always_inline]
private partial def stateQueryDot (state queryShared : Cuda.DevicePtr Float32)
    (head column row : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < linearDimension then
    let stateValue ← Cuda.loadFloat32 state (linearStateRowMajorIndex head row column).toUSize
    let queryValue ← Cuda.loadFloat32 queryShared row.toUSize
    stateQueryDot state queryShared head column (row + 1)
      (Cuda.fma stateValue queryValue accumulator)
  else
    return accumulator

/-- Fuse one decode recurrence around a warp-resident, column-major state tile. Each of the eight
warps owns one column at a time; its lanes hold rows `(lane, lane+32, lane+64, lane+96)`. This
reduces each state element from four global reads plus two writes to one read plus one write and
removes the three intermediate block barriers. The state is private to this persistent launch and
is zero-initialized, so the compile-time schedule also owns its physical layout. -/
@[always_inline, convergent]
private partial def recurrentWarpResidentColumns
    (state keyShared queryShared : Cuda.DevicePtr Float32)
    (convolved attentionOutput : Cuda.DevicePtr Cuda.BFloat16)
    (head warp lane columnBase columnStride : UInt32) (decay beta : Float32) :
    Cuda.DeviceM Unit := do
  if columnBase < linearDimension then
    let column := columnBase + warp
    let address0 := linearStateColumnMajorIndex head column lane
    let address1 := address0 + 32
    let address2 := address0 + 64
    let address3 := address0 + 96
    let state0 := (← Cuda.loadFloat32 state address0.toUSize) * decay
    let state1 := (← Cuda.loadFloat32 state address1.toUSize) * decay
    let state2 := (← Cuda.loadFloat32 state address2.toUSize) * decay
    let state3 := (← Cuda.loadFloat32 state address3.toUSize) * decay
    let memoryPartial := Cuda.fma state0 (← Cuda.loadFloat32 keyShared lane.toUSize) zero
    let memoryPartial := Cuda.fma state1
      (← Cuda.loadFloat32 keyShared (lane + 32).toUSize) memoryPartial
    let memoryPartial := Cuda.fma state2
      (← Cuda.loadFloat32 keyShared (lane + 64).toUSize) memoryPartial
    let memoryPartial := Cuda.fma state3
      (← Cuda.loadFloat32 keyShared (lane + 96).toUSize) memoryPartial
    let memory ← Cuda.Qwen36.Primitives.warpSumBroadcast memoryPartial
    let value ← loadBFloat convolved (linearKeyElements * 2 + head * linearDimension + column)
    let delta := (value - memory) * beta
    let state0 := Cuda.fma (← Cuda.loadFloat32 keyShared lane.toUSize) delta state0
    let state1 := Cuda.fma (← Cuda.loadFloat32 keyShared (lane + 32).toUSize) delta state1
    let state2 := Cuda.fma (← Cuda.loadFloat32 keyShared (lane + 64).toUSize) delta state2
    let state3 := Cuda.fma (← Cuda.loadFloat32 keyShared (lane + 96).toUSize) delta state3
    let resultPartial := Cuda.fma state0 (← Cuda.loadFloat32 queryShared lane.toUSize) zero
    let resultPartial := Cuda.fma state1
      (← Cuda.loadFloat32 queryShared (lane + 32).toUSize) resultPartial
    let resultPartial := Cuda.fma state2
      (← Cuda.loadFloat32 queryShared (lane + 64).toUSize) resultPartial
    let resultPartial := Cuda.fma state3
      (← Cuda.loadFloat32 queryShared (lane + 96).toUSize) resultPartial
    let result ← Cuda.Qwen36.Primitives.warpSumBroadcast resultPartial
    if lane == 0 then
      storeBFloat attentionOutput (head * linearDimension + column) result
    Cuda.storeFloat32 state address0.toUSize state0
    Cuda.storeFloat32 state address1.toUSize state1
    Cuda.storeFloat32 state address2.toUSize state2
    Cuda.storeFloat32 state address3.toUSize state3
    recurrentWarpResidentColumns state keyShared queryShared convolved attentionOutput
      head warp lane (columnBase + columnStride) columnStride decay beta

@[always_inline]
private partial def waitRecurrentArrivals (counter : Cuda.DevicePtr UInt32)
    (required : UInt32) : Cuda.DeviceM Unit := do
  let observed ← Cuda.atomicLoadRelaxedUInt32 counter
  if observed >= required then
    Cuda.fenceAcquireDevice
  else
    Cuda.nanosleep 16
    waitRecurrentArrivals counter required

@[always_inline, convergent]
private def finishLinearRecurrent (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers) (scratch : Cuda.Collective.BlockScratch 8)
    (head thread : UInt32) : Cuda.DeviceM Unit := do
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
private def runLinearRecurrent (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (head thread group : UInt32) : Cuda.DeviceM Unit := do
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
  match recurrentStateSchedule with
  | .phasedRowMajor =>
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
    finishLinearRecurrent item weights buffers scratch head thread
  | .warpResidentColumnMajor =>
    let warp ← Cuda.warpId
    let lane ← Cuda.laneId
    let firstColumn := group * recurrentWarpsPerBlock
    let columnStride := match recurrentGridSchedule with
      | .oneBlockPerHead => recurrentWarpsPerBlock
      | .fullResidentGrid => recurrentGroupCount * recurrentWarpsPerBlock
    recurrentWarpResidentColumns state keyShared queryShared convolved buffers.attentionOutput
      head warp lane firstColumn columnStride decay beta
    Cuda.blockSync
    match recurrentGridSchedule with
    | .oneBlockPerHead => finishLinearRecurrent item weights buffers scratch head thread
    | .fullResidentGrid =>
      let arrivals : Cuda.DevicePtr UInt32 := buffers.partialMaximum
      let counter := arrivals + head.toUSize * 4
      if thread == 0 then
        Cuda.atomicReduceAddReleaseUInt32 counter 1
      Cuda.blockSync
      if group == 0 then
        if thread == 0 then
          waitRecurrentArrivals counter recurrentGroupCount
        Cuda.blockSync
        finishLinearRecurrent item weights buffers scratch head thread
        Cuda.blockSync
        if thread == 0 then
          Cuda.atomicStoreReleaseUInt32 counter 0

@[always_inline, convergent]
private def runLinearStateStep (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread : UInt32) :
    Cuda.DeviceM Unit := do
  runLinearConvolution item (← weightPointer weights item weightAttention1) buffers block thread
  Cuda.gridSync
  match recurrentStateSchedule, recurrentGridSchedule with
  | .phasedRowMajor, _ =>
    if block < linearHeads then
      runLinearRecurrent item weights buffers shared block thread 0
  | .warpResidentColumnMajor, .oneBlockPerHead =>
    if block < linearHeads then
      runLinearRecurrent item weights buffers shared block thread 0
  | .warpResidentColumnMajor, .fullResidentGrid =>
    runLinearRecurrent item weights buffers shared (block % linearHeads) thread
      (block / linearHeads)
  Cuda.gridSync

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
  runLinearStateStep item weights buffers shared block thread
  runResidualProjection buffers.attentionOutput (← weightPointer weights item weightAttention8)
    buffers.hidden hiddenSizeU32 linearValueSizeU32 block thread scratch

@[always_inline, convergent]
private def runLinearLayerBatch2 (firstItem secondItem : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8) (firstBuffers secondBuffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights firstItem weightAttention5)
    firstBuffers.workspace0 secondBuffers.workspace0
    linearQkvSizeU32 hiddenSizeU32 block thread scratch
  runProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights firstItem weightAttention6)
    firstBuffers.workspace1 secondBuffers.workspace1
    linearValueSizeU32 hiddenSizeU32 block thread scratch
  runProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights firstItem weightAttention3)
    firstBuffers.workspace2 secondBuffers.workspace2
    linearHeads hiddenSizeU32 block thread scratch
  runProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights firstItem weightAttention4)
    (bfloatOffset firstBuffers.workspace2 linearHeads)
    (bfloatOffset secondBuffers.workspace2 linearHeads)
    linearHeads hiddenSizeU32 block thread scratch
  runLinearStateStep firstItem weights firstBuffers shared block thread
  runLinearStateStep secondItem weights secondBuffers shared block thread
  runResidualProjectionBatch2 firstBuffers.attentionOutput secondBuffers.attentionOutput
    (← weightPointer weights firstItem weightAttention8)
    firstBuffers.hidden secondBuffers.hidden hiddenSizeU32 linearValueSizeU32 block thread scratch

@[always_inline]
private def fullCacheIndex (buffers : Buffers) (item : ModelItem)
    (position head channel : UInt32) : UInt32 :=
  ((item.stateSlot * buffers.contextCapacity + position) *
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
      let cacheIndex := fullCacheIndex buffers item item.position head thread
      storeBFloat buffers.keyCache cacheIndex key
      let value ← loadBFloat buffers.workspace2 (head * fullDimension + thread)
      storeBFloat buffers.valueCache cacheIndex value

@[always_inline]
private partial def fullScoreDot (item : ModelItem) (buffers : Buffers)
    (queryHead keyValueHead position channel : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if channel < fullDimension then
    let query ← loadBFloat buffers.attentionOutput (queryHead * fullDimension + channel)
    let key ← loadBFloat buffers.keyCache
      (fullCacheIndex buffers item position keyValueHead channel)
    fullScoreDot item buffers queryHead keyValueHead position
      (channel + blockThreadCount)
      (Cuda.fma query key accumulator)
  else
    return accumulator

/-!
Online softmax keeps one running maximum, denominator, and value accumulator per output channel.
The score collective broadcasts one scalar to all 256 channels, so causal attention no longer
reserves one shared Float32 per possible context position.
-/
@[always_inline, convergent]
private partial def onlineFullAttention (item : ModelItem) (buffers : Buffers)
    (queryHead keyValueHead position thread : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8)
    (maximum denominator accumulator : Float32) : Cuda.DeviceM Float32 := do
  if position <= item.position then
    let localScore ← fullScoreDot item buffers queryHead keyValueHead position thread zero
    let score ← Cuda.Collective.blockSum scratch localScore
    let score := score * attentionScale
    let value ← loadBFloat buffers.valueCache
      (fullCacheIndex buffers item position keyValueHead thread)
    if position == 0 then
      onlineFullAttention item buffers queryHead keyValueHead (position + 1) thread scratch
        score one value
    else
      let nextMaximum := max maximum score
      let previousScale ← Cuda.fastExp (maximum - nextMaximum)
      let currentScale ← Cuda.fastExp (score - nextMaximum)
      let nextDenominator := Cuda.fma denominator previousScale currentScale
      let nextAccumulator := Cuda.fma accumulator previousScale (currentScale * value)
      onlineFullAttention item buffers queryHead keyValueHead (position + 1) thread scratch
        nextMaximum nextDenominator nextAccumulator
  else
    Cuda.fastDivide accumulator denominator

@[always_inline, convergent]
private def runFullAttention (item : ModelItem) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (head thread : UInt32) : Cuda.DeviceM Unit := do
  if head < fullQueryHeads then
    let scratch : Cuda.Collective.BlockScratch 8 := shared
    let keyValueHead := head / fullGroups
    let attention ← onlineFullAttention item buffers head keyValueHead 0 thread scratch
      negativeInfinity zero zero
    let gateInput ← loadBFloat buffers.workspace0
      (head * (2 * fullDimension) + fullDimension + thread)
    let gate := rounded (← deviceSigmoid gateInput)
    storeBFloat buffers.attentionOutput (head * fullDimension + thread)
      (rounded attention * gate)

@[always_inline, convergent]
private def runFullAttentionStep (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread : UInt32) :
    Cuda.DeviceM Unit := do
  runFullPrepare item weights buffers shared block thread
  Cuda.gridSync
  runFullAttention item buffers shared block thread
  Cuda.gridSync

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
  runFullAttentionStep item weights buffers shared block thread
  runResidualProjection buffers.attentionOutput (← weightPointer weights item weightAttention5)
    buffers.hidden hiddenSizeU32 fullOutputSize.toUInt32 block thread scratch

@[always_inline, convergent]
private def runFullLayerBatch2 (firstItem secondItem : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8) (firstBuffers secondBuffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  runProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights firstItem weightAttention2)
    firstBuffers.workspace0 secondBuffers.workspace0
    fullQueryProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights firstItem weightAttention3)
    firstBuffers.workspace1 secondBuffers.workspace1
    fullKeyValueProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights firstItem weightAttention4)
    firstBuffers.workspace2 secondBuffers.workspace2
    fullKeyValueProjectionSize.toUInt32 hiddenSizeU32 block thread scratch
  runFullAttentionStep firstItem weights firstBuffers shared block thread
  runFullAttentionStep secondItem weights secondBuffers shared block thread
  runResidualProjectionBatch2 firstBuffers.attentionOutput secondBuffers.attentionOutput
    (← weightPointer weights firstItem weightAttention5)
    firstBuffers.hidden secondBuffers.hidden hiddenSizeU32 fullOutputSize.toUInt32
    block thread scratch

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

@[always_inline, convergent]
private def runMLPBatch2 (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (firstBuffers secondBuffers : Buffers)
    (shared : Cuda.DevicePtr Float32) (block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  let normWeight ← weightPointer weights item weightPostNorm
  runRmsNorm firstBuffers.hidden normWeight firstBuffers.normalized firstBuffers scratch
    block thread globalThread globalThreads
  runRmsNorm secondBuffers.hidden normWeight secondBuffers.normalized secondBuffers scratch
    block thread globalThread globalThreads
  runSwiGLUProjectionBatch2 firstBuffers.normalized secondBuffers.normalized
    (← weightPointer weights item weightMlpGate) (← weightPointer weights item weightMlpUp)
    firstBuffers.workspace2 secondBuffers.workspace2 intermediateSizeU32 hiddenSizeU32
    block thread scratch
  runResidualProjectionBatch2 firstBuffers.workspace2 secondBuffers.workspace2
    (← weightPointer weights item weightMlpDown) firstBuffers.hidden secondBuffers.hidden
    hiddenSizeU32 intermediateSizeU32 block thread scratch

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

@[struct]
private structure IndexedMaximum where
  value : Float32
  index : UInt32
  deriving Nonempty

@[always_inline]
private def selectIndexedMaximum (candidate current : IndexedMaximum) : IndexedMaximum :=
  if candidate.value > current.value ||
      (candidate.value == current.value && candidate.index < current.index) then
    candidate
  else
    current

@[always_inline]
private partial def blockOwnedMaximumLoop (logits : Cuda.DevicePtr Float32)
    (row rowStride : UInt32) (best : IndexedMaximum) : Cuda.DeviceM IndexedMaximum := do
  if row < vocabularySizeU32 then
    let first ← Cuda.loadFloat32 logits row.toUSize
    let best := selectIndexedMaximum { value := first, index := row } best
    let secondRow := row + 1
    let best ←
      if secondRow < vocabularySizeU32 then
        let second ← Cuda.loadFloat32 logits secondRow.toUSize
        pure <| selectIndexedMaximum { value := second, index := secondRow } best
      else
        pure best
    blockOwnedMaximumLoop logits (row + rowStride) rowStride best
  else
    return best

@[always_inline]
private def loadIndexedPartial (buffers : Buffers) (slot : UInt32) :
    Cuda.DeviceM IndexedMaximum := do
  return {
    value := ← Cuda.loadFloat32 buffers.partialMaximum slot.toUSize
    index := ← Cuda.loadUInt32 (buffers.partialSum : Cuda.DevicePtr UInt32) slot.toUSize
  }

@[always_inline, convergent]
private def runBlockOwnedGreedySampling
    (finalHidden lmHeadWeight : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32)
    (sampled : Cuda.DevicePtr UInt32) (buffers : Buffers)
    (scratch : Cuda.Collective.BlockScratch 8) (block thread : UInt32) :
    Cuda.DeviceM Unit := do
  if thread == 0 then
    let best ← blockOwnedMaximumLoop logits (block * 2) (gridBlockCount * 2)
      { value := negativeInfinity, index := 0xffffffff }
    Cuda.storeFloat32 buffers.partialMaximum block.toUSize best.value
    Cuda.storeUInt32 (buffers.partialSum : Cuda.DevicePtr UInt32) block.toUSize best.index
  Cuda.gridSync
  let approximate ← loadIndexedPartial buffers block
  let localExact ← dotProjectionRow finalHidden lmHeadWeight approximate.index
    (hiddenSizeU32 / projectionValuesPerTransactionNat.toUInt32) thread zero
  let exact ← Cuda.Collective.blockSumLeader scratch localExact
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialMaximum block.toUSize exact
  Cuda.gridSync
  if block == 0 then
    let first ← loadIndexedPartial buffers thread
    let secondSlot := thread + blockThreadCount
    let candidate ←
      if secondSlot < gridBlockCount then
        pure <| selectIndexedMaximum (← loadIndexedPartial buffers secondSlot) first
      else
        pure first
    let maximum ← Cuda.Collective.blockMax scratch candidate.value
    if thread == 0 then
      Cuda.storeUInt32 sampled 0 0xffffffff
    Cuda.blockSync
    if candidate.value == maximum then
      discard <| Cuda.atomicMinUInt32 sampled candidate.index
    Cuda.blockSync

@[always_inline, convergent]
private def runGlobalFinalReduction (item : ModelItem) (buffers : Buffers)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (sampled : Cuda.DevicePtr UInt32) (scratch : Cuda.Collective.BlockScratch 8)
    (outputSlot block thread globalThread globalThreads : UInt32) : Cuda.DeviceM Unit := do
  if block == 0 && thread == 0 then
    Cuda.storeUInt32 sampled 0 0xffffffff
  Cuda.gridSync
  let localMaximum ← localMaximumLoop logits globalThread globalThreads negativeInfinity
  let blockMaximum ← Cuda.Collective.blockMaxLeader scratch localMaximum
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialMaximum block.toUSize blockMaximum
  Cuda.gridSync
  let maximum ←
    match finalMaximumReductionSchedule with
    | .repeatedGridPartials =>
      maximumPartials buffers.partialMaximum 0 negativeInfinity
    | .singleBlockOwner => do
      if block == 0 then
        let localMaximum ← maximumPartialsStrided buffers.partialMaximum thread blockThreadCount
          negativeInfinity
        let maximum ← Cuda.Collective.blockMaxLeader scratch localMaximum
        if thread == 0 then
          Cuda.storeFloat32 buffers.partialMaximum 0 maximum
      Cuda.gridSync
      Cuda.loadFloat32 buffers.partialMaximum 0
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
      Cuda.storeFloat32 buffers.loss outputSlot.toUSize
        (maximum + (← Cuda.fastLog denominator) - targetLogit)
  else
    sampleMaximumLoop logits sampled maximum globalThread globalThreads

@[always_inline, convergent]
private def runFinal (item : ModelItem) (weights : Cuda.DevicePtrTable UInt8)
    (buffers : Buffers)
    (outputQueue : Cuda.Mailbox.Device TokenEvent)
    (shared : Cuda.DevicePtr Float32)
    (outputSlot block thread globalThread globalThreads : UInt32) :
    Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  let logits := floatOffset buffers.logits (outputSlot * vocabularySizeU32)
  let gradient := bfloatOffset buffers.gradient (outputSlot * vocabularySizeU32)
  let sampled := buffers.sampled + outputSlot.toUSize * 4
  let finalHidden := bfloatOffset buffers.finalHidden (outputSlot * hiddenSizeU32)
  let lmHeadWeight ← weightPointer weights item weightAttention0
  runRmsNorm buffers.hidden (← weightPointer weights item weightInputNorm) finalHidden buffers scratch
    block thread globalThread globalThreads
  if useGreedyLmHeadMXFP8 && item.kind == kindGreedyFinal then
    projectRowPairsFloatMXFP8 finalHidden buffers.lmHeadData buffers.lmHeadScales logits
      vocabularySizeU32 hiddenSizeU32 (block * 2) (gridBlockCount * 2) thread scratch
  else
    projectRowPairsFloat finalHidden lmHeadWeight logits
      vocabularySizeU32 hiddenSizeU32 (block * 2) (gridBlockCount * 2) thread scratch
  match greedySamplingSchedule with
  | .globalLogitRescan =>
    runGlobalFinalReduction item buffers logits gradient sampled scratch
      outputSlot block thread globalThread globalThreads
  | .blockOwnedRows =>
    if item.kind == kindGreedyFinal then
      runBlockOwnedGreedySampling finalHidden lmHeadWeight logits sampled buffers scratch block thread
    else
      runGlobalFinalReduction item buffers logits gradient sampled scratch
        outputSlot block thread globalThread globalThreads
  Cuda.gridSync
  if block == 0 && thread == 0 then
    let event := buffers.tokenEvents + outputSlot.toUSize * 8
    Cuda.storeUInt32 (event : Cuda.DevicePtr UInt32) 0 item.position
    Cuda.storeUInt32 (event : Cuda.DevicePtr UInt32) 1 (← Cuda.loadUInt32 sampled 0)
    let result ← Cuda.Mailbox.Device.trySendFrom outputQueue event
    unless result.isSent do
      Cuda.panic 0x3603 result.code.toUInt64

@[always_inline]
private def prefillLayerItem (layer position token : UInt32) : ModelItem :=
  let full := (layer + 1) % 4 == 0
  {
    kind := if full then kindFull else kindLinear
    stateSlot := if full then layer / 4 else layer - ((layer + 1) / 4)
    weightSlot := layer + 1
    token
    position
    target := 0
    descriptor := layer + 1
    count := 0
  }

@[always_inline, convergent]
private partial def runPrefillLayers (layer position token : UInt32)
    (weights : Cuda.DevicePtrTable UInt8) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32)
    (block thread globalThread globalThreads : UInt32) : Cuda.DeviceM Unit := do
  if layer < layerCount.toUInt32 then
    let layerItem := prefillLayerItem layer position token
    runRmsNorm buffers.hidden (← weightPointer weights layerItem weightInputNorm)
      buffers.normalized buffers shared block thread globalThread globalThreads
    if layerItem.kind == kindFull then
      runFullLayer layerItem weights buffers shared block thread
    else
      runLinearLayer layerItem weights buffers shared block thread
    runMLP layerItem weights buffers shared block thread globalThread globalThreads
    Cuda.gridSync
    if block == 0 && thread == 0 then
      Cuda.atomicAddUInt32At_ buffers.counts layerItem.descriptor 1
    Cuda.gridSync
    runPrefillLayers (layer + 1) position token weights buffers shared
      block thread globalThread globalThreads

@[always_inline, convergent]
private partial def runPrefillLayersBatch2
    (layer firstPosition secondPosition firstToken secondToken : UInt32)
    (weights : Cuda.DevicePtrTable UInt8) (firstBuffers secondBuffers : Buffers)
    (shared : Cuda.DevicePtr Float32)
    (block thread globalThread globalThreads : UInt32) : Cuda.DeviceM Unit := do
  if layer < layerCount.toUInt32 then
    let firstItem := prefillLayerItem layer firstPosition firstToken
    let secondItem := prefillLayerItem layer secondPosition secondToken
    let inputNorm ← weightPointer weights firstItem weightInputNorm
    runRmsNorm firstBuffers.hidden inputNorm firstBuffers.normalized firstBuffers shared
      block thread globalThread globalThreads
    runRmsNorm secondBuffers.hidden inputNorm secondBuffers.normalized secondBuffers shared
      block thread globalThread globalThreads
    if firstItem.kind == kindFull then
      runFullLayerBatch2 firstItem secondItem weights firstBuffers secondBuffers
        shared block thread
    else
      runLinearLayerBatch2 firstItem secondItem weights firstBuffers secondBuffers
        shared block thread
    runMLPBatch2 firstItem weights firstBuffers secondBuffers shared
      block thread globalThread globalThreads
    Cuda.gridSync
    if block == 0 && thread == 0 then
      Cuda.atomicAddUInt32At_ firstBuffers.counts firstItem.descriptor 2
    Cuda.gridSync
    runPrefillLayersBatch2 (layer + 1) firstPosition secondPosition firstToken secondToken
      weights firstBuffers secondBuffers shared block thread globalThread globalThreads

@[always_inline, convergent]
private partial def runPrefillPositions (start count offset : UInt32)
    (weights : Cuda.DevicePtrTable UInt8) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32)
    (block thread globalThread globalThreads : UInt32) : Cuda.DeviceM Unit := do
  if offset < count then
    let position := start + offset
    let token ← Cuda.loadUInt32 buffers.promptTokens position.toUSize
    let embedding : ModelItem := {
        kind := kindEmbedding
        stateSlot := 0
        weightSlot := 0
        token
        position
        target := 0
        descriptor := 0
        count := 0
      }
    let firstBuffers := prefillRowBuffers buffers 0
    if offset + 1 < count then
      let secondPosition := position + 1
      let secondToken ← Cuda.loadUInt32 buffers.promptTokens secondPosition.toUSize
      let secondBuffers := prefillRowBuffers buffers 1
      let embeddingWeight ← weightPointer weights embedding weightInputNorm
      embeddingLoop embeddingWeight firstBuffers.hidden token globalThread globalThreads
      embeddingLoop embeddingWeight secondBuffers.hidden secondToken globalThread globalThreads
      if globalThread < linearHeads then
        Cuda.storeFloat32 buffers.partialMaximum globalThread.toUSize zero
      Cuda.gridSync
      if block == 0 && thread == 0 then
        Cuda.atomicAddUInt32At_ buffers.counts (0 : UInt32) 2
      Cuda.gridSync
      runPrefillLayersBatch2 0 position secondPosition token secondToken weights
        firstBuffers secondBuffers shared block thread globalThread globalThreads
      runPrefillPositions start count (offset + 2) weights buffers shared
        block thread globalThread globalThreads
    else
      embeddingLoop (← weightPointer weights embedding weightInputNorm)
        firstBuffers.hidden token globalThread globalThreads
      if globalThread < linearHeads then
        Cuda.storeFloat32 buffers.partialMaximum globalThread.toUSize zero
      Cuda.gridSync
      if block == 0 && thread == 0 then
        Cuda.atomicAddUInt32At_ buffers.counts (0 : UInt32) 1
      Cuda.gridSync
      runPrefillLayers 0 position token weights firstBuffers shared
        block thread globalThread globalThreads
      runPrefillPositions start count (offset + 1) weights buffers shared
        block thread globalThread globalThreads

@[always_inline, convergent]
private def runModelItem (item : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8)
    (contextCapacity : UInt32)
    (lmHeadData : Cuda.DevicePtr Cuda.Float8E4M3)
    (lmHeadScales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput
      convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (promptTokens : Cuda.DevicePtr UInt32)
    (keyCache valueCache : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled counts : Cuda.DevicePtr UInt32) (tokenEvents : Cuda.DevicePtr TokenEvent)
    (outputStorage : Cuda.DeviceSlice UInt8) (outputLayout : Cuda.Mailbox.Layout)
    (outputSlot : UInt32) :
    Cuda.DeviceM Unit := do
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let globalThread ← Cuda.globalThreadIdxX
  let globalThreads ← Cuda.globalThreadCountX
  let shared ← Cuda.dynamicShared (α := Float32)
  let buffers : Buffers := {
    contextCapacity,
    hidden, normalized, finalHidden, workspace0, workspace1, workspace2, attentionOutput,
    lmHeadData, lmHeadScales, convolutionState, recurrentState,
    promptTokens, keyCache, valueCache, logits, gradient,
    partialMaximum,
    partialSum, loss, sampled, counts, tokenEvents
  }
  let outputQueue : Cuda.Mailbox.Device TokenEvent :=
    Cuda.Mailbox.attachSystem outputStorage outputLayout
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  if contextCapacity > maxSequenceTokensU32 || item.position >= contextCapacity then
    Cuda.panic 0x3601 item.position.toUInt64
  else if item.kind == kindPrefill &&
      (item.count == 0 || item.count > contextCapacity - item.position) then
    Cuda.panic 0x3604 item.count.toUInt64
  else if item.kind == kindPrefill then
    runPrefillPositions item.position item.count 0 weights buffers shared
      block thread globalThread globalThreads
  else if item.kind == kindEmbedding then
    embeddingLoop (← weightPointer weights item weightInputNorm) buffers.hidden item.token
      globalThread globalThreads
    if globalThread < linearHeads then
      Cuda.storeFloat32 buffers.partialMaximum globalThread.toUSize zero
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
  else if item.kind == kindFinal || item.kind == kindGreedyFinal ||
      item.kind == kindReferenceFinal then
    runFinal item weights buffers outputQueue shared outputSlot
      block thread globalThread globalThreads
  else
    Cuda.panic 0x3602 item.kind.toUInt64
  if item.kind != kindPrefill && block == 0 && thread == 0 then
    Cuda.atomicAddUInt32At_ buffers.counts item.descriptor 1

@[always_inline, convergent]
private def runTrainingBatch2Item (item : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8)
    (contextCapacity : UInt32)
    (lmHeadData : Cuda.DevicePtr Cuda.Float8E4M3)
    (lmHeadScales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput
      convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (promptTokens : Cuda.DevicePtr UInt32)
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
    contextCapacity,
    hidden, normalized, finalHidden, workspace0, workspace1, workspace2, attentionOutput,
    lmHeadData, lmHeadScales, convolutionState, recurrentState,
    promptTokens, keyCache, valueCache, logits, gradient,
    partialMaximum, partialSum, loss, sampled, counts, tokenEvents
  }
  let outputQueue : Cuda.Mailbox.Device TokenEvent :=
    Cuda.Mailbox.attachSystem outputStorage outputLayout
  if item.kind != kindTrainingBatch2 || contextCapacity > maxTrainingSequenceTokensU32 ||
      item.position >= contextCapacity || item.count != contextCapacity then
    Cuda.panic 0x3605 item.position.toUInt64
  else
    let firstBuffers := trainingBatchRowBuffers buffers 0
    let secondBuffers := trainingBatchRowBuffers buffers 1
    let embeddingItem : ModelItem := {
      kind := kindEmbedding
      stateSlot := 0
      weightSlot := 0
      token := item.token
      position := item.position
      target := item.target
      descriptor := 0
      count := 0
    }
    let embeddingWeight ← weightPointer weights embeddingItem weightInputNorm
    embeddingLoop embeddingWeight firstBuffers.hidden item.token globalThread globalThreads
    embeddingLoop embeddingWeight secondBuffers.hidden item.stateSlot globalThread globalThreads
    if globalThread < linearHeads then
      Cuda.storeFloat32 buffers.partialMaximum globalThread.toUSize zero
    Cuda.gridSync
    runPrefillLayersBatch2 0 item.position item.position item.token item.stateSlot weights
      firstBuffers secondBuffers shared block thread globalThread globalThreads
    let firstSlot := item.position
    let secondSlot := item.count + item.position
    let firstFinalHidden := bfloatOffset buffers.finalHidden (firstSlot * hiddenSizeU32)
    let secondFinalHidden := bfloatOffset buffers.finalHidden (secondSlot * hiddenSizeU32)
    let finalItem : ModelItem := {
      kind := kindFinal
      stateSlot := 0
      weightSlot := (layerCount + 1).toUInt32
      token := item.token
      position := item.position
      target := item.target
      descriptor := 0
      count := 0
    }
    let scratch : Cuda.Collective.BlockScratch 8 := shared
    let finalNorm ← weightPointer weights finalItem weightInputNorm
    runRmsNorm firstBuffers.hidden finalNorm firstFinalHidden firstBuffers scratch
      block thread globalThread globalThreads
    runRmsNorm secondBuffers.hidden finalNorm secondFinalHidden secondBuffers scratch
      block thread globalThread globalThreads
    let lmHeadWeight ← weightPointer weights finalItem weightAttention0
    projectRowsBatch2Float firstFinalHidden secondFinalHidden lmHeadWeight
      (floatOffset buffers.logits (firstSlot * vocabularySizeU32))
      (floatOffset buffers.logits (secondSlot * vocabularySizeU32))
      vocabularySizeU32 hiddenSizeU32 block gridBlockCount thread scratch
    Cuda.gridSync
    if block == 0 && thread == 0 then
      let firstEvent := buffers.tokenEvents + firstSlot.toUSize * 8
      Cuda.storeUInt32 (firstEvent : Cuda.DevicePtr UInt32) 0 firstSlot
      Cuda.storeUInt32 (firstEvent : Cuda.DevicePtr UInt32) 1 0
      let firstResult ← Cuda.Mailbox.Device.trySendFrom outputQueue firstEvent
      unless firstResult.isSent do
        Cuda.panic 0x3606 firstResult.code.toUInt64
      let secondEvent := buffers.tokenEvents + secondSlot.toUSize * 8
      Cuda.storeUInt32 (secondEvent : Cuda.DevicePtr UInt32) 0 secondSlot
      Cuda.storeUInt32 (secondEvent : Cuda.DevicePtr UInt32) 1 0
      let secondResult ← Cuda.Mailbox.Device.trySendFrom outputQueue secondEvent
      unless secondResult.isSent do
        Cuda.panic 0x3607 secondResult.code.toUInt64

/--
Two-sequence retained-training kernel. Every projection applies each checkpoint weight transaction
to both independent activation rows before advancing, while recurrent and KV state remain isolated.
-/
@[cuda_grid_persistent]
def qwen36TrainingBatch2Model (item : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8)
    (contextCapacity : UInt32)
    (lmHeadData : Cuda.DevicePtr Cuda.Float8E4M3)
    (lmHeadScales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput
      convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (promptTokens : Cuda.DevicePtr UInt32)
    (keyCache valueCache : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled counts : Cuda.DevicePtr UInt32) (tokenEvents : Cuda.DevicePtr TokenEvent)
    (outputStorage : Cuda.DeviceSlice UInt8) (outputLayout : Cuda.Mailbox.Layout) :
    Cuda.DeviceM Unit :=
  runTrainingBatch2Item item weights contextCapacity lmHeadData lmHeadScales
    hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput convolutionState
    recurrentState promptTokens keyCache valueCache logits gradient partialMaximum partialSum loss
    sampled counts tokenEvents outputStorage outputLayout
/-- Position-major persistent kernel for retained training activations and logit gradients. -/
@[cuda_grid_persistent]
def qwen36TrainingModel (item : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8)
    (contextCapacity : UInt32)
    (lmHeadData : Cuda.DevicePtr Cuda.Float8E4M3)
    (lmHeadScales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput
      convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (promptTokens : Cuda.DevicePtr UInt32)
    (keyCache valueCache : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled counts : Cuda.DevicePtr UInt32) (tokenEvents : Cuda.DevicePtr TokenEvent)
    (outputStorage : Cuda.DeviceSlice UInt8) (outputLayout : Cuda.Mailbox.Layout) :
    Cuda.DeviceM Unit :=
  runModelItem item weights contextCapacity lmHeadData lmHeadScales
    hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput convolutionState
    recurrentState promptTokens keyCache valueCache logits gradient partialMaximum partialSum loss
    sampled counts tokenEvents outputStorage outputLayout item.position

/-- Rolling-output persistent kernel for autoregressive inference over a request-sized KV cache. -/
@[cuda_grid_persistent]
def qwen36InferenceModel (item : ModelItem)
    (weights : Cuda.DevicePtrTable UInt8)
    (contextCapacity : UInt32)
    (lmHeadData : Cuda.DevicePtr Cuda.Float8E4M3)
    (lmHeadScales : Cuda.DevicePtr Cuda.ScaleUE8M0)
    (hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput
      convolutionState : Cuda.DevicePtr Cuda.BFloat16)
    (recurrentState : Cuda.DevicePtr Float32)
    (promptTokens : Cuda.DevicePtr UInt32)
    (keyCache valueCache : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled counts : Cuda.DevicePtr UInt32) (tokenEvents : Cuda.DevicePtr TokenEvent)
    (outputStorage : Cuda.DeviceSlice UInt8) (outputLayout : Cuda.Mailbox.Layout) :
    Cuda.DeviceM Unit :=
  runModelItem item weights contextCapacity lmHeadData lmHeadScales
    hidden normalized finalHidden workspace0 workspace1 workspace2 attentionOutput convolutionState
    recurrentState promptTokens keyCache valueCache logits gradient partialMaximum partialSum loss
    sampled counts tokenEvents outputStorage outputLayout 0

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
    (counts : Cuda.DevicePtr UInt32)
    (countElements convolutionElements recurrentElements resetState : UInt32) :
    Cuda.DeviceM Unit := do
  let index ← Cuda.globalThreadIdxX
  let stride ← Cuda.globalThreadCountX
  if resetState != 0 then
    zeroBFloatLoop convolutionState convolutionElements index stride
    zeroFloatLoop recurrentState recurrentElements index stride
  -- Every causal K/V position is written before it is read; unlike recurrent state it needs no
  -- eager zero-fill, which also avoids work proportional to the requested inference context.
  zeroUIntLoop counts countElements index stride

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

/-- Checked host-side ownership layout for one bounded persistent sequence. -/
structure PersistentLayout where
  tokenCount : Nat
  outputTokenCount : Nat
  retainsPositionOutputs : Bool
  descriptorCount : Nat
  countElements : Nat
  queueCapacity : Nat
  convolutionElements : Nat
  recurrentElements : Nat
  keyValueCacheElements : Nat
  logitsElements : Nat
  deriving Repr, BEq

private def PersistentLayout.forTokensWith (tokenCount limit : Nat)
    (retainsPositionOutputs : Bool) (purpose : String) : Except String PersistentLayout := do
  if tokenCount == 0 then
    throw s!"persistent Qwen3.8 {purpose} requires at least one token"
  if tokenCount > limit then
    throw s!"persistent Qwen3.8 {purpose} supports at most {limit} tokens, got {tokenCount}"
  let outputTokenCount := if retainsPositionOutputs then tokenCount else 1
  let descriptorCount := tokenCount * descriptorsPerTokenNat
  return {
    tokenCount
    outputTokenCount
    retainsPositionOutputs
    descriptorCount
    countElements := if retainsPositionOutputs then descriptorCount else descriptorsPerTokenNat
    queueCapacity := if retainsPositionOutputs then descriptorCount else descriptorsPerTokenNat
    convolutionElements := linearLayerCount * convolutionElementsPerLayerNat
    recurrentElements := linearLayerCount * linearStateElementsPerLayerNat
    keyValueCacheElements :=
      fullAttentionLayerCount * tokenCount * fullKeyValueProjectionSize
    logitsElements := outputTokenCount * vocabularySize
  }

/-- Retain all position-major outputs needed by loss and LoRA training. -/
def PersistentLayout.forTrainingTokens (tokenCount : Nat) : Except String PersistentLayout :=
  PersistentLayout.forTokensWith tokenCount maxTrainingSequenceTokens true "training"
/-- Two independent causal sequences sharing every projection weight read in a two-row tile. -/
private def PersistentLayout.forTrainingBatch2 (sequenceLength : Nat) :
    Except String PersistentLayout := do
  let base ← PersistentLayout.forTrainingTokens sequenceLength
  return {
    base with
    outputTokenCount := 2 * sequenceLength
    queueCapacity := sequenceLength
    convolutionElements := 2 * base.convolutionElements
    recurrentElements := 2 * base.recurrentElements
    keyValueCacheElements := 2 * base.keyValueCacheElements
    logitsElements := 2 * base.logitsElements
  }

/-- Recycle one output row while sizing only the causal KV cache to the inference request. -/
def PersistentLayout.forInferenceTokens (tokenCount : Nat) : Except String PersistentLayout :=
  PersistentLayout.forTokensWith tokenCount maxSequenceTokens false "inference"

/-- Compatibility spelling for the retained training layout. -/
def PersistentLayout.forTokens (tokenCount : Nat) : Except String PersistentLayout :=
  PersistentLayout.forTrainingTokens tokenCount

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
  promptTokens : Cuda.Buffer UInt32
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

/-!
The inference arena owns physical context capacity independently of any one request. Its recurrent
and convolution snapshots mark a reusable canonical prompt prefix; full-attention KV entries
before that prefix are immutable and therefore remain in the arena directly.
-/
private structure InferenceArena where
  buffers : ModelBuffers
  convolutionSnapshot : Cuda.Buffer Cuda.BFloat16
  recurrentSnapshot : Cuda.Buffer Float32
  uploadStream : Cuda.Stream

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
  let hidden ← Cuda.Buffer.alloc Cuda.BFloat16
    (prefillTileRowsNat * hiddenSize).toUSize
  let normalized ← Cuda.Buffer.alloc Cuda.BFloat16
    (prefillTileRowsNat * hiddenSize).toUSize
  let finalHidden ← Cuda.Buffer.alloc Cuda.BFloat16
    (layout.outputTokenCount * hiddenSize).toUSize
  let workspace0 ← Cuda.Buffer.alloc Cuda.BFloat16
    (prefillTileRowsNat * intermediateSize).toUSize
  let workspace1 ← Cuda.Buffer.alloc Cuda.BFloat16
    (prefillTileRowsNat * intermediateSize).toUSize
  let workspace2 ← Cuda.Buffer.alloc Cuda.BFloat16
    (prefillTileRowsNat * intermediateSize).toUSize
  let attentionOutput ← Cuda.Buffer.alloc Cuda.BFloat16
    (prefillTileRowsNat * linearValueSize).toUSize
  let convolutionState ← Cuda.Buffer.alloc Cuda.BFloat16
    layout.convolutionElements.toUSize
  let recurrentState ← Cuda.Buffer.alloc Float32
    layout.recurrentElements.toUSize
  let promptTokens ← Cuda.Buffer.alloc UInt32 layout.tokenCount.toUSize
  let keyCache ← Cuda.Buffer.alloc Cuda.BFloat16
    layout.keyValueCacheElements.toUSize
  let valueCache ← Cuda.Buffer.alloc Cuda.BFloat16
    layout.keyValueCacheElements.toUSize
  let logits ← Cuda.Buffer.alloc Float32 layout.logitsElements.toUSize
  let gradient ← Cuda.Buffer.alloc Cuda.BFloat16 layout.logitsElements.toUSize
  let partialMaximum ← Cuda.Buffer.alloc Float32 gridBlockCountNat.toUSize
  let partialSum ← Cuda.Buffer.alloc Float32 gridBlockCountNat.toUSize
  let loss ← Cuda.Buffer.alloc Float32 layout.outputTokenCount.toUSize
  let sampled ← Cuda.Buffer.alloc UInt32 layout.outputTokenCount.toUSize
  let counts ← Cuda.Buffer.alloc UInt32 layout.countElements.toUSize
  let tokenEvents ← Cuda.Buffer.alloc TokenEvent layout.outputTokenCount.toUSize
  return {
    layout, hidden, normalized, finalHidden, workspace0, workspace1, workspace2, attentionOutput,
    convolutionState, recurrentState, promptTokens, keyCache, valueCache, logits, gradient,
    partialMaximum,
    partialSum, loss, sampled, counts, tokenEvents
  }

private def allocateInferenceArena : IO InferenceArena := do
  let layout ← match PersistentLayout.forInferenceTokens maxSequenceTokens with
    | .ok layout => pure layout
    | .error error => throw <| IO.userError error
  let buffers ← allocateBuffers layout
  let convolutionSnapshot ← Cuda.Buffer.alloc Cuda.BFloat16
    layout.convolutionElements.toUSize
  let recurrentSnapshot ← Cuda.Buffer.alloc Float32
    layout.recurrentElements.toUSize
  let uploadStream ← Cuda.Stream.create
  return { buffers, convolutionSnapshot, recurrentSnapshot, uploadStream }

private def promptTokenBytes (tokens : @& Array UInt32) : ByteArray := Id.run do
  let mut bytes := ByteArray.emptyWithCapacity (tokens.size * 4)
  for token in tokens do
    bytes := (((bytes.push token.toUInt8).push (token >>> 8).toUInt8).push
      (token >>> 16).toUInt8).push (token >>> 24).toUInt8
  return bytes

private def uploadPrompt (arena : @& InferenceArena)
    (tokens : @& Array UInt32) : IO Unit := do
  arena.buffers.promptTokens.copyFrom (promptTokenBytes tokens) arena.uploadStream
  arena.uploadStream.synchronize

private def snapshotInferenceState (arena : @& InferenceArena)
    (stream : @& Cuda.Stream) : IO Unit := do
  arena.buffers.convolutionState.copyPeerTo arena.convolutionSnapshot stream
  arena.buffers.recurrentState.copyPeerTo arena.recurrentSnapshot stream

private def restoreInferenceState (arena : @& InferenceArena)
    (stream : @& Cuda.Stream) : IO Unit := do
  arena.convolutionSnapshot.copyPeerTo arena.buffers.convolutionState stream
  arena.recurrentSnapshot.copyPeerTo arena.buffers.recurrentState stream

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
  count := 0
}

private def embeddingItem (token position descriptor : UInt32) : ModelItem := {
  blankItem with
  kind := kindEmbedding
  token
  position
  descriptor
}

private def prefillItem (position count : UInt32) : ModelItem := {
  blankItem with kind := kindPrefill, position, count
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
    (retainsPositionOutputs greedyOnly : Bool) : ModelItem := {
  blankItem with
  kind :=
    if greedyOnly then kindGreedyFinal
    else if retainsPositionOutputs then kindFinal
    else kindReferenceFinal
  weightSlot := (layerCount + 1).toUInt32
  position
  target
  descriptor
}

private def trainingBatch2Item (firstToken secondToken firstTarget secondTarget position
    sequenceLength : UInt32) : ModelItem := {
  blankItem with
  kind := kindTrainingBatch2
  stateSlot := secondToken
  weightSlot := secondTarget
  token := firstToken
  position
  target := firstTarget
  descriptor := 0
  count := sequenceLength
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
  queueCapacity := layout.queueCapacity.toUSize
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

/-- Immutable real-checkpoint tensor views in persistent record/field order. -/
structure TrainingWeightTable where
  entries : Array Primitives.FrozenBFloat16Weight

namespace TrainingWeightTable

def recordCount : Nat := 66

def get (table : @& TrainingWeightTable) (record field : Nat) :
    IO Primitives.FrozenBFloat16Weight := do
  unless record < recordCount && field < 14 do
    throw <| IO.userError s!"Qwen3.8 training weight index [{record}, {field}] is out of range"
  let some weight := table.entries[record * 14 + field]?
    | throw <| IO.userError "Qwen3.8 training weight table is truncated"
  return weight

def embedding (table : @& TrainingWeightTable) : IO Primitives.FrozenBFloat16Weight :=
  table.get 0 0

def layer (table : @& TrainingWeightTable) (index field : Nat) :
    IO Primitives.FrozenBFloat16Weight :=
  table.get (index + 1) field

def finalNorm (table : @& TrainingWeightTable) : IO Primitives.FrozenBFloat16Weight :=
  table.get 65 0

def lmHead (table : @& TrainingWeightTable) : IO Primitives.FrozenBFloat16Weight :=
  table.get 65 5

end TrainingWeightTable

/-- Published fixed Qwen3.8-27B dimensions used by the checkpoint reverse graph. -/
structure TrainingArchitecture where
  layers : Nat
  hidden : UInt32
  intermediate : UInt32
  vocabulary : UInt32
  linearKeyHeads : UInt32
  linearValueHeads : UInt32
  linearKeyWidth : UInt32
  linearValueWidth : UInt32
  attentionQueryHeads : UInt32
  attentionKeyValueHeads : UInt32
  attentionHeadWidth : UInt32
  rotaryHalf : UInt32
  epsilon : Float32
  queryScale : Float32
  attentionScale : Float32
  deriving Repr

def trainingArchitecture : TrainingArchitecture := {
  layers := layerCount
  hidden := hiddenSizeU32
  intermediate := intermediateSizeU32
  vocabulary := vocabularySizeU32
  linearKeyHeads := 16
  linearValueHeads := linearHeads
  linearKeyWidth := linearDimension
  linearValueWidth := linearDimension
  attentionQueryHeads := fullQueryHeads
  attentionKeyValueHeads := fullKeyValueHeads
  attentionHeadWidth := fullDimension
  rotaryHalf
  epsilon := rmsEpsilon
  queryScale
  attentionScale
}

def TrainingArchitecture.layerUsesFullAttention (_ : @& TrainingArchitecture) (index : Nat) : Bool :=
  (index + 1) % 4 == 0

private def resolveTrainingWeightTable (weights : @& LoadedWeights) (layout : @& TextLayout) :
    IO TrainingWeightTable := do
  let mut records := #[embeddingWeights layout]
  for layer in layout.layers do
    records := records.push (layerWeights layer)
  records := records.push (finalWeights layout)
  let mut entries := #[]
  for record in records do
    for weight in record.toArray do
      let (owner, byteOffset) ← weights.resolve weight.shard weight.byteOffset weight.byteCount
      entries := entries.push { owner, byteOffset, byteCount := weight.byteCount }
  unless entries.size == TrainingWeightTable.recordCount * weightFieldCount.toNat do
    throw <| IO.userError "Qwen3.8 resolved training weight table has an inconsistent field count"
  return { entries }

/--
A single-consumer host view of a live Qwen3.8 persistent launch.

Prefill submits one causal position without paying for an unused LM-head projection.
Enqueue submits a position that publishes a token, and nextToken waits only for a published
position. Finish closes a complete launch without copying the full logits tensor. FinishEarly
cleanly closes a partially consumed launch, for example after an end token. Collect additionally
returns all position-major outputs from a complete launch.
-/
structure InferenceStream where
  expectedPositions : Nat
  prefill : UInt32 → IO Unit
  prefillBatch : Nat → IO Unit
  enqueue : UInt32 → UInt32 → IO Unit
  enqueueGreedy : UInt32 → IO Unit
  nextToken : IO TokenEvent
  finish : IO Unit
  finishEarly : IO Unit
  collect : IO InferenceResult
  abort : IO Unit

private structure InferenceStreamState where
  nextPosition : Nat := 0
  submitted : Nat := 0
  published : Nat := 0
  received : Nat := 0
  firstPublishedPosition : Option Nat := none
  finalized : Bool := false
  aborted : Bool := false

private def enqueuePosition (submit : ModelItem → IO Unit)
    (textLayout : @& TextLayout) (persistentLayout : @& PersistentLayout)
    (position : Nat) (token target : UInt32) (publishToken greedyOnly : Bool) : IO Unit := do
  let descriptorBase :=
    if persistentLayout.retainsPositionOutputs then position * descriptorsPerTokenNat else 0
  submit (embeddingItem token position.toUInt32
    descriptorBase.toUInt32)
  let mut linearSlot : UInt32 := 0
  let mut fullSlot : UInt32 := 0
  for index in [:textLayout.layers.size] do
    let some layer := textLayout.layers[index]?
      | throw <| IO.userError s!"missing resolved Qwen3.8 layer {index}"
    let slot := match layer.attention with
      | .linear _ => linearSlot
      | .full _ => fullSlot
    submit (layerItem layer slot position.toUInt32
      (descriptorBase + index + 1).toUInt32 token target)
    match layer.attention with
    | .linear _ => linearSlot := linearSlot + 1
    | .full _ => fullSlot := fullSlot + 1
  if publishToken then
    submit (finalItem position.toUInt32 target
      (descriptorBase + layerCount + 1).toUInt32
      persistentLayout.retainsPositionOutputs greedyOnly)

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

private def validateDescriptorCounts (buffers : @& ModelBuffers) (positions published : Nat)
    (stream : @& Cuda.Stream) : IO Unit := do
  let counts ← buffers.counts.copyTo stream
  if buffers.layout.retainsPositionOutputs then
    for descriptor in [:positions * descriptorsPerTokenNat] do
      unless readUInt32 counts descriptor == 1 do
        throw <| IO.userError <|
          s!"Qwen3.8 training descriptor {descriptor} executed " ++
          s!"{readUInt32 counts descriptor} times"
  else
    for descriptor in [:descriptorsPerTokenNat] do
      let expected := if descriptor == layerCount + 1 then published else positions
      unless readUInt32 counts descriptor == expected.toUInt32 do
        throw <| IO.userError <|
          s!"Qwen3.8 inference descriptor {descriptor} executed " ++
          s!"{readUInt32 counts descriptor} times, expected {expected}"

private def controlLoadedStream (layout : @& TextLayout)
    (buffers : ModelBuffers) (stream : Cuda.Stream)
    (outputQueue : Cuda.HostIO.Queue TokenEvent)
    (handle : Cuda.GridPersistentHandle ModelItem)
    (submit : ModelItem → IO Unit) (startPosition expectedEndPosition : Nat) :
    IO InferenceStream := do
  let persistentLayout := buffers.layout
  unless startPosition ≤ expectedEndPosition &&
      expectedEndPosition ≤ persistentLayout.tokenCount do
    throw <| IO.userError "Qwen3.8 inference stream plan exceeds its physical context"
  let state ← IO.mkRef ({ nextPosition := startPosition } : InferenceStreamState)
  let close (requireComplete : Bool) : IO Unit := do
    let current ← state.get
    if current.finalized then
      return
    if current.aborted then
      throw <| IO.userError "Qwen3.8 inference stream was aborted"
    unless current.received == current.published do
      throw <| IO.userError <|
        s!"Qwen3.8 inference stream published {current.published} tokens but received " ++
        s!"{current.received}"
    if requireComplete && current.nextPosition != expectedEndPosition then
      throw <| IO.userError <|
        s!"Qwen3.8 inference stream expected to end at {expectedEndPosition}, reached " ++
        s!"{current.nextPosition}"
    handle.shutdown
    handle.waitChecked "Qwen3.8 model megakernel"
    validateDescriptorCounts buffers current.submitted current.published stream
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
  let enqueue (publishToken greedyOnly : Bool) (token target : UInt32) : IO Unit := do
    let current ← state.get
    if current.finalized || current.aborted then
      throw <| IO.userError "cannot enqueue into a closed Qwen3.8 inference stream"
    if current.nextPosition >= expectedEndPosition then
      throw <| IO.userError "Qwen3.8 inference stream received too many positions"
    if token >= vocabularySizeU32 || (!greedyOnly && target >= vocabularySizeU32) then
      throw <| IO.userError "Qwen3.8 token and target IDs must be below 248320"
    let position := current.nextPosition
    enqueuePosition submit layout persistentLayout position token target publishToken greedyOnly
    let firstPublishedPosition :=
      if publishToken && current.firstPublishedPosition.isNone then some position
      else current.firstPublishedPosition
    state.set {
      current with
      nextPosition := position + 1
      submitted := current.submitted + 1
      published := current.published + if publishToken then 1 else 0
      firstPublishedPosition
    }
  let prefillBatch (count : Nat) : IO Unit := do
    let current ← state.get
    if current.finalized || current.aborted then
      throw <| IO.userError "cannot prefill into a closed Qwen3.8 inference stream"
    if count == 0 || count > expectedEndPosition - current.nextPosition then
      throw <| IO.userError "Qwen3.8 prefill span exceeds the inference stream"
    submit (prefillItem current.nextPosition.toUInt32 count.toUInt32)
    state.set {
      current with
      nextPosition := current.nextPosition + count
      submitted := current.submitted + count
    }
  return {
    expectedPositions := expectedEndPosition - startPosition
    prefill := fun token => enqueue false true token 0
    prefillBatch
    enqueue := enqueue true false
    enqueueGreedy := fun token => enqueue true true token 0
    nextToken := do
      let current ← state.get
      if current.finalized || current.aborted then
        throw <| IO.userError "cannot receive from a closed Qwen3.8 inference stream"
      unless current.received < current.published do
        throw <| IO.userError "Qwen3.8 inference stream has no submitted token to receive"
      let event ← awaitTokenEvent outputQueue handle
      let some firstPosition := current.firstPublishedPosition
        | throw <| IO.userError "Qwen3.8 inference stream has no published position"
      let expectedPosition := firstPosition + current.received
      unless event.position.toNat == expectedPosition do
        throw <| IO.userError <|
          s!"Qwen3.8 token stream expected position {expectedPosition}, got {event.position}"
      state.set { current with received := current.received + 1 }
      return event
    finish := finalize
    finishEarly := close false
    collect := do
      unless persistentLayout.retainsPositionOutputs do
        throw <| IO.userError
          "rolling Qwen3.8 inference does not retain position-major training outputs"
      finalize
      return {
        logits := ← buffers.logits.copyTo stream
        loss := ← buffers.loss.copyTo stream
        sampled := ← buffers.sampled.copyTo stream
      }
    abort
  }

private def initializeBuffers (buffers : @& ModelBuffers) (stream : @& Cuda.Stream)
    (resetState : Bool) : IO Unit := do
  (← initializeModelState.launchOn stream initializerConfig buffers.convolutionState
    buffers.recurrentState buffers.counts buffers.layout.countElements.toUInt32
    buffers.layout.convolutionElements.toUInt32 buffers.layout.recurrentElements.toUInt32
    (if resetState then 1 else 0)).waitChecked
    "Qwen3.8 state initialization"

/-- Event-queue capacity rounded up to the mailbox's power-of-two contract. -/
private def mailboxEventCapacity (events : Nat) : USize :=
  (Nat.nextPowerOfTwo (max 2 events)).toUSize

private def startLoadedTrainingStream (layout : @& TextLayout)
    (weights : @& Cuda.BufferTable UInt8) (lmHead : @& QuantizedLmHead)
    (buffers : ModelBuffers) (stream : Cuda.Stream) : IO InferenceStream := do
  initializeBuffers buffers stream true
  let persistentLayout := buffers.layout
  let outputQueue ← Cuda.HostIO.Queue.alloc TokenEvent
    (mailboxEventCapacity persistentLayout.outputTokenCount)
  let handle ← qwen36TrainingModel.startOn stream (persistentConfig persistentLayout)
    weights persistentLayout.tokenCount.toUInt32
    lmHead.data lmHead.scales
    buffers.hidden buffers.normalized buffers.finalHidden buffers.workspace0 buffers.workspace1
    buffers.workspace2 buffers.attentionOutput buffers.convolutionState buffers.recurrentState
    buffers.promptTokens buffers.keyCache buffers.valueCache buffers.logits buffers.gradient
    buffers.partialMaximum
    buffers.partialSum buffers.loss buffers.sampled buffers.counts buffers.tokenEvents
    outputQueue.buffer outputQueue.layout
  controlLoadedStream layout buffers stream outputQueue handle
    (fun item => qwen36TrainingModel.enqueue handle item) 0 persistentLayout.tokenCount

private def startLoadedInferenceStream (layout : @& TextLayout)
    (weights : @& Cuda.BufferTable UInt8) (lmHead : @& QuantizedLmHead)
    (buffers : ModelBuffers) (stream : Cuda.Stream)
    (startPosition expectedEndPosition : Nat) (resetState : Bool) : IO InferenceStream := do
  initializeBuffers buffers stream resetState
  let persistentLayout := buffers.layout
  let outputQueue ← Cuda.HostIO.Queue.alloc TokenEvent
    (mailboxEventCapacity persistentLayout.outputTokenCount)
  let handle ← qwen36InferenceModel.startOn stream (persistentConfig persistentLayout)
    weights persistentLayout.tokenCount.toUInt32
    lmHead.data lmHead.scales
    buffers.hidden buffers.normalized buffers.finalHidden buffers.workspace0 buffers.workspace1
    buffers.workspace2 buffers.attentionOutput buffers.convolutionState buffers.recurrentState
    buffers.promptTokens buffers.keyCache buffers.valueCache buffers.logits buffers.gradient
    buffers.partialMaximum
    buffers.partialSum buffers.loss buffers.sampled buffers.counts buffers.tokenEvents
    outputQueue.buffer outputQueue.layout
  controlLoadedStream layout buffers stream outputQueue handle
    (fun item => qwen36InferenceModel.enqueue handle item) startPosition expectedEndPosition

private def runOnce (layout : TextLayout) (weights : Cuda.BufferTable UInt8)
    (lmHead : QuantizedLmHead)
    (buffers : ModelBuffers)
    (tokens targets : Array UInt32) (stream : Cuda.Stream) : IO InferenceResult := do
  unless !tokens.isEmpty && tokens.size == targets.size &&
      tokens.size <= maxTrainingSequenceTokens do
    throw <| IO.userError <|
      s!"Qwen3.8 token/target arrays must have equal positive length at most " ++
      s!"{maxTrainingSequenceTokens}"
  let persistentLayout ← match PersistentLayout.forTrainingTokens tokens.size with
    | .ok persistentLayout => pure persistentLayout
    | .error error => throw <| IO.userError error
  unless buffers.layout == persistentLayout do
    throw <| IO.userError "Qwen3.8 persistent buffers were allocated for a different sequence layout"
  let inference ← startLoadedTrainingStream layout weights lmHead buffers stream
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
  unless !tokens.isEmpty && tokens.size == targets.size &&
      tokens.size <= maxTrainingSequenceTokens do
    throw <| IO.userError <|
      s!"Qwen3.8 token/target arrays must have equal positive length at most " ++
      s!"{maxTrainingSequenceTokens}"
  let persistentLayout ← match PersistentLayout.forTrainingTokens tokens.size with
    | .ok persistentLayout => pure persistentLayout
    | .error error => throw <| IO.userError error
  let buffers ← allocateBuffers persistentLayout
  let inference ← startLoadedTrainingStream layout weights lmHead buffers stream
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

private def runTrainingBatch2Forward (_layout : TextLayout)
    (weights : Cuda.BufferTable UInt8) (lmHead : QuantizedLmHead)
    (firstTokens secondTokens firstTargets secondTargets : Array UInt32)
    (stream : Cuda.Stream) : IO TrainingBatch := do
  let sequenceLength := firstTokens.size
  unless sequenceLength > 0 && secondTokens.size == sequenceLength &&
      firstTargets.size == sequenceLength && secondTargets.size == sequenceLength do
    throw <| IO.userError "Qwen3.8 two-row training requires four equal nonempty sequences"
  let persistentLayout ← match PersistentLayout.forTrainingBatch2 sequenceLength with
    | .ok persistentLayout => pure persistentLayout
    | .error error => throw <| IO.userError error
  let buffers ← allocateBuffers persistentLayout
  initializeBuffers buffers stream true
  let outputQueue ← Cuda.HostIO.Queue.alloc TokenEvent
    (mailboxEventCapacity persistentLayout.outputTokenCount)
  let handle ← qwen36TrainingBatch2Model.startOn stream (persistentConfig persistentLayout)
    weights persistentLayout.tokenCount.toUInt32
    lmHead.data lmHead.scales
    buffers.hidden buffers.normalized buffers.finalHidden buffers.workspace0 buffers.workspace1
    buffers.workspace2 buffers.attentionOutput buffers.convolutionState buffers.recurrentState
    buffers.promptTokens buffers.keyCache buffers.valueCache buffers.logits buffers.gradient
    buffers.partialMaximum buffers.partialSum buffers.loss buffers.sampled buffers.counts
    buffers.tokenEvents outputQueue.buffer outputQueue.layout
  try
    for position in [:sequenceLength] do
      let firstToken := firstTokens[position]!
      let secondToken := secondTokens[position]!
      let firstTarget := firstTargets[position]!
      let secondTarget := secondTargets[position]!
      if firstToken >= vocabularySizeU32 || secondToken >= vocabularySizeU32 ||
          firstTarget >= vocabularySizeU32 || secondTarget >= vocabularySizeU32 then
        throw <| IO.userError "Qwen3.8 token and target IDs must be below 248320"
      qwen36TrainingBatch2Model.enqueue handle <|
        trainingBatch2Item firstToken secondToken firstTarget secondTarget position.toUInt32
          sequenceLength.toUInt32
    for _ in [:persistentLayout.outputTokenCount] do
      discard <| awaitTokenEvent outputQueue handle
    handle.shutdown
    handle.waitChecked "Qwen3.8 two-row training megakernel"
    return {
      rows := persistentLayout.outputTokenCount.toUInt32
      inputFeatures := hiddenSizeU32
      outputFeatures := vocabularySizeU32
      finalHidden := buffers.finalHidden
      logits := buffers.logits
    }
  catch error =>
    try
      handle.shutdown
      discard <| handle.wait
    catch _ =>
      pure ()
    throw error
/--
Collect independent causal sequences into one row-major tensor. Sequence pairs execute in one
persistent two-row kernel and reuse every projection weight transaction; only an odd tail uses the
single-sequence retained-training route.
-/
private def runTrainingBatchForward (layout : TextLayout)
    (weights : Cuda.BufferTable UInt8) (lmHead : QuantizedLmHead)
    (tokens targets : Array (Array UInt32)) (stream : Cuda.Stream) : IO TrainingBatch := do
  unless !tokens.isEmpty && tokens.size == targets.size do
    throw <| IO.userError <|
      "Qwen3.8 training batch must contain equally many token and target sequences"
  let sequenceLength := tokens[0]!.size
  unless sequenceLength > 0 && sequenceLength ≤ maxTrainingSequenceTokens do
    throw <| IO.userError <|
      s!"Qwen3.8 training sequence length must lie in [1, {maxTrainingSequenceTokens}]"
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
  let hiddenCount := sequenceLength * hiddenSize
  let logitCount := sequenceLength * vocabularySize
  let mut batch := 0
  while batch + 1 < tokens.size do
    let result ← runTrainingBatch2Forward layout weights lmHead
      tokens[batch]! tokens[batch + 1]! targets[batch]! targets[batch + 1]! stream
    (← copyTrainingBF16Kernel.launchOn stream
      (Primitives.elementConfig (2 * hiddenCount).toUInt32) result.finalHidden finalHidden
      (batch * hiddenCount).toUInt32 (2 * hiddenCount).toUInt32).waitChecked
      "Qwen3.8 native-batch final-hidden copy"
    (← copyTrainingF32Kernel.launchOn stream
      (Primitives.elementConfig (2 * logitCount).toUInt32) result.logits logits
      (batch * logitCount).toUInt32 (2 * logitCount).toUInt32).waitChecked
      "Qwen3.8 native-batch logit copy"
    batch := batch + 2
  if batch < tokens.size then
    let result ← runTrainingForward layout weights lmHead tokens[batch]! targets[batch]! stream
    (← copyTrainingBF16Kernel.launchOn stream
      (Primitives.elementConfig hiddenCount.toUInt32) result.finalHidden finalHidden
      (batch * hiddenCount).toUInt32 hiddenCount.toUInt32).waitChecked
      "Qwen3.8 odd-tail final-hidden copy"
    (← copyTrainingF32Kernel.launchOn stream
      (Primitives.elementConfig logitCount.toUInt32) result.logits logits
      (batch * logitCount).toUInt32 logitCount.toUInt32).waitChecked
      "Qwen3.8 odd-tail logit copy"
  return {
    rows := rows.toUInt32
    inputFeatures := hiddenSizeU32
    outputFeatures := vocabularySizeU32
    finalHidden
    logits
  }

/-- A loaded checkpoint with distinct retained-training and rolling-inference launches. -/
structure Checkpoint where
  start : Nat → IO InferenceStream
  startInference : Nat → IO InferenceStream

/-- A shared real checkpoint with both inference and device-resident frozen-base training views. -/
structure TrainableCheckpoint where
  checkpoint : Checkpoint
  stream : Cuda.Stream
  inputFeatures : UInt32
  outputFeatures : UInt32
  weights : TrainingWeightTable
  forward : Array UInt32 → Array UInt32 → IO TrainingBatch
  forwardBatch : Array (Array UInt32) → Array (Array UInt32) → IO TrainingBatch

private structure LoadedCheckpoint where
  layout : TextLayout
  weights : Cuda.BufferTable UInt8
  trainingWeights : TrainingWeightTable
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
  let trainingWeights ← resolveTrainingWeightTable loadedWeights layout
  let lmHead ← quantizeLmHead loadedWeights layout stream
  return { layout, weights, trainingWeights, lmHead, stream }

private def checkpointOfLoaded (loaded : LoadedCheckpoint) : Checkpoint := {
  start := fun positions => do
    let persistentLayout ← match PersistentLayout.forTrainingTokens positions with
      | .ok layout => pure layout
      | .error error => throw <| IO.userError error
    let buffers ← allocateBuffers persistentLayout
    startLoadedTrainingStream loaded.layout loaded.weights loaded.lmHead buffers loaded.stream
  startInference := fun positions => do
    let persistentLayout ← match PersistentLayout.forInferenceTokens positions with
      | .ok layout => pure layout
      | .error error => throw <| IO.userError error
    let buffers ← allocateBuffers persistentLayout
    startLoadedInferenceStream loaded.layout loaded.weights loaded.lmHead buffers loaded.stream
      0 positions true
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
    weights := loaded.trainingWeights
    forward := fun tokens targets =>
      runTrainingForward loaded.layout loaded.weights loaded.lmHead tokens targets loaded.stream
    forwardBatch := fun tokens targets =>
      runTrainingBatchForward loaded.layout loaded.weights loaded.lmHead tokens targets loaded.stream
  }

/-- Run explicit input/target positions while invoking `onToken` as each device token is ready. -/
def Checkpoint.runStreaming (checkpoint : @& Checkpoint) (tokens targets : Array UInt32)
    (onToken : TokenEvent → IO Unit) : IO InferenceResult := do
  unless !tokens.isEmpty && tokens.size == targets.size &&
      tokens.size <= maxTrainingSequenceTokens do
    throw <| IO.userError <|
      s!"Qwen3.8 token/target arrays must have equal positive length at most " ++
      s!"{maxTrainingSequenceTokens}"
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

/--
A serialized chat-generation session. Physical inference buffers are allocated once, while the
canonical prompt prefix is retained as full-attention KV plus snapshotted DeltaNet state.
`cachePrefixTokens` identifies the stable history before the assistant-generation suffix.
-/
structure GenerationSession where
  generateStreamingWith :
    Array UInt32 → Nat → GenerationConfig → (TokenEvent → IO Unit) → IO (Array UInt32)

/-- One loaded weight set shared by stateless correctness/benchmark paths and cached chat. -/
structure ServingCheckpoint where
  checkpoint : Checkpoint
  generation : GenerationSession

/-- Language-model head selected for autoregressive generation. -/
private inductive GenerationHead where
  | bf16Reference
  | mxfp8Greedy

private def enqueueGeneration (inference : @& InferenceStream) (head : GenerationHead)
    (token : UInt32) : IO Unit :=
  match head with
  | .bf16Reference => inference.enqueue token 0
  | .mxfp8Greedy => inference.enqueueGreedy token

/-- Whether a generated token cleanly terminates this request. -/
def GenerationConfig.shouldStop (config : @& GenerationConfig) (token : UInt32) : Bool :=
  config.stopTokens.contains token

private partial def generateRemaining (inference : @& InferenceStream)
    (head : GenerationHead) (stopTokens : @& Array UInt32)
    (remaining : Nat) (latest : TokenEvent)
    (generated : Array UInt32) (onToken : @& TokenEvent → IO Unit) : IO (Array UInt32) := do
  if remaining == 0 then
    inference.finish
    return generated
  enqueueGeneration inference head latest.token
  let latest ← inference.nextToken
  let generated := generated.push latest.token
  onToken latest
  if stopTokens.contains latest.token then
    inference.finishEarly
    return generated
  generateRemaining inference head stopTokens (remaining - 1) latest generated onToken

private def validateGenerationRequest (prompt : @& Array UInt32)
    (config : @& GenerationConfig) : IO Nat := do
  if prompt.isEmpty then
    throw <| IO.userError "Qwen3.8 streaming generation requires a nonempty prompt"
  unless prompt.all (· < vocabularySizeU32) do
    throw <| IO.userError "Qwen3.8 prompt token IDs must be below 248320"
  unless config.stopTokens.all (· < vocabularySizeU32) do
    throw <| IO.userError "Qwen3.8 stop token IDs must be below 248320"
  if config.maxNewTokens == 0 then
    return prompt.size
  let positions := prompt.size + config.maxNewTokens - 1
  if positions > maxSequenceTokens then
    throw <| IO.userError
      s!"Qwen3.8 prompt plus generation requires {positions} positions, maximum is {maxSequenceTokens}"
  return positions

private def generatePromptOnStream (inference : @& InferenceStream)
    (head : GenerationHead) (prompt : @& Array UInt32) (startPosition : Nat)
    (integratedPrefill : Bool) (config : @& GenerationConfig)
    (onToken : @& TokenEvent → IO Unit) :
    IO (Array UInt32) := do
  unless startPosition < prompt.size do
    throw <| IO.userError "Qwen3.8 generation suffix must contain at least one prompt token"
  let prefillCount := prompt.size - 1 - startPosition
  if prefillCount > 0 then
    if integratedPrefill then
      inference.prefillBatch prefillCount
    else
      for position in [startPosition:(prompt.size - 1)] do
        inference.prefill prompt[position]!
  enqueueGeneration inference head prompt[prompt.size - 1]!
  let latest ← inference.nextToken
  let generated := #[latest.token]
  onToken latest
  if config.shouldStop latest.token then
    inference.finishEarly
    return generated
  generateRemaining inference head config.stopTokens (config.maxNewTokens - 1)
    latest generated onToken

/--
Autoregressively feed each sampled token back into the same live persistent launch and publish
generated tokens through `onToken`. The prompt is consumed causally; only generated tokens are
reported to the callback and returned. A configured stop token finishes the partially filled
launch normally after publishing that token.
-/
private def Checkpoint.generateStreamingWithHead (checkpoint : @& Checkpoint)
    (head : GenerationHead) (prompt : Array UInt32)
    (config : GenerationConfig) (onToken : TokenEvent → IO Unit) : IO (Array UInt32) := do
  if config.maxNewTokens == 0 then
    return #[]
  let positions ← validateGenerationRequest prompt config
  let inference ← checkpoint.startInference positions
  try
    generatePromptOnStream inference head prompt 0 false config onToken
  catch error =>
    inference.abort
    throw error

private structure PrefixCacheState where
  tokens : Array UInt32 := #[]
  valid : Bool := false

private def hasExactPrefix (tokens candidatePrefix : @& Array UInt32) : Bool :=
  candidatePrefix.size ≤ tokens.size &&
    tokens.extract 0 candidatePrefix.size == candidatePrefix

private def newGenerationSession (loaded : LoadedCheckpoint) : IO GenerationSession := do
  let arena ← allocateInferenceArena
  let cache ← IO.mkRef ({} : PrefixCacheState)
  return {
    generateStreamingWith := fun prompt cachePrefixTokens config onToken => do
      if config.maxNewTokens == 0 then
        return #[]
      let positions ← validateGenerationRequest prompt config
      unless 0 < cachePrefixTokens && cachePrefixTokens < prompt.size do
        throw <| IO.userError
          "Qwen3.8 cache prefix must be nonempty and end before the generation suffix"
      uploadPrompt arena prompt
      let previous ← cache.get
      let reusePrefix :=
        previous.valid && previous.tokens.size ≤ cachePrefixTokens &&
          hasExactPrefix prompt previous.tokens
      let startPosition := if reusePrefix then previous.tokens.size else 0
      if reusePrefix then
        restoreInferenceState arena loaded.stream
      if startPosition < cachePrefixTokens then
        let prefixStream ← startLoadedInferenceStream loaded.layout loaded.weights loaded.lmHead
          arena.buffers loaded.stream startPosition cachePrefixTokens (!reusePrefix)
        try
          prefixStream.prefillBatch (cachePrefixTokens - startPosition)
          prefixStream.finishEarly
        catch error =>
          prefixStream.abort
          cache.set {}
          throw error
        try
          snapshotInferenceState arena loaded.stream
        catch error =>
          cache.set {}
          throw error
      cache.set {
        tokens := prompt.extract 0 cachePrefixTokens
        valid := true
      }
      let inference ← startLoadedInferenceStream loaded.layout loaded.weights loaded.lmHead
        arena.buffers loaded.stream cachePrefixTokens positions false
      try
        generatePromptOnStream inference .mxfp8Greedy prompt cachePrefixTokens true config onToken
      catch error =>
        inference.abort
        throw error
  }

/--
Load one immutable checkpoint, one reusable full-context arena, and a canonical-prefix cache for
the serialized chat-serving path.
-/
def loadServingCheckpoint (modelDirectory : System.FilePath) : IO ServingCheckpoint := do
  let loaded ← loadCheckpointData modelDirectory
  return {
    checkpoint := checkpointOfLoaded loaded
    generation := ← newGenerationSession loaded
  }

/-- Generate through the production MXFP8 head. -/
def Checkpoint.generateStreamingWith (checkpoint : @& Checkpoint) (prompt : Array UInt32)
    (config : GenerationConfig) (onToken : TokenEvent → IO Unit) : IO (Array UInt32) :=
  checkpoint.generateStreamingWithHead .mxfp8Greedy prompt config onToken

/-- Generate through the published BF16 head as an untimed numerical reference. -/
def Checkpoint.generateStreamingReferenceWith (checkpoint : @& Checkpoint)
    (prompt : Array UInt32) (config : GenerationConfig)
    (onToken : TokenEvent → IO Unit) : IO (Array UInt32) :=
  checkpoint.generateStreamingWithHead .bf16Reference prompt config onToken

/-- Generate a fixed maximum number of tokens without an early-stop token. -/
def Checkpoint.generateStreaming (checkpoint : @& Checkpoint) (prompt : Array UInt32)
    (maxNewTokens : Nat) (onToken : TokenEvent → IO Unit) : IO (Array UInt32) :=
  checkpoint.generateStreamingWith prompt { maxNewTokens } onToken

/-- Generate a fixed number of untimed BF16 reference tokens. -/
def Checkpoint.generateStreamingReference (checkpoint : @& Checkpoint)
    (prompt : Array UInt32) (maxNewTokens : Nat)
    (onToken : TokenEvent → IO Unit) : IO (Array UInt32) :=
  checkpoint.generateStreamingReferenceWith prompt { maxNewTokens } onToken

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
  match recurrentStateSchedule with
  | .phasedRowMajor => pure ()
  | .warpResidentColumnMajor =>
    let warpWidth := 32
    let warpsPerBlock := blockThreadCountNat / warpWidth
    unless blockThreadCountNat % warpWidth == 0 &&
        linearHeadDimension == 4 * warpWidth && warpsPerBlock > 0 do
      throw <| IO.userError
        "Qwen3.8 warp-resident state schedule does not tile the configured block and head"
  match recurrentGridSchedule with
  | .oneBlockPerHead => pure ()
  | .fullResidentGrid =>
    unless gridBlockCountNat % linearHeadCount == 0 && recurrentGroupCountNat > 0 &&
        recurrentGroupCountNat * recurrentWarpsPerBlockNat <= linearHeadDimension do
      throw <| IO.userError
        "Qwen3.8 recurrent full-grid schedule does not tile the value heads"
  match PersistentLayout.forTrainingTokens 0 with
  | .ok _ => throw <| IO.userError "Qwen3.8 persistent layout accepted an empty sequence"
  | .error _ => pure ()
  match PersistentLayout.forTrainingTokens (maxTrainingSequenceTokens + 1) with
  | .ok _ => throw <| IO.userError "Qwen3.8 training layout accepted an over-limit sequence"
  | .error _ => pure ()
  match PersistentLayout.forInferenceTokens (maxSequenceTokens + 1) with
  | .ok _ => throw <| IO.userError "Qwen3.8 inference layout accepted an over-context sequence"
  | .error _ => pure ()
  let trainingLayout ← match PersistentLayout.forTrainingTokens 2 with
    | .ok layout => pure layout
    | .error error => throw <| IO.userError error
  unless trainingLayout.outputTokenCount == 2 &&
      trainingLayout.retainsPositionOutputs &&
      trainingLayout.descriptorCount == 2 * descriptorsPerTokenNat &&
      trainingLayout.countElements == 2 * descriptorsPerTokenNat &&
      trainingLayout.queueCapacity == 2 * descriptorsPerTokenNat &&
      trainingLayout.logitsElements == 2 * vocabularySize &&
      trainingLayout.keyValueCacheElements ==
        fullAttentionLayerCount * 2 * fullKeyValueProjectionSize do
    throw <| IO.userError "Qwen3.8 training layout derived inconsistent state extents"
  let inferenceLayout ← match PersistentLayout.forInferenceTokens 257 with
    | .ok layout => pure layout
    | .error error => throw <| IO.userError error
  unless inferenceLayout.outputTokenCount == 1 &&
      !inferenceLayout.retainsPositionOutputs &&
      inferenceLayout.descriptorCount == 257 * descriptorsPerTokenNat &&
      inferenceLayout.countElements == descriptorsPerTokenNat &&
      inferenceLayout.queueCapacity == descriptorsPerTokenNat &&
      inferenceLayout.logitsElements == vocabularySize &&
      inferenceLayout.keyValueCacheElements ==
        fullAttentionLayerCount * 257 * fullKeyValueProjectionSize do
    throw <| IO.userError "Qwen3.8 inference layout derived inconsistent state extents"

def main : IO Unit := do
  checkPersistentLayout
  let some modelDirectory ← IO.getEnv "QWEN_MODEL_DIR" | do
    IO.println "Skipping real Qwen3.8 persistent inference: QWEN_MODEL_DIR is not set"
    return
  let tokens ← parseIds "QWEN38_TOKEN_IDS"
    ((← IO.getEnv "QWEN38_TOKEN_IDS").getD "1234,5678")
  let targets ← parseIds "QWEN38_TARGET_IDS"
    ((← IO.getEnv "QWEN38_TARGET_IDS").getD "1234,5678")
  unless tokens.size == targets.size do
    throw <| IO.userError "QWEN38_TOKEN_IDS and QWEN38_TARGET_IDS must have equal length"
  unless tokens.size <= maxTrainingSequenceTokens do
    throw <| IO.userError
      s!"Qwen3.8 retained sequence exceeds training limit {maxTrainingSequenceTokens}"
  let schema ← introspect modelDirectory
  unless schema.isDirectory do
    throw <| IO.userError "Qwen3.8-27B must be loaded from its sharded checkpoint directory"
  let manifest ← schema.payloadManifest
  let layout ← readTextLayout manifest
  let stream ← Cuda.Stream.default
  let loadedWeights ← loadWeights manifest layout stream
  let weights ← resolveWeightTable loadedWeights layout
  let lmHead ← quantizeLmHead loadedWeights layout stream
  let persistentLayout ← match PersistentLayout.forTrainingTokens tokens.size with
    | .ok layout => pure layout
    | .error error => throw <| IO.userError error
  let buffers ← allocateBuffers persistentLayout
  let start ← IO.monoMsNow
  let first ← runOnce layout weights lmHead buffers tokens targets stream
  let elapsed := (← IO.monoMsNow) - start
  if (← IO.getEnv "QWEN38_REPEAT_DETERMINISM") == some "1" then
    let second ← runOnce layout weights lmHead buffers tokens targets stream
    unless first == second do
      throw <| IO.userError "repeated Qwen3.8 multi-token megakernel runs differ"
  if let some outputPath ← IO.getEnv "QWEN38_LOGITS_OUTPUT" then
    IO.FS.writeBinFile outputPath first.logits
  for position in [:tokens.size] do
    IO.println s!"Qwen3.8 position={position}, token={tokens[position]!}, sampled={readUInt32 first.sampled position}, loss_bits={readUInt32 first.loss position}"
  IO.println s!"Lean Qwen3.8 real-checkpoint {tokens.size}-token persistent inference ok; elapsed_ms={elapsed}"

end Cuda.Qwen36.Megakernel
