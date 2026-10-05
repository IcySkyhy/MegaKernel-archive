/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda

/-!
# Qwen3.6-27B gated full-attention decode core

This executable covers the nonlinear/stateful part of each fourth decoder layer at the exact
checkpoint geometry: 24 query heads, four key/value heads, 256 channels per head, six-way grouped
query attention, head-local `(1 + weight)` RMSNorm, 64-channel partial RoPE, a BF16 KV cache, and
the sigmoid output gate carried in the doubled query projection.

Projection outputs and precomputed RoPE cosine/sine tensors are explicit inputs, matching the
model boundary. Four token descriptors execute in one cooperative-grid launch. A two-launch-per-
token route and an independent host Lean replay are correctness oracles.
-/

namespace Qwen36Attention

private def tokenCount : Nat := 4
private def queryHeadCount : Nat := 24
private def keyValueHeadCount : Nat := 4
private def headDimensionNat : Nat := 256
private def rotaryDimensionNat : Nat := 64
private def rotaryHalfNat : Nat := rotaryDimensionNat / 2
private def queryProjectionHeadStrideNat : Nat := headDimensionNat * 2
private def queryElementCount : Nat := queryHeadCount * headDimensionNat
private def keyValueElementCount : Nat := keyValueHeadCount * headDimensionNat

private def queryHeads : UInt32 := queryHeadCount.toUInt32
private def keyValueHeads : UInt32 := keyValueHeadCount.toUInt32
private def headDimension : UInt32 := headDimensionNat.toUInt32
private def rotaryDimension : UInt32 := rotaryDimensionNat.toUInt32
private def rotaryHalf : UInt32 := rotaryHalfNat.toUInt32
private def queryProjectionHeadStride : UInt32 := queryProjectionHeadStrideNat.toUInt32
private def queryGroups : UInt32 := queryHeads / keyValueHeads
private def zero : Float32 := Float32.ofBits 0
private def one : Float32 := Float32.ofBits 0x3f800000
private def rmsEpsilon : Float32 := Float32.ofBits 0x358637bd
private def inverseHeadDimension : Float32 := Float32.ofBits 0x3b800000
private def attentionScale : Float32 := Float32.ofBits 0x3d800000

private def reductionBytes : Nat := 8 * 4
private def vectorOffset : Nat := reductionBytes
private def vectorBytes : Nat := headDimensionNat * 4
private def scoresOffset : Nat := vectorOffset + vectorBytes
private def scoresBytes : Nat := tokenCount * 4
private def sharedBytes : Nat := scoresOffset + scoresBytes

structure TokenItem where
  token : UInt32
  deriving Cuda.POD

@[struct] structure Inputs where
  projectedQueryGate : Cuda.DevicePtr Cuda.BFloat16
  projectedKey : Cuda.DevicePtr Cuda.BFloat16
  projectedValue : Cuda.DevicePtr Cuda.BFloat16
  queryNormWeight : Cuda.DevicePtr Cuda.BFloat16
  keyNormWeight : Cuda.DevicePtr Cuda.BFloat16
  ropeCos : Cuda.DevicePtr Cuda.BFloat16
  ropeSin : Cuda.DevicePtr Cuda.BFloat16

@[struct] structure Buffers where
  query : Cuda.DevicePtr Cuda.BFloat16
  keyCache : Cuda.DevicePtr Cuda.BFloat16
  valueCache : Cuda.DevicePtr Cuda.BFloat16
  output : Cuda.DevicePtr Cuda.BFloat16

@[always_inline]
private def queryProjectionIndex (token head channel : UInt32) : UInt32 :=
  (token * queryHeads + head) * queryProjectionHeadStride + channel

@[always_inline]
private def queryIndex (token head channel : UInt32) : UInt32 :=
  (token * queryHeads + head) * headDimension + channel

@[always_inline]
private def keyValueIndex (token head channel : UInt32) : UInt32 :=
  (token * keyValueHeads + head) * headDimension + channel

@[always_inline]
private def ropeIndex (token frequency : UInt32) : UInt32 :=
  token * rotaryHalf + frequency

@[always_inline]
private def loadBFloat (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) : Cuda.DeviceM Float32 := do
  return (← Cuda.loadBFloat16 pointer index.toUSize).toFloat32

@[always_inline]
private def storeBFloat (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) (value : Float32) : Cuda.DeviceM Unit :=
  Cuda.storeBFloat16 pointer index.toUSize (Cuda.BFloat16.ofFloat32 value)

@[always_inline, convergent]
private def normalizeVector (source weight : Cuda.DevicePtr Cuda.BFloat16)
    (sourceBase thread : UInt32) (scratch : Cuda.Collective.BlockScratch 8)
    (vector : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let value ← loadBFloat source (sourceBase + thread)
  let squareSum ← Cuda.Collective.blockSum scratch (value * value)
  let inverse ← Cuda.fastRsqrt (squareSum * inverseHeadDimension + rmsEpsilon)
  let normWeight ← loadBFloat weight thread
  Cuda.storeFloat32 vector thread.toUSize (value * inverse * (one + normWeight))
  Cuda.blockSync

@[always_inline]
private def applyPartialRope (token thread : UInt32) (inputs : Inputs)
    (vector : Cuda.DevicePtr Float32) : Cuda.DeviceM Float32 := do
  let value ← Cuda.loadFloat32 vector thread.toUSize
  if thread < rotaryDimension then
    let frequency := thread % rotaryHalf
    let partnerIndex := if thread < rotaryHalf then thread + rotaryHalf else thread - rotaryHalf
    let partner ← Cuda.loadFloat32 vector partnerIndex.toUSize
    let cosine ← loadBFloat inputs.ropeCos (ropeIndex token frequency)
    let sine ← loadBFloat inputs.ropeSin (ropeIndex token frequency)
    if thread < rotaryHalf then
      return Cuda.fma value cosine (-(partner * sine))
    else
      return Cuda.fma partner sine (value * cosine)
  else
    return value

@[always_inline, convergent]
private def runPrepare (token head thread : UInt32) (inputs : Inputs) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  let vector ← Cuda.dynamicShared (α := Float32) vectorOffset.toUSize
  let queryBase := queryProjectionIndex token head 0
  normalizeVector inputs.projectedQueryGate inputs.queryNormWeight queryBase thread scratch vector
  let queryValue ← applyPartialRope token thread inputs vector
  storeBFloat buffers.query (queryIndex token head thread) queryValue
  Cuda.blockSync
  if head < keyValueHeads then
    let keyBase := keyValueIndex token head 0
    normalizeVector inputs.projectedKey inputs.keyNormWeight keyBase thread scratch vector
    let keyValue ← applyPartialRope token thread inputs vector
    storeBFloat buffers.keyCache (keyValueIndex token head thread) keyValue
    let value ← loadBFloat inputs.projectedValue (keyValueIndex token head thread)
    storeBFloat buffers.valueCache (keyValueIndex token head thread) value

@[always_inline]
private partial def scoreDot (token queryHead keyValueHead past channel : UInt32)
    (buffers : Buffers) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if channel < headDimension then
    let query ← loadBFloat buffers.query (queryIndex token queryHead channel)
    let key ← loadBFloat buffers.keyCache (keyValueIndex past keyValueHead channel)
    scoreDot token queryHead keyValueHead past (channel + 256) buffers
      (Cuda.fma query key accumulator)
  else
    return accumulator

@[always_inline, convergent]
private partial def writeScores (token queryHead keyValueHead past thread : UInt32)
    (buffers : Buffers) (scratch : Cuda.Collective.BlockScratch 8)
    (scores : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  if past <= token then
    let localScore ← scoreDot token queryHead keyValueHead past thread buffers zero
    let score ← Cuda.Collective.blockSum scratch localScore
    if thread == 0 then
      Cuda.storeFloat32 scores past.toUSize (score * attentionScale)
    writeScores token queryHead keyValueHead (past + 1) thread buffers scratch scores

@[always_inline]
private partial def scoreMaximum (scores : Cuda.DevicePtr Float32) (limit index : UInt32)
    (maximum : Float32) : Cuda.DeviceM Float32 := do
  if index < limit then
    let value ← Cuda.loadFloat32 scores index.toUSize
    scoreMaximum scores limit (index + 1) (if value > maximum then value else maximum)
  else
    return maximum

@[always_inline]
private partial def exponentialSum (scores : Cuda.DevicePtr Float32) (limit index : UInt32)
    (maximum accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < limit then
    let score ← Cuda.loadFloat32 scores index.toUSize
    let exponential ← Cuda.fastExp (score - maximum)
    Cuda.storeFloat32 scores index.toUSize exponential
    exponentialSum scores limit (index + 1) maximum (accumulator + exponential)
  else
    return accumulator

@[always_inline]
private partial def normalizeProbabilities (scores : Cuda.DevicePtr Float32)
    (limit index : UInt32) (inverse : Float32) : Cuda.DeviceM Unit := do
  if index < limit then
    let exponential ← Cuda.loadFloat32 scores index.toUSize
    let probability := (Cuda.BFloat16.ofFloat32 (exponential * inverse)).toFloat32
    Cuda.storeFloat32 scores index.toUSize probability
    normalizeProbabilities scores limit (index + 1) inverse

@[always_inline]
private partial def weightedValue (keyValueHead channel past limit : UInt32)
    (buffers : Buffers) (scores : Cuda.DevicePtr Float32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if past < limit then
    let probability ← Cuda.loadFloat32 scores past.toUSize
    let value ← loadBFloat buffers.valueCache (keyValueIndex past keyValueHead channel)
    weightedValue keyValueHead channel (past + 1) limit buffers scores
      (Cuda.fma probability value accumulator)
  else
    return accumulator

@[always_inline]
private def deviceSigmoid (value : Float32) : Cuda.DeviceM Float32 := do
  return one / (one + (← Cuda.fastExp (-value)))

@[always_inline, convergent]
private def runAttention (token head thread : UInt32) (inputs : Inputs) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  let scores ← Cuda.dynamicShared (α := Float32) scoresOffset.toUSize
  let keyValueHead := head / queryGroups
  writeScores token head keyValueHead 0 thread buffers scratch scores
  if thread == 0 then
    let limit := token + 1
    let first ← Cuda.loadFloat32 scores 0
    let maximum ← scoreMaximum scores limit 1 first
    let denominator ← exponentialSum scores limit 0 maximum zero
    normalizeProbabilities scores limit 0 (one / denominator)
  Cuda.blockSync
  let attention ← weightedValue keyValueHead thread 0 (token + 1) buffers scores zero
  let roundedAttention := (Cuda.BFloat16.ofFloat32 attention).toFloat32
  let gateInput ← loadBFloat inputs.projectedQueryGate
    (queryProjectionIndex token head (headDimension + thread))
  let gate := (Cuda.BFloat16.ofFloat32 (← deviceSigmoid gateInput)).toFloat32
  storeBFloat buffers.output (queryIndex token head thread) (roundedAttention * gate)

@[cuda_kernel]
def qwen36AttentionPrepareStep (token : UInt32)
    (projectedQueryGate projectedKey projectedValue queryNormWeight keyNormWeight ropeCos ropeSin
      query keyCache valueCache output : Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runPrepare token (← Cuda.blockIdxX) (← Cuda.threadIdxX) {
    projectedQueryGate, projectedKey, projectedValue, queryNormWeight, keyNormWeight, ropeCos, ropeSin
  } { query, keyCache, valueCache, output } shared

@[cuda_kernel]
def qwen36AttentionComputeStep (token : UInt32)
    (projectedQueryGate projectedKey projectedValue queryNormWeight keyNormWeight ropeCos ropeSin
      query keyCache valueCache output : Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runAttention token (← Cuda.blockIdxX) (← Cuda.threadIdxX) {
    projectedQueryGate, projectedKey, projectedValue, queryNormWeight, keyNormWeight, ropeCos, ropeSin
  } { query, keyCache, valueCache, output } shared

@[cuda_grid_persistent]
def qwen36Attention (item : TokenItem)
    (projectedQueryGate projectedKey projectedValue queryNormWeight keyNormWeight ropeCos ropeSin
      query keyCache valueCache output : Cuda.DevicePtr Cuda.BFloat16)
    (counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let shared ← Cuda.dynamicShared (α := Float32)
  let inputs : Inputs := {
    projectedQueryGate, projectedKey, projectedValue, queryNormWeight, keyNormWeight, ropeCos, ropeSin
  }
  let buffers : Buffers := { query, keyCache, valueCache, output }
  runPrepare item.token block thread inputs buffers shared
  Cuda.gridSync
  runAttention item.token block thread inputs buffers shared
  Cuda.gridSync
  if block == 0 && thread == 0 then
    Cuda.atomicAddUInt32At_ counts item.token 1

private structure InputBuffers where
  projectedQueryGate : Cuda.Buffer Cuda.BFloat16
  projectedKey : Cuda.Buffer Cuda.BFloat16
  projectedValue : Cuda.Buffer Cuda.BFloat16
  queryNormWeight : Cuda.Buffer Cuda.BFloat16
  keyNormWeight : Cuda.Buffer Cuda.BFloat16
  ropeCos : Cuda.Buffer Cuda.BFloat16
  ropeSin : Cuda.Buffer Cuda.BFloat16

private structure RunBuffers where
  query : Cuda.Buffer Cuda.BFloat16
  keyCache : Cuda.Buffer Cuda.BFloat16
  valueCache : Cuda.Buffer Cuda.BFloat16
  output : Cuda.Buffer Cuda.BFloat16
  counts : Cuda.Buffer UInt32

private structure Oracle where
  keyCache : Array Float32
  valueCache : Array Float32
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

private def zeroBytes (count : Nat) : ByteArray :=
  List.replicate count 0 |>.toByteArray

private def makeBFloatData (count : Nat) (value : Nat → Float32) : ByteArray :=
  (List.range count).foldl (init := ByteArray.empty) fun bytes index =>
    pushUInt16 bytes (Cuda.BFloat16.ofFloat32 (value index)).bits

private def signedValue (index modulus center divisor : Nat) : Float32 :=
  (Int32.ofInt (Int.ofNat (index % modulus) - Int.ofNat center)).toFloat32 /
    divisor.toUInt32.toFloat32

private def queryProjectionValue (token head channel : Nat) : Float32 :=
  signedValue (token * 103 + head * 29 + channel * 7 + 3) 37 18 24

private def keyProjectionValue (token head channel : Nat) : Float32 :=
  signedValue (token * 73 + head * 31 + channel * 11 + 5) 41 20 28

private def valueProjectionValue (token head channel : Nat) : Float32 :=
  signedValue (token * 59 + head * 23 + channel * 13 + 9) 43 21 20

private def queryNormValue (channel : Nat) : Float32 :=
  signedValue (channel * 5 + 1) 13 6 128

private def keyNormValue (channel : Nat) : Float32 :=
  signedValue (channel * 7 + 3) 17 8 128

private def ropeAngle (token frequency : Nat) : Float32 :=
  (token * (frequency + 1)).toUInt32.toFloat32 / Float32.ofBits 0x42400000

private def rounded (value : Float32) : Float32 :=
  (Cuda.BFloat16.ofFloat32 value).toFloat32

private def hostSigmoid (value : Float32) : Float32 :=
  one / (one + Float32.exp (-value))

private def initializeInputs (stream : Cuda.Stream) : IO InputBuffers := do
  let projectedQueryGate ← Cuda.Buffer.alloc Cuda.BFloat16
    (tokenCount * queryHeadCount * queryProjectionHeadStrideNat).toUSize
  let projectedKey ← Cuda.Buffer.alloc Cuda.BFloat16
    (tokenCount * keyValueElementCount).toUSize
  let projectedValue ← Cuda.Buffer.alloc Cuda.BFloat16
    (tokenCount * keyValueElementCount).toUSize
  let queryNormWeight ← Cuda.Buffer.alloc Cuda.BFloat16 headDimensionNat.toUSize
  let keyNormWeight ← Cuda.Buffer.alloc Cuda.BFloat16 headDimensionNat.toUSize
  let ropeCos ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * rotaryHalfNat).toUSize
  let ropeSin ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * rotaryHalfNat).toUSize
  projectedQueryGate.copyFrom
    (makeBFloatData (tokenCount * queryHeadCount * queryProjectionHeadStrideNat) fun index =>
      let token := index / (queryHeadCount * queryProjectionHeadStrideNat)
      let rest := index % (queryHeadCount * queryProjectionHeadStrideNat)
      queryProjectionValue token (rest / queryProjectionHeadStrideNat)
        (rest % queryProjectionHeadStrideNat)) stream
  projectedKey.copyFrom (makeBFloatData (tokenCount * keyValueElementCount) fun index =>
    let token := index / keyValueElementCount
    let rest := index % keyValueElementCount
    keyProjectionValue token (rest / headDimensionNat) (rest % headDimensionNat)) stream
  projectedValue.copyFrom (makeBFloatData (tokenCount * keyValueElementCount) fun index =>
    let token := index / keyValueElementCount
    let rest := index % keyValueElementCount
    valueProjectionValue token (rest / headDimensionNat) (rest % headDimensionNat)) stream
  queryNormWeight.copyFrom (makeBFloatData headDimensionNat queryNormValue) stream
  keyNormWeight.copyFrom (makeBFloatData headDimensionNat keyNormValue) stream
  ropeCos.copyFrom (makeBFloatData (tokenCount * rotaryHalfNat) fun index =>
    Float32.cos (ropeAngle (index / rotaryHalfNat) (index % rotaryHalfNat))) stream
  ropeSin.copyFrom (makeBFloatData (tokenCount * rotaryHalfNat) fun index =>
    Float32.sin (ropeAngle (index / rotaryHalfNat) (index % rotaryHalfNat))) stream
  return {
    projectedQueryGate, projectedKey, projectedValue, queryNormWeight, keyNormWeight, ropeCos,
    ropeSin
  }

private def initializeRunBuffers (stream : Cuda.Stream) : IO RunBuffers := do
  let query ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * queryElementCount).toUSize
  let keyCache ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * keyValueElementCount).toUSize
  let valueCache ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * keyValueElementCount).toUSize
  let output ← Cuda.Buffer.alloc Cuda.BFloat16 (tokenCount * queryElementCount).toUSize
  let counts ← Cuda.Buffer.alloc UInt32 tokenCount.toUSize
  query.copyFrom (zeroBytes (tokenCount * queryElementCount * 2)) stream
  keyCache.copyFrom (zeroBytes (tokenCount * keyValueElementCount * 2)) stream
  valueCache.copyFrom (zeroBytes (tokenCount * keyValueElementCount * 2)) stream
  output.copyFrom (zeroBytes (tokenCount * queryElementCount * 2)) stream
  counts.copyFrom (zeroBytes (tokenCount * 4)) stream
  return { query, keyCache, valueCache, output, counts }

private def launchConfig : Cuda.LaunchConfig := {
  grid := { x := queryHeads }
  block := { x := 256 }
  sharedMemoryBytes := sharedBytes.toUSize
  blockArenaBytes := 16 * 1024
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := queryHeads }
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
    waitKernel "Qwen3.6 attention prepare" <|
      ← qwen36AttentionPrepareStep.launch launchConfig token.toUInt32 inputs.projectedQueryGate
        inputs.projectedKey inputs.projectedValue inputs.queryNormWeight inputs.keyNormWeight
        inputs.ropeCos inputs.ropeSin buffers.query buffers.keyCache buffers.valueCache buffers.output
    waitKernel "Qwen3.6 attention compute" <|
      ← qwen36AttentionComputeStep.launch launchConfig token.toUInt32 inputs.projectedQueryGate
        inputs.projectedKey inputs.projectedValue inputs.queryNormWeight inputs.keyNormWeight
        inputs.ropeCos inputs.ropeSin buffers.query buffers.keyCache buffers.valueCache buffers.output

private def runPersistent (inputs : InputBuffers) (buffers : RunBuffers) : IO Unit := do
  let handle ← qwen36Attention.start persistentConfig inputs.projectedQueryGate inputs.projectedKey
    inputs.projectedValue inputs.queryNormWeight inputs.keyNormWeight inputs.ropeCos inputs.ropeSin
    buffers.query buffers.keyCache buffers.valueCache buffers.output buffers.counts
  for token in [:tokenCount] do
    qwen36Attention.enqueue handle { token := token.toUInt32 }
  handle.shutdown
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"Qwen3.6 attention megakernel failed: {repr status}"

private def hostNormalizeRope (token : Nat) (raw weight : Nat → Float32) : Array Float32 := Id.run do
  let raw := (List.range headDimensionNat).map (fun channel => rounded (raw channel)) |>.toArray
  let mut squareSum := zero
  for value in raw do
    squareSum := Cuda.fma value value squareSum
  let inverse := one / Float32.sqrt (squareSum * inverseHeadDimension + rmsEpsilon)
  let normalized := (List.range headDimensionNat).map (fun channel =>
    raw[channel]! * inverse * (one + rounded (weight channel))) |>.toArray
  return (List.range headDimensionNat).map (fun channel =>
    if channel < rotaryDimensionNat then
      let frequency := channel % rotaryHalfNat
      let partnerIndex := if channel < rotaryHalfNat then channel + rotaryHalfNat
        else channel - rotaryHalfNat
      let cosine := rounded (Float32.cos (ropeAngle token frequency))
      let sine := rounded (Float32.sin (ropeAngle token frequency))
      let value := if channel < rotaryHalfNat then
          Cuda.fma normalized[channel]! cosine (-(normalized[partnerIndex]! * sine))
        else
          Cuda.fma normalized[partnerIndex]! sine (normalized[channel]! * cosine)
      rounded value
    else
      rounded normalized[channel]!) |>.toArray

private def hostOracle : Oracle := Id.run do
  let mut query := Array.replicate (tokenCount * queryElementCount) zero
  let mut keyCache := Array.replicate (tokenCount * keyValueElementCount) zero
  let mut valueCache := Array.replicate (tokenCount * keyValueElementCount) zero
  let mut output := Array.replicate (tokenCount * queryElementCount) zero
  for token in [:tokenCount] do
    for head in [:queryHeadCount] do
      let vector := hostNormalizeRope token
        (queryProjectionValue token head) queryNormValue
      for channel in [:headDimensionNat] do
        query := query.set! ((token * queryHeadCount + head) * headDimensionNat + channel)
          vector[channel]!
    for head in [:keyValueHeadCount] do
      let vector := hostNormalizeRope token (keyProjectionValue token head) keyNormValue
      for channel in [:headDimensionNat] do
        keyCache := keyCache.set! ((token * keyValueHeadCount + head) * headDimensionNat + channel)
          vector[channel]!
        valueCache := valueCache.set!
          ((token * keyValueHeadCount + head) * headDimensionNat + channel)
          (rounded (valueProjectionValue token head channel))
    for head in [:queryHeadCount] do
      let keyValueHead := head / (queryHeadCount / keyValueHeadCount)
      let mut scores := Array.replicate (token + 1) zero
      let mut maximum := -Float32.ofBits 0x7f800000
      for past in [:token + 1] do
        let mut score := zero
        for channel in [:headDimensionNat] do
          score := Cuda.fma query[(token * queryHeadCount + head) * headDimensionNat + channel]!
            keyCache[(past * keyValueHeadCount + keyValueHead) * headDimensionNat + channel]!
            score
        score := score * attentionScale
        scores := scores.set! past score
        if score > maximum then maximum := score
      let mut denominator := zero
      for past in [:token + 1] do
        let exponential := Float32.exp (scores[past]! - maximum)
        scores := scores.set! past exponential
        denominator := denominator + exponential
      for past in [:token + 1] do
        scores := scores.set! past (rounded (scores[past]! / denominator))
      for channel in [:headDimensionNat] do
        let mut attention := zero
        for past in [:token + 1] do
          attention := Cuda.fma scores[past]!
            valueCache[(past * keyValueHeadCount + keyValueHead) * headDimensionNat + channel]!
            attention
        let roundedAttention := rounded attention
        let gateInput := rounded (queryProjectionValue token head (headDimensionNat + channel))
        let gate := rounded (hostSigmoid gateInput)
        output := output.set! ((token * queryHeadCount + head) * headDimensionNat + channel)
          (rounded (roundedAttention * gate))
  return { keyCache, valueCache, output }

private def requireEqual (label : String) (actual expected : ByteArray) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError s!"persistent and separate-launch {label} differ"

private def requireClose (label : String) (actual : ByteArray) (expected : Array Float32)
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
  requireEqual "attention key cache" (← persistent.keyCache.copyTo stream)
    (← separate.keyCache.copyTo stream)
  requireEqual "attention value cache" (← persistent.valueCache.copyTo stream)
    (← separate.valueCache.copyTo stream)
  requireEqual "attention output" (← persistent.output.copyTo stream)
    (← separate.output.copyTo stream)
  let counts ← persistent.counts.copyTo stream
  for token in [:tokenCount] do
    unless readUInt32 counts token == 1 do
      throw <| IO.userError s!"Qwen3.6 attention token {token} did not execute exactly once"
  let oracle := hostOracle
  requireClose "attention key cache" (← persistent.keyCache.copyTo stream) oracle.keyCache
    (Float32.ofBits 0x3b800000)
  requireClose "attention value cache" (← persistent.valueCache.copyTo stream) oracle.valueCache zero
  requireClose "attention output" (← persistent.output.copyTo stream) oracle.output
    (Float32.ofBits 0x3c800000)
  IO.println "Lean Qwen3.6 gated full-attention KV-cache megakernel ok"

end Qwen36Attention

def main : IO Unit := Qwen36Attention.main
