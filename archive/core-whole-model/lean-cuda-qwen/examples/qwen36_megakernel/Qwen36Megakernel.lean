/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda

/-!
# Qwen3.6-27B recurrent Gated DeltaNet megakernel

This is the first architecture-specific slice of a plain-Lean CUDA implementation of
`Qwen/Qwen3.6-27B`. The checkpoint has 64 language layers in a repeating three-linear/one-full
attention schedule. Its 48 linear-attention layers use 16 query/key heads, 48 value heads, and
128 channels per head.

`qwen36Decode` keeps the exact recurrent Gated DeltaNet state on the device and consumes multiple
token descriptors in one cooperative-grid persistent launch. Each of the 48 resident blocks owns
one value head. Query and key heads remain compact; three value-head blocks share each projected
query/key head exactly as the reference model's `repeat_interleave(3)` does.

For every token and head, the Lean kernel performs the reference recurrence in FP32:

* L2-normalize query and key and scale query by `1 / sqrt(128)`;
* decay the `128 x 128` recurrent state by `exp(g)`;
* compute `delta = beta * (value - key^T state)`;
* apply the rank-one update `state += key * delta^T`;
* publish `query^T state`.

The executable compares the persistent launch byte-for-byte with three ordinary launches and
checks both routes against an independent host Lean replay. Projection, causal convolution,
gated RMSNorm, output projection, the 16 full-attention layers, MLPs, embedding/LM head, and all
backward/update phases remain subsequent slices; this file does not claim full-model inference or
training yet.
-/

namespace Qwen36

private def tokenCount : Nat := 3
private def keyHeadCount : Nat := 16
private def valueHeadCount : Nat := 48
private def headDimensionNat : Nat := 128
private def stateElementsPerHeadNat : Nat := headDimensionNat * headDimensionNat
private def stateElementCount : Nat := valueHeadCount * stateElementsPerHeadNat
private def outputElementCount : Nat := tokenCount * valueHeadCount * headDimensionNat

private def keyHeads : UInt32 := keyHeadCount.toUInt32
private def valueHeads : UInt32 := valueHeadCount.toUInt32
private def headDimension : UInt32 := headDimensionNat.toUInt32
private def valueHeadsPerKeyHead : UInt32 := valueHeads / keyHeads
private def stateElementsPerHead : UInt32 := headDimension * headDimension

/-- Exact Float32 encoding of `1.0e-6`, matching the reference L2-normalization epsilon. -/
private def normalizationEpsilon : Float32 := Float32.ofBits 0x358637bd

/-- Exact Float32 encoding of `1 / sqrt(128)`. -/
private def queryScale : Float32 := Float32.ofBits 0x3db504f3

/-- Dynamic shared-memory layout: reduction scratch followed by Q, K, and delta vectors. -/
private def reductionBytes : Nat := 8 * 4
private def vectorBytes : Nat := headDimensionNat * 4
private def queryOffset : Nat := reductionBytes
private def keyOffset : Nat := queryOffset + vectorBytes
private def deltaOffset : Nat := keyOffset + vectorBytes
private def decodeSharedBytes : Nat := deltaOffset + vectorBytes

/-- One token consumed by the resident decode grid. -/
structure DecodeItem where
  token : UInt32
  deriving Cuda.POD

/-- Device buffers for one linear-attention layer's recurrent core. -/
@[struct] structure Buffers where
  /-- Compact BF16 query projection `[tokens, 16, 128]`. -/
  query : Cuda.DevicePtr Cuda.BFloat16
  /-- Compact BF16 key projection `[tokens, 16, 128]`. -/
  key : Cuda.DevicePtr Cuda.BFloat16
  /-- BF16 value projection `[tokens, 48, 128]`. -/
  value : Cuda.DevicePtr Cuda.BFloat16
  /-- FP32 logarithmic decay inputs `[tokens, 48]`. -/
  decayLog : Cuda.DevicePtr Float32
  /-- FP32 delta-rule interpolation coefficients `[tokens, 48]`. -/
  beta : Cuda.DevicePtr Float32
  /-- Mutable FP32 recurrent state `[48, 128, 128]`. -/
  state : Cuda.DevicePtr Float32
  /-- FP32 recurrent output `[tokens, 48, 128]`. -/
  output : Cuda.DevicePtr Float32

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
private def loadProjected (pointer : Cuda.DevicePtr Cuda.BFloat16)
    (index : UInt32) : Cuda.DeviceM Float32 := do
  return (← Cuda.loadBFloat16 pointer index.toUSize).toFloat32

@[always_inline]
private partial def decayState (state : Cuda.DevicePtr Float32) (head index stride : UInt32)
    (decay : Float32) : Cuda.DeviceM Unit := do
  if index < stateElementsPerHead then
    let address := stateLinearIndex head index
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
    let address := stateLinearIndex head index
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

/-- Execute the exact single-token recurrent delta-rule core for one value head per block. -/
@[always_inline, convergent]
private def recurrentStep (token : UInt32) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let head ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let compactHead := head / valueHeadsPerKeyHead
  let reduction : Cuda.Collective.BlockScratch 8 := shared
  let queryShared ← Cuda.dynamicShared (α := Float32) queryOffset.toUSize
  let keyShared ← Cuda.dynamicShared (α := Float32) keyOffset.toUSize
  let deltaShared ← Cuda.dynamicShared (α := Float32) deltaOffset.toUSize

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

  if thread == 0 then
    let decay ← Cuda.fastExp
      (← Cuda.loadFloat32 buffers.decayLog (token * valueHeads + head).toUSize)
    Cuda.storeFloat32 reduction 0 decay
  Cuda.blockSync
  let decay ← Cuda.loadFloat32 reduction 0
  decayState buffers.state head thread 256 decay
  Cuda.blockSync

  if thread < headDimension then
    let memory ← stateKeyDot buffers.state keyShared head thread 0 (Float32.ofBits 0)
    let value ← loadProjected buffers.value (valueIndex token head thread)
    let beta ← Cuda.loadFloat32 buffers.beta (token * valueHeads + head).toUSize
    Cuda.storeFloat32 deltaShared thread.toUSize ((value - memory) * beta)
  Cuda.blockSync

  updateState buffers.state keyShared deltaShared head thread 256
  Cuda.blockSync

  if thread < headDimension then
    let result ← stateQueryDot buffers.state queryShared head thread 0 (Float32.ofBits 0)
    Cuda.storeFloat32 buffers.output (valueIndex token head thread).toUSize result

/-- Conventional one-token launch used as an exact same-code baseline. -/
@[cuda_kernel]
def qwen36DecodeStep (token : UInt32)
    (query key value : Cuda.DevicePtr Cuda.BFloat16)
    (decayLog beta state output : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  recurrentStep token { query, key, value, decayLog, beta, state, output } shared

/--
Cooperative-grid resident Qwen3.6 recurrent decode worker.

All 48 blocks execute every token descriptor. The grid-persistent runtime provides an epoch
boundary between descriptors; the explicit grid barrier makes the observable completion counter
unambiguous and keeps this worker safe if later slices add cross-head reductions.
-/
@[cuda_grid_persistent]
def qwen36Decode (item : DecodeItem)
    (query key value : Cuda.DevicePtr Cuda.BFloat16)
    (decayLog beta state output : Cuda.DevicePtr Float32)
    (counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  recurrentStep item.token { query, key, value, decayLog, beta, state, output } shared
  Cuda.gridSync
  if (← Cuda.blockIdxX) == 0 && (← Cuda.threadIdxX) == 0 then
    Cuda.atomicAddUInt32At_ counts item.token 1

private structure InputBuffers where
  query : Cuda.Buffer Cuda.BFloat16
  key : Cuda.Buffer Cuda.BFloat16
  value : Cuda.Buffer Cuda.BFloat16
  decayLog : Cuda.Buffer Float32
  beta : Cuda.Buffer Float32

private structure RunBuffers where
  state : Cuda.Buffer Float32
  output : Cuda.Buffer Float32
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

private def queryValue (token head channel : Nat) : Float32 :=
  signedValue (token * 101 + head * 31 + channel * 7 + 3) 29 14 32

private def keyValue (token head channel : Nat) : Float32 :=
  signedValue (token * 79 + head * 43 + channel * 11 + 5) 31 15 36

private def projectedValue (token head channel : Nat) : Float32 :=
  signedValue (token * 61 + head * 17 + channel * 13 + 9) 37 18 40

private def decayLogValue (token head : Nat) : Float32 :=
  -(head % 7 + token + 1).toUInt32.toFloat32 / Float32.ofBits 0x42c80000

private def betaValue (token head : Nat) : Float32 :=
  Float32.ofBits 0x3e800000 +
    (head % 5 + token).toUInt32.toFloat32 / Float32.ofBits 0x41200000

private def initialStateValue (index : Nat) : Float32 :=
  signedValue (index * 19 + 7) 41 20 512

private def makeCompactProjection (value : Nat → Nat → Nat → Float32) : ByteArray :=
  makeBFloat16Data (tokenCount * keyHeadCount * headDimensionNat) fun index =>
    let token := index / (keyHeadCount * headDimensionNat)
    let within := index % (keyHeadCount * headDimensionNat)
    value token (within / headDimensionNat) (within % headDimensionNat)

private def makeValueProjection : ByteArray :=
  makeBFloat16Data outputElementCount fun index =>
    let token := index / (valueHeadCount * headDimensionNat)
    let within := index % (valueHeadCount * headDimensionNat)
    projectedValue token (within / headDimensionNat) (within % headDimensionNat)

private def initializeInputs (stream : Cuda.Stream) : IO InputBuffers := do
  let query ← Cuda.Buffer.alloc Cuda.BFloat16
    (tokenCount * keyHeadCount * headDimensionNat).toUSize
  let key ← Cuda.Buffer.alloc Cuda.BFloat16
    (tokenCount * keyHeadCount * headDimensionNat).toUSize
  let value ← Cuda.Buffer.alloc Cuda.BFloat16 outputElementCount.toUSize
  let decayLog ← Cuda.Buffer.alloc Float32 (tokenCount * valueHeadCount).toUSize
  let beta ← Cuda.Buffer.alloc Float32 (tokenCount * valueHeadCount).toUSize
  query.copyFrom (makeCompactProjection queryValue) stream
  key.copyFrom (makeCompactProjection keyValue) stream
  value.copyFrom makeValueProjection stream
  decayLog.copyFrom (makeFloat32Data (tokenCount * valueHeadCount) fun index =>
    decayLogValue (index / valueHeadCount) (index % valueHeadCount)) stream
  beta.copyFrom (makeFloat32Data (tokenCount * valueHeadCount) fun index =>
    betaValue (index / valueHeadCount) (index % valueHeadCount)) stream
  return { query, key, value, decayLog, beta }

private def initializeRunBuffers (stream : Cuda.Stream) : IO RunBuffers := do
  let state ← Cuda.Buffer.alloc Float32 stateElementCount.toUSize
  let output ← Cuda.Buffer.alloc Float32 outputElementCount.toUSize
  let counts ← Cuda.Buffer.alloc UInt32 tokenCount.toUSize
  state.copyFrom (makeFloat32Data stateElementCount initialStateValue) stream
  output.copyFrom (zeroBytes (outputElementCount * 4)) stream
  counts.copyFrom (zeroBytes (tokenCount * 4)) stream
  return { state, output, counts }

private def waitKernel (label : String) (handle : Cuda.KernelHandle) : IO Unit := do
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"{label} failed: {repr status}"

private def oneShotConfig : Cuda.LaunchConfig := {
  grid := { x := valueHeads }
  block := { x := 256 }
  sharedMemoryBytes := decodeSharedBytes.toUSize
  blockArenaBytes := 16 * 1024
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := valueHeads }
  block := { x := 256 }
  sharedMemoryBytes := decodeSharedBytes.toUSize
  blockArenaBytes := 16 * 1024
  queueCapacity := tokenCount.toUSize
}

private def runSeparate (inputs : InputBuffers) (buffers : RunBuffers) : IO Unit := do
  for token in [:tokenCount] do
    waitKernel "Qwen3.6 separate decode step" <| ← qwen36DecodeStep.launch oneShotConfig
      token.toUInt32 inputs.query inputs.key inputs.value inputs.decayLog inputs.beta buffers.state
      buffers.output

private def runPersistent (inputs : InputBuffers) (buffers : RunBuffers) : IO Unit := do
  let handle ← qwen36Decode.start persistentConfig inputs.query inputs.key inputs.value
    inputs.decayLog inputs.beta buffers.state buffers.output buffers.counts
  for token in [:tokenCount] do
    qwen36Decode.enqueue handle { token := token.toUInt32 }
  let running ← handle.query
  unless running.state == .running do
    throw <| IO.userError s!"Qwen3.6 decode megakernel exited before shutdown: {repr running}"
  handle.shutdown
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"Qwen3.6 decode megakernel failed: {repr status}"

private def roundedQueryValue (token head channel : Nat) : Float32 :=
  (Cuda.BFloat16.ofFloat32 (queryValue token head channel)).toFloat32

private def roundedKeyValue (token head channel : Nat) : Float32 :=
  (Cuda.BFloat16.ofFloat32 (keyValue token head channel)).toFloat32

private def roundedProjectedValue (token head channel : Nat) : Float32 :=
  (Cuda.BFloat16.ofFloat32 (projectedValue token head channel)).toFloat32

private structure OracleResult where
  state : Array Float32
  output : Array Float32

/-- Independent host replay of the recurrent equations, including BF16 projection rounding. -/
private def hostOracle : OracleResult := Id.run do
  let mut state := (List.range stateElementCount).map initialStateValue |>.toArray
  let mut output := Array.replicate outputElementCount (Float32.ofBits 0)
  for token in [:tokenCount] do
    for head in [:valueHeadCount] do
      let compactHead := head / (valueHeadCount / keyHeadCount)
      let mut querySquareSum := Float32.ofBits 0
      let mut keySquareSum := Float32.ofBits 0
      for channel in [:headDimensionNat] do
        let query := roundedQueryValue token compactHead channel
        let key := roundedKeyValue token compactHead channel
        querySquareSum := querySquareSum + query * query
        keySquareSum := keySquareSum + key * key
      let queryInverse := Float32.ofBits 0x3f800000 /
        Float32.sqrt (querySquareSum + normalizationEpsilon)
      let keyInverse := Float32.ofBits 0x3f800000 /
        Float32.sqrt (keySquareSum + normalizationEpsilon)
      let queryNormalized := (List.range headDimensionNat).map (fun channel =>
        roundedQueryValue token compactHead channel * queryInverse * queryScale) |>.toArray
      let keyNormalized := (List.range headDimensionNat).map (fun channel =>
        roundedKeyValue token compactHead channel * keyInverse) |>.toArray
      let decay := Float32.exp (decayLogValue token head)
      let headBase := head * stateElementsPerHeadNat
      for index in [:stateElementsPerHeadNat] do
        state := state.set! (headBase + index) (state[headBase + index]! * decay)
      let mut delta := Array.replicate headDimensionNat (Float32.ofBits 0)
      for column in [:headDimensionNat] do
        let mut memory := Float32.ofBits 0
        for row in [:headDimensionNat] do
          memory := Cuda.fma state[headBase + row * headDimensionNat + column]!
            keyNormalized[row]! memory
        delta := delta.set! column
          ((roundedProjectedValue token head column - memory) * betaValue token head)
      for row in [:headDimensionNat] do
        for column in [:headDimensionNat] do
          let index := headBase + row * headDimensionNat + column
          state := state.set! index (Cuda.fma keyNormalized[row]! delta[column]! state[index]!)
      for column in [:headDimensionNat] do
        let mut result := Float32.ofBits 0
        for row in [:headDimensionNat] do
          result := Cuda.fma state[headBase + row * headDimensionNat + column]!
            queryNormalized[row]! result
        let outputIndex := (token * valueHeadCount + head) * headDimensionNat + column
        output := output.set! outputIndex result
  return { state, output }

private def requireEqual (label : String) (actual expected : ByteArray) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError s!"persistent and separate-launch {label} differ"

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

def main : IO Unit := do
  let stream ← Cuda.Stream.default
  let inputs ← initializeInputs stream
  let separate ← initializeRunBuffers stream
  let persistent ← initializeRunBuffers stream
  stream.synchronize

  let separateStart ← Cuda.Event.createTiming
  let separateStop ← Cuda.Event.createTiming
  separateStart.record stream
  runSeparate inputs separate
  separateStop.record stream
  separateStop.wait
  let separateMilliseconds ← Cuda.Event.elapsedMilliseconds separateStart separateStop

  let persistentStart ← Cuda.Event.createTiming
  let persistentStop ← Cuda.Event.createTiming
  persistentStart.record stream
  runPersistent inputs persistent
  persistentStop.record stream
  persistentStop.wait
  let persistentMilliseconds ← Cuda.Event.elapsedMilliseconds persistentStart persistentStop

  let separateState ← separate.state.copyTo stream
  let separateOutput ← separate.output.copyTo stream
  let persistentState ← persistent.state.copyTo stream
  let persistentOutput ← persistent.output.copyTo stream
  requireEqual "Qwen3.6 recurrent state" persistentState separateState
  requireEqual "Qwen3.6 recurrent output" persistentOutput separateOutput
  let countBytes ← persistent.counts.copyTo stream
  for token in [:tokenCount] do
    unless readUInt32 countBytes token == 1 do
      throw <| IO.userError s!"Qwen3.6 token descriptor {token} did not execute exactly once"

  let oracle := hostOracle
  requireClose "Qwen3.6 recurrent state" persistentState oracle.state (Float32.ofBits 0x3a83126f)
  requireClose "Qwen3.6 recurrent output" persistentOutput oracle.output
    (Float32.ofBits 0x3b03126f)
  IO.println <| s!"Lean Qwen3.6 recurrent decode megakernel ok; one persistent launch = " ++
    s!"{persistentMilliseconds} ms, three separate launches = {separateMilliseconds} ms"

end Qwen36

def main : IO Unit := Qwen36.main
