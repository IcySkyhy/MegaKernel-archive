/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda

/-!
# Qwen3.6 token-boundary megakernel

This plain-Lean CUDA kernel closes the token boundary at the checkpoint's exact text geometry:

* lookup from a BF16 `[248320, 5120]` embedding allocation;
* final `(1 + weight)` RMSNorm over 5120 channels;
* stable softmax, greedy sampling, cross-entropy, and `probability - target` over all 248320
  vocabulary entries;
* one cooperative launch, with a conventional finite cooperative launch used as the equivalence
  route.

The validation allocates the full embedding address space but initializes only the selected row;
the vocabulary-wide logits and gradient are fully materialized. The LM-head projection producing
those logits is tested separately by the descriptor-driven projection engine.
-/

namespace Qwen36Token

private def hiddenSizeNat : Nat := 5120
private def vocabularySizeNat : Nat := 248320
private def embeddingElementsNat : Nat := vocabularySizeNat * hiddenSizeNat
private def embeddingElements : UInt32 := embeddingElementsNat.toUInt32
private def projectionColumn : UInt32 := 2
private def blockCountNat : Nat := 48
private def inputTokenNat : Nat := vocabularySizeNat - 1
private def targetTokenNat : Nat := 12345
private def sampledTokenNat : Nat := 147777

private def hiddenSize : UInt32 := hiddenSizeNat.toUInt32
private def vocabularySize : UInt32 := vocabularySizeNat.toUInt32
private def blockCount : UInt32 := blockCountNat.toUInt32
private def inputToken : UInt32 := inputTokenNat.toUInt32
private def targetToken : UInt32 := targetTokenNat.toUInt32
private def sampledToken : UInt32 := sampledTokenNat.toUInt32
private def zero : Float32 := Float32.ofBits 0
private def one : Float32 := Float32.ofBits 0x3f800000
private def negativeInfinity : Float32 := Float32.ofBits 0xff800000
private def inverseHiddenSize : Float32 := Float32.ofBits 0x394ccccd
private def rmsEpsilon : Float32 := Float32.ofBits 0x358637bd
private def log2e : Float32 := Float32.ofBits 0x3fb8aa3b

structure TokenItem where
  token : UInt32
  target : UInt32
  descriptor : UInt32
  deriving Cuda.POD

@[struct] private structure Buffers where
  embedding : Cuda.DevicePtr Cuda.BFloat16
  normWeight : Cuda.DevicePtr Cuda.BFloat16
  lmHead : Cuda.DevicePtr Cuda.BFloat16
  hidden : Cuda.DevicePtr Cuda.BFloat16
  normalized : Cuda.DevicePtr Cuda.BFloat16
  logits : Cuda.DevicePtr Float32
  gradient : Cuda.DevicePtr Cuda.BFloat16
  partialMaximum : Cuda.DevicePtr Float32
  partialSum : Cuda.DevicePtr Float32
  loss : Cuda.DevicePtr Float32
  sampled : Cuda.DevicePtr UInt32

@[always_inline]
private partial def initializeLoop (item : TokenItem)
    (embedding normWeight lmHead : Cuda.DevicePtr Cuda.BFloat16)
    (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < embeddingElements then
    let row := index / hiddenSize
    let column := index % hiddenSize
    let headResidue := (row * 13 + 7) % 257
    let baseWeight := (headResidue.toFloat32 - Float32.ofBits 0x43000000) /
      Float32.ofBits 0x42000000
    let headWeight := if row == sampledToken then
        baseWeight + Float32.ofBits 0x41800000
      else
        baseWeight
    let weight := if column == projectionColumn then headWeight else zero
    Cuda.storeBFloat16 lmHead index.toUSize (Cuda.BFloat16.ofFloat32 weight)
    if index < hiddenSize then
      let embeddingResidue := (index * 11 + 5) % 33
      let embeddingValue := (embeddingResidue.toFloat32 - Float32.ofBits 0x41800000) /
        Float32.ofBits 0x42000000
      let normResidue := (index * 7 + 3) % 15
      let normValue := (normResidue.toFloat32 - Float32.ofBits 0x40e00000) /
        Float32.ofBits 0x43800000
      Cuda.storeBFloat16 embedding (item.token * hiddenSize + index).toUSize
        (Cuda.BFloat16.ofFloat32 embeddingValue)
      Cuda.storeBFloat16 normWeight index.toUSize (Cuda.BFloat16.ofFloat32 normValue)
    initializeLoop item embedding normWeight lmHead (index + stride) stride

@[cuda_kernel]
def initializeTokenInputs (item : TokenItem)
    (embedding normWeight lmHead : Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  initializeLoop item embedding normWeight lmHead (← Cuda.globalThreadIdxX)
    (← Cuda.globalThreadCountX)

@[always_inline]
private partial def copyEmbeddingLoop (item : TokenItem) (buffers : Buffers)
    (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < hiddenSize then
    let value ← Cuda.loadBFloat16 buffers.embedding (item.token * hiddenSize + index).toUSize
    Cuda.storeBFloat16 buffers.hidden index.toUSize value
    copyEmbeddingLoop item buffers (index + stride) stride

@[always_inline]
private partial def squareSumLoop (hidden : Cuda.DevicePtr Cuda.BFloat16)
    (index stride : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < hiddenSize then
    let value := (← Cuda.loadBFloat16 hidden index.toUSize).toFloat32
    squareSumLoop hidden (index + stride) stride (Cuda.fma value value accumulator)
  else
    return accumulator

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
private partial def normalizeLoop (buffers : Buffers) (inverse : Float32)
    (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < hiddenSize then
    let value := (← Cuda.loadBFloat16 buffers.hidden index.toUSize).toFloat32
    let weight := (← Cuda.loadBFloat16 buffers.normWeight index.toUSize).toFloat32
    Cuda.storeBFloat16 buffers.normalized index.toUSize
      (Cuda.BFloat16.ofFloat32 (value * inverse * (one + weight)))
    normalizeLoop buffers inverse (index + stride) stride

@[always_inline]
private partial def dotLMHeadRow (buffers : Buffers) (row column : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < hiddenSize then
    let input := (← Cuda.loadBFloat16 buffers.normalized column.toUSize).toFloat32
    let weight := (← Cuda.loadBFloat16 buffers.lmHead
      (row * hiddenSize + column).toUSize).toFloat32
    dotLMHeadRow buffers row (column + 256) (Cuda.fma weight input accumulator)
  else
    return accumulator

@[always_inline, convergent]
private partial def projectRows (buffers : Buffers)
    (scratch : Cuda.Collective.BlockScratch 8)
    (row rowStride thread : UInt32) : Cuda.DeviceM Unit := do
  if row < vocabularySize then
    let localSum ← dotLMHeadRow buffers row thread zero
    let total ← Cuda.Collective.blockSum scratch localSum
    if thread == 0 then
      Cuda.storeFloat32 buffers.logits row.toUSize total
    projectRows buffers scratch (row + rowStride) rowStride thread

@[always_inline]
private partial def localMaximumLoop (logits : Cuda.DevicePtr Float32)
    (index stride : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < vocabularySize then
    localMaximumLoop logits (index + stride) stride
      (max accumulator (← Cuda.loadFloat32 logits index.toUSize))
  else
    return accumulator

@[always_inline]
private partial def localExponentialSumLoop (logits : Cuda.DevicePtr Float32)
    (maximum : Float32) (index stride : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if index < vocabularySize then
    let logit ← Cuda.loadFloat32 logits index.toUSize
    let exponential ← Cuda.fastExp2Fma ((logit - maximum) * log2e)
    localExponentialSumLoop logits maximum (index + stride) stride (accumulator + exponential)
  else
    return accumulator

@[always_inline]
private partial def gradientAndSampleLoop (item : TokenItem) (buffers : Buffers)
    (maximum inverseDenominator : Float32) (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < vocabularySize then
    let logit ← Cuda.loadFloat32 buffers.logits index.toUSize
    let exponential ← Cuda.fastExp2Fma ((logit - maximum) * log2e)
    let probability := exponential * inverseDenominator
    let target := if index == item.target then one else zero
    Cuda.storeBFloat16 buffers.gradient index.toUSize
      (Cuda.BFloat16.ofFloat32 (probability - target))
    if logit == maximum then
      discard <| Cuda.atomicMinUInt32 buffers.sampled index
    gradientAndSampleLoop item buffers maximum inverseDenominator (index + stride) stride

@[always_inline, convergent]
private def runToken (item : TokenItem) (buffers : Buffers)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let globalThread ← Cuda.globalThreadIdxX
  let globalThreads ← Cuda.globalThreadCountX
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  if block == 0 && thread == 0 then
    Cuda.storeUInt32 buffers.sampled 0 0xffffffff
  copyEmbeddingLoop item buffers globalThread globalThreads
  Cuda.gridSync

  let localSquares ← squareSumLoop buffers.hidden globalThread globalThreads zero
  let blockSquares ← Cuda.Collective.blockSum scratch localSquares
  if thread == 0 then
    Cuda.storeFloat32 buffers.partialSum block.toUSize blockSquares
  Cuda.gridSync
  let squareSum ← sumPartials buffers.partialSum 0 zero
  let inverse ← Cuda.fastRsqrt (squareSum * inverseHiddenSize + rmsEpsilon)
  normalizeLoop buffers inverse globalThread globalThreads
  Cuda.gridSync
  projectRows buffers scratch block blockCount thread
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
    Cuda.storeFloat32 buffers.loss 0
      (maximum + (← Cuda.fastLog denominator) - targetLogit)
  Cuda.gridSync

@[cuda_kernel]
def qwen36TokenStep (item : TokenItem)
    (embedding normWeight lmHead hidden normalized : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runToken item {
    embedding, normWeight, lmHead, hidden, normalized, logits, gradient,
    partialMaximum, partialSum, loss, sampled
  } shared

@[cuda_grid_persistent]
def qwen36Token (item : TokenItem)
    (embedding normWeight lmHead hidden normalized : Cuda.DevicePtr Cuda.BFloat16)
    (logits : Cuda.DevicePtr Float32) (gradient : Cuda.DevicePtr Cuda.BFloat16)
    (partialMaximum partialSum loss : Cuda.DevicePtr Float32)
    (sampled counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runToken item {
    embedding, normWeight, lmHead, hidden, normalized, logits, gradient,
    partialMaximum, partialSum, loss, sampled
  } shared
  if (← Cuda.blockIdxX) == 0 && (← Cuda.threadIdxX) == 0 then
    Cuda.atomicAddUInt32At_ counts item.descriptor 1

private structure SharedBuffers where
  embedding : Cuda.Buffer Cuda.BFloat16
  normWeight : Cuda.Buffer Cuda.BFloat16
  lmHead : Cuda.Buffer Cuda.BFloat16

private structure RunBuffers where
  hidden : Cuda.Buffer Cuda.BFloat16
  normalized : Cuda.Buffer Cuda.BFloat16
  logits : Cuda.Buffer Float32
  gradient : Cuda.Buffer Cuda.BFloat16
  partialMaximum : Cuda.Buffer Float32
  partialSum : Cuda.Buffer Float32
  loss : Cuda.Buffer Float32
  sampled : Cuda.Buffer UInt32

private def zeroBytes (count : Nat) : ByteArray :=
  List.replicate count 0 |>.toByteArray

private def allocateShared : IO SharedBuffers := do
  let embedding ← Cuda.Buffer.alloc Cuda.BFloat16 embeddingElementsNat.toUSize
  let normWeight ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let lmHead ← Cuda.Buffer.alloc Cuda.BFloat16 embeddingElementsNat.toUSize
  return { embedding, normWeight, lmHead }

private def allocateRun : IO RunBuffers := do
  let hidden ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let normalized ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let logits ← Cuda.Buffer.alloc Float32 vocabularySizeNat.toUSize
  let gradient ← Cuda.Buffer.alloc Cuda.BFloat16 vocabularySizeNat.toUSize
  let partialMaximum ← Cuda.Buffer.alloc Float32 blockCountNat.toUSize
  let partialSum ← Cuda.Buffer.alloc Float32 blockCountNat.toUSize
  let loss ← Cuda.Buffer.alloc Float32 1
  let sampled ← Cuda.Buffer.alloc UInt32 1
  return { hidden, normalized, logits, gradient, partialMaximum, partialSum, loss, sampled }

private def item : TokenItem := {
  token := inputToken
  target := targetToken
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
  blockArenaBytes := 16 * 1024
  cooperative := true
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := blockCount }
  block := { x := 256 }
  sharedMemoryBytes := (8 * 4).toUSize
  blockArenaBytes := 16 * 1024
  queueCapacity := 1
}

private def waitKernel (label : String) (handle : Cuda.KernelHandle) : IO Unit := do
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"{label} failed: {repr status}"

private def pushUInt16 (bytes : ByteArray) (value : UInt16) : ByteArray :=
  (bytes.push value.toUInt8).push (value >>> 8).toUInt8

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

private def signedValue (index modulus center divisor : Nat) : Float32 :=
  (Int32.ofInt (Int.ofNat (index % modulus) - Int.ofNat center)).toFloat32 /
    divisor.toUInt32.toFloat32

private def rounded (value : Float32) : Float32 :=
  (Cuda.BFloat16.ofFloat32 value).toFloat32

private def embeddingValue (channel : Nat) : Float32 :=
  rounded (signedValue (channel * 11 + 5) 33 16 32)

private def normWeightValue (channel : Nat) : Float32 :=
  rounded (signedValue (channel * 7 + 3) 15 7 256)

private def lmHeadWeightValue (token : Nat) : Float32 :=
  let base := signedValue (token * 13 + 7) 257 128 32
  rounded (if token == sampledTokenNat then base + Float32.ofBits 0x41800000 else base)

private def expectedNormalized : Array Float32 := Id.run do
  let mut squareSum := zero
  for channel in [:hiddenSizeNat] do
    let value := embeddingValue channel
    squareSum := Cuda.fma value value squareSum
  let inverse := one / Float32.sqrt (squareSum * inverseHiddenSize + rmsEpsilon)
  return (List.range hiddenSizeNat).map (fun channel =>
    rounded (embeddingValue channel * inverse * (one + normWeightValue channel))) |>.toArray

private structure LossOracle where
  logits : Array Float32
  loss : Float32
  gradient : Array Float32

private def lossOracle : LossOracle := Id.run do
  let normalized := expectedNormalized
  let scale := normalized[projectionColumn.toNat]!
  let logits := (List.range vocabularySizeNat).map (fun token =>
    scale * lmHeadWeightValue token) |>.toArray
  let mut maximum := negativeInfinity
  for token in [:vocabularySizeNat] do
    maximum := max maximum logits[token]!
  let mut denominator := zero
  for token in [:vocabularySizeNat] do
    denominator := denominator + Float32.exp (logits[token]! - maximum)
  let inverseDenominator := one / denominator
  let gradient := (List.range vocabularySizeNat).map (fun token =>
    let probability := Float32.exp (logits[token]! - maximum) * inverseDenominator
    rounded (probability - if token == targetTokenNat then one else zero)) |>.toArray
  return {
    logits
    loss := maximum + Float32.log denominator - logits[targetTokenNat]!
    gradient
  }

private def requireBFloatClose (label : String) (actual : ByteArray)
    (expected : Array Float32) (tolerance : Float32) : IO Unit := do
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
  IO.println s!"Qwen3.6 {label} host-oracle max error: {maximumError}"
private def requireFloatClose (label : String) (actual : ByteArray)
    (expected : Array Float32) (tolerance : Float32) : IO Unit := do
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
  IO.println s!"Qwen3.6 {label} host-oracle max error: {maximumError}"


private def copyRun (run : RunBuffers) (stream : Cuda.Stream) : IO
    (ByteArray × ByteArray × ByteArray × ByteArray × ByteArray × ByteArray) := do
  let hidden ← run.hidden.copyTo stream
  let normalized ← run.normalized.copyTo stream
  let logits ← run.logits.copyTo stream
  let gradient ← run.gradient.copyTo stream
  let loss ← run.loss.copyTo stream
  let sampled ← run.sampled.copyTo stream
  return (hidden, normalized, logits, gradient, loss, sampled)

def main : IO Unit := do
  let stream ← Cuda.Stream.default
  let shared ← allocateShared
  let finite ← allocateRun
  let persistent ← allocateRun
  let counts ← Cuda.Buffer.alloc UInt32 1
  counts.copyFrom (zeroBytes 4) stream
  waitKernel "Qwen3.6 token input initialization" <|
    ← initializeTokenInputs.launch initializerConfig item shared.embedding shared.normWeight
      shared.lmHead
  waitKernel "Qwen3.6 finite token boundary" <|
    ← qwen36TokenStep.launch launchConfig item shared.embedding shared.normWeight shared.lmHead
      finite.hidden finite.normalized finite.logits finite.gradient finite.partialMaximum finite.partialSum
      finite.loss finite.sampled
  let handle ← qwen36Token.start persistentConfig shared.embedding shared.normWeight shared.lmHead
    persistent.hidden persistent.normalized persistent.logits persistent.gradient
    persistent.partialMaximum persistent.partialSum persistent.loss persistent.sampled counts
  qwen36Token.enqueue handle item
  handle.shutdown
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"Qwen3.6 token megakernel failed: {repr status}"
  let finiteResult ← copyRun finite stream
  let persistentResult ← copyRun persistent stream
  unless finiteResult == persistentResult do
    throw <| IO.userError "persistent and finite Qwen3.6 token-boundary results differ"
  let (hiddenBytes, normalizedBytes, logitsBytes, gradientBytes, lossBytes, sampledBytes) :=
    persistentResult
  for channel in [:hiddenSizeNat] do
    unless readBFloat hiddenBytes channel == embeddingValue channel do
      throw <| IO.userError s!"embedding lookup differs at channel {channel}"
  requireBFloatClose "final RMSNorm" normalizedBytes expectedNormalized
    (Float32.ofBits 0x3c000000)
  let oracle := lossOracle
  requireFloatClose "LM-head logits" logitsBytes oracle.logits (Float32.ofBits 0x39800000)
  requireBFloatClose "vocabulary gradient" gradientBytes oracle.gradient
    (Float32.ofBits 0x3c000000)
  let lossError := Float32.abs (readFloat lossBytes 0 - oracle.loss)
  unless lossError <= Float32.ofBits 0x3c800000 do
    throw <| IO.userError
      s!"cross-entropy differs: actual={readFloat lossBytes 0}, expected={oracle.loss}, error={lossError}"
  unless readUInt32 sampledBytes 0 == sampledToken do
    throw <| IO.userError
      s!"greedy sample differs: actual={readUInt32 sampledBytes 0}, expected={sampledToken}"
  let countBytes ← counts.copyTo stream
  unless readUInt32 countBytes 0 == 1 do
    throw <| IO.userError "Qwen3.6 token descriptor did not execute exactly once"
  IO.println s!"Qwen3.6 exact-vocabulary cross-entropy host-oracle error: {lossError}"
  IO.println "Lean Qwen3.6 embedding/RMSNorm/LM-head/loss/sampling megakernel ok"

end Qwen36Token

def main : IO Unit := Qwen36Token.main
