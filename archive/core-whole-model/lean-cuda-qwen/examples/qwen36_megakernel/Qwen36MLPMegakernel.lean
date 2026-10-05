/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Cuda

/-!
# Qwen3.6-27B decode MLP megakernel

One cooperative-grid token item executes the exact dense decoder MLP boundary:

`RMSNorm[5120] -> gate/up [17408,5120] -> SiLU(gate) * up -> down [5120,17408] -> residual`.

The test allocates and reads all 267,386,880 BF16 matrix elements (about 510 MiB). Sparse,
deterministic validation weights make a complete host Lean oracle practical while retaining the
real checkpoint shapes and memory traffic. The resident path must match five ordinary Lean CUDA
launches byte-for-byte.
-/

namespace Qwen36MLP

private def hiddenSizeNat : Nat := 5120
private def intermediateSizeNat : Nat := 17408
private def blockCountNat : Nat := 48
private def hiddenSize : UInt32 := hiddenSizeNat.toUInt32
private def intermediateSize : UInt32 := intermediateSizeNat.toUInt32
private def blockCount : UInt32 := blockCountNat.toUInt32
private def zero : Float32 := Float32.ofBits 0
private def one : Float32 := Float32.ofBits 0x3f800000
private def rmsEpsilon : Float32 := Float32.ofBits 0x358637bd
private def inverseHiddenSize : Float32 := Float32.ofBits 0x394ccccd

structure TokenItem where
  descriptor : UInt32
  deriving Cuda.POD

@[struct] structure Weights where
  norm : Cuda.DevicePtr Cuda.BFloat16
  gate : Cuda.DevicePtr Cuda.BFloat16
  up : Cuda.DevicePtr Cuda.BFloat16
  down : Cuda.DevicePtr Cuda.BFloat16

@[struct] structure Buffers where
  residual : Cuda.DevicePtr Cuda.BFloat16
  normalized : Cuda.DevicePtr Cuda.BFloat16
  gate : Cuda.DevicePtr Cuda.BFloat16
  up : Cuda.DevicePtr Cuda.BFloat16
  intermediate : Cuda.DevicePtr Cuda.BFloat16
  projected : Cuda.DevicePtr Cuda.BFloat16
  output : Cuda.DevicePtr Cuda.BFloat16

@[always_inline]
private def selectedColumn (row columns salt : UInt32) : UInt32 :=
  (row * (salt * 2 + 3) + salt * 17 + 5) % columns

@[always_inline]
private def selectedWeight (row salt : UInt32) : Float32 :=
  let magnitude := if salt % 2 == 0 then Float32.ofBits 0x3f000000
    else Float32.ofBits 0x3e800000
  if (row + salt) % 2 == 0 then magnitude else -magnitude

@[always_inline]
private partial def initializeMatrixLoop (weight : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns salt index stride : UInt32) : Cuda.DeviceM Unit := do
  let count := rows * columns
  if index < count then
    let row := index / columns
    let column := index % columns
    let value := if column == selectedColumn row columns salt then selectedWeight row salt else zero
    Cuda.storeBFloat16 weight index.toUSize (Cuda.BFloat16.ofFloat32 value)
    initializeMatrixLoop weight rows columns salt (index + stride) stride

@[cuda_kernel]
def initializeMatrix (rows columns salt : UInt32)
    (weight : Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  initializeMatrixLoop weight rows columns salt (← Cuda.globalThreadIdxX)
    (← Cuda.globalThreadCountX)

@[always_inline]
private partial def dotRow (input weight : Cuda.DevicePtr Cuda.BFloat16)
    (row columns column stride : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < columns then
    let inputValue := (← Cuda.loadBFloat16 input column.toUSize).toFloat32
    let weightValue := (← Cuda.loadBFloat16 weight (row * columns + column).toUSize).toFloat32
    dotRow input weight row columns (column + stride) stride
      (Cuda.fma weightValue inputValue accumulator)
  else
    return accumulator

@[always_inline, convergent]
private partial def projectRows (input weight output : Cuda.DevicePtr Cuda.BFloat16)
    (rows columns : UInt32) (scratch : Cuda.Collective.BlockScratch 8)
    (row rowStride thread : UInt32) : Cuda.DeviceM Unit := do
  if row < rows then
    let localSum ← dotRow input weight row columns thread 256 zero
    let total ← Cuda.Collective.blockSum scratch localSum
    if thread == 0 then
      Cuda.storeBFloat16 output row.toUSize (Cuda.BFloat16.ofFloat32 total)
    projectRows input weight output rows columns scratch (row + rowStride) rowStride thread

@[always_inline, convergent]
private partial def projectDualRows (input gateWeight upWeight gateOutput upOutput :
    Cuda.DevicePtr Cuda.BFloat16) (rows columns : UInt32)
    (scratch : Cuda.Collective.BlockScratch 8) (row rowStride thread : UInt32) :
    Cuda.DeviceM Unit := do
  if row < rows then
    let gateLocal ← dotRow input gateWeight row columns thread 256 zero
    let gateTotal ← Cuda.Collective.blockSum scratch gateLocal
    let upLocal ← dotRow input upWeight row columns thread 256 zero
    let upTotal ← Cuda.Collective.blockSum scratch upLocal
    if thread == 0 then
      Cuda.storeBFloat16 gateOutput row.toUSize (Cuda.BFloat16.ofFloat32 gateTotal)
      Cuda.storeBFloat16 upOutput row.toUSize (Cuda.BFloat16.ofFloat32 upTotal)
    projectDualRows input gateWeight upWeight gateOutput upOutput rows columns scratch
      (row + rowStride) rowStride thread

@[always_inline]
private partial def sumSquares (input : Cuda.DevicePtr Cuda.BFloat16)
    (index stride : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if index < hiddenSize then
    let value := (← Cuda.loadBFloat16 input index.toUSize).toFloat32
    sumSquares input (index + stride) stride (Cuda.fma value value accumulator)
  else
    return accumulator

@[always_inline]
private partial def writeNormalized (input norm output : Cuda.DevicePtr Cuda.BFloat16)
    (inverse : Float32) (index stride : UInt32) : Cuda.DeviceM Unit := do
  if index < hiddenSize then
    let value := (← Cuda.loadBFloat16 input index.toUSize).toFloat32
    let weight := (← Cuda.loadBFloat16 norm index.toUSize).toFloat32
    Cuda.storeBFloat16 output index.toUSize
      (Cuda.BFloat16.ofFloat32 (value * inverse * (one + weight)))
    writeNormalized input norm output inverse (index + stride) stride

@[always_inline, convergent]
private def runNorm (buffers : Buffers) (weights : Weights)
    (shared : Cuda.DevicePtr Float32) (block thread : UInt32) : Cuda.DeviceM Unit := do
  if block == 0 then
    let scratch : Cuda.Collective.BlockScratch 8 := shared
    let localSum ← sumSquares buffers.residual thread 256 zero
    let squareSum ← Cuda.Collective.blockSum scratch localSum
    let inverse ← Cuda.fastRsqrt (squareSum * inverseHiddenSize + rmsEpsilon)
    if thread == 0 then
      Cuda.storeFloat32 shared 0 inverse
    Cuda.blockSync
    let inverse ← Cuda.loadFloat32 shared 0
    writeNormalized buffers.residual weights.norm buffers.normalized inverse thread 256

@[always_inline]
private def deviceSilu (value : Float32) : Cuda.DeviceM Float32 := do
  return value / (one + (← Cuda.fastExp (-value)))

@[always_inline]
private partial def activateLoop (buffers : Buffers) (index stride : UInt32) :
    Cuda.DeviceM Unit := do
  if index < intermediateSize then
    let gate := (← Cuda.loadBFloat16 buffers.gate index.toUSize).toFloat32
    let up := (← Cuda.loadBFloat16 buffers.up index.toUSize).toFloat32
    Cuda.storeBFloat16 buffers.intermediate index.toUSize
      (Cuda.BFloat16.ofFloat32 ((← deviceSilu gate) * up))
    activateLoop buffers (index + stride) stride

@[always_inline]
private partial def residualLoop (buffers : Buffers) (index stride : UInt32) :
    Cuda.DeviceM Unit := do
  if index < hiddenSize then
    let residual := (← Cuda.loadBFloat16 buffers.residual index.toUSize).toFloat32
    let projected := (← Cuda.loadBFloat16 buffers.projected index.toUSize).toFloat32
    Cuda.storeBFloat16 buffers.output index.toUSize
      (Cuda.BFloat16.ofFloat32 (residual + projected))
    residualLoop buffers (index + stride) stride

@[always_inline, convergent]
private def runDualProjection (buffers : Buffers) (weights : Weights)
    (shared : Cuda.DevicePtr Float32) (block blocks thread : UInt32) : Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  projectDualRows buffers.normalized weights.gate weights.up buffers.gate buffers.up
    intermediateSize hiddenSize scratch block blocks thread

@[always_inline, convergent]
private def runDownProjection (buffers : Buffers) (weights : Weights)
    (shared : Cuda.DevicePtr Float32) (block blocks thread : UInt32) : Cuda.DeviceM Unit := do
  let scratch : Cuda.Collective.BlockScratch 8 := shared
  projectRows buffers.intermediate weights.down buffers.projected hiddenSize intermediateSize
    scratch block blocks thread

@[cuda_kernel]
def qwen36NormStep (norm : Cuda.DevicePtr Cuda.BFloat16)
    (residual normalized gate up intermediate projected output :
      Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runNorm { residual, normalized, gate, up, intermediate, projected, output }
    { norm, gate := norm, up := norm, down := norm } shared (← Cuda.blockIdxX) (← Cuda.threadIdxX)

@[cuda_kernel]
def qwen36DualProjectionStep (gateWeight upWeight : Cuda.DevicePtr Cuda.BFloat16)
    (residual normalized gate up intermediate projected output :
      Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runDualProjection { residual, normalized, gate, up, intermediate, projected, output }
    { norm := gateWeight, gate := gateWeight, up := upWeight, down := gateWeight } shared
    (← Cuda.blockIdxX) (← Cuda.gridDimX) (← Cuda.threadIdxX)

@[cuda_kernel]
def qwen36ActivationStep (residual normalized gate up intermediate projected output :
    Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  activateLoop { residual, normalized, gate, up, intermediate, projected, output }
    (← Cuda.globalThreadIdxX) (← Cuda.globalThreadCountX)

@[cuda_kernel]
def qwen36DownProjectionStep (downWeight : Cuda.DevicePtr Cuda.BFloat16)
    (residual normalized gate up intermediate projected output :
      Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  runDownProjection { residual, normalized, gate, up, intermediate, projected, output }
    { norm := downWeight, gate := downWeight, up := downWeight, down := downWeight } shared
    (← Cuda.blockIdxX) (← Cuda.gridDimX) (← Cuda.threadIdxX)

@[cuda_kernel]
def qwen36ResidualStep (residual normalized gate up intermediate projected output :
    Cuda.DevicePtr Cuda.BFloat16) : Cuda.DeviceM Unit := do
  residualLoop { residual, normalized, gate, up, intermediate, projected, output }
    (← Cuda.globalThreadIdxX) (← Cuda.globalThreadCountX)

@[cuda_grid_persistent]
def qwen36MLP (item : TokenItem)
    (norm gateWeight upWeight downWeight : Cuda.DevicePtr Cuda.BFloat16)
    (residual normalized gate up intermediate projected output :
      Cuda.DevicePtr Cuda.BFloat16) (counts : Cuda.DevicePtr UInt32) : Cuda.DeviceM Unit := do
  let shared ← Cuda.dynamicShared (α := Float32)
  let block ← Cuda.blockIdxX
  let blocks ← Cuda.gridDimX
  let thread ← Cuda.threadIdxX
  let weights : Weights := { norm, gate := gateWeight, up := upWeight, down := downWeight }
  let buffers : Buffers := { residual, normalized, gate, up, intermediate, projected, output }
  runNorm buffers weights shared block thread
  Cuda.gridSync
  runDualProjection buffers weights shared block blocks thread
  Cuda.gridSync
  activateLoop buffers (block * 256 + thread) (blocks * 256)
  Cuda.gridSync
  runDownProjection buffers weights shared block blocks thread
  Cuda.gridSync
  residualLoop buffers (block * 256 + thread) (blocks * 256)
  Cuda.gridSync
  if block == 0 && thread == 0 then
    Cuda.atomicAddUInt32At_ counts item.descriptor 1

private structure DeviceBuffers where
  norm : Cuda.Buffer Cuda.BFloat16
  gateWeight : Cuda.Buffer Cuda.BFloat16
  upWeight : Cuda.Buffer Cuda.BFloat16
  downWeight : Cuda.Buffer Cuda.BFloat16
  residual : Cuda.Buffer Cuda.BFloat16
  normalized : Cuda.Buffer Cuda.BFloat16
  gate : Cuda.Buffer Cuda.BFloat16
  up : Cuda.Buffer Cuda.BFloat16
  intermediate : Cuda.Buffer Cuda.BFloat16
  projected : Cuda.Buffer Cuda.BFloat16
  output : Cuda.Buffer Cuda.BFloat16
  counts : Cuda.Buffer UInt32

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

private def residualValue (index : Nat) : Float32 :=
  signedValue (index * 11 + 5) 31 15 32

private def normValue (index : Nat) : Float32 :=
  signedValue (index * 7 + 3) 17 8 128

private def rounded (value : Float32) : Float32 :=
  (Cuda.BFloat16.ofFloat32 value).toFloat32

private def selectedColumnHost (row columns salt : Nat) : Nat :=
  (row * (salt * 2 + 3) + salt * 17 + 5) % columns

private def selectedWeightHost (row salt : Nat) : Float32 :=
  let magnitude := if salt % 2 == 0 then Float32.ofBits 0x3f000000
    else Float32.ofBits 0x3e800000
  if (row + salt) % 2 == 0 then magnitude else -magnitude

private def initializeBuffers (stream : Cuda.Stream) : IO DeviceBuffers := do
  let norm ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let gateWeight ← Cuda.Buffer.alloc Cuda.BFloat16 (intermediateSizeNat * hiddenSizeNat).toUSize
  let upWeight ← Cuda.Buffer.alloc Cuda.BFloat16 (intermediateSizeNat * hiddenSizeNat).toUSize
  let downWeight ← Cuda.Buffer.alloc Cuda.BFloat16 (hiddenSizeNat * intermediateSizeNat).toUSize
  let residual ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let normalized ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let gate ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSizeNat.toUSize
  let up ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSizeNat.toUSize
  let intermediate ← Cuda.Buffer.alloc Cuda.BFloat16 intermediateSizeNat.toUSize
  let projected ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let output ← Cuda.Buffer.alloc Cuda.BFloat16 hiddenSizeNat.toUSize
  let counts ← Cuda.Buffer.alloc UInt32 1
  norm.copyFrom (makeBFloatData hiddenSizeNat normValue) stream
  residual.copyFrom (makeBFloatData hiddenSizeNat residualValue) stream
  normalized.copyFrom (zeroBytes (hiddenSizeNat * 2)) stream
  gate.copyFrom (zeroBytes (intermediateSizeNat * 2)) stream
  up.copyFrom (zeroBytes (intermediateSizeNat * 2)) stream
  intermediate.copyFrom (zeroBytes (intermediateSizeNat * 2)) stream
  projected.copyFrom (zeroBytes (hiddenSizeNat * 2)) stream
  output.copyFrom (zeroBytes (hiddenSizeNat * 2)) stream
  counts.copyFrom (zeroBytes 4) stream
  return {
    norm, gateWeight, upWeight, downWeight, residual, normalized, gate, up, intermediate,
    projected, output, counts
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
  blockArenaBytes := 8 * 1024
}

private def persistentConfig : Cuda.GridPersistentConfig := {
  grid := { x := blockCount }
  block := { x := 256 }
  sharedMemoryBytes := (8 * 4).toUSize
  blockArenaBytes := 8 * 1024
  queueCapacity := 1
}

private def waitKernel (label : String) (handle : Cuda.KernelHandle) : IO Unit := do
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"{label} failed: {repr status}"

private def runSeparate (buffers : DeviceBuffers) : IO Unit := do
  waitKernel "Qwen3.6 MLP RMSNorm" <| ← qwen36NormStep.launch launchConfig buffers.norm
    buffers.residual buffers.normalized buffers.gate buffers.up buffers.intermediate
    buffers.projected buffers.output
  waitKernel "Qwen3.6 MLP gate/up projections" <|
    ← qwen36DualProjectionStep.launch launchConfig buffers.gateWeight buffers.upWeight
      buffers.residual buffers.normalized buffers.gate buffers.up buffers.intermediate
      buffers.projected buffers.output
  waitKernel "Qwen3.6 MLP SwiGLU" <| ← qwen36ActivationStep.launch launchConfig
    buffers.residual buffers.normalized buffers.gate buffers.up buffers.intermediate
    buffers.projected buffers.output
  waitKernel "Qwen3.6 MLP down projection" <|
    ← qwen36DownProjectionStep.launch launchConfig buffers.downWeight buffers.residual
      buffers.normalized buffers.gate buffers.up buffers.intermediate buffers.projected buffers.output
  waitKernel "Qwen3.6 MLP residual" <| ← qwen36ResidualStep.launch launchConfig
    buffers.residual buffers.normalized buffers.gate buffers.up buffers.intermediate
    buffers.projected buffers.output

private def runPersistent (buffers : DeviceBuffers) : IO Unit := do
  let handle ← qwen36MLP.start persistentConfig buffers.norm buffers.gateWeight buffers.upWeight
    buffers.downWeight buffers.residual buffers.normalized buffers.gate buffers.up
    buffers.intermediate buffers.projected buffers.output buffers.counts
  qwen36MLP.enqueue handle { descriptor := 0 }
  handle.shutdown
  let status ← handle.wait
  unless status.state == .completed do
    throw <| IO.userError s!"Qwen3.6 MLP megakernel failed: {repr status}"

private def hostOracle : Array Float32 := Id.run do
  let residual := (List.range hiddenSizeNat).map (fun index => rounded (residualValue index)) |>.toArray
  let mut squareSum := zero
  for value in residual do
    squareSum := Cuda.fma value value squareSum
  let inverse := one / Float32.sqrt (squareSum * inverseHiddenSize + rmsEpsilon)
  let normalized := (List.range hiddenSizeNat).map (fun index =>
    rounded (residual[index]! * inverse * (one + rounded (normValue index)))) |>.toArray
  let gate := (List.range intermediateSizeNat).map (fun row =>
    rounded (selectedWeightHost row 1 * normalized[selectedColumnHost row hiddenSizeNat 1]!))
    |>.toArray
  let up := (List.range intermediateSizeNat).map (fun row =>
    rounded (selectedWeightHost row 2 * normalized[selectedColumnHost row hiddenSizeNat 2]!))
    |>.toArray
  let intermediate := (List.range intermediateSizeNat).map (fun row =>
    rounded ((gate[row]! / (one + Float32.exp (-gate[row]!))) * up[row]!)) |>.toArray
  return (List.range hiddenSizeNat).map (fun row =>
    let projected := rounded (selectedWeightHost row 3 *
      intermediate[selectedColumnHost row intermediateSizeNat 3]!)
    rounded (residual[row]! + projected)) |>.toArray

private def requireClose (actual : ByteArray) (expected : Array Float32)
    (tolerance : Float32) : IO Unit := do
  let mut maximumError := zero
  let mut maximumIndex := 0
  for index in [:expected.size] do
    let error := Float32.abs (readBFloat actual index - expected[index]!)
    if error > maximumError then
      maximumError := error
      maximumIndex := index
  unless maximumError <= tolerance do
    throw <| IO.userError <| s!"MLP output differs from host Lean oracle at {maximumIndex}: " ++
      s!"actual={readBFloat actual maximumIndex}, expected={expected[maximumIndex]!}, " ++
      s!"error={maximumError}, tolerance={tolerance}"
  IO.println s!"Qwen3.6 MLP host-oracle max error: {maximumError}"

def main : IO Unit := do
  let stream ← Cuda.Stream.default
  let separate ← initializeBuffers stream
  let persistent ← initializeBuffers stream
  stream.synchronize
  for buffers in [separate, persistent] do
    waitKernel "Qwen3.6 gate matrix initialization" <|
      ← initializeMatrix.launch initializerConfig intermediateSize hiddenSize 1 buffers.gateWeight
    waitKernel "Qwen3.6 up matrix initialization" <|
      ← initializeMatrix.launch initializerConfig intermediateSize hiddenSize 2 buffers.upWeight
    waitKernel "Qwen3.6 down matrix initialization" <|
      ← initializeMatrix.launch initializerConfig hiddenSize intermediateSize 3 buffers.downWeight
  runSeparate separate
  runPersistent persistent
  let separateOutput ← separate.output.copyTo stream
  let persistentOutput ← persistent.output.copyTo stream
  unless separateOutput == persistentOutput do
    throw <| IO.userError "persistent and separate Qwen3.6 MLP outputs differ"
  let counts ← persistent.counts.copyTo stream
  unless readUInt32 counts 0 == 1 do
    throw <| IO.userError "Qwen3.6 MLP descriptor did not execute exactly once"
  requireClose persistentOutput hostOracle (Float32.ofBits 0x3c000000)
  IO.println "Lean Qwen3.6 exact-shape RMSNorm/SwiGLU/residual megakernel ok"

end Qwen36MLP

def main : IO Unit := Qwen36MLP.main
