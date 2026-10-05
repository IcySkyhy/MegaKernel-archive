/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-!
# Qwen3.6 mixed-adapter LoRA device gate

Checks interleaved adapter forward/VJPs and one masked AdamW update against the deterministic
NumPy oracle. It also proves sequence-permutation invariance, the no-adapter route, frozen-base
preservation, and byte-exact isolation of the masked adapter and its optimizer state.
-/

namespace Qwen36LoRAGate

open Cuda.Qwen36.LoRA

private def batch : Nat := 4
private def sequence : Nat := 2
private def rows : Nat := batch * sequence
private def adapters : Nat := 2
private def rank : Nat := 16
private def inputFeatures : Nat := 16
private def outputFeatures : Nat := 16
private def rowElements : Nat := rows * inputFeatures
private def aElements : Nat := adapters * rank * inputFeatures
private def bElements : Nat := adapters * outputFeatures * rank

private def descriptor : Descriptor := {
  batchSize := batch
  sequenceLength := sequence
  adapterCount := adapters
  rank
  inputFeatures
  outputFeatures
  alpha := 16
  adapterIds := #[0, 1, 0, 1]
}

private def fixture (directory name : String) (bytes : Nat) : IO ByteArray := do
  let value ← IO.FS.readBinFile (directory ++ "/" ++ name)
  LeanTest.assertTrue (value.size == bytes)
    s!"fixture {name}: expected {bytes} bytes, found {value.size}"
  return value

private def uploadF32 (directory name : String) (count : Nat) (stream : Cuda.Stream) :
    IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 count.toUSize
  buffer.copyFrom (← fixture directory name (count * 4)) stream
  return buffer

private def uploadU32 (directory name : String) (count : Nat) (stream : Cuda.Stream) :
    IO (Cuda.Buffer UInt32) := do
  let buffer ← Cuda.Buffer.alloc UInt32 count.toUSize
  buffer.copyFrom (← fixture directory name (count * 4)) stream
  return buffer

private def zerosF32 (count : Nat) (stream : Cuda.Stream) : IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 count.toUSize
  buffer.copyFrom (List.replicate (count * 4) 0).toByteArray stream
  return buffer

private def readFloat32 (bytes : ByteArray) (index : Nat) : Float32 :=
  let offset := index * 4
  Float32.ofBits <| bytes[offset]!.toUInt32 |||
    (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def checkClose (label : String) (actual expected : ByteArray) (count : Nat)
    (tolerance : Float32 := 3e-5) : IO Unit := do
  let mut worst : Float32 := 0
  let mut worstAt : Nat := 0
  for index in [:count] do
    let error := Float32.abs (readFloat32 actual index - readFloat32 expected index)
    if error > worst then
      worst := error
      worstAt := index
  unless worst ≤ tolerance do
    LeanTest.fail
      (s!"{label}: max abs err {worst} at {worstAt} exceeds {tolerance} " ++
       s!"(actual {readFloat32 actual worstAt}, expected {readFloat32 expected worstAt})")

private def checkBuffer (directory label fixtureName : String) (buffer : Cuda.Buffer Float32)
    (count : Nat) (stream : Cuda.Stream) (tolerance : Float32 := 3e-5) : IO Unit := do
  checkClose label (← buffer.copyTo stream) (← fixture directory fixtureName (count * 4)) count
    tolerance

private def checkExact (label : String) (actual expected : ByteArray) : IO Unit := do
  LeanTest.assertTrue (actual == expected) s!"{label}: bytes changed"

private def checkExactRange (label : String) (actual expected : ByteArray)
    (start count : Nat) : IO Unit := do
  for index in [:count] do
    LeanTest.assertTrue (actual[start + index]! == expected[start + index]!)
      s!"{label}: byte {index} changed"

private structure Result where
  rankActivation : Cuda.Buffer Float32
  output : Cuda.Buffer Float32
  inputGradient : Cuda.Buffer Float32
  adapterAGradient : Cuda.Buffer Float32
  adapterBGradient : Cuda.Buffer Float32

private def evaluate (directory label inputName idsName outputGradientName rankName outputName
    inputGradientName aGradientName bGradientName : String)
    (base adapterA adapterB : Cuda.Buffer Float32) (shape : Shape) (stream : Cuda.Stream) :
    IO Result := do
  let input ← uploadF32 directory inputName rowElements stream
  let ids ← uploadU32 directory idsName batch stream
  let outputGradient ← uploadF32 directory outputGradientName (rows * outputFeatures) stream
  let rankActivation ← zerosF32 (rows * rank) stream
  let output ← zerosF32 (rows * outputFeatures) stream
  let inputGradient ← zerosF32 rowElements stream
  let adapterAGradient ← zerosF32 aElements stream
  let adapterBGradient ← zerosF32 bElements stream
  forwardF32 stream input base adapterA adapterB ids rankActivation output shape
  backwardF32 stream input base adapterA adapterB ids rankActivation outputGradient inputGradient
    adapterAGradient adapterBGradient shape
  checkBuffer directory (label ++ " rank activation") rankName rankActivation (rows * rank) stream
  checkBuffer directory (label ++ " output") outputName output (rows * outputFeatures) stream
  checkBuffer directory (label ++ " input VJP") inputGradientName inputGradient rowElements stream
  checkBuffer directory (label ++ " adapter-A VJP") aGradientName adapterAGradient aElements stream
  checkBuffer directory (label ++ " adapter-B VJP") bGradientName adapterBGradient bElements stream
  return { rankActivation, output, inputGradient, adapterAGradient, adapterBGradient }

private def checkPermutationRows (label : String) (original permuted : ByteArray) : IO Unit := do
  let order : Array Nat := #[2, 0, 3, 1]
  let mut worst : Float32 := 0
  for permutedBatch in [:batch] do
    for token in [:sequence] do
      for column in [:outputFeatures] do
        let originalRow := order[permutedBatch]! * sequence + token
        let permutedRow := permutedBatch * sequence + token
        let error := Float32.abs
          (readFloat32 original (originalRow * outputFeatures + column) -
           readFloat32 permuted (permutedRow * outputFeatures + column))
        if error > worst then
          worst := error
  LeanTest.assertTrue (worst == 0) s!"{label}: row-unpermuted max abs err {worst}"

private def optimizerConfig : Cuda.Training.AdamW.Config := {
  learningRate := 2e-3
  beta1 := 0.9
  beta2 := 0.999
  inverseBiasCorrection1 := 10
  inverseBiasCorrection2 := 1000
  epsilon := 1e-8
  weightDecay := 0.01
}

@[test]
def mixedAdapterForwardVjpAndUpdate : IO Unit := do
  let directory ← match ← IO.getEnv "QWEN36_LORA_FIXTURES" with
    | some directory => pure directory
    | none => LeanTest.fail "QWEN36_LORA_FIXTURES is not set"
  let shape ← match descriptor.toShape with
    | .ok shape => pure shape
    | .error error => LeanTest.fail error
  let stream ← Cuda.Stream.default
  let base ← uploadF32 directory "lora.base.16x16.f32.bin"
    (outputFeatures * inputFeatures) stream
  let adapterA ← uploadF32 directory "lora.adapter_a.2x16x16.f32.bin" aElements stream
  let adapterB ← uploadF32 directory "lora.adapter_b.2x16x16.f32.bin" bElements stream
  let baseBefore ← base.copyTo stream
  let adapterABefore ← adapterA.copyTo stream
  let adapterBBefore ← adapterB.copyTo stream

  let original ← evaluate directory "interleaved"
    "lora.input.8x16.f32.bin" "lora.adapter_ids.4.u32.bin"
    "lora.output_gradient.8x16.f32.bin" "lora.rank_activation.8x16.f32.bin"
    "lora.output.8x16.f32.bin" "lora.input_gradient.8x16.f32.bin"
    "lora.adapter_a_gradient.2x16x16.f32.bin"
    "lora.adapter_b_gradient.2x16x16.f32.bin" base adapterA adapterB shape stream

  let additiveInput ← uploadF32 directory "lora.input.8x16.f32.bin" rowElements stream
  let additiveIds ← uploadU32 directory "lora.adapter_ids.4.u32.bin" batch stream
  let additiveOutputGradient ← uploadF32 directory
    "lora.output_gradient.8x16.f32.bin" (rows * outputFeatures) stream
  let additiveRank ← zerosF32 (rows * rank) stream
  let additiveOutput ← zerosF32 (rows * outputFeatures) stream
  (← Cuda.Qwen36.Linear.forwardF32 stream additiveInput base additiveOutput rows.toUInt32
    inputFeatures.toUInt32 outputFeatures.toUInt32).waitChecked
    "Qwen3.6 additive-LoRA frozen-base projection"
  addForwardF32 stream additiveInput adapterA adapterB additiveIds additiveRank additiveOutput shape
  checkClose "additive LoRA matches fused projection" (← additiveOutput.copyTo stream)
    (← original.output.copyTo stream) (rows * outputFeatures)
  let additiveAGradient ← zerosF32 aElements stream
  let additiveBGradient ← zerosF32 bElements stream
  backwardAdaptersF32 stream additiveInput adapterB additiveIds additiveRank
    additiveOutputGradient additiveAGradient additiveBGradient shape
  checkClose "adapter-only A VJP matches fused VJP" (← additiveAGradient.copyTo stream)
    (← original.adapterAGradient.copyTo stream) aElements
  checkClose "adapter-only B VJP matches fused VJP" (← additiveBGradient.copyTo stream)
    (← original.adapterBGradient.copyTo stream) bElements

  let permuted ← evaluate directory "permuted"
    "lora.permuted_input.8x16.f32.bin" "lora.permuted_adapter_ids.4.u32.bin"
    "lora.permuted_output_gradient.8x16.f32.bin"
    "lora.permuted_rank_activation.8x16.f32.bin" "lora.permuted_output.8x16.f32.bin"
    "lora.permuted_input_gradient.8x16.f32.bin"
    "lora.permuted_adapter_a_gradient.2x16x16.f32.bin"
    "lora.permuted_adapter_b_gradient.2x16x16.f32.bin" base adapterA adapterB shape stream
  checkPermutationRows "LoRA forward permutation invariance"
    (← original.output.copyTo stream) (← permuted.output.copyTo stream)
  checkPermutationRows "LoRA input-VJP permutation invariance"
    (← original.inputGradient.copyTo stream) (← permuted.inputGradient.copyTo stream)
  checkClose "LoRA adapter-A gradient permutation invariance"
    (← original.adapterAGradient.copyTo stream) (← permuted.adapterAGradient.copyTo stream)
    aElements
  checkClose "LoRA adapter-B gradient permutation invariance"
    (← original.adapterBGradient.copyTo stream) (← permuted.adapterBGradient.copyTo stream)
    bElements

  let sentinelInput ← uploadF32 directory "lora.input.8x16.f32.bin" rowElements stream
  let sentinelIds ← uploadU32 directory "lora.sentinel_adapter_ids.4.u32.bin" batch stream
  let sentinelRank ← zerosF32 (rows * rank) stream
  let sentinelOutput ← zerosF32 (rows * outputFeatures) stream
  forwardF32 stream sentinelInput base adapterA adapterB sentinelIds sentinelRank sentinelOutput shape
  checkBuffer directory "no-adapter rank activation"
    "lora.sentinel_rank_activation.8x16.f32.bin" sentinelRank (rows * rank) stream
  checkBuffer directory "no-adapter frozen-base output" "lora.sentinel_output.8x16.f32.bin"
    sentinelOutput (rows * outputFeatures) stream

  let masked ← evaluate directory "masked adapter"
    "lora.input.8x16.f32.bin" "lora.adapter_ids.4.u32.bin"
    "lora.masked_output_gradient.8x16.f32.bin" "lora.rank_activation.8x16.f32.bin"
    "lora.output.8x16.f32.bin" "lora.masked_input_gradient.8x16.f32.bin"
    "lora.masked_adapter_a_gradient.2x16x16.f32.bin"
    "lora.masked_adapter_b_gradient.2x16x16.f32.bin" base adapterA adapterB shape stream
  let adapterAFirst ← uploadF32 directory "lora.adapter_a_first.2x16x16.f32.bin"
    aElements stream
  let adapterASecond ← uploadF32 directory "lora.adapter_a_second.2x16x16.f32.bin"
    aElements stream
  let adapterBFirst ← uploadF32 directory "lora.adapter_b_first.2x16x16.f32.bin"
    bElements stream
  let adapterBSecond ← uploadF32 directory "lora.adapter_b_second.2x16x16.f32.bin"
    bElements stream
  let updateMask ← uploadU32 directory "lora.update_mask.2.u32.bin" adapters stream
  let adapterAFirstBefore ← adapterAFirst.copyTo stream
  let adapterASecondBefore ← adapterASecond.copyTo stream
  let adapterBFirstBefore ← adapterBFirst.copyTo stream
  let adapterBSecondBefore ← adapterBSecond.copyTo stream
  let update ← adamWF32 stream optimizerConfig adapterA adapterB masked.adapterAGradient
    masked.adapterBGradient adapterAFirst adapterASecond adapterBFirst adapterBSecond updateMask shape
  update.waitChecked "Qwen3.6 masked mixed-adapter AdamW"

  checkBuffer directory "updated adapter A" "lora.updated_adapter_a.2x16x16.f32.bin"
    adapterA aElements stream
  checkBuffer directory "updated adapter B" "lora.updated_adapter_b.2x16x16.f32.bin"
    adapterB bElements stream
  checkBuffer directory "updated adapter-A first moment"
    "lora.updated_adapter_a_first.2x16x16.f32.bin" adapterAFirst aElements stream
  checkBuffer directory "updated adapter-A second moment"
    "lora.updated_adapter_a_second.2x16x16.f32.bin" adapterASecond aElements stream
  checkBuffer directory "updated adapter-B first moment"
    "lora.updated_adapter_b_first.2x16x16.f32.bin" adapterBFirst bElements stream
  checkBuffer directory "updated adapter-B second moment"
    "lora.updated_adapter_b_second.2x16x16.f32.bin" adapterBSecond bElements stream

  checkExact "frozen base weight" (← base.copyTo stream) baseBefore
  let adapterStrideBytes := rank * inputFeatures * 4
  checkExactRange "masked adapter A" (← adapterA.copyTo stream) adapterABefore
    adapterStrideBytes adapterStrideBytes
  checkExactRange "masked adapter B" (← adapterB.copyTo stream) adapterBBefore
    adapterStrideBytes adapterStrideBytes
  checkExactRange "masked adapter-A first moment" (← adapterAFirst.copyTo stream)
    adapterAFirstBefore adapterStrideBytes adapterStrideBytes
  checkExactRange "masked adapter-A second moment" (← adapterASecond.copyTo stream)
    adapterASecondBefore adapterStrideBytes adapterStrideBytes
  checkExactRange "masked adapter-B first moment" (← adapterBFirst.copyTo stream)
    adapterBFirstBefore adapterStrideBytes adapterStrideBytes
  checkExactRange "masked adapter-B second moment" (← adapterBSecond.copyTo stream)
    adapterBSecondBefore adapterStrideBytes adapterStrideBytes

end Qwen36LoRAGate
