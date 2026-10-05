/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-!
# Qwen3.6 complete model gate

Drives embedding, all four tiny hybrid decoder layers, final normalization, the language-model
head, mean cross-entropy, and the entire reverse pass against the deterministic Float32 oracle.
-/

namespace Qwen36ModelGate

open Cuda.Qwen36.Model
open Cuda.Qwen36.Training

private def tokens : Nat := 8
private def hidden : Nat := 256
private def intermediate : Nat := 512
private def vocabulary : Nat := 512
private def keyHeads : Nat := 2
private def valueHeads : Nat := 4
private def keyWidth : Nat := 64
private def valueWidth : Nat := 64
private def queryHeads : Nat := 4
private def keyValueHeads : Nat := 2
private def attentionWidth : Nat := 64
private def convolutionWidth : Nat := 512
private def hiddenElements : Nat := tokens * hidden
private def intermediateElements : Nat := tokens * intermediate
private def compactElements : Nat := tokens * keyHeads * keyWidth
private def valueElements : Nat := tokens * valueHeads * valueWidth
private def stateElements : Nat := valueHeads * keyWidth * valueWidth
private def queryElements : Nat := tokens * queryHeads * attentionWidth
private def keyValueElements : Nat := tokens * keyValueHeads * attentionWidth
private def probabilityElements : Nat := queryHeads * tokens * tokens

private def shape : ShapeF32 := {
  tokens := 8, hidden := 256, intermediate := 512, vocabulary := 512,
  linearKeyHeads := 2, linearValueHeads := 4, linearKeyWidth := 64,
  linearValueWidth := 64, attentionQueryHeads := 4, attentionKeyValueHeads := 2,
  attentionHeadWidth := 64, rotaryHalf := 8, epsilon := 1e-6,
  queryScale := 0.125, attentionScale := 0.125
}

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def pushUInt16 (bytes : ByteArray) (value : UInt16) : ByteArray :=
  (bytes.push value.toUInt8).push (value >>> 8).toUInt8

private def readFloat32 (bytes : ByteArray) (index : Nat) : Float32 :=
  let offset := index * 4
  Float32.ofBits <| bytes[offset]!.toUInt32 |||
    (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def fixture (dir name : String) (count : Nat) : IO ByteArray := do
  let bytes ← IO.FS.readBinFile (dir ++ "/" ++ name ++ ".f32.bin")
  unless bytes.size == count * 4 do
    throw <| IO.userError
      s!"fixture {name}: expected {count * 4} bytes, found {bytes.size}"
  return bytes

private def upload (dir name : String) (count : Nat) (stream : Cuda.Stream) :
    IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 count.toUSize
  buffer.copyFrom (← fixture dir name count) stream
  return buffer

private def uploadIds (dir name : String) (count : Nat) (stream : Cuda.Stream) :
    IO (Cuda.Buffer UInt32) := do
  let source ← fixture dir name count
  let mut packed := ByteArray.emptyWithCapacity (count * 4)
  for i in [:count] do
    packed := pushUInt32 packed (readFloat32 source i).toUInt32
  let buffer ← Cuda.Buffer.alloc UInt32 count.toUSize
  buffer.copyFrom packed stream
  return buffer

private def zeros (count : Nat) (stream : Cuda.Stream) : IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 count.toUSize
  buffer.copyFrom (List.replicate (count * 4) 0).toByteArray stream
  return buffer

private def uploadUInt32Values (values : Array UInt32) (stream : Cuda.Stream) :
    IO (Cuda.Buffer UInt32) := do
  let mut bytes := ByteArray.emptyWithCapacity (values.size * 4)
  for value in values do
    bytes := pushUInt32 bytes value
  let buffer ← Cuda.Buffer.alloc UInt32 values.size.toUSize
  buffer.copyFrom bytes stream
  return buffer

private def adapterValue (salt index : Nat) : Float32 :=
  let magnitude := ((index * 13 + salt * 7) % 17 + 1).toUInt32.toFloat32 / 4096
  if (index + salt) % 2 == 0 then magnitude else -magnitude

private def generatedAdapter (count salt : Nat) (stream : Cuda.Stream) :
    IO (Cuda.Buffer Float32) := do
  let mut bytes := ByteArray.emptyWithCapacity (count * 4)
  for index in [:count] do
    bytes := pushUInt32 bytes (adapterValue salt index).toBits
  let buffer ← Cuda.Buffer.alloc Float32 count.toUSize
  buffer.copyFrom bytes stream
  return buffer

private def freezeBF16 (baseWeight : Cuda.Buffer Float32) (elements : Nat)
    (stream : Cuda.Stream) : IO Cuda.Qwen36.Primitives.FrozenBFloat16Weight := do
  let source ← baseWeight.copyTo stream
  let mut packed := ByteArray.emptyWithCapacity (elements * 2)
  for index in [:elements] do
    packed := pushUInt16 packed (Cuda.BFloat16.ofFloat32 (readFloat32 source index)).bits
  let owner ← Cuda.Buffer.alloc UInt8 packed.size.toUSize
  owner.copyFrom packed stream
  return { owner, byteOffset := 0, byteCount := packed.size.toUInt64 }

private def adaptProjection (baseWeight : Cuda.Buffer Float32)
    (adapterIds updateMask : Cuda.Buffer UInt32) (inputFeatures outputFeatures salt : Nat)
    (stream : Cuda.Stream) :
    IO (Cuda.Qwen36.Projection.WeightsF32 × Cuda.Qwen36.Projection.AdapterF32) := do
  let adapterCount : Nat := 2
  let rank : Nat := 16
  let aElements := adapterCount * rank * inputFeatures
  let bElements := adapterCount * outputFeatures * rank
  let adapterA ← generatedAdapter aElements salt stream
  let adapterB ← generatedAdapter bElements (salt + 1) stream
  let rankActivation ← zeros (tokens * rank) stream
  let adapterAGradient ← zeros aElements stream
  let adapterBGradient ← zeros bElements stream
  let adapterAFirstMoment ← zeros aElements stream
  let adapterASecondMoment ← zeros aElements stream
  let adapterBFirstMoment ← zeros bElements stream
  let adapterBSecondMoment ← zeros bElements stream
  let shape : Cuda.Qwen36.LoRA.Shape := {
    rows := tokens.toUInt32
    sequenceLength := 2
    adapterCount := adapterCount.toUInt32
    rank := rank.toUInt32
    inputFeatures := inputFeatures.toUInt32
    outputFeatures := outputFeatures.toUInt32
    scale := 1
  }
  let adapter : Cuda.Qwen36.Projection.AdapterF32 := {
    adapterA, adapterB, adapterIds, rankActivation, adapterAGradient, adapterBGradient,
    adapterAFirstMoment, adapterASecondMoment, adapterBFirstMoment, adapterBSecondMoment,
    updateMask, weightDecay := 0, shape
  }
  let frozen : Cuda.Qwen36.Linear.FrozenBFloat16F32 := {
    weight := ← freezeBF16 baseWeight (inputFeatures * outputFeatures) stream
    inputBF16 := ← Cuda.Buffer.alloc Cuda.BFloat16 (tokens * inputFeatures).toUSize
    outputGradientBF16 := ← Cuda.Buffer.alloc Cuda.BFloat16
      (tokens * outputFeatures).toUSize
  }
  return (.loraFrozenBF16 frozen adapter, adapter)

private def requireBaseWeight (projection : Cuda.Qwen36.Projection.WeightsF32) :
    IO (Cuda.Buffer Float32) :=
  match projection.baseWeight? with
  | some weight => pure weight
  | none => throw <| IO.userError "model regression expected a Float32 projection base"

private def adaptMLPWeights (weights : MLPWeightsF32)
    (adapterIds updateMask : Cuda.Buffer UInt32) (salt : Nat) (stream : Cuda.Stream) :
    IO (MLPWeightsF32 × Array Cuda.Qwen36.Projection.AdapterF32 ×
      Array (Cuda.Buffer Float32)) := do
  let gateBase ← requireBaseWeight weights.gate
  let upBase ← requireBaseWeight weights.up
  let downBase ← requireBaseWeight weights.down
  let (gate, gateAdapter) ← adaptProjection gateBase adapterIds updateMask hidden
    intermediate salt stream
  let (up, upAdapter) ← adaptProjection upBase adapterIds updateMask hidden
    intermediate (salt + 2) stream
  let (down, downAdapter) ← adaptProjection downBase adapterIds updateMask intermediate
    hidden (salt + 4) stream
  return ({ weights with gate, up, down }, #[gateAdapter, upAdapter, downAdapter],
    #[gateBase, upBase, downBase])

private def adaptDeltaWeights (weights : Cuda.Qwen36.DeltaNet.StageWeightsF32)
    (adapterIds updateMask : Cuda.Buffer UInt32) (salt : Nat) (stream : Cuda.Stream) :
    IO (Cuda.Qwen36.DeltaNet.StageWeightsF32 ×
      Array Cuda.Qwen36.Projection.AdapterF32 × Array (Cuda.Buffer Float32)) := do
  let queryKeyValueBase ← requireBaseWeight weights.queryKeyValue
  let zBase ← requireBaseWeight weights.z
  let bBase ← requireBaseWeight weights.b
  let aBase ← requireBaseWeight weights.a
  let outputBase ← requireBaseWeight weights.output
  let (queryKeyValue, queryKeyValueAdapter) ← adaptProjection
    queryKeyValueBase adapterIds updateMask hidden convolutionWidth salt stream
  let (z, zAdapter) ← adaptProjection zBase adapterIds updateMask hidden
    (valueHeads * valueWidth) (salt + 2) stream
  let (b, bAdapter) ← adaptProjection bBase adapterIds updateMask hidden valueHeads
    (salt + 4) stream
  let (a, aAdapter) ← adaptProjection aBase adapterIds updateMask hidden valueHeads
    (salt + 6) stream
  let (output, outputAdapter) ← adaptProjection outputBase adapterIds updateMask
    (valueHeads * valueWidth) hidden (salt + 8) stream
  return ({ weights with queryKeyValue, z, b, a, output },
    #[queryKeyValueAdapter, zAdapter, bAdapter, aAdapter, outputAdapter],
    #[queryKeyValueBase, zBase, bBase, aBase, outputBase])

private def adaptAttentionWeights (weights : Cuda.Qwen36.Attention.StageWeightsF32)
    (adapterIds updateMask : Cuda.Buffer UInt32) (salt : Nat) (stream : Cuda.Stream) :
    IO (Cuda.Qwen36.Attention.StageWeightsF32 ×
      Array Cuda.Qwen36.Projection.AdapterF32 × Array (Cuda.Buffer Float32)) := do
  let queryBase ← requireBaseWeight weights.query
  let keyBase ← requireBaseWeight weights.key
  let valueBase ← requireBaseWeight weights.value
  let outputBase ← requireBaseWeight weights.output
  let (query, queryAdapter) ← adaptProjection queryBase adapterIds updateMask hidden
    (2 * queryHeads * attentionWidth) salt stream
  let (key, keyAdapter) ← adaptProjection keyBase adapterIds updateMask hidden
    (keyValueHeads * attentionWidth) (salt + 2) stream
  let (value, valueAdapter) ← adaptProjection valueBase adapterIds updateMask hidden
    (keyValueHeads * attentionWidth) (salt + 4) stream
  let (output, outputAdapter) ← adaptProjection outputBase adapterIds updateMask
    (queryHeads * attentionWidth) hidden (salt + 6) stream
  return ({ weights with query, key, value, output },
    #[queryAdapter, keyAdapter, valueAdapter, outputAdapter],
    #[queryBase, keyBase, valueBase, outputBase])

private structure AdapterSnapshot where
  adapterA : ByteArray
  adapterB : ByteArray
  adapterAFirstMoment : ByteArray
  adapterASecondMoment : ByteArray
  adapterBFirstMoment : ByteArray
  adapterBSecondMoment : ByteArray

private def snapshotAdapter (adapter : Cuda.Qwen36.Projection.AdapterF32)
    (stream : Cuda.Stream) : IO AdapterSnapshot := do
  return {
    adapterA := ← adapter.adapterA.copyTo stream
    adapterB := ← adapter.adapterB.copyTo stream
    adapterAFirstMoment := ← adapter.adapterAFirstMoment.copyTo stream
    adapterASecondMoment := ← adapter.adapterASecondMoment.copyTo stream
    adapterBFirstMoment := ← adapter.adapterBFirstMoment.copyTo stream
    adapterBSecondMoment := ← adapter.adapterBSecondMoment.copyTo stream
  }

private def restoreAdapter (adapter : Cuda.Qwen36.Projection.AdapterF32)
    (snapshot : AdapterSnapshot) (stream : Cuda.Stream) : IO Unit := do
  adapter.adapterA.copyFrom snapshot.adapterA stream
  adapter.adapterB.copyFrom snapshot.adapterB stream
  adapter.adapterAFirstMoment.copyFrom snapshot.adapterAFirstMoment stream
  adapter.adapterASecondMoment.copyFrom snapshot.adapterASecondMoment stream
  adapter.adapterBFirstMoment.copyFrom snapshot.adapterBFirstMoment stream
  adapter.adapterBSecondMoment.copyFrom snapshot.adapterBSecondMoment stream

private def checkBytesExact (label : String) (actual expected : ByteArray) : IO Unit := do
  unless actual.size == expected.size do
    LeanTest.fail s!"{label}: byte count {actual.size} != {expected.size}"
  for offset in [:actual.size] do
    unless actual[offset]! == expected[offset]! do
      LeanTest.fail s!"{label}: byte {offset}: actual {actual[offset]!}, expected {expected[offset]!}"

private def checkAdapterSnapshotExact (label : String) (index : Nat)
    (actual expected : AdapterSnapshot) : IO Unit := do
  checkBytesExact s!"{label} projection {index} adapter A" actual.adapterA expected.adapterA
  checkBytesExact s!"{label} projection {index} adapter B" actual.adapterB expected.adapterB
  checkBytesExact s!"{label} projection {index} adapter-A first moment"
    actual.adapterAFirstMoment expected.adapterAFirstMoment
  checkBytesExact s!"{label} projection {index} adapter-A second moment"
    actual.adapterASecondMoment expected.adapterASecondMoment
  checkBytesExact s!"{label} projection {index} adapter-B first moment"
    actual.adapterBFirstMoment expected.adapterBFirstMoment
  checkBytesExact s!"{label} projection {index} adapter-B second moment"
    actual.adapterBSecondMoment expected.adapterBSecondMoment

private structure ParameterGroupSnapshot where
  parameter : ByteArray
  firstMoment : ByteArray
  secondMoment : ByteArray

private def snapshotParameterGroup (group : ParameterGroupF32) (stream : Cuda.Stream) :
    IO ParameterGroupSnapshot := do
  return {
    parameter := ← group.parameter.copyTo stream
    firstMoment := ← group.firstMoment.copyTo stream
    secondMoment := ← group.secondMoment.copyTo stream
  }

private def restoreParameterGroup (group : ParameterGroupF32) (snapshot : ParameterGroupSnapshot)
    (stream : Cuda.Stream) : IO Unit := do
  group.parameter.copyFrom snapshot.parameter stream
  group.firstMoment.copyFrom snapshot.firstMoment stream
  group.secondMoment.copyFrom snapshot.secondMoment stream

private def checkParameterGroupSnapshotExact (actual expected : ParameterGroupSnapshot) :
    IO Unit := do
  LeanTest.assertEqual actual.parameter expected.parameter
  LeanTest.assertEqual actual.firstMoment expected.firstMoment
  LeanTest.assertEqual actual.secondMoment expected.secondMoment

private def elementSlice (bytes : ByteArray) (start count : Nat) : ByteArray :=
  bytes.extract (start * 4) ((start + count) * 4)

private def checkMaskedAdapterUpdate (index : Nat)
    (adapter : Cuda.Qwen36.Projection.AdapterF32) (before : AdapterSnapshot)
    (stream : Cuda.Stream) : IO Unit := do
  let after ← snapshotAdapter adapter stream
  let aStride := adapter.shape.rank.toNat * adapter.shape.inputFeatures.toNat
  let bStride := adapter.shape.outputFeatures.toNat * adapter.shape.rank.toNat
  let enabledParametersChanged :=
    elementSlice after.adapterA 0 aStride != elementSlice before.adapterA 0 aStride ||
    elementSlice after.adapterB 0 bStride != elementSlice before.adapterB 0 bStride
  LeanTest.assertTrue enabledParametersChanged
    s!"projection {index}: enabled adapter parameters did not change"
  let enabledMomentsChanged :=
    elementSlice after.adapterAFirstMoment 0 aStride !=
        elementSlice before.adapterAFirstMoment 0 aStride ||
      elementSlice after.adapterASecondMoment 0 aStride !=
        elementSlice before.adapterASecondMoment 0 aStride ||
      elementSlice after.adapterBFirstMoment 0 bStride !=
        elementSlice before.adapterBFirstMoment 0 bStride ||
      elementSlice after.adapterBSecondMoment 0 bStride !=
        elementSlice before.adapterBSecondMoment 0 bStride
  LeanTest.assertTrue enabledMomentsChanged
    s!"projection {index}: enabled adapter optimizer state did not change"
  LeanTest.assertEqual (elementSlice after.adapterA aStride aStride)
    (elementSlice before.adapterA aStride aStride)
  LeanTest.assertEqual (elementSlice after.adapterB bStride bStride)
    (elementSlice before.adapterB bStride bStride)
  LeanTest.assertEqual (elementSlice after.adapterAFirstMoment aStride aStride)
    (elementSlice before.adapterAFirstMoment aStride aStride)
  LeanTest.assertEqual (elementSlice after.adapterASecondMoment aStride aStride)
    (elementSlice before.adapterASecondMoment aStride aStride)
  LeanTest.assertEqual (elementSlice after.adapterBFirstMoment bStride bStride)
    (elementSlice before.adapterBFirstMoment bStride bStride)
  LeanTest.assertEqual (elementSlice after.adapterBSecondMoment bStride bStride)
    (elementSlice before.adapterBSecondMoment bStride bStride)

private def scalarUInt32 (value : UInt32) (stream : Cuda.Stream) :
    IO (Cuda.Buffer UInt32) := do
  let buffer ← Cuda.Buffer.alloc UInt32 1
  buffer.copyFrom (pushUInt32 ByteArray.empty value) stream
  return buffer

private def scalarFloat32 (value : Float32) (stream : Cuda.Stream) :
    IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 1
  buffer.copyFrom (pushUInt32 ByteArray.empty value.toBits) stream
  return buffer

private def allocateAdamWSchedule (stream : Cuda.Stream) : IO AdamWScheduleF32 := do
  let step ← scalarUInt32 0 stream
  let beta1Power ← scalarFloat32 1 stream
  let beta2Power ← scalarFloat32 1 stream
  let inverseBiasCorrection1 ← scalarFloat32 0 stream
  let inverseBiasCorrection2 ← scalarFloat32 0 stream
  return {
    step, beta1Power, beta2Power, inverseBiasCorrection1, inverseBiasCorrection2
  }

private def allocateFullModelOptimizer (weights : WeightsF32) (backward : BackwardF32)
    (stream : Cuda.Stream) : IO FullModelOptimizerF32 := do
  let specs ← fullModelParameterSpecsF32 weights backward shape
  let mut states : Array ParameterStateF32 := #[]
  for spec in specs do
    let firstMoment ← zeros spec.elements.toNat stream
    let secondMoment ← zeros spec.elements.toNat stream
    states := states.push { firstMoment, secondMoment, weightDecay := 0 }
  return { states }

private def adamWHyperparameters : AdamWHyperparametersF32 := {
  learningRate := 0.001
  beta1 := 0.9
  beta2 := 0.99
  epsilon := 1e-8
}

private def checkClose (label : String) (actual expected : ByteArray) (count : Nat)
    (tolerance : Float32) : IO Unit := do
  let mut worst : Float32 := 0
  let mut worstAt : Nat := 0
  for index in [:count] do
    let error := Float32.abs (readFloat32 actual index - readFloat32 expected index)
    if error > worst then
      worst := error
      worstAt := index
  unless worst ≤ tolerance do
    throw <| IO.userError
      (s!"{label}: max abs err {worst} at {worstAt} exceeds {tolerance} " ++
       s!"(actual {readFloat32 actual worstAt}, expected {readFloat32 expected worstAt})")

private def checkBuffer (dir label fixtureName : String) (buffer : Cuda.Buffer Float32)
    (count : Nat) (stream : Cuda.Stream) : IO Unit := do
  checkClose label (← buffer.copyTo stream) (← fixture dir fixtureName count) count 3e-5

private def layerPrefix (layer : Nat) : String := s!"model.layers.{layer}."
private def fixturePrefix (layer : Nat) : String := s!"layer{layer}."
private def gradPrefix (layer : Nat) : String := s!"grad.model.layers.{layer}."

private def loadMLPWeights (dir : String) (layer : Nat) (stream : Cuda.Stream) :
    IO MLPWeightsF32 := do
  let base := layerPrefix layer
  let norm ← upload dir (base ++ "post_attention_layernorm.weight.256") hidden stream
  let gate ← upload dir (base ++ "mlp.gate_proj.weight.512x256")
    (intermediate * hidden) stream
  let up ← upload dir (base ++ "mlp.up_proj.weight.512x256")
    (intermediate * hidden) stream
  let down ← upload dir (base ++ "mlp.down_proj.weight.256x512")
    (hidden * intermediate) stream
  return { norm, gate, up, down }

private def loadDeltaWeights (dir : String) (layer : Nat) (stream : Cuda.Stream) :
    IO Cuda.Qwen36.DeltaNet.StageWeightsF32 := do
  let base := layerPrefix layer
  let inputNorm ← upload dir (base ++ "input_layernorm.weight.256") hidden stream
  let queryKeyValue ← upload dir (base ++ "linear_attn.in_proj_qkv.weight.512x256")
    (convolutionWidth * hidden) stream
  let z ← upload dir (base ++ "linear_attn.in_proj_z.weight.256x256")
    (hidden * hidden) stream
  let b ← upload dir (base ++ "linear_attn.in_proj_b.weight.4x256")
    (valueHeads * hidden) stream
  let a ← upload dir (base ++ "linear_attn.in_proj_a.weight.4x256")
    (valueHeads * hidden) stream
  let convolution ← upload dir (base ++ "linear_attn.conv1d.weight.512x4")
    (convolutionWidth * 4) stream
  let aLog ← upload dir (base ++ "linear_attn.A_log.4") valueHeads stream
  let dtBias ← upload dir (base ++ "linear_attn.dt_bias.4") valueHeads stream
  let gatedNorm ← upload dir (base ++ "linear_attn.norm.weight.64") valueWidth stream
  let output ← upload dir (base ++ "linear_attn.out_proj.weight.256x256")
    (hidden * hidden) stream
  return {
    inputNorm, queryKeyValue, z, b, a, convolution, aLog, dtBias, gatedNorm, output
  }

private def loadAttentionWeights (dir : String) (layer : Nat) (stream : Cuda.Stream) :
    IO Cuda.Qwen36.Attention.StageWeightsF32 := do
  let base := layerPrefix layer
  let inputNorm ← upload dir (base ++ "input_layernorm.weight.256") hidden stream
  let query ← upload dir (base ++ "self_attn.q_proj.weight.512x256")
    (2 * queryHeads * attentionWidth * hidden) stream
  let key ← upload dir (base ++ "self_attn.k_proj.weight.128x256")
    (keyValueHeads * attentionWidth * hidden) stream
  let value ← upload dir (base ++ "self_attn.v_proj.weight.128x256")
    (keyValueHeads * attentionWidth * hidden) stream
  let queryNorm ← upload dir (base ++ "self_attn.q_norm.weight.64") attentionWidth stream
  let keyNorm ← upload dir (base ++ "self_attn.k_norm.weight.64") attentionWidth stream
  let output ← upload dir (base ++ "self_attn.o_proj.weight.256x256")
    (hidden * hidden) stream
  let inverseFrequency ← zeros 8 stream
  let handle ← Cuda.Qwen36.Primitives.ropeInvFreq stream inverseFrequency 8 16 1e7
  handle.waitChecked "Qwen3.6 model inverse frequency"
  return { inputNorm, query, key, value, queryNorm, keyNorm, output, inverseFrequency }

private def allocMLPForward (stream : Cuda.Stream) : IO MLPForwardF32 := do
  let normalized ← zeros hiddenElements stream
  let inverseRms ← zeros tokens stream
  let gate ← zeros intermediateElements stream
  let up ← zeros intermediateElements stream
  let activation ← zeros intermediateElements stream
  let output ← zeros hiddenElements stream
  return { normalized, inverseRms, gate, up, activation, output }

private def allocDeltaForward (input : Cuda.Buffer Float32) (stream : Cuda.Stream) :
    IO Cuda.Qwen36.DeltaNet.StageForwardF32 := do
  let normalizedInput ← zeros hiddenElements stream
  let inputInverseRms ← zeros tokens stream
  let queryKeyValuePreConv ← zeros (tokens * convolutionWidth) stream
  let z ← zeros valueElements stream
  let b ← zeros (tokens * valueHeads) stream
  let a ← zeros (tokens * valueHeads) stream
  let queryKeyValuePostConv ← zeros (tokens * convolutionWidth) stream
  let queryCompact ← zeros compactElements stream
  let keyCompact ← zeros compactElements stream
  let value ← zeros valueElements stream
  let queryRepeated ← zeros valueElements stream
  let keyRepeated ← zeros valueElements stream
  let queryNormed ← zeros valueElements stream
  let keyNormed ← zeros valueElements stream
  let decayLog ← zeros (tokens * valueHeads) stream
  let beta ← zeros (tokens * valueHeads) stream
  -- Large enough for both the batch-1 reference gates and batch-4 x sequence-2 LoRA gate.
  let stateInput ← zeros (4 * stateElements) stream
  let stateOutput ← zeros (4 * stateElements) stream
  let stateHistory ← zeros (4 * (2 + 1) * stateElements) stream
  let deltaOutput ← zeros valueElements stream
  let gatedInverseRms ← zeros (tokens * valueHeads) stream
  let gatedOutput ← zeros valueElements stream
  let output ← zeros hiddenElements stream
  return {
    input, normalizedInput, inputInverseRms, queryKeyValuePreConv, z, b, a,
    queryKeyValuePostConv, queryCompact, keyCompact, value, queryRepeated, keyRepeated,
    queryNormed, keyNormed, decayLog, beta, stateInput, stateOutput, stateHistory,
    deltaOutput, gatedInverseRms, gatedOutput, output
  }

private def allocAttentionForward (input : Cuda.Buffer Float32) (stream : Cuda.Stream) :
    IO Cuda.Qwen36.Attention.StageForwardF32 := do
  let normalizedInput ← zeros hiddenElements stream
  let inputInverseRms ← zeros tokens stream
  let queryGateProjection ← zeros (tokens * 2 * queryHeads * attentionWidth) stream
  let queryPreNorm ← zeros queryElements stream
  let gate ← zeros queryElements stream
  let keyPreNorm ← zeros keyValueElements stream
  let value ← zeros keyValueElements stream
  let queryNormed ← zeros queryElements stream
  let queryInverseRms ← zeros (tokens * queryHeads) stream
  let keyNormed ← zeros keyValueElements stream
  let keyInverseRms ← zeros (tokens * keyValueHeads) stream
  let queryRope ← zeros queryElements stream
  let keyRope ← zeros keyValueElements stream
  let probabilities ← zeros probabilityElements stream
  let preGate ← zeros queryElements stream
  let postGate ← zeros queryElements stream
  let output ← zeros hiddenElements stream
  return {
    input, normalizedInput, inputInverseRms, queryGateProjection, queryPreNorm, gate,
    keyPreNorm, value, queryNormed, queryInverseRms, keyNormed, keyInverseRms,
    queryRope, keyRope, probabilities, preGate, postGate, output
  }

private def allocLayerForward (mixer : MixerForwardF32) (stream : Cuda.Stream) :
    IO LayerForwardF32 := do
  let mixerResidual ← zeros hiddenElements stream
  let mlp ← allocMLPForward stream
  let output ← zeros hiddenElements stream
  return { mixer, mixerResidual, mlp, output }

private def allocMLPBackward (stream : Cuda.Stream) : IO MLPBackwardF32 := do
  let activationGradient ← zeros intermediateElements stream
  let gateGradient ← zeros intermediateElements stream
  let upGradient ← zeros intermediateElements stream
  let normGradientGate ← zeros hiddenElements stream
  let normGradientUp ← zeros hiddenElements stream
  let normGradient ← zeros hiddenElements stream
  let inputGradient ← zeros hiddenElements stream
  let normWeightGradient ← zeros hidden stream
  let gateWeightGradient ← zeros (intermediate * hidden) stream
  let upWeightGradient ← zeros (intermediate * hidden) stream
  let downWeightGradient ← zeros (hidden * intermediate) stream
  return {
    activationGradient, gateGradient, upGradient, normGradientGate, normGradientUp,
    normGradient, inputGradient, normWeightGradient, gateWeightGradient,
    upWeightGradient, downWeightGradient
  }

private def allocDeltaBackward (stream : Cuda.Stream) :
    IO Cuda.Qwen36.DeltaNet.StageBackwardF32 := do
  let outputGradient ← zeros hiddenElements stream
  let gatedOutputGradient ← zeros valueElements stream
  let deltaOutputGradient ← zeros valueElements stream
  let zGradient ← zeros valueElements stream
  let gatedNormWeightGradient ← zeros valueWidth stream
  let queryNormGradient ← zeros valueElements stream
  let keyNormGradient ← zeros valueElements stream
  let valueGradient ← zeros valueElements stream
  let decayLogGradient ← zeros (tokens * valueHeads) stream
  let betaGradient ← zeros (tokens * valueHeads) stream
  let stateGradient ← zeros (4 * stateElements) stream
  let queryRepeatedGradient ← zeros valueElements stream
  let keyRepeatedGradient ← zeros valueElements stream
  let queryCompactGradient ← zeros compactElements stream
  let keyCompactGradient ← zeros compactElements stream
  let queryKeyValuePostConvGradient ← zeros (tokens * convolutionWidth) stream
  let queryKeyValuePreConvGradient ← zeros (tokens * convolutionWidth) stream
  let convolutionWeightGradient ← zeros (convolutionWidth * 4) stream
  let aGradient ← zeros (tokens * valueHeads) stream
  let bGradient ← zeros (tokens * valueHeads) stream
  let aLogGradient ← zeros valueHeads stream
  let dtBiasGradient ← zeros valueHeads stream
  let queryKeyValueInputGradient ← zeros hiddenElements stream
  let zInputGradient ← zeros hiddenElements stream
  let bInputGradient ← zeros hiddenElements stream
  let aInputGradient ← zeros hiddenElements stream
  let queryKeyValueZInputGradient ← zeros hiddenElements stream
  let queryKeyValueZBInputGradient ← zeros hiddenElements stream
  let normalizedInputGradient ← zeros hiddenElements stream
  let inputGradient ← zeros hiddenElements stream
  let inputNormWeightGradient ← zeros hidden stream
  let queryKeyValueWeightGradient ← zeros (convolutionWidth * hidden) stream
  let zWeightGradient ← zeros (hidden * hidden) stream
  let bWeightGradient ← zeros (valueHeads * hidden) stream
  let aWeightGradient ← zeros (valueHeads * hidden) stream
  let outputWeightGradient ← zeros (hidden * hidden) stream
  return {
    outputGradient, gatedOutputGradient, deltaOutputGradient, zGradient,
    gatedNormWeightGradient, queryNormGradient, keyNormGradient, valueGradient,
    decayLogGradient, betaGradient, stateGradient, queryRepeatedGradient,
    keyRepeatedGradient, queryCompactGradient, keyCompactGradient,
    queryKeyValuePostConvGradient, queryKeyValuePreConvGradient,
    convolutionWeightGradient, aGradient, bGradient, aLogGradient, dtBiasGradient,
    queryKeyValueInputGradient, zInputGradient, bInputGradient, aInputGradient,
    queryKeyValueZInputGradient, queryKeyValueZBInputGradient, normalizedInputGradient,
    inputGradient, inputNormWeightGradient, queryKeyValueWeightGradient,
    zWeightGradient, bWeightGradient, aWeightGradient, outputWeightGradient
  }

private def allocAttentionBackward (stream : Cuda.Stream) :
    IO Cuda.Qwen36.Attention.StageBackwardF32 := do
  let outputGradient ← zeros hiddenElements stream
  let postGateGradient ← zeros queryElements stream
  let preGateGradient ← zeros queryElements stream
  let gateGradient ← zeros queryElements stream
  let probabilityGradient ← zeros probabilityElements stream
  let scoreGradient ← zeros probabilityElements stream
  let queryRopeGradient ← zeros queryElements stream
  let keyRopeGradient ← zeros keyValueElements stream
  let valueGradient ← zeros keyValueElements stream
  let queryNormGradient ← zeros queryElements stream
  let keyNormGradient ← zeros keyValueElements stream
  let queryPreNormGradient ← zeros queryElements stream
  let keyPreNormGradient ← zeros keyValueElements stream
  let queryNormWeightGradient ← zeros attentionWidth stream
  let keyNormWeightGradient ← zeros attentionWidth stream
  let queryGateProjectionGradient ← zeros (tokens * 2 * queryHeads * attentionWidth) stream
  let queryInputGradient ← zeros hiddenElements stream
  let keyInputGradient ← zeros hiddenElements stream
  let valueInputGradient ← zeros hiddenElements stream
  let queryKeyInputGradient ← zeros hiddenElements stream
  let normalizedInputGradient ← zeros hiddenElements stream
  let inputGradient ← zeros hiddenElements stream
  let inputNormWeightGradient ← zeros hidden stream
  let queryWeightGradient ← zeros (2 * queryHeads * attentionWidth * hidden) stream
  let keyWeightGradient ← zeros (keyValueHeads * attentionWidth * hidden) stream
  let valueWeightGradient ← zeros (keyValueHeads * attentionWidth * hidden) stream
  let outputWeightGradient ← zeros (hidden * hidden) stream
  return {
    outputGradient, postGateGradient, preGateGradient, gateGradient,
    probabilityGradient, scoreGradient, queryRopeGradient, keyRopeGradient,
    valueGradient, queryNormGradient, keyNormGradient, queryPreNormGradient,
    keyPreNormGradient, queryNormWeightGradient, keyNormWeightGradient,
    queryGateProjectionGradient, queryInputGradient, keyInputGradient,
    valueInputGradient, queryKeyInputGradient, normalizedInputGradient,
    inputGradient, inputNormWeightGradient, queryWeightGradient, keyWeightGradient,
    valueWeightGradient, outputWeightGradient
  }

private def allocLayerBackward (mixer : MixerBackwardF32) (stream : Cuda.Stream) :
    IO LayerBackwardF32 := do
  let mlp ← allocMLPBackward stream
  let mixerResidualGradient ← zeros hiddenElements stream
  let inputGradient ← zeros hiddenElements stream
  return { mlp, mixerResidualGradient, mixer, inputGradient }

private def checkLayerForward (dir : String) (layer : Nat) (buffers : LayerForwardF32)
    (mixerOutput : Cuda.Buffer Float32) (stream : Cuda.Stream) : IO Unit := do
  let base := fixturePrefix layer
  checkBuffer dir s!"layer {layer} mixer output" (base ++ "mixer_out.8x256")
    mixerOutput hiddenElements stream
  checkBuffer dir s!"layer {layer} mixer residual" (base ++ "mixer_residual.8x256")
    buffers.mixerResidual hiddenElements stream
  checkBuffer dir s!"layer {layer} MLP norm" (base ++ "norm2_out.8x256")
    buffers.mlp.normalized hiddenElements stream
  checkBuffer dir s!"layer {layer} MLP gate" (base ++ "mlp_gate.8x512")
    buffers.mlp.gate intermediateElements stream
  checkBuffer dir s!"layer {layer} MLP up" (base ++ "mlp_up.8x512")
    buffers.mlp.up intermediateElements stream
  checkBuffer dir s!"layer {layer} MLP activation" (base ++ "mlp_act.8x512")
    buffers.mlp.activation intermediateElements stream
  checkBuffer dir s!"layer {layer} MLP output" (base ++ "mlp_out.8x256")
    buffers.mlp.output hiddenElements stream
  checkBuffer dir s!"layer {layer} hidden output" (base ++ "hidden_out.8x256")
    buffers.output hiddenElements stream

private def checkMLPBackward (dir : String) (layer : Nat) (buffers : MLPBackwardF32)
    (stream : Cuda.Stream) : IO Unit := do
  let debug := s!"grad.layer{layer}.mlp_"
  let weights := gradPrefix layer
  checkBuffer dir s!"layer {layer} MLP dactivation" (debug ++ "act.8x512")
    buffers.activationGradient intermediateElements stream
  checkBuffer dir s!"layer {layer} MLP dgate" (debug ++ "gate.8x512")
    buffers.gateGradient intermediateElements stream
  checkBuffer dir s!"layer {layer} MLP dup" (debug ++ "up.8x512")
    buffers.upGradient intermediateElements stream
  checkBuffer dir s!"layer {layer} MLP dnorm" (debug ++ "norm2.8x256")
    buffers.normGradient hiddenElements stream
  checkBuffer dir s!"layer {layer} MLP dinput" (debug ++ "input.8x256")
    buffers.inputGradient hiddenElements stream
  checkBuffer dir s!"layer {layer} MLP dnorm weight"
    (weights ++ "post_attention_layernorm.weight.256") buffers.normWeightGradient hidden stream
  checkBuffer dir s!"layer {layer} MLP dgate weight"
    (weights ++ "mlp.gate_proj.weight.512x256") buffers.gateWeightGradient
    (intermediate * hidden) stream
  checkBuffer dir s!"layer {layer} MLP dup weight"
    (weights ++ "mlp.up_proj.weight.512x256") buffers.upWeightGradient
    (intermediate * hidden) stream
  checkBuffer dir s!"layer {layer} MLP ddown weight"
    (weights ++ "mlp.down_proj.weight.256x512") buffers.downWeightGradient
    (hidden * intermediate) stream

private def checkDeltaWeights (dir : String) (layer : Nat)
    (buffers : Cuda.Qwen36.DeltaNet.StageBackwardF32) (stream : Cuda.Stream) : IO Unit := do
  let base := gradPrefix layer
  checkBuffer dir s!"layer {layer} DeltaNet mixer dinput"
    s!"grad.layer{layer}.deltanet_input.8x256" buffers.inputGradient hiddenElements stream
  checkBuffer dir s!"layer {layer} DeltaNet dinput norm"
    (base ++ "input_layernorm.weight.256") buffers.inputNormWeightGradient hidden stream
  checkBuffer dir s!"layer {layer} DeltaNet dQKV"
    (base ++ "linear_attn.in_proj_qkv.weight.512x256")
    buffers.queryKeyValueWeightGradient (convolutionWidth * hidden) stream
  checkBuffer dir s!"layer {layer} DeltaNet dz"
    (base ++ "linear_attn.in_proj_z.weight.256x256") buffers.zWeightGradient
    (hidden * hidden) stream
  checkBuffer dir s!"layer {layer} DeltaNet db"
    (base ++ "linear_attn.in_proj_b.weight.4x256") buffers.bWeightGradient
    (valueHeads * hidden) stream
  checkBuffer dir s!"layer {layer} DeltaNet da"
    (base ++ "linear_attn.in_proj_a.weight.4x256") buffers.aWeightGradient
    (valueHeads * hidden) stream
  checkBuffer dir s!"layer {layer} DeltaNet dconv"
    (base ++ "linear_attn.conv1d.weight.512x4") buffers.convolutionWeightGradient
    (convolutionWidth * 4) stream
  checkBuffer dir s!"layer {layer} DeltaNet dA_log"
    (base ++ "linear_attn.A_log.4") buffers.aLogGradient valueHeads stream
  checkBuffer dir s!"layer {layer} DeltaNet ddt_bias"
    (base ++ "linear_attn.dt_bias.4") buffers.dtBiasGradient valueHeads stream
  checkBuffer dir s!"layer {layer} DeltaNet dgated norm"
    (base ++ "linear_attn.norm.weight.64") buffers.gatedNormWeightGradient valueWidth stream
  checkBuffer dir s!"layer {layer} DeltaNet dout"
    (base ++ "linear_attn.out_proj.weight.256x256") buffers.outputWeightGradient
    (hidden * hidden) stream

private def checkAttentionWeights (dir : String) (layer : Nat)
    (buffers : Cuda.Qwen36.Attention.StageBackwardF32) (stream : Cuda.Stream) : IO Unit := do
  let base := gradPrefix layer
  checkBuffer dir "attention mixer dinput" s!"grad.layer{layer}.attention_input.8x256"
    buffers.inputGradient hiddenElements stream
  checkBuffer dir "attention dinput norm" (base ++ "input_layernorm.weight.256")
    buffers.inputNormWeightGradient hidden stream
  checkBuffer dir "attention dq norm" (base ++ "self_attn.q_norm.weight.64")
    buffers.queryNormWeightGradient attentionWidth stream
  checkBuffer dir "attention dk norm" (base ++ "self_attn.k_norm.weight.64")
    buffers.keyNormWeightGradient attentionWidth stream
  checkBuffer dir "attention dq weight" (base ++ "self_attn.q_proj.weight.512x256")
    buffers.queryWeightGradient (2 * queryHeads * attentionWidth * hidden) stream
  checkBuffer dir "attention dk weight" (base ++ "self_attn.k_proj.weight.128x256")
    buffers.keyWeightGradient (keyValueHeads * attentionWidth * hidden) stream
  checkBuffer dir "attention dv weight" (base ++ "self_attn.v_proj.weight.128x256")
    buffers.valueWeightGradient (keyValueHeads * attentionWidth * hidden) stream
  checkBuffer dir "attention do weight" (base ++ "self_attn.o_proj.weight.256x256")
    buffers.outputWeightGradient (hidden * hidden) stream

private inductive ExecutionMode where
  | sequential
  | graph
  | trainingGraph

private def execute (mode : ExecutionMode) (stream : Cuda.Stream) (weights : WeightsF32)
    (forward : ForwardF32) (backward : BackwardF32) (tokenIds targets : Cuda.Buffer UInt32) :
    IO Unit :=
  match mode with
  | .sequential => do
      forwardLossF32 stream weights forward tokenIds targets shape
      backwardF32 stream weights forward backward tokenIds shape
  | .graph => do
      let builder ← Cuda.GraphBuilder.create
      submitForwardLossF32 (.graph builder) weights forward tokenIds targets shape
      submitBackwardF32 (.graph builder) weights forward backward tokenIds shape
      let graph ← builder.instantiate
      (← graph.launchOn stream).waitChecked "Qwen3.6 complete model graph"
  | .trainingGraph => do
      let lmHead ← requireBaseWeight weights.lmHead
      let initialLmHead ← lmHead.copyTo stream
      let schedule ← allocateAdamWSchedule stream
      let optimizer ← allocateFullModelOptimizer weights backward stream
      LeanTest.assertTrue (optimizer.states.size == 56)
        s!"complete four-layer optimizer registry has {optimizer.states.size} entries"
      let descriptor : Descriptor := {
        steps := 1
        batchSize := 1
        sequenceLength := tokens
        model := shape
        profile := .fullModel
      }
      let step : FullModelStepF32 := {
        weights, forward, backward, tokenIds, targets,
        updateState := .fullModel optimizer
      }
      let groups ← fullModelParameterGroupsF32 weights backward shape optimizer
      let mut initialGroups : Array ParameterGroupSnapshot := #[]
      for group in groups do
        initialGroups := initialGroups.push (← snapshotParameterGroup group stream)
      let builder ← Cuda.GraphBuilder.create
      submitFullModelStepF32 (.graph builder) adamWHyperparameters schedule descriptor step
      let graph ← builder.instantiate
      (← graph.launchOn stream).waitChecked "Qwen3.6 complete model training graph"
      let mut graphGroups : Array ParameterGroupSnapshot := #[]
      for index in [:groups.size] do
        match groups[index]?, initialGroups[index]? with
        | some group, some initial =>
            let result ← snapshotParameterGroup group stream
            LeanTest.assertTrue (result.parameter != initial.parameter)
              s!"full-model registry entry {index} did not update"
            graphGroups := graphGroups.push result
        | _, _ => LeanTest.fail s!"full-model graph snapshot mismatch at {index}"
      let graphStep := ← schedule.step.copyTo stream
      let graphBeta1Power := ← schedule.beta1Power.copyTo stream
      let graphBeta2Power := ← schedule.beta2Power.copyTo stream
      let graphInverse1 := ← schedule.inverseBiasCorrection1.copyTo stream
      let graphInverse2 := ← schedule.inverseBiasCorrection2.copyTo stream

      for index in [:groups.size] do
        match groups[index]?, initialGroups[index]? with
        | some group, some initial => restoreParameterGroup group initial stream
        | _, _ => LeanTest.fail s!"full-model restore mismatch at {index}"
      let sequentialSchedule ← allocateAdamWSchedule stream
      submitFullModelStepF32 (.sequential stream) adamWHyperparameters sequentialSchedule descriptor
        step
      for index in [:groups.size] do
        match groups[index]?, graphGroups[index]? with
        | some group, some expected =>
            checkParameterGroupSnapshotExact (← snapshotParameterGroup group stream) expected
        | _, _ => LeanTest.fail s!"full-model parity mismatch at {index}"
      LeanTest.assertEqual (← sequentialSchedule.step.copyTo stream) graphStep
      LeanTest.assertEqual (← sequentialSchedule.beta1Power.copyTo stream) graphBeta1Power
      LeanTest.assertEqual (← sequentialSchedule.beta2Power.copyTo stream) graphBeta2Power
      LeanTest.assertEqual (← sequentialSchedule.inverseBiasCorrection1.copyTo stream) graphInverse1
      LeanTest.assertEqual (← sequentialSchedule.inverseBiasCorrection2.copyTo stream) graphInverse2
      LeanTest.assertTrue ((← lmHead.copyTo stream) != initialLmHead)
        "full-model training did not update the LM head"

private def run (referenceDir : String) (mode : ExecutionMode) : IO Unit := do
  let stream ← Cuda.Stream.default
  let tokenIds ← uploadIds referenceDir "tokens.8" tokens stream
  let targets ← uploadIds referenceDir "targets.8" tokens stream
  let embeddingWeight ← upload referenceDir "model.embed_tokens.weight.512x256"
    (vocabulary * hidden) stream
  let finalNorm ← upload referenceDir "model.norm.weight.256" hidden stream
  let lmHead ← upload referenceDir "lm_head.weight.512x256" (vocabulary * hidden) stream

  let deltaWeights0 ← loadDeltaWeights referenceDir 0 stream
  let deltaWeights1 ← loadDeltaWeights referenceDir 1 stream
  let deltaWeights2 ← loadDeltaWeights referenceDir 2 stream
  let attentionWeights3 ← loadAttentionWeights referenceDir 3 stream
  let mlpWeights0 ← loadMLPWeights referenceDir 0 stream
  let mlpWeights1 ← loadMLPWeights referenceDir 1 stream
  let mlpWeights2 ← loadMLPWeights referenceDir 2 stream
  let mlpWeights3 ← loadMLPWeights referenceDir 3 stream
  let weights : WeightsF32 := {
    embedding := embeddingWeight,
    layers := #[
      { mixer := .deltaNet deltaWeights0, mlp := mlpWeights0 },
      { mixer := .deltaNet deltaWeights1, mlp := mlpWeights1 },
      { mixer := .deltaNet deltaWeights2, mlp := mlpWeights2 },
      { mixer := .attention attentionWeights3, mlp := mlpWeights3 }
    ],
    finalNorm, lmHead
  }

  let embedding ← zeros hiddenElements stream
  let deltaForward0 ← allocDeltaForward embedding stream
  let layerForward0 ← allocLayerForward (.deltaNet deltaForward0) stream
  let deltaForward1 ← allocDeltaForward embedding stream
  let layerForward1 ← allocLayerForward (.deltaNet deltaForward1) stream
  let deltaForward2 ← allocDeltaForward embedding stream
  let layerForward2 ← allocLayerForward (.deltaNet deltaForward2) stream
  let attentionForward3 ← allocAttentionForward embedding stream
  let layerForward3 ← allocLayerForward (.attention attentionForward3) stream
  let finalHidden ← zeros hiddenElements stream
  let finalInverseRms ← zeros tokens stream
  let logits ← zeros (tokens * vocabulary) stream
  let rowLoss ← zeros tokens stream
  let loss ← zeros 1 stream
  let logitGradient ← zeros (tokens * vocabulary) stream
  let forward : ForwardF32 := {
    embedding,
    layers := #[layerForward0, layerForward1, layerForward2, layerForward3],
    finalHidden, finalInverseRms, logits, rowLoss, loss, logitGradient
  }

  let deltaBackward0 ← allocDeltaBackward stream
  let layerBackward0 ← allocLayerBackward (.deltaNet deltaBackward0) stream
  let deltaBackward1 ← allocDeltaBackward stream
  let layerBackward1 ← allocLayerBackward (.deltaNet deltaBackward1) stream
  let deltaBackward2 ← allocDeltaBackward stream
  let layerBackward2 ← allocLayerBackward (.deltaNet deltaBackward2) stream
  let attentionBackward3 ← allocAttentionBackward stream
  let layerBackward3 ← allocLayerBackward (.attention attentionBackward3) stream
  let finalHiddenGradient ← zeros hiddenElements stream
  let lmHeadWeightGradient ← zeros (vocabulary * hidden) stream
  let decoderOutputGradient ← zeros hiddenElements stream
  let finalNormWeightGradient ← zeros hidden stream
  let embeddingWeightGradient ← zeros (vocabulary * hidden) stream
  let backward : BackwardF32 := {
    finalHiddenGradient, lmHeadWeightGradient, decoderOutputGradient,
    finalNormWeightGradient,
    layers := #[layerBackward0, layerBackward1, layerBackward2, layerBackward3],
    embeddingWeightGradient
  }
  execute mode stream weights forward backward tokenIds targets

  checkBuffer referenceDir "model embedding" "embed_out.8x256" embedding hiddenElements stream
  checkLayerForward referenceDir 0 layerForward0 deltaForward0.output stream
  checkLayerForward referenceDir 1 layerForward1 deltaForward1.output stream
  checkLayerForward referenceDir 2 layerForward2 deltaForward2.output stream
  checkLayerForward referenceDir 3 layerForward3 attentionForward3.output stream
  checkBuffer referenceDir "model final hidden" "final_hidden.8x256" finalHidden
    hiddenElements stream
  checkBuffer referenceDir "model logits" "logits.8x512" logits (tokens * vocabulary) stream
  checkBuffer referenceDir "model loss" "loss.1" loss 1 stream
  checkBuffer referenceDir "model dlogits" "grad.logits.8x512" logitGradient
    (tokens * vocabulary) stream

  checkBuffer referenceDir "model dfinal hidden" "grad.final_hidden.8x256"
    finalHiddenGradient hiddenElements stream
  checkBuffer referenceDir "model dlm head" "grad.lm_head.weight.512x256"
    lmHeadWeightGradient (vocabulary * hidden) stream
  checkBuffer referenceDir "model dfinal norm" "grad.model.norm.weight.256"
    finalNormWeightGradient hidden stream
  checkBuffer referenceDir "model dembedding activation before scatter" "grad.embed_out.8x256"
    layerBackward0.inputGradient hiddenElements stream
  checkBuffer referenceDir "model dembedding" "grad.model.embed_tokens.weight.512x256"
    embeddingWeightGradient (vocabulary * hidden) stream

  checkMLPBackward referenceDir 0 layerBackward0.mlp stream
  checkMLPBackward referenceDir 1 layerBackward1.mlp stream
  checkMLPBackward referenceDir 2 layerBackward2.mlp stream
  checkMLPBackward referenceDir 3 layerBackward3.mlp stream
  checkBuffer referenceDir "layer 0 residual-complete input gradient" "grad.embed_out.8x256"
    layerBackward0.inputGradient hiddenElements stream
  checkBuffer referenceDir "layer 1 residual-complete input gradient"
    "grad.layer0.mlp_output.8x256" layerBackward1.inputGradient hiddenElements stream
  checkBuffer referenceDir "layer 2 residual-complete input gradient"
    "grad.layer1.mlp_output.8x256" layerBackward2.inputGradient hiddenElements stream
  checkBuffer referenceDir "layer 3 residual-complete input gradient"
    "grad.layer2.mlp_output.8x256" layerBackward3.inputGradient hiddenElements stream
  checkBuffer referenceDir "layer 0 mixer output gradient" "grad.layer0.deltanet_output.8x256"
    layerBackward0.mixerResidualGradient hiddenElements stream
  checkBuffer referenceDir "layer 1 mixer output gradient" "grad.layer1.deltanet_output.8x256"
    layerBackward1.mixerResidualGradient hiddenElements stream
  checkBuffer referenceDir "layer 2 mixer output gradient" "grad.layer2.deltanet_output.8x256"
    layerBackward2.mixerResidualGradient hiddenElements stream
  checkBuffer referenceDir "layer 3 mixer output gradient" "grad.layer3.attention_output.8x256"
    layerBackward3.mixerResidualGradient hiddenElements stream
  checkDeltaWeights referenceDir 0 deltaBackward0 stream
  checkDeltaWeights referenceDir 1 deltaBackward1 stream
  checkDeltaWeights referenceDir 2 deltaBackward2 stream
  checkAttentionWeights referenceDir 3 attentionBackward3 stream

  match mode with
  | .trainingGraph =>
      let lossBefore := readFloat32 (← loss.copyTo stream) 0
      forwardLossF32 stream weights forward tokenIds targets shape
      let lossAfter := readFloat32 (← loss.copyTo stream) 0
      LeanTest.assertTrue (lossAfter < lossBefore)
        s!"full-model update did not lower loss: {lossBefore} -> {lossAfter}"
  | _ => pure ()


private def referenceDirectory : IO String := do
  match ← IO.getEnv "QWEN36_REFERENCE_DIR" with
    | some directory => pure directory
    | none => throw <| IO.userError "QWEN36_REFERENCE_DIR is required"

@[test]
def fullModelSequentialMatchesReference : IO Unit := do
  run (← referenceDirectory) .sequential

@[test]
def fullModelGraphMatchesReference : IO Unit := do
  run (← referenceDirectory) .graph

@[test]
def fullModelTrainingGraphUpdatesAndLowersLoss : IO Unit := do
  run (← referenceDirectory) .trainingGraph

@[test]
def fullModelMixedLoRAGraphUpdatesOnlyEnabledAdapters : IO Unit := do
  let referenceDir ← referenceDirectory
  let stream ← Cuda.Stream.default
  let tokenIds ← uploadIds referenceDir "tokens.8" tokens stream
  let targets ← uploadIds referenceDir "targets.8" tokens stream
  let adapterIds ← uploadUInt32Values #[0, 1, 0, 1] stream
  let updateMask ← uploadUInt32Values #[1, 0] stream

  let embeddingWeight ← upload referenceDir "model.embed_tokens.weight.512x256"
    (vocabulary * hidden) stream
  let embeddingFrozen ← freezeBF16 embeddingWeight (vocabulary * hidden) stream
  let finalNorm ← upload referenceDir "model.norm.weight.256" hidden stream
  let lmHeadBase ← upload referenceDir "lm_head.weight.512x256" (vocabulary * hidden) stream

  let deltaBase0 ← loadDeltaWeights referenceDir 0 stream
  let deltaBase1 ← loadDeltaWeights referenceDir 1 stream
  let deltaBase2 ← loadDeltaWeights referenceDir 2 stream
  let attentionBase ← loadAttentionWeights referenceDir 3 stream
  let mlpBase0 ← loadMLPWeights referenceDir 0 stream
  let mlpBase1 ← loadMLPWeights referenceDir 1 stream
  let mlpBase2 ← loadMLPWeights referenceDir 2 stream
  let mlpBase3 ← loadMLPWeights referenceDir 3 stream
  let (deltaWeights0, deltaAdapters0, deltaBases0) ←
    adaptDeltaWeights deltaBase0 adapterIds updateMask 1 stream
  let (deltaWeights1, deltaAdapters1, deltaBases1) ←
    adaptDeltaWeights deltaBase1 adapterIds updateMask 20 stream
  let (deltaWeights2, deltaAdapters2, deltaBases2) ←
    adaptDeltaWeights deltaBase2 adapterIds updateMask 40 stream
  let (attentionWeights, attentionAdapters, attentionBases) ←
    adaptAttentionWeights attentionBase adapterIds updateMask 60 stream
  let attentionOutputAdapter ← match attentionAdapters[3]? with
    | some adapter => pure adapter
    | none => throw <| IO.userError "tiny attention output adapter is missing"
  let (mlpWeights0, mlpAdapters0, mlpBases0) ←
    adaptMLPWeights mlpBase0 adapterIds updateMask 80 stream
  let (mlpWeights1, mlpAdapters1, mlpBases1) ←
    adaptMLPWeights mlpBase1 adapterIds updateMask 90 stream
  let (mlpWeights2, mlpAdapters2, mlpBases2) ←
    adaptMLPWeights mlpBase2 adapterIds updateMask 100 stream
  let (mlpWeights3, mlpAdapters3, mlpBases3) ←
    adaptMLPWeights mlpBase3 adapterIds updateMask 110 stream
  let (lmHead, lmHeadAdapter) ← adaptProjection lmHeadBase adapterIds updateMask hidden
    vocabulary 120 stream

  let adapters :=
    (deltaAdapters0 ++ mlpAdapters0 ++ deltaAdapters1 ++ mlpAdapters1 ++ deltaAdapters2 ++
      mlpAdapters2 ++ attentionAdapters ++ mlpAdapters3).push lmHeadAdapter
  let projectionBases :=
    (deltaBases0 ++ mlpBases0 ++ deltaBases1 ++ mlpBases1 ++ deltaBases2 ++ mlpBases2 ++
      attentionBases ++ mlpBases3).push lmHeadBase
  let frozenBuffers := projectionBases ++ #[
    embeddingWeight, finalNorm,
    deltaBase0.inputNorm, deltaBase0.convolution, deltaBase0.aLog, deltaBase0.dtBias,
    deltaBase0.gatedNorm,
    deltaBase1.inputNorm, deltaBase1.convolution, deltaBase1.aLog, deltaBase1.dtBias,
    deltaBase1.gatedNorm,
    deltaBase2.inputNorm, deltaBase2.convolution, deltaBase2.aLog, deltaBase2.dtBias,
    deltaBase2.gatedNorm,
    attentionBase.inputNorm, attentionBase.queryNorm, attentionBase.keyNorm,
    attentionBase.inverseFrequency,
    mlpBase0.norm, mlpBase1.norm, mlpBase2.norm, mlpBase3.norm
  ]
  LeanTest.assertTrue (adapters.size == 32)
    s!"expected adapters on all 32 eligible projections, found {adapters.size}"

  let weights : WeightsF32 := {
    embedding := .frozenBF16 embeddingFrozen
    layers := #[
      { mixer := .deltaNet deltaWeights0, mlp := mlpWeights0 },
      { mixer := .deltaNet deltaWeights1, mlp := mlpWeights1 },
      { mixer := .deltaNet deltaWeights2, mlp := mlpWeights2 },
      { mixer := .attention attentionWeights, mlp := mlpWeights3 }
    ]
    finalNorm
    lmHead
  }

  let embedding ← zeros hiddenElements stream
  let deltaForward0 ← allocDeltaForward embedding stream
  let layerForward0 ← allocLayerForward (.deltaNet deltaForward0) stream
  let deltaForward1 ← allocDeltaForward embedding stream
  let layerForward1 ← allocLayerForward (.deltaNet deltaForward1) stream
  let deltaForward2 ← allocDeltaForward embedding stream
  let layerForward2 ← allocLayerForward (.deltaNet deltaForward2) stream
  let attentionForward3 ← allocAttentionForward embedding stream
  let layerForward3 ← allocLayerForward (.attention attentionForward3) stream
  let finalHidden ← zeros hiddenElements stream
  let finalInverseRms ← zeros tokens stream
  let logits ← zeros (tokens * vocabulary) stream
  let rowLoss ← zeros tokens stream
  let loss ← zeros 1 stream
  let logitGradient ← zeros (tokens * vocabulary) stream
  let forward : ForwardF32 := {
    embedding
    layers := #[layerForward0, layerForward1, layerForward2, layerForward3]
    finalHidden, finalInverseRms, logits, rowLoss, loss, logitGradient
  }

  let deltaBackward0 ← allocDeltaBackward stream
  let layerBackward0 ← allocLayerBackward (.deltaNet deltaBackward0) stream
  let deltaBackward1 ← allocDeltaBackward stream
  let layerBackward1 ← allocLayerBackward (.deltaNet deltaBackward1) stream
  let deltaBackward2 ← allocDeltaBackward stream
  let layerBackward2 ← allocLayerBackward (.deltaNet deltaBackward2) stream
  let attentionBackward3 ← allocAttentionBackward stream
  let layerBackward3 ← allocLayerBackward (.attention attentionBackward3) stream
  let finalHiddenGradient ← zeros hiddenElements stream
  let lmHeadWeightGradient ← zeros (vocabulary * hidden) stream
  let decoderOutputGradient ← zeros hiddenElements stream
  let finalNormWeightGradient ← zeros hidden stream
  let embeddingWeightGradient ← zeros (vocabulary * hidden) stream
  let backward : BackwardF32 := {
    finalHiddenGradient, lmHeadWeightGradient, decoderOutputGradient,
    finalNormWeightGradient
    layers := #[layerBackward0, layerBackward1, layerBackward2, layerBackward3]
    embeddingWeightGradient
  }

  let layout : Cuda.Qwen36.SequenceLayout := {
    batchSize := 4
    sequenceLength := 2
  }
  submitBatchedForwardLossF32 (.sequential stream) weights forward tokenIds targets layout shape
  let lossBefore := readFloat32 (← loss.copyTo stream) 0
  let initialAttentionPostGate ← attentionForward3.postGate.copyTo stream
  let initialAttentionOutputRank ← attentionOutputAdapter.rankActivation.copyTo stream
  let initialLogitGradient ← forward.logitGradient.copyTo stream

  let mut frozenSnapshots : Array ByteArray := #[]
  for buffer in frozenBuffers do
    frozenSnapshots := frozenSnapshots.push (← buffer.copyTo stream)
  let mut adapterSnapshots : Array AdapterSnapshot := #[]
  for adapter in adapters do
    adapterSnapshots := adapterSnapshots.push (← snapshotAdapter adapter stream)

  let residentSteps : Nat := 1
  let schedule ← AdamWScheduleF32.allocate stream
  let descriptor : Descriptor := {
    steps := residentSteps
    batchSize := 4
    sequenceLength := 2
    model := shape
    profile := .lora {
      adapterCount := 2
      rank := 16
      alpha := 16
      adapterIds := #[0, 1, 0, 1]
    }
  }
  let step : FullModelStepF32 := {
    weights, forward, backward, tokenIds, targets
    updateState := .lora
  }
  let optimizer : AdamWHyperparametersF32 := {
    learningRate := 0.0001
    beta1 := 0.9
    beta2 := 0.99
    epsilon := 1e-8
  }
  let trainer ← ResidentTrainerF32.build optimizer schedule descriptor step
  trainer.run stream

  LeanTest.assertEqual (← schedule.step.copyTo stream)
    (pushUInt32 ByteArray.empty residentSteps.toUInt32)
  for index in [:frozenBuffers.size] do
    match frozenBuffers[index]?, frozenSnapshots[index]? with
    | some buffer, some before =>
        LeanTest.assertEqual (← buffer.copyTo stream) before
    | _, _ =>
        LeanTest.fail s!"frozen-buffer snapshot index mismatch at {index}"
  for index in [:adapters.size] do
    match adapters[index]?, adapterSnapshots[index]? with
    | some adapter, some before =>
        checkMaskedAdapterUpdate index adapter before stream
    | _, _ =>
        LeanTest.fail s!"adapter snapshot index mismatch at {index}"

  submitBatchedForwardLossF32 (.sequential stream) weights forward tokenIds targets layout shape
  let lossAfter := readFloat32 (← loss.copyTo stream) 0
  LeanTest.assertTrue (lossAfter < lossBefore)
    s!"mixed-LoRA full-model step did not lower loss: {lossBefore} -> {lossAfter}"

  let graphLoss := ← loss.copyTo stream
  let mut graphAdapterSnapshots : Array AdapterSnapshot := #[]
  for adapter in adapters do
    graphAdapterSnapshots := graphAdapterSnapshots.push (← snapshotAdapter adapter stream)
  let graphAttentionOutputBGradient ← attentionOutputAdapter.adapterBGradient.copyTo stream
  let graphAttentionOutputGradient ← layerBackward3.mixerResidualGradient.copyTo stream
  let graphDecoderOutputGradient ← decoderOutputGradient.copyTo stream
  let graphFinalHiddenGradient ← finalHiddenGradient.copyTo stream
  let graphAttentionMlpInputGradient ← layerBackward3.mlp.inputGradient.copyTo stream
  let graphStep := ← schedule.step.copyTo stream
  let graphBeta1Power := ← schedule.beta1Power.copyTo stream
  let graphBeta2Power := ← schedule.beta2Power.copyTo stream
  let graphInverse1 := ← schedule.inverseBiasCorrection1.copyTo stream
  let graphInverse2 := ← schedule.inverseBiasCorrection2.copyTo stream

  for index in [:adapters.size] do
    match adapters[index]?, adapterSnapshots[index]? with
    | some adapter, some initial => restoreAdapter adapter initial stream
    | _, _ => LeanTest.fail s!"adapter restore index mismatch at {index}"
  let sequentialSchedule ← AdamWScheduleF32.allocate stream
  for _ in [:residentSteps] do
    submitFullModelStepF32 (.sequential stream) optimizer sequentialSchedule descriptor step
  for index in [:adapters.size] do
    match adapters[index]?, graphAdapterSnapshots[index]? with
    | some adapter, some expected =>
      checkAdapterSnapshotExact "sequential" index (← snapshotAdapter adapter stream) expected
    | _, _ => LeanTest.fail s!"adapter parity index mismatch at {index}"
  LeanTest.assertEqual (← sequentialSchedule.step.copyTo stream) graphStep
  LeanTest.assertEqual (← sequentialSchedule.beta1Power.copyTo stream) graphBeta1Power
  LeanTest.assertEqual (← sequentialSchedule.beta2Power.copyTo stream) graphBeta2Power
  LeanTest.assertEqual (← sequentialSchedule.inverseBiasCorrection1.copyTo stream) graphInverse1
  LeanTest.assertEqual (← sequentialSchedule.inverseBiasCorrection2.copyTo stream) graphInverse2
  submitBatchedForwardLossF32 (.sequential stream) weights forward tokenIds targets layout shape
  LeanTest.assertEqual (← loss.copyTo stream) graphLoss

  for index in [:adapters.size] do
    match adapters[index]?, adapterSnapshots[index]? with
    | some adapter, some initial => restoreAdapter adapter initial stream
    | _, _ => LeanTest.fail s!"megakernel adapter restore index mismatch at {index}"
  let megakernelSchedule ← AdamWScheduleF32.allocate stream
  let program ← Cuda.Qwen36.FullTrainingMegakernel.Program.build stream descriptor step
  LeanTest.assertTrue (program.descriptor.projectionCount == 32)
    s!"tiny full-training tape registered {program.descriptor.projectionCount} projections"
  for _ in [:residentSteps] do
    (← Cuda.Qwen36.FullPretrainMegakernel.launch stream program optimizer
      megakernelSchedule).waitChecked "Qwen3.6 complete-model pretraining megakernel"
  checkBytesExact "megakernel attention post-gate" (← attentionForward3.postGate.copyTo stream)
    initialAttentionPostGate
  checkBytesExact "megakernel attention output rank activation"
    (← attentionOutputAdapter.rankActivation.copyTo stream) initialAttentionOutputRank
  checkBytesExact "megakernel logit gradient"
    (← forward.logitGradient.copyTo stream) initialLogitGradient
  checkBytesExact "megakernel LM-head input gradient"
    (← finalHiddenGradient.copyTo stream) graphFinalHiddenGradient
  checkBytesExact "megakernel decoder output gradient"
    (← decoderOutputGradient.copyTo stream) graphDecoderOutputGradient
  checkBytesExact "megakernel attention MLP input gradient"
    (← layerBackward3.mlp.inputGradient.copyTo stream) graphAttentionMlpInputGradient
  checkBytesExact "megakernel attention output gradient"
    (← layerBackward3.mixerResidualGradient.copyTo stream) graphAttentionOutputGradient
  checkBytesExact "megakernel attention output dB"
    (← attentionOutputAdapter.adapterBGradient.copyTo stream) graphAttentionOutputBGradient
  for index in [:adapters.size] do
    match adapters[index]?, graphAdapterSnapshots[index]? with
    | some adapter, some expected =>
        checkAdapterSnapshotExact "megakernel" index (← snapshotAdapter adapter stream) expected
    | _, _ => LeanTest.fail s!"megakernel parity index mismatch at {index}"
  LeanTest.assertEqual (← megakernelSchedule.step.copyTo stream) graphStep
  LeanTest.assertEqual (← megakernelSchedule.beta1Power.copyTo stream) graphBeta1Power
  LeanTest.assertEqual (← megakernelSchedule.beta2Power.copyTo stream) graphBeta2Power
  LeanTest.assertEqual (← megakernelSchedule.inverseBiasCorrection1.copyTo stream) graphInverse1
  LeanTest.assertEqual (← megakernelSchedule.inverseBiasCorrection2.copyTo stream) graphInverse2
  submitBatchedForwardLossF32 (.sequential stream) weights forward tokenIds targets layout shape
  LeanTest.assertEqual (← loss.copyTo stream) graphLoss

end Qwen36ModelGate
