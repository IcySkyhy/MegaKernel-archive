/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-!
# Qwen3.6 recurrent DeltaNet gates

Checks all three tiny linear-attention layers against the NumPy/HF reference fixtures. The
multi-token recurrent baseline must agree with the chunked oracle, and single-token decode must
preserve the supplied recurrent state exactly within the fp32 stage tolerance.
-/

namespace Qwen36DeltaNetGate

open Cuda.Qwen36.DeltaNet

private def tokens : Nat := 8
private def keyHeads : Nat := 2
private def valueHeads : Nat := 4
private def keyWidth : Nat := 64
private def valueWidth : Nat := 64
private def stateElements : Nat := valueHeads * keyWidth * valueWidth
private def prefillElements : Nat := tokens * valueHeads * valueWidth

private def tokens32 : UInt32 := 8
private def keyHeads32 : UInt32 := 2
private def valueHeads32 : UInt32 := 4
private def keyWidth32 : UInt32 := 64
private def valueWidth32 : UInt32 := 64
private def epsilon : Float32 := 1e-6
private def queryScale : Float32 := 0.125

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

private def zeros (count : Nat) (stream : Cuda.Stream) : IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 count.toUSize
  buffer.copyFrom (List.replicate (count * 4) 0).toByteArray stream
  return buffer

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
    LeanTest.fail
      (s!"{label}: max abs err {worst} at {worstAt} exceeds {tolerance} " ++
       s!"(actual {readFloat32 actual worstAt}, expected {readFloat32 expected worstAt})")

private def checkBuffer (dir label fixtureName : String) (buffer : Cuda.Buffer Float32)
    (count : Nat) (stream : Cuda.Stream) : IO Unit := do
  checkClose label (← buffer.copyTo stream) (← fixture dir fixtureName count) count 3e-5

private def layerName (layer : Nat) (suffix : String) : String :=
  s!"layer{layer}.{suffix}"

private def decodeName (layer : Nat) (suffix : String) : String :=
  s!"decode.layer{layer}.{suffix}"

private def gatePrefill (dir : String) (layer : Nat) (stream : Cuda.Stream) : IO Unit := do
  let query ← upload dir (layerName layer "q_post_conv.8x128") (tokens * keyHeads * keyWidth) stream
  let key ← upload dir (layerName layer "k_post_conv.8x128") (tokens * keyHeads * keyWidth) stream
  let value ← upload dir (layerName layer "v_heads.8x4x64") prefillElements stream
  let decayLog ← upload dir (layerName layer "g.8x4") (tokens * valueHeads) stream
  let beta ← upload dir (layerName layer "beta.8x4") (tokens * valueHeads) stream
  let stateInput ← zeros stateElements stream
  let output ← zeros prefillElements stream
  let stateOutput ← zeros stateElements stream
  let handle ← recurrentPrefillF32 stream query key value decayLog beta stateInput output stateOutput
    tokens32 keyHeads32 valueHeads32 keyWidth32 valueWidth32 epsilon queryScale
  handle.waitChecked s!"Qwen3.6 recurrent prefill layer {layer}"
  checkClose s!"layer{layer} recurrent-vs-chunked output" (← output.copyTo stream)
    (← fixture dir (layerName layer "delta_out.8x4x64") prefillElements)
    prefillElements 3e-5
  checkClose s!"layer{layer} recurrent-vs-chunked state" (← stateOutput.copyTo stream)
    (← fixture dir (layerName layer "S_final.4x64x64") stateElements)
    stateElements 3e-5

private def gateDecode (dir : String) (layer : Nat) (stream : Cuda.Stream) : IO Unit := do
  let query ← upload dir (decodeName layer "q_post_conv.128") (keyHeads * keyWidth) stream
  let key ← upload dir (decodeName layer "k_post_conv.128") (keyHeads * keyWidth) stream
  let value ← upload dir (decodeName layer "v_heads.4x64") (valueHeads * valueWidth) stream
  let decayLog ← upload dir (decodeName layer "g.4") valueHeads stream
  let beta ← upload dir (decodeName layer "beta.4") valueHeads stream
  let stateInput ← upload dir (decodeName layer "S_in.4x64x64") stateElements stream
  let output ← zeros (valueHeads * valueWidth) stream
  let stateOutput ← zeros stateElements stream
  let handle ← recurrentDecodeF32 stream query key value decayLog beta stateInput output stateOutput
    keyHeads32 valueHeads32 keyWidth32 valueWidth32 epsilon queryScale
  handle.waitChecked s!"Qwen3.6 recurrent decode layer {layer}"
  checkClose s!"layer{layer} decode output" (← output.copyTo stream)
    (← fixture dir (decodeName layer "delta_out.4x64") (valueHeads * valueWidth))
    (valueHeads * valueWidth) 3e-5
  checkClose s!"layer{layer} decode state" (← stateOutput.copyTo stream)
    (← fixture dir (decodeName layer "S_out.4x64x64") stateElements)
    stateElements 3e-5

private def gateReverseRecurrence (dir : String) (stream : Cuda.Stream) : IO Unit := do
  let query ← upload dir "layer0.q_l2.8x4x64" prefillElements stream
  let key ← upload dir "layer0.k_l2.8x4x64" prefillElements stream
  let value ← upload dir "layer0.v_heads.8x4x64" prefillElements stream
  let decayLog ← upload dir "layer0.g.8x4" (tokens * valueHeads) stream
  let beta ← upload dir "layer0.beta.8x4" (tokens * valueHeads) stream
  let stateInput ← zeros stateElements stream
  let output ← zeros prefillElements stream
  let stateOutput ← zeros stateElements stream
  let stateHistory ← zeros ((tokens + 1) * stateElements) stream
  let forward ← recurrentNormalizedPrefillF32 stream query key value decayLog beta stateInput output
    stateOutput stateHistory tokens32 keyHeads32 valueHeads32 keyWidth32 valueWidth32 queryScale
  forward.waitChecked "Qwen3.6 normalized recurrent prefill with history"
  checkClose "normalized recurrent output" (← output.copyTo stream)
    (← fixture dir "layer0.delta_out.8x4x64" prefillElements) prefillElements 3e-5
  checkClose "normalized recurrent final state" (← stateOutput.copyTo stream)
    (← fixture dir "layer0.S_final.4x64x64" stateElements) stateElements 3e-5

  let outputGradient ← upload dir "grad.layer0.deltanet_delta_out.8x4x64"
    prefillElements stream
  let queryGradient ← zeros prefillElements stream
  let keyGradient ← zeros prefillElements stream
  let valueGradient ← zeros prefillElements stream
  let decayLogGradient ← zeros (tokens * valueHeads) stream
  let betaGradient ← zeros (tokens * valueHeads) stream
  let stateGradient ← zeros stateElements stream
  let backward ← recurrentNormalizedBackwardF32 stream query key value decayLog beta stateHistory
    outputGradient queryGradient keyGradient valueGradient decayLogGradient betaGradient
    stateGradient tokens32 keyHeads32 valueHeads32 keyWidth32 valueWidth32 queryScale
  backward.waitChecked "Qwen3.6 normalized reverse recurrence"
  checkClose "recurrent dq normalized" (← queryGradient.copyTo stream)
    (← fixture dir "grad.layer0.deltanet_q_l2_repeated.8x4x64" prefillElements)
    prefillElements 3e-5
  checkClose "recurrent dk normalized" (← keyGradient.copyTo stream)
    (← fixture dir "grad.layer0.deltanet_k_l2_repeated.8x4x64" prefillElements)
    prefillElements 3e-5
  checkClose "recurrent dv" (← valueGradient.copyTo stream)
    (← fixture dir "grad.layer0.deltanet_v_heads.8x4x64" prefillElements)
    prefillElements 3e-5
  checkClose "recurrent dg" (← decayLogGradient.copyTo stream)
    (← fixture dir "grad.layer0.deltanet_g.8x4" (tokens * valueHeads))
    (tokens * valueHeads) 3e-5
  checkClose "recurrent dbeta" (← betaGradient.copyTo stream)
    (← fixture dir "grad.layer0.deltanet_beta.8x4" (tokens * valueHeads))
    (tokens * valueHeads) 3e-5

private def gateFullStage (dir : String) (stream : Cuda.Stream) : IO Unit := do
  let hidden : Nat := 256
  let convolutionWidth : Nat := 512
  let compactElements : Nat := tokens * keyHeads * keyWidth
  let valueElements : Nat := tokens * valueHeads * valueWidth
  let gateElements : Nat := tokens * valueHeads
  let inputNorm ← upload dir "model.layers.0.input_layernorm.weight.256" hidden stream
  let queryKeyValueWeight ← upload dir
    "model.layers.0.linear_attn.in_proj_qkv.weight.512x256" (convolutionWidth * hidden) stream
  let zWeight ← upload dir "model.layers.0.linear_attn.in_proj_z.weight.256x256"
    (hidden * hidden) stream
  let bWeight ← upload dir "model.layers.0.linear_attn.in_proj_b.weight.4x256"
    (valueHeads * hidden) stream
  let aWeight ← upload dir "model.layers.0.linear_attn.in_proj_a.weight.4x256"
    (valueHeads * hidden) stream
  let convolution ← upload dir "model.layers.0.linear_attn.conv1d.weight.512x4"
    (convolutionWidth * 4) stream
  let aLog ← upload dir "model.layers.0.linear_attn.A_log.4" valueHeads stream
  let dtBias ← upload dir "model.layers.0.linear_attn.dt_bias.4" valueHeads stream
  let gatedNorm ← upload dir "model.layers.0.linear_attn.norm.weight.64" valueWidth stream
  let outputWeight ← upload dir "model.layers.0.linear_attn.out_proj.weight.256x256"
    (hidden * hidden) stream
  let weights : StageWeightsF32 := {
    inputNorm, queryKeyValue := queryKeyValueWeight, z := zWeight, b := bWeight, a := aWeight,
    convolution, aLog, dtBias, gatedNorm, output := outputWeight
  }
  let input ← upload dir "layer0.hidden_in.8x256" (tokens * hidden) stream
  let normalizedInput ← zeros (tokens * hidden) stream
  let inputInverseRms ← zeros tokens stream
  let queryKeyValuePreConv ← zeros (tokens * convolutionWidth) stream
  let z ← zeros valueElements stream
  let b ← zeros gateElements stream
  let a ← zeros gateElements stream
  let queryKeyValuePostConv ← zeros (tokens * convolutionWidth) stream
  let queryCompact ← zeros compactElements stream
  let keyCompact ← zeros compactElements stream
  let value ← zeros valueElements stream
  let queryRepeated ← zeros valueElements stream
  let keyRepeated ← zeros valueElements stream
  let queryNormed ← zeros valueElements stream
  let keyNormed ← zeros valueElements stream
  let decayLog ← zeros gateElements stream
  let beta ← zeros gateElements stream
  let stateInput ← zeros stateElements stream
  let stateOutput ← zeros stateElements stream
  let stateHistory ← zeros ((tokens + 1) * stateElements) stream
  let deltaOutput ← zeros valueElements stream
  let gatedInverseRms ← zeros (tokens * valueHeads) stream
  let gatedOutput ← zeros valueElements stream
  let stageOutput ← zeros (tokens * hidden) stream
  let forward : StageForwardF32 := {
    input, normalizedInput, inputInverseRms, queryKeyValuePreConv, z, b, a,
    queryKeyValuePostConv, queryCompact, keyCompact, value, queryRepeated, keyRepeated,
    queryNormed, keyNormed, decayLog, beta, stateInput, stateOutput, stateHistory, deltaOutput,
    gatedInverseRms, gatedOutput, output := stageOutput
  }
  forwardStageF32 stream weights forward 8 256 2 4 64 64 epsilon queryScale
  checkBuffer dir "DeltaNet input norm" "layer0.norm1_out.8x256" normalizedInput
    (tokens * hidden) stream
  checkBuffer dir "DeltaNet QKV projection" "layer0.qkv_pre_conv.8x512"
    queryKeyValuePreConv (tokens * convolutionWidth) stream
  checkBuffer dir "DeltaNet q post-conv" "layer0.q_post_conv.8x128" queryCompact
    compactElements stream
  checkBuffer dir "DeltaNet k post-conv" "layer0.k_post_conv.8x128" keyCompact
    compactElements stream
  checkBuffer dir "DeltaNet v post-conv" "layer0.v_post_conv.8x256" value valueElements stream
  checkBuffer dir "DeltaNet q normalized" "layer0.q_l2.8x4x64" queryNormed
    valueElements stream
  checkBuffer dir "DeltaNet k normalized" "layer0.k_l2.8x4x64" keyNormed
    valueElements stream
  checkBuffer dir "DeltaNet decay log" "layer0.g.8x4" decayLog gateElements stream
  checkBuffer dir "DeltaNet beta" "layer0.beta.8x4" beta gateElements stream
  checkBuffer dir "DeltaNet recurrent output" "layer0.delta_out.8x4x64" deltaOutput
    valueElements stream
  checkBuffer dir "DeltaNet final state" "layer0.S_final.4x64x64" stateOutput
    stateElements stream
  checkBuffer dir "DeltaNet gated output" "layer0.gated_out.8x256" gatedOutput
    valueElements stream
  checkBuffer dir "DeltaNet mixer output" "layer0.mixer_out.8x256" stageOutput
    (tokens * hidden) stream

  let outputGradient ← upload dir "grad.layer0.deltanet_output.8x256" (tokens * hidden) stream
  let gatedOutputGradient ← zeros valueElements stream
  let deltaOutputGradient ← zeros valueElements stream
  let zGradient ← zeros valueElements stream
  let gatedNormWeightGradient ← zeros valueWidth stream
  let queryNormGradient ← zeros valueElements stream
  let keyNormGradient ← zeros valueElements stream
  let valueGradient ← zeros valueElements stream
  let decayLogGradient ← zeros gateElements stream
  let betaGradient ← zeros gateElements stream
  let stateGradient ← zeros stateElements stream
  let queryRepeatedGradient ← zeros valueElements stream
  let keyRepeatedGradient ← zeros valueElements stream
  let queryCompactGradient ← zeros compactElements stream
  let keyCompactGradient ← zeros compactElements stream
  let queryKeyValuePostConvGradient ← zeros (tokens * convolutionWidth) stream
  let queryKeyValuePreConvGradient ← zeros (tokens * convolutionWidth) stream
  let convolutionWeightGradient ← zeros (convolutionWidth * 4) stream
  let aGradient ← zeros gateElements stream
  let bGradient ← zeros gateElements stream
  let aLogGradient ← zeros valueHeads stream
  let dtBiasGradient ← zeros valueHeads stream
  let queryKeyValueInputGradient ← zeros (tokens * hidden) stream
  let zInputGradient ← zeros (tokens * hidden) stream
  let bInputGradient ← zeros (tokens * hidden) stream
  let aInputGradient ← zeros (tokens * hidden) stream
  let queryKeyValueZInputGradient ← zeros (tokens * hidden) stream
  let queryKeyValueZBInputGradient ← zeros (tokens * hidden) stream
  let normalizedInputGradient ← zeros (tokens * hidden) stream
  let inputGradient ← zeros (tokens * hidden) stream
  let inputNormWeightGradient ← zeros hidden stream
  let queryKeyValueWeightGradient ← zeros (convolutionWidth * hidden) stream
  let zWeightGradient ← zeros (hidden * hidden) stream
  let bWeightGradient ← zeros (valueHeads * hidden) stream
  let aWeightGradient ← zeros (valueHeads * hidden) stream
  let outputWeightGradient ← zeros (hidden * hidden) stream
  let backward : StageBackwardF32 := {
    outputGradient, gatedOutputGradient, deltaOutputGradient, zGradient,
    gatedNormWeightGradient, queryNormGradient, keyNormGradient, valueGradient,
    decayLogGradient, betaGradient, stateGradient, queryRepeatedGradient, keyRepeatedGradient,
    queryCompactGradient, keyCompactGradient, queryKeyValuePostConvGradient,
    queryKeyValuePreConvGradient, convolutionWeightGradient, aGradient, bGradient, aLogGradient,
    dtBiasGradient, queryKeyValueInputGradient, zInputGradient, bInputGradient, aInputGradient,
    queryKeyValueZInputGradient, queryKeyValueZBInputGradient, normalizedInputGradient,
    inputGradient, inputNormWeightGradient, queryKeyValueWeightGradient, zWeightGradient,
    bWeightGradient, aWeightGradient, outputWeightGradient
  }
  backwardStageF32 stream weights forward backward 8 256 2 4 64 64 epsilon queryScale
  checkBuffer dir "DeltaNet dgated output" "grad.layer0.deltanet_gated_out.8x256"
    gatedOutputGradient valueElements stream
  checkBuffer dir "DeltaNet ddelta output" "grad.layer0.deltanet_delta_out.8x4x64"
    deltaOutputGradient valueElements stream
  checkBuffer dir "DeltaNet dz" "grad.layer0.deltanet_z.8x256" zGradient valueElements stream
  checkBuffer dir "DeltaNet dq normalized" "grad.layer0.deltanet_q_l2_repeated.8x4x64"
    queryNormGradient valueElements stream
  checkBuffer dir "DeltaNet dk normalized" "grad.layer0.deltanet_k_l2_repeated.8x4x64"
    keyNormGradient valueElements stream
  checkBuffer dir "DeltaNet dv" "grad.layer0.deltanet_v_heads.8x4x64" valueGradient
    valueElements stream
  checkBuffer dir "DeltaNet dg" "grad.layer0.deltanet_g.8x4" decayLogGradient
    gateElements stream
  checkBuffer dir "DeltaNet dbeta" "grad.layer0.deltanet_beta.8x4" betaGradient
    gateElements stream
  checkBuffer dir "DeltaNet dq repeated" "grad.layer0.deltanet_q_repeated.8x4x64"
    queryRepeatedGradient valueElements stream
  checkBuffer dir "DeltaNet dk repeated" "grad.layer0.deltanet_k_repeated.8x4x64"
    keyRepeatedGradient valueElements stream
  checkBuffer dir "DeltaNet dq post-conv" "grad.layer0.deltanet_q_post_conv.8x2x64"
    queryCompactGradient compactElements stream
  checkBuffer dir "DeltaNet dk post-conv" "grad.layer0.deltanet_k_post_conv.8x2x64"
    keyCompactGradient compactElements stream
  checkBuffer dir "DeltaNet dQKV post-conv" "grad.layer0.deltanet_qkv_post_conv.8x512"
    queryKeyValuePostConvGradient (tokens * convolutionWidth) stream
  checkBuffer dir "DeltaNet dQKV pre-conv" "grad.layer0.deltanet_qkv_pre_conv.8x512"
    queryKeyValuePreConvGradient (tokens * convolutionWidth) stream
  checkBuffer dir "DeltaNet da" "grad.layer0.deltanet_a.8x4" aGradient gateElements stream
  checkBuffer dir "DeltaNet db" "grad.layer0.deltanet_b.8x4" bGradient gateElements stream
  checkBuffer dir "DeltaNet dnormalized input" "grad.layer0.deltanet_norm1.8x256"
    normalizedInputGradient (tokens * hidden) stream
  checkBuffer dir "DeltaNet dinput" "grad.layer0.deltanet_input.8x256" inputGradient
    (tokens * hidden) stream
  checkBuffer dir "DeltaNet dgated norm weight"
    "grad.model.layers.0.linear_attn.norm.weight.64" gatedNormWeightGradient valueWidth stream
  checkBuffer dir "DeltaNet dA_log" "grad.model.layers.0.linear_attn.A_log.4"
    aLogGradient valueHeads stream
  checkBuffer dir "DeltaNet ddt_bias" "grad.model.layers.0.linear_attn.dt_bias.4"
    dtBiasGradient valueHeads stream
  checkBuffer dir "DeltaNet dconv weight"
    "grad.model.layers.0.linear_attn.conv1d.weight.512x4" convolutionWeightGradient
    (convolutionWidth * 4) stream
  checkBuffer dir "DeltaNet dinput norm weight"
    "grad.model.layers.0.input_layernorm.weight.256" inputNormWeightGradient hidden stream
  checkBuffer dir "DeltaNet dQKV weight"
    "grad.model.layers.0.linear_attn.in_proj_qkv.weight.512x256"
    queryKeyValueWeightGradient (convolutionWidth * hidden) stream
  checkBuffer dir "DeltaNet dz weight"
    "grad.model.layers.0.linear_attn.in_proj_z.weight.256x256" zWeightGradient
    (hidden * hidden) stream
  checkBuffer dir "DeltaNet db weight"
    "grad.model.layers.0.linear_attn.in_proj_b.weight.4x256" bWeightGradient
    (valueHeads * hidden) stream
  checkBuffer dir "DeltaNet da weight"
    "grad.model.layers.0.linear_attn.in_proj_a.weight.4x256" aWeightGradient
    (valueHeads * hidden) stream
  checkBuffer dir "DeltaNet dout weight"
    "grad.model.layers.0.linear_attn.out_proj.weight.256x256" outputWeightGradient
    (hidden * hidden) stream

private def referenceDirectory : IO String := do
  let some directory ← IO.getEnv "QWEN36_REFERENCE_DIR"
    | LeanTest.fail "QWEN36_REFERENCE_DIR is required"
  return directory

@[test]
def recurrentPrefillDecodeAndVjpMatchReference : IO Unit := do
  let referenceDir ← referenceDirectory
  let stream ← Cuda.Stream.default
  for layer in [:3] do
    gatePrefill referenceDir layer stream
    gateDecode referenceDir layer stream
  gateReverseRecurrence referenceDir stream
  gateFullStage referenceDir stream

end Qwen36DeltaNetGate
