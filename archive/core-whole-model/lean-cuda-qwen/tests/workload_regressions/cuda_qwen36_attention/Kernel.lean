/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-!
# Qwen3.6 grouped-query attention gate

Checks causal probabilities and the pre/post sigmoid-gate outputs of the tiny full-attention layer
against the deterministic NumPy/HF reference.
-/

namespace Qwen36AttentionGate

open Cuda.Qwen36.Attention

private def tokens : Nat := 8
private def queryHeads : Nat := 4
private def keyValueHeads : Nat := 2
private def width : Nat := 64
private def queryElements : Nat := tokens * queryHeads * width
private def keyValueElements : Nat := tokens * keyValueHeads * width
private def probabilityElements : Nat := queryHeads * tokens * tokens

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

private def gateFullStage (referenceDir : String) (stream : Cuda.Stream) : IO Unit := do
  let hidden : Nat := 256
  let queryProjection : Nat := 512
  let queryElements : Nat := 8 * 4 * 64
  let keyValueElements : Nat := 8 * 2 * 64
  let probabilityElements : Nat := 4 * 8 * 8
  let input ← upload referenceDir "layer3.hidden_in.8x256" (8 * hidden) stream
  let inputNorm ← upload referenceDir "model.layers.3.input_layernorm.weight.256" hidden stream
  let queryWeight ← upload referenceDir "model.layers.3.self_attn.q_proj.weight.512x256"
    (queryProjection * hidden) stream
  let keyWeight ← upload referenceDir "model.layers.3.self_attn.k_proj.weight.128x256"
    (128 * hidden) stream
  let valueWeight ← upload referenceDir "model.layers.3.self_attn.v_proj.weight.128x256"
    (128 * hidden) stream
  let queryNormWeight ← upload referenceDir "model.layers.3.self_attn.q_norm.weight.64" 64 stream
  let keyNormWeight ← upload referenceDir "model.layers.3.self_attn.k_norm.weight.64" 64 stream
  let outputWeight ← upload referenceDir "model.layers.3.self_attn.o_proj.weight.256x256"
    (hidden * hidden) stream
  let inverseFrequency ← zeros 8 stream
  let inverseHandle ← Cuda.Qwen36.Primitives.ropeInvFreq stream inverseFrequency 8 16 1e7
  inverseHandle.waitChecked "Qwen3.6 attention inverse frequency"
  let weights : StageWeightsF32 := {
    inputNorm, query := queryWeight, key := keyWeight, value := valueWeight,
    queryNorm := queryNormWeight, keyNorm := keyNormWeight, output := outputWeight,
    inverseFrequency
  }
  let normalizedInput ← zeros (8 * hidden) stream
  let inputInverseRms ← zeros 8 stream
  let queryGateProjection ← zeros (8 * queryProjection) stream
  let queryPreNorm ← zeros queryElements stream
  let stageGate ← zeros queryElements stream
  let keyPreNorm ← zeros keyValueElements stream
  let stageValue ← zeros keyValueElements stream
  let queryNormed ← zeros queryElements stream
  let queryInverseRms ← zeros (8 * 4) stream
  let keyNormed ← zeros keyValueElements stream
  let keyInverseRms ← zeros (8 * 2) stream
  let queryRope ← zeros queryElements stream
  let keyRope ← zeros keyValueElements stream
  let probabilities ← zeros probabilityElements stream
  let preGate ← zeros queryElements stream
  let postGate ← zeros queryElements stream
  let stageOutput ← zeros (8 * hidden) stream
  let forward : StageForwardF32 := {
    input, normalizedInput, inputInverseRms, queryGateProjection, queryPreNorm, gate := stageGate,
    keyPreNorm, value := stageValue, queryNormed, queryInverseRms, keyNormed, keyInverseRms,
    queryRope, keyRope, probabilities, preGate, postGate, output := stageOutput
  }
  forwardStageF32 stream weights forward 8 256 4 2 64 8 1e-6 0.125
  checkBuffer referenceDir "attention input norm" "layer3.norm1_out.8x256" normalizedInput
    (8 * hidden) stream
  checkBuffer referenceDir "attention q/g projection" "layer3.qg_proj.8x512"
    queryGateProjection (8 * queryProjection) stream
  checkBuffer referenceDir "attention q pre-norm" "layer3.q_pre_norm.8x4x64" queryPreNorm
    queryElements stream
  checkBuffer referenceDir "attention gate split" "layer3.gate.8x256" stageGate queryElements stream
  checkBuffer referenceDir "attention k pre-norm" "layer3.k_pre_norm.8x2x64" keyPreNorm
    keyValueElements stream
  checkBuffer referenceDir "attention value projection" "layer3.v_proj.8x128" stageValue
    keyValueElements stream
  checkBuffer referenceDir "attention q norm" "layer3.q_norm.8x4x64" queryNormed
    queryElements stream
  checkBuffer referenceDir "attention k norm" "layer3.k_norm.8x2x64" keyNormed
    keyValueElements stream
  checkBuffer referenceDir "attention q RoPE" "layer3.q_rope.8x4x64" queryRope
    queryElements stream
  checkBuffer referenceDir "attention k RoPE" "layer3.k_rope.8x2x64" keyRope
    keyValueElements stream
  checkBuffer referenceDir "attention stage probabilities" "layer3.attn_probs.4x8x8"
    probabilities probabilityElements stream
  checkBuffer referenceDir "attention stage pre-gate" "layer3.attn_out_pre_gate.8x256" preGate
    queryElements stream
  checkBuffer referenceDir "attention stage post-gate" "layer3.attn_out_post_gate.8x256"
    postGate queryElements stream
  checkBuffer referenceDir "attention mixer output" "layer3.mixer_out.8x256" stageOutput
    (8 * hidden) stream

  let outputGradient ← upload referenceDir "grad.layer3.attention_output.8x256" (8 * hidden) stream
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
  let queryNormWeightGradient ← zeros 64 stream
  let keyNormWeightGradient ← zeros 64 stream
  let queryGateProjectionGradient ← zeros (8 * queryProjection) stream
  let queryInputGradient ← zeros (8 * hidden) stream
  let keyInputGradient ← zeros (8 * hidden) stream
  let valueInputGradient ← zeros (8 * hidden) stream
  let queryKeyInputGradient ← zeros (8 * hidden) stream
  let normalizedInputGradient ← zeros (8 * hidden) stream
  let inputGradient ← zeros (8 * hidden) stream
  let inputNormWeightGradient ← zeros hidden stream
  let queryWeightGradient ← zeros (queryProjection * hidden) stream
  let keyWeightGradient ← zeros (128 * hidden) stream
  let valueWeightGradient ← zeros (128 * hidden) stream
  let outputWeightGradient ← zeros (hidden * hidden) stream
  let backward : StageBackwardF32 := {
    outputGradient, postGateGradient, preGateGradient, gateGradient, probabilityGradient,
    scoreGradient, queryRopeGradient, keyRopeGradient, valueGradient, queryNormGradient,
    keyNormGradient, queryPreNormGradient, keyPreNormGradient, queryNormWeightGradient,
    keyNormWeightGradient, queryGateProjectionGradient, queryInputGradient, keyInputGradient,
    valueInputGradient, queryKeyInputGradient, normalizedInputGradient, inputGradient,
    inputNormWeightGradient, queryWeightGradient, keyWeightGradient, valueWeightGradient,
    outputWeightGradient
  }
  backwardStageF32 stream weights forward backward 8 256 4 2 64 8 0.125
  checkBuffer referenceDir "attention dpost-gate" "grad.layer3.attention_post_gate.8x256"
    postGateGradient queryElements stream
  checkBuffer referenceDir "attention dpre-gate" "grad.layer3.attention_pre_gate.8x256"
    preGateGradient queryElements stream
  checkBuffer referenceDir "attention dgate" "grad.layer3.attention_gate.8x256" gateGradient
    queryElements stream
  checkBuffer referenceDir "attention dprobabilities" "grad.layer3.attention_probs.4x8x8"
    probabilityGradient probabilityElements stream
  checkBuffer referenceDir "attention dscores" "grad.layer3.attention_scores.4x8x8"
    scoreGradient probabilityElements stream
  checkBuffer referenceDir "attention dq RoPE" "grad.layer3.attention_q_rope.8x4x64"
    queryRopeGradient queryElements stream
  checkBuffer referenceDir "attention dk RoPE" "grad.layer3.attention_k_rope.8x2x64"
    keyRopeGradient keyValueElements stream
  checkBuffer referenceDir "attention dv" "grad.layer3.attention_v_heads.8x2x64" valueGradient
    keyValueElements stream
  checkBuffer referenceDir "attention dq norm" "grad.layer3.attention_q_norm.8x4x64"
    queryNormGradient queryElements stream
  checkBuffer referenceDir "attention dk norm" "grad.layer3.attention_k_norm.8x2x64"
    keyNormGradient keyValueElements stream
  checkBuffer referenceDir "attention dq pre-norm"
    "grad.layer3.attention_q_pre_norm.8x4x64" queryPreNormGradient queryElements stream
  checkBuffer referenceDir "attention dk pre-norm"
    "grad.layer3.attention_k_pre_norm.8x2x64" keyPreNormGradient keyValueElements stream
  checkBuffer referenceDir "attention dq/g projection" "grad.layer3.attention_qg_proj.8x512"
    queryGateProjectionGradient (8 * queryProjection) stream
  checkBuffer referenceDir "attention dnormalized input" "grad.layer3.attention_norm1.8x256"
    normalizedInputGradient (8 * hidden) stream
  checkBuffer referenceDir "attention dinput" "grad.layer3.attention_input.8x256" inputGradient
    (8 * hidden) stream
  checkBuffer referenceDir "attention dq norm weight"
    "grad.model.layers.3.self_attn.q_norm.weight.64" queryNormWeightGradient 64 stream
  checkBuffer referenceDir "attention dk norm weight"
    "grad.model.layers.3.self_attn.k_norm.weight.64" keyNormWeightGradient 64 stream
  checkBuffer referenceDir "attention dinput norm weight"
    "grad.model.layers.3.input_layernorm.weight.256" inputNormWeightGradient hidden stream
  checkBuffer referenceDir "attention dq weight"
    "grad.model.layers.3.self_attn.q_proj.weight.512x256" queryWeightGradient
    (queryProjection * hidden) stream
  checkBuffer referenceDir "attention dk weight"
    "grad.model.layers.3.self_attn.k_proj.weight.128x256" keyWeightGradient (128 * hidden) stream
  checkBuffer referenceDir "attention dv weight"
    "grad.model.layers.3.self_attn.v_proj.weight.128x256" valueWeightGradient (128 * hidden) stream
  checkBuffer referenceDir "attention do weight"
    "grad.model.layers.3.self_attn.o_proj.weight.256x256" outputWeightGradient
    (hidden * hidden) stream

private def referenceDirectory : IO String := do
  let some directory ← IO.getEnv "QWEN36_REFERENCE_DIR"
    | LeanTest.fail "QWEN36_REFERENCE_DIR is required"
  return directory

@[test]
def causalGqaForwardAndVjpMatchReference : IO Unit := do
  let referenceDir ← referenceDirectory
  let stream ← Cuda.Stream.default
  let query ← upload referenceDir "layer3.q_rope.8x4x64" queryElements stream
  let key ← upload referenceDir "layer3.k_rope.8x2x64" keyValueElements stream
  let value ← upload referenceDir "layer3.v_heads.8x2x64" keyValueElements stream
  let gate ← upload referenceDir "layer3.gate.8x256" queryElements stream
  let pre ← zeros queryElements stream
  let post ← zeros queryElements stream
  let probabilities ← zeros probabilityElements stream
  let handle ← causalForwardF32 stream query key value gate pre post probabilities
    8 4 2 64 0.125
  handle.waitChecked "Qwen3.6 grouped-query attention"
  checkClose "attention probabilities" (← probabilities.copyTo stream)
    (← fixture referenceDir "layer3.attn_probs.4x8x8" probabilityElements)
    probabilityElements 3e-5
  checkClose "attention output before gate" (← pre.copyTo stream)
    (← fixture referenceDir "layer3.attn_out_pre_gate.8x256" queryElements)
    queryElements 3e-5
  checkClose "attention output after gate" (← post.copyTo stream)
    (← fixture referenceDir "layer3.attn_out_post_gate.8x256" queryElements)
    queryElements 3e-5
  gateFullStage referenceDir stream

end Qwen36AttentionGate
