/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-!
# Qwen3.6 MLP forward and VJP gate

Checks the tiny layer-zero RMSNorm/SwiGLU MLP, every saved forward activation, its input VJP, and
all four trainable gradients against the deterministic full-model NumPy reference.
-/

namespace Qwen36MLPGate

open Cuda.Qwen36.MLP

private def rows : Nat := 8
private def hidden : Nat := 256
private def intermediate : Nat := 512
private def hiddenElements : Nat := rows * hidden
private def intermediateElements : Nat := rows * intermediate

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

private def referenceDirectory : IO String := do
  let some directory ← IO.getEnv "QWEN36_REFERENCE_DIR"
    | LeanTest.fail "QWEN36_REFERENCE_DIR is required"
  return directory

@[test]
def forwardAndVjpMatchReference : IO Unit := do
  let referenceDir ← referenceDirectory
  let stream ← Cuda.Stream.default
  let input ← upload referenceDir "layer0.mixer_residual.8x256" hiddenElements stream
  let normWeight ← upload referenceDir
    "model.layers.0.post_attention_layernorm.weight.256" hidden stream
  let gateWeight ← upload referenceDir "model.layers.0.mlp.gate_proj.weight.512x256"
    (intermediate * hidden) stream
  let upWeight ← upload referenceDir "model.layers.0.mlp.up_proj.weight.512x256"
    (intermediate * hidden) stream
  let downWeight ← upload referenceDir "model.layers.0.mlp.down_proj.weight.256x512"
    (hidden * intermediate) stream
  let normalized ← zeros hiddenElements stream
  let inverseRms ← zeros rows stream
  let gate ← zeros intermediateElements stream
  let up ← zeros intermediateElements stream
  let activation ← zeros intermediateElements stream
  let output ← zeros hiddenElements stream
  forwardF32 stream input normWeight normalized inverseRms gate up activation output gateWeight
    upWeight downWeight 8 256 512 1e-6
  checkBuffer referenceDir "MLP normalized input" "layer0.norm2_out.8x256" normalized
    hiddenElements stream
  checkBuffer referenceDir "MLP gate projection" "layer0.mlp_gate.8x512" gate
    intermediateElements stream
  checkBuffer referenceDir "MLP up projection" "layer0.mlp_up.8x512" up
    intermediateElements stream
  checkBuffer referenceDir "MLP SwiGLU activation" "layer0.mlp_act.8x512" activation
    intermediateElements stream
  checkBuffer referenceDir "MLP output" "layer0.mlp_out.8x256" output hiddenElements stream

  let outputGradient ← upload referenceDir "grad.layer0.mlp_output.8x256" hiddenElements stream
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
  backwardF32 stream input normWeight normalized inverseRms gate up activation outputGradient
    activationGradient gateGradient upGradient normGradientGate normGradientUp normGradient
    inputGradient normWeightGradient gateWeightGradient upWeightGradient downWeightGradient
    gateWeight upWeight downWeight 8 256 512
  checkBuffer referenceDir "MLP dactivation" "grad.layer0.mlp_act.8x512"
    activationGradient intermediateElements stream
  checkBuffer referenceDir "MLP dgate" "grad.layer0.mlp_gate.8x512"
    gateGradient intermediateElements stream
  checkBuffer referenceDir "MLP dup" "grad.layer0.mlp_up.8x512" upGradient
    intermediateElements stream
  checkBuffer referenceDir "MLP dnormalized" "grad.layer0.mlp_norm2.8x256" normGradient
    hiddenElements stream
  checkBuffer referenceDir "MLP dinput" "grad.layer0.mlp_input.8x256" inputGradient
    hiddenElements stream
  checkBuffer referenceDir "MLP dnorm weight"
    "grad.model.layers.0.post_attention_layernorm.weight.256" normWeightGradient hidden stream
  checkBuffer referenceDir "MLP dgate weight"
    "grad.model.layers.0.mlp.gate_proj.weight.512x256" gateWeightGradient
    (intermediate * hidden) stream
  checkBuffer referenceDir "MLP dup weight"
    "grad.model.layers.0.mlp.up_proj.weight.512x256" upWeightGradient
    (intermediate * hidden) stream
  checkBuffer referenceDir "MLP ddown weight"
    "grad.model.layers.0.mlp.down_proj.weight.256x512" downWeightGradient
    (hidden * intermediate) stream

end Qwen36MLPGate
