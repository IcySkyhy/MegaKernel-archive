/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Qwen36Checkpoint

namespace Qwen36CheckpointTests

open Qwen36.Checkpoint

@[test]
def safetensorsHeaderParser : IO Unit := do
  let text :=
    "{\"weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}," ++
    "\"bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[12,20]}}"
  let header ← IO.ofExcept <| parseHeaderText text.toUTF8.size text
  LeanTest.assertEqual header.dataBytes 20
  LeanTest.assertEqual header.tensors.size 2
  LeanTest.assertEqual (Option.map (fun (tensor : TensorInfo) => tensor.shape) (header.findTensor? "weight")) (some #[2, 3])
  LeanTest.assertEqual (Option.map (fun (tensor : TensorInfo) => tensor.dtype) (header.findTensor? "bias")) (some "F32")

private def checkpointRoot : IO System.FilePath := do
  let some configured ← IO.getEnv "QWEN36_MODEL_DIR"
    | throw (IO.userError "set QWEN36_MODEL_DIR to the local Qwen/Qwen3.6-27B snapshot")
  IO.FS.realPath configured

private def assertTensorShape
    (manifest : Manifest) (name dtype : String) (shape : Array Nat) : IO Unit := do
  let some (_, tensor) := manifest.findTensor? name
    | throw <| IO.userError s!"missing checkpoint tensor: {name}"
  LeanTest.assertEqual tensor.dtype dtype (some s!"wrong dtype for {name}")
  LeanTest.assertEqual tensor.shape shape (some s!"wrong shape for {name}")

@[test_ignore]
def realCheckpointManifestAndTensorStream : IO Unit := do
  let manifest ← readManifest (← checkpointRoot)
  LeanTest.assertEqual manifest.shards.size 15
  LeanTest.assertEqual manifest.index.weights.size 1199
  LeanTest.assertEqual manifest.index.totalSize 55562855904
  assertTensorShape manifest "model.language_model.embed_tokens.weight" "BF16" #[248320, 5120]
  assertTensorShape manifest
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" "BF16" #[10240, 5120]
  assertTensorShape manifest
    "model.language_model.layers.0.linear_attn.conv1d.weight" "BF16" #[10240, 1, 4]
  assertTensorShape manifest
    "model.language_model.layers.0.linear_attn.A_log" "BF16" #[48]
  assertTensorShape manifest
    "model.language_model.layers.11.self_attn.q_proj.weight" "BF16" #[12288, 5120]
  assertTensorShape manifest
    "model.language_model.layers.11.self_attn.k_proj.weight" "BF16" #[1024, 5120]
  assertTensorShape manifest
    "model.language_model.layers.0.mlp.gate_proj.weight" "BF16" #[17408, 5120]
  assertTensorShape manifest "lm_head.weight" "BF16" #[248320, 5120]
  let (tensor, bytes) ← readTensor manifest
    "model.language_model.layers.32.input_layernorm.weight"
  LeanTest.assertEqual tensor.shape #[5120]
  LeanTest.assertEqual bytes.size 10240
  LeanTest.assertEqual (bytes.extract 0 32) <| ByteArray.mk #[
    0x4e, 0xbc, 0x41, 0xbe, 0xa7, 0x3e, 0x95, 0xbc,
    0x6c, 0xbe, 0x89, 0x3c, 0xed, 0x3d, 0x70, 0xbe,
    0x53, 0xbe, 0x62, 0x3d, 0x46, 0xbc, 0x46, 0x3d,
    0x79, 0x3d, 0xd1, 0xbd, 0x8f, 0xbc, 0x25, 0xbe]

end Qwen36CheckpointTests
