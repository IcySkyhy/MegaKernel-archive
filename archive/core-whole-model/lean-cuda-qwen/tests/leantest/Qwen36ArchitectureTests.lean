/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Qwen36Architecture

namespace Qwen36ArchitectureTests

open Qwen36.Architecture
open Qwen36.Checkpoint

@[test]
def layerSchedule : IO Unit := do
  let mut linear := 0
  let mut full := 0
  for layer in [:layerCount] do
    match layerKind layer with
    | .linearAttention => linear := linear + 1
    | .fullAttention =>
      full := full + 1
      LeanTest.assertEqual ((layer + 1) % 4) 0
  LeanTest.assertEqual linear linearLayerCount
  LeanTest.assertEqual full fullAttentionLayerCount

private def checkpointRoot : IO System.FilePath := do
  let some configured ← IO.getEnv "QWEN36_MODEL_DIR"
    | throw (IO.userError "set QWEN36_MODEL_DIR to the local Qwen/Qwen3.6-27B snapshot")
  IO.FS.realPath configured

@[test_ignore]
def realTextModelLayout : IO Unit := do
  let manifest ← readManifest (← checkpointRoot)
  let layout ← readTextLayout manifest
  LeanTest.assertEqual layout.layers.size layerCount
  LeanTest.assertEqual layout.linearLayers linearLayerCount
  LeanTest.assertEqual layout.fullAttentionLayers fullAttentionLayerCount
  let tensors := layout.tensorRefs
  LeanTest.assertEqual tensors.size 851
  let mut names : Array String := #[]
  for tensor in tensors do
    LeanTest.assertFalse (names.contains tensor.name) s!"duplicate layout tensor: {tensor.name}"
    LeanTest.assertTrue (tensor.shard < 15) s!"invalid shard for {tensor.name}"
    LeanTest.assertEqual (tensor.byteOffset % 2) 0 (some s!"unaligned BF16 tensor: {tensor.name}")
    LeanTest.assertEqual (tensor.byteCount % 2) 0 (some s!"odd BF16 size: {tensor.name}")
    names := names.push tensor.name
  for layer in [:layerCount] do
    let some weights := layout.layers[layer]?
      | throw <| IO.userError s!"missing resolved layer {layer}"
    LeanTest.assertEqual weights.index layer
    match layerKind layer, weights.attention with
    | .linearAttention, .linear _ => pure ()
    | .fullAttention, .full _ => pure ()
    | expected, actual =>
      throw <| IO.userError
        s!"layer {layer} kind mismatch: expected {repr expected}, got {repr actual}"
  LeanTest.assertTrue (layout.totalBytes > 40 * 1024 * 1024 * 1024)
    "resolved text weights should exceed 40 GiB"
  LeanTest.assertTrue (layout.totalBytes < manifest.index.totalSize.toUInt64)
    "text-only layout must exclude vision and MTP weights"
  IO.println s!"Qwen3.6 text layout bytes: {layout.totalBytes}"

end Qwen36ArchitectureTests
