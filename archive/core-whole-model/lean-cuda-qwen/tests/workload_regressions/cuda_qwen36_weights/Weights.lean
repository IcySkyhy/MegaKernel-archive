/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Lean.Cuda.Types
import Lean.Cuda.Device
import LeanCudaQwen.Qwen36.SafeTensors

/-!
# Qwen3.6 safetensors loader gates

Two modes, both driven off the checked-in fixtures under `fixtures/` (regenerate with
`gen_fixtures.py`; payloads follow `byte j = (j*131 + seed*17 + 7) % 256`):

* `weights_test host <fixtures>` — hardware-independent: header parse, whole-file
  `parseHeader` agreement, single-file and sharded-directory introspection, host-side
  byte extraction, the HF-name registry (tiny config + vendored trimmed real 27B index),
  and `checkComplete` accept/reject cases.
* `weights_test device <fixtures>` — needs an sm_121 GPU: `loadTensor`/`loadTensors`
  upload fixture tensors to device buffers and read them back byte-exact, including
  index-based shard resolution and forced multi-chunk payload uploads.
* `weights_test stream-checkpoint <directory>` — safely exercise every real payload through the
  bounded loader one shard at a time, releasing it before the next allocation and reporting the
  process resident/high-water memory after each shard.
-/

open Cuda.Qwen36.SafeTensors

/-- Deterministic fixture payload byte; mirrors `gen_fixtures.py`. -/
def patternByte (j seed : Nat) : UInt8 := ((j * 131 + seed * 17 + 7) % 256).toUInt8

def pattern (size seed : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity size
  for j in Array.range size do
    out := out.push (patternByte j seed)
  return out

def fail (msg : String) : IO α := LeanTest.fail msg

def check (what : String) (cond : Bool) : IO Unit := do
  unless cond do fail what

def checkEq [BEq α] [Repr α] (what : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    fail s!"{what}\n  expected: {reprStr expected}\n  actual:   {reprStr actual}"

def checkBytes (what : String) (actual expected : ByteArray) : IO Unit := do
  if actual == expected then
    return
  let mismatch := Id.run (α := Option Nat) do
    for j in Array.range (min actual.size expected.size) do
      if actual[j]! != expected[j]! then return some j
    return none
  fail s!"{what}: sizes {actual.size} vs {expected.size}, first mismatch at {mismatch}"

/-- Single-file and sharded fixtures: parse, introspect, and check host-side bytes. -/
def hostGates (fixtures : System.FilePath) : IO Unit := do
  let tinyPath := fixtures / "tiny.safetensors"

  let hdr ← readHeader tinyPath
  checkEq "tiny.safetensors: header tensor count (__metadata__ skipped)" hdr.size 3
  let get (n : String) := hdr.findSome? fun t => if t.name == n then some t else none
  match get "w_qkv_f32" with
  | some t =>
    check "w_qkv_f32 schema"
      (t.dtype == .f32 && t.shape == #[8, 16] && t.byteSize == 512 && t.span == 512 &&
        t.dataBegin == 0)
  | none => fail "w_qkv_f32 missing from tiny.safetensors"
  match get "w_gate_bf16" with
  | some t =>
    check "w_gate_bf16 schema"
      (t.dtype == .bf16 && t.shape == #[4, 32] && t.byteSize == 256 && t.dataBegin == 512)
  | none => fail "w_gate_bf16 missing from tiny.safetensors"
  match get "w_up_f16" with
  | some t =>
    check "w_up_f16 schema"
      (t.dtype == .f16 && t.shape == #[16] && t.byteSize == 32 && t.dataBegin == 768)
  | none => fail "w_up_f16 missing from tiny.safetensors"

  let fileBytes ← IO.FS.readBinFile tinyPath
  match parseHeader fileBytes with
  | .error e => fail s!"parseHeader rejected tiny.safetensors: {e}"
  | .ok ts =>
    checkEq "parseHeader (whole-file) agrees with readHeader" (ts.map (·.name)) (hdr.map (·.name))

  let tiny ← introspect tinyPath
  check "introspect single file" (!tiny.isDirectory && tiny.tensors.size == 3)

  for (name, seed) in [("w_qkv_f32", 1), ("w_gate_bf16", 2), ("w_up_f16", 3)] do
    let t := (tiny.find? name).get!
    let bytes ← readTensorBytes (tiny.tensorPath t) t
    checkBytes s!"host bytes of '{name}'" bytes (pattern t.byteSize seed)

  let sharded ← introspect (fixtures / "sharded")
  check "sharded directory introspection" (sharded.isDirectory && sharded.tensors.size == 3)
  let alpha := (sharded.find? "tensor.alpha").get!
  let beta := (sharded.find? "tensor.beta").get!
  let gamma := (sharded.find? "tensor.gamma").get!
  checkEq "tensor.alpha resolves to shard 1" alpha.sourceFile "model-00001-of-00002.safetensors"
  checkEq "tensor.beta resolves to shard 2" beta.sourceFile "model-00002-of-00002.safetensors"
  checkEq "tensor.gamma resolves to shard 2" gamma.sourceFile "model-00002-of-00002.safetensors"
  for (t, seed) in [(alpha, 4), (beta, 5), (gamma, 6)] do
    let bytes ← readTensorBytes (sharded.tensorPath t) t
    checkBytes s!"sharded host bytes of '{t.name}'" bytes (pattern t.byteSize seed)

  -- Persistent kernels retain the published shard payloads and refer to tensors by a stable
  -- shard index plus a byte offset relative to the uploaded payload.
  let payloadManifest ← sharded.payloadManifest
  checkEq "payload manifest has two stable shards" payloadManifest.shards.size 2
  checkEq "payload shard order is lexical/index order"
    (payloadManifest.shards.map (·.filename))
    #["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
  let alphaRef := (payloadManifest.findTensor? "tensor.alpha").get!
  let betaRef := (payloadManifest.findTensor? "tensor.beta").get!
  let gammaRef := (payloadManifest.findTensor? "tensor.gamma").get!
  check "payload tensor references preserve shard and relative offset"
    (alphaRef.shard == 0 && alphaRef.byteOffset == alpha.dataBegin.toUInt64 &&
      betaRef.shard == 1 && betaRef.byteOffset == beta.dataBegin.toUInt64 &&
      gammaRef.shard == 1 && gammaRef.byteOffset == gamma.dataBegin.toUInt64)
  checkBytes "payload shard 1 host bytes" (← payloadManifest.readShard 0)
    (pattern alpha.byteSize 4)
  checkBytes "payload shard 2 host bytes" (← payloadManifest.readShard 1)
    (pattern beta.byteSize 5 ++ pattern gamma.byteSize 6)

  -- HF-name registry over the tiny configuration (4 layers: lin, lin, lin, full).
  let tinyCfg : RegistryConfig := { numLayers := 4 }
  checkEq "tiny registry: expected tensor count" tinyCfg.expected.size 56
  checkEq "hfName: deltanet A_log carries no .weight suffix"
    (WeightFamily.deltanetALog 2).hfName "model.language_model.layers.2.linear_attn.A_log"
  checkEq "hfName: deltanet dt_bias carries no .weight suffix"
    (WeightFamily.deltanetDtBias 0).hfName "model.language_model.layers.0.linear_attn.dt_bias"
  checkEq "hfName: full-attention o_proj"
    (WeightFamily.attnO 3).hfName "model.language_model.layers.3.self_attn.o_proj.weight"
  checkEq "hfName: final norm"
    WeightFamily.finalNorm.hfName "model.language_model.norm.weight"
  let roundtripped := tinyCfg.expected.filter fun f => WeightFamily.ofHFName? f.hfName == some f
  checkEq "registry roundtrip over all expected tiny names" roundtripped.size tinyCfg.expected.size

  -- Vendored trimmed copy of the real Qwen3.6-27B index: text keys must classify,
  -- roundtrip, and be expected by the 64-layer registry; vision/MTP keys are out of scope.
  let fullCfg : RegistryConfig := { numLayers := 64 }
  let weightMap ← parseIndexWeightMap (fixtures / "qwen36_27b_index_trimmed.json")
  checkEq "trimmed real 27B index: entry count" weightMap.size 32
  let mut problems : Array String := #[]
  for (name, _) in weightMap do
    if name.startsWith "model.visual." || name.startsWith "mtp." then
      unless (WeightFamily.ofHFName? name).isNone do
        problems := problems.push s!"'{name}' should be out of scope"
    else
      match WeightFamily.ofHFName? name with
      | none => problems := problems.push s!"'{name}' does not classify"
      | some family =>
        unless family.hfName == name do
          problems := problems.push s!"'{name}' roundtrip produced '{family.hfName}'"
        unless fullCfg.expectedNames.contains name do
          problems := problems.push s!"'{name}' is not expected by the 64-layer registry"
  check "trimmed real 27B index: every text key classifies as expected" problems.isEmpty

  -- checkComplete: accept the complete tiny checkpoint, reject missing/unexpected tensors.
  let complete ← introspect (fixtures / "complete")
  let report := checkComplete tinyCfg complete
  check "complete tiny checkpoint accepted" report.isClean
  checkEq "complete tiny checkpoint: nothing missing" report.missing #[]
  checkEq "complete tiny checkpoint: nothing unexpected" report.unexpected #[]

  let dropped := "model.language_model.layers.2.linear_attn.A_log"
  let missingSchema := { complete with tensors := complete.tensors.filter (·.name != dropped) }
  let missingReport := checkComplete tinyCfg missingSchema
  check "missing tensor reported" (!missingReport.isClean && missingReport.missing == #[dropped])

  let bogusAttn : TensorSchema := {
    name := "model.language_model.layers.0.self_attn.q_proj.weight"
    dtype := .bf16, shape := #[512, 256], dataOffset := 0, dataBegin := 0, dataEnd := 0 }
  let visual : TensorSchema := {
    name := "model.visual.blocks.0.attn.qkv.weight"
    dtype := .bf16, shape := #[1], dataOffset := 0, dataBegin := 0, dataEnd := 0 }
  let extraSchema := { complete with tensors := complete.tensors.push bogusAttn |>.push visual }
  let extraReport := checkComplete tinyCfg extraSchema
  check "attention tensor on a deltanet layer reported unexpected"
    (!extraReport.isClean && extraReport.unexpected == #[bogusAttn.name])
  checkEq "vision tensor reported out of scope, not unexpected"
    extraReport.outOfScope #[visual.name]

/-- Device roundtrips: upload through `Cuda.Buffer` and read back byte-exact. -/
def deviceGates (fixtures : System.FilePath) : IO Unit := do
  if (← Cuda.Device.count) == 0 then
    return
  let stream ← Cuda.Stream.create

  let tiny ← introspect (fixtures / "tiny.safetensors")
  for (name, seed) in [("w_qkv_f32", 1), ("w_gate_bf16", 2), ("w_up_f16", 3)] do
    let (t, buf) ← loadTensor tiny name stream
    checkEq s!"'{name}' device buffer byte size" (← buf.byteSize) t.byteSize.toUSize
    let readback ← Cuda.Buffer.copyTo buf stream
    stream.synchronize
    checkBytes s!"'{name}' device roundtrip is byte-exact" readback (pattern t.byteSize seed)

  let sharded ← introspect (fixtures / "sharded")
  let alpha := (sharded.find? "tensor.alpha").get!
  let beta := (sharded.find? "tensor.beta").get!
  let (gamma, gammaBuf) ← loadTensor sharded "tensor.gamma" stream
  checkEq "device: 'tensor.gamma' resolved through the index to shard 2"
    gamma.sourceFile "model-00002-of-00002.safetensors"
  let gammaReadback ← Cuda.Buffer.copyTo gammaBuf stream
  stream.synchronize
  checkBytes "'tensor.gamma' (shard 2) device roundtrip is byte-exact"
    gammaReadback (pattern gamma.byteSize 6)

  let all ← loadTensors tiny stream
  checkEq "loadTensors loads every tensor" all.size 3
  for (t, buf) in all do
    let seed := if t.name == "w_qkv_f32" then 1 else if t.name == "w_gate_bf16" then 2 else 3
    let readback ← Cuda.Buffer.copyTo buf stream
    stream.synchronize
    checkBytes s!"loadTensors '{t.name}' roundtrip is byte-exact" readback (pattern t.byteSize seed)

  let payloadManifest ← sharded.payloadManifest
  -- Seven bytes is deliberately smaller than both fixture payloads. This forces the production
  -- loader through multiple ranged uploads without requiring a model-scale test artifact.
  let payloads ← payloadManifest.loadShards stream 7
  checkEq "bounded whole-payload loader preserves shard count" payloads.size 2
  let some firstPayload := payloads[0]?
    | fail "whole-payload shard 1 missing"
  let some secondPayload := payloads[1]?
    | fail "whole-payload shard 2 missing"
  let firstReadback ← firstPayload.2.copyTo stream
  let secondReadback ← secondPayload.2.copyTo stream
  stream.synchronize
  checkBytes "bounded whole-payload shard 1 device roundtrip" firstReadback
    (pattern alpha.byteSize 4)
  checkBytes "bounded whole-payload shard 2 device roundtrip" secondReadback
    (pattern beta.byteSize 5 ++ pattern gamma.byteSize 6)

  -- The production model path selects only text tensors and independently aligns their device
  -- addresses. A 16-byte fixture alignment forces two bytes of padding between beta and gamma;
  -- seven-byte chunks also retain the bounded streaming path above.
  let packed ← payloadManifest.loadPacked
    #["tensor.alpha", "tensor.beta", "tensor.gamma"] stream 16 7
  checkEq "aligned packed payload preserves the two nonempty source shards" packed.buffers.size 2
  checkEq "aligned packed payload retains every selected tensor" packed.tensors.size 3
  checkEq "aligned packed payload records its alignment" packed.alignment 16
  let packedAlpha := (packed.findTensor? "tensor.alpha").get!
  let packedBeta := (packed.findTensor? "tensor.beta").get!
  let packedGamma := (packed.findTensor? "tensor.gamma").get!
  check "aligned packed tensor references use dense shards and aligned offsets"
    (packedAlpha.shard == 0 && packedAlpha.byteOffset == 0 &&
      packedBeta.shard == 1 && packedBeta.byteOffset == 0 &&
      packedGamma.shard == 1 && packedGamma.byteOffset == 32 &&
      packed.tensors.all (fun tensor => tensor.byteOffset.toNat % packed.alignment == 0))
  let some packedFirstBuffer := packed.buffers[0]?
    | fail "aligned packed shard 1 buffer missing"
  let some packedSecondBuffer := packed.buffers[1]?
    | fail "aligned packed shard 2 buffer missing"
  let packedFirst ← packedFirstBuffer.copyTo stream
  let packedSecond ← packedSecondBuffer.copyTo stream
  checkEq "aligned packed shard 1 byte size" packedFirst.size alpha.byteSize
  checkEq "aligned packed shard 2 byte size" packedSecond.size (32 + gamma.byteSize)
  checkBytes "aligned packed alpha bytes" packedFirst (pattern alpha.byteSize 4)
  checkBytes "aligned packed beta bytes" (packedSecond.extract 0 beta.byteSize)
    (pattern beta.byteSize 5)
  checkBytes "aligned packed gamma bytes" (packedSecond.extract 32 (32 + gamma.byteSize))
    (pattern gamma.byteSize 6)

private def printProcessMemory (label : String) : IO Unit := do
  let statusPath : System.FilePath := "/proc/self/status"
  unless ← statusPath.pathExists do
    IO.println s!"memory after {label}: unavailable on this platform"
    return
  let status ← IO.FS.readFile statusPath
  let lines := status.splitOn "\n" |>.filter fun line =>
    line.startsWith "VmRSS:" || line.startsWith "VmHWM:"
  IO.println s!"memory after {label}: {String.intercalate ", " lines}"

private def streamOneCheckpointShard (manifest : PayloadManifest) (index : Nat)
    (stream : Cuda.Stream) : IO Unit := do
  let (shard, buffer) ← manifest.loadShard index stream
  checkEq s!"real payload '{shard.filename}' device byte size"
    (← buffer.byteSize) shard.byteSize.toUSize

/-- Model-scale memory gate which deliberately never owns more than one device shard. Fixture
gates above establish ranged-upload byte exactness; this gate establishes bounded host residency
against the actual checkpoint sizes without competing for enough memory to run the model. -/
def streamCheckpointGates (directory : System.FilePath) : IO Unit := do
  if (← Cuda.Device.count) == 0 then
    fail "real checkpoint bounded-stream gate requires a CUDA device"
  let schema ← introspect directory
  let manifest ← schema.payloadManifest
  let stream ← Cuda.Stream.create
  printProcessMemory "checkpoint introspection"
  for index in [:manifest.shards.size] do
    streamOneCheckpointShard manifest index stream
    printProcessMemory s!"releasing shard {index + 1}/{manifest.shards.size}"

private def fixtureDirectory : IO System.FilePath := do
  let some directory ← IO.getEnv "QWEN36_WEIGHTS_FIXTURES"
    | LeanTest.fail "QWEN36_WEIGHTS_FIXTURES is required"
  return ⟨directory⟩

@[test]
def hostSafetensorsRegistryAndBytes : IO Unit := do
  hostGates (← fixtureDirectory)

@[test]
def deviceSafetensorsRoundtrips : IO Unit := do
  if (← IO.getEnv "QWEN36_WEIGHTS_RUN_DEVICE") == some "1" then
    deviceGates (← fixtureDirectory)

@[test_ignore]
def realCheckpointBoundedShardStreaming : IO Unit := do
  let some directory ← IO.getEnv "QWEN36_STREAM_CHECKPOINT_DIR"
    | LeanTest.fail "QWEN36_STREAM_CHECKPOINT_DIR is required"
  streamCheckpointGates ⟨directory⟩
