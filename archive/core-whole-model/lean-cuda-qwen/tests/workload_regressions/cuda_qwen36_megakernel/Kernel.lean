/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-!
# Qwen3.8 persistent multi-token inference gate

Runs the bounded multi-token cache/RoPE gate on every supported GPU. When `QWEN_MODEL_DIR` is
set, it additionally returns logits and greedy tokens from the published Qwen3.8-27B checkpoint.
-/

namespace Qwen36MegakernelGate

private def vocabulary : Nat := 248320

private def parseIds (name raw : String) : IO (Array UInt32) := do
  let mut values := #[]
  for part in raw.splitOn "," do
    let some value := part.toNat?
      | LeanTest.fail s!"{name} must be a comma-separated list of token IDs"
    LeanTest.assertTrue (value < vocabulary)
      s!"{name} token ID {value} is outside the Qwen3.8 vocabulary"
    values := values.push value.toUInt32
  LeanTest.assertTrue (!values.isEmpty) s!"{name} must contain at least one token ID"
  return values

private def readUInt32 (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def finiteFloat32 (value : Float32) : Bool :=
  value.toBits &&& 0x7f800000 != 0x7f800000

private def readFloat32At (bytes : ByteArray) (index : Nat) : Float32 :=
  Float32.ofBits (readUInt32 bytes index)

private def readBFloat16At (bytes : ByteArray) (index : Nat) : Float32 :=
  let offset := index * 2
  Float32.ofBits ((bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8)) <<< 16)

private def maxAbsDiff (read : ByteArray → Nat → Float32) (left right : ByteArray)
    (leftStart rightStart elements : Nat) : Float32 := Id.run do
  let mut worst : Float32 := 0
  for index in [:elements] do
    let delta := read left (leftStart + index) - read right (rightStart + index)
    let magnitude := Float32.ofBits (delta.toBits &&& 0x7fffffff)
    if magnitude > worst then
      worst := magnitude
  return worst

private def bytesDifferInRange (left right : ByteArray) (start count : Nat) : Bool :=
  (List.range count).any fun index => left[start + index]? != right[start + index]?

@[test]
def persistentLayoutContract : IO Unit :=
  Cuda.Qwen36.Megakernel.checkPersistentLayout

@[test]
def benchmarkRouteDescribesCompiledKernel : IO Unit := do
  let route := Cuda.Qwen36.Megakernel.benchmarkRoute
  LeanTest.assertTrue (route.contains "grid") "benchmark route omitted the launch grid"
  LeanTest.assertTrue (route.contains "bf16x16") "benchmark route omitted the projection branch width"
  LeanTest.assertTrue (route.contains "strided256")
    "benchmark route omitted the coalesced block-stride access order"
  LeanTest.assertTrue (route.contains "statewarp_resident_colmajor")
    "benchmark route omitted the warp-resident recurrent-state schedule"
  LeanTest.assertTrue (route.contains "stategrid6x8warps")
    "benchmark route omitted full-grid recurrent-state parallelism"
  LeanTest.assertTrue (route.contains "attn_online_softmax")
    "benchmark route omitted constant-shared-memory online attention"
  LeanTest.assertTrue (route.contains "rms_block1_owner")
    "benchmark route omitted the single-owner RMSNorm reduction schedule"
  LeanTest.assertTrue (route.contains "max_block1_owner")
    "benchmark route omitted the single-owner final-maximum reduction schedule"
  LeanTest.assertTrue (route.contains "sample_blockrows_bf16refine")
    "benchmark route omitted block-owned greedy sampling"
  LeanTest.assertTrue (route.contains "rowpair2")
    "benchmark route omitted paired output-row scheduling"
  LeanTest.assertTrue (route.contains "fused_residual")
    "benchmark route omitted fused residual projection"
  LeanTest.assertTrue (route.contains "integrated_prefill_prefix_cache")
    "benchmark route omitted integrated prefill and prefix caching"
  LeanTest.assertTrue (route.contains "prefix_cache_tile2")
    "benchmark route omitted two-row prefill weight reuse"
  LeanTest.assertTrue (route.contains "aligned128")
    "benchmark route omitted aligned text-weight placement"

@[test]
def checkpointLoRAMixedDescriptorContract : IO Unit := do
  let descriptor : Cuda.Qwen36.CheckpointLoRA.Descriptor := {
    batchSize := 3
    sequenceLength := 2
    adapterCount := 2
    adapterIds := #[0, 1, Cuda.Qwen36.LoRA.noAdapter]
    updateMask := #[1, 0]
  }
  LeanTest.assertTrue descriptor.check.isOk "valid mixed checkpoint LoRA descriptor was rejected"
  LeanTest.assertTrue (!({ descriptor with adapterIds := #[0, 2, 1] }).check.isOk)
    "out-of-range checkpoint adapter ID was accepted"
  LeanTest.assertTrue (!({ descriptor with updateMask := #[1, 2] }).check.isOk)
    "non-binary checkpoint update mask was accepted"

@[test]
def tokenEventCodecRoundTrips : IO Unit := do
  let event : Cuda.Qwen36.Megakernel.TokenEvent := {
    position := 7
    token := 1234
  }
  match Cuda.Qwen36.Megakernel.TokenEvent.ofBytes event.toBytes with
  | .ok decoded =>
      LeanTest.assertEqual decoded.position event.position
      LeanTest.assertEqual decoded.token event.token
  | .error message => LeanTest.fail message

@[test]
def generationStopsCleanlyAtConfiguredToken : IO Unit := do
  let config : Cuda.Qwen36.Megakernel.GenerationConfig := {
    maxNewTokens := 3
    stopTokens := #[248046]
  }
  LeanTest.assertTrue (config.shouldStop 248046) "configured end token did not stop generation"
  LeanTest.assertTrue (!config.shouldStop 10) "ordinary token stopped generation"

  let samples : Array UInt32 := #[10, 248046, 99]
  let nextSample ← IO.mkRef 0
  let prefillEnqueued ← IO.mkRef (#[] : Array UInt32)
  let trainingEnqueued ← IO.mkRef (#[] : Array UInt32)
  let greedyEnqueued ← IO.mkRef (#[] : Array UInt32)
  let streamed ← IO.mkRef (#[] : Array Cuda.Qwen36.Megakernel.TokenEvent)
  let finished ← IO.mkRef false
  let finishedEarly ← IO.mkRef false
  let aborted ← IO.mkRef false
  let checkpoint : Cuda.Qwen36.Megakernel.Checkpoint := {
    start := fun positions =>
      return {
        expectedPositions := positions
        prefill := fun token => prefillEnqueued.modify (·.push token)
        prefillBatch := fun _ => pure ()
        enqueue := fun token _ => trainingEnqueued.modify (·.push token)
        enqueueGreedy := fun token => greedyEnqueued.modify (·.push token)
        nextToken := do
          let index ← nextSample.get
          let some token := samples[index]?
            | LeanTest.fail "fake generation stream ran out of samples"
          nextSample.set (index + 1)
          return { position := index.toUInt32, token }
        finish := finished.set true
        finishEarly := finishedEarly.set true
        collect := LeanTest.fail "generation must not collect full logits"
        abort := aborted.set true
      }
    startInference := fun positions =>
      return {
        expectedPositions := positions
        prefill := fun token => prefillEnqueued.modify (·.push token)
        prefillBatch := fun _ => pure ()
        enqueue := fun token _ => trainingEnqueued.modify (·.push token)
        enqueueGreedy := fun token => greedyEnqueued.modify (·.push token)
        nextToken := do
          let index ← nextSample.get
          let some token := samples[index]?
            | LeanTest.fail "fake generation stream ran out of samples"
          nextSample.set (index + 1)
          return { position := index.toUInt32, token }
        finish := finished.set true
        finishEarly := finishedEarly.set true
        collect := LeanTest.fail "generation must not collect full logits"
        abort := aborted.set true
      }
  }
  let generated ← checkpoint.generateStreamingWith #[7] config fun event =>
    streamed.modify (·.push event)
  LeanTest.assertEqual generated #[10, 248046]
  let events ← streamed.get
  LeanTest.assertEqual (events.map fun event => event.token) generated
  LeanTest.assertEqual (← trainingEnqueued.get) #[]
  LeanTest.assertEqual (← prefillEnqueued.get) #[]
  LeanTest.assertEqual (← greedyEnqueued.get) #[7, 10]
  LeanTest.assertTrue (← finishedEarly.get) "stop token did not finish the stream cleanly"
  LeanTest.assertTrue (!(← finished.get)) "stop token completed a full stream"
  LeanTest.assertTrue (!(← aborted.get)) "stop token aborted instead of finishing cleanly"

  nextSample.set 0
  trainingEnqueued.set #[]
  greedyEnqueued.set #[]
  finished.set false
  finishedEarly.set false
  let reference ← checkpoint.generateStreamingReference #[7] 2 fun _ => pure ()
  LeanTest.assertEqual reference #[10, 248046]
  LeanTest.assertEqual (← trainingEnqueued.get) #[7, 10]
  LeanTest.assertEqual (← greedyEnqueued.get) #[]
  LeanTest.assertTrue (← finished.get) "BF16 reference did not complete its stream"
  LeanTest.assertTrue (!(← finishedEarly.get)) "BF16 reference stopped early"

  nextSample.set 0
  prefillEnqueued.set #[]
  trainingEnqueued.set #[]
  greedyEnqueued.set #[]
  finished.set false
  finishedEarly.set false
  let prefetched ← checkpoint.generateStreamingWith #[7, 8, 9] {
    maxNewTokens := 1
  } fun _ => pure ()
  LeanTest.assertEqual prefetched #[10]
  LeanTest.assertEqual (← prefillEnqueued.get) #[7, 8]
  LeanTest.assertEqual (← greedyEnqueued.get) #[9]
  LeanTest.assertEqual (← trainingEnqueued.get) #[]
  LeanTest.assertTrue (← finished.get) "integrated prefill did not complete its stream"

/-- Run the published checkpoint only when configured; all success conditions are assertions and
the optional logits file is an artifact for the separate Transformers numerical comparison. -/
@[test_ignore]
def configuredRealCheckpointReturnsMultiTokenOutputs : IO Unit := do
  let some modelDirectory ← IO.getEnv "QWEN_MODEL_DIR"
    | LeanTest.fail "QWEN_MODEL_DIR is required for the real-checkpoint test"
  let tokens ← parseIds "QWEN38_TOKEN_IDS"
    ((← IO.getEnv "QWEN38_TOKEN_IDS").getD "1234,5678")
  let targets ← parseIds "QWEN38_TARGET_IDS"
    ((← IO.getEnv "QWEN38_TARGET_IDS").getD "1234,5678")
  LeanTest.assertTrue (tokens.size == targets.size)
    "QWEN38_TOKEN_IDS and QWEN38_TARGET_IDS must have equal length"
  let checkpoint ← Cuda.Qwen36.Megakernel.loadCheckpoint modelDirectory
  let streamed ← IO.mkRef (#[] : Array Cuda.Qwen36.Megakernel.TokenEvent)
  let result ← checkpoint.runStreaming tokens targets fun event =>
      streamed.modify (·.push event)
  let logits := result.logits
  let loss := result.loss
  let sampled := result.sampled
  LeanTest.assertTrue (logits.size == tokens.size * vocabulary * 4)
    s!"real checkpoint returned {logits.size} logit bytes"
  LeanTest.assertTrue (loss.size == tokens.size * 4)
    s!"real checkpoint returned {loss.size} loss bytes"
  LeanTest.assertTrue (sampled.size == tokens.size * 4)
    s!"real checkpoint returned {sampled.size} sampled-token bytes"
  let events ← streamed.get
  LeanTest.assertTrue (events.size == tokens.size)
    s!"real checkpoint streamed {events.size} tokens for {tokens.size} positions"
  for position in [:tokens.size] do
    let sampledToken := readUInt32 sampled position
    let some event := events[position]?
      | LeanTest.fail s!"missing streamed token at position {position}"
    LeanTest.assertEqual event.position position.toUInt32
    LeanTest.assertEqual event.token sampledToken
    LeanTest.assertTrue (sampledToken < vocabulary.toUInt32)
      s!"sampled token at position {position} is outside the vocabulary"
  if let some expectedRaw ← IO.getEnv "QWEN38_EXPECTED_SAMPLED" then
    let expected ← parseIds "QWEN38_EXPECTED_SAMPLED" expectedRaw
    LeanTest.assertTrue (expected.size == tokens.size)
      "QWEN38_EXPECTED_SAMPLED must contain one token per position"
    for position in [:tokens.size] do
      LeanTest.assertTrue (readUInt32 sampled position == expected[position]!)
        s!"sampled token mismatch at position {position}"
  if let some outputPath ← IO.getEnv "QWEN38_LOGITS_OUTPUT" then
    IO.FS.writeBinFile outputPath logits

  let generatedEvents ← IO.mkRef (#[] : Array Cuda.Qwen36.Megakernel.TokenEvent)
  let generated ← checkpoint.generateStreaming #[tokens[0]!] 2 fun event =>
    generatedEvents.modify (·.push event)
  let events ← generatedEvents.get
  LeanTest.assertEqual generated.size 2
  LeanTest.assertEqual events.size generated.size
  for position in [:generated.size] do
    let some event := events[position]?
      | LeanTest.fail s!"missing autoregressive streamed token at position {position}"
    LeanTest.assertEqual event.position position.toUInt32
    LeanTest.assertEqual event.token generated[position]!
    LeanTest.assertTrue (event.token < vocabulary.toUInt32)
      s!"autoregressive token at position {position} is outside the vocabulary"
  LeanTest.assertEqual generated[0]! (readUInt32 sampled 0)
  let bf16Generated ← checkpoint.generateStreamingReference #[tokens[0]!] 2 fun _ => pure ()
  LeanTest.assertEqual generated bf16Generated

  let stoppedEvents ← IO.mkRef (#[] : Array Cuda.Qwen36.Megakernel.TokenEvent)
  let stopped ← checkpoint.generateStreamingWith #[tokens[0]!] {
    maxNewTokens := 2
    stopTokens := #[readUInt32 sampled 0]
  } fun event => stoppedEvents.modify (·.push event)
  LeanTest.assertEqual stopped #[readUInt32 sampled 0]
  LeanTest.assertEqual (← stoppedEvents.get).size 1

/-- Prove a genuine optimizer step against the published BF16 checkpoint, not a toy model. -/
@[test_ignore]
def configuredRealCheckpointLoRAUpdatesAndLowersLoss : IO Unit := do
  let some modelDirectory ← IO.getEnv "QWEN_MODEL_DIR"
    | LeanTest.fail "QWEN_MODEL_DIR is required for the real LoRA test"
  let tokens : Array (Array UInt32) := #[#[1234], #[4321]]
  let targets : Array (Array UInt32) := #[#[5678], #[8765]]
  let trainer ← Cuda.Qwen36.CheckpointLoRA.load modelDirectory {
    batchSize := 2
    sequenceLength := 1
    adapterCount := 2
    adapterIds := #[0, 1]
    updateMask := #[1, 1]
  } {
    rank := 16
    alpha := 16
    learningRate := 0.001
    weightDecay := 0
  }
  let mut baseBefore : Array Cuda.Qwen36.Megakernel.InferenceResult := #[]
  for batch in [:tokens.size] do
    baseBefore := baseBefore.push
      (← trainer.checkpoint.runStreaming tokens[batch]! targets[batch]! fun _ => pure ())
  let initial ← trainer.snapshot
  let lossBefore ← trainer.evaluate tokens targets
  let firstStepLoss ← trainer.step tokens targets
  let afterFirst ← trainer.snapshot
  let secondStepLoss ← trainer.step tokens targets
  let afterSecond ← trainer.snapshot
  let lossAfter ← trainer.evaluate tokens targets
  LeanTest.assertTrue (finiteFloat32 lossBefore && finiteFloat32 firstStepLoss &&
      finiteFloat32 secondStepLoss && finiteFloat32 lossAfter)
    (s!"real LoRA produced a non-finite loss: {lossBefore}, {firstStepLoss}, " ++
      s!"{secondStepLoss}, {lossAfter}")
  LeanTest.assertTrue (afterFirst.adapter.adapterB != initial.adapter.adapterB)
    "real mixed LoRA step did not update adapter B"
  let aStride := trainer.adapter.shape.rank.toNat * trainer.adapter.shape.inputFeatures.toNat * 4
  let bStride := trainer.adapter.shape.outputFeatures.toNat * trainer.adapter.shape.rank.toNat * 4
  for adapter in [:2] do
    LeanTest.assertTrue (bytesDifferInRange afterFirst.adapter.adapterB initial.adapter.adapterB
        (adapter * bStride) bStride)
      s!"real mixed LoRA step did not update adapter {adapter} B"
  LeanTest.assertEqual afterFirst.adapter.adapterA initial.adapter.adapterA
  for adapter in [:2] do
    LeanTest.assertTrue (bytesDifferInRange afterSecond.adapter.adapterA initial.adapter.adapterA
        (adapter * aStride) aStride)
      s!"real mixed LoRA second step did not open adapter {adapter} A"
  LeanTest.assertTrue (lossAfter < lossBefore)
    s!"real LoRA did not lower CE: {lossBefore} -> {lossAfter}"
  for batch in [:tokens.size] do
    let baseAfter ← trainer.checkpoint.runStreaming tokens[batch]! targets[batch]! fun _ => pure ()
    let some before := baseBefore[batch]?
      | LeanTest.fail s!"missing frozen-base result for batch row {batch}"
    LeanTest.assertEqual baseAfter.logits before.logits
    LeanTest.assertEqual baseAfter.loss before.loss
    LeanTest.assertEqual baseAfter.sampled before.sampled

/-- Prove the paired retained-training kernel reproduces the sequential single-sequence
forward for both rows before any adapter update relies on it. -/
@[test_ignore]
def configuredRealCheckpointPairedForwardMatchesSequential : IO Unit := do
  let some modelDirectory ← IO.getEnv "QWEN_MODEL_DIR"
    | LeanTest.fail "QWEN_MODEL_DIR is required for the real paired-forward test"
  let trainable ← Cuda.Qwen36.Megakernel.loadTrainableCheckpoint modelDirectory
  let firstTokens : Array UInt32 := #[1234, 5678, 90]
  let firstTargets : Array UInt32 := #[5678, 90, 11]
  let secondTokens : Array UInt32 := #[4321, 8765, 321]
  let secondTargets : Array UInt32 := #[8765, 321, 22]
  let firstSingle ← trainable.forward firstTokens firstTargets
  let secondSingle ← trainable.forward secondTokens secondTargets
  let paired ← trainable.forwardBatch #[firstTokens, secondTokens] #[firstTargets, secondTargets]
  LeanTest.assertEqual paired.rows (2 * firstTokens.size).toUInt32
  let firstSingleHidden ← firstSingle.finalHidden.copyTo trainable.stream
  let secondSingleHidden ← secondSingle.finalHidden.copyTo trainable.stream
  let pairedHidden ← paired.finalHidden.copyTo trainable.stream
  let firstSingleLogits ← firstSingle.logits.copyTo trainable.stream
  let secondSingleLogits ← secondSingle.logits.copyTo trainable.stream
  let pairedLogits ← paired.logits.copyTo trainable.stream
  let hiddenElements := firstTokens.size * paired.inputFeatures.toNat
  let logitElements := firstTokens.size * paired.outputFeatures.toNat
  let firstHiddenDiff := maxAbsDiff readBFloat16At pairedHidden firstSingleHidden
    0 0 hiddenElements
  let secondHiddenDiff := maxAbsDiff readBFloat16At pairedHidden secondSingleHidden
    hiddenElements 0 hiddenElements
  let firstLogitDiff := maxAbsDiff readFloat32At pairedLogits firstSingleLogits
    0 0 logitElements
  let secondLogitDiff := maxAbsDiff readFloat32At pairedLogits secondSingleLogits
    logitElements 0 logitElements
  LeanTest.assertTrue (firstHiddenDiff < 0.02)
    s!"paired forward first-row hidden diverged from sequential: {firstHiddenDiff}"
  LeanTest.assertTrue (secondHiddenDiff < 0.02)
    s!"paired forward second-row hidden diverged from sequential: {secondHiddenDiff}"
  LeanTest.assertTrue (firstLogitDiff < 0.02)
    s!"paired forward first-row logits diverged from sequential: {firstLogitDiff}"
  LeanTest.assertTrue (secondLogitDiff < 0.02)
    s!"paired forward second-row logits diverged from sequential: {secondLogitDiff}"

end Qwen36MegakernelGate
