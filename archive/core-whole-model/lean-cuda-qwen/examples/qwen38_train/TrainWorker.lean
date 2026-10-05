/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import Lean.Data.Json
import LeanCudaQwen.Foundation
import LeanCudaQwen.Qwen36.CheckpointLoRA
import PreferenceDataset
import GRPODataset

/-!
# Qwen3.8 LoRA training worker

One-shot projection-wide training CLI over the real BF16 checkpoint. Reads packed little-endian
UInt32 token streams, runs AdamW LoRA steps on all 497 transformer and LM-head projections with
bounded device-side gradient diagnostics, reports validation cross-entropy, and saves or resumes
versioned adapter checkpoints. Every progress record is one NDJSON line on stdout.
-/

namespace Qwen38Train

open Lean
open Cuda.Qwen36.CheckpointLoRA

private inductive Objective where
  | sft
  | dpo
  | grpo
  deriving BEq

private def Objective.name : Objective → String
  | .sft => "sft"
  | .dpo => "dpo"
  | .grpo => "grpo"

private def parseObjective (raw : String) : Except String Objective :=
  match raw with
  | "sft" => pure .sft
  | "dpo" => pure .dpo
  | "grpo" => pure .grpo
  | _ => throw s!"objective must be sft, dpo, or grpo, got {raw}"

private structure CliConfig where
  modelDirectory : System.FilePath
  dataset : System.FilePath
  valDataset : Option System.FilePath := none
  checkpointOut : Option System.FilePath := none
  resume : Option System.FilePath := none
  steps : Nat := 100
  batchSize : Nat := 1
  sequenceLength : Nat := 128
  rank : Nat := 16
  alpha : Float32 := 16
  learningRate : Float32 := 0.001
  objective : Objective := .sft
  dpoBeta : Float32 := 0.1
  grpoGroupSize : Nat := 2
  grpoUpdates : Nat := 1
  grpoClipEpsilon : Float32 := 0.2
  grpoKLBeta : Float32 := 0.04
  grpoAdvantageEpsilon : Float32 := 1e-6
  seed : UInt64 := 0x3636
  logEvery : Nat := 1
  valEvery : Nat := 0
  valBatches : Nat := 4
  saveEvery : Nat := 0

private def usage : String :=
  "qwen38_train --model-dir PATH --dataset TOKENS.bin [--val-dataset TOKENS.bin] " ++
  "[--objective sft|dpo|grpo] [--steps N] [--batch-size N] [--sequence-length N] " ++
  "[--rank N] [--alpha F] [--learning-rate F] [--dpo-beta F] " ++
  "[--grpo-group-size N] [--grpo-updates N] [--grpo-clip-epsilon F] " ++
  "[--grpo-kl-beta F] [--grpo-advantage-epsilon F] [--seed N] " ++
  "[--log-every N] [--val-every N] [--val-batches N] [--checkpoint-out PATH] " ++
  "[--resume PATH] [--save-every N]"

private def parseNat (name raw : String) : Except String Nat := do
  let some value := raw.toNat?
    | throw s!"{name} must be an unsigned integer, got {raw}"
  return value

private def finiteFloat32 (value : Float32) : Bool :=
  value.toBits &&& 0x7f800000 != 0x7f800000

private def parseFloat32 (name raw : String) : Except String Float32 := do
  let json ← Lean.Json.parse raw
  let .num number := json
    | throw s!"{name} must be a number, got {raw}"
  let value := number.toFloat.toFloat32
  unless finiteFloat32 value do
    throw s!"{name} must be finite"
  return value

private def requirePositive (name : String) (value : Nat) : Except String Nat := do
  unless value > 0 do
    throw s!"{name} must be positive"
  return value

private partial def parseArgs (args : List String) (config : CliConfig) :
    Except String CliConfig := do
  match args with
  | [] => return config
  | flag :: value :: rest =>
    let config ← match flag with
      | "--model-dir" => pure { config with modelDirectory := value }
      | "--dataset" => pure { config with dataset := value }
      | "--val-dataset" => pure { config with valDataset := some value }
      | "--objective" => pure { config with objective := ← parseObjective value }
      | "--checkpoint-out" => pure { config with checkpointOut := some value }
      | "--resume" => pure { config with resume := some value }
      | "--steps" => pure { config with steps := ← parseNat "steps" value }
      | "--batch-size" => pure { config with batchSize := ← parseNat "batch-size" value }
      | "--sequence-length" =>
          pure { config with sequenceLength := ← parseNat "sequence-length" value }
      | "--rank" => pure { config with rank := ← parseNat "rank" value }
      | "--alpha" => pure { config with alpha := ← parseFloat32 "alpha" value }
      | "--learning-rate" =>
          pure { config with learningRate := ← parseFloat32 "learning-rate" value }
      | "--dpo-beta" => pure { config with dpoBeta := ← parseFloat32 "dpo-beta" value }
      | "--grpo-group-size" =>
          pure { config with grpoGroupSize := ← parseNat "grpo-group-size" value }
      | "--grpo-updates" =>
          pure { config with grpoUpdates := ← parseNat "grpo-updates" value }
      | "--grpo-clip-epsilon" =>
          pure { config with grpoClipEpsilon := ← parseFloat32 "grpo-clip-epsilon" value }
      | "--grpo-kl-beta" =>
          pure { config with grpoKLBeta := ← parseFloat32 "grpo-kl-beta" value }
      | "--grpo-advantage-epsilon" =>
          pure { config with grpoAdvantageEpsilon := ← parseFloat32 "grpo-advantage-epsilon" value }
      | "--seed" => pure { config with seed := (← parseNat "seed" value).toUInt64 }
      | "--log-every" => pure { config with logEvery := ← parseNat "log-every" value }
      | "--val-every" => pure { config with valEvery := ← parseNat "val-every" value }
      | "--val-batches" => pure { config with valBatches := ← parseNat "val-batches" value }
      | "--save-every" => pure { config with saveEvery := ← parseNat "save-every" value }
      | unknown => throw s!"unknown flag {unknown}"
    parseArgs rest config
  | [flag] => throw s!"flag {flag} requires a value"

private def CliConfig.check (config : CliConfig) : Except String CliConfig := do
  discard <| requirePositive "steps" config.steps
  discard <| requirePositive "batch-size" config.batchSize
  discard <| requirePositive "sequence-length" config.sequenceLength
  unless config.rank > 0 && config.rank % 16 == 0 do
    throw "rank must be a positive multiple of 16"
  unless config.alpha > 0 && finiteFloat32 config.alpha do
    throw "alpha must be positive and finite"
  unless config.learningRate > 0 && finiteFloat32 config.learningRate do
    throw "learning-rate must be positive and finite"
  if config.sequenceLength > Cuda.Qwen36.Megakernel.maxTrainingSequenceTokens then
    throw s!"sequence-length must be at most {Cuda.Qwen36.Megakernel.maxTrainingSequenceTokens}"
  if config.objective == .dpo then
    let sequences := config.batchSize * 2
    if sequences * config.sequenceLength > 128 ||
        sequences * (config.sequenceLength + 1) > 132 then
      throw <|
        "DPO pair-batch geometry exceeds 128 token rows or 132 recurrent-state rows; " ++
        "use --batch-size 1 --sequence-length 64 or a smaller shape"
    unless config.dpoBeta > 0 &&
        (config.dpoBeta.toBits &&& 0x7f800000) != 0x7f800000 do
      throw "dpo-beta must be positive and finite"
  if config.objective == .grpo then
    unless config.grpoGroupSize >= 2 do
      throw "grpo-group-size must be at least two"
    discard <| requirePositive "grpo-updates" config.grpoUpdates
    let sequences := config.batchSize * config.grpoGroupSize
    if sequences * config.sequenceLength > 128 ||
        sequences * (config.sequenceLength + 1) > 132 then
      throw <|
        "GRPO grouped-batch geometry exceeds 128 token rows or 132 recurrent-state rows; " ++
        "reduce --batch-size, --grpo-group-size, or --sequence-length"
    unless config.grpoClipEpsilon > 0 && config.grpoClipEpsilon < 1 &&
        (config.grpoClipEpsilon.toBits &&& 0x7f800000) != 0x7f800000 do
      throw "grpo-clip-epsilon must be finite and lie in (0, 1)"
    unless config.grpoKLBeta >= 0 &&
        (config.grpoKLBeta.toBits &&& 0x7f800000) != 0x7f800000 do
      throw "grpo-kl-beta must be nonnegative and finite"
    unless config.grpoAdvantageEpsilon > 0 &&
        (config.grpoAdvantageEpsilon.toBits &&& 0x7f800000) != 0x7f800000 do
      throw "grpo-advantage-epsilon must be positive and finite"
  discard <| requirePositive "log-every" config.logEvery
  discard <| requirePositive "val-batches" config.valBatches
  if config.saveEvery > 0 && config.checkpointOut.isNone then
    throw "--save-every requires --checkpoint-out"
  return config

private def vocabulary : Nat := 248320

/-- A wrapped cursor over a packed little-endian UInt32 token stream. -/
private structure Batcher where
  bytes : ByteArray
  tokenCount : Nat
  sequenceLength : Nat
  cursor : Nat := 0
  epoch : Nat := 0

private def readToken (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def loadBatcher (path : System.FilePath) (sequenceLength : Nat) : IO Batcher := do
  let bytes ← IO.FS.readBinFile path
  unless bytes.size % 4 == 0 do
    throw <| IO.userError s!"{path}: token stream size {bytes.size} is not a multiple of 4"
  let tokenCount := bytes.size / 4
  unless tokenCount > sequenceLength do
    throw <| IO.userError
      s!"{path}: need more than {sequenceLength} tokens, found {tokenCount}"
  for index in [:tokenCount] do
    unless readToken bytes index < vocabulary.toUInt32 do
      throw <| IO.userError s!"{path}: token {index} is outside the Qwen3.8 vocabulary"
  return { bytes, tokenCount, sequenceLength }

/-- Draw one contiguous window; targets are the same stream shifted by one token. -/
private def Batcher.next (batcher : Batcher) : Batcher × Array UInt32 × Array UInt32 := Id.run do
  let mut batcher := batcher
  if batcher.cursor + batcher.sequenceLength + 1 > batcher.tokenCount then
    batcher := { batcher with cursor := 0, epoch := batcher.epoch + 1 }
  let mut tokens := Array.mkEmpty batcher.sequenceLength
  let mut targets := Array.mkEmpty batcher.sequenceLength
  for offset in [:batcher.sequenceLength] do
    tokens := tokens.push (readToken batcher.bytes (batcher.cursor + offset))
    targets := targets.push (readToken batcher.bytes (batcher.cursor + offset + 1))
  return ({ batcher with cursor := batcher.cursor + batcher.sequenceLength }, tokens, targets)

private def jsonFloat32 (value : Float32) : Lean.Json :=
  toJson value.toFloat

private def emit (fields : List (String × Lean.Json)) : IO Unit :=
  IO.println (Lean.Json.mkObj fields).compress

private def gradientStatsJson (stats : GradientStats) : Lean.Json :=
  Lean.Json.mkObj [
    ("elements", toJson stats.elements),
    ("finite", toJson stats.finiteElements),
    ("nonzero", toJson stats.nonzeroElements),
    ("maxAbs", jsonFloat32 stats.maxAbs),
    ("l2", jsonFloat32 stats.l2Norm),
    ("rms", jsonFloat32 stats.rms)
  ]

private def namedAdapterStatsJson (stats : NamedAdapterGradientStats) : Lean.Json :=
  Lean.Json.mkObj [
    ("projection", toJson stats.projection),
    ("adapter", toJson stats.adapterId),
    ("a", gradientStatsJson stats.adapterA),
    ("b", gradientStatsJson stats.adapterB)
  ]


private def requireFinite (label : String) (value : Float32) : IO Unit :=
  unless finiteFloat32 value do
    throw <| IO.userError s!"nonfinite training result: {label}"

private def checkGradientStats (label : String) (stats : GradientStats) : IO Unit := do
  unless stats.finiteElements == stats.elements do
    throw <| IO.userError
      s!"nonfinite training gradient: {label} has {stats.finiteElements}/{stats.elements} finite values"
  requireFinite (label ++ ".maxAbs") stats.maxAbs
  requireFinite (label ++ ".l2") stats.l2Norm
  requireFinite (label ++ ".rms") stats.rms

private def checkGradients (loss : Float32)
    (gradients : Array NamedAdapterGradientStats) : IO Unit := do
  requireFinite "loss" loss
  for gradient in gradients do
    checkGradientStats (gradient.projection ++ ".adapterA") gradient.adapterA
    checkGradientStats (gradient.projection ++ ".adapterB") gradient.adapterB

private def checkDPOResult (result : DPOResult) : IO Unit := do
  requireFinite "loss" result.loss
  requireFinite "rewardAccuracy" result.rewardAccuracy
  requireFinite "meanChosenReward" result.meanChosenReward
  requireFinite "meanRejectedReward" result.meanRejectedReward
  requireFinite "meanRewardMargin" result.meanRewardMargin

private def checkGRPOResult (result : GRPOResult) : IO Unit := do
  requireFinite "loss" result.loss
  requireFinite "meanReward" result.meanReward
  requireFinite "rewardStd" result.rewardStd
  requireFinite "meanKL" result.meanKL
  requireFinite "clipFraction" result.clipFraction
  requireFinite "meanAbsoluteAdvantage" result.meanAbsoluteAdvantage

private def elapsedMilliseconds (start finish : Nat) : Float :=
  (finish - start).toFloat / 1000000.0

private def checkDPOProjectionResult (result : DPOProjectionStepResult) : IO Unit := do
  requireFinite "loss" result.loss
  requireFinite "rewardAccuracy" result.rewardAccuracy
  requireFinite "meanChosenReward" result.meanChosenReward
  requireFinite "meanRejectedReward" result.meanRejectedReward
  requireFinite "meanRewardMargin" result.meanRewardMargin
  checkGradients result.loss result.gradients

private def checkGRPOProjectionResult (result : GRPOProjectionStepResult) : IO Unit := do
  requireFinite "loss" result.loss
  requireFinite "meanReward" result.meanReward
  requireFinite "rewardStd" result.rewardStd
  requireFinite "meanKL" result.meanKL
  requireFinite "clipFraction" result.clipFraction
  requireFinite "meanAbsoluteAdvantage" result.meanAbsoluteAdvantage
  checkGradients result.loss result.gradients

private def evaluateBatches (trainer : ProjectionTrainer) (batcher : Batcher)
    (batchSize batches : Nat) : IO Float32 := do
  let mut current := batcher
  let mut total : Float32 := 0
  for _ in [:batches] do
    let mut tokens := Array.mkEmpty batchSize
    let mut targets := Array.mkEmpty batchSize
    for _ in [:batchSize] do
      let (next, sequenceTokens, sequenceTargets) := current.next
      current := next
      tokens := tokens.push sequenceTokens
      targets := targets.push sequenceTargets
    let loss ← trainer.evaluate tokens targets
    requireFinite "validation loss" loss
    total := total + loss
  return total / batches.toUInt32.toFloat32

private def evaluateDPOBatches (trainer : ProjectionTrainer)
    (batcher : PreferenceDataset.Batcher) (pairBatchSize batches : Nat) : IO DPOResult := do
  let mut current := batcher
  let mut totalLoss : Float32 := 0
  let mut totalAccuracy : Float32 := 0
  let mut totalChosen : Float32 := 0
  let mut totalRejected : Float32 := 0
  let mut totalMargin : Float32 := 0
  for _ in [:batches] do
    let (next, batch) := current.next pairBatchSize
    current := next
    let result ← trainer.evaluateDPO batch.tokens batch.targets batch.masks
    checkDPOResult result
    totalLoss := totalLoss + result.loss
    totalAccuracy := totalAccuracy + result.rewardAccuracy
    totalChosen := totalChosen + result.meanChosenReward
    totalRejected := totalRejected + result.meanRejectedReward
    totalMargin := totalMargin + result.meanRewardMargin
  let inverse := 1 / batches.toUInt32.toFloat32
  return {
    loss := totalLoss * inverse
    rewardAccuracy := totalAccuracy * inverse
    meanChosenReward := totalChosen * inverse
    meanRejectedReward := totalRejected * inverse
    meanRewardMargin := totalMargin * inverse
  }

private def evaluateGRPOBatches (trainer : ProjectionTrainer)
    (batcher : GRPODataset.Batcher) (groupBatchSize batches : Nat) : IO GRPOResult := do
  let mut current := batcher
  let mut totalLoss : Float32 := 0
  let mut totalReward : Float32 := 0
  let mut totalRewardStd : Float32 := 0
  let mut totalKL : Float32 := 0
  let mut totalClipFraction : Float32 := 0
  let mut totalAbsoluteAdvantage : Float32 := 0
  for _ in [:batches] do
    let (next, batch) := current.next groupBatchSize
    current := next
    let result ← trainer.evaluateGRPO batch.tokens batch.targets batch.masks
      batch.behaviorLogProbabilities batch.rewards
    totalLoss := totalLoss + result.loss
    checkGRPOResult result
    totalReward := totalReward + result.meanReward
    totalRewardStd := totalRewardStd + result.rewardStd
    totalKL := totalKL + result.meanKL
    totalClipFraction := totalClipFraction + result.clipFraction
    totalAbsoluteAdvantage := totalAbsoluteAdvantage + result.meanAbsoluteAdvantage
  let inverse := 1 / batches.toUInt32.toFloat32
  return {
    loss := totalLoss * inverse
    meanReward := totalReward * inverse
    rewardStd := totalRewardStd * inverse
    meanKL := totalKL * inverse
    clipFraction := totalClipFraction * inverse
    meanAbsoluteAdvantage := totalAbsoluteAdvantage * inverse
  }

private def buildCompleteProgram (trainer : @& ProjectionTrainer) :
    IO Cuda.Qwen36.FullTrainingMegakernel.Program := do
  let program ← trainer.buildFullMegakernelProgram
  unless program.descriptor.projectionCount.toNat == trainer.adapters.size do
    throw <| IO.userError <|
      s!"complete-model tape registered {program.descriptor.projectionCount} projections, " ++
      s!"expected {trainer.adapters.size}"
  return program

private def stepCompleteSFTWithDiagnostics (trainer : @& ProjectionTrainer)
    (program : @& Cuda.Qwen36.FullTrainingMegakernel.Program)
    (tokens targets : Array (Array UInt32)) : IO ProjectionStepResult := do
  let loss ← trainer.stepFullMegakernelProgram program tokens targets
  let result : ProjectionStepResult := { loss, gradients := ← trainer.gradientDiagnostics }
  checkGradients result.loss result.gradients
  return result

private def stepCompleteGRPOWithDiagnostics (trainer : @& ProjectionTrainer)
    (program : @& Cuda.Qwen36.FullTrainingMegakernel.Program)
    (tokens targets masks : Array (Array UInt32)) (behavior : Array (Array Float32))
    (rewards : Array Float32)
    (updates : Nat) : IO GRPOProjectionStepResult := do
  let result ← trainer.stepGRPOFullMegakernelProgramUpdates program tokens targets masks behavior
    rewards updates
  let projectionResult : GRPOProjectionStepResult := {
    loss := result.loss
    meanReward := result.meanReward
    rewardStd := result.rewardStd
    meanKL := result.meanKL
    clipFraction := result.clipFraction
    meanAbsoluteAdvantage := result.meanAbsoluteAdvantage
    gradients := ← trainer.gradientDiagnostics
  }
  checkGRPOProjectionResult projectionResult
  return projectionResult

private def runSFTTraining (config : CliConfig) : IO Unit := do
  let train ← loadBatcher config.dataset config.sequenceLength
  let validation ← match config.valDataset with
    | some path => pure (some (← loadBatcher path config.sequenceLength))
    | none => pure none
  let trainer ← Cuda.Qwen36.CheckpointLoRA.loadProjectionWide config.modelDirectory {
    batchSize := config.batchSize
    sequenceLength := config.sequenceLength
    adapterCount := 1
    adapterIds := Array.replicate config.batchSize 0
    updateMask := #[1]
  } {
    rank := config.rank
    alpha := config.alpha
    learningRate := config.learningRate
    seed := config.seed
  }
  match config.resume with
  | some path =>
    trainer.resumeCheckpoint path
    emit [("type", toJson "resume"), ("path", toJson path.toString)]
  | none => pure ()
  let program ← buildCompleteProgram trainer
  emit [
    ("type", toJson "start"), ("objective", toJson "sft"),
    ("datasetTokens", toJson train.tokenCount), ("steps", toJson config.steps),
    ("batchSize", toJson config.batchSize), ("sequenceLength", toJson config.sequenceLength),
    ("rank", toJson config.rank), ("learningRate", jsonFloat32 config.learningRate),
    ("profile", toJson "projection-wide"),
    ("kernel", toJson "full-pretraining-megakernel"),
    ("projections", toJson trainer.adapters.size)
  ]
  let mut batcher := train
  let mut lastLoss : Float32 := 0
  for step in [:config.steps] do
    let mut tokens := Array.mkEmpty config.batchSize
    let mut targets := Array.mkEmpty config.batchSize
    for _ in [:config.batchSize] do
      let (next, sequenceTokens, sequenceTargets) := batcher.next
      batcher := next
      tokens := tokens.push sequenceTokens
      targets := targets.push sequenceTargets
    let started ← IO.monoNanosNow
    let result ← stepCompleteSFTWithDiagnostics trainer program tokens targets
    let finished ← IO.monoNanosNow
    lastLoss := result.loss
    if (step + 1) % config.logEvery == 0 then
      emit [
        ("type", toJson "step"), ("objective", toJson "sft"), ("step", toJson (step + 1)),
        ("epoch", toJson batcher.epoch), ("loss", jsonFloat32 result.loss),
        ("ms", toJson (elapsedMilliseconds started finished)),
        ("adapters", Lean.Json.arr (result.gradients.map namedAdapterStatsJson))
      ]
    if config.valEvery > 0 && (step + 1) % config.valEvery == 0 then
      if let some valBatcher := validation then
        let valLoss ← evaluateBatches trainer valBatcher config.batchSize config.valBatches
        emit [("type", toJson "val"), ("objective", toJson "sft"),
          ("step", toJson (step + 1)), ("loss", jsonFloat32 valLoss)]
    if config.saveEvery > 0 && step + 1 != config.steps && (step + 1) % config.saveEvery == 0 then
      if let some path := config.checkpointOut then
        trainer.saveCheckpoint path
        emit [("type", toJson "checkpoint"), ("step", toJson (step + 1)),
          ("path", toJson path.toString)]
  match config.checkpointOut with
  | some path =>
    trainer.saveCheckpoint path
    emit [("type", toJson "checkpoint"), ("step", toJson config.steps),
      ("path", toJson path.toString)]
  | none => pure ()
  emit [("type", toJson "done"), ("objective", toJson "sft"),
    ("steps", toJson config.steps), ("finalLoss", jsonFloat32 lastLoss)]

private def runDPOTraining (config : CliConfig) : IO Unit := do
  let train ← PreferenceDataset.load config.dataset config.sequenceLength
  let validation ← match config.valDataset with
    | some path => pure (some (← PreferenceDataset.load path config.sequenceLength))
    | none => pure none
  let sequenceBatchSize := config.batchSize * 2
  let trainer ← Cuda.Qwen36.CheckpointLoRA.loadProjectionWide config.modelDirectory {
    batchSize := sequenceBatchSize
    sequenceLength := config.sequenceLength
    adapterCount := 1
    adapterIds := Array.replicate sequenceBatchSize 0
    updateMask := #[1]
  } {
    rank := config.rank
    alpha := config.alpha
    learningRate := config.learningRate
    enableDPO := true
    dpoBeta := config.dpoBeta
    seed := config.seed
  }
  match config.resume with
  | some path =>
    trainer.resumeCheckpoint path
    emit [("type", toJson "resume"), ("path", toJson path.toString)]
  | none => pure ()
  emit [
    ("type", toJson "start"), ("objective", toJson "dpo"),
    ("datasetPairs", toJson train.pairCount), ("steps", toJson config.steps),
    ("pairBatchSize", toJson config.batchSize), ("sequenceBatchSize", toJson sequenceBatchSize),
    ("sequenceLength", toJson config.sequenceLength), ("rank", toJson config.rank),
    ("learningRate", jsonFloat32 config.learningRate), ("dpoBeta", jsonFloat32 config.dpoBeta),
    ("profile", toJson "projection-wide"), ("kernel", toJson "cuda-graph"),
    ("projections", toJson trainer.adapters.size)
  ]
  let mut batcher := train
  let mut lastLoss : Float32 := 0
  for step in [:config.steps] do
    let (next, batch) := batcher.next config.batchSize
    batcher := next
    let started ← IO.monoNanosNow
    let result ← trainer.stepDPOWithDiagnostics batch.tokens batch.targets batch.masks
    let finished ← IO.monoNanosNow
    lastLoss := result.loss
    checkDPOProjectionResult result
    if (step + 1) % config.logEvery == 0 then
      emit [
        ("type", toJson "step"), ("objective", toJson "dpo"), ("step", toJson (step + 1)),
        ("epoch", toJson batcher.epoch), ("loss", jsonFloat32 result.loss),
        ("rewardAccuracy", jsonFloat32 result.rewardAccuracy),
        ("meanChosenReward", jsonFloat32 result.meanChosenReward),
        ("meanRejectedReward", jsonFloat32 result.meanRejectedReward),
        ("meanRewardMargin", jsonFloat32 result.meanRewardMargin),
        ("ms", toJson (elapsedMilliseconds started finished)),
        ("adapters", Lean.Json.arr (result.gradients.map namedAdapterStatsJson))
      ]
    if config.valEvery > 0 && (step + 1) % config.valEvery == 0 then
      if let some valBatcher := validation then
        let result ← evaluateDPOBatches trainer valBatcher config.batchSize config.valBatches
        checkDPOResult result
        emit [
          ("type", toJson "val"), ("objective", toJson "dpo"), ("step", toJson (step + 1)),
          ("loss", jsonFloat32 result.loss),
          ("rewardAccuracy", jsonFloat32 result.rewardAccuracy),
          ("meanChosenReward", jsonFloat32 result.meanChosenReward),
          ("meanRejectedReward", jsonFloat32 result.meanRejectedReward),
          ("meanRewardMargin", jsonFloat32 result.meanRewardMargin)
        ]
    if config.saveEvery > 0 && step + 1 != config.steps && (step + 1) % config.saveEvery == 0 then
      if let some path := config.checkpointOut then
        trainer.saveCheckpoint path
        emit [("type", toJson "checkpoint"), ("step", toJson (step + 1)),
          ("path", toJson path.toString)]
  match config.checkpointOut with
  | some path =>
    trainer.saveCheckpoint path
    emit [("type", toJson "checkpoint"), ("step", toJson config.steps),
      ("path", toJson path.toString)]
  | none => pure ()
  emit [("type", toJson "done"), ("objective", toJson "dpo"),
    ("steps", toJson config.steps), ("finalLoss", jsonFloat32 lastLoss)]

private def runGRPOTraining (config : CliConfig) : IO Unit := do
  let train ← GRPODataset.load config.dataset config.sequenceLength config.grpoGroupSize
  let validation ← match config.valDataset with
    | some path =>
        pure (some (← GRPODataset.load path config.sequenceLength config.grpoGroupSize))
    | none => pure none
  let sequenceBatchSize := config.batchSize * config.grpoGroupSize
  let trainer ← Cuda.Qwen36.CheckpointLoRA.loadProjectionWide config.modelDirectory {
    batchSize := sequenceBatchSize
    sequenceLength := config.sequenceLength
    adapterCount := 1
    adapterIds := Array.replicate sequenceBatchSize 0
    updateMask := #[1]
  } {
    rank := config.rank
    alpha := config.alpha
    learningRate := config.learningRate
    enableGRPO := true
    grpoGroupSize := config.grpoGroupSize
    grpoClipEpsilon := config.grpoClipEpsilon
    grpoKLBeta := config.grpoKLBeta
    grpoAdvantageEpsilon := config.grpoAdvantageEpsilon
    seed := config.seed
  }
  match config.resume with
  | some path =>
    trainer.resumeCheckpoint path
    emit [("type", toJson "resume"), ("path", toJson path.toString)]
  | none => pure ()
  let program ← buildCompleteProgram trainer
  emit [
    ("type", toJson "start"), ("objective", toJson "grpo"),
    ("datasetGroups", toJson train.groupCount), ("steps", toJson config.steps),
    ("groupBatchSize", toJson config.batchSize), ("groupSize", toJson config.grpoGroupSize),
    ("sequenceBatchSize", toJson sequenceBatchSize),
    ("sequenceLength", toJson config.sequenceLength), ("rank", toJson config.rank),
    ("learningRate", jsonFloat32 config.learningRate),
    ("updatesPerBatch", toJson config.grpoUpdates),
    ("clipEpsilon", jsonFloat32 config.grpoClipEpsilon),
    ("klBeta", jsonFloat32 config.grpoKLBeta),
    ("advantageEpsilon", jsonFloat32 config.grpoAdvantageEpsilon),
    ("profile", toJson "projection-wide"), ("kernel", toJson "full-grpo-megakernel"),
    ("projections", toJson trainer.adapters.size)
  ]
  let mut batcher := train
  let mut lastLoss : Float32 := 0
  for step in [:config.steps] do
    let (next, batch) := batcher.next config.batchSize
    batcher := next
    let started ← IO.monoNanosNow
    let result ← stepCompleteGRPOWithDiagnostics trainer program batch.tokens batch.targets
      batch.masks batch.behaviorLogProbabilities batch.rewards config.grpoUpdates
    let finished ← IO.monoNanosNow
    lastLoss := result.loss
    if (step + 1) % config.logEvery == 0 then
      emit [
        ("type", toJson "step"), ("objective", toJson "grpo"), ("step", toJson (step + 1)),
        ("optimizerUpdates", toJson ((step + 1) * config.grpoUpdates)),
        ("epoch", toJson batcher.epoch), ("loss", jsonFloat32 result.loss),
        ("meanReward", jsonFloat32 result.meanReward),
        ("rewardStd", jsonFloat32 result.rewardStd), ("meanKL", jsonFloat32 result.meanKL),
        ("clipFraction", jsonFloat32 result.clipFraction),
        ("meanAbsoluteAdvantage", jsonFloat32 result.meanAbsoluteAdvantage),
        ("ms", toJson (elapsedMilliseconds started finished)),
        ("adapters", Lean.Json.arr (result.gradients.map namedAdapterStatsJson))
      ]
    if config.valEvery > 0 && (step + 1) % config.valEvery == 0 then
      if let some valBatcher := validation then
        let result ← evaluateGRPOBatches trainer valBatcher config.batchSize config.valBatches
        checkGRPOResult result
        emit [
          ("type", toJson "val"), ("objective", toJson "grpo"),
          ("step", toJson (step + 1)), ("loss", jsonFloat32 result.loss),
          ("meanReward", jsonFloat32 result.meanReward),
          ("rewardStd", jsonFloat32 result.rewardStd), ("meanKL", jsonFloat32 result.meanKL),
          ("clipFraction", jsonFloat32 result.clipFraction),
          ("meanAbsoluteAdvantage", jsonFloat32 result.meanAbsoluteAdvantage)
        ]
    if config.saveEvery > 0 && step + 1 != config.steps && (step + 1) % config.saveEvery == 0 then
      if let some path := config.checkpointOut then
        trainer.saveCheckpoint path
        emit [("type", toJson "checkpoint"), ("step", toJson (step + 1)),
          ("path", toJson path.toString)]
  match config.checkpointOut with
  | some path =>
    trainer.saveCheckpoint path
    emit [("type", toJson "checkpoint"), ("step", toJson config.steps),
      ("path", toJson path.toString)]
  | none => pure ()
  emit [("type", toJson "done"), ("objective", toJson "grpo"),
    ("steps", toJson config.steps),
    ("optimizerUpdates", toJson (config.steps * config.grpoUpdates)),
    ("finalLoss", jsonFloat32 lastLoss)]

def run (args : List String) : IO UInt32 := do
  let config ← match parseArgs args { modelDirectory := "", dataset := "" } with
    | .ok config => pure config
    | .error message =>
      IO.eprintln s!"error: {message}"
      IO.eprintln usage
      return 2
  let config ← match config.check with
    | .ok config => pure config
    | .error message =>
      IO.eprintln s!"error: {message}"
      IO.eprintln usage
      return 2
  let envDirectory := (← IO.getEnv "QWEN_MODEL_DIR").getD ""
  let modelDirectory : System.FilePath :=
    if config.modelDirectory.toString.isEmpty then envDirectory else config.modelDirectory
  let config := { config with modelDirectory }
  if config.modelDirectory.toString.isEmpty then
    IO.eprintln "error: --model-dir or QWEN_MODEL_DIR is required"
    IO.eprintln usage
    return 2
  if config.dataset.toString.isEmpty then
    IO.eprintln "error: --dataset is required"
    IO.eprintln usage
    return 2
  try
    match config.objective with
    | .sft => runSFTTraining config
    | .dpo => runDPOTraining config
    | .grpo => runGRPOTraining config
    return 0
  catch error =>
    IO.eprintln s!"error: {error}"
    return 1

end Qwen38Train

public def main (args : List String) : IO UInt32 :=
  Qwen38Train.run args
