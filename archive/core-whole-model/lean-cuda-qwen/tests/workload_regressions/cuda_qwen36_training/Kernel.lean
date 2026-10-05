/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-! Device-tail recurrence gate for the shared sequential/graph training executor. -/

namespace Qwen36TrainingGate

open Cuda.Qwen36.Training

private def elements : Nat := 32
private def steps : Nat := 4

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def pushUInt64 (bytes : ByteArray) (value : UInt64) : ByteArray := Id.run do
  let mut result := bytes
  for shift in [:8] do
    result := result.push (value >>> (8 * shift).toUInt64).toUInt8
  return result

private def floats (count : Nat) (value : Nat → Float32) : ByteArray :=
  (List.range count).foldl (init := ByteArray.empty) fun bytes index =>
    pushUInt32 bytes (value index).toBits

private def uint32s (values : Array UInt32) : ByteArray :=
  values.foldl (init := ByteArray.empty) pushUInt32

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

private def initialParameter (index : Nat) : Float32 :=
  (Int32.ofInt (Int.ofNat (index % 13) - 6)).toFloat32 / 32

private def gradient (index : Nat) : Float32 :=
  (Int32.ofInt (Int.ofNat (index % 9) - 4)).toFloat32 / 64

private structure State where
  group : ParameterGroupF32
  initial : ByteArray

private def allocate (stream : Cuda.Stream) : IO State := do
  let parameter ← Cuda.Buffer.alloc Float32 elements.toUSize
  let gradientBuffer ← Cuda.Buffer.alloc Float32 elements.toUSize
  let firstMoment ← Cuda.Buffer.alloc Float32 elements.toUSize
  let secondMoment ← Cuda.Buffer.alloc Float32 elements.toUSize
  let initial := floats elements initialParameter
  parameter.copyFrom initial stream
  gradientBuffer.copyFrom (floats elements gradient) stream
  firstMoment.copyFrom (floats elements fun _ => 0) stream
  secondMoment.copyFrom (floats elements fun _ => 0) stream
  return {
    initial
    group := {
      parameter
      gradient := gradientBuffer
      firstMoment
      secondMoment
      elements := elements.toUInt32
      weightDecay := 0.01
    }
  }

private def config : Cuda.Training.AdamW.Config := {
  learningRate := 0.01
  beta1 := 0.9
  beta2 := 0.99
  inverseBiasCorrection1 := 10
  inverseBiasCorrection2 := 100
  epsilon := 1e-8
  weightDecay := 0
}

private def hyperparameters : AdamWHyperparametersF32 := {
  learningRate := 0.01
  beta1 := 0.9
  beta2 := 0.99
  epsilon := 1e-8
}

private def allocateSchedule (stream : Cuda.Stream) : IO AdamWScheduleF32 := do
  let step ← Cuda.Buffer.alloc UInt32 1
  let beta1Power ← Cuda.Buffer.alloc Float32 1
  let beta2Power ← Cuda.Buffer.alloc Float32 1
  let inverseBiasCorrection1 ← Cuda.Buffer.alloc Float32 1
  let inverseBiasCorrection2 ← Cuda.Buffer.alloc Float32 1
  step.copyFrom (pushUInt32 ByteArray.empty 0) stream
  beta1Power.copyFrom (floats 1 fun _ => 1) stream
  beta2Power.copyFrom (floats 1 fun _ => 1) stream
  inverseBiasCorrection1.copyFrom (floats 1 fun _ => 0) stream
  inverseBiasCorrection2.copyFrom (floats 1 fun _ => 0) stream
  return {
    step, beta1Power, beta2Power, inverseBiasCorrection1, inverseBiasCorrection2
  }

@[test]
def adapterAllocationAndOptimizerStateResume : IO Unit := do
  let stream ← Cuda.Stream.default
  let descriptor : Cuda.Qwen36.LoRA.Descriptor := {
    batchSize := 4
    sequenceLength := 2
    adapterCount := 2
    rank := 16
    inputFeatures := 16
    outputFeatures := 16
    alpha := 16
    adapterIds := #[0, 1, 0, 1]
  }
  let adapter ← Cuda.Qwen36.Projection.AdapterF32.allocate descriptor #[1, 0]
    0xc0ffee 0.02 0.01 stream
  adapter.check 8 16 16
  let aElements := 2 * 16 * 16
  let bElements := 2 * 16 * 16
  let rankElements := 8 * 16
  let zeroA := floats aElements fun _ => 0
  let zeroB := floats bElements fun _ => 0
  LeanTest.assertTrue ((← adapter.adapterA.copyTo stream) != zeroA)
    "deterministic LoRA adapter-A initialization was degenerate"
  LeanTest.assertEqual (← adapter.adapterB.copyTo stream) zeroB
  LeanTest.assertEqual (← adapter.rankActivation.copyTo stream) (floats rankElements fun _ => 0)
  LeanTest.assertEqual (← adapter.adapterAGradient.copyTo stream) zeroA
  LeanTest.assertEqual (← adapter.adapterBGradient.copyTo stream) zeroB
  LeanTest.assertEqual (← adapter.adapterAFirstMoment.copyTo stream) zeroA
  LeanTest.assertEqual (← adapter.adapterASecondMoment.copyTo stream) zeroA
  LeanTest.assertEqual (← adapter.adapterBFirstMoment.copyTo stream) zeroB
  LeanTest.assertEqual (← adapter.adapterBSecondMoment.copyTo stream) zeroB
  LeanTest.assertEqual (← adapter.adapterIds.copyTo stream) (uint32s #[0, 1, 0, 1])
  LeanTest.assertEqual (← adapter.updateMask.copyTo stream) (uint32s #[1, 0])

  let sameSeed ← Cuda.Qwen36.Projection.AdapterF32.allocate descriptor #[1, 0]
    0xc0ffee 0.02 0.01 stream
  LeanTest.assertEqual (← sameSeed.adapterA.copyTo stream) (← adapter.adapterA.copyTo stream)

  let snapshot ← adapter.snapshot stream
  let changedA := floats aElements fun _ => 0
  let changedB := floats bElements fun _ => 0.25
  let changedMomentA := floats aElements fun _ => -0.125
  let changedMomentB := floats bElements fun _ => 0.5
  adapter.adapterA.copyFrom changedA stream
  adapter.adapterB.copyFrom changedB stream
  adapter.adapterIds.copyFrom (uint32s #[1, 0, 1, 0]) stream
  adapter.adapterAFirstMoment.copyFrom changedMomentA stream
  adapter.adapterASecondMoment.copyFrom changedMomentA stream
  adapter.adapterBFirstMoment.copyFrom changedMomentB stream
  adapter.adapterBSecondMoment.copyFrom changedMomentB stream
  adapter.updateMask.copyFrom (uint32s #[0, 1]) stream
  adapter.restore snapshot stream
  LeanTest.assertTrue ((← adapter.snapshot stream) == snapshot)
    "LoRA adapter state did not restore byte-exactly"

  let schedule ← AdamWScheduleF32.allocate stream
  let scheduleSnapshot ← schedule.snapshot stream
  submitAdvanceAdamWScheduleF32 (.sequential stream) hyperparameters schedule
  LeanTest.assertTrue ((← schedule.snapshot stream) != scheduleSnapshot)
    "AdamW schedule did not advance before restore"
  schedule.restore scheduleSnapshot stream
  LeanTest.assertTrue ((← schedule.snapshot stream) == scheduleSnapshot)
    "AdamW schedule did not restore byte-exactly"

private def assertGroupEqual (expected actual : ParameterGroupF32)
    (stream : Cuda.Stream) : IO Unit := do
  LeanTest.assertEqual (← actual.parameter.copyTo stream) (← expected.parameter.copyTo stream)
  LeanTest.assertEqual (← actual.firstMoment.copyTo stream) (← expected.firstMoment.copyTo stream)
  LeanTest.assertEqual (← actual.secondMoment.copyTo stream) (← expected.secondMoment.copyTo stream)

private def assertScheduleEqual (expected actual : AdamWScheduleF32)
    (stream : Cuda.Stream) : IO Unit := do
  LeanTest.assertEqual (← actual.step.copyTo stream) (← expected.step.copyTo stream)
  LeanTest.assertEqual (← actual.beta1Power.copyTo stream) (← expected.beta1Power.copyTo stream)
  LeanTest.assertEqual (← actual.beta2Power.copyTo stream) (← expected.beta2Power.copyTo stream)
  LeanTest.assertEqual (← actual.inverseBiasCorrection1.copyTo stream)
    (← expected.inverseBiasCorrection1.copyTo stream)
  LeanTest.assertEqual (← actual.inverseBiasCorrection2.copyTo stream)
    (← expected.inverseBiasCorrection2.copyTo stream)

@[test]
def deviceTailAdamWMatchesSequential : IO Unit := do
  let stream ← Cuda.Stream.default
  let baseline ← allocate stream
  let candidate ← allocate stream
  baseline.group.check
  candidate.group.check
  for _ in [:steps] do
    submitAdamWF32 (.sequential stream) config baseline.group

  let graphHandle ← Cuda.Buffer.alloc UInt64 1
  let remaining ← Cuda.Buffer.alloc UInt32 1
  let builder ← Cuda.GraphBuilder.create
  submitAdamWF32 (.graph builder) config candidate.group
  appendTailTrainingGraph builder graphHandle remaining
  let graph ← builder.instantiate
  graphHandle.copyFrom (pushUInt64 ByteArray.empty (← graph.deviceHandle)) stream
  remaining.copyFrom (pushUInt32 ByteArray.empty steps.toUInt32) stream
  (← graph.launchOn stream).waitChecked "Qwen3.6 device-tail AdamW recurrence"

  assertGroupEqual baseline.group candidate.group stream
  LeanTest.assertEqual (← remaining.copyTo stream) (pushUInt32 ByteArray.empty 0)
  LeanTest.assertTrue ((← candidate.group.parameter.copyTo stream) != candidate.initial)
    "resident AdamW did not change parameters"

@[test]
def deviceTailResidentAdamWMatchesSequential : IO Unit := do
  let stream ← Cuda.Stream.default
  let baseline ← allocate stream
  let candidate ← allocate stream
  let baselineSchedule ← allocateSchedule stream
  let candidateSchedule ← allocateSchedule stream
  baseline.group.check
  candidate.group.check
  baselineSchedule.check
  candidateSchedule.check
  hyperparameters.check
  for _ in [:steps] do
    submitAdvanceAdamWScheduleF32 (.sequential stream) hyperparameters baselineSchedule
    submitResidentAdamWF32 (.sequential stream) hyperparameters baselineSchedule baseline.group

  let graphHandle ← Cuda.Buffer.alloc UInt64 1
  let remaining ← Cuda.Buffer.alloc UInt32 1
  let builder ← Cuda.GraphBuilder.create
  submitAdvanceAdamWScheduleF32 (.graph builder) hyperparameters candidateSchedule
  submitResidentAdamWF32 (.graph builder) hyperparameters candidateSchedule candidate.group
  appendTailTrainingGraph builder graphHandle remaining
  let graph ← builder.instantiate
  graphHandle.copyFrom (pushUInt64 ByteArray.empty (← graph.deviceHandle)) stream
  remaining.copyFrom (pushUInt32 ByteArray.empty steps.toUInt32) stream
  (← graph.launchOn stream).waitChecked "Qwen3.6 resident AdamW recurrence"

  assertGroupEqual baseline.group candidate.group stream
  assertScheduleEqual baselineSchedule candidateSchedule stream
  LeanTest.assertEqual (← candidateSchedule.step.copyTo stream)
    (pushUInt32 ByteArray.empty steps.toUInt32)
  LeanTest.assertEqual (← remaining.copyTo stream) (pushUInt32 ByteArray.empty 0)
  LeanTest.assertTrue ((← candidate.group.parameter.copyTo stream) != candidate.initial)
    "bias-corrected resident AdamW did not change parameters"

@[test]
def dpoObjectiveAndGradientMatchAnalytic : IO Unit := do
  let stream ← Cuda.Stream.default
  let layout : Cuda.Qwen36.SequenceLayout := { batchSize := 2, sequenceLength := 2 }
  let rows := layout.tokens.toNat
  let classes : UInt32 := 4
  let elementCount := rows * classes.toNat
  let logits ← Cuda.Buffer.alloc Float32 elementCount.toUSize
  let targets ← Cuda.Buffer.alloc UInt32 rows.toUSize
  let masks ← Cuda.Buffer.alloc UInt32 rows.toUSize
  let rowLogProbability ← Cuda.Buffer.alloc Float32 rows.toUSize
  let policy ← Cuda.Buffer.alloc Float32 layout.batchSize.toUSize
  let reference ← Cuda.Buffer.alloc Float32 layout.batchSize.toUSize
  let coefficients ← Cuda.Buffer.alloc Float32 layout.batchSize.toUSize
  let loss ← Cuda.Buffer.alloc Float32 1
  let gradientBuffer ← Cuda.Buffer.alloc Float32 elementCount.toUSize
  logits.copyFrom (floats elementCount fun _ => 0) stream
  targets.copyFrom (uint32s #[0, 2, 0, 1]) stream
  masks.copyFrom (uint32s #[0, 1, 0, 1]) stream
  reference.copyFrom (floats 2 fun index => if index == 0 then -2 else -3) stream

  (← Cuda.Qwen36.Model.tokenLogProbabilityF32 stream logits targets masks
    rowLogProbability layout.tokens classes).waitChecked "DPO token log probability"
  (← Cuda.Qwen36.Model.reduceSequenceLogProbabilityF32 stream rowLogProbability policy
    layout).waitChecked "DPO sequence log probability"
  let beta : Float32 := 0.2
  (← Cuda.Qwen36.Model.dpoObjectiveF32 stream policy reference coefficients loss layout
    beta).waitChecked "DPO objective"
  (← Cuda.Qwen36.Model.dpoGradientF32 stream logits targets masks coefficients
    gradientBuffer layout classes).waitChecked "DPO logit VJP"

  let logProbability := -Float32.log (4 : Float32)
  checkClose "DPO masked row log probabilities" (← rowLogProbability.copyTo stream)
    (floats rows fun index => if index == 1 || index == 3 then logProbability else 0) rows
  checkClose "DPO sequence log probabilities" (← policy.copyTo stream)
    (floats 2 fun _ => logProbability) 2
  let margin := Cuda.Qwen36.Model.dpoMarginF32 beta logProbability logProbability (-2) (-3)
  let coefficient := Cuda.Qwen36.Model.dpoChosenCoefficientF32 beta margin
  checkClose "DPO loss" (← loss.copyTo stream)
    (floats 1 fun _ => Cuda.Qwen36.Model.dpoLossFromMarginF32 margin) 1
  checkClose "DPO pair coefficients" (← coefficients.copyTo stream)
    (floats 2 fun index => if index == 0 then coefficient else -coefficient) 2
  let expectedGradient := floats elementCount fun index =>
    let row := index / classes.toNat
    let column := index % classes.toNat
    if row == 1 then
      coefficient * (if column == 2 then -0.75 else 0.25)
    else if row == 3 then
      (-coefficient) * (if column == 1 then -0.75 else 0.25)
    else
      0
  checkClose "DPO completion-masked logit VJP" (← gradientBuffer.copyTo stream)
    expectedGradient elementCount

@[test]
def grpoObjectiveAndGradientMatchAnalytic : IO Unit := do
  let stream ← Cuda.Stream.default
  let layout : Cuda.Qwen36.SequenceLayout := { batchSize := 2, sequenceLength := 2 }
  let rows := layout.tokens.toNat
  let classes : UInt32 := 4
  let elementCount := rows * classes.toNat
  let logits ← Cuda.Buffer.alloc Float32 elementCount.toUSize
  let targets ← Cuda.Buffer.alloc UInt32 rows.toUSize
  let masks ← Cuda.Buffer.alloc UInt32 rows.toUSize
  let policy ← Cuda.Buffer.alloc Float32 rows.toUSize
  let old ← Cuda.Buffer.alloc Float32 rows.toUSize
  let reference ← Cuda.Buffer.alloc Float32 rows.toUSize
  let rewards ← Cuda.Buffer.alloc Float32 layout.batchSize.toUSize
  let advantages ← Cuda.Buffer.alloc Float32 layout.batchSize.toUSize
  let coefficients ← Cuda.Buffer.alloc Float32 rows.toUSize
  let statistics ← Cuda.Buffer.alloc Float32 2
  let loss ← Cuda.Buffer.alloc Float32 1
  let gradientBuffer ← Cuda.Buffer.alloc Float32 elementCount.toUSize
  logits.copyFrom (floats elementCount fun _ => 0) stream
  targets.copyFrom (uint32s #[0, 2, 0, 1]) stream
  masks.copyFrom (uint32s #[0, 1, 0, 1]) stream
  rewards.copyFrom (floats 2 fun index => if index == 0 then 1 else 0) stream

  (← Cuda.Qwen36.Model.tokenLogProbabilityF32 stream logits targets masks policy
    layout.tokens classes).waitChecked "GRPO policy token log probability"
  let logProbability := -Float32.log (4 : Float32)
  let highRatio : Float32 := 1.5
  old.copyFrom (floats rows fun index =>
    if index == 1 then logProbability - Float32.log highRatio else logProbability) stream
  reference.copyFrom (floats rows fun index =>
    if index == 3 then logProbability + Float32.log (2 : Float32) else logProbability) stream
  let clipEpsilon : Float32 := 0.2
  let klBeta : Float32 := 0.1
  let advantageEpsilon : Float32 := 1e-6
  (← Cuda.Qwen36.Model.grpoObjectiveF32 stream policy old reference rewards advantages
    coefficients statistics loss masks layout 2 clipEpsilon klBeta
    advantageEpsilon).waitChecked "GRPO objective"
  (← Cuda.Qwen36.Model.grpoGradientF32 stream logits targets masks coefficients
    gradientBuffer layout classes 2).waitChecked "GRPO logit VJP"

  let meanReward : Float32 := 0.5
  let rewardVariance : Float32 := 0.25
  let positiveAdvantage := Cuda.Qwen36.Model.grpoAdvantageF32 1 meanReward rewardVariance
    advantageEpsilon
  let negativeAdvantage := Cuda.Qwen36.Model.grpoAdvantageF32 0 meanReward rewardVariance
    advantageEpsilon
  checkClose "GRPO group-relative advantages" (← advantages.copyTo stream)
    (floats 2 fun index => if index == 0 then positiveAdvantage else negativeAdvantage) 2
  let secondReferenceRatio : Float32 := 2
  let secondKL := secondReferenceRatio - Float32.log secondReferenceRatio - 1
  let secondCoefficient := (negativeAdvantage + klBeta * (secondReferenceRatio - 1)) * 0.5
  checkClose "GRPO per-row coefficients" (← coefficients.copyTo stream)
    (floats rows fun index => if index == 3 then secondCoefficient else 0) rows
  let expectedLoss :=
    (-positiveAdvantage * (1 + clipEpsilon) -
      negativeAdvantage + klBeta * secondKL) * 0.5
  checkClose "GRPO clipped loss" (← loss.copyTo stream) (floats 1 fun _ => expectedLoss) 1
  checkClose "GRPO KL and clip statistics" (← statistics.copyTo stream)
    (floats 2 fun index => if index == 0 then secondKL * 0.5 else 0.5) 2
  let expectedGradient := floats elementCount fun index =>
    let row := index / classes.toNat
    let column := index % classes.toNat
    if row == 3 then
      secondCoefficient * (if column == 1 then -0.75 else 0.25)
    else
      0
  checkClose "GRPO completion-masked logit VJP" (← gradientBuffer.copyTo stream)
    expectedGradient elementCount

private def projectionShape : Cuda.Qwen36.LoRA.Shape := {
  rows := 4
  sequenceLength := 2
  adapterCount := 1
  rank := 16
  inputFeatures := 16
  outputFeatures := 4
  scale := 1
}

private def projectionStep : Cuda.Qwen36.TrainingMegakernel.ProjectionStepF32 := {
  shape := projectionShape
  optimizer := {
    learningRate := 0.01
    beta1 := 0.9
    beta2 := 0.99
    inverseBiasCorrection1 := 10
    inverseBiasCorrection2 := 100
    epsilon := 1e-8
    weightDecay := 0.01
  }
}

private structure ProjectionKernelState where
  input : Cuda.Buffer Float32
  baseWeight : Cuda.Buffer Float32
  adapterA : Cuda.Buffer Float32
  adapterB : Cuda.Buffer Float32
  adapterIds : Cuda.Buffer UInt32
  rankActivation : Cuda.Buffer Float32
  logits : Cuda.Buffer Float32
  rowStatistics : Cuda.Buffer Float32
  logitGradient : Cuda.Buffer Float32
  rowLoss : Cuda.Buffer Float32
  loss : Cuda.Buffer Float32
  adapterAGradient : Cuda.Buffer Float32
  adapterBGradient : Cuda.Buffer Float32
  adapterAFirstMoment : Cuda.Buffer Float32
  adapterASecondMoment : Cuda.Buffer Float32
  adapterBFirstMoment : Cuda.Buffer Float32
  adapterBSecondMoment : Cuda.Buffer Float32
  updateMask : Cuda.Buffer UInt32

private def allocateF32 (stream : Cuda.Stream) (count : Nat)
    (value : Nat → Float32) : IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 count.toUSize
  buffer.copyFrom (floats count value) stream
  return buffer

private def allocateProjectionKernelState (stream : Cuda.Stream) : IO ProjectionKernelState := do
  let shape := projectionShape
  let rows := shape.rows.toNat
  let inputs := rows * shape.inputFeatures.toNat
  let logits := rows * shape.outputFeatures.toNat
  let aElements := shape.adapterCount.toNat * shape.rank.toNat * shape.inputFeatures.toNat
  let bElements := shape.adapterCount.toNat * shape.outputFeatures.toNat * shape.rank.toNat
  let input ← allocateF32 stream inputs fun index =>
    (Int32.ofInt (Int.ofNat (index % 11) - 5)).toFloat32 / 16
  let baseWeight ← allocateF32 stream
    (shape.outputFeatures.toNat * shape.inputFeatures.toNat) fun index =>
      (Int32.ofInt (Int.ofNat (index % 13) - 6)).toFloat32 / 32
  let adapterA ← allocateF32 stream aElements fun index =>
    (Int32.ofInt (Int.ofNat (index % 7) - 3)).toFloat32 / 32
  let adapterB ← allocateF32 stream bElements fun index =>
    (Int32.ofInt (Int.ofNat (index % 5) - 2)).toFloat32 / 64
  let adapterIds ← Cuda.Buffer.alloc UInt32 2
  adapterIds.copyFrom (uint32s #[0, 0]) stream
  let rankActivation ← allocateF32 stream (rows * shape.rank.toNat) fun _ => 0
  let output ← allocateF32 stream logits fun _ => 0
  let rowStatistics ← allocateF32 stream (rows * 2) fun _ => 0
  let logitGradient ← allocateF32 stream logits fun _ => 0
  let rowLoss ← allocateF32 stream rows fun _ => 0
  let loss ← allocateF32 stream 1 fun _ => 0
  let adapterAGradient ← allocateF32 stream aElements fun _ => 0
  let adapterBGradient ← allocateF32 stream bElements fun _ => 0
  let adapterAFirstMoment ← allocateF32 stream aElements fun _ => 0
  let adapterASecondMoment ← allocateF32 stream aElements fun _ => 0
  let adapterBFirstMoment ← allocateF32 stream bElements fun _ => 0
  let adapterBSecondMoment ← allocateF32 stream bElements fun _ => 0
  let updateMask ← Cuda.Buffer.alloc UInt32 1
  updateMask.copyFrom (uint32s #[1]) stream
  return {
    input, baseWeight, adapterA, adapterB, adapterIds, rankActivation, logits := output,
    rowStatistics, logitGradient, rowLoss, loss, adapterAGradient, adapterBGradient,
    adapterAFirstMoment, adapterASecondMoment, adapterBFirstMoment, adapterBSecondMoment, updateMask
  }

private def checkProjectionKernelState (label : String) (expected actual : ProjectionKernelState)
    (stream : Cuda.Stream) (tolerance : Float32 := 2e-4) : IO Unit := do
  let shape := projectionShape
  let rows := shape.rows.toNat
  let logits := rows * shape.outputFeatures.toNat
  let aElements := shape.adapterCount.toNat * shape.rank.toNat * shape.inputFeatures.toNat
  let bElements := shape.adapterCount.toNat * shape.outputFeatures.toNat * shape.rank.toNat
  checkClose (label ++ " rank activation") (← actual.rankActivation.copyTo stream)
    (← expected.rankActivation.copyTo stream) (rows * shape.rank.toNat) tolerance
  checkClose (label ++ " logits") (← actual.logits.copyTo stream)
    (← expected.logits.copyTo stream) logits tolerance
  checkClose (label ++ " logit VJP") (← actual.logitGradient.copyTo stream)
    (← expected.logitGradient.copyTo stream) logits tolerance
  checkClose (label ++ " loss") (← actual.loss.copyTo stream)
    (← expected.loss.copyTo stream) 1 tolerance
  checkClose (label ++ " adapter-A VJP") (← actual.adapterAGradient.copyTo stream)
    (← expected.adapterAGradient.copyTo stream) aElements tolerance
  checkClose (label ++ " adapter-B VJP") (← actual.adapterBGradient.copyTo stream)
    (← expected.adapterBGradient.copyTo stream) bElements tolerance
  checkClose (label ++ " adapter A") (← actual.adapterA.copyTo stream)
    (← expected.adapterA.copyTo stream) aElements tolerance
  checkClose (label ++ " adapter B") (← actual.adapterB.copyTo stream)
    (← expected.adapterB.copyTo stream) bElements tolerance
  checkClose (label ++ " adapter-A first moment") (← actual.adapterAFirstMoment.copyTo stream)
    (← expected.adapterAFirstMoment.copyTo stream) aElements tolerance
  checkClose (label ++ " adapter-A second moment") (← actual.adapterASecondMoment.copyTo stream)
    (← expected.adapterASecondMoment.copyTo stream) aElements tolerance
  checkClose (label ++ " adapter-B first moment") (← actual.adapterBFirstMoment.copyTo stream)
    (← expected.adapterBFirstMoment.copyTo stream) bElements tolerance
  checkClose (label ++ " adapter-B second moment") (← actual.adapterBSecondMoment.copyTo stream)
    (← expected.adapterBSecondMoment.copyTo stream) bElements tolerance

@[test]
def pretrainingMegakernelMatchesModularOracle : IO Unit := do
  let stream ← Cuda.Stream.default
  let baseline ← allocateProjectionKernelState stream
  let fused ← allocateProjectionKernelState stream
  let targets ← Cuda.Buffer.alloc UInt32 projectionShape.rows.toUSize
  targets.copyFrom (uint32s #[0, 2, 0, 1]) stream

  Cuda.Qwen36.LoRA.forwardF32 stream baseline.input baseline.baseWeight baseline.adapterA
    baseline.adapterB baseline.adapterIds baseline.rankActivation baseline.logits projectionShape
  (← Cuda.Qwen36.Model.crossEntropyF32 stream baseline.logits targets baseline.logitGradient
    baseline.rowLoss projectionShape.rows projectionShape.outputFeatures).waitChecked
    "modular pretraining cross entropy"
  (← Cuda.Qwen36.Model.reduceLossF32 stream baseline.rowLoss baseline.loss
    projectionShape.rows).waitChecked "modular pretraining loss reduction"
  Cuda.Qwen36.LoRA.backwardAdaptersF32 stream baseline.input baseline.adapterB baseline.adapterIds
    baseline.rankActivation baseline.logitGradient baseline.adapterAGradient
    baseline.adapterBGradient projectionShape
  (← Cuda.Qwen36.LoRA.adamWF32 stream projectionStep.optimizer baseline.adapterA baseline.adapterB
    baseline.adapterAGradient baseline.adapterBGradient baseline.adapterAFirstMoment
    baseline.adapterASecondMoment baseline.adapterBFirstMoment baseline.adapterBSecondMoment
    baseline.updateMask projectionShape).waitChecked "modular pretraining AdamW"

  (← Cuda.Qwen36.PretrainMegakernel.launchProjectionStepF32 stream fused.input fused.baseWeight
    fused.adapterA fused.adapterB fused.adapterIds fused.rankActivation fused.logits
    fused.rowStatistics fused.logitGradient fused.rowLoss fused.loss fused.adapterAGradient
    fused.adapterBGradient fused.adapterAFirstMoment fused.adapterASecondMoment
    fused.adapterBFirstMoment fused.adapterBSecondMoment targets fused.updateMask projectionStep
    2).waitChecked "one-launch pretraining projection megakernel"
  checkProjectionKernelState "pretraining megakernel" baseline fused stream

private structure GRPOScratch where
  policy : Cuda.Buffer Float32
  oldPolicy : Cuda.Buffer Float32
  reference : Cuda.Buffer Float32
  rewards : Cuda.Buffer Float32
  advantages : Cuda.Buffer Float32
  rowCoefficients : Cuda.Buffer Float32
  statistics : Cuda.Buffer Float32

private def allocateGRPOScratch (stream : Cuda.Stream) : IO GRPOScratch := do
  let rows := projectionShape.rows.toNat
  let policy ← allocateF32 stream rows fun _ => 0
  let oldPolicy ← allocateF32 stream rows fun _ => 0
  let reference ← allocateF32 stream rows fun _ => 0
  let rewards ← allocateF32 stream 2 fun index => if index == 0 then 1 else 0
  let advantages ← allocateF32 stream 2 fun _ => 0
  let rowCoefficients ← allocateF32 stream rows fun _ => 0
  let statistics ← allocateF32 stream 2 fun _ => 0
  return { policy, oldPolicy, reference, rewards, advantages, rowCoefficients, statistics }

@[test]
def grpoMegakernelMatchesModularOracle : IO Unit := do
  let stream ← Cuda.Stream.default
  let baseline ← allocateProjectionKernelState stream
  let fused ← allocateProjectionKernelState stream
  let baselineGRPO ← allocateGRPOScratch stream
  let fusedGRPO ← allocateGRPOScratch stream
  let targets ← Cuda.Buffer.alloc UInt32 projectionShape.rows.toUSize
  let masks ← Cuda.Buffer.alloc UInt32 projectionShape.rows.toUSize
  targets.copyFrom (uint32s #[0, 2, 0, 1]) stream
  masks.copyFrom (uint32s #[0, 1, 0, 1]) stream
  let layout : Cuda.Qwen36.SequenceLayout := { batchSize := 2, sequenceLength := 2 }
  let objective : Cuda.Qwen36.TrainingMegakernel.GRPOConfigF32 := {
    groupSize := 2
    clipEpsilon := 0.2
    klBeta := 0.1
    advantageEpsilon := 1e-6
  }

  Cuda.Qwen36.LoRA.forwardF32 stream baseline.input baseline.baseWeight baseline.adapterA
    baseline.adapterB baseline.adapterIds baseline.rankActivation baseline.logits projectionShape
  (← Cuda.Qwen36.Model.tokenLogProbabilityF32 stream baseline.logits targets masks
    baselineGRPO.policy projectionShape.rows projectionShape.outputFeatures).waitChecked
    "modular GRPO policy score"
  (← Cuda.Qwen36.Model.grpoObjectiveF32 stream baselineGRPO.policy baselineGRPO.oldPolicy
    baselineGRPO.reference baselineGRPO.rewards baselineGRPO.advantages
    baselineGRPO.rowCoefficients baselineGRPO.statistics baseline.loss masks layout
    objective.groupSize objective.clipEpsilon objective.klBeta objective.advantageEpsilon).waitChecked
    "modular GRPO objective"
  (← Cuda.Qwen36.Model.grpoGradientF32 stream baseline.logits targets masks
    baselineGRPO.rowCoefficients baseline.logitGradient layout projectionShape.outputFeatures
    objective.groupSize).waitChecked "modular GRPO logit VJP"
  Cuda.Qwen36.LoRA.backwardAdaptersF32 stream baseline.input baseline.adapterB baseline.adapterIds
    baseline.rankActivation baseline.logitGradient baseline.adapterAGradient
    baseline.adapterBGradient projectionShape
  (← Cuda.Qwen36.LoRA.adamWF32 stream projectionStep.optimizer baseline.adapterA baseline.adapterB
    baseline.adapterAGradient baseline.adapterBGradient baseline.adapterAFirstMoment
    baseline.adapterASecondMoment baseline.adapterBFirstMoment baseline.adapterBSecondMoment
    baseline.updateMask projectionShape).waitChecked "modular GRPO AdamW"

  (← Cuda.Qwen36.GRPOMegakernel.launchProjectionStepF32 stream fused.input fused.baseWeight
    fused.adapterA fused.adapterB fused.adapterIds fused.rankActivation fused.logits
    fused.rowStatistics fused.logitGradient fusedGRPO.policy fusedGRPO.oldPolicy
    fusedGRPO.reference fusedGRPO.rewards fusedGRPO.advantages fusedGRPO.rowCoefficients
    fusedGRPO.statistics fused.loss fused.adapterAGradient fused.adapterBGradient
    fused.adapterAFirstMoment fused.adapterASecondMoment fused.adapterBFirstMoment
    fused.adapterBSecondMoment targets masks fused.updateMask projectionStep objective 2).waitChecked
    "one-launch GRPO projection megakernel"
  checkProjectionKernelState "GRPO megakernel" baseline fused stream
  checkClose "GRPO megakernel policy score" (← fusedGRPO.policy.copyTo stream)
    (← baselineGRPO.policy.copyTo stream) projectionShape.rows.toNat 2e-4
  checkClose "GRPO megakernel advantages" (← fusedGRPO.advantages.copyTo stream)
    (← baselineGRPO.advantages.copyTo stream) 2 2e-4
  checkClose "GRPO megakernel row coefficients" (← fusedGRPO.rowCoefficients.copyTo stream)
    (← baselineGRPO.rowCoefficients.copyTo stream) projectionShape.rows.toNat 2e-4
  checkClose "GRPO megakernel statistics" (← fusedGRPO.statistics.copyTo stream)
    (← baselineGRPO.statistics.copyTo stream) 2 2e-4

end Qwen36TrainingGate
