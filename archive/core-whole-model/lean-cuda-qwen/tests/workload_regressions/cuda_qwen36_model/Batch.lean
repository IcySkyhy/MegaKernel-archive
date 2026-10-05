/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-!
# Qwen3.6 batch-isolation gates

Compares one flattened `batch=4, sequence=2` execution with four independent two-token
executions. Equality is byte-exact because both routes call the same numerical device bodies;
only the sequence-layout scheduling differs.
-/

namespace Qwen36BatchIsolationGate

private def pushUInt32 (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def floatData (count : Nat) (value : Nat → Float32) : ByteArray := Id.run do
  let mut bytes := ByteArray.emptyWithCapacity (count * 4)
  for index in [:count] do
    bytes := pushUInt32 bytes (value index).toBits
  return bytes

private def signal (salt index : Nat) : Float32 :=
  ((index * 7 + salt * 11) % 29 + 1).toUInt32.toFloat32 / 64

private def sliceElements (bytes : ByteArray) (start count : Nat) : ByteArray :=
  bytes.extract (start * 4) ((start + count) * 4)

private def uploadF32 (bytes : ByteArray) (stream : Cuda.Stream) :
    IO (Cuda.Buffer Float32) := do
  let buffer ← Cuda.Buffer.alloc Float32 (bytes.size / 4).toUSize
  buffer.copyFrom bytes stream
  return buffer

private def zerosF32 (count : Nat) (stream : Cuda.Stream) : IO (Cuda.Buffer Float32) :=
  uploadF32 (List.replicate (count * 4) 0 |>.toByteArray) stream

private def assertBufferExact (label : String) (buffer : Cuda.Buffer Float32)
    (expected : ByteArray) (stream : Cuda.Stream) : IO Unit := do
  LeanTest.assertEqual (← buffer.copyTo stream) expected label

@[test]
def attentionForwardAndVjpAreSequenceIsolated : IO Unit := do
  let batch : Nat := 4
  let sequence : Nat := 2
  let queryHeads : Nat := 2
  let keyValueHeads : Nat := 1
  let width : Nat := 4
  let queryPerBatch := sequence * queryHeads * width
  let keyValuePerBatch := sequence * keyValueHeads * width
  let probabilityPerBatch := queryHeads * sequence * sequence
  let queryCount := batch * queryPerBatch
  let keyValueCount := batch * keyValuePerBatch
  let probabilityCount := batch * probabilityPerBatch
  let layout : Cuda.Qwen36.SequenceLayout := {
    batchSize := batch.toUInt32
    sequenceLength := sequence.toUInt32
  }
  let scale : Float32 := 0.5
  let queryData := floatData queryCount (signal 1)
  let keyData := floatData keyValueCount (signal 2)
  let valueData := floatData keyValueCount (signal 3)
  let gateData := floatData queryCount fun index => signal 4 index - 0.25
  let outputGradientData := floatData queryCount fun index => signal 5 index - 0.125
  let stream ← Cuda.Stream.default
  let query ← uploadF32 queryData stream
  let key ← uploadF32 keyData stream
  let value ← uploadF32 valueData stream
  let gate ← uploadF32 gateData stream
  let preGate ← zerosF32 queryCount stream
  let postGate ← zerosF32 queryCount stream
  let probabilities ← zerosF32 probabilityCount stream
  Cuda.Qwen36.Attention.submitCausalBatchedForwardF32 (.sequential stream)
    "batch-isolated attention forward" query key value gate preGate postGate probabilities layout
    queryHeads.toUInt32 keyValueHeads.toUInt32 width.toUInt32 scale

  let outputGradient ← uploadF32 outputGradientData stream
  let probabilityGradient ← zerosF32 probabilityCount stream
  let scoreGradient ← zerosF32 probabilityCount stream
  let queryGradient ← zerosF32 queryCount stream
  let keyGradient ← zerosF32 keyValueCount stream
  let valueGradient ← zerosF32 keyValueCount stream
  Cuda.Qwen36.Attention.submitProbabilityBatchedGradientF32 (.sequential stream)
    "batch-isolated probability VJP" outputGradient value probabilityGradient layout
    queryHeads.toUInt32 keyValueHeads.toUInt32 width.toUInt32
  Cuda.Qwen36.Attention.submitScoreBatchedGradientF32 (.sequential stream)
    "batch-isolated score VJP" probabilities probabilityGradient scoreGradient layout
    queryHeads.toUInt32 keyValueHeads.toUInt32 width.toUInt32 scale
  Cuda.Qwen36.Attention.submitQueryBatchedGradientF32 (.sequential stream)
    "batch-isolated query VJP" scoreGradient key queryGradient layout queryHeads.toUInt32
    keyValueHeads.toUInt32 width.toUInt32
  Cuda.Qwen36.Attention.submitKeyValueBatchedGradientF32 (.sequential stream)
    "batch-isolated key/value VJP" scoreGradient probabilities query outputGradient keyGradient
    valueGradient layout queryHeads.toUInt32 keyValueHeads.toUInt32 width.toUInt32

  let mut expectedPreGate := ByteArray.empty
  let mut expectedPostGate := ByteArray.empty
  let mut expectedProbabilities := ByteArray.empty
  let mut expectedProbabilityGradient := ByteArray.empty
  let mut expectedScoreGradient := ByteArray.empty
  let mut expectedQueryGradient := ByteArray.empty
  let mut expectedKeyGradient := ByteArray.empty
  let mut expectedValueGradient := ByteArray.empty
  for batchIndex in [:batch] do
    let localQuery ← uploadF32
      (sliceElements queryData (batchIndex * queryPerBatch) queryPerBatch) stream
    let localKey ← uploadF32
      (sliceElements keyData (batchIndex * keyValuePerBatch) keyValuePerBatch) stream
    let localValue ← uploadF32
      (sliceElements valueData (batchIndex * keyValuePerBatch) keyValuePerBatch) stream
    let localGate ← uploadF32
      (sliceElements gateData (batchIndex * queryPerBatch) queryPerBatch) stream
    let localOutputGradient ← uploadF32
      (sliceElements outputGradientData (batchIndex * queryPerBatch) queryPerBatch) stream
    let localPreGate ← zerosF32 queryPerBatch stream
    let localPostGate ← zerosF32 queryPerBatch stream
    let localProbabilities ← zerosF32 probabilityPerBatch stream
    (← Cuda.Qwen36.Attention.causalForwardF32 stream localQuery localKey localValue localGate
      localPreGate localPostGate localProbabilities sequence.toUInt32 queryHeads.toUInt32
      keyValueHeads.toUInt32 width.toUInt32 scale).waitChecked "single-sequence attention forward"
    let localProbabilityGradient ← zerosF32 probabilityPerBatch stream
    let localScoreGradient ← zerosF32 probabilityPerBatch stream
    let localQueryGradient ← zerosF32 queryPerBatch stream
    let localKeyGradient ← zerosF32 keyValuePerBatch stream
    let localValueGradient ← zerosF32 keyValuePerBatch stream
    (← Cuda.Qwen36.Attention.probabilityGradientF32 stream localOutputGradient localValue
      localProbabilityGradient sequence.toUInt32 queryHeads.toUInt32 keyValueHeads.toUInt32
      width.toUInt32).waitChecked "single-sequence probability VJP"
    (← Cuda.Qwen36.Attention.scoreGradientF32 stream localProbabilities
      localProbabilityGradient localScoreGradient sequence.toUInt32 queryHeads.toUInt32
      keyValueHeads.toUInt32 width.toUInt32 scale).waitChecked "single-sequence score VJP"
    (← Cuda.Qwen36.Attention.queryGradientF32 stream localScoreGradient localKey
      localQueryGradient sequence.toUInt32 queryHeads.toUInt32 keyValueHeads.toUInt32
      width.toUInt32).waitChecked "single-sequence query VJP"
    (← Cuda.Qwen36.Attention.keyValueGradientF32 stream localScoreGradient localProbabilities
      localQuery localOutputGradient localKeyGradient localValueGradient sequence.toUInt32
      queryHeads.toUInt32 keyValueHeads.toUInt32 width.toUInt32).waitChecked
        "single-sequence key/value VJP"
    expectedPreGate := expectedPreGate ++ (← localPreGate.copyTo stream)
    expectedPostGate := expectedPostGate ++ (← localPostGate.copyTo stream)
    expectedProbabilities := expectedProbabilities ++ (← localProbabilities.copyTo stream)
    expectedProbabilityGradient :=
      expectedProbabilityGradient ++ (← localProbabilityGradient.copyTo stream)
    expectedScoreGradient := expectedScoreGradient ++ (← localScoreGradient.copyTo stream)
    expectedQueryGradient := expectedQueryGradient ++ (← localQueryGradient.copyTo stream)
    expectedKeyGradient := expectedKeyGradient ++ (← localKeyGradient.copyTo stream)
    expectedValueGradient := expectedValueGradient ++ (← localValueGradient.copyTo stream)
  assertBufferExact "batched attention pre-gate differs" preGate expectedPreGate stream
  assertBufferExact "batched attention post-gate differs" postGate expectedPostGate stream
  assertBufferExact "batched attention probabilities differ" probabilities expectedProbabilities stream
  assertBufferExact "batched attention probability VJP differs" probabilityGradient
    expectedProbabilityGradient stream
  assertBufferExact "batched attention score VJP differs" scoreGradient expectedScoreGradient stream
  assertBufferExact "batched attention query VJP differs" queryGradient expectedQueryGradient stream
  assertBufferExact "batched attention key VJP differs" keyGradient expectedKeyGradient stream
  assertBufferExact "batched attention value VJP differs" valueGradient expectedValueGradient stream

@[test]
def deltaNetRecurrenceAndVjpAreSequenceIsolated : IO Unit := do
  let batch : Nat := 4
  let sequence : Nat := 2
  let keyHeads : Nat := 1
  let valueHeads : Nat := 2
  let keyWidth : Nat := 2
  let valueWidth : Nat := 3
  let keyPerBatch := sequence * valueHeads * keyWidth
  let valuePerBatch := sequence * valueHeads * valueWidth
  let gatePerBatch := sequence * valueHeads
  let statePerBatch := valueHeads * keyWidth * valueWidth
  let historyPerBatch := (sequence + 1) * statePerBatch
  let keyCount := batch * keyPerBatch
  let valueCount := batch * valuePerBatch
  let gateCount := batch * gatePerBatch
  let stateCount := batch * statePerBatch
  let historyCount := batch * historyPerBatch
  let layout : Cuda.Qwen36.SequenceLayout := {
    batchSize := batch.toUInt32
    sequenceLength := sequence.toUInt32
  }
  let queryScale : Float32 := 0.5
  let queryData := floatData keyCount fun index => signal 6 index - 0.2
  let keyData := floatData keyCount fun index => signal 7 index - 0.2
  let valueData := floatData valueCount fun index => signal 8 index - 0.2
  let decayData := floatData gateCount fun index => -signal 9 index
  let betaData := floatData gateCount (signal 10)
  let stateInputData := floatData stateCount fun index => signal 11 index - 0.2
  let outputGradientData := floatData valueCount fun index => signal 12 index - 0.2
  let stream ← Cuda.Stream.default
  let query ← uploadF32 queryData stream
  let key ← uploadF32 keyData stream
  let value ← uploadF32 valueData stream
  let decayLog ← uploadF32 decayData stream
  let beta ← uploadF32 betaData stream
  let stateInput ← uploadF32 stateInputData stream
  let output ← zerosF32 valueCount stream
  let stateOutput ← zerosF32 stateCount stream
  let stateHistory ← zerosF32 historyCount stream
  Cuda.Qwen36.DeltaNet.submitRecurrentNormalizedBatchedPrefillF32 (.sequential stream)
    "batch-isolated DeltaNet recurrence" query key value decayLog beta stateInput output
    stateOutput stateHistory layout keyHeads.toUInt32 valueHeads.toUInt32 keyWidth.toUInt32
    valueWidth.toUInt32 queryScale

  let outputGradient ← uploadF32 outputGradientData stream
  let queryGradient ← zerosF32 keyCount stream
  let keyGradient ← zerosF32 keyCount stream
  let valueGradient ← zerosF32 valueCount stream
  let decayGradient ← zerosF32 gateCount stream
  let betaGradient ← zerosF32 gateCount stream
  let stateGradient ← zerosF32 stateCount stream
  Cuda.Qwen36.DeltaNet.submitRecurrentNormalizedBatchedBackwardF32 (.sequential stream)
    "batch-isolated DeltaNet recurrence VJP" query key value decayLog beta stateHistory
    outputGradient queryGradient keyGradient valueGradient decayGradient betaGradient stateGradient
    layout keyHeads.toUInt32 valueHeads.toUInt32 keyWidth.toUInt32 valueWidth.toUInt32 queryScale

  let mut expectedOutput := ByteArray.empty
  let mut expectedStateOutput := ByteArray.empty
  let mut expectedStateHistory := ByteArray.empty
  let mut expectedQueryGradient := ByteArray.empty
  let mut expectedKeyGradient := ByteArray.empty
  let mut expectedValueGradient := ByteArray.empty
  let mut expectedDecayGradient := ByteArray.empty
  let mut expectedBetaGradient := ByteArray.empty
  let mut expectedStateGradient := ByteArray.empty
  for batchIndex in [:batch] do
    let localQuery ← uploadF32
      (sliceElements queryData (batchIndex * keyPerBatch) keyPerBatch) stream
    let localKey ← uploadF32
      (sliceElements keyData (batchIndex * keyPerBatch) keyPerBatch) stream
    let localValue ← uploadF32
      (sliceElements valueData (batchIndex * valuePerBatch) valuePerBatch) stream
    let localDecay ← uploadF32
      (sliceElements decayData (batchIndex * gatePerBatch) gatePerBatch) stream
    let localBeta ← uploadF32
      (sliceElements betaData (batchIndex * gatePerBatch) gatePerBatch) stream
    let localStateInput ← uploadF32
      (sliceElements stateInputData (batchIndex * statePerBatch) statePerBatch) stream
    let localOutputGradient ← uploadF32
      (sliceElements outputGradientData (batchIndex * valuePerBatch) valuePerBatch) stream
    let localOutput ← zerosF32 valuePerBatch stream
    let localStateOutput ← zerosF32 statePerBatch stream
    let localStateHistory ← zerosF32 historyPerBatch stream
    (← Cuda.Qwen36.DeltaNet.recurrentNormalizedPrefillF32 stream localQuery localKey localValue
      localDecay localBeta localStateInput localOutput localStateOutput localStateHistory
      sequence.toUInt32 keyHeads.toUInt32 valueHeads.toUInt32 keyWidth.toUInt32
      valueWidth.toUInt32 queryScale).waitChecked "single-sequence DeltaNet recurrence"
    let localQueryGradient ← zerosF32 keyPerBatch stream
    let localKeyGradient ← zerosF32 keyPerBatch stream
    let localValueGradient ← zerosF32 valuePerBatch stream
    let localDecayGradient ← zerosF32 gatePerBatch stream
    let localBetaGradient ← zerosF32 gatePerBatch stream
    let localStateGradient ← zerosF32 statePerBatch stream
    (← Cuda.Qwen36.DeltaNet.recurrentNormalizedBackwardF32 stream localQuery localKey localValue
      localDecay localBeta localStateHistory localOutputGradient localQueryGradient
      localKeyGradient localValueGradient localDecayGradient localBetaGradient localStateGradient
      sequence.toUInt32 keyHeads.toUInt32 valueHeads.toUInt32 keyWidth.toUInt32
      valueWidth.toUInt32 queryScale).waitChecked "single-sequence DeltaNet recurrence VJP"
    expectedOutput := expectedOutput ++ (← localOutput.copyTo stream)
    expectedStateOutput := expectedStateOutput ++ (← localStateOutput.copyTo stream)
    expectedStateHistory := expectedStateHistory ++ (← localStateHistory.copyTo stream)
    expectedQueryGradient := expectedQueryGradient ++ (← localQueryGradient.copyTo stream)
    expectedKeyGradient := expectedKeyGradient ++ (← localKeyGradient.copyTo stream)
    expectedValueGradient := expectedValueGradient ++ (← localValueGradient.copyTo stream)
    expectedDecayGradient := expectedDecayGradient ++ (← localDecayGradient.copyTo stream)
    expectedBetaGradient := expectedBetaGradient ++ (← localBetaGradient.copyTo stream)
    expectedStateGradient := expectedStateGradient ++ (← localStateGradient.copyTo stream)
  assertBufferExact "batched DeltaNet output differs" output expectedOutput stream
  assertBufferExact "batched DeltaNet final state differs" stateOutput expectedStateOutput stream
  assertBufferExact "batched DeltaNet state history differs" stateHistory expectedStateHistory stream
  assertBufferExact "batched DeltaNet query VJP differs" queryGradient expectedQueryGradient stream
  assertBufferExact "batched DeltaNet key VJP differs" keyGradient expectedKeyGradient stream
  assertBufferExact "batched DeltaNet value VJP differs" valueGradient expectedValueGradient stream
  assertBufferExact "batched DeltaNet decay VJP differs" decayGradient expectedDecayGradient stream
  assertBufferExact "batched DeltaNet beta VJP differs" betaGradient expectedBetaGradient stream
  assertBufferExact "batched DeltaNet initial-state VJP differs" stateGradient
    expectedStateGradient stream

end Qwen36BatchIsolationGate
