/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

public section

/-! # Fixed-shape Qwen DPO preference dataset -/

namespace Qwen38Train.PreferenceDataset

private def magic : ByteArray := "LCQDPO1".toUTF8 |>.push 0
private def version : UInt32 := 1
private def headerWords : Nat := 5
private def vocabulary : UInt32 := 248320

private def readWord (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

/-- One ordered chosen/rejected minibatch ready for the projection-wide DPO trainer. -/
structure Batch where
  tokens : Array (Array UInt32)
  targets : Array (Array UInt32)
  masks : Array (Array UInt32)

/-- Wrapped cursor over a checked `LCQDPO1` file. -/
structure Batcher where
  bytes : ByteArray
  pairCount : Nat
  sequenceLength : Nat
  cursor : Nat := 0
  epoch : Nat := 0

private def roleWords (sequenceLength : Nat) : Nat :=
  2 * sequenceLength + 1

private def recordWords (sequenceLength : Nat) : Nat :=
  2 * roleWords sequenceLength

private def sequenceBase (sequenceLength record role : Nat) : Nat :=
  headerWords + record * recordWords sequenceLength + role * roleWords sequenceLength

private def readSequence (batcher : Batcher) (record role : Nat) :
    Array UInt32 × Array UInt32 × Array UInt32 := Id.run do
  let base := sequenceBase batcher.sequenceLength record role
  let maskBase := base + batcher.sequenceLength + 1
  let mut tokens := Array.mkEmpty batcher.sequenceLength
  let mut targets := Array.mkEmpty batcher.sequenceLength
  let mut masks := Array.mkEmpty batcher.sequenceLength
  for offset in [:batcher.sequenceLength] do
    tokens := tokens.push (readWord batcher.bytes (base + offset))
    targets := targets.push (readWord batcher.bytes (base + offset + 1))
    masks := masks.push (readWord batcher.bytes (maskBase + offset))
  return (tokens, targets, masks)

private def checkMask (path : System.FilePath) (record role : Nat)
    (mask : Array UInt32) : IO Nat := do
  let mut first := mask.size
  let mut seenCompletion := false
  let mut seenPadding := false
  for index in [:mask.size] do
    match mask[index]! with
    | 0 =>
        if seenCompletion then
          seenPadding := true
    | 1 =>
        if seenPadding then
          throw <| IO.userError
            s!"{path}: record {record} role {role} mask is not one contiguous completion"
        if !seenCompletion then
          first := index
        seenCompletion := true
    | value =>
        throw <| IO.userError
          s!"{path}: record {record} role {role} has invalid mask value {value}"
  unless seenCompletion do
    throw <| IO.userError s!"{path}: record {record} role {role} has an empty completion"
  return first

/-- Read and completely validate one fixed-shape preference file. -/
def load (path : System.FilePath) (expectedSequenceLength : Nat) : IO Batcher := do
  let bytes ← IO.FS.readBinFile path
  unless bytes.size ≥ 20 && bytes.size % 4 == 0 && bytes.extract 0 magic.size == magic do
    throw <| IO.userError s!"{path}: invalid LCQDPO1 preference dataset header"
  let fileVersion := readWord bytes 2
  unless fileVersion == version do
    throw <| IO.userError s!"{path}: unsupported LCQDPO1 version {fileVersion}"
  let sequenceLength := (readWord bytes 3).toNat
  let pairCount := (readWord bytes 4).toNat
  unless sequenceLength == expectedSequenceLength do
    throw <| IO.userError
      s!"{path}: sequence length {sequenceLength} does not match {expectedSequenceLength}"
  unless pairCount > 0 do
    throw <| IO.userError s!"{path}: preference dataset is empty"
  let expectedBytes := (headerWords + pairCount * recordWords sequenceLength) * 4
  unless bytes.size == expectedBytes do
    throw <| IO.userError
      s!"{path}: preference dataset size {bytes.size} does not match header {expectedBytes}"
  let batcher : Batcher := { bytes, pairCount, sequenceLength }
  for record in [:pairCount] do
    let (chosenTokens, chosenTargets, chosenMask) := readSequence batcher record 0
    let (rejectedTokens, rejectedTargets, rejectedMask) := readSequence batcher record 1
    for token in chosenTokens ++ chosenTargets ++ rejectedTokens ++ rejectedTargets do
      unless token < vocabulary do
        throw <| IO.userError s!"{path}: record {record} token is outside the vocabulary"
    let chosenStart ← checkMask path record 0 chosenMask
    let rejectedStart ← checkMask path record 1 rejectedMask
    unless chosenStart == rejectedStart do
      throw <| IO.userError s!"{path}: record {record} chosen/rejected prompt lengths differ"
    for index in [:chosenStart + 1] do
      unless chosenTokens[index]! == rejectedTokens[index]! do
        throw <| IO.userError s!"{path}: record {record} chosen/rejected prompts differ"
  return batcher

/-- Draw `pairBatchSize` records, flattened as adjacent chosen/rejected sequences. -/
def Batcher.next (batcher : Batcher) (pairBatchSize : Nat) : Batcher × Batch := Id.run do
  let mut current := batcher
  let mut tokens := Array.mkEmpty (pairBatchSize * 2)
  let mut targets := Array.mkEmpty (pairBatchSize * 2)
  let mut masks := Array.mkEmpty (pairBatchSize * 2)
  for _ in [:pairBatchSize] do
    if current.cursor == current.pairCount then
      current := { current with cursor := 0, epoch := current.epoch + 1 }
    for role in [:2] do
      let (sequenceTokens, sequenceTargets, sequenceMask) :=
        readSequence current current.cursor role
      tokens := tokens.push sequenceTokens
      targets := targets.push sequenceTargets
      masks := masks.push sequenceMask
    current := { current with cursor := current.cursor + 1 }
  return (current, { tokens, targets, masks })

end Qwen38Train.PreferenceDataset
