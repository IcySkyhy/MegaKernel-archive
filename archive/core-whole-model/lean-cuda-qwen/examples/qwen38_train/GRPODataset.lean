/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

public section

/-! # Fixed-shape grouped Qwen rollout dataset for outcome-GRPO -/

namespace Qwen38Train.GRPODataset

private def magic : ByteArray := "LCQGRP2".toUTF8 |>.push 0
private def version : UInt32 := 2
private def headerWords : Nat := 6
private def policyWords : Nat := 8
private def vocabulary : UInt32 := 248320

private def readWord (bytes : ByteArray) (index : Nat) : UInt32 :=
  let offset := index * 4
  bytes[offset]!.toUInt32 ||| (bytes[offset + 1]!.toUInt32 <<< 8) |||
    (bytes[offset + 2]!.toUInt32 <<< 16) ||| (bytes[offset + 3]!.toUInt32 <<< 24)

private def finiteFloat32 (value : Float32) : Bool :=
  value.toBits &&& 0x7f800000 != 0x7f800000

/-- One or more complete prompt groups flattened in stable group/member order. -/
structure Batch where
  tokens : Array (Array UInt32)
  targets : Array (Array UInt32)
  masks : Array (Array UInt32)
  behaviorLogProbabilities : Array (Array Float32)
  behaviorPolicySHA256 : Array ByteArray
  rewards : Array Float32

/-- Wrapped cursor over a checked `LCQGRP2` file. -/
structure Batcher where
  bytes : ByteArray
  groupCount : Nat
  groupSize : Nat
  sequenceLength : Nat
  cursor : Nat := 0
  epoch : Nat := 0

private def sequenceWords (sequenceLength : Nat) : Nat :=
  3 * sequenceLength + 2

private def groupWords (sequenceLength groupSize : Nat) : Nat :=
  policyWords + groupSize * sequenceWords sequenceLength

private def sequenceBase (batcher : Batcher) (group member : Nat) : Nat :=
  headerWords + group * groupWords batcher.sequenceLength batcher.groupSize +
    policyWords + member * sequenceWords batcher.sequenceLength

private def policyDigest (batcher : Batcher) (group : Nat) : ByteArray :=
  let base := (headerWords + group * groupWords batcher.sequenceLength batcher.groupSize) * 4
  batcher.bytes.extract base (base + 32)

private def readSequence (batcher : Batcher) (group member : Nat) :
    Array UInt32 × Array UInt32 × Array UInt32 × Array Float32 × Float32 := Id.run do
  let base := sequenceBase batcher group member
  let maskBase := base + batcher.sequenceLength + 1
  let oldBase := maskBase + batcher.sequenceLength
  let rewardIndex := oldBase + batcher.sequenceLength
  let mut tokens := Array.mkEmpty batcher.sequenceLength
  let mut targets := Array.mkEmpty batcher.sequenceLength
  let mut masks := Array.mkEmpty batcher.sequenceLength
  let mut behavior := Array.mkEmpty batcher.sequenceLength
  for offset in [:batcher.sequenceLength] do
    tokens := tokens.push (readWord batcher.bytes (base + offset))
    targets := targets.push (readWord batcher.bytes (base + offset + 1))
    masks := masks.push (readWord batcher.bytes (maskBase + offset))
    behavior := behavior.push (Float32.ofBits (readWord batcher.bytes (oldBase + offset)))
  return (tokens, targets, masks, behavior,
    Float32.ofBits (readWord batcher.bytes rewardIndex))

private def checkMask (path : System.FilePath) (group member : Nat)
    (mask : Array UInt32) : IO Nat := do
  let mut first := mask.size
  for index in [:mask.size] do
    match mask[index]! with
    | 0 => pure ()
    | 1 =>
        if first == mask.size then
          first := index
    | value =>
        throw <| IO.userError
          s!"{path}: group {group} member {member} has invalid mask value {value}"
  if first == mask.size then
    throw <| IO.userError s!"{path}: group {group} member {member} has an empty action mask"
  return first

private def checkBehaviorLogProbabilities (path : System.FilePath) (group member : Nat)
    (mask : Array UInt32) (behavior : Array Float32) : IO Unit := do
  for index in [:behavior.size] do
    unless finiteFloat32 behavior[index]! do
      throw <| IO.userError
        s!"{path}: group {group} member {member} behavior log probability {index} is not finite"
    if behavior[index]! > 0 then
      throw <| IO.userError
        s!"{path}: group {group} member {member} behavior log probability {index} is positive"
    if mask[index]! == 0 && behavior[index]! != 0 then
      throw <| IO.userError
        s!"{path}: group {group} member {member} masked behavior log probability {index} is nonzero"

/-- Read and completely validate one fixed-shape grouped rollout file. -/
def load (path : System.FilePath) (expectedSequenceLength expectedGroupSize : Nat) : IO Batcher := do
  let bytes ← IO.FS.readBinFile path
  unless bytes.size ≥ 24 && bytes.size % 4 == 0 && bytes.extract 0 magic.size == magic do
    throw <| IO.userError s!"{path}: invalid LCQGRP2 rollout dataset header"
  let fileVersion := readWord bytes 2
  unless fileVersion == version do
    throw <| IO.userError s!"{path}: unsupported LCQGRP2 version {fileVersion}"
  let sequenceLength := (readWord bytes 3).toNat
  let groupSize := (readWord bytes 4).toNat
  let groupCount := (readWord bytes 5).toNat
  unless sequenceLength == expectedSequenceLength do
    throw <| IO.userError
      s!"{path}: sequence length {sequenceLength} does not match {expectedSequenceLength}"
  unless groupSize == expectedGroupSize do
    throw <| IO.userError s!"{path}: group size {groupSize} does not match {expectedGroupSize}"
  unless groupSize ≥ 2 do
    throw <| IO.userError s!"{path}: GRPO group size must be at least two"
  unless groupCount > 0 do
    throw <| IO.userError s!"{path}: grouped rollout dataset is empty"
  let expectedBytes :=
    (headerWords + groupCount * groupWords sequenceLength groupSize) * 4
  unless bytes.size == expectedBytes do
    throw <| IO.userError
      s!"{path}: rollout dataset size {bytes.size} does not match header {expectedBytes}"
  let batcher : Batcher := { bytes, groupCount, groupSize, sequenceLength }
  for group in [:groupCount] do
    let (firstTokens, firstTargets, firstMask, firstBehavior, firstReward) :=
      readSequence batcher group 0
    checkBehaviorLogProbabilities path group 0 firstMask firstBehavior
    let firstStart ← checkMask path group 0 firstMask
    for token in firstTokens ++ firstTargets do
      unless token < vocabulary do
        throw <| IO.userError s!"{path}: group {group} member 0 token is outside the vocabulary"
    unless finiteFloat32 firstReward do
      throw <| IO.userError s!"{path}: group {group} member 0 reward is not finite"
    for member in [1:groupSize] do
      let (tokens, targets, mask, behavior, reward) := readSequence batcher group member
      checkBehaviorLogProbabilities path group member mask behavior
      for token in tokens ++ targets do
        unless token < vocabulary do
          throw <| IO.userError
            s!"{path}: group {group} member {member} token is outside the vocabulary"
      unless finiteFloat32 reward do
        throw <| IO.userError
          s!"{path}: group {group} member {member} reward is not finite"
      let start ← checkMask path group member mask
      unless start == firstStart do
        throw <| IO.userError s!"{path}: group {group} prompt lengths differ"
      for index in [:firstStart + 1] do
        unless tokens[index]! == firstTokens[index]! do
          throw <| IO.userError s!"{path}: group {group} prompts differ"
  return batcher

/-- Draw complete groups and flatten them in adjacent group/member sequence order. -/
def Batcher.next (batcher : Batcher) (groupBatchSize : Nat) : Batcher × Batch := Id.run do
  let mut current := batcher
  let sequenceCount := groupBatchSize * batcher.groupSize
  let mut tokens := Array.mkEmpty sequenceCount
  let mut targets := Array.mkEmpty sequenceCount
  let mut masks := Array.mkEmpty sequenceCount
  let mut behaviorLogProbabilities := Array.mkEmpty sequenceCount
  let mut behaviorPolicySHA256 := Array.mkEmpty groupBatchSize
  let mut rewards := Array.mkEmpty sequenceCount
  for _ in [:groupBatchSize] do
    if current.cursor == current.groupCount then
      current := { current with cursor := 0, epoch := current.epoch + 1 }
    behaviorPolicySHA256 := behaviorPolicySHA256.push (policyDigest current current.cursor)
    for member in [:current.groupSize] do
      let (sequenceTokens, sequenceTargets, sequenceMask, behavior, reward) :=
        readSequence current current.cursor member
      tokens := tokens.push sequenceTokens
      targets := targets.push sequenceTargets
      masks := masks.push sequenceMask
      behaviorLogProbabilities := behaviorLogProbabilities.push behavior
      rewards := rewards.push reward
    current := { current with cursor := current.cursor + 1 }
  return (current,
    { tokens, targets, masks, behaviorLogProbabilities, behaviorPolicySHA256, rewards })

end Qwen38Train.GRPODataset
