/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Training

public section

/-!
# Model-wide Qwen3.6 training megakernel core

This module lowers `Training.FullModelStepF32` into a device-resident tape of POD UInt32 records.
One cooperative grid interprets the complete embedding, hybrid decoder, language-model head, exact
reverse pass, all projection-local LoRA reductions, and all masked AdamW updates. Frozen checkpoint
matrices, Float32 activations, and UInt32 routing buffers live in separately typed pointer tables;
no raw device address is encoded in a POD value.

The objective is intentionally absent from this module. Dedicated pretraining and GRPO entry
programs execute the shared forward prefix, their compile-time objective body, the shared reverse
suffix, and the shared optimizer publication in one launch.
-/

namespace Cuda.Qwen36.FullTrainingMegakernel

open Cuda.Qwen36.Primitives

/-- Fixed record width. A deliberately roomy record keeps decoding branch-free within an op and
leaves room for recurrent reverse descriptors without an auxiliary indirection. -/
def operationWords : UInt32 := 32

namespace Op

def embedding : UInt32 := 1
def projectionForward : UInt32 := 2
def rmsForward : UInt32 := 3
def add : UInt32 := 4
def swigluForward : UInt32 := 5
def convolutionForward : UInt32 := 6
def splitQkv : UInt32 := 7
def repeatCompact : UInt32 := 8
def l2Forward : UInt32 := 9
def deltaGateForward : UInt32 := 10
def recurrenceForward : UInt32 := 11
def weightedGatedRmsForward : UInt32 := 12
def splitQueryGate : UInt32 := 13
def ropeForward : UInt32 := 14
def attentionForward : UInt32 := 15

def projectionBackward : UInt32 := 20
def rmsBackward : UInt32 := 21
def swigluBackward : UInt32 := 22
def weightedGatedRmsBackward : UInt32 := 23
def recurrenceBackward : UInt32 := 24
def l2Backward : UInt32 := 25
def reduceRepeated : UInt32 := 26
def concatenateQkv : UInt32 := 27
def convolutionBackward : UInt32 := 28
def deltaGateBackward : UInt32 := 29
def attentionGateBackward : UInt32 := 30
def probabilityBackward : UInt32 := 31
def scoreBackward : UInt32 := 32
def queryBackward : UInt32 := 33
def keyValueBackward : UInt32 := 34
def ropeBackward : UInt32 := 35
def mergeQueryGate : UInt32 := 36

end Op

/-- POD launch metadata shared by both objective-specific kernels. -/
structure ProgramDescriptor where
  forwardOperations : UInt32
  operationCount : UInt32
  projectionCount : UInt32
  adapterIndexBase : UInt32
  deriving Cuda.POD

/-- Owning host-side result of lowering one complete LoRA full-model step. -/
structure Program where
  operations : Cuda.Buffer UInt32
  activations : Cuda.BufferTable Float32
  adapters : Cuda.BufferTable Float32
  matrices : Cuda.BufferTable UInt8
  indices : Cuda.BufferTable UInt32
  projectionMetadata : Cuda.Buffer UInt32
  descriptor : ProgramDescriptor
  rows : UInt32
  classes : UInt32
  blocks : UInt32
  sharedMemoryBytes : USize
  logits : Cuda.Buffer Float32
  logitGradient : Cuda.Buffer Float32
  rowLoss : Cuda.Buffer Float32
  loss : Cuda.Buffer Float32
  targets : Cuda.Buffer UInt32

/-- One projection's shape plus its optimizer decay, reconstructed from eight UInt32 words. -/
@[struct] structure ProjectionMetadata where
  shape : Cuda.Qwen36.LoRA.Shape
  weightDecay : Float32

namespace Internal

def adapterFields : UInt32 := 9
def adapterIndexFields : UInt32 := 2
def projectionMetadataWords : UInt32 := 8

def adapterAField : UInt32 := 0
def adapterBField : UInt32 := 1
def rankActivationField : UInt32 := 2
def adapterAGradientField : UInt32 := 3
def adapterBGradientField : UInt32 := 4
def adapterAFirstMomentField : UInt32 := 5
def adapterASecondMomentField : UInt32 := 6
def adapterBFirstMomentField : UInt32 := 7
def adapterBSecondMomentField : UInt32 := 8

@[expose, cuda_device, always_inline]
def word (operations : Cuda.DevicePtr UInt32) (operation offset : UInt32) :
    Cuda.DeviceM UInt32 :=
  Cuda.loadReadOnlyUInt32 operations (operation * operationWords + offset).toUSize

@[expose, cuda_device, always_inline]
def activation (table : Cuda.DevicePtrTable Float32) (index : UInt32) :
    Cuda.DeviceM (Cuda.DevicePtr Float32) :=
  table.get index.toUSize

@[expose, cuda_device, always_inline]
def indexBuffer (table : Cuda.DevicePtrTable UInt32) (index : UInt32) :
    Cuda.DeviceM (Cuda.DevicePtr UInt32) :=
  table.get index.toUSize

@[expose, cuda_device, always_inline]
def adapterBuffer (table : Cuda.DevicePtrTable Float32) (projection field : UInt32) :
    Cuda.DeviceM (Cuda.DevicePtr Float32) :=
  table.get (projection * adapterFields + field).toUSize

@[expose, cuda_device, always_inline]
def adapterIndex (table : Cuda.DevicePtrTable UInt32) (base projection field : UInt32) :
    Cuda.DeviceM (Cuda.DevicePtr UInt32) :=
  table.get (base + projection * adapterIndexFields + field).toUSize

@[expose, cuda_device, always_inline]
def matrix (table : Cuda.DevicePtrTable UInt8) (projection : UInt32) :
    Cuda.DeviceM (Cuda.DevicePtr Cuda.BFloat16) := do
  let pointer ← table.get (projection + 1).toUSize
  return pointer

@[expose, cuda_device, always_inline]
def loadProjectionMetadata (metadata : Cuda.DevicePtr UInt32) (projection : UInt32) :
    Cuda.DeviceM ProjectionMetadata := do
  let base := projection * projectionMetadataWords
  return {
    shape := {
      rows := ← Cuda.loadReadOnlyUInt32 metadata base.toUSize
      sequenceLength := ← Cuda.loadReadOnlyUInt32 metadata (base + 1).toUSize
      adapterCount := ← Cuda.loadReadOnlyUInt32 metadata (base + 2).toUSize
      rank := ← Cuda.loadReadOnlyUInt32 metadata (base + 3).toUSize
      inputFeatures := ← Cuda.loadReadOnlyUInt32 metadata (base + 4).toUSize
      outputFeatures := ← Cuda.loadReadOnlyUInt32 metadata (base + 5).toUSize
      scale := Float32.ofBits (← Cuda.loadReadOnlyUInt32 metadata (base + 6).toUSize)
    }
    weightDecay := Float32.ofBits (← Cuda.loadReadOnlyUInt32 metadata (base + 7).toUSize)
  }

@[always_inline]
partial def projectionRankLoop (activations : Cuda.DevicePtrTable Float32)
    (adapters : Cuda.DevicePtrTable Float32) (indices : Cuda.DevicePtrTable UInt32)
    (operations : Cuda.DevicePtr UInt32) (descriptor : ProgramDescriptor)
    (metadata : ProjectionMetadata) (operation linear stride : UInt32) : Cuda.DeviceM Unit := do
  if linear < metadata.shape.rows * metadata.shape.rank then
    let projection ← word operations operation 1
    let input ← activation activations (← word operations operation 2)
    let adapterA ← adapterBuffer adapters projection adapterAField
    let adapterIds ← adapterIndex indices descriptor.adapterIndexBase projection 0
    let rankActivation ← adapterBuffer adapters projection rankActivationField
    Cuda.Qwen36.LoRA.rankForwardElementF32 input adapterA adapterIds rankActivation metadata.shape
      linear
    projectionRankLoop activations adapters indices operations descriptor metadata operation
      (linear + stride) stride

@[always_inline]
partial def projectionForwardLoop (activations : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (adapters : Cuda.DevicePtrTable Float32)
    (indices : Cuda.DevicePtrTable UInt32) (operations : Cuda.DevicePtr UInt32)
    (descriptor : ProgramDescriptor) (metadata : ProjectionMetadata)
    (operation linear stride : UInt32) : Cuda.DeviceM Unit := do
  if linear < metadata.shape.rows * metadata.shape.outputFeatures then
    let projection ← word operations operation 1
    let input ← activation activations (← word operations operation 2)
    let output ← activation activations (← word operations operation 3)
    Cuda.Qwen36.Linear.forwardFrozenBF16ElementF32 input (← matrix matrices projection) output
      { rows := metadata.shape.rows, inputFeatures := metadata.shape.inputFeatures,
        outputFeatures := metadata.shape.outputFeatures } linear
    let adapterB ← adapterBuffer adapters projection adapterBField
    let adapterIds ← adapterIndex indices descriptor.adapterIndexBase projection 0
    let rankActivation ← adapterBuffer adapters projection rankActivationField
    Cuda.Qwen36.LoRA.addForwardElementF32 adapterB adapterIds rankActivation output metadata.shape
      linear
    projectionForwardLoop activations matrices adapters indices operations descriptor metadata
      operation (linear + stride) stride

@[always_inline]
partial def projectionBaseBackwardLoop (activations : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (operations : Cuda.DevicePtr UInt32)
    (metadata : ProjectionMetadata) (operation linear stride : UInt32) : Cuda.DeviceM Unit := do
  if linear < metadata.shape.rows * metadata.shape.inputFeatures then
    let projection ← word operations operation 1
    let outputGradient ← activation activations (← word operations operation 3)
    let inputGradient ← activation activations (← word operations operation 4)
    Cuda.Qwen36.Linear.backwardInputFrozenBF16ElementF32 outputGradient
      (← matrix matrices projection) inputGradient {
        rows := metadata.shape.rows, inputFeatures := metadata.shape.inputFeatures,
        outputFeatures := metadata.shape.outputFeatures
      } linear
    projectionBaseBackwardLoop activations matrices operations metadata operation
      (linear + stride) stride

@[always_inline]
partial def projectionAdapterBackwardLoop (activations : Cuda.DevicePtrTable Float32)
    (adapters : Cuda.DevicePtrTable Float32) (indices : Cuda.DevicePtrTable UInt32)
    (operations : Cuda.DevicePtr UInt32) (descriptor : ProgramDescriptor)
    (metadata : ProjectionMetadata) (operation linear stride : UInt32) : Cuda.DeviceM Unit := do
  let shape := metadata.shape
  let inputCount := shape.rows * shape.inputFeatures
  let aCount := shape.adapterCount * shape.rank * shape.inputFeatures
  let bCount := shape.adapterCount * shape.outputFeatures * shape.rank
  if linear < max inputCount (max aCount bCount) then
    let projection ← word operations operation 1
    let input ← activation activations (← word operations operation 2)
    let outputGradient ← activation activations (← word operations operation 3)
    let inputGradient ← activation activations (← word operations operation 4)
    let adapterA ← adapterBuffer adapters projection adapterAField
    let adapterB ← adapterBuffer adapters projection adapterBField
    let adapterIds ← adapterIndex indices descriptor.adapterIndexBase projection 0
    if linear < inputCount then
      Cuda.Qwen36.LoRA.addBackwardInputElementF32 outputGradient adapterA adapterB adapterIds
        inputGradient shape linear
    if linear < aCount then
      Cuda.Qwen36.LoRA.backwardAElementF32 input outputGradient adapterB adapterIds
        (← adapterBuffer adapters projection adapterAGradientField) shape linear
    if linear < bCount then
      Cuda.Qwen36.LoRA.backwardBElementF32
        (← adapterBuffer adapters projection rankActivationField) outputGradient adapterIds
        (← adapterBuffer adapters projection adapterBGradientField) shape linear
    projectionAdapterBackwardLoop activations adapters indices operations descriptor metadata
      operation (linear + stride) stride

@[always_inline]
def runElementOperationBody (kind : UInt32) (activations : Cuda.DevicePtrTable Float32)
    (operations : Cuda.DevicePtr UInt32) (operation linear : UInt32) : Cuda.DeviceM Unit := do
  if kind == Op.add then
    Cuda.Qwen36.Linear.addElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3)) (← word operations operation 4) linear
  else if kind == Op.swigluForward then
    Cuda.Qwen36.Primitives.swigluForwardElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3)) (← word operations operation 4) linear
  else if kind == Op.swigluBackward then
    Cuda.Qwen36.Primitives.swigluBackwardElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3))
      (← activation activations (← word operations operation 4)) (← activation activations (← word operations operation 5)) (← word operations operation 6) linear
  else if kind == Op.convolutionForward then
    Cuda.Qwen36.Primitives.conv1dSiLUBatchedForwardElementF32
      (← activation activations (← word operations operation 1)) (← activation activations (← word operations operation 2))
      (← activation activations (← word operations operation 3)) (← word operations operation 4) (← word operations operation 5) (← word operations operation 6) linear
  else if kind == Op.convolutionBackward then
    Cuda.Qwen36.Primitives.conv1dSiLUBatchedBackwardInputElementF32
      (← activation activations (← word operations operation 1)) (← activation activations (← word operations operation 2))
      (← activation activations (← word operations operation 3)) (← activation activations (← word operations operation 4))
      (← word operations operation 5) (← word operations operation 6) (← word operations operation 7) linear
  else if kind == Op.splitQkv then
    Cuda.Qwen36.DeltaNet.splitQkvElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3))
      (← activation activations (← word operations operation 4)) (← word operations operation 5) {
        keyHeads := ← word operations operation 6, valueHeads := ← word operations operation 7, keyWidth := ← word operations operation 8, valueWidth := ← word operations operation 9
      } linear
  else if kind == Op.repeatCompact then
    Cuda.Qwen36.DeltaNet.repeatCompactElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← word operations operation 3) {
        keyHeads := ← word operations operation 4, valueHeads := ← word operations operation 5, keyWidth := ← word operations operation 6, valueWidth := ← word operations operation 7
      } linear
  else if kind == Op.reduceRepeated then
    Cuda.Qwen36.DeltaNet.reduceRepeatedGradientElementF32
      (← activation activations (← word operations operation 1)) (← activation activations (← word operations operation 2)) (← word operations operation 3) {
        keyHeads := ← word operations operation 4, valueHeads := ← word operations operation 5, keyWidth := ← word operations operation 6, valueWidth := ← word operations operation 7
      } linear
  else if kind == Op.concatenateQkv then
    Cuda.Qwen36.DeltaNet.concatenateQkvGradientElementF32
      (← activation activations (← word operations operation 1)) (← activation activations (← word operations operation 2))
      (← activation activations (← word operations operation 3)) (← activation activations (← word operations operation 4)) (← word operations operation 5) {
        keyHeads := ← word operations operation 6, valueHeads := ← word operations operation 7, keyWidth := ← word operations operation 8, valueWidth := ← word operations operation 9
      } linear
  else if kind == Op.deltaGateForward then
    Cuda.Qwen36.DeltaNet.gateForwardElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3))
      (← activation activations (← word operations operation 4)) (← activation activations (← word operations operation 5))
      (← activation activations (← word operations operation 6)) (← word operations operation 7) (← word operations operation 8) linear
  else if kind == Op.deltaGateBackward then
    Cuda.Qwen36.DeltaNet.gateBackwardElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3))
      (← activation activations (← word operations operation 4)) (← activation activations (← word operations operation 5))
      (← activation activations (← word operations operation 6)) (← activation activations (← word operations operation 7))
      (← activation activations (← word operations operation 8)) (← word operations operation 9) (← word operations operation 10) linear
  else if kind == Op.splitQueryGate then
    Cuda.Qwen36.Attention.splitQueryGateElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3)) {
        tokens := ← word operations operation 4, queryHeads := ← word operations operation 5, keyValueHeads := ← word operations operation 6, width := ← word operations operation 7
      } linear
  else if kind == Op.mergeQueryGate then
    Cuda.Qwen36.Attention.mergeQueryGateGradientElementF32
      (← activation activations (← word operations operation 1)) (← activation activations (← word operations operation 2))
      (← activation activations (← word operations operation 3)) {
        tokens := ← word operations operation 4, queryHeads := ← word operations operation 5, keyValueHeads := ← word operations operation 6, width := ← word operations operation 7
      } linear
  else if kind == Op.attentionGateBackward then
    Cuda.Qwen36.Attention.gateBackwardElementF32 (← activation activations (← word operations operation 1))
      (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3))
      (← activation activations (← word operations operation 4)) (← activation activations (← word operations operation 5)) (← word operations operation 6) linear
  else if kind == Op.probabilityBackward then
    let batchSize ← word operations operation 7
    let sequenceLength ← word operations operation 8
    let queryHeads ← word operations operation 9
    let keyValueHeads ← word operations operation 10
    let width ← word operations operation 11
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    let queryElements := sequenceLength * queryHeads * width
    let keyValueElements := sequenceLength * keyValueHeads * width
    if linear < batchSize * probabilityElements then
      let batch := linear / probabilityElements
      let localIndex := linear % probabilityElements
      Cuda.Qwen36.Attention.probabilityGradientElementF32
        ((← activation activations (← word operations operation 1)) + (batch * queryElements * 4).toUSize)
        ((← activation activations (← word operations operation 2)) + (batch * keyValueElements * 4).toUSize)
        ((← activation activations (← word operations operation 3)) + (batch * probabilityElements * 4).toUSize)
        { tokens := sequenceLength, queryHeads, keyValueHeads, width } localIndex
  else if kind == Op.scoreBackward then
    let batchSize ← word operations operation 7
    let sequenceLength ← word operations operation 8
    let queryHeads ← word operations operation 9
    let keyValueHeads ← word operations operation 10
    let width ← word operations operation 11
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    if linear < batchSize * probabilityElements then
      let batch := linear / probabilityElements
      let localIndex := linear % probabilityElements
      Cuda.Qwen36.Attention.scoreGradientElementF32
        ((← activation activations (← word operations operation 1)) + (batch * probabilityElements * 4).toUSize)
        ((← activation activations (← word operations operation 2)) + (batch * probabilityElements * 4).toUSize)
        ((← activation activations (← word operations operation 3)) + (batch * probabilityElements * 4).toUSize)
        { tokens := sequenceLength, queryHeads, keyValueHeads, width }
        (Float32.ofBits (← word operations operation 12)) localIndex
  else if kind == Op.queryBackward then
    let batchSize ← word operations operation 7
    let sequenceLength ← word operations operation 8
    let queryHeads ← word operations operation 9
    let keyValueHeads ← word operations operation 10
    let width ← word operations operation 11
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    let queryElements := sequenceLength * queryHeads * width
    let keyValueElements := sequenceLength * keyValueHeads * width
    if linear < batchSize * queryElements then
      let batch := linear / queryElements
      let localIndex := linear % queryElements
      Cuda.Qwen36.Attention.queryGradientElementF32
        ((← activation activations (← word operations operation 1)) + (batch * probabilityElements * 4).toUSize)
        ((← activation activations (← word operations operation 2)) + (batch * keyValueElements * 4).toUSize)
        ((← activation activations (← word operations operation 3)) + (batch * queryElements * 4).toUSize)
        { tokens := sequenceLength, queryHeads, keyValueHeads, width } localIndex
  else if kind == Op.keyValueBackward then
    let batchSize ← word operations operation 7
    let sequenceLength ← word operations operation 8
    let queryHeads ← word operations operation 9
    let keyValueHeads ← word operations operation 10
    let width ← word operations operation 11
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    let queryElements := sequenceLength * queryHeads * width
    let keyValueElements := sequenceLength * keyValueHeads * width
    if linear < batchSize * keyValueElements then
      let batch := linear / keyValueElements
      let localIndex := linear % keyValueElements
      Cuda.Qwen36.Attention.keyValueGradientElementF32
        ((← activation activations (← word operations operation 1)) + (batch * probabilityElements * 4).toUSize)
        ((← activation activations (← word operations operation 2)) + (batch * probabilityElements * 4).toUSize)
        ((← activation activations (← word operations operation 3)) + (batch * queryElements * 4).toUSize)
        ((← activation activations (← word operations operation 4)) + (batch * queryElements * 4).toUSize)
        ((← activation activations (← word operations operation 5)) + (batch * keyValueElements * 4).toUSize)
        ((← activation activations (← word operations operation 6)) + (batch * keyValueElements * 4).toUSize)
        { tokens := sequenceLength, queryHeads, keyValueHeads, width } localIndex
  else if kind == Op.ropeForward then
    let batchSize ← word operations operation 4
    let sequenceLength ← word operations operation 5
    let heads ← word operations operation 6
    let headDim ← word operations operation 7
    let half ← word operations operation 8
    if linear < batchSize * sequenceLength * heads * headDim then
      let token := linear / (heads * headDim)
      let position := token % sequenceLength
      let base := (linear / headDim) * headDim
      let d := linear % headDim
      Cuda.Qwen36.Primitives.ropeRotateFwdAt (← activation activations (← word operations operation 1))
        (← activation activations (← word operations operation 3))
        (← activation activations (← word operations operation 2)) base d half position.toFloat32
  else if kind == Op.ropeBackward then
    let batchSize ← word operations operation 4
    let sequenceLength ← word operations operation 5
    let heads ← word operations operation 6
    let headDim ← word operations operation 7
    let half ← word operations operation 8
    if linear < batchSize * sequenceLength * heads * headDim then
      let token := linear / (heads * headDim)
      let position := token % sequenceLength
      let base := (linear / headDim) * headDim
      let d := linear % headDim
      Cuda.Qwen36.Primitives.ropeRotateBwdAt (← activation activations (← word operations operation 1))
        (← activation activations (← word operations operation 3))
        (← activation activations (← word operations operation 2)) base d half position.toFloat32

@[always_inline]
partial def runElementOperation (kind : UInt32) (activations : Cuda.DevicePtrTable Float32)
    (operations : Cuda.DevicePtr UInt32) (operation linear stride : UInt32) :
    Cuda.DeviceM Unit := do
  let workItems ← word operations operation 31
  if linear < workItems then
    runElementOperationBody kind activations operations operation linear
    runElementOperation kind activations operations operation (linear + stride) stride

@[always_inline]
partial def embeddingLoop (matrices : Cuda.DevicePtrTable UInt8)
    (indices : Cuda.DevicePtrTable UInt32) (activations : Cuda.DevicePtrTable Float32)
    (operations : Cuda.DevicePtr UInt32) (operation linear stride : UInt32) : Cuda.DeviceM Unit := do
  let count ← word operations operation 4
  let width ← word operations operation 5
  if linear < count * width then
    let raw ← matrices.get (← word operations operation 1).toUSize
    let table : Cuda.DevicePtr Cuda.BFloat16 := raw
    let ids ← indexBuffer indices (← word operations operation 2)
    let output ← activation activations (← word operations operation 3)
    let row := linear / width
    let column := linear % width
    let id ← Cuda.loadUInt32 ids row.toUSize
    Cuda.storeFloat32 output linear.toUSize
      (← Cuda.loadBFloat16 table (id * width + column).toUSize).toFloat32
    embeddingLoop matrices indices activations operations operation (linear + stride) stride

@[always_inline, convergent]
partial def rowOperationLoop (kind : UInt32) (activations : Cuda.DevicePtrTable Float32)
    (operations : Cuda.DevicePtr UInt32) (operation row stride : UInt32) : Cuda.DeviceM Unit := do
  if kind == Op.rmsForward then
    let rows ← word operations operation 5
    if row < rows then
      Cuda.Qwen36.Primitives.rmsNormRowFwdF32 (← activation activations (← word operations operation 1))
        (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3))
        (← activation activations (← word operations operation 4)) row (← word operations operation 6) (Float32.ofBits (← word operations operation 7))
      rowOperationLoop kind activations operations operation (row + stride) stride
  else if kind == Op.rmsBackward then
    let rows ← word operations operation 6
    if row < rows then
      Cuda.Qwen36.Primitives.rmsNormRowBwdF32 (← activation activations (← word operations operation 1))
        (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3))
        (← activation activations (← word operations operation 4)) (← activation activations (← word operations operation 5)) row (← word operations operation 7)
      rowOperationLoop kind activations operations operation (row + stride) stride
  else if kind == Op.l2Forward then
    let rows ← word operations operation 3
    if row < rows then
      Cuda.Qwen36.Primitives.l2normRowFwdF32 (← activation activations (← word operations operation 1))
        (← activation activations (← word operations operation 2)) row (← word operations operation 4) (Float32.ofBits (← word operations operation 5))
      rowOperationLoop kind activations operations operation (row + stride) stride
  else if kind == Op.l2Backward then
    let rows ← word operations operation 4
    if row < rows then
      Cuda.Qwen36.Primitives.l2normRowBwdF32 (← activation activations (← word operations operation 1))
        (← activation activations (← word operations operation 2)) (← activation activations (← word operations operation 3)) row (← word operations operation 5)
        (Float32.ofBits (← word operations operation 6))
      rowOperationLoop kind activations operations operation (row + stride) stride
  else if kind == Op.weightedGatedRmsForward then
    let rows ← word operations operation 6
    if row < rows then
      Cuda.Qwen36.Primitives.weightedGatedRmsNormRowFwdF32
        (← activation activations (← word operations operation 1)) (← activation activations (← word operations operation 2))
        (← activation activations (← word operations operation 3)) (← activation activations (← word operations operation 4))
        (← activation activations (← word operations operation 5)) row (← word operations operation 7) (Float32.ofBits (← word operations operation 8))
      rowOperationLoop kind activations operations operation (row + stride) stride
  else if kind == Op.weightedGatedRmsBackward then
    let rows ← word operations operation 8
    if row < rows then
      Cuda.Qwen36.Primitives.weightedGatedRmsNormRowBwdF32
        (← activation activations (← word operations operation 1)) (← activation activations (← word operations operation 2))
        (← activation activations (← word operations operation 3)) (← activation activations (← word operations operation 4))
        (← activation activations (← word operations operation 5)) (← activation activations (← word operations operation 6))
        (← activation activations (← word operations operation 7)) row (← word operations operation 9)
      rowOperationLoop kind activations operations operation (row + stride) stride

@[always_inline, convergent]
partial def attentionForwardLoop (activations : Cuda.DevicePtrTable Float32)
    (operations : Cuda.DevicePtr UInt32) (operation role stride : UInt32)
    (scores : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let batchSize ← word operations operation 8
  let sequenceLength ← word operations operation 9
  let queryHeads ← word operations operation 10
  let keyValueHeads ← word operations operation 11
  let width ← word operations operation 12
  let rolesPerBatch := sequenceLength * queryHeads
  if role < batchSize * rolesPerBatch then
    let batch := role / rolesPerBatch
    let localRole := role % rolesPerBatch
    let queryElements := sequenceLength * queryHeads * width
    let keyValueElements := sequenceLength * keyValueHeads * width
    let probabilityElements := queryHeads * sequenceLength * sequenceLength
    let buffers : Cuda.Qwen36.Attention.Buffers := {
      query := (← activation activations (← word operations operation 1)) + (batch * queryElements * 4).toUSize
      key := (← activation activations (← word operations operation 2)) + (batch * keyValueElements * 4).toUSize
      value := (← activation activations (← word operations operation 3)) + (batch * keyValueElements * 4).toUSize
      gate := (← activation activations (← word operations operation 4)) + (batch * queryElements * 4).toUSize
      outputPreGate := (← activation activations (← word operations operation 5)) +
        (batch * queryElements * 4).toUSize
      outputPostGate := (← activation activations (← word operations operation 6)) +
        (batch * queryElements * 4).toUSize
      probabilities := (← activation activations (← word operations operation 7)) +
        (batch * probabilityElements * 4).toUSize
    }
    Cuda.Qwen36.Attention.attentionRowF32 buffers scores {
      tokens := sequenceLength, queryHeads, keyValueHeads, width
    } (localRole / queryHeads) (localRole % queryHeads) (Float32.ofBits (← word operations operation 13))
    attentionForwardLoop activations operations operation (role + stride) stride scores

@[always_inline, convergent]
partial def recurrenceForwardLoop (activations : Cuda.DevicePtrTable Float32)
    (operations : Cuda.DevicePtr UInt32) (operation role stride : UInt32)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let batchSize ← word operations operation 10
  let sequenceLength ← word operations operation 11
  let keyHeads ← word operations operation 12
  let valueHeads ← word operations operation 13
  let keyWidth ← word operations operation 14
  let valueWidth ← word operations operation 15
  if role < batchSize * valueHeads then
    Cuda.Qwen36.DeltaNet.recurrentNormalizedBatchHeadF32 {
      query := ← activation activations (← word operations operation 1)
      key := ← activation activations (← word operations operation 2)
      value := ← activation activations (← word operations operation 3)
      decayLog := ← activation activations (← word operations operation 4)
      beta := ← activation activations (← word operations operation 5)
      stateInput := ← activation activations (← word operations operation 6)
      output := ← activation activations (← word operations operation 7)
      stateOutput := ← activation activations (← word operations operation 8)
      stateHistory := ← activation activations (← word operations operation 9)
    } shared {
      batchSize := batchSize
      sequenceLength := sequenceLength
      recurrent := {
        keyHeads := keyHeads
        valueHeads := valueHeads
        keyWidth := keyWidth
        valueWidth := valueWidth
      }
    } role (Float32.ofBits (← word operations operation 16))
    recurrenceForwardLoop activations operations operation (role + stride) stride shared

@[always_inline, convergent]
partial def recurrenceBackwardLoop (activations : Cuda.DevicePtrTable Float32)
    (operations : Cuda.DevicePtr UInt32) (operation role stride : UInt32)
    (shared : Cuda.DevicePtr Float32) : Cuda.DeviceM Unit := do
  let batchSize ← word operations operation 14
  let sequenceLength ← word operations operation 15
  let keyHeads ← word operations operation 16
  let valueHeads ← word operations operation 17
  let keyWidth ← word operations operation 18
  let valueWidth ← word operations operation 19
  if role < batchSize * valueHeads then
    Cuda.Qwen36.DeltaNet.recurrentNormalizedBatchHeadBackwardF32 {
      query := ← activation activations (← word operations operation 1)
      key := ← activation activations (← word operations operation 2)
      value := ← activation activations (← word operations operation 3)
      decayLog := ← activation activations (← word operations operation 4)
      beta := ← activation activations (← word operations operation 5)
      stateHistory := ← activation activations (← word operations operation 6)
      outputGradient := ← activation activations (← word operations operation 7)
      queryGradient := ← activation activations (← word operations operation 8)
      keyGradient := ← activation activations (← word operations operation 9)
      valueGradient := ← activation activations (← word operations operation 10)
      decayLogGradient := ← activation activations (← word operations operation 11)
      betaGradient := ← activation activations (← word operations operation 12)
      stateGradient := ← activation activations (← word operations operation 13)
    } shared {
      batchSize := batchSize
      sequenceLength := sequenceLength
      recurrent := {
        keyHeads := keyHeads
        valueHeads := valueHeads
        keyWidth := keyWidth
        valueWidth := valueWidth
      }
    } role (Float32.ofBits (← word operations operation 20))
    recurrenceBackwardLoop activations operations operation (role + stride) stride shared

@[expose, cuda_device, always_inline, convergent]
def executeOperation (operations : Cuda.DevicePtr UInt32)
    (activations adapters : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (indices : Cuda.DevicePtrTable UInt32)
    (projectionMetadata : Cuda.DevicePtr UInt32) (descriptor : ProgramDescriptor)
    (operation : UInt32) : Cuda.DeviceM Unit := do
  let kind ← word operations operation 0
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  let blocks ← Cuda.gridDimX
  let threads ← Cuda.blockDimX
  let linear := block * threads + thread
  let linearStride := blocks * threads
  if kind == Op.embedding then
    embeddingLoop matrices indices activations operations operation linear linearStride
  else if kind == Op.projectionForward then
    let projection ← word operations operation 1
    let metadata ← loadProjectionMetadata projectionMetadata projection
    projectionRankLoop activations adapters indices operations descriptor metadata operation linear
      linearStride
    Cuda.gridSync
    projectionForwardLoop activations matrices adapters indices operations descriptor metadata
      operation linear linearStride
  else if kind == Op.projectionBackward then
    let projection ← word operations operation 1
    let metadata ← loadProjectionMetadata projectionMetadata projection
    projectionBaseBackwardLoop activations matrices operations metadata operation linear linearStride
    Cuda.gridSync
    projectionAdapterBackwardLoop activations adapters indices operations descriptor metadata
      operation linear linearStride
  else if kind == Op.rmsForward || kind == Op.rmsBackward || kind == Op.l2Forward ||
      kind == Op.l2Backward || kind == Op.weightedGatedRmsForward ||
      kind == Op.weightedGatedRmsBackward then
    rowOperationLoop kind activations operations operation block blocks
  else if kind == Op.attentionForward then
    let shared ← Cuda.dynamicShared (α := Float32)
    attentionForwardLoop activations operations operation block blocks shared
  else if kind == Op.recurrenceForward then
    let shared ← Cuda.dynamicShared (α := Float32)
    recurrenceForwardLoop activations operations operation block blocks shared
  else if kind == Op.recurrenceBackward then
    let shared ← Cuda.dynamicShared (α := Float32)
    recurrenceBackwardLoop activations operations operation block blocks shared
  else
    runElementOperation kind activations operations operation linear linearStride

@[always_inline, convergent]
partial def operationLoop (operations : Cuda.DevicePtr UInt32)
    (activations adapters : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (indices : Cuda.DevicePtrTable UInt32)
    (projectionMetadata : Cuda.DevicePtr UInt32) (descriptor : ProgramDescriptor)
    (operation endOperation : UInt32) : Cuda.DeviceM Unit := do
  if operation < endOperation then
    executeOperation operations activations adapters matrices indices projectionMetadata descriptor
      operation
    Cuda.gridSync
    operationLoop operations activations adapters matrices indices projectionMetadata descriptor
      (operation + 1) endOperation

@[expose, cuda_device, always_inline, convergent]
def runForward (operations : Cuda.DevicePtr UInt32)
    (activations adapters : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (indices : Cuda.DevicePtrTable UInt32)
    (projectionMetadata : Cuda.DevicePtr UInt32) (descriptor : ProgramDescriptor) :
    Cuda.DeviceM Unit :=
  operationLoop operations activations adapters matrices indices projectionMetadata descriptor 0
    descriptor.forwardOperations

@[expose, cuda_device, always_inline, convergent]
def runBackward (operations : Cuda.DevicePtr UInt32)
    (activations adapters : Cuda.DevicePtrTable Float32)
    (matrices : Cuda.DevicePtrTable UInt8) (indices : Cuda.DevicePtrTable UInt32)
    (projectionMetadata : Cuda.DevicePtr UInt32) (descriptor : ProgramDescriptor) :
    Cuda.DeviceM Unit :=
  operationLoop operations activations adapters matrices indices projectionMetadata descriptor
    descriptor.forwardOperations descriptor.operationCount

@[always_inline]
partial def optimizeProjectionElements (adapters : Cuda.DevicePtrTable Float32)
    (indices : Cuda.DevicePtrTable UInt32) (projectionMetadata : Cuda.DevicePtr UInt32)
    (descriptor : ProgramDescriptor) (config : Cuda.Training.AdamW.Config)
    (projection linear stride : UInt32) : Cuda.DeviceM Unit := do
  let metadata ← loadProjectionMetadata projectionMetadata projection
  let shape := metadata.shape
  let count := shape.adapterCount *
    (shape.rank * shape.inputFeatures + shape.outputFeatures * shape.rank)
  if linear < count then
    Cuda.Qwen36.LoRA.adamWElementF32 { config with weightDecay := metadata.weightDecay } {
      adapterA := ← adapterBuffer adapters projection adapterAField
      adapterB := ← adapterBuffer adapters projection adapterBField
      adapterAGradient := ← adapterBuffer adapters projection adapterAGradientField
      adapterBGradient := ← adapterBuffer adapters projection adapterBGradientField
      adapterAFirstMoment := ← adapterBuffer adapters projection adapterAFirstMomentField
      adapterASecondMoment := ← adapterBuffer adapters projection adapterASecondMomentField
      adapterBFirstMoment := ← adapterBuffer adapters projection adapterBFirstMomentField
      adapterBSecondMoment := ← adapterBuffer adapters projection adapterBSecondMomentField
      updateMask := ← adapterIndex indices descriptor.adapterIndexBase projection 1
    } shape linear
    optimizeProjectionElements adapters indices projectionMetadata descriptor config projection
      (linear + stride) stride

@[always_inline]
partial def optimizeProjectionLoop (adapters : Cuda.DevicePtrTable Float32)
    (indices : Cuda.DevicePtrTable UInt32) (projectionMetadata : Cuda.DevicePtr UInt32)
    (descriptor : ProgramDescriptor) (config : Cuda.Training.AdamW.Config)
    (projection linear stride : UInt32) : Cuda.DeviceM Unit := do
  if projection < descriptor.projectionCount then
    optimizeProjectionElements adapters indices projectionMetadata descriptor config projection
      linear stride
    optimizeProjectionLoop adapters indices projectionMetadata descriptor config (projection + 1)
      linear stride

@[expose, cuda_device, always_inline, convergent]
def advanceAndOptimize (adapters : Cuda.DevicePtrTable Float32)
    (indices : Cuda.DevicePtrTable UInt32) (projectionMetadata : Cuda.DevicePtr UInt32)
    (descriptor : ProgramDescriptor) (hyperparameters : Cuda.Qwen36.Training.AdamWHyperparametersF32)
    (schedule : Cuda.Qwen36.Training.AdamWScheduleBuffersF32) : Cuda.DeviceM Unit := do
  let block ← Cuda.blockIdxX
  let thread ← Cuda.threadIdxX
  if block == 0 && thread == 0 then
    Cuda.Qwen36.Training.advanceAdamWScheduleF32 hyperparameters schedule
  Cuda.gridSync
  let config : Cuda.Training.AdamW.Config := {
    learningRate := hyperparameters.learningRate
    beta1 := hyperparameters.beta1
    beta2 := hyperparameters.beta2
    inverseBiasCorrection1 := ← Cuda.loadFloat32 schedule.inverseBiasCorrection1 0
    inverseBiasCorrection2 := ← Cuda.loadFloat32 schedule.inverseBiasCorrection2 0
    epsilon := hyperparameters.epsilon
    weightDecay := 0
  }
  let linear := block * (← Cuda.blockDimX) + thread
  let stride := (← Cuda.gridDimX) * (← Cuda.blockDimX)
  optimizeProjectionLoop adapters indices projectionMetadata descriptor config 0 linear stride

@[struct] structure SoftmaxStatistics where
  maximum : Float32
  denominator : Float32
  inverseSum : Float32

@[always_inline]
partial def rowMaximum (logits : Cuda.DevicePtr Float32) (row classes column stride : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < classes then
    rowMaximum logits row classes (column + stride) stride
      (max accumulator (← Cuda.loadFloat32 logits (row * classes + column).toUSize))
  else
    return accumulator

@[always_inline]
partial def rowExponentSum (logits : Cuda.DevicePtr Float32) (row classes column stride : UInt32)
    (maximum accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < classes then
    let value ← Cuda.loadFloat32 logits (row * classes + column).toUSize
    rowExponentSum logits row classes (column + stride) stride maximum
      (accumulator + Float32.exp (value - maximum))
  else
    return accumulator

@[expose, cuda_device, always_inline, convergent]
def rowSoftmax (logits : Cuda.DevicePtr Float32) (row classes : UInt32) :
    Cuda.DeviceM SoftmaxStatistics := do
  let lane ← Cuda.laneId
  -- The modular objective uses a 256-thread block. This megakernel must remain one warp wide for
  -- the attention and recurrence bodies, so each physical lane evaluates the corresponding lane
  -- in all eight logical warps. The two levels of warp reductions below reproduce the modular
  -- block collective's floating-point reduction tree exactly.
  let negativeInfinity := Float32.ofBits 0xff800000
  let maximum0 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes lane 256 negativeInfinity)
  let maximum1 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes (lane + 32) 256 negativeInfinity)
  let maximum2 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes (lane + 64) 256 negativeInfinity)
  let maximum3 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes (lane + 96) 256 negativeInfinity)
  let maximum4 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes (lane + 128) 256 negativeInfinity)
  let maximum5 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes (lane + 160) 256 negativeInfinity)
  let maximum6 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes (lane + 192) 256 negativeInfinity)
  let maximum7 ← Cuda.Qwen36.Attention.warpMaxBroadcast
    (← rowMaximum logits row classes (lane + 224) 256 negativeInfinity)
  let maximumPartial := if lane == 0 then maximum0 else if lane == 1 then maximum1 else
    if lane == 2 then maximum2 else if lane == 3 then maximum3 else
    if lane == 4 then maximum4 else if lane == 5 then maximum5 else
    if lane == 6 then maximum6 else if lane == 7 then maximum7 else negativeInfinity
  let maximum ← Cuda.Qwen36.Attention.warpMaxBroadcast maximumPartial
  let sum0 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes lane 256 maximum 0)
  let sum1 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes (lane + 32) 256 maximum 0)
  let sum2 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes (lane + 64) 256 maximum 0)
  let sum3 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes (lane + 96) 256 maximum 0)
  let sum4 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes (lane + 128) 256 maximum 0)
  let sum5 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes (lane + 160) 256 maximum 0)
  let sum6 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes (lane + 192) 256 maximum 0)
  let sum7 ← Cuda.Qwen36.Primitives.warpSumBroadcast
    (← rowExponentSum logits row classes (lane + 224) 256 maximum 0)
  let sumPartial := if lane == 0 then sum0 else if lane == 1 then sum1 else
    if lane == 2 then sum2 else if lane == 3 then sum3 else if lane == 4 then sum4 else
    if lane == 5 then sum5 else if lane == 6 then sum6 else if lane == 7 then sum7 else 0
  let denominator ← Cuda.Qwen36.Primitives.warpSumBroadcast sumPartial
  return SoftmaxStatistics.mk maximum denominator
    (Cuda.Qwen36.Primitives.oneF32 / denominator)

@[always_inline]
partial def storeLogitGradient (logits gradient : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32) (row classes column stride : UInt32)
    (statistics : SoftmaxStatistics) (coefficient : Float32) : Cuda.DeviceM Unit := do
  if column < classes then
    let value ← Cuda.loadFloat32 logits (row * classes + column).toUSize
    let probability := Float32.exp (value - statistics.maximum) * statistics.inverseSum
    let label ← Cuda.loadUInt32 targets row.toUSize
    let target := if column == label then Cuda.Qwen36.Primitives.oneF32 else 0
    Cuda.storeFloat32 gradient (row * classes + column).toUSize
      ((probability - target) * coefficient)
    storeLogitGradient logits gradient targets row classes (column + stride) stride statistics
      coefficient

@[always_inline, convergent]
partial def pretrainingRows (logits gradient rowLoss : Cuda.DevicePtr Float32)
    (targets : Cuda.DevicePtr UInt32) (rows classes row stride : UInt32) : Cuda.DeviceM Unit := do
  if row < rows then
    let lane ← Cuda.laneId
    let statistics ← rowSoftmax logits row classes
    let scale := Cuda.Qwen36.Primitives.oneF32 / rows.toFloat32
    storeLogitGradient logits gradient targets row classes lane 32 statistics scale
    if lane == 0 then
      let label ← Cuda.loadUInt32 targets row.toUSize
      let target ← Cuda.loadFloat32 logits (row * classes + label).toUSize
      Cuda.storeFloat32 rowLoss row.toUSize
        ((statistics.maximum + Float32.log statistics.denominator - target) * scale)
    pretrainingRows logits gradient rowLoss targets rows classes (row + stride) stride

@[always_inline, convergent]
partial def policyRows (logits policy : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (rows classes row stride : UInt32) :
    Cuda.DeviceM Unit := do
  if row < rows then
    let lane ← Cuda.laneId
    let statistics ← rowSoftmax logits row classes
    if lane == 0 then
      let enabled ← Cuda.loadUInt32 masks row.toUSize
      let value ← if enabled == 0 then pure 0 else do
        let label ← Cuda.loadUInt32 targets row.toUSize
        let target ← Cuda.loadFloat32 logits (row * classes + label).toUSize
        pure (target - statistics.maximum + Float32.log statistics.inverseSum)
      Cuda.storeFloat32 policy row.toUSize value
    policyRows logits policy targets masks rows classes (row + stride) stride

@[always_inline, convergent]
partial def grpoGradientRows (logits gradient coefficients : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (rows classes row stride : UInt32) :
    Cuda.DeviceM Unit := do
  if row < rows then
    let lane ← Cuda.laneId
    let statistics ← rowSoftmax logits row classes
    let enabled ← Cuda.loadUInt32 masks row.toUSize
    let coefficient ← if enabled == 0 then pure 0 else Cuda.loadFloat32 coefficients row.toUSize
    storeLogitGradient logits gradient targets row classes lane 32 statistics coefficient
    grpoGradientRows logits gradient coefficients targets masks rows classes (row + stride) stride

@[always_inline]
partial def sumRows (values : Cuda.DevicePtr Float32) (rows row : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < rows then
    sumRows values rows (row + 1) (accumulator + (← Cuda.loadFloat32 values row.toUSize))
  else
    return accumulator

end Internal

namespace Program

private structure BuildState where
  operations : IO.Ref (Array UInt32)
  activations : IO.Ref (Array (Cuda.Buffer Float32))
  matrices : IO.Ref (Array (Cuda.Buffer UInt8))
  matrixOffsets : IO.Ref (Array USize)
  adapters : IO.Ref (Array (Cuda.Buffer Float32))
  indices : IO.Ref (Array (Cuda.Buffer UInt32))
  metadata : IO.Ref (Array UInt32)

private inductive MixerProjectionIds where
  | deltaNet (qkv z b a output : UInt32)
  | attention (query key value output : UInt32)

private structure LayerProjectionIds where
  mixer : MixerProjectionIds
  gate : UInt32
  up : UInt32
  down : UInt32

private def appendUInt32LE (bytes : ByteArray) (value : UInt32) : ByteArray :=
  (((bytes.push value.toUInt8).push (value >>> 8).toUInt8).push
    (value >>> 16).toUInt8).push (value >>> 24).toUInt8

private def packUInt32 (values : @& Array UInt32) : ByteArray := Id.run do
  let mut bytes := ByteArray.emptyWithCapacity (values.size * 4)
  for value in values do
    bytes := appendUInt32LE bytes value
  return bytes

private def addActivation (state : @& BuildState) (buffer : Cuda.Buffer Float32) : IO UInt32 := do
  let buffers ← state.activations.get
  state.activations.set (buffers.push buffer)
  return buffers.size.toUInt32

private def addIndex (state : @& BuildState) (buffer : Cuda.Buffer UInt32) : IO UInt32 := do
  let buffers ← state.indices.get
  state.indices.set (buffers.push buffer)
  return buffers.size.toUInt32

private def emit (state : @& BuildState) (fields : Array UInt32)
    (workItems : UInt32 := 0) : IO Unit := do
  unless fields.size ≤ operationWords.toNat do
    throw <| IO.userError s!"Qwen3.6 full-training operation has {fields.size} words"
  let record := (fields ++ Array.replicate (operationWords.toNat - fields.size) 0).set! 31
    workItems
  state.operations.modify fun operations => operations ++ record

private def activationIds (state : @& BuildState)
    (buffers : @& Array (Cuda.Buffer Float32)) : IO (Array UInt32) := do
  let mut result := #[]
  for buffer in buffers do
    result := result.push (← addActivation state buffer)
  return result

private def emitF32 (state : @& BuildState) (kind : UInt32)
    (buffers : Array (Cuda.Buffer Float32)) (tail : Array UInt32 := #[])
    (workItems : UInt32 := 0) : IO Unit := do
  emit state (#[kind] ++ (← activationIds state buffers) ++ tail) workItems

private def registerProjection (state : @& BuildState) (weights : @& Projection.WeightsF32)
    (rows inputFeatures outputFeatures : UInt32) : IO UInt32 := do
  let .loraFrozenBF16 base adapter := weights
    | throw <| IO.userError
        "full-training megakernel currently requires frozen-BF16 LoRA on every projection"
  unless adapter.shape.rows == rows && adapter.shape.inputFeatures == inputFeatures &&
      adapter.shape.outputFeatures == outputFeatures do
    throw <| IO.userError "full-training megakernel projection shape mismatch"
  adapter.check rows inputFeatures outputFeatures
  let matrixOwners ← state.matrices.get
  let projection := (matrixOwners.size - 1).toUInt32
  state.matrices.set (matrixOwners.push base.weight.owner)
  state.matrixOffsets.modify fun offsets => offsets.push base.weight.byteOffset.toUSize
  state.adapters.modify fun buffers => buffers ++ #[
    adapter.adapterA, adapter.adapterB, adapter.rankActivation, adapter.adapterAGradient,
    adapter.adapterBGradient, adapter.adapterAFirstMoment, adapter.adapterASecondMoment,
    adapter.adapterBFirstMoment, adapter.adapterBSecondMoment
  ]
  state.indices.modify fun buffers => buffers ++ #[adapter.adapterIds, adapter.updateMask]
  let shape := adapter.shape
  state.metadata.modify fun values => values ++ #[
    shape.rows, shape.sequenceLength, shape.adapterCount, shape.rank, shape.inputFeatures,
    shape.outputFeatures, shape.scale.toBits, adapter.weightDecay.toBits
  ]
  return projection

private def emitProjectionForward (state : @& BuildState) (projection : UInt32)
    (input output : Cuda.Buffer Float32) : IO Unit := do
  let inputId ← addActivation state input
  let outputId ← addActivation state output
  emit state #[Op.projectionForward, projection, inputId, outputId]

private def emitProjectionBackward (state : @& BuildState) (projection : UInt32)
    (input outputGradient inputGradient : Cuda.Buffer Float32) : IO Unit := do
  let inputId ← addActivation state input
  let outputGradientId ← addActivation state outputGradient
  let inputGradientId ← addActivation state inputGradient
  emit state #[Op.projectionBackward, projection, inputId, outputGradientId, inputGradientId]

private def emitRmsForward (state : @& BuildState) (input weight output inverseRms :
    Cuda.Buffer Float32) (rows width : UInt32) (epsilon : Float32) : IO Unit :=
  emitF32 state Op.rmsForward #[input, weight, output, inverseRms]
    #[rows, width, epsilon.toBits]

private def emitRmsBackward (state : @& BuildState) (input weight outputGradient inputGradient
    inverseRms : Cuda.Buffer Float32) (rows width : UInt32) : IO Unit :=
  emitF32 state Op.rmsBackward #[input, weight, outputGradient, inputGradient, inverseRms]
    #[rows, width]

private def emitAdd (state : @& BuildState) (left right output : Cuda.Buffer Float32)
    (count : UInt32) : IO Unit :=
  emitF32 state Op.add #[left, right, output] #[count] count

private def emitL2Forward (state : @& BuildState) (input output : Cuda.Buffer Float32)
    (rows width : UInt32) (epsilon : Float32) : IO Unit :=
  emitF32 state Op.l2Forward #[input, output] #[rows, width, epsilon.toBits]

private def emitL2Backward (state : @& BuildState) (input outputGradient inputGradient :
    Cuda.Buffer Float32) (rows width : UInt32) (epsilon : Float32) : IO Unit :=
  emitF32 state Op.l2Backward #[input, outputGradient, inputGradient]
    #[rows, width, epsilon.toBits]

private def emitRope (state : @& BuildState) (kind : UInt32)
    (input inverseFrequency output : Cuda.Buffer Float32) (batchSize sequenceLength heads width
      half : UInt32) : IO Unit :=
  emitF32 state kind #[input, inverseFrequency, output]
    #[batchSize, sequenceLength, heads, width, half]
    (batchSize * sequenceLength * heads * width)

private def emitAttentionBackwardElement (state : @& BuildState) (kind : UInt32)
    (buffers : Array (Cuda.Buffer Float32)) (batchSize sequenceLength queryHeads keyValueHeads width :
      UInt32) (scale? : Option Float32 := none) : IO Unit := do
  let ids ← activationIds state buffers
  let fields := #[kind] ++ ids
  let fields := fields ++ Array.replicate (7 - fields.size) 0
  let fields := fields ++ #[batchSize, sequenceLength, queryHeads, keyValueHeads, width]
  let fields := match scale? with
    | some scale => fields.push scale.toBits
    | none => fields
  let workItems := if kind == Op.probabilityBackward || kind == Op.scoreBackward then
      batchSize * queryHeads * sequenceLength * sequenceLength
    else if kind == Op.queryBackward then
      batchSize * sequenceLength * queryHeads * width
    else
      batchSize * sequenceLength * keyValueHeads * width
  emit state fields workItems

private def emitDeltaForward (state : @& BuildState) (weights : @& DeltaNet.StageWeightsF32)
    (buffers : @& DeltaNet.StageForwardF32) (input : Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& Model.ShapeF32) :
    IO MixerProjectionIds := do
  let tokens := shape.tokens
  let qkvWidth := 2 * shape.linearKeyHeads * shape.linearKeyWidth +
    shape.linearValueHeads * shape.linearValueWidth
  emitRmsForward state input weights.inputNorm buffers.normalizedInput buffers.inputInverseRms
    tokens shape.hidden shape.epsilon
  let qkv ← registerProjection state weights.queryKeyValue tokens shape.hidden qkvWidth
  emitProjectionForward state qkv buffers.normalizedInput buffers.queryKeyValuePreConv
  let z ← registerProjection state weights.z tokens shape.hidden
    (shape.linearValueHeads * shape.linearValueWidth)
  emitProjectionForward state z buffers.normalizedInput buffers.z
  let b ← registerProjection state weights.b tokens shape.hidden shape.linearValueHeads
  emitProjectionForward state b buffers.normalizedInput buffers.b
  let a ← registerProjection state weights.a tokens shape.hidden shape.linearValueHeads
  emitProjectionForward state a buffers.normalizedInput buffers.a
  emitF32 state Op.convolutionForward
    #[buffers.queryKeyValuePreConv, weights.convolution, buffers.queryKeyValuePostConv]
    #[layout.batchSize, layout.sequenceLength, qkvWidth]
    (layout.batchSize * layout.sequenceLength * qkvWidth)
  emitF32 state Op.splitQkv #[buffers.queryKeyValuePostConv, buffers.queryCompact,
    buffers.keyCompact, buffers.value] #[tokens, shape.linearKeyHeads, shape.linearValueHeads,
      shape.linearKeyWidth, shape.linearValueWidth] (tokens * qkvWidth)
  emitF32 state Op.repeatCompact #[buffers.queryCompact, buffers.queryRepeated]
    #[tokens, shape.linearKeyHeads, shape.linearValueHeads, shape.linearKeyWidth,
      shape.linearValueWidth] (tokens * shape.linearValueHeads * shape.linearKeyWidth)
  emitF32 state Op.repeatCompact #[buffers.keyCompact, buffers.keyRepeated]
    #[tokens, shape.linearKeyHeads, shape.linearValueHeads, shape.linearKeyWidth,
      shape.linearValueWidth] (tokens * shape.linearValueHeads * shape.linearKeyWidth)
  emitL2Forward state buffers.queryRepeated buffers.queryNormed
    (tokens * shape.linearValueHeads) shape.linearKeyWidth shape.epsilon
  emitL2Forward state buffers.keyRepeated buffers.keyNormed
    (tokens * shape.linearValueHeads) shape.linearKeyWidth shape.epsilon
  emitF32 state Op.deltaGateForward #[buffers.a, buffers.b, weights.aLog, weights.dtBias,
    buffers.decayLog, buffers.beta] #[tokens, shape.linearValueHeads]
    (tokens * shape.linearValueHeads)
  emitF32 state Op.recurrenceForward #[buffers.queryNormed, buffers.keyNormed, buffers.value,
    buffers.decayLog, buffers.beta, buffers.stateInput, buffers.deltaOutput, buffers.stateOutput,
    buffers.stateHistory] #[layout.batchSize, layout.sequenceLength, shape.linearKeyHeads,
      shape.linearValueHeads, shape.linearKeyWidth, shape.linearValueWidth, shape.queryScale.toBits]
  emitF32 state Op.weightedGatedRmsForward #[buffers.deltaOutput, buffers.z, weights.gatedNorm,
    buffers.gatedOutput, buffers.gatedInverseRms]
    #[tokens * shape.linearValueHeads, shape.linearValueWidth, shape.epsilon.toBits]
  let output ← registerProjection state weights.output tokens
    (shape.linearValueHeads * shape.linearValueWidth) shape.hidden
  emitProjectionForward state output buffers.gatedOutput buffers.output
  return .deltaNet qkv z b a output

private def emitAttentionForward (state : @& BuildState) (weights : @& Attention.StageWeightsF32)
    (buffers : @& Attention.StageForwardF32) (input : Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& Model.ShapeF32) :
    IO MixerProjectionIds := do
  let tokens := shape.tokens
  let querySize := shape.attentionQueryHeads * shape.attentionHeadWidth
  let keyValueSize := shape.attentionKeyValueHeads * shape.attentionHeadWidth
  emitRmsForward state input weights.inputNorm buffers.normalizedInput buffers.inputInverseRms
    tokens shape.hidden shape.epsilon
  let query ← registerProjection state weights.query tokens shape.hidden (2 * querySize)
  emitProjectionForward state query buffers.normalizedInput buffers.queryGateProjection
  emitF32 state Op.splitQueryGate #[buffers.queryGateProjection, buffers.queryPreNorm, buffers.gate]
    #[tokens, shape.attentionQueryHeads, shape.attentionKeyValueHeads,
      shape.attentionHeadWidth] (tokens * querySize)
  let key ← registerProjection state weights.key tokens shape.hidden keyValueSize
  emitProjectionForward state key buffers.normalizedInput buffers.keyPreNorm
  let value ← registerProjection state weights.value tokens shape.hidden keyValueSize
  emitProjectionForward state value buffers.normalizedInput buffers.value
  emitRmsForward state buffers.queryPreNorm weights.queryNorm buffers.queryNormed
    buffers.queryInverseRms (tokens * shape.attentionQueryHeads) shape.attentionHeadWidth shape.epsilon
  emitRmsForward state buffers.keyPreNorm weights.keyNorm buffers.keyNormed buffers.keyInverseRms
    (tokens * shape.attentionKeyValueHeads) shape.attentionHeadWidth shape.epsilon
  emitRope state Op.ropeForward buffers.queryNormed weights.inverseFrequency buffers.queryRope
    layout.batchSize layout.sequenceLength shape.attentionQueryHeads shape.attentionHeadWidth
    shape.rotaryHalf
  emitRope state Op.ropeForward buffers.keyNormed weights.inverseFrequency buffers.keyRope
    layout.batchSize layout.sequenceLength shape.attentionKeyValueHeads shape.attentionHeadWidth
    shape.rotaryHalf
  emitF32 state Op.attentionForward #[buffers.queryRope, buffers.keyRope, buffers.value,
    buffers.gate, buffers.preGate, buffers.postGate, buffers.probabilities]
    #[layout.batchSize, layout.sequenceLength, shape.attentionQueryHeads,
      shape.attentionKeyValueHeads, shape.attentionHeadWidth, shape.attentionScale.toBits]
  let output ← registerProjection state weights.output tokens querySize shape.hidden
  emitProjectionForward state output buffers.postGate buffers.output
  return .attention query key value output

private def emitMlpForward (state : @& BuildState) (weights : @& Model.MLPWeightsF32)
    (buffers : @& Model.MLPForwardF32) (input : Cuda.Buffer Float32)
    (shape : @& Model.ShapeF32) : IO (UInt32 × UInt32 × UInt32) := do
  emitRmsForward state input weights.norm buffers.normalized buffers.inverseRms shape.tokens
    shape.hidden shape.epsilon
  let gate ← registerProjection state weights.gate shape.tokens shape.hidden shape.intermediate
  emitProjectionForward state gate buffers.normalized buffers.gate
  let up ← registerProjection state weights.up shape.tokens shape.hidden shape.intermediate
  emitProjectionForward state up buffers.normalized buffers.up
  emitF32 state Op.swigluForward #[buffers.gate, buffers.up, buffers.activation]
    #[shape.tokens * shape.intermediate] (shape.tokens * shape.intermediate)
  let down ← registerProjection state weights.down shape.tokens shape.intermediate shape.hidden
  emitProjectionForward state down buffers.activation buffers.output
  return (gate, up, down)

private def emitDeltaBackward (state : @& BuildState) (ids : MixerProjectionIds)
    (weights : @& DeltaNet.StageWeightsF32) (forward : @& DeltaNet.StageForwardF32)
    (backward : @& DeltaNet.StageBackwardF32) (layout : Cuda.Qwen36.SequenceLayout)
    (input outputGradient : Cuda.Buffer Float32) (shape : @& Model.ShapeF32) : IO Unit := do
  let .deltaNet qkv z b a output := ids
    | throw <| IO.userError "full-training DeltaNet projection registry mismatch"
  let tokens := shape.tokens
  let qkvWidth := 2 * shape.linearKeyHeads * shape.linearKeyWidth +
    shape.linearValueHeads * shape.linearValueWidth
  emitProjectionBackward state output forward.gatedOutput outputGradient
    backward.gatedOutputGradient
  emitF32 state Op.weightedGatedRmsBackward #[forward.deltaOutput, forward.z, weights.gatedNorm,
    backward.gatedOutputGradient, backward.deltaOutputGradient, backward.zGradient,
    forward.gatedInverseRms] #[tokens * shape.linearValueHeads, shape.linearValueWidth]
  emitF32 state Op.recurrenceBackward #[forward.queryNormed, forward.keyNormed, forward.value,
    forward.decayLog, forward.beta, forward.stateHistory, backward.deltaOutputGradient,
    backward.queryNormGradient, backward.keyNormGradient, backward.valueGradient,
    backward.decayLogGradient, backward.betaGradient, backward.stateGradient]
    #[layout.batchSize, layout.sequenceLength, shape.linearKeyHeads, shape.linearValueHeads,
      shape.linearKeyWidth, shape.linearValueWidth, shape.queryScale.toBits]
  emitL2Backward state forward.queryRepeated backward.queryNormGradient
    backward.queryRepeatedGradient (tokens * shape.linearValueHeads) shape.linearKeyWidth shape.epsilon
  emitL2Backward state forward.keyRepeated backward.keyNormGradient backward.keyRepeatedGradient
    (tokens * shape.linearValueHeads) shape.linearKeyWidth shape.epsilon
  emitF32 state Op.reduceRepeated #[backward.queryRepeatedGradient,
    backward.queryCompactGradient] #[tokens, shape.linearKeyHeads, shape.linearValueHeads,
      shape.linearKeyWidth, shape.linearValueWidth]
    (tokens * shape.linearKeyHeads * shape.linearKeyWidth)
  emitF32 state Op.reduceRepeated #[backward.keyRepeatedGradient, backward.keyCompactGradient]
    #[tokens, shape.linearKeyHeads, shape.linearValueHeads, shape.linearKeyWidth,
      shape.linearValueWidth] (tokens * shape.linearKeyHeads * shape.linearKeyWidth)
  emitF32 state Op.concatenateQkv #[backward.queryCompactGradient, backward.keyCompactGradient,
    backward.valueGradient, backward.queryKeyValuePostConvGradient]
    #[tokens, shape.linearKeyHeads, shape.linearValueHeads, shape.linearKeyWidth,
      shape.linearValueWidth] (tokens * qkvWidth)
  emitF32 state Op.convolutionBackward #[forward.queryKeyValuePreConv, weights.convolution,
    backward.queryKeyValuePostConvGradient, backward.queryKeyValuePreConvGradient]
    #[layout.batchSize, layout.sequenceLength, qkvWidth]
    (layout.batchSize * layout.sequenceLength * qkvWidth)
  emitF32 state Op.deltaGateBackward #[forward.a, forward.b, weights.aLog, weights.dtBias,
    backward.decayLogGradient, backward.betaGradient, backward.aGradient, backward.bGradient]
    #[tokens, shape.linearValueHeads] (tokens * shape.linearValueHeads)
  emitProjectionBackward state qkv forward.normalizedInput
    backward.queryKeyValuePreConvGradient backward.queryKeyValueInputGradient
  emitProjectionBackward state z forward.normalizedInput backward.zGradient backward.zInputGradient
  emitProjectionBackward state b forward.normalizedInput backward.bGradient backward.bInputGradient
  emitProjectionBackward state a forward.normalizedInput backward.aGradient backward.aInputGradient
  emitAdd state backward.queryKeyValueInputGradient backward.zInputGradient
    backward.queryKeyValueZInputGradient (tokens * shape.hidden)
  emitAdd state backward.queryKeyValueZInputGradient backward.bInputGradient
    backward.queryKeyValueZBInputGradient (tokens * shape.hidden)
  emitAdd state backward.queryKeyValueZBInputGradient backward.aInputGradient
    backward.normalizedInputGradient (tokens * shape.hidden)
  emitRmsBackward state input weights.inputNorm backward.normalizedInputGradient
    backward.inputGradient forward.inputInverseRms tokens shape.hidden

private def emitAttentionBackward (state : @& BuildState) (ids : MixerProjectionIds)
    (weights : @& Attention.StageWeightsF32) (forward : @& Attention.StageForwardF32)
    (backward : @& Attention.StageBackwardF32) (layout : Cuda.Qwen36.SequenceLayout)
    (input outputGradient : Cuda.Buffer Float32) (shape : @& Model.ShapeF32) : IO Unit := do
  let .attention query key value output := ids
    | throw <| IO.userError "full-training attention projection registry mismatch"
  let tokens := shape.tokens
  let queryElements := tokens * shape.attentionQueryHeads * shape.attentionHeadWidth
  emitProjectionBackward state output forward.postGate outputGradient
    backward.postGateGradient
  emitF32 state Op.attentionGateBackward #[forward.preGate, forward.gate,
    backward.postGateGradient, backward.preGateGradient, backward.gateGradient] #[queryElements]
    queryElements
  emitAttentionBackwardElement state Op.probabilityBackward
    #[backward.preGateGradient, forward.value, backward.probabilityGradient]
    layout.batchSize layout.sequenceLength shape.attentionQueryHeads
    shape.attentionKeyValueHeads shape.attentionHeadWidth
  emitAttentionBackwardElement state Op.scoreBackward
    #[forward.probabilities, backward.probabilityGradient, backward.scoreGradient]
    layout.batchSize layout.sequenceLength shape.attentionQueryHeads
    shape.attentionKeyValueHeads shape.attentionHeadWidth (some shape.attentionScale)
  emitAttentionBackwardElement state Op.queryBackward
    #[backward.scoreGradient, forward.keyRope, backward.queryRopeGradient]
    layout.batchSize layout.sequenceLength shape.attentionQueryHeads
    shape.attentionKeyValueHeads shape.attentionHeadWidth
  emitAttentionBackwardElement state Op.keyValueBackward
    #[backward.scoreGradient, forward.probabilities, forward.queryRope,
      backward.preGateGradient, backward.keyRopeGradient, backward.valueGradient]
    layout.batchSize layout.sequenceLength shape.attentionQueryHeads
    shape.attentionKeyValueHeads shape.attentionHeadWidth
  emitRope state Op.ropeBackward backward.queryRopeGradient weights.inverseFrequency
    backward.queryNormGradient layout.batchSize layout.sequenceLength shape.attentionQueryHeads
    shape.attentionHeadWidth shape.rotaryHalf
  emitRope state Op.ropeBackward backward.keyRopeGradient weights.inverseFrequency
    backward.keyNormGradient layout.batchSize layout.sequenceLength shape.attentionKeyValueHeads
    shape.attentionHeadWidth shape.rotaryHalf
  emitRmsBackward state forward.queryPreNorm weights.queryNorm backward.queryNormGradient
    backward.queryPreNormGradient forward.queryInverseRms (tokens * shape.attentionQueryHeads)
    shape.attentionHeadWidth
  emitRmsBackward state forward.keyPreNorm weights.keyNorm backward.keyNormGradient
    backward.keyPreNormGradient forward.keyInverseRms (tokens * shape.attentionKeyValueHeads)
    shape.attentionHeadWidth
  emitF32 state Op.mergeQueryGate #[backward.queryPreNormGradient, backward.gateGradient,
    backward.queryGateProjectionGradient] #[tokens, shape.attentionQueryHeads,
      shape.attentionKeyValueHeads, shape.attentionHeadWidth] queryElements
  emitProjectionBackward state query forward.normalizedInput
    backward.queryGateProjectionGradient backward.queryInputGradient
  emitProjectionBackward state key forward.normalizedInput backward.keyPreNormGradient
    backward.keyInputGradient
  emitProjectionBackward state value forward.normalizedInput backward.valueGradient
    backward.valueInputGradient
  emitAdd state backward.queryInputGradient backward.keyInputGradient
    backward.queryKeyInputGradient (tokens * shape.hidden)
  emitAdd state backward.queryKeyInputGradient backward.valueInputGradient
    backward.normalizedInputGradient (tokens * shape.hidden)
  emitRmsBackward state input weights.inputNorm backward.normalizedInputGradient
    backward.inputGradient forward.inputInverseRms tokens shape.hidden

private def emitMlpBackward (state : @& BuildState) (ids : LayerProjectionIds)
    (weights : @& Model.MLPWeightsF32) (forward : @& Model.MLPForwardF32)
    (backward : @& Model.MLPBackwardF32) (input outputGradient : Cuda.Buffer Float32)
    (shape : @& Model.ShapeF32) : IO Unit := do
  emitProjectionBackward state ids.down forward.activation outputGradient
    backward.activationGradient
  emitF32 state Op.swigluBackward #[forward.gate, forward.up, backward.activationGradient,
    backward.gateGradient, backward.upGradient] #[shape.tokens * shape.intermediate]
    (shape.tokens * shape.intermediate)
  emitProjectionBackward state ids.gate forward.normalized backward.gateGradient
    backward.normGradientGate
  emitProjectionBackward state ids.up forward.normalized backward.upGradient
    backward.normGradientUp
  emitAdd state backward.normGradientGate backward.normGradientUp backward.normGradient
    (shape.tokens * shape.hidden)
  emitRmsBackward state input weights.norm backward.normGradient backward.inputGradient
    forward.inverseRms shape.tokens shape.hidden

/-- Lower one checked LoRA full-model step to the model-wide cooperative operation tape. -/
def build (stream : @& Cuda.Stream) (descriptor : Training.Descriptor)
    (step : @& Training.FullModelStepF32) : IO Program := do
  let descriptor ← match descriptor.check with
    | .ok descriptor => pure descriptor
    | .error message => throw <| IO.userError message
  let .lora _ := descriptor.profile
    | throw <| IO.userError "full-training megakernel currently supports the LoRA profile"
  unless step.weights.layers.size == step.forward.layers.size &&
      step.weights.layers.size == step.backward.layers.size do
    throw <| IO.userError "full-training megakernel layer-count mismatch"
  let embedding ← match step.weights.embedding with
    | .frozenBF16 embedding => pure embedding
    | .f32 _ => throw (IO.userError
        "full-training megakernel currently requires a frozen BF16 embedding")
  let operationsRef ← IO.mkRef (#[] : Array UInt32)
  let activationsRef ← IO.mkRef (#[] : Array (Cuda.Buffer Float32))
  let matricesRef ← IO.mkRef #[embedding.owner]
  let matrixOffsetsRef ← IO.mkRef #[embedding.byteOffset.toUSize]
  let adaptersRef ← IO.mkRef (#[] : Array (Cuda.Buffer Float32))
  let indicesRef ← IO.mkRef #[step.tokenIds, step.targets]
  let metadataRef ← IO.mkRef (#[] : Array UInt32)
  let state : BuildState := {
    operations := operationsRef, activations := activationsRef, matrices := matricesRef,
    matrixOffsets := matrixOffsetsRef, adapters := adaptersRef, indices := indicesRef,
    metadata := metadataRef
  }
  let shape := descriptor.model
  let layout : Cuda.Qwen36.SequenceLayout := {
    batchSize := descriptor.batchSize.toUInt32
    sequenceLength := descriptor.sequenceLength.toUInt32
  }
  let embeddingOutput ← addActivation state step.forward.embedding
  emit state #[Op.embedding, 0, 0, embeddingOutput, shape.tokens, shape.hidden]
  let mut input := step.forward.embedding
  let mut projectionIds : Array LayerProjectionIds := #[]
  for layerIndex in [:step.weights.layers.size] do
    let some layerWeights := step.weights.layers[layerIndex]?
      | throw <| IO.userError "full-training megakernel missing layer weights"
    let some layerForward := step.forward.layers[layerIndex]?
      | throw <| IO.userError "full-training megakernel missing layer activations"
    let mixerIds ← match layerWeights.mixer, layerForward.mixer with
      | .deltaNet weights, .deltaNet buffers =>
          emitDeltaForward state weights buffers input layout shape
      | .attention weights, .attention buffers =>
          emitAttentionForward state weights buffers input layout shape
      | _, _ => throw <| IO.userError "full-training megakernel mixer forward-kind mismatch"
    let mixerOutput := match layerForward.mixer with
      | .deltaNet buffers => buffers.output
      | .attention buffers => buffers.output
    emitAdd state input mixerOutput layerForward.mixerResidual (shape.tokens * shape.hidden)
    let (gate, up, down) ← emitMlpForward state layerWeights.mlp layerForward.mlp
      layerForward.mixerResidual shape
    emitAdd state layerForward.mixerResidual layerForward.mlp.output layerForward.output
      (shape.tokens * shape.hidden)
    projectionIds := projectionIds.push { mixer := mixerIds, gate, up, down }
    input := layerForward.output
  emitRmsForward state input step.weights.finalNorm step.forward.finalHidden
    step.forward.finalInverseRms shape.tokens shape.hidden shape.epsilon
  let lmHead ← registerProjection state step.weights.lmHead shape.tokens shape.hidden shape.vocabulary
  emitProjectionForward state lmHead step.forward.finalHidden step.forward.logits
  let operationsAfterForward ← state.operations.get
  let forwardOperations := (operationsAfterForward.size / operationWords.toNat).toUInt32
  emitProjectionBackward state lmHead step.forward.finalHidden step.forward.logitGradient
    step.backward.finalHiddenGradient
  emitRmsBackward state input step.weights.finalNorm step.backward.finalHiddenGradient
    step.backward.decoderOutputGradient step.forward.finalInverseRms shape.tokens shape.hidden
  let mut outputGradient := step.backward.decoderOutputGradient
  for reverseIndex in [:step.weights.layers.size] do
    let layerIndex := step.weights.layers.size - 1 - reverseIndex
    let some layerWeights := step.weights.layers[layerIndex]?
      | throw <| IO.userError "full-training megakernel reverse layer weights missing"
    let some layerForward := step.forward.layers[layerIndex]?
      | throw <| IO.userError "full-training megakernel reverse layer activations missing"
    let some layerBackward := step.backward.layers[layerIndex]?
      | throw <| IO.userError "full-training megakernel reverse layer scratch missing"
    let some ids := projectionIds[layerIndex]?
      | throw <| IO.userError "full-training megakernel projection registry missing"
    let layerInput ← if layerIndex == 0 then pure step.forward.embedding else
      match step.forward.layers[layerIndex - 1]? with
      | some previous => pure previous.output
      | none => throw <| IO.userError "full-training megakernel previous layer output missing"
    emitMlpBackward state ids layerWeights.mlp layerForward.mlp layerBackward.mlp
      layerForward.mixerResidual outputGradient shape
    emitAdd state outputGradient layerBackward.mlp.inputGradient
      layerBackward.mixerResidualGradient (shape.tokens * shape.hidden)
    match layerWeights.mixer, layerForward.mixer, layerBackward.mixer with
    | .deltaNet weights, .deltaNet forward, .deltaNet backward =>
        emitDeltaBackward state ids.mixer weights forward backward layout layerInput
          layerBackward.mixerResidualGradient shape
    | .attention weights, .attention forward, .attention backward =>
        emitAttentionBackward state ids.mixer weights forward backward layout layerInput
          layerBackward.mixerResidualGradient shape
    | _, _, _ => throw <| IO.userError "full-training megakernel mixer reverse-kind mismatch"
    let mixerInputGradient := match layerBackward.mixer with
      | .deltaNet buffers => buffers.inputGradient
      | .attention buffers => buffers.inputGradient
    emitAdd state layerBackward.mixerResidualGradient mixerInputGradient layerBackward.inputGradient
      (shape.tokens * shape.hidden)
    outputGradient := layerBackward.inputGradient
  let operationWordsHost ← state.operations.get
  let operationCount := (operationWordsHost.size / operationWords.toNat).toUInt32
  let projectionMetadataHost ← state.metadata.get
  let projectionCount := (projectionMetadataHost.size /
    Internal.projectionMetadataWords.toNat).toUInt32
  unless projectionCount == lmHead + 1 do
    throw <| IO.userError "full-training megakernel projection registry is not contiguous"
  let operationBuffer ← Cuda.Buffer.alloc UInt32 operationWordsHost.size.toUSize
  operationBuffer.copyFrom (packUInt32 operationWordsHost) stream
  let metadataBuffer ← Cuda.Buffer.alloc UInt32 projectionMetadataHost.size.toUSize
  metadataBuffer.copyFrom (packUInt32 projectionMetadataHost) stream
  let activationOwners ← state.activations.get
  let matrixOwners ← state.matrices.get
  let matrixOffsets ← state.matrixOffsets.get
  let adapterOwners ← state.adapters.get
  let indexOwners ← state.indices.get
  let activations ← Cuda.BufferTable.create activationOwners
  let matrices ← Cuda.BufferTable.createAtByteOffsets matrixOwners matrixOffsets
  let adapters ← Cuda.BufferTable.create adapterOwners
  let indices ← Cuda.BufferTable.create indexOwners
  let maxRoles := max (layout.batchSize * shape.linearValueHeads)
    (shape.tokens * shape.attentionQueryHeads)
  let blocks := min 288 (max 1 maxRoles)
  let sharedWords := max layout.sequenceLength (3 * shape.linearValueWidth)
  return {
    operations := operationBuffer, activations, adapters, matrices, indices,
    projectionMetadata := metadataBuffer,
    descriptor := {
      forwardOperations, operationCount, projectionCount, adapterIndexBase := 2
    },
    rows := shape.tokens, classes := shape.vocabulary,
    blocks, sharedMemoryBytes := (sharedWords * 4).toUSize,
    logits := step.forward.logits, logitGradient := step.forward.logitGradient,
    rowLoss := step.forward.rowLoss, loss := step.forward.loss, targets := step.targets
  }

/-- Cooperative launch geometry selected by the lowered model and sequence shape. -/
def launchConfig (program : @& Program) : Cuda.LaunchConfig := {
  grid := { x := program.blocks }
  block := { x := 32 }
  sharedMemoryBytes := program.sharedMemoryBytes
  blockArenaBytes := 0
  cooperative := true
}

end Program

end Cuda.Qwen36.FullTrainingMegakernel
