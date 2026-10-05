/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Primitives
public import Lean.Cuda.Training.Gemm.GB10

public section

/-!
# Qwen3.6 dense projection baseline

Generic Float32 `Y = X W^T` forward, input VJP, and weight VJP bodies for row-major activations
and HF-style `[out, in]` weights. These scalar kernels are the numerical baseline for tiny oracle
gates and decode tails; production Qwen shapes route tile-compatible calls through the native
BF16 GEMM runners. The element bodies remain exposed so persistent schedules can reuse exactly
the same equations without launching child kernels.
-/

namespace Cuda.Qwen36.Linear

open Cuda.Qwen36.Primitives

/-- Zero-copy BF16 base matrix plus reusable mixed-precision operand scratch. -/
structure FrozenBFloat16F32 where
  weight : FrozenBFloat16Weight
  inputBF16 : Cuda.Buffer Cuda.BFloat16
  outputGradientBF16 : Cuda.Buffer Cuda.BFloat16

namespace FrozenBFloat16F32

/-- Check the checkpoint range and the two shared operand buffers for one projection shape. -/
def check (weights : @& FrozenBFloat16F32) (rows inputFeatures outputFeatures : UInt32) :
    IO Unit := do
  let requiredWeightBytes := outputFeatures.toUInt64 * inputFeatures.toUInt64 * 2
  unless weights.weight.byteCount == requiredWeightBytes do
    throw <| IO.userError "Qwen3.6 frozen BF16 projection size mismatch"
  unless weights.weight.byteOffset + weights.weight.byteCount <=
      (← weights.weight.owner.byteSize).toUInt64 do
    throw <| IO.userError "Qwen3.6 frozen BF16 projection range exceeds its owner"
  unless (← weights.inputBF16.byteSize) >= (rows * inputFeatures * 2).toUSize do
    throw <| IO.userError "Qwen3.6 frozen BF16 input scratch is too small"
  unless (← weights.outputGradientBF16.byteSize) >= (rows * outputFeatures * 2).toUSize do
    throw <| IO.userError "Qwen3.6 frozen BF16 output-gradient scratch is too small"

end FrozenBFloat16F32

@[struct] structure Shape where
  rows : UInt32
  inputFeatures : UInt32
  outputFeatures : UInt32

namespace Internal

@[always_inline]
partial def forwardDotBF16 (input weight : Cuda.DevicePtr Cuda.BFloat16) (shape : Shape)
    (row outputFeature inputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if inputFeature < shape.inputFeatures then
    let x := (← Cuda.loadBFloat16 input
      (row * shape.inputFeatures + inputFeature).toUSize).toFloat32
    let w := (← Cuda.loadBFloat16 weight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize).toFloat32
    forwardDotBF16 input weight shape row outputFeature (inputFeature + 1)
      (Cuda.fma x w accumulator)
  else
    return accumulator

@[always_inline]
partial def inputGradientDotBF16 (outputGradient weight : Cuda.DevicePtr Cuda.BFloat16)
    (shape : Shape) (row inputFeature outputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if outputFeature < shape.outputFeatures then
    let dy := (← Cuda.loadBFloat16 outputGradient
      (row * shape.outputFeatures + outputFeature).toUSize).toFloat32
    let w := (← Cuda.loadBFloat16 weight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize).toFloat32
    inputGradientDotBF16 outputGradient weight shape row inputFeature (outputFeature + 1)
      (Cuda.fma dy w accumulator)
  else
    return accumulator

@[always_inline]
partial def forwardDotF32 (input weight : Cuda.DevicePtr Float32) (shape : Shape)
    (row outputFeature inputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if inputFeature < shape.inputFeatures then
    let x ← Cuda.loadFloat32 input (row * shape.inputFeatures + inputFeature).toUSize
    let w ← Cuda.loadFloat32 weight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize
    forwardDotF32 input weight shape row outputFeature (inputFeature + 1)
      (Cuda.fma x w accumulator)
  else
    return accumulator

@[always_inline]
partial def inputGradientDotF32 (outputGradient weight : Cuda.DevicePtr Float32)
    (shape : Shape) (row inputFeature outputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if outputFeature < shape.outputFeatures then
    let dy ← Cuda.loadFloat32 outputGradient
      (row * shape.outputFeatures + outputFeature).toUSize
    let w ← Cuda.loadFloat32 weight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize
    inputGradientDotF32 outputGradient weight shape row inputFeature (outputFeature + 1)
      (Cuda.fma dy w accumulator)
  else
    return accumulator

@[always_inline]
partial def weightGradientDotF32 (input outputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (outputFeature inputFeature row : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if row < shape.rows then
    let x ← Cuda.loadFloat32 input (row * shape.inputFeatures + inputFeature).toUSize
    let dy ← Cuda.loadFloat32 outputGradient
      (row * shape.outputFeatures + outputFeature).toUSize
    weightGradientDotF32 input outputGradient shape outputFeature inputFeature (row + 1)
      (Cuda.fma dy x accumulator)
  else
    return accumulator

@[always_inline]
partial def forwardDotRoundedBF16F32 (input : Cuda.DevicePtr Float32)
    (weight : Cuda.DevicePtr Cuda.BFloat16) (shape : Shape)
    (row outputFeature inputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if inputFeature < shape.inputFeatures then
    let x := (Cuda.BFloat16.ofFloat32
      (← Cuda.loadFloat32 input (row * shape.inputFeatures + inputFeature).toUSize)).toFloat32
    let w := (← Cuda.loadBFloat16 weight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize).toFloat32
    forwardDotRoundedBF16F32 input weight shape row outputFeature (inputFeature + 1)
      (Cuda.fma x w accumulator)
  else
    return accumulator

@[always_inline]
partial def inputGradientDotRoundedBF16F32 (outputGradient : Cuda.DevicePtr Float32)
    (weight : Cuda.DevicePtr Cuda.BFloat16) (shape : Shape)
    (row inputFeature outputFeature : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if outputFeature < shape.outputFeatures then
    let dy := (Cuda.BFloat16.ofFloat32 (← Cuda.loadFloat32 outputGradient
      (row * shape.outputFeatures + outputFeature).toUSize)).toFloat32
    let w := (← Cuda.loadBFloat16 weight
      (outputFeature * shape.inputFeatures + inputFeature).toUSize).toFloat32
    inputGradientDotRoundedBF16F32 outputGradient weight shape row inputFeature
      (outputFeature + 1) (Cuda.fma dy w accumulator)
  else
    return accumulator

end Internal

/-- Scalar frozen-BF16 projection element over Float32 activations. The activation is rounded to
BF16 at the multiply boundary, matching the production mixed-precision projection contract. -/
@[expose, cuda_device, always_inline]
def forwardFrozenBF16ElementF32 (input : Cuda.DevicePtr Float32)
    (weight : Cuda.DevicePtr Cuda.BFloat16) (output : Cuda.DevicePtr Float32) (shape : Shape)
    (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.outputFeatures then
    let row := linear / shape.outputFeatures
    let outputFeature := linear % shape.outputFeatures
    Cuda.storeFloat32 output linear.toUSize
      (← Internal.forwardDotRoundedBF16F32 input weight shape row outputFeature 0 zeroF32)

/-- Scalar frozen-BF16 input VJP over a Float32 upstream gradient. The upstream value is rounded
to BF16 at the multiply boundary, exactly like the production GB10 operand staging path. -/
@[expose, cuda_device, always_inline]
def backwardInputFrozenBF16ElementF32 (outputGradient : Cuda.DevicePtr Float32)
    (weight : Cuda.DevicePtr Cuda.BFloat16) (inputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.inputFeatures then
    let row := linear / shape.inputFeatures
    let inputFeature := linear % shape.inputFeatures
    Cuda.storeFloat32 inputGradient linear.toUSize
      (← Internal.inputGradientDotRoundedBF16F32 outputGradient weight shape row inputFeature 0
        zeroF32)

@[cuda_kernel]
def forwardFrozenBF16ScalarKernel (input : Cuda.DevicePtr Cuda.BFloat16)
    (weightBytes : Cuda.DevicePtr UInt8) (weightByteOffset : UInt64)
    (output : Cuda.DevicePtr Float32) (rows inputFeatures outputFeatures : UInt32) :
    Cuda.DeviceM Unit := do
  let shape : Shape := { rows, inputFeatures, outputFeatures }
  let linear ← elementIndex
  if linear < rows * outputFeatures then
    let weight : Cuda.DevicePtr Cuda.BFloat16 := weightBytes + weightByteOffset.toUSize
    let row := linear / outputFeatures
    let outputFeature := linear % outputFeatures
    let value ← Internal.forwardDotBF16 input weight shape row outputFeature 0 zeroF32
    Cuda.storeFloat32 output linear.toUSize value

@[cuda_kernel]
def backwardInputFrozenBF16ScalarKernel (outputGradient : Cuda.DevicePtr Cuda.BFloat16)
    (weightBytes : Cuda.DevicePtr UInt8) (weightByteOffset : UInt64)
    (inputGradient : Cuda.DevicePtr Float32) (rows inputFeatures outputFeatures : UInt32) :
    Cuda.DeviceM Unit := do
  let shape : Shape := { rows, inputFeatures, outputFeatures }
  let linear ← elementIndex
  if linear < rows * inputFeatures then
    let weight : Cuda.DevicePtr Cuda.BFloat16 := weightBytes + weightByteOffset.toUSize
    let row := linear / inputFeatures
    let inputFeature := linear % inputFeatures
    let value ← Internal.inputGradientDotBF16 outputGradient weight shape row inputFeature 0 zeroF32
    Cuda.storeFloat32 inputGradient linear.toUSize value

@[cuda_kernel]
def forwardFrozenBF16GB10Kernel (input : Cuda.DevicePtr Cuda.BFloat16)
    (weightBytes : Cuda.DevicePtr UInt8) (weightByteOffset : UInt64)
    (output : Cuda.DevicePtr Float32) (rows inputFeatures outputFeatures : UInt32) :
    Cuda.DeviceM Unit := do
  let weight : Cuda.DevicePtr Cuda.BFloat16 := weightBytes + weightByteOffset.toUSize
  -- `dynamicShared` takes a byte offset; the launch config owns the allocation size.
  let scratch ← Cuda.dynamicShared (α := UInt8)
  Cuda.Training.Gemm.GB10.runNT {
    left := input, right := weight, output
    m := rows, n := outputFeatures, k := inputFeatures
  } scratch

@[cuda_kernel]
def backwardInputFrozenBF16GB10Kernel (outputGradient : Cuda.DevicePtr Cuda.BFloat16)
    (weightBytes : Cuda.DevicePtr UInt8) (weightByteOffset : UInt64)
    (inputGradient : Cuda.DevicePtr Float32) (rows inputFeatures outputFeatures : UInt32) :
    Cuda.DeviceM Unit := do
  let weight : Cuda.DevicePtr Cuda.BFloat16 := weightBytes + weightByteOffset.toUSize
  -- Keep the scratch base at offset zero; passing `sharedBytes` here points past the allocation.
  let scratch ← Cuda.dynamicShared (α := UInt8)
  Cuda.Training.Gemm.GB10.runNN {
    left := outputGradient, right := weight, output := inputGradient
    m := rows, n := inputFeatures, k := outputFeatures
  } scratch

/-- Evaluate and publish one element of `X W^T`. -/
@[expose, cuda_device, always_inline]
def forwardElementF32 (input weight output : Cuda.DevicePtr Float32) (shape : Shape)
    (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.outputFeatures then
    let row := linear / shape.outputFeatures
    let outputFeature := linear % shape.outputFeatures
    let value ← Internal.forwardDotF32 input weight shape row outputFeature 0 zeroF32
    Cuda.storeFloat32 output linear.toUSize value

/-- Evaluate and publish one element of the input VJP `dY W`. -/
@[expose, cuda_device, always_inline]
def backwardInputElementF32 (outputGradient weight inputGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.rows * shape.inputFeatures then
    let row := linear / shape.inputFeatures
    let inputFeature := linear % shape.inputFeatures
    let value ← Internal.inputGradientDotF32 outputGradient weight shape row inputFeature 0 zeroF32
    Cuda.storeFloat32 inputGradient linear.toUSize value

/-- Evaluate and publish one element of the weight VJP `dY^T X`. -/
@[expose, cuda_device, always_inline]
def backwardWeightElementF32 (input outputGradient weightGradient : Cuda.DevicePtr Float32)
    (shape : Shape) (linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < shape.outputFeatures * shape.inputFeatures then
    let outputFeature := linear / shape.inputFeatures
    let inputFeature := linear % shape.inputFeatures
    let value ← Internal.weightGradientDotF32 input outputGradient shape outputFeature inputFeature 0
      zeroF32
    Cuda.storeFloat32 weightGradient linear.toUSize value

/-- Add one pair of contiguous Float32 elements. -/
@[expose, cuda_device, always_inline]
def addElementF32 (left right output : Cuda.DevicePtr Float32) (count linear : UInt32) :
    Cuda.DeviceM Unit := do
  if linear < count then
    let x ← Cuda.loadFloat32 left linear.toUSize
    let y ← Cuda.loadFloat32 right linear.toUSize
    Cuda.storeFloat32 output linear.toUSize (x + y)

@[cuda_kernel]
def forwardF32Kernel (input weight output : Cuda.DevicePtr Float32)
    (rows inputFeatures outputFeatures : UInt32) : Cuda.DeviceM Unit := do
  forwardElementF32 input weight output { rows, inputFeatures, outputFeatures } (← elementIndex)

@[cuda_kernel]
def backwardInputF32Kernel (outputGradient weight inputGradient : Cuda.DevicePtr Float32)
    (rows inputFeatures outputFeatures : UInt32) : Cuda.DeviceM Unit := do
  backwardInputElementF32 outputGradient weight inputGradient { rows, inputFeatures, outputFeatures }
    (← elementIndex)

@[cuda_kernel]
def backwardWeightF32Kernel (input outputGradient weightGradient : Cuda.DevicePtr Float32)
    (rows inputFeatures outputFeatures : UInt32) : Cuda.DeviceM Unit := do
  backwardWeightElementF32 input outputGradient weightGradient { rows, inputFeatures, outputFeatures }
    (← elementIndex)

@[cuda_kernel]
def addF32Kernel (left right output : Cuda.DevicePtr Float32) (count : UInt32) :
    Cuda.DeviceM Unit := do
  addElementF32 left right output count (← elementIndex)

/-- Numerical Float32 projection baseline `Y = X W^T`. -/
def forwardF32 (stream : @& Cuda.Stream) (input weight output : @& Cuda.Buffer Float32)
    (rows inputFeatures outputFeatures : UInt32) : IO Cuda.KernelHandle :=
  forwardF32Kernel.launchOn stream (elementConfig (rows * outputFeatures)) input weight output rows
    inputFeatures outputFeatures

/-- Numerical Float32 input VJP baseline. -/
def backwardInputF32 (stream : @& Cuda.Stream)
    (outputGradient weight inputGradient : @& Cuda.Buffer Float32)
    (rows inputFeatures outputFeatures : UInt32) : IO Cuda.KernelHandle :=
  backwardInputF32Kernel.launchOn stream (elementConfig (rows * inputFeatures)) outputGradient
    weight inputGradient rows inputFeatures outputFeatures

/-- Numerical Float32 weight VJP baseline. -/
def backwardWeightF32 (stream : @& Cuda.Stream)
    (input outputGradient weightGradient : @& Cuda.Buffer Float32)
    (rows inputFeatures outputFeatures : UInt32) : IO Cuda.KernelHandle :=
  backwardWeightF32Kernel.launchOn stream (elementConfig (outputFeatures * inputFeatures)) input
    outputGradient weightGradient rows inputFeatures outputFeatures

/-- Add two contiguous Float32 buffers. -/
def addF32 (stream : @& Cuda.Stream) (left right output : @& Cuda.Buffer Float32)
    (count : UInt32) : IO Cuda.KernelHandle :=
  addF32Kernel.launchOn stream (elementConfig count) left right output count

/-! ## Shared sequential/device-graph submissions -/

def submitForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input weight output : @& Cuda.Buffer Float32)
    (rows inputFeatures outputFeatures : UInt32) : IO Unit :=
  let launch := elementConfig (rows * outputFeatures)
  executor.submit label
    (fun stream => forwardF32Kernel.launchOn stream launch input weight output rows
      inputFeatures outputFeatures)
    (fun builder => forwardF32Kernel.addToGraph builder launch input weight output rows
      inputFeatures outputFeatures)

def submitBackwardInputF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (outputGradient weight inputGradient : @& Cuda.Buffer Float32)
    (rows inputFeatures outputFeatures : UInt32) : IO Unit :=
  let launch := elementConfig (rows * inputFeatures)
  executor.submit label
    (fun stream => backwardInputF32Kernel.launchOn stream launch outputGradient weight
      inputGradient rows inputFeatures outputFeatures)
    (fun builder => backwardInputF32Kernel.addToGraph builder launch outputGradient weight
      inputGradient rows inputFeatures outputFeatures)

def submitBackwardWeightF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input outputGradient weightGradient : @& Cuda.Buffer Float32)
    (rows inputFeatures outputFeatures : UInt32) : IO Unit :=
  let launch := elementConfig (outputFeatures * inputFeatures)
  executor.submit label
    (fun stream => backwardWeightF32Kernel.launchOn stream launch input outputGradient
      weightGradient rows inputFeatures outputFeatures)
    (fun builder => backwardWeightF32Kernel.addToGraph builder launch input outputGradient
      weightGradient rows inputFeatures outputFeatures)

private def gb10Config (tiles : UInt32) : Cuda.LaunchConfig := {
  grid := { x := tiles }
  block := { x := 256 }
  sharedMemoryBytes := Cuda.Training.Gemm.GB10.sharedBytes.toUSize
  blockArenaBytes := 0
}

/--
Submit `X W^T` from an immutable BF16 checkpoint matrix. Tile-compatible real-model shapes use
the Lean-native GB10 runner; compact/non-multiple oracle shapes retain a deterministic scalar path.
-/
def submitForwardFrozenBF16F32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input : @& Cuda.Buffer Float32) (weights : @& FrozenBFloat16F32)
    (output : @& Cuda.Buffer Float32) (rows inputFeatures outputFeatures : UInt32) : IO Unit := do
  weights.check rows inputFeatures outputFeatures
  submitCastF32ToBF16 executor (label ++ " input BF16 cast") input weights.inputBF16
    (rows * inputFeatures)
  let owner := weights.weight.owner
  let offset := weights.weight.byteOffset
  if rows % 128 == 0 && outputFeatures % 64 == 0 && inputFeatures % 32 == 0 then
    let launch := gb10Config (rows / 128 * (outputFeatures / 64))
    executor.submit label
      (fun stream => forwardFrozenBF16GB10Kernel.launchOn stream launch weights.inputBF16 owner
        offset output rows inputFeatures outputFeatures)
      (fun builder => forwardFrozenBF16GB10Kernel.addToGraph builder launch weights.inputBF16 owner
        offset output rows inputFeatures outputFeatures)
  else
    let launch := elementConfig (rows * outputFeatures)
    executor.submit label
      (fun stream => forwardFrozenBF16ScalarKernel.launchOn stream launch weights.inputBF16 owner
        offset output rows inputFeatures outputFeatures)
      (fun builder => forwardFrozenBF16ScalarKernel.addToGraph builder launch weights.inputBF16 owner
        offset output rows inputFeatures outputFeatures)

/-- Submit the frozen BF16 input VJP `dX = dY W`, publishing Float32 gradients. -/
def submitBackwardInputFrozenBF16F32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (outputGradient : @& Cuda.Buffer Float32) (weights : @& FrozenBFloat16F32)
    (inputGradient : @& Cuda.Buffer Float32) (rows inputFeatures outputFeatures : UInt32) :
    IO Unit := do
  weights.check rows inputFeatures outputFeatures
  submitCastF32ToBF16 executor (label ++ " output-gradient BF16 cast") outputGradient
    weights.outputGradientBF16 (rows * outputFeatures)
  let owner := weights.weight.owner
  let offset := weights.weight.byteOffset
  if rows % 128 == 0 && inputFeatures % 64 == 0 && outputFeatures % 32 == 0 then
    let launch := gb10Config (rows / 128 * (inputFeatures / 64))
    executor.submit label
      (fun stream => backwardInputFrozenBF16GB10Kernel.launchOn stream launch
        weights.outputGradientBF16 owner offset inputGradient rows inputFeatures outputFeatures)
      (fun builder => backwardInputFrozenBF16GB10Kernel.addToGraph builder launch
        weights.outputGradientBF16 owner offset inputGradient rows inputFeatures outputFeatures)
  else
    let launch := elementConfig (rows * inputFeatures)
    executor.submit label
      (fun stream => backwardInputFrozenBF16ScalarKernel.launchOn stream launch
        weights.outputGradientBF16 owner offset inputGradient rows inputFeatures outputFeatures)
      (fun builder => backwardInputFrozenBF16ScalarKernel.addToGraph builder launch
        weights.outputGradientBF16 owner offset inputGradient rows inputFeatures outputFeatures)

def submitAddF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (left right output : @& Cuda.Buffer Float32) (count : UInt32) : IO Unit :=
  let launch := elementConfig count
  executor.submit label
    (fun stream => addF32Kernel.launchOn stream launch left right output count)
    (fun builder => addF32Kernel.addToGraph builder launch left right output count)

end Cuda.Qwen36.Linear
