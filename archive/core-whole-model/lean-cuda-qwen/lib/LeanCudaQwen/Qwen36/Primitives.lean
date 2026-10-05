/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import Lean.Cuda.Collective
public import Lean.Cuda.Half
public import Lean.Cuda.Launch
import Lean.Compiler.StructAttr

public section

/-!
# Qwen3.6 shared CUDA device primitives

Device bodies, finite kernels, and typed host launchers for the operations every Qwen3.6 stage
builds on: the zero-init `(1 + w)` RMSNorm over the last dimension, the unit-weight gated
RMSNorm `rmsnorm(x) * silu(z)` of the DeltaNet output, partial GPT-NeoX RoPE with its
inverse-frequency table, the causal depthwise conv1d (kernel 4, SiLU) in full-sequence and
state-carrying decode forms, per-head l2 normalization, SiLU and SwiGLU, embedding gathers,
and BF16↔FP32 casts.

Row-parallel bodies assign one warp per row: reductions are full-warp butterflies, so no shared
memory or block synchronization crosses these bodies, and every lane recomputes the row scalar
redundantly instead of publishing it through memory. Elementwise kernels take one thread per
element. All math is exact IEEE Float32 (`expf`, `sqrtf`, `sinf`, `cosf`): these are the
reference bodies the `1e-5` fixture gates measure, so they deliberately avoid the fast-math
intrinsics used by the production attention kernels. Accumulation is always Float32, including
the BF16 I/O variants, which round only at loads and stores.

SiLU/SwiGLU restate the `Lean.Cuda.MoE.KernelBody` tile formulas; the tile bodies there are
BF16-only with routed-tile strides, while these primitives need FP32 and dense contiguous forms.

Kernel names end in `Kernel`; the same name without the suffix is the typed host wrapper, which
sizes the launch from the logical shape and forwards to the generated launcher. Row-parallel
wrappers launch `rowWarps` warps per block; elementwise wrappers launch 128-thread blocks.
-/

namespace Cuda.Qwen36

/-- Explicit row-major sequence boundaries for flattened `[batch, sequence, ...]` tensors. -/
structure SequenceLayout where
  batchSize : UInt32
  sequenceLength : UInt32
  deriving Repr

namespace SequenceLayout

/-- Flattened row count without losing the original sequence boundary. -/
def tokens (layout : SequenceLayout) : UInt32 :=
  layout.batchSize * layout.sequenceLength

/-- One independent sequence containing every row. -/
def single (tokens : UInt32) : SequenceLayout := {
  batchSize := 1
  sequenceLength := tokens
}

/-- Reject empty layouts and UInt32 multiplication overflow. -/
def check (layout : SequenceLayout) : Except String SequenceLayout := do
  if layout.batchSize == 0 || layout.sequenceLength == 0 then
    throw "Qwen3.6 sequence layout dimensions must be positive"
  let rows := layout.batchSize.toUInt64 * layout.sequenceLength.toUInt64
  if rows > 0xffffffff then
    throw "Qwen3.6 sequence layout row count exceeds UInt32"
  return layout

end SequenceLayout

/-- Where a finite Qwen operation sequence is submitted. -/
inductive Executor where
  | sequential (stream : Cuda.Stream)
  | graph (builder : Cuda.GraphBuilder)

namespace Executor

/-- Submit one finite kernel in the selected mode while preserving a common operation order. -/
def submit (executor : @& Executor) (label : String)
    (launch : Cuda.Stream → IO Cuda.KernelHandle)
    (append : Cuda.GraphBuilder → IO Unit) : IO Unit :=
  match executor with
  | .sequential stream => do
      if (← IO.getEnv "QWEN_TRAIN_TRACE_KERNELS") == some "1" then
        IO.eprintln s!"Qwen3.8 CUDA: {label}"
      (← launch stream).waitChecked label
  | .graph builder =>
      append builder

end Executor

end Cuda.Qwen36

namespace Cuda.Qwen36.Primitives

/-- An immutable BF16 tensor retained inside one owning checkpoint allocation. -/
structure FrozenBFloat16Weight where
  owner : Cuda.Buffer UInt8
  byteOffset : UInt64
  byteCount : UInt64

/-- Float32 zero as an exact device bit pattern. -/
@[expose, cuda_device, always_inline] def zeroF32 : Float32 := Float32.ofBits 0

/-- Float32 one as an exact device bit pattern. -/
@[expose, cuda_device, always_inline] def oneF32 : Float32 := Float32.ofBits 0x3f800000

/-- Warps per block in row-parallel launches; `rowWarpConfig` matches it. -/
@[expose, cuda_device, always_inline] def rowWarps : UInt32 := 4

/-- Block-linear row owned by the calling warp in a row-parallel launch. -/
@[expose, cuda_device, always_inline] def warpRow : Cuda.DeviceM UInt32 := do
  return (← Cuda.blockIdxX) * rowWarps + (← Cuda.warpId)

/-- Grid-linear element index owned by the calling thread. -/
@[expose, cuda_device, always_inline] def elementIndex : Cuda.DeviceM UInt32 := do
  return (← Cuda.blockIdxX) * (← Cuda.blockDimX) + (← Cuda.threadIdxX)

/-- Exact Float32 sigmoid `1 / (1 + exp(-x))`. -/
@[expose, cuda_device, always_inline] def sigmoidF32 (x : Float32) : Float32 :=
  oneF32 / (oneF32 + Float32.exp (-x))

/-- Exact Float32 SiLU `x * sigmoid(x)`. -/
@[expose, cuda_device, always_inline] def siluF32 (x : Float32) : Float32 :=
  x * sigmoidF32 x

/-- Exact Float32 SiLU derivative `σ(x) * (1 + x * (1 - σ(x)))`. -/
@[expose, cuda_device, always_inline] def siluGradF32 (x : Float32) : Float32 :=
  let s := sigmoidF32 x
  s * (oneF32 + x * (oneF32 - s))

/-- Sum one Float32 per lane and broadcast the complete lane-zero reduction to the warp. -/
@[expose, cuda_device, always_inline, convergent]
def warpSumBroadcast (value : Float32) : Cuda.DeviceM Float32 := do
  let total ← Cuda.Collective.warpSum value
  return Float32.ofBits (← Cuda.shuffleUInt32 total.toBits 0)

namespace Internal

/-- Lane-strided Float32 sum of squares over `x[base, base + width)`; stride is one warp. -/
@[always_inline] partial def squareSumF32 (x : Cuda.DevicePtr Float32) (base width lane : UInt32)
    (acc : Float32) : Cuda.DeviceM Float32 := do
  if lane >= width then
    return acc
  let value ← Cuda.loadFloat32 x (base + lane).toUSize
  squareSumF32 x base width (lane + 32) (Cuda.fma value value acc)

/-- Lane-strided Float32 sum of squares over a BF16 row; accumulation is Float32. -/
@[always_inline] partial def squareSumBF16 (x : Cuda.DevicePtr Cuda.BFloat16)
    (base width lane : UInt32) (acc : Float32) : Cuda.DeviceM Float32 := do
  if lane >= width then
    return acc
  let value := (← Cuda.loadBFloat16 x (base + lane).toUSize).toFloat32
  squareSumBF16 x base width (lane + 32) (Cuda.fma value value acc)

/-- Lane-strided `(1 + w)` RMSNorm publication of one normalized Float32 row. -/
@[always_inline] partial def storeNormedF32 (x weight output : Cuda.DevicePtr Float32)
    (base width lane : UInt32) (inv : Float32) : Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value ← Cuda.loadFloat32 x index
  let w ← Cuda.loadFloat32 weight lane.toUSize
  Cuda.storeFloat32 output index (value * inv * (oneF32 + w))
  storeNormedF32 x weight output base width (lane + 32) inv

/-- Lane-strided `(1 + w)` RMSNorm publication of one normalized row, BF16 I/O. -/
@[always_inline] partial def storeNormedBF16 (x weight output : Cuda.DevicePtr Cuda.BFloat16)
    (base width lane : UInt32) (inv : Float32) : Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value := (← Cuda.loadBFloat16 x index).toFloat32
  let w := (← Cuda.loadBFloat16 weight lane.toUSize).toFloat32
  Cuda.storeBFloat16 output index (Cuda.BFloat16.ofFloat32 (value * inv * (oneF32 + w)))
  storeNormedBF16 x weight output base width (lane + 32) inv

/-- Lane-strided dot `Σ dy·(1+w)·x` over one Float32 row for the RMSNorm input VJP. -/
@[always_inline] partial def dotWeightedF32 (outputGradient x weight : Cuda.DevicePtr Float32)
    (base width lane : UInt32) (acc : Float32) : Cuda.DeviceM Float32 := do
  if lane >= width then
    return acc
  let index := (base + lane).toUSize
  let d ← Cuda.loadFloat32 outputGradient index
  let value ← Cuda.loadFloat32 x index
  let w ← Cuda.loadFloat32 weight lane.toUSize
  dotWeightedF32 outputGradient x weight base width (lane + 32)
    (Cuda.fma (d * (oneF32 + w)) value acc)

/-- Lane-strided dot `Σ dy·(1+w)·x` over one BF16 row; accumulation is Float32. -/
@[always_inline] partial def dotWeightedBF16 (outputGradient x weight :
    Cuda.DevicePtr Cuda.BFloat16) (base width lane : UInt32) (acc : Float32) :
    Cuda.DeviceM Float32 := do
  if lane >= width then
    return acc
  let index := (base + lane).toUSize
  let d := (← Cuda.loadBFloat16 outputGradient index).toFloat32
  let value := (← Cuda.loadBFloat16 x index).toFloat32
  let w := (← Cuda.loadBFloat16 weight lane.toUSize).toFloat32
  dotWeightedBF16 outputGradient x weight base width (lane + 32)
    (Cuda.fma (d * (oneF32 + w)) value acc)

/-- Lane-strided RMSNorm input VJP `inv·dy·(1+w) - x·correction` over one Float32 row. -/
@[always_inline] partial def storeInputGradF32 (x weight outputGradient inputGradient :
    Cuda.DevicePtr Float32) (base width lane : UInt32) (inv correction : Float32) :
    Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value ← Cuda.loadFloat32 x index
  let w ← Cuda.loadFloat32 weight lane.toUSize
  let d ← Cuda.loadFloat32 outputGradient index
  Cuda.storeFloat32 inputGradient index
    (Cuda.fma (-value) correction (d * (oneF32 + w) * inv))
  storeInputGradF32 x weight outputGradient inputGradient base width (lane + 32) inv correction

/-- Lane-strided RMSNorm input VJP over one row, BF16 I/O with Float32 accumulation. -/
@[always_inline] partial def storeInputGradBF16 (x weight outputGradient inputGradient :
    Cuda.DevicePtr Cuda.BFloat16) (base width lane : UInt32) (inv correction : Float32) :
    Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value := (← Cuda.loadBFloat16 x index).toFloat32
  let w := (← Cuda.loadBFloat16 weight lane.toUSize).toFloat32
  let d := (← Cuda.loadBFloat16 outputGradient index).toFloat32
  Cuda.storeBFloat16 inputGradient index
    (Cuda.BFloat16.ofFloat32 (Cuda.fma (-value) correction (d * (oneF32 + w) * inv)))
  storeInputGradBF16 x weight outputGradient inputGradient base width (lane + 32) inv correction

/-- Row loop of the uniquely-owned RMSNorm weight-gradient column `Σ_r dy·x·invRms`. -/
@[always_inline] partial def weightGradColumnF32 (x outputGradient inverseRms :
    Cuda.DevicePtr Float32) (rows width column row : UInt32) (acc : Float32) :
    Cuda.DeviceM Float32 := do
  if row >= rows then
    return acc
  let index := (row * width + column).toUSize
  let value ← Cuda.loadFloat32 x index
  let d ← Cuda.loadFloat32 outputGradient index
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  weightGradColumnF32 x outputGradient inverseRms rows width column (row + 1)
    (Cuda.fma d (value * inv) acc)

/-- Row loop of the uniquely-owned RMSNorm weight-gradient column, BF16 inputs. -/
@[always_inline] partial def weightGradColumnBF16 (x outputGradient :
    Cuda.DevicePtr Cuda.BFloat16) (inverseRms : Cuda.DevicePtr Float32)
    (rows width column row : UInt32) (acc : Float32) : Cuda.DeviceM Float32 := do
  if row >= rows then
    return acc
  let index := (row * width + column).toUSize
  let value := (← Cuda.loadBFloat16 x index).toFloat32
  let d := (← Cuda.loadBFloat16 outputGradient index).toFloat32
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  weightGradColumnBF16 x outputGradient inverseRms rows width column (row + 1)
    (Cuda.fma d (value * inv) acc)

/-- Lane-strided gated-RMSNorm publication `x·inv·silu(z)` over one Float32 row. -/
@[always_inline] partial def storeGatedF32 (x z output : Cuda.DevicePtr Float32)
    (base width lane : UInt32) (inv : Float32) : Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value ← Cuda.loadFloat32 x index
  let gate ← Cuda.loadFloat32 z index
  Cuda.storeFloat32 output index (value * inv * siluF32 gate)
  storeGatedF32 x z output base width (lane + 32) inv

/-- Lane-strided gated-RMSNorm publication over one row, BF16 I/O. -/
@[always_inline] partial def storeGatedBF16 (x z output : Cuda.DevicePtr Cuda.BFloat16)
    (base width lane : UInt32) (inv : Float32) : Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value := (← Cuda.loadBFloat16 x index).toFloat32
  let gate := (← Cuda.loadBFloat16 z index).toFloat32
  Cuda.storeBFloat16 output index (Cuda.BFloat16.ofFloat32 (value * inv * siluF32 gate))
  storeGatedBF16 x z output base width (lane + 32) inv

/-- Lane-strided dot `Σ dy·silu(z)·x` over one Float32 row for the gated-RMSNorm input VJP. -/
@[always_inline] partial def dotGatedF32 (outputGradient z x : Cuda.DevicePtr Float32)
    (base width lane : UInt32) (acc : Float32) : Cuda.DeviceM Float32 := do
  if lane >= width then
    return acc
  let index := (base + lane).toUSize
  let d ← Cuda.loadFloat32 outputGradient index
  let gate ← Cuda.loadFloat32 z index
  let value ← Cuda.loadFloat32 x index
  dotGatedF32 outputGradient z x base width (lane + 32)
    (Cuda.fma (d * siluF32 gate) value acc)

/-- Lane-strided dot `Σ dy·silu(z)·x` over one BF16 row; accumulation is Float32. -/
@[always_inline] partial def dotGatedBF16 (outputGradient z x : Cuda.DevicePtr Cuda.BFloat16)
    (base width lane : UInt32) (acc : Float32) : Cuda.DeviceM Float32 := do
  if lane >= width then
    return acc
  let index := (base + lane).toUSize
  let d := (← Cuda.loadBFloat16 outputGradient index).toFloat32
  let gate := (← Cuda.loadBFloat16 z index).toFloat32
  let value := (← Cuda.loadBFloat16 x index).toFloat32
  dotGatedBF16 outputGradient z x base width (lane + 32)
    (Cuda.fma (d * siluF32 gate) value acc)

/-- Lane-strided gated-RMSNorm VJP over one Float32 row: `dx = inv·dy·silu(z) - x·correction`,
`dz = dy·(x·inv)·silu'(z)`. -/
@[always_inline] partial def storeGatedGradsF32 (x z outputGradient inputGradient gateGradient :
    Cuda.DevicePtr Float32) (base width lane : UInt32) (inv correction : Float32) :
    Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value ← Cuda.loadFloat32 x index
  let gate ← Cuda.loadFloat32 z index
  let d ← Cuda.loadFloat32 outputGradient index
  Cuda.storeFloat32 inputGradient index
    (Cuda.fma (-value) correction (d * siluF32 gate * inv))
  Cuda.storeFloat32 gateGradient index (d * (value * inv) * siluGradF32 gate)
  storeGatedGradsF32 x z outputGradient inputGradient gateGradient base width (lane + 32) inv
    correction

/-- Lane-strided gated-RMSNorm VJP over one row, BF16 I/O with Float32 accumulation. -/
@[always_inline] partial def storeGatedGradsBF16 (x z outputGradient inputGradient gateGradient :
    Cuda.DevicePtr Cuda.BFloat16) (base width lane : UInt32) (inv correction : Float32) :
    Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value := (← Cuda.loadBFloat16 x index).toFloat32
  let gate := (← Cuda.loadBFloat16 z index).toFloat32
  let d := (← Cuda.loadBFloat16 outputGradient index).toFloat32
  Cuda.storeBFloat16 inputGradient index
    (Cuda.BFloat16.ofFloat32 (Cuda.fma (-value) correction (d * siluF32 gate * inv)))
  Cuda.storeBFloat16 gateGradient index
    (Cuda.BFloat16.ofFloat32 (d * (value * inv) * siluGradF32 gate))
  storeGatedGradsBF16 x z outputGradient inputGradient gateGradient base width (lane + 32) inv
    correction

/-- Weighted gated-RMSNorm publication `weight·x·inv·silu(z)`. -/
@[always_inline] partial def storeWeightedGatedF32 (x z weight output :
    Cuda.DevicePtr Float32) (base width lane : UInt32) (inv : Float32) : Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value ← Cuda.loadFloat32 x index
  let gate ← Cuda.loadFloat32 z index
  let weightValue ← Cuda.loadFloat32 weight lane.toUSize
  Cuda.storeFloat32 output index (weightValue * value * inv * siluF32 gate)
  storeWeightedGatedF32 x z weight output base width (lane + 32) inv

/-- Weighted gated-RMSNorm reduction `Σ dy·weight·silu(z)·x`. -/
@[always_inline] partial def dotWeightedGatedF32 (outputGradient z x weight :
    Cuda.DevicePtr Float32) (base width lane : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if lane >= width then
    return accumulator
  let index := (base + lane).toUSize
  let gradient ← Cuda.loadFloat32 outputGradient index
  let gate ← Cuda.loadFloat32 z index
  let value ← Cuda.loadFloat32 x index
  let weightValue ← Cuda.loadFloat32 weight lane.toUSize
  dotWeightedGatedF32 outputGradient z x weight base width (lane + 32)
    (Cuda.fma (gradient * weightValue * siluF32 gate) value accumulator)

/-- Weighted gated-RMSNorm input and gate VJP publication. -/
@[always_inline] partial def storeWeightedGatedGradsF32 (x z weight outputGradient inputGradient
    gateGradient : Cuda.DevicePtr Float32) (base width lane : UInt32)
    (inv correction : Float32) : Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value ← Cuda.loadFloat32 x index
  let gate ← Cuda.loadFloat32 z index
  let weightValue ← Cuda.loadFloat32 weight lane.toUSize
  let gradient ← Cuda.loadFloat32 outputGradient index
  Cuda.storeFloat32 inputGradient index
    (Cuda.fma (-value) correction (gradient * weightValue * siluF32 gate * inv))
  Cuda.storeFloat32 gateGradient index
    (gradient * weightValue * (value * inv) * siluGradF32 gate)
  storeWeightedGatedGradsF32 x z weight outputGradient inputGradient gateGradient base width
    (lane + 32) inv correction

/-- One trainable gated-RMSNorm weight-gradient column. -/
@[always_inline] partial def weightedGatedWeightGradColumnF32 (x z outputGradient inverseRms :
    Cuda.DevicePtr Float32) (rows width column row : UInt32) (accumulator : Float32) :
    Cuda.DeviceM Float32 := do
  if row >= rows then
    return accumulator
  let index := (row * width + column).toUSize
  let value ← Cuda.loadFloat32 x index
  let gate ← Cuda.loadFloat32 z index
  let gradient ← Cuda.loadFloat32 outputGradient index
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  weightedGatedWeightGradColumnF32 x z outputGradient inverseRms rows width column (row + 1)
    (Cuda.fma gradient (value * inv * siluF32 gate) accumulator)

/-- Lane-strided dot `Σ dy·y` over one Float32 row for the l2norm VJP; `y = x / den`. -/
@[always_inline] partial def dotNormalizedF32 (outputGradient x : Cuda.DevicePtr Float32)
    (base width lane : UInt32) (den : Float32) (acc : Float32) : Cuda.DeviceM Float32 := do
  if lane >= width then
    return acc
  let index := (base + lane).toUSize
  let d ← Cuda.loadFloat32 outputGradient index
  let value ← Cuda.loadFloat32 x index
  dotNormalizedF32 outputGradient x base width (lane + 32) den
    (Cuda.fma d (value / den) acc)

/-- Lane-strided l2norm publication `x / den` over one Float32 row. -/
@[always_inline] partial def storeNormalizedF32 (x output : Cuda.DevicePtr Float32)
    (base width lane : UInt32) (den : Float32) : Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  Cuda.storeFloat32 output index ((← Cuda.loadFloat32 x index) / den)
  storeNormalizedF32 x output base width (lane + 32) den

/-- Lane-strided l2norm VJP `(dy - y·dot) / den` over one Float32 row. -/
@[always_inline] partial def storeL2GradF32 (x outputGradient inputGradient :
    Cuda.DevicePtr Float32) (base width lane : UInt32) (den dot : Float32) :
    Cuda.DeviceM Unit := do
  if lane >= width then
    return
  let index := (base + lane).toUSize
  let value ← Cuda.loadFloat32 x index
  let d ← Cuda.loadFloat32 outputGradient index
  Cuda.storeFloat32 inputGradient index ((d - (value / den) * dot) / den)
  storeL2GradF32 x outputGradient inputGradient base width (lane + 32) den dot

/-- Causal depthwise conv1d tap sum `Σ_j w[c,j]·x[t-3+j, c]` with zero left padding. -/
@[always_inline] partial def convSumF32 (x weight : Cuda.DevicePtr Float32)
    (channels t c j : UInt32) (acc : Float32) : Cuda.DeviceM Float32 := do
  if j >= 4 then
    return acc
  let acc ←
    if t + j >= 3 then
      let xv ← Cuda.loadFloat32 x ((t + j - 3) * channels + c).toUSize
      let wv ← Cuda.loadFloat32 weight (c * 4 + j).toUSize
      pure (Cuda.fma wv xv acc)
    else
      pure acc
  convSumF32 x weight channels t c (j + 1) acc

/-- Row loop of the conv weight VJP `dw[c,j] = Σ_t x[t-3+j,c]·ds[t,c]`; `ds` recomputes the
pre-activation at every step. -/
@[always_inline] partial def convWeightGradFrom (x weight outputGradient :
    Cuda.DevicePtr Float32) (seq channels c j t : UInt32) (acc : Float32) :
    Cuda.DeviceM Float32 := do
  if t >= seq then
    return acc
  let u ← convSumF32 x weight channels t c 0 zeroF32
  let d ← Cuda.loadFloat32 outputGradient (t * channels + c).toUSize
  let xv ← Cuda.loadFloat32 x ((t + j - 3) * channels + c).toUSize
  convWeightGradFrom x weight outputGradient seq channels c j (t + 1)
    (Cuda.fma xv (d * siluGradF32 u) acc)

/-- Tap loop of the conv input VJP `dx[t,c] = Σ_j w[c,j]·ds[t+3-j,c]` over valid sources. -/
@[always_inline] partial def convInputGradFrom (x weight outputGradient :
    Cuda.DevicePtr Float32) (seq channels t c j : UInt32) (acc : Float32) :
    Cuda.DeviceM Float32 := do
  if j >= 4 then
    return acc
  let source := t + 3 - j
  let acc ←
    if source < seq then
      let u ← convSumF32 x weight channels source c 0 zeroF32
      let d ← Cuda.loadFloat32 outputGradient (source * channels + c).toUSize
      let wv ← Cuda.loadFloat32 weight (c * 4 + j).toUSize
      pure (Cuda.fma wv (d * siluGradF32 u) acc)
    else
      pure acc
  convInputGradFrom x weight outputGradient seq channels t c (j + 1) acc

/-- Batch loop for a causal-convolution weight VJP with independent sequence boundaries. -/
@[always_inline] partial def batchedConvWeightGradFrom (x weight outputGradient :
    Cuda.DevicePtr Float32) (batchSize sequenceLength channels c j batch : UInt32)
    (acc : Float32) : Cuda.DeviceM Float32 := do
  if batch >= batchSize then
    return acc
  let byteOffset := (batch * sequenceLength * channels * 4).toUSize
  let batchInput : Cuda.DevicePtr Float32 := x + byteOffset
  let batchGradient : Cuda.DevicePtr Float32 := outputGradient + byteOffset
  let acc ← convWeightGradFrom batchInput weight batchGradient sequenceLength channels c j
    (3 - j) acc
  batchedConvWeightGradFrom x weight outputGradient batchSize sequenceLength channels c j
    (batch + 1) acc

end Internal

/-! ## RMSNorm `(1 + w)` over the last dimension -/

/-- Forward body of one `(1 + w)` RMSNorm row, Float32 I/O with Float32 accumulation. Every lane
of the calling warp executes convergently; `row` indexes row-major `[rows, width]` buffers. The
per-row inverse RMS is published for the backward pass. -/
@[expose, cuda_device, always_inline]
def rmsNormRowFwdF32 (x weight output inverseRms : Cuda.DevicePtr Float32) (row width : UInt32)
    (epsilon : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let total ← warpSumBroadcast (← Internal.squareSumF32 x base width lane zeroF32)
  let inv := oneF32 / Float32.sqrt (total / width.toFloat32 + epsilon)
  if lane == 0 then
    Cuda.storeFloat32 inverseRms row.toUSize inv
  Internal.storeNormedF32 x weight output base width lane inv

/-- Forward body of one `(1 + w)` RMSNorm row, BF16 I/O with Float32 accumulation. -/
@[expose, cuda_device, always_inline]
def rmsNormRowFwdBF16 (x weight output : Cuda.DevicePtr Cuda.BFloat16)
    (inverseRms : Cuda.DevicePtr Float32) (row width : UInt32) (epsilon : Float32) :
    Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let total ← warpSumBroadcast (← Internal.squareSumBF16 x base width lane zeroF32)
  let inv := oneF32 / Float32.sqrt (total / width.toFloat32 + epsilon)
  if lane == 0 then
    Cuda.storeFloat32 inverseRms row.toUSize inv
  Internal.storeNormedBF16 x weight output base width lane inv

/-- Input-gradient body of one `(1 + w)` RMSNorm row, Float32 I/O. Consumes the inverse RMS
published by the forward body: `dx = inv·dy·(1+w) - x·(Σ dy·(1+w)·x)·inv³/width`. -/
@[expose, cuda_device, always_inline]
def rmsNormRowBwdF32 (x weight outputGradient inputGradient inverseRms :
    Cuda.DevicePtr Float32) (row width : UInt32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  let dot ← warpSumBroadcast
    (← Internal.dotWeightedF32 outputGradient x weight base width lane zeroF32)
  let correction := dot * inv * inv * inv / width.toFloat32
  Internal.storeInputGradF32 x weight outputGradient inputGradient base width lane inv correction

/-- Input-gradient body of one `(1 + w)` RMSNorm row, BF16 I/O with Float32 accumulation. -/
@[expose, cuda_device, always_inline]
def rmsNormRowBwdBF16 (x weight outputGradient inputGradient : Cuda.DevicePtr Cuda.BFloat16)
    (inverseRms : Cuda.DevicePtr Float32) (row width : UInt32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  let dot ← warpSumBroadcast
    (← Internal.dotWeightedBF16 outputGradient x weight base width lane zeroF32)
  let correction := dot * inv * inv * inv / width.toFloat32
  Internal.storeInputGradBF16 x weight outputGradient inputGradient base width lane inv correction

/-- One uniquely-owned RMSNorm weight-gradient column `dw[j] = Σ_r dy[r,j]·x[r,j]·invRms[r]`,
Float32 I/O. -/
@[expose, cuda_device, always_inline]
def rmsNormWeightGradF32 (x outputGradient inverseRms : Cuda.DevicePtr Float32)
    (rows width column : UInt32) : Cuda.DeviceM Float32 := do
  Internal.weightGradColumnF32 x outputGradient inverseRms rows width column 0 zeroF32

/-- One uniquely-owned RMSNorm weight-gradient column, BF16 inputs with Float32 accumulation. -/
@[expose, cuda_device, always_inline]
def rmsNormWeightGradBF16 (x outputGradient : Cuda.DevicePtr Cuda.BFloat16)
    (inverseRms : Cuda.DevicePtr Float32) (rows width column : UInt32) :
    Cuda.DeviceM Float32 := do
  Internal.weightGradColumnBF16 x outputGradient inverseRms rows width column 0 zeroF32

/-- `(1 + w)` RMSNorm forward, Float32 I/O: one warp per row of a `[rows, width]` activation. -/
@[cuda_kernel]
def rmsNormForwardF32Kernel (x weight output inverseRms : Cuda.DevicePtr Float32)
    (rows width : UInt32) (epsilon : Float32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    rmsNormRowFwdF32 x weight output inverseRms row width epsilon

/-- `(1 + w)` RMSNorm forward, BF16 I/O: one warp per row of a `[rows, width]` activation. -/
@[cuda_kernel]
def rmsNormForwardBF16Kernel (x weight output : Cuda.DevicePtr Cuda.BFloat16)
    (inverseRms : Cuda.DevicePtr Float32) (rows width : UInt32) (epsilon : Float32) :
    Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    rmsNormRowFwdBF16 x weight output inverseRms row width epsilon

/-- `(1 + w)` RMSNorm input VJP, Float32 I/O: one warp per row. -/
@[cuda_kernel]
def rmsNormBackwardInputF32Kernel (x weight outputGradient inputGradient inverseRms :
    Cuda.DevicePtr Float32) (rows width : UInt32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    rmsNormRowBwdF32 x weight outputGradient inputGradient inverseRms row width

/-- `(1 + w)` RMSNorm input VJP, BF16 I/O with Float32 accumulation: one warp per row. -/
@[cuda_kernel]
def rmsNormBackwardInputBF16Kernel (x weight outputGradient inputGradient :
    Cuda.DevicePtr Cuda.BFloat16) (inverseRms : Cuda.DevicePtr Float32) (rows width : UInt32) :
    Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    rmsNormRowBwdBF16 x weight outputGradient inputGradient inverseRms row width

/-- `(1 + w)` RMSNorm weight VJP, Float32 output: one thread per column, rows reduced serially. -/
@[cuda_kernel]
def rmsNormBackwardWeightF32Kernel (x outputGradient inverseRms weightGradient :
    Cuda.DevicePtr Float32) (rows width : UInt32) : Cuda.DeviceM Unit := do
  let column ← elementIndex
  if column < width then
    let gradient ← rmsNormWeightGradF32 x outputGradient inverseRms rows width column
    Cuda.storeFloat32 weightGradient column.toUSize gradient

/-- `(1 + w)` RMSNorm weight VJP, BF16 I/O with Float32 accumulation: one thread per column. -/
@[cuda_kernel]
def rmsNormBackwardWeightBF16Kernel (x outputGradient : Cuda.DevicePtr Cuda.BFloat16)
    (inverseRms : Cuda.DevicePtr Float32) (weightGradient : Cuda.DevicePtr Cuda.BFloat16)
    (rows width : UInt32) : Cuda.DeviceM Unit := do
  let column ← elementIndex
  if column < width then
    let gradient ← rmsNormWeightGradBF16 x outputGradient inverseRms rows width column
    Cuda.storeBFloat16 weightGradient column.toUSize (Cuda.BFloat16.ofFloat32 gradient)

/-! ## Gated RMSNorm (unit weight) -/

/-- Forward body of one unit-weight gated RMSNorm row `rmsnorm(x) * silu(z)`, Float32 I/O with
Float32 accumulation. Every lane of the calling warp executes convergently. -/
@[expose, cuda_device, always_inline]
def gatedRmsNormRowFwdF32 (x z output inverseRms : Cuda.DevicePtr Float32) (row width : UInt32)
    (epsilon : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let total ← warpSumBroadcast (← Internal.squareSumF32 x base width lane zeroF32)
  let inv := oneF32 / Float32.sqrt (total / width.toFloat32 + epsilon)
  if lane == 0 then
    Cuda.storeFloat32 inverseRms row.toUSize inv
  Internal.storeGatedF32 x z output base width lane inv

/-- Forward body of one gated RMSNorm row, BF16 I/O with Float32 accumulation. -/
@[expose, cuda_device, always_inline]
def gatedRmsNormRowFwdBF16 (x z output : Cuda.DevicePtr Cuda.BFloat16)
    (inverseRms : Cuda.DevicePtr Float32) (row width : UInt32) (epsilon : Float32) :
    Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let total ← warpSumBroadcast (← Internal.squareSumBF16 x base width lane zeroF32)
  let inv := oneF32 / Float32.sqrt (total / width.toFloat32 + epsilon)
  if lane == 0 then
    Cuda.storeFloat32 inverseRms row.toUSize inv
  Internal.storeGatedBF16 x z output base width lane inv

/-- Backward body of one gated RMSNorm row, Float32 I/O. Publishes both the input gradient
`dx = inv·dy·silu(z) - x·(Σ dy·silu(z)·x)·inv³/width` and the gate gradient
`dz = dy·(x·inv)·silu'(z)`. -/
@[expose, cuda_device, always_inline]
def gatedRmsNormRowBwdF32 (x z outputGradient inputGradient gateGradient inverseRms :
    Cuda.DevicePtr Float32) (row width : UInt32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  let dot ← warpSumBroadcast
    (← Internal.dotGatedF32 outputGradient z x base width lane zeroF32)
  let correction := dot * inv * inv * inv / width.toFloat32
  Internal.storeGatedGradsF32 x z outputGradient inputGradient gateGradient base width lane inv
    correction

/-- Backward body of one gated RMSNorm row, BF16 I/O with Float32 accumulation. -/
@[expose, cuda_device, always_inline]
def gatedRmsNormRowBwdBF16 (x z outputGradient inputGradient gateGradient :
    Cuda.DevicePtr Cuda.BFloat16) (inverseRms : Cuda.DevicePtr Float32) (row width : UInt32) :
    Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  let dot ← warpSumBroadcast
    (← Internal.dotGatedBF16 outputGradient z x base width lane zeroF32)
  let correction := dot * inv * inv * inv / width.toFloat32
  Internal.storeGatedGradsBF16 x z outputGradient inputGradient gateGradient base width lane inv
    correction

/-- Forward body of trainable-weight gated RMSNorm. -/
@[expose, cuda_device, always_inline]
def weightedGatedRmsNormRowFwdF32 (x z weight output inverseRms : Cuda.DevicePtr Float32)
    (row width : UInt32) (epsilon : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let total ← warpSumBroadcast (← Internal.squareSumF32 x base width lane zeroF32)
  let inv := oneF32 / Float32.sqrt (total / width.toFloat32 + epsilon)
  if lane == 0 then
    Cuda.storeFloat32 inverseRms row.toUSize inv
  Internal.storeWeightedGatedF32 x z weight output base width lane inv

/-- Input and gate VJP body of trainable-weight gated RMSNorm. -/
@[expose, cuda_device, always_inline]
def weightedGatedRmsNormRowBwdF32 (x z weight outputGradient inputGradient gateGradient
    inverseRms : Cuda.DevicePtr Float32) (row width : UInt32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let inv ← Cuda.loadFloat32 inverseRms row.toUSize
  let dot ← warpSumBroadcast
    (← Internal.dotWeightedGatedF32 outputGradient z x weight base width lane zeroF32)
  let correction := dot * inv * inv * inv / width.toFloat32
  Internal.storeWeightedGatedGradsF32 x z weight outputGradient inputGradient gateGradient base
    width lane inv correction

/-- Gated RMSNorm forward, Float32 I/O: one warp per row of `[rows, width]` activations. -/
@[cuda_kernel]
def gatedRmsNormForwardF32Kernel (x z output inverseRms : Cuda.DevicePtr Float32)
    (rows width : UInt32) (epsilon : Float32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    gatedRmsNormRowFwdF32 x z output inverseRms row width epsilon

/-- Gated RMSNorm forward, BF16 I/O: one warp per row of `[rows, width]` activations. -/
@[cuda_kernel]
def gatedRmsNormForwardBF16Kernel (x z output : Cuda.DevicePtr Cuda.BFloat16)
    (inverseRms : Cuda.DevicePtr Float32) (rows width : UInt32) (epsilon : Float32) :
    Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    gatedRmsNormRowFwdBF16 x z output inverseRms row width epsilon

/-- Gated RMSNorm VJP, Float32 I/O: one warp per row; publishes input and gate gradients. -/
@[cuda_kernel]
def gatedRmsNormBackwardF32Kernel (x z outputGradient inputGradient gateGradient inverseRms :
    Cuda.DevicePtr Float32) (rows width : UInt32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    gatedRmsNormRowBwdF32 x z outputGradient inputGradient gateGradient inverseRms row width

/-- Gated RMSNorm VJP, BF16 I/O with Float32 accumulation: one warp per row. -/
@[cuda_kernel]
def gatedRmsNormBackwardBF16Kernel (x z outputGradient inputGradient gateGradient :
    Cuda.DevicePtr Cuda.BFloat16) (inverseRms : Cuda.DevicePtr Float32) (rows width : UInt32) :
    Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    gatedRmsNormRowBwdBF16 x z outputGradient inputGradient gateGradient inverseRms row width

@[cuda_kernel]
def weightedGatedRmsNormForwardF32Kernel (x z weight output inverseRms :
    Cuda.DevicePtr Float32) (rows width : UInt32) (epsilon : Float32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    weightedGatedRmsNormRowFwdF32 x z weight output inverseRms row width epsilon

@[cuda_kernel]
def weightedGatedRmsNormBackwardF32Kernel (x z weight outputGradient inputGradient gateGradient
    inverseRms : Cuda.DevicePtr Float32) (rows width : UInt32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    weightedGatedRmsNormRowBwdF32 x z weight outputGradient inputGradient gateGradient inverseRms
      row width

@[cuda_kernel]
def weightedGatedRmsNormBackwardWeightF32Kernel (x z outputGradient inverseRms weightGradient :
    Cuda.DevicePtr Float32) (rows width : UInt32) : Cuda.DeviceM Unit := do
  let column ← elementIndex
  if column < width then
    let gradient ← Internal.weightedGatedWeightGradColumnF32 x z outputGradient inverseRms rows
      width column 0 zeroF32
    Cuda.storeFloat32 weightGradient column.toUSize gradient

/-! ## Partial RoPE (GPT-NeoX half rotation) -/

/-- Inverse-frequency table entry `theta^(-2i/rotaryDim)` in exact Float32 (`expf`/`logf`). -/
@[expose, cuda_device, always_inline]
def ropeInvFreqAt (theta : Float32) (i rotaryDim : UInt32) : Float32 :=
  Float32.exp (-((2 * i).toFloat32 / rotaryDim.toFloat32 * Float32.log theta))

/-- GPT-NeoX half-rotation of the pair `(x[d], x[d + half])` at one position; dims at and past
`2 * half` pass through. Written for one tensor; the kernel applies it to q and k. -/
@[expose, cuda_device, always_inline]
def ropeRotateFwdAt (input output invFreq : Cuda.DevicePtr Float32) (base d half : UInt32)
    (position : Float32) : Cuda.DeviceM Unit := do
  if d < half then
    let freq ← Cuda.loadFloat32 invFreq d.toUSize
    let angle := position * freq
    let c := Float32.cos angle
    let s := Float32.sin angle
    let x1 ← Cuda.loadFloat32 input (base + d).toUSize
    let x2 ← Cuda.loadFloat32 input (base + d + half).toUSize
    Cuda.storeFloat32 output (base + d).toUSize (x1 * c - x2 * s)
    Cuda.storeFloat32 output (base + d + half).toUSize (x2 * c + x1 * s)
  else if d >= 2 * half then
    let value ← Cuda.loadFloat32 input (base + d).toUSize
    Cuda.storeFloat32 output (base + d).toUSize value

/-- VJP of `ropeRotateFwdAt`: with `dy1, dy2` the pair gradients,
`dx1 = dy1·c + dy2·s`, `dx2 = dy2·c - dy1·s`. -/
@[expose, cuda_device, always_inline]
def ropeRotateBwdAt (outputGradient inputGradient invFreq : Cuda.DevicePtr Float32)
    (base d half : UInt32) (position : Float32) : Cuda.DeviceM Unit := do
  if d < half then
    let freq ← Cuda.loadFloat32 invFreq d.toUSize
    let angle := position * freq
    let c := Float32.cos angle
    let s := Float32.sin angle
    let dy1 ← Cuda.loadFloat32 outputGradient (base + d).toUSize
    let dy2 ← Cuda.loadFloat32 outputGradient (base + d + half).toUSize
    Cuda.storeFloat32 inputGradient (base + d).toUSize (dy1 * c + dy2 * s)
    Cuda.storeFloat32 inputGradient (base + d + half).toUSize (dy2 * c - dy1 * s)
  else if d >= 2 * half then
    let value ← Cuda.loadFloat32 outputGradient (base + d).toUSize
    Cuda.storeFloat32 inputGradient (base + d).toUSize value

/-- RoPE inverse-frequency table: `output[i] = theta^(-2i/rotaryDim)` for `i < half`. -/
@[cuda_kernel]
def ropeInvFreqKernel (output : Cuda.DevicePtr Float32) (half rotaryDim : UInt32)
    (theta : Float32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < half then
    Cuda.storeFloat32 output i.toUSize (ropeInvFreqAt theta i rotaryDim)

/-- Partial rotary applied to single-position q and k of shape `[heads, headDim]`; the first
`2 * half` channels rotate as GPT-NeoX pairs, the rest pass through. -/
@[cuda_kernel]
def ropeApplyForwardF32Kernel (q k invFreq qOut kOut : Cuda.DevicePtr Float32)
    (heads headDim half : UInt32) (position : Float32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < heads * headDim then
    let base := (linear / headDim) * headDim
    let d := linear % headDim
    ropeRotateFwdAt q qOut invFreq base d half position
    ropeRotateFwdAt k kOut invFreq base d half position

/-- VJP of `ropeApplyForwardF32Kernel` at one position: pre-rotary gradients of q and k. -/
@[cuda_kernel]
def ropeApplyBackwardF32Kernel (qGrad kGrad invFreq qPreGrad kPreGrad : Cuda.DevicePtr Float32)
    (heads headDim half : UInt32) (position : Float32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < heads * headDim then
    let base := (linear / headDim) * headDim
    let d := linear % headDim
    ropeRotateBwdAt qGrad qPreGrad invFreq base d half position
    ropeRotateBwdAt kGrad kPreGrad invFreq base d half position

/-- Partial RoPE over a contiguous `[tokens, heads, headDim]` tensor. -/
@[cuda_kernel]
def ropeSequenceForwardF32Kernel (input invFreq output : Cuda.DevicePtr Float32)
    (tokens heads headDim half : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < tokens * heads * headDim then
    let token := linear / (heads * headDim)
    let base := (linear / headDim) * headDim
    let d := linear % headDim
    ropeRotateFwdAt input output invFreq base d half token.toFloat32

/-- VJP of full-sequence partial RoPE. -/
@[cuda_kernel]
def ropeSequenceBackwardF32Kernel (outputGradient invFreq inputGradient :
    Cuda.DevicePtr Float32) (tokens heads headDim half : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < tokens * heads * headDim then
    let token := linear / (heads * headDim)
    let base := (linear / headDim) * headDim
    let d := linear % headDim
    ropeRotateBwdAt outputGradient inputGradient invFreq base d half token.toFloat32

/-- Partial rotary over `[batch, sequence, heads, headDim]`, resetting position per sequence. -/
@[cuda_kernel]
def ropeBatchedForwardF32Kernel (input invFreq output : Cuda.DevicePtr Float32)
    (batchSize sequenceLength heads headDim half : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let tokens := batchSize * sequenceLength
  if linear < tokens * heads * headDim then
    let token := linear / (heads * headDim)
    let position := token % sequenceLength
    let base := (linear / headDim) * headDim
    let d := linear % headDim
    ropeRotateFwdAt input output invFreq base d half position.toFloat32

/-- VJP of batch-isolated partial rotary. -/
@[cuda_kernel]
def ropeBatchedBackwardF32Kernel (outputGradient invFreq inputGradient :
    Cuda.DevicePtr Float32) (batchSize sequenceLength heads headDim half : UInt32) :
    Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let tokens := batchSize * sequenceLength
  if linear < tokens * heads * headDim then
    let token := linear / (heads * headDim)
    let position := token % sequenceLength
    let base := (linear / headDim) * headDim
    let d := linear % headDim
    ropeRotateBwdAt outputGradient inputGradient invFreq base d half position.toFloat32

/-! ## Causal depthwise conv1d (kernel 4) with SiLU -/

/-- One element of batch-isolated causal depthwise convolution with SiLU. Exposed separately so a
model-wide cooperative program can reuse the numerical body without a child launch. -/
@[expose, cuda_device, always_inline]
def conv1dSiLUBatchedForwardElementF32 (x weight output : Cuda.DevicePtr Float32)
    (batchSize sequenceLength channels linear : UInt32) : Cuda.DeviceM Unit := do
  let tokens := batchSize * sequenceLength
  if linear < tokens * channels then
    let token := linear / channels
    let batch := token / sequenceLength
    let t := token % sequenceLength
    let c := linear % channels
    let byteOffset := (batch * sequenceLength * channels * 4).toUSize
    let batchInput : Cuda.DevicePtr Float32 := x + byteOffset
    let u ← Internal.convSumF32 batchInput weight channels t c 0 zeroF32
    Cuda.storeFloat32 output linear.toUSize (siluF32 u)

/-- One element of the batch-isolated causal-convolution input VJP. -/
@[expose, cuda_device, always_inline]
def conv1dSiLUBatchedBackwardInputElementF32 (x weight outputGradient inputGradient :
    Cuda.DevicePtr Float32) (batchSize sequenceLength channels linear : UInt32) :
    Cuda.DeviceM Unit := do
  let tokens := batchSize * sequenceLength
  if linear < tokens * channels then
    let token := linear / channels
    let batch := token / sequenceLength
    let t := token % sequenceLength
    let c := linear % channels
    let byteOffset := (batch * sequenceLength * channels * 4).toUSize
    let batchInput : Cuda.DevicePtr Float32 := x + byteOffset
    let batchOutputGradient : Cuda.DevicePtr Float32 := outputGradient + byteOffset
    let gradient ← Internal.convInputGradFrom batchInput weight batchOutputGradient sequenceLength
      channels t c 0 zeroF32
    Cuda.storeFloat32 inputGradient linear.toUSize gradient

/-- Full-sequence causal depthwise conv1d with SiLU: `out[t,c] = silu(Σ_j w[c,j]·x[t-3+j,c])`
with zero left padding; weight layout is the HF `[channels, 4]`. -/
@[cuda_kernel]
def conv1dSiLUForwardF32Kernel (x weight output : Cuda.DevicePtr Float32) (seq channels : UInt32) :
    Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < seq * channels then
    let t := linear / channels
    let c := linear % channels
    let u ← Internal.convSumF32 x weight channels t c 0 zeroF32
    Cuda.storeFloat32 output linear.toUSize (siluF32 u)

/-- Conv input VJP over the full sequence: `dx[t,c] = Σ_j w[c,j]·ds[t+3-j,c]` with `ds` the
pre-activation gradient recomputed at each source position. -/
@[cuda_kernel]
def conv1dSiLUBackwardInputF32Kernel (x weight outputGradient inputGradient :
    Cuda.DevicePtr Float32) (seq channels : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < seq * channels then
    let t := linear / channels
    let c := linear % channels
    let gradient ← Internal.convInputGradFrom x weight outputGradient seq channels t c 0 zeroF32
    Cuda.storeFloat32 inputGradient linear.toUSize gradient

/-- Conv weight VJP over the full sequence: one thread per `(c, j)` tap, rows reduced serially. -/
@[cuda_kernel]
def conv1dSiLUBackwardWeightF32Kernel (x weight outputGradient weightGradient :
    Cuda.DevicePtr Float32) (seq channels : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < channels * 4 then
    let c := linear / 4
    let j := linear % 4
    let gradient ← Internal.convWeightGradFrom x weight outputGradient seq channels c j (3 - j)
      zeroF32
    Cuda.storeFloat32 weightGradient linear.toUSize gradient

/-- Batch-isolated causal depthwise convolution over `[batch, sequence, channels]`. -/
@[cuda_kernel]
def conv1dSiLUBatchedForwardF32Kernel (x weight output : Cuda.DevicePtr Float32)
    (batchSize sequenceLength channels : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let tokens := batchSize * sequenceLength
  if linear < tokens * channels then
    let token := linear / channels
    let batch := token / sequenceLength
    let t := token % sequenceLength
    let c := linear % channels
    let byteOffset := (batch * sequenceLength * channels * 4).toUSize
    let batchInput : Cuda.DevicePtr Float32 := x + byteOffset
    let u ← Internal.convSumF32 batchInput weight channels t c 0 zeroF32
    Cuda.storeFloat32 output linear.toUSize (siluF32 u)

/-- Input VJP of batch-isolated causal depthwise convolution. -/
@[cuda_kernel]
def conv1dSiLUBatchedBackwardInputF32Kernel (x weight outputGradient inputGradient :
    Cuda.DevicePtr Float32) (batchSize sequenceLength channels : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  let tokens := batchSize * sequenceLength
  if linear < tokens * channels then
    let token := linear / channels
    let batch := token / sequenceLength
    let t := token % sequenceLength
    let c := linear % channels
    let byteOffset := (batch * sequenceLength * channels * 4).toUSize
    let batchInput : Cuda.DevicePtr Float32 := x + byteOffset
    let batchOutputGradient : Cuda.DevicePtr Float32 := outputGradient + byteOffset
    let gradient ← Internal.convInputGradFrom batchInput weight batchOutputGradient sequenceLength
      channels t c 0 zeroF32
    Cuda.storeFloat32 inputGradient linear.toUSize gradient

/-- Weight VJP reduced deterministically across independent sequences. -/
@[cuda_kernel]
def conv1dSiLUBatchedBackwardWeightF32Kernel (x weight outputGradient weightGradient :
    Cuda.DevicePtr Float32) (batchSize sequenceLength channels : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < channels * 4 then
    let c := linear / 4
    let j := linear % 4
    let gradient ← Internal.batchedConvWeightGradFrom x weight outputGradient batchSize
      sequenceLength channels c j 0 zeroF32
    Cuda.storeFloat32 weightGradient linear.toUSize gradient

/-- Single-token decode step: taps read `state` rows `(x[t-3], x[t-2], x[t-1])` and the new
token, publish `silu` of the tap sum, and shift the state. -/
@[cuda_kernel]
def conv1dSiLUDecodeForwardF32Kernel (state input weight output stateOut :
    Cuda.DevicePtr Float32) (channels : UInt32) : Cuda.DeviceM Unit := do
  let c ← elementIndex
  if c < channels then
    let s0 ← Cuda.loadFloat32 state c.toUSize
    let s1 ← Cuda.loadFloat32 state (channels + c).toUSize
    let s2 ← Cuda.loadFloat32 state (2 * channels + c).toUSize
    let value ← Cuda.loadFloat32 input c.toUSize
    let w0 ← Cuda.loadFloat32 weight (c * 4).toUSize
    let w1 ← Cuda.loadFloat32 weight (c * 4 + 1).toUSize
    let w2 ← Cuda.loadFloat32 weight (c * 4 + 2).toUSize
    let w3 ← Cuda.loadFloat32 weight (c * 4 + 3).toUSize
    let u := Cuda.fma w0 s0 (Cuda.fma w1 s1 (Cuda.fma w2 s2 (w3 * value)))
    Cuda.storeFloat32 output c.toUSize (siluF32 u)
    Cuda.storeFloat32 stateOut c.toUSize s1
    Cuda.storeFloat32 stateOut (channels + c).toUSize s2
    Cuda.storeFloat32 stateOut (2 * channels + c).toUSize value

/-- VJP of the decode step: gradients of the incoming state rows, the new token, and the four
taps at this step (no cross-step accumulation). -/
@[cuda_kernel]
def conv1dSiLUDecodeBackwardF32Kernel (state input weight outputGradient inputGradient
    stateGradient weightGradientStep : Cuda.DevicePtr Float32) (channels : UInt32) :
    Cuda.DeviceM Unit := do
  let c ← elementIndex
  if c < channels then
    let s0 ← Cuda.loadFloat32 state c.toUSize
    let s1 ← Cuda.loadFloat32 state (channels + c).toUSize
    let s2 ← Cuda.loadFloat32 state (2 * channels + c).toUSize
    let value ← Cuda.loadFloat32 input c.toUSize
    let w0 ← Cuda.loadFloat32 weight (c * 4).toUSize
    let w1 ← Cuda.loadFloat32 weight (c * 4 + 1).toUSize
    let w2 ← Cuda.loadFloat32 weight (c * 4 + 2).toUSize
    let w3 ← Cuda.loadFloat32 weight (c * 4 + 3).toUSize
    let u := Cuda.fma w0 s0 (Cuda.fma w1 s1 (Cuda.fma w2 s2 (w3 * value)))
    let d ← Cuda.loadFloat32 outputGradient c.toUSize
    let ds := d * siluGradF32 u
    Cuda.storeFloat32 inputGradient c.toUSize (w3 * ds)
    Cuda.storeFloat32 stateGradient c.toUSize (w0 * ds)
    Cuda.storeFloat32 stateGradient (channels + c).toUSize (w1 * ds)
    Cuda.storeFloat32 stateGradient (2 * channels + c).toUSize (w2 * ds)
    Cuda.storeFloat32 weightGradientStep (c * 4).toUSize (s0 * ds)
    Cuda.storeFloat32 weightGradientStep (c * 4 + 1).toUSize (s1 * ds)
    Cuda.storeFloat32 weightGradientStep (c * 4 + 2).toUSize (s2 * ds)
    Cuda.storeFloat32 weightGradientStep (c * 4 + 3).toUSize (value * ds)

/-! ## Per-head l2 normalization (eps inside the square root) -/

/-- Forward body of one l2norm row `x / sqrt(Σx² + eps)`, Float32 I/O. Every lane of the
calling warp executes convergently. -/
@[expose, cuda_device, always_inline]
def l2normRowFwdF32 (x output : Cuda.DevicePtr Float32) (row width : UInt32)
    (epsilon : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let total ← warpSumBroadcast (← Internal.squareSumF32 x base width lane zeroF32)
  let den := Float32.sqrt (total + epsilon)
  Internal.storeNormalizedF32 x output base width lane den

/-- Backward body of one l2norm row, Float32 I/O: `dx = (dy - y·(Σ dy·y)) / den` with `y` and
`den` recomputed from `x`. -/
@[expose, cuda_device, always_inline]
def l2normRowBwdF32 (x outputGradient inputGradient : Cuda.DevicePtr Float32) (row width : UInt32)
    (epsilon : Float32) : Cuda.DeviceM Unit := do
  let lane ← Cuda.laneId
  let base := row * width
  let total ← warpSumBroadcast (← Internal.squareSumF32 x base width lane zeroF32)
  let den := Float32.sqrt (total + epsilon)
  let dot ← warpSumBroadcast
    (← Internal.dotNormalizedF32 outputGradient x base width lane den zeroF32)
  Internal.storeL2GradF32 x outputGradient inputGradient base width lane den dot

/-- Per-head l2 normalization forward: one warp per row of a `[rows, width]` activation. -/
@[cuda_kernel]
def l2normForwardF32Kernel (x output : Cuda.DevicePtr Float32) (rows width : UInt32)
    (epsilon : Float32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    l2normRowFwdF32 x output row width epsilon

/-- Per-head l2 normalization VJP: one warp per row. -/
@[cuda_kernel]
def l2normBackwardF32Kernel (x outputGradient inputGradient : Cuda.DevicePtr Float32)
    (rows width : UInt32) (epsilon : Float32) : Cuda.DeviceM Unit := do
  let row ← warpRow
  if row < rows then
    l2normRowBwdF32 x outputGradient inputGradient row width epsilon

/-! ## SiLU and SwiGLU elementwise bodies -/

/-- One Float32 SwiGLU forward element, reusable by resident programs. -/
@[expose, cuda_device, always_inline]
def swigluForwardElementF32 (gate up hidden : Cuda.DevicePtr Float32) (count linear : UInt32) :
    Cuda.DeviceM Unit := do
  if linear < count then
    let g ← Cuda.loadFloat32 gate linear.toUSize
    let u ← Cuda.loadFloat32 up linear.toUSize
    Cuda.storeFloat32 hidden linear.toUSize (siluF32 g * u)

/-- One Float32 SwiGLU VJP element, reusable by resident programs. -/
@[expose, cuda_device, always_inline]
def swigluBackwardElementF32 (gate up hiddenGradient gateGradient upGradient :
    Cuda.DevicePtr Float32) (count linear : UInt32) : Cuda.DeviceM Unit := do
  if linear < count then
    let g ← Cuda.loadFloat32 gate linear.toUSize
    let u ← Cuda.loadFloat32 up linear.toUSize
    let dh ← Cuda.loadFloat32 hiddenGradient linear.toUSize
    Cuda.storeFloat32 gateGradient linear.toUSize (dh * u * siluGradF32 g)
    Cuda.storeFloat32 upGradient linear.toUSize (dh * siluF32 g)

/-- Elementwise SiLU forward, Float32 I/O. -/
@[cuda_kernel]
def siluForwardF32Kernel (x y : Cuda.DevicePtr Float32) (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    Cuda.storeFloat32 y i.toUSize (siluF32 (← Cuda.loadFloat32 x i.toUSize))

/-- Elementwise SiLU forward, BF16 I/O with Float32 math. -/
@[cuda_kernel]
def siluForwardBF16Kernel (x y : Cuda.DevicePtr Cuda.BFloat16) (count : UInt32) :
    Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    let value := (← Cuda.loadBFloat16 x i.toUSize).toFloat32
    Cuda.storeBFloat16 y i.toUSize (Cuda.BFloat16.ofFloat32 (siluF32 value))

/-- Elementwise SiLU VJP `dx = dy·silu'(x)`, Float32 I/O. -/
@[cuda_kernel]
def siluBackwardF32Kernel (x outputGradient inputGradient : Cuda.DevicePtr Float32)
    (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    let value ← Cuda.loadFloat32 x i.toUSize
    let d ← Cuda.loadFloat32 outputGradient i.toUSize
    Cuda.storeFloat32 inputGradient i.toUSize (d * siluGradF32 value)

/-- Elementwise SiLU VJP, BF16 I/O with Float32 math. -/
@[cuda_kernel]
def siluBackwardBF16Kernel (x outputGradient inputGradient : Cuda.DevicePtr Cuda.BFloat16)
    (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    let value := (← Cuda.loadBFloat16 x i.toUSize).toFloat32
    let d := (← Cuda.loadBFloat16 outputGradient i.toUSize).toFloat32
    Cuda.storeBFloat16 inputGradient i.toUSize
      (Cuda.BFloat16.ofFloat32 (d * siluGradF32 value))

/-- Elementwise SwiGLU forward `silu(gate) * up`, Float32 I/O. -/
@[cuda_kernel]
def swigluForwardF32Kernel (gate up hidden : Cuda.DevicePtr Float32) (count : UInt32) :
    Cuda.DeviceM Unit := do
  let i ← elementIndex
  swigluForwardElementF32 gate up hidden count i

/-- Elementwise SwiGLU forward, BF16 I/O with Float32 math. -/
@[cuda_kernel]
def swigluForwardBF16Kernel (gate up hidden : Cuda.DevicePtr Cuda.BFloat16) (count : UInt32) :
    Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    let g := (← Cuda.loadBFloat16 gate i.toUSize).toFloat32
    let u := (← Cuda.loadBFloat16 up i.toUSize).toFloat32
    Cuda.storeBFloat16 hidden i.toUSize (Cuda.BFloat16.ofFloat32 (siluF32 g * u))

/-- Elementwise SwiGLU VJP, Float32 I/O: `dgate = dh·up·silu'(gate)`, `dup = dh·silu(gate)`. -/
@[cuda_kernel]
def swigluBackwardF32Kernel (gate up hiddenGradient gateGradient upGradient :
    Cuda.DevicePtr Float32) (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  swigluBackwardElementF32 gate up hiddenGradient gateGradient upGradient count i

/-- Elementwise SwiGLU VJP, BF16 I/O with Float32 math. -/
@[cuda_kernel]
def swigluBackwardBF16Kernel (gate up hiddenGradient gateGradient upGradient :
    Cuda.DevicePtr Cuda.BFloat16) (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    let g := (← Cuda.loadBFloat16 gate i.toUSize).toFloat32
    let u := (← Cuda.loadBFloat16 up i.toUSize).toFloat32
    let dh := (← Cuda.loadBFloat16 hiddenGradient i.toUSize).toFloat32
    Cuda.storeBFloat16 gateGradient i.toUSize
      (Cuda.BFloat16.ofFloat32 (dh * u * siluGradF32 g))
    Cuda.storeBFloat16 upGradient i.toUSize (Cuda.BFloat16.ofFloat32 (dh * siluF32 g))

/-! ## Embedding gather and casts -/

/-- Embedding gather of `count` rows of width `width`: `output[r] = table[ids[r]]`. -/
@[cuda_kernel]
def embeddingGatherF32Kernel (table : Cuda.DevicePtr Float32) (ids : Cuda.DevicePtr UInt32)
    (output : Cuda.DevicePtr Float32) (count width : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < count * width then
    let row := linear / width
    let column := linear % width
    let id ← Cuda.loadUInt32 ids row.toUSize
    let value ← Cuda.loadFloat32 table (id * width + column).toUSize
    Cuda.storeFloat32 output linear.toUSize value

/-- Embedding gather of `count` rows of width `width`, BF16 table and output. -/
@[cuda_kernel]
def embeddingGatherBF16Kernel (table : Cuda.DevicePtr Cuda.BFloat16) (ids : Cuda.DevicePtr UInt32)
    (output : Cuda.DevicePtr Cuda.BFloat16) (count width : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < count * width then
    let row := linear / width
    let column := linear % width
    let id ← Cuda.loadUInt32 ids row.toUSize
    let value ← Cuda.loadBFloat16 table (id * width + column).toUSize
    Cuda.storeBFloat16 output linear.toUSize value

/-- Gather one frozen checkpoint embedding directly into Float32 training activations. -/
@[cuda_kernel]
def embeddingGatherFrozenBF16F32Kernel (sourceBytes : Cuda.DevicePtr UInt8)
    (sourceByteOffset : UInt64) (ids : Cuda.DevicePtr UInt32)
    (output : Cuda.DevicePtr Float32) (count width : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < count * width then
    let row := linear / width
    let column := linear % width
    let id ← Cuda.loadUInt32 ids row.toUSize
    let table : Cuda.DevicePtr Cuda.BFloat16 := sourceBytes + sourceByteOffset.toUSize
    let value ← Cuda.loadBFloat16 table (id * width + column).toUSize
    Cuda.storeFloat32 output linear.toUSize value.toFloat32

/-- Widen an immutable BF16 checkpoint tensor into a separately owned Float32 buffer. -/
@[cuda_kernel]
def copyFrozenBF16ToF32Kernel (sourceBytes : Cuda.DevicePtr UInt8)
    (sourceByteOffset : UInt64) (output : Cuda.DevicePtr Float32) (count : UInt32) :
    Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < count then
    let source : Cuda.DevicePtr Cuda.BFloat16 := sourceBytes + sourceByteOffset.toUSize
    Cuda.storeFloat32 output linear.toUSize
      (← Cuda.loadBFloat16 source linear.toUSize).toFloat32

/-- Elementwise Float32 → BF16 cast (round to nearest even). -/
@[cuda_kernel]
def castF32ToBF16Kernel (input : Cuda.DevicePtr Float32) (output : Cuda.DevicePtr Cuda.BFloat16)
    (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    Cuda.storeBFloat16 output i.toUSize
      (Cuda.BFloat16.ofFloat32 (← Cuda.loadFloat32 input i.toUSize))

/-- Elementwise BF16 → Float32 cast (exact widening). -/
@[cuda_kernel]
def castBF16ToF32Kernel (input : Cuda.DevicePtr Cuda.BFloat16) (output : Cuda.DevicePtr Float32)
    (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    Cuda.storeFloat32 output i.toUSize (← Cuda.loadBFloat16 input i.toUSize).toFloat32

@[cuda_kernel]
def zeroF32Kernel (output : Cuda.DevicePtr Float32) (count : UInt32) : Cuda.DeviceM Unit := do
  let i ← elementIndex
  if i < count then
    Cuda.storeFloat32 output i.toUSize zeroF32

/-! ## Typed host launchers -/

/-- Launch configuration for row-parallel warp kernels: `rowWarps` warps per block. -/
def rowWarpConfig (rows : UInt32) : Cuda.LaunchConfig := {
  grid := { x := (rows + rowWarps - 1) / rowWarps }
  block := { x := rowWarps * 32 }
  blockArenaBytes := 0
}

/-- Launch configuration for `count` independent elements in 128-thread blocks. -/
def elementConfig (count : UInt32) : Cuda.LaunchConfig :=
  Cuda.LaunchConfig.forElements count.toUSize 128

/-- `(1 + w)` RMSNorm forward over a `[rows, width]` Float32 activation. -/
def rmsNormForwardF32 (stream : @& Cuda.Stream) (x weight output inverseRms :
    @& Cuda.Buffer Float32) (rows width : UInt32) (epsilon : Float32) : IO Cuda.KernelHandle :=
  rmsNormForwardF32Kernel.launchOn stream (rowWarpConfig rows) x weight output inverseRms rows
    width epsilon

/-- `(1 + w)` RMSNorm forward over a `[rows, width]` BF16 activation. -/
def rmsNormForwardBF16 (stream : @& Cuda.Stream) (x weight output :
    @& Cuda.Buffer Cuda.BFloat16) (inverseRms : @& Cuda.Buffer Float32) (rows width : UInt32)
    (epsilon : Float32) : IO Cuda.KernelHandle :=
  rmsNormForwardBF16Kernel.launchOn stream (rowWarpConfig rows) x weight output inverseRms rows
    width epsilon

/-- `(1 + w)` RMSNorm input VJP over `[rows, width]`, Float32 I/O. -/
def rmsNormBackwardInputF32 (stream : @& Cuda.Stream)
    (x weight outputGradient inputGradient inverseRms : @& Cuda.Buffer Float32)
    (rows width : UInt32) : IO Cuda.KernelHandle :=
  rmsNormBackwardInputF32Kernel.launchOn stream (rowWarpConfig rows) x weight outputGradient
    inputGradient inverseRms rows width

/-- `(1 + w)` RMSNorm input VJP over `[rows, width]`, BF16 I/O. -/
def rmsNormBackwardInputBF16 (stream : @& Cuda.Stream)
    (x weight outputGradient inputGradient : @& Cuda.Buffer Cuda.BFloat16)
    (inverseRms : @& Cuda.Buffer Float32) (rows width : UInt32) : IO Cuda.KernelHandle :=
  rmsNormBackwardInputBF16Kernel.launchOn stream (rowWarpConfig rows) x weight outputGradient
    inputGradient inverseRms rows width

/-- `(1 + w)` RMSNorm weight VJP over `[rows, width]`, Float32 I/O. -/
def rmsNormBackwardWeightF32 (stream : @& Cuda.Stream)
    (x outputGradient inverseRms weightGradient : @& Cuda.Buffer Float32) (rows width : UInt32) :
    IO Cuda.KernelHandle :=
  rmsNormBackwardWeightF32Kernel.launchOn stream (elementConfig width) x outputGradient
    inverseRms weightGradient rows width

/-- `(1 + w)` RMSNorm weight VJP over `[rows, width]`, BF16 I/O. -/
def rmsNormBackwardWeightBF16 (stream : @& Cuda.Stream)
    (x outputGradient : @& Cuda.Buffer Cuda.BFloat16) (inverseRms : @& Cuda.Buffer Float32)
    (weightGradient : @& Cuda.Buffer Cuda.BFloat16) (rows width : UInt32) :
    IO Cuda.KernelHandle :=
  rmsNormBackwardWeightBF16Kernel.launchOn stream (elementConfig width) x outputGradient
    inverseRms weightGradient rows width

/-- Gated RMSNorm forward over `[rows, width]`, Float32 I/O. -/
def gatedRmsNormForwardF32 (stream : @& Cuda.Stream) (x z output inverseRms :
    @& Cuda.Buffer Float32) (rows width : UInt32) (epsilon : Float32) : IO Cuda.KernelHandle :=
  gatedRmsNormForwardF32Kernel.launchOn stream (rowWarpConfig rows) x z output inverseRms rows
    width epsilon

/-- Gated RMSNorm forward over `[rows, width]`, BF16 I/O. -/
def gatedRmsNormForwardBF16 (stream : @& Cuda.Stream) (x z output :
    @& Cuda.Buffer Cuda.BFloat16) (inverseRms : @& Cuda.Buffer Float32) (rows width : UInt32)
    (epsilon : Float32) : IO Cuda.KernelHandle :=
  gatedRmsNormForwardBF16Kernel.launchOn stream (rowWarpConfig rows) x z output inverseRms rows
    width epsilon

/-- Gated RMSNorm VJP over `[rows, width]`, Float32 I/O. -/
def gatedRmsNormBackwardF32 (stream : @& Cuda.Stream)
    (x z outputGradient inputGradient gateGradient inverseRms : @& Cuda.Buffer Float32)
    (rows width : UInt32) : IO Cuda.KernelHandle :=
  gatedRmsNormBackwardF32Kernel.launchOn stream (rowWarpConfig rows) x z outputGradient
    inputGradient gateGradient inverseRms rows width

/-- Gated RMSNorm VJP over `[rows, width]`, BF16 I/O. -/
def gatedRmsNormBackwardBF16 (stream : @& Cuda.Stream)
    (x z outputGradient inputGradient gateGradient : @& Cuda.Buffer Cuda.BFloat16)
    (inverseRms : @& Cuda.Buffer Float32) (rows width : UInt32) : IO Cuda.KernelHandle :=
  gatedRmsNormBackwardBF16Kernel.launchOn stream (rowWarpConfig rows) x z outputGradient
    inputGradient gateGradient inverseRms rows width

/-- Trainable-weight gated RMSNorm forward over `[rows, width]`. -/
def weightedGatedRmsNormForwardF32 (stream : @& Cuda.Stream)
    (x z weight output inverseRms : @& Cuda.Buffer Float32) (rows width : UInt32)
    (epsilon : Float32) : IO Cuda.KernelHandle :=
  weightedGatedRmsNormForwardF32Kernel.launchOn stream (rowWarpConfig rows) x z weight output
    inverseRms rows width epsilon

/-- Trainable-weight gated RMSNorm input and gate VJP. -/
def weightedGatedRmsNormBackwardF32 (stream : @& Cuda.Stream)
    (x z weight outputGradient inputGradient gateGradient inverseRms : @& Cuda.Buffer Float32)
    (rows width : UInt32) : IO Cuda.KernelHandle :=
  weightedGatedRmsNormBackwardF32Kernel.launchOn stream (rowWarpConfig rows) x z weight
    outputGradient inputGradient gateGradient inverseRms rows width

/-- Trainable gated-RMSNorm weight VJP. -/
def weightedGatedRmsNormBackwardWeightF32 (stream : @& Cuda.Stream)
    (x z outputGradient inverseRms weightGradient : @& Cuda.Buffer Float32)
    (rows width : UInt32) : IO Cuda.KernelHandle :=
  weightedGatedRmsNormBackwardWeightF32Kernel.launchOn stream (elementConfig width) x z
    outputGradient inverseRms weightGradient rows width

/-- RoPE inverse-frequency table of `half` entries for `rotaryDim` and `theta`. -/
def ropeInvFreq (stream : @& Cuda.Stream) (output : @& Cuda.Buffer Float32)
    (half rotaryDim : UInt32) (theta : Float32) : IO Cuda.KernelHandle :=
  ropeInvFreqKernel.launchOn stream (elementConfig half) output half rotaryDim theta

/-- Partial RoPE applied to single-position `[heads, headDim]` q and k at `position`. -/
def ropeApplyForwardF32 (stream : @& Cuda.Stream) (q k invFreq qOut kOut :
    @& Cuda.Buffer Float32) (heads headDim half : UInt32) (position : Float32) :
    IO Cuda.KernelHandle :=
  ropeApplyForwardF32Kernel.launchOn stream (elementConfig (heads * headDim)) q k invFreq qOut
    kOut heads headDim half position

/-- RoPE VJP at `position`: pre-rotary gradients of `[heads, headDim]` q and k. -/
def ropeApplyBackwardF32 (stream : @& Cuda.Stream) (qGrad kGrad invFreq qPreGrad kPreGrad :
    @& Cuda.Buffer Float32) (heads headDim half : UInt32) (position : Float32) :
    IO Cuda.KernelHandle :=
  ropeApplyBackwardF32Kernel.launchOn stream (elementConfig (heads * headDim)) qGrad kGrad
    invFreq qPreGrad kPreGrad heads headDim half position

/-- Partial RoPE over `[tokens, heads, headDim]`, using token indices as positions. -/
def ropeSequenceForwardF32 (stream : @& Cuda.Stream) (input invFreq output :
    @& Cuda.Buffer Float32) (tokens heads headDim half : UInt32) : IO Cuda.KernelHandle :=
  ropeSequenceForwardF32Kernel.launchOn stream (elementConfig (tokens * heads * headDim)) input
    invFreq output tokens heads headDim half

/-- VJP of full-sequence partial RoPE. -/
def ropeSequenceBackwardF32 (stream : @& Cuda.Stream) (outputGradient invFreq inputGradient :
    @& Cuda.Buffer Float32) (tokens heads headDim half : UInt32) : IO Cuda.KernelHandle :=
  ropeSequenceBackwardF32Kernel.launchOn stream (elementConfig (tokens * heads * headDim))
    outputGradient invFreq inputGradient tokens heads headDim half

/-- Full-sequence causal depthwise conv1d (kernel 4) with SiLU over `[seq, channels]`. -/
def conv1dSiLUForwardF32 (stream : @& Cuda.Stream) (x weight output : @& Cuda.Buffer Float32)
    (seq channels : UInt32) : IO Cuda.KernelHandle :=
  conv1dSiLUForwardF32Kernel.launchOn stream (elementConfig (seq * channels)) x weight output
    seq channels

/-- Conv input VJP over `[seq, channels]`. -/
def conv1dSiLUBackwardInputF32 (stream : @& Cuda.Stream)
    (x weight outputGradient inputGradient : @& Cuda.Buffer Float32) (seq channels : UInt32) :
    IO Cuda.KernelHandle :=
  conv1dSiLUBackwardInputF32Kernel.launchOn stream (elementConfig (seq * channels)) x weight
    outputGradient inputGradient seq channels

/-- Conv weight VJP over `[seq, channels]`; weight layout `[channels, 4]`. -/
def conv1dSiLUBackwardWeightF32 (stream : @& Cuda.Stream)
    (x weight outputGradient weightGradient : @& Cuda.Buffer Float32) (seq channels : UInt32) :
    IO Cuda.KernelHandle :=
  conv1dSiLUBackwardWeightF32Kernel.launchOn stream (elementConfig (channels * 4)) x weight
    outputGradient weightGradient seq channels

/-- Single-token decode step with state carry `[3, channels]`. -/
def conv1dSiLUDecodeForwardF32 (stream : @& Cuda.Stream)
    (state input weight output stateOut : @& Cuda.Buffer Float32) (channels : UInt32) :
    IO Cuda.KernelHandle :=
  conv1dSiLUDecodeForwardF32Kernel.launchOn stream (elementConfig channels) state input weight
    output stateOut channels

/-- VJP of the single-token decode step. -/
def conv1dSiLUDecodeBackwardF32 (stream : @& Cuda.Stream)
    (state input weight outputGradient inputGradient stateGradient weightGradientStep :
    @& Cuda.Buffer Float32) (channels : UInt32) : IO Cuda.KernelHandle :=
  conv1dSiLUDecodeBackwardF32Kernel.launchOn stream (elementConfig channels) state input weight
    outputGradient inputGradient stateGradient weightGradientStep channels

/-- Per-head l2 normalization forward over `[rows, width]`. -/
def l2normForwardF32 (stream : @& Cuda.Stream) (x output : @& Cuda.Buffer Float32)
    (rows width : UInt32) (epsilon : Float32) : IO Cuda.KernelHandle :=
  l2normForwardF32Kernel.launchOn stream (rowWarpConfig rows) x output rows width epsilon

/-- Per-head l2 normalization VJP over `[rows, width]`. -/
def l2normBackwardF32 (stream : @& Cuda.Stream) (x outputGradient inputGradient :
    @& Cuda.Buffer Float32) (rows width : UInt32) (epsilon : Float32) : IO Cuda.KernelHandle :=
  l2normBackwardF32Kernel.launchOn stream (rowWarpConfig rows) x outputGradient inputGradient
    rows width epsilon

/-- Elementwise SiLU forward over `count` Float32 values. -/
def siluForwardF32 (stream : @& Cuda.Stream) (x y : @& Cuda.Buffer Float32) (count : UInt32) :
    IO Cuda.KernelHandle :=
  siluForwardF32Kernel.launchOn stream (elementConfig count) x y count

/-- Elementwise SiLU forward over `count` BF16 values. -/
def siluForwardBF16 (stream : @& Cuda.Stream) (x y : @& Cuda.Buffer Cuda.BFloat16)
    (count : UInt32) : IO Cuda.KernelHandle :=
  siluForwardBF16Kernel.launchOn stream (elementConfig count) x y count

/-- Elementwise SiLU VJP over `count` Float32 values. -/
def siluBackwardF32 (stream : @& Cuda.Stream) (x outputGradient inputGradient :
    @& Cuda.Buffer Float32) (count : UInt32) : IO Cuda.KernelHandle :=
  siluBackwardF32Kernel.launchOn stream (elementConfig count) x outputGradient inputGradient
    count

/-- Elementwise SiLU VJP over `count` BF16 values. -/
def siluBackwardBF16 (stream : @& Cuda.Stream) (x outputGradient inputGradient :
    @& Cuda.Buffer Cuda.BFloat16) (count : UInt32) : IO Cuda.KernelHandle :=
  siluBackwardBF16Kernel.launchOn stream (elementConfig count) x outputGradient inputGradient
    count

/-- Elementwise SwiGLU forward over `count` Float32 values. -/
def swigluForwardF32 (stream : @& Cuda.Stream) (gate up hidden : @& Cuda.Buffer Float32)
    (count : UInt32) : IO Cuda.KernelHandle :=
  swigluForwardF32Kernel.launchOn stream (elementConfig count) gate up hidden count

/-- Elementwise SwiGLU forward over `count` BF16 values. -/
def swigluForwardBF16 (stream : @& Cuda.Stream) (gate up hidden :
    @& Cuda.Buffer Cuda.BFloat16) (count : UInt32) : IO Cuda.KernelHandle :=
  swigluForwardBF16Kernel.launchOn stream (elementConfig count) gate up hidden count

/-- Elementwise SwiGLU VJP over `count` Float32 values. -/
def swigluBackwardF32 (stream : @& Cuda.Stream) (gate up hiddenGradient gateGradient upGradient :
    @& Cuda.Buffer Float32) (count : UInt32) : IO Cuda.KernelHandle :=
  swigluBackwardF32Kernel.launchOn stream (elementConfig count) gate up hiddenGradient
    gateGradient upGradient count

/-- Elementwise SwiGLU VJP over `count` BF16 values. -/
def swigluBackwardBF16 (stream : @& Cuda.Stream) (gate up hiddenGradient gateGradient
    upGradient : @& Cuda.Buffer Cuda.BFloat16) (count : UInt32) : IO Cuda.KernelHandle :=
  swigluBackwardBF16Kernel.launchOn stream (elementConfig count) gate up hiddenGradient
    gateGradient upGradient count

/-- Embedding gather of `count` rows of width `width`, Float32 table. -/
def embeddingGatherF32 (stream : @& Cuda.Stream) (table : @& Cuda.Buffer Float32)
    (ids : @& Cuda.Buffer UInt32) (output : @& Cuda.Buffer Float32) (count width : UInt32) :
    IO Cuda.KernelHandle :=
  embeddingGatherF32Kernel.launchOn stream (elementConfig (count * width)) table ids output
    count width

/-- Embedding gather of `count` rows of width `width`, BF16 table. -/
def embeddingGatherBF16 (stream : @& Cuda.Stream) (table : @& Cuda.Buffer Cuda.BFloat16)
    (ids : @& Cuda.Buffer UInt32) (output : @& Cuda.Buffer Cuda.BFloat16) (count width : UInt32) :
    IO Cuda.KernelHandle :=
  embeddingGatherBF16Kernel.launchOn stream (elementConfig (count * width)) table ids output
    count width

/-- Elementwise Float32 → BF16 cast over `count` values. -/
def castF32ToBF16 (stream : @& Cuda.Stream) (input : @& Cuda.Buffer Float32)
    (output : @& Cuda.Buffer Cuda.BFloat16) (count : UInt32) : IO Cuda.KernelHandle :=
  castF32ToBF16Kernel.launchOn stream (elementConfig count) input output count

/-- Elementwise BF16 → Float32 cast over `count` values. -/
def castBF16ToF32 (stream : @& Cuda.Stream) (input : @& Cuda.Buffer Cuda.BFloat16)
    (output : @& Cuda.Buffer Float32) (count : UInt32) : IO Cuda.KernelHandle :=
  castBF16ToF32Kernel.launchOn stream (elementConfig count) input output count

/-- Zero one Float32 allocation without constructing a host-sized byte array. -/
def zeroBufferF32 (stream : @& Cuda.Stream) (output : @& Cuda.Buffer Float32)
    (count : UInt32) : IO Cuda.KernelHandle :=
  zeroF32Kernel.launchOn stream (elementConfig count) output count

/-! ## Shared sequential/device-graph submissions -/

def submitRmsNormForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x weight output inverseRms : @& Cuda.Buffer Float32) (rows width : UInt32)
    (epsilon : Float32) : IO Unit :=
  executor.submit label
    (fun stream => rmsNormForwardF32Kernel.launchOn stream (rowWarpConfig rows)
      x weight output inverseRms rows width epsilon)
    (fun builder => rmsNormForwardF32Kernel.addToGraph builder (rowWarpConfig rows)
      x weight output inverseRms rows width epsilon)

def submitRmsNormBackwardInputF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x weight outputGradient inputGradient inverseRms : @& Cuda.Buffer Float32)
    (rows width : UInt32) : IO Unit :=
  executor.submit label
    (fun stream => rmsNormBackwardInputF32Kernel.launchOn stream (rowWarpConfig rows)
      x weight outputGradient inputGradient inverseRms rows width)
    (fun builder => rmsNormBackwardInputF32Kernel.addToGraph builder (rowWarpConfig rows)
      x weight outputGradient inputGradient inverseRms rows width)

def submitRmsNormBackwardWeightF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x outputGradient inverseRms weightGradient : @& Cuda.Buffer Float32)
    (rows width : UInt32) : IO Unit :=
  executor.submit label
    (fun stream => rmsNormBackwardWeightF32Kernel.launchOn stream (elementConfig width)
      x outputGradient inverseRms weightGradient rows width)
    (fun builder => rmsNormBackwardWeightF32Kernel.addToGraph builder (elementConfig width)
      x outputGradient inverseRms weightGradient rows width)

def submitWeightedGatedRmsNormForwardF32 (executor : @& Cuda.Qwen36.Executor)
    (label : String) (x z weight output inverseRms : @& Cuda.Buffer Float32)
    (rows width : UInt32) (epsilon : Float32) : IO Unit :=
  executor.submit label
    (fun stream => weightedGatedRmsNormForwardF32Kernel.launchOn stream (rowWarpConfig rows)
      x z weight output inverseRms rows width epsilon)
    (fun builder => weightedGatedRmsNormForwardF32Kernel.addToGraph builder (rowWarpConfig rows)
      x z weight output inverseRms rows width epsilon)

def submitWeightedGatedRmsNormBackwardF32 (executor : @& Cuda.Qwen36.Executor)
    (label : String) (x z weight outputGradient inputGradient gateGradient inverseRms :
      @& Cuda.Buffer Float32) (rows width : UInt32) : IO Unit :=
  executor.submit label
    (fun stream => weightedGatedRmsNormBackwardF32Kernel.launchOn stream (rowWarpConfig rows)
      x z weight outputGradient inputGradient gateGradient inverseRms rows width)
    (fun builder => weightedGatedRmsNormBackwardF32Kernel.addToGraph builder
      (rowWarpConfig rows) x z weight outputGradient inputGradient gateGradient inverseRms
      rows width)

def submitWeightedGatedRmsNormBackwardWeightF32 (executor : @& Cuda.Qwen36.Executor)
    (label : String) (x z outputGradient inverseRms weightGradient : @& Cuda.Buffer Float32)
    (rows width : UInt32) : IO Unit :=
  executor.submit label
    (fun stream => weightedGatedRmsNormBackwardWeightF32Kernel.launchOn stream
      (elementConfig width) x z outputGradient inverseRms weightGradient rows width)
    (fun builder => weightedGatedRmsNormBackwardWeightF32Kernel.addToGraph builder
      (elementConfig width) x z outputGradient inverseRms weightGradient rows width)

def submitRopeSequenceForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input invFreq output : @& Cuda.Buffer Float32)
    (tokens heads headDim half : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * heads * headDim)
  executor.submit label
    (fun stream => ropeSequenceForwardF32Kernel.launchOn stream launch input invFreq output
      tokens heads headDim half)
    (fun builder => ropeSequenceForwardF32Kernel.addToGraph builder launch input invFreq output
      tokens heads headDim half)

def submitRopeSequenceBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (outputGradient invFreq inputGradient : @& Cuda.Buffer Float32)
    (tokens heads headDim half : UInt32) : IO Unit :=
  let launch := elementConfig (tokens * heads * headDim)
  executor.submit label
    (fun stream => ropeSequenceBackwardF32Kernel.launchOn stream launch outputGradient invFreq
      inputGradient tokens heads headDim half)
    (fun builder => ropeSequenceBackwardF32Kernel.addToGraph builder launch outputGradient invFreq
      inputGradient tokens heads headDim half)

def submitRopeBatchedForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input invFreq output : @& Cuda.Buffer Float32) (layout : Cuda.Qwen36.SequenceLayout)
    (heads headDim half : UInt32) : IO Unit :=
  let launch := elementConfig (layout.tokens * heads * headDim)
  executor.submit label
    (fun stream => ropeBatchedForwardF32Kernel.launchOn stream launch input invFreq output
      layout.batchSize layout.sequenceLength heads headDim half)
    (fun builder => ropeBatchedForwardF32Kernel.addToGraph builder launch input invFreq output
      layout.batchSize layout.sequenceLength heads headDim half)

def submitRopeBatchedBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (outputGradient invFreq inputGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (heads headDim half : UInt32) : IO Unit :=
  let launch := elementConfig (layout.tokens * heads * headDim)
  executor.submit label
    (fun stream => ropeBatchedBackwardF32Kernel.launchOn stream launch outputGradient invFreq
      inputGradient layout.batchSize layout.sequenceLength heads headDim half)
    (fun builder => ropeBatchedBackwardF32Kernel.addToGraph builder launch outputGradient invFreq
      inputGradient layout.batchSize layout.sequenceLength heads headDim half)

def submitConv1dSiLUForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x weight output : @& Cuda.Buffer Float32) (seq channels : UInt32) : IO Unit :=
  let launch := elementConfig (seq * channels)
  executor.submit label
    (fun stream => conv1dSiLUForwardF32Kernel.launchOn stream launch x weight output seq channels)
    (fun builder => conv1dSiLUForwardF32Kernel.addToGraph builder launch x weight output
      seq channels)

def submitConv1dSiLUBackwardInputF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x weight outputGradient inputGradient : @& Cuda.Buffer Float32)
    (seq channels : UInt32) : IO Unit :=
  let launch := elementConfig (seq * channels)
  executor.submit label
    (fun stream => conv1dSiLUBackwardInputF32Kernel.launchOn stream launch x weight
      outputGradient inputGradient seq channels)
    (fun builder => conv1dSiLUBackwardInputF32Kernel.addToGraph builder launch x weight
      outputGradient inputGradient seq channels)

def submitConv1dSiLUBackwardWeightF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x weight outputGradient weightGradient : @& Cuda.Buffer Float32)
    (seq channels : UInt32) : IO Unit :=
  let launch := elementConfig (channels * 4)
  executor.submit label
    (fun stream => conv1dSiLUBackwardWeightF32Kernel.launchOn stream launch x weight
      outputGradient weightGradient seq channels)
    (fun builder => conv1dSiLUBackwardWeightF32Kernel.addToGraph builder launch x weight
      outputGradient weightGradient seq channels)

def submitConv1dSiLUBatchedForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x weight output : @& Cuda.Buffer Float32) (layout : Cuda.Qwen36.SequenceLayout)
    (channels : UInt32) : IO Unit :=
  let launch := elementConfig (layout.tokens * channels)
  executor.submit label
    (fun stream => conv1dSiLUBatchedForwardF32Kernel.launchOn stream launch x weight output
      layout.batchSize layout.sequenceLength channels)
    (fun builder => conv1dSiLUBatchedForwardF32Kernel.addToGraph builder launch x weight output
      layout.batchSize layout.sequenceLength channels)

def submitConv1dSiLUBatchedBackwardInputF32 (executor : @& Cuda.Qwen36.Executor)
    (label : String) (x weight outputGradient inputGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (channels : UInt32) : IO Unit :=
  let launch := elementConfig (layout.tokens * channels)
  executor.submit label
    (fun stream => conv1dSiLUBatchedBackwardInputF32Kernel.launchOn stream launch x weight
      outputGradient inputGradient layout.batchSize layout.sequenceLength channels)
    (fun builder => conv1dSiLUBatchedBackwardInputF32Kernel.addToGraph builder launch x weight
      outputGradient inputGradient layout.batchSize layout.sequenceLength channels)

def submitConv1dSiLUBatchedBackwardWeightF32 (executor : @& Cuda.Qwen36.Executor)
    (label : String) (x weight outputGradient weightGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (channels : UInt32) : IO Unit :=
  let launch := elementConfig (channels * 4)
  executor.submit label
    (fun stream => conv1dSiLUBatchedBackwardWeightF32Kernel.launchOn stream launch x weight
      outputGradient weightGradient layout.batchSize layout.sequenceLength channels)
    (fun builder => conv1dSiLUBatchedBackwardWeightF32Kernel.addToGraph builder launch x weight
      outputGradient weightGradient layout.batchSize layout.sequenceLength channels)

def submitL2normForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x output : @& Cuda.Buffer Float32) (rows width : UInt32) (epsilon : Float32) : IO Unit :=
  executor.submit label
    (fun stream => l2normForwardF32Kernel.launchOn stream (rowWarpConfig rows)
      x output rows width epsilon)
    (fun builder => l2normForwardF32Kernel.addToGraph builder (rowWarpConfig rows)
      x output rows width epsilon)

def submitL2normBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (x outputGradient inputGradient : @& Cuda.Buffer Float32)
    (rows width : UInt32) (epsilon : Float32) : IO Unit :=
  executor.submit label
    (fun stream => l2normBackwardF32Kernel.launchOn stream (rowWarpConfig rows)
      x outputGradient inputGradient rows width epsilon)
    (fun builder => l2normBackwardF32Kernel.addToGraph builder (rowWarpConfig rows)
      x outputGradient inputGradient rows width epsilon)

def submitSwigluForwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (gate up hidden : @& Cuda.Buffer Float32) (count : UInt32) : IO Unit :=
  executor.submit label
    (fun stream => swigluForwardF32Kernel.launchOn stream (elementConfig count)
      gate up hidden count)
    (fun builder => swigluForwardF32Kernel.addToGraph builder (elementConfig count)
      gate up hidden count)

def submitSwigluBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (gate up hiddenGradient gateGradient upGradient : @& Cuda.Buffer Float32)
    (count : UInt32) : IO Unit :=
  executor.submit label
    (fun stream => swigluBackwardF32Kernel.launchOn stream (elementConfig count)
      gate up hiddenGradient gateGradient upGradient count)
    (fun builder => swigluBackwardF32Kernel.addToGraph builder (elementConfig count)
      gate up hiddenGradient gateGradient upGradient count)

def submitEmbeddingGatherF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (table : @& Cuda.Buffer Float32) (ids : @& Cuda.Buffer UInt32)
    (output : @& Cuda.Buffer Float32) (count width : UInt32) : IO Unit :=
  let launch := elementConfig (count * width)
  executor.submit label
    (fun stream => embeddingGatherF32Kernel.launchOn stream launch table ids output count width)
    (fun builder => embeddingGatherF32Kernel.addToGraph builder launch table ids output count width)

/-- Submit an elementwise Float32 to BF16 cast to either a stream or a CUDA graph. -/
def submitCastF32ToBF16 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (input : @& Cuda.Buffer Float32) (output : @& Cuda.Buffer Cuda.BFloat16)
    (count : UInt32) : IO Unit :=
  let launch := elementConfig count
  executor.submit label
    (fun stream => castF32ToBF16Kernel.launchOn stream launch input output count)
    (fun builder => castF32ToBF16Kernel.addToGraph builder launch input output count)

/-- Submit a zero-copy frozen-BF16 embedding gather with Float32 publication. -/
def submitEmbeddingGatherFrozenBF16F32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (weight : @& FrozenBFloat16Weight) (ids : @& Cuda.Buffer UInt32)
    (output : @& Cuda.Buffer Float32) (count width : UInt32) : IO Unit := do
  unless weight.byteCount >= width.toUInt64 * 2 do
    throw <| IO.userError s!"Qwen3.6 {label} frozen BF16 row is truncated"
  unless weight.byteOffset + weight.byteCount <= (← weight.owner.byteSize).toUInt64 do
    throw <| IO.userError s!"Qwen3.6 {label} frozen BF16 range exceeds its owner"
  let launch := elementConfig (count * width)
  executor.submit label
    (fun stream => embeddingGatherFrozenBF16F32Kernel.launchOn stream launch weight.owner
      weight.byteOffset ids output count width)
    (fun builder => embeddingGatherFrozenBF16F32Kernel.addToGraph builder launch weight.owner
      weight.byteOffset ids output count width)

/-- Copy a checked immutable BF16 tensor into a newly allocated Float32 training vector. -/
def copyFrozenBF16ToF32 (weight : @& FrozenBFloat16Weight) (count : UInt32)
    (stream : @& Cuda.Stream) : IO (Cuda.Buffer Float32) := do
  unless weight.byteCount == count.toUInt64 * 2 do
    throw <| IO.userError "Qwen3.6 frozen BF16 vector size mismatch"
  unless weight.byteOffset + weight.byteCount <= (← weight.owner.byteSize).toUInt64 do
    throw <| IO.userError "Qwen3.6 frozen BF16 vector range exceeds its owner"
  let output ← Cuda.Buffer.alloc Float32 count.toUSize
  (← copyFrozenBF16ToF32Kernel.launchOn stream (elementConfig count) weight.owner
    weight.byteOffset output count).waitChecked "Qwen3.6 frozen BF16 vector widening"
  return output

end Cuda.Qwen36.Primitives
