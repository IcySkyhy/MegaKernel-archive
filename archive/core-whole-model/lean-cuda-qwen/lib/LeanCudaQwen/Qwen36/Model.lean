/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
module

prelude
public import LeanCudaQwen.Qwen36.Attention
public import LeanCudaQwen.Qwen36.DeltaNet
public import LeanCudaQwen.Qwen36.MLP

public section

/-!
# Qwen3.6 complete numerical model

Float32 reference composition for embedding, an arbitrary hybrid decoder stack, final RMSNorm,
the language-model head, stable mean cross-entropy, the complete reverse pass, and deterministic
embedding scatter. Component launches are intentionally sequenced and checked: this is the
independent numerical route used to gate the later single-launch persistent training schedule.

The generic cross-entropy and embedding-adjoint device bodies are exposed so the persistent
schedule can reuse their equations without device-side child launches.
-/

namespace Cuda.Qwen36.Model

open Cuda.Qwen36.Primitives

/-! ## Direct preference optimization scalars -/

/-- DPO logit for one chosen/rejected pair against a frozen reference policy. -/
@[expose, cuda_device, always_inline]
def dpoMarginF32 (beta policyChosen policyRejected referenceChosen referenceRejected : Float32) :
    Float32 :=
  beta * ((policyChosen - policyRejected) - (referenceChosen - referenceRejected))

/-- Stable `-log sigmoid(margin)` used by direct preference optimization. -/
@[expose, cuda_device, always_inline]
def dpoLossFromMarginF32 (margin : Float32) : Float32 :=
  DeltaNet.softplusF32 (-margin)

/-- Positive multiplier of the chosen sequence's cross-entropy logit gradient. -/
@[expose, cuda_device, always_inline]
def dpoChosenCoefficientF32 (beta margin : Float32) : Float32 :=
  beta * sigmoidF32 (-margin)

/-! ## Group-relative policy optimization scalars -/

/-- Population-normalized reward used as the outcome advantage for one GRPO group member. -/
@[expose, cuda_device, always_inline]
def grpoAdvantageF32 (reward mean variance epsilon : Float32) : Float32 :=
  (reward - mean) / Float32.sqrt (variance + epsilon)

/-- PPO probability ratio for one sampled target token. -/
@[expose, cuda_device, always_inline]
def grpoRatioF32 (policyLogProbability oldLogProbability : Float32) : Float32 :=
  Float32.exp (policyLogProbability - oldLogProbability)

/-- Clamp one GRPO ratio to the symmetric PPO trust region. -/
@[expose, cuda_device, always_inline]
def grpoClippedRatioF32 (ratio clipEpsilon : Float32) : Float32 :=
  let lower := oneF32 - clipEpsilon
  let upper := oneF32 + clipEpsilon
  if ratio < lower then lower else if ratio > upper then upper else ratio

/-- Whether the unclipped GRPO surrogate still contributes a policy gradient. -/
@[expose, cuda_device, always_inline]
def grpoRatioActiveF32 (advantage ratio clipEpsilon : Float32) : Bool :=
  if advantage >= 0 then ratio <= oneF32 + clipEpsilon
  else ratio >= oneF32 - clipEpsilon

/-- Minimum of the unclipped and clipped PPO surrogates. -/
@[expose, cuda_device, always_inline]
def grpoSurrogateF32 (advantage ratio clipEpsilon : Float32) : Float32 :=
  let unclipped := advantage * ratio
  let clipped := advantage * grpoClippedRatioF32 ratio clipEpsilon
  if unclipped < clipped then unclipped else clipped

/-- Positive, unbiased sampled-token estimator of `KL(policy || reference)`. -/
@[expose, cuda_device, always_inline]
def grpoKLF32 (policyLogProbability referenceLogProbability : Float32) : Float32 :=
  let referenceRatio := Float32.exp (referenceLogProbability - policyLogProbability)
  referenceRatio + policyLogProbability - referenceLogProbability - oneF32

/--
Cross-entropy-form coefficient for the clipped policy surrogate plus direct reference KL.
The logit VJP is `coefficient * (softmax(logits) - oneHot(target))`.
-/
@[expose, cuda_device, always_inline]
def grpoCoefficientF32 (advantage ratio referenceRatio clipEpsilon klBeta : Float32) : Float32 :=
  let policy := if grpoRatioActiveF32 advantage ratio clipEpsilon then advantage * ratio else 0
  policy + klBeta * (referenceRatio - oneF32)

/-- POD objective descriptor shared by the modular kernel and fused training programs. -/
structure GRPOObjectiveF32 where
  sequences : UInt32
  sequenceLength : UInt32
  groupSize : UInt32
  clipEpsilon : Float32
  klBeta : Float32
  advantageEpsilon : Float32
  deriving Cuda.POD

/-- Runtime dimensions shared by every layer in one Qwen3.6 text model execution. -/
structure ShapeF32 where
  tokens : UInt32
  hidden : UInt32
  intermediate : UInt32
  vocabulary : UInt32
  linearKeyHeads : UInt32
  linearValueHeads : UInt32
  linearKeyWidth : UInt32
  linearValueWidth : UInt32
  attentionQueryHeads : UInt32
  attentionKeyValueHeads : UInt32
  attentionHeadWidth : UInt32
  rotaryHalf : UInt32
  epsilon : Float32
  queryScale : Float32
  attentionScale : Float32

/-- Immutable MLP weights shared by either mixer kind. -/
structure MLPWeightsF32 where
  norm : Cuda.Buffer Float32
  gate : Projection.WeightsF32
  up : Projection.WeightsF32
  down : Projection.WeightsF32

/-- Immutable mixer weights for one decoder layer. -/
inductive MixerWeightsF32 where
  | deltaNet (weights : DeltaNet.StageWeightsF32)
  | attention (weights : Attention.StageWeightsF32)

/-- Immutable weights for one decoder layer. -/
structure LayerWeightsF32 where
  mixer : MixerWeightsF32
  mlp : MLPWeightsF32

/-- Token-embedding storage for either Float32 oracle weights or a frozen BF16 checkpoint view. -/
inductive EmbeddingWeightsF32 where
  | f32 (weight : Cuda.Buffer Float32)
  | frozenBF16 (weight : FrozenBFloat16Weight)

instance : Coe (Cuda.Buffer Float32) EmbeddingWeightsF32 where
  coe := .f32

/-- Immutable full-model weights. -/
structure WeightsF32 where
  embedding : EmbeddingWeightsF32
  layers : Array LayerWeightsF32
  finalNorm : Cuda.Buffer Float32
  lmHead : Projection.WeightsF32

/-- Saved MLP activations for one decoder layer. -/
structure MLPForwardF32 where
  normalized : Cuda.Buffer Float32
  inverseRms : Cuda.Buffer Float32
  gate : Cuda.Buffer Float32
  up : Cuda.Buffer Float32
  activation : Cuda.Buffer Float32
  output : Cuda.Buffer Float32

/-- Saved mixer activations for one decoder layer. -/
inductive MixerForwardF32 where
  | deltaNet (buffers : DeltaNet.StageForwardF32)
  | attention (buffers : Attention.StageForwardF32)

/-- Saved forward activations for one decoder layer. -/
structure LayerForwardF32 where
  mixer : MixerForwardF32
  mixerResidual : Cuda.Buffer Float32
  mlp : MLPForwardF32
  output : Cuda.Buffer Float32

/-- Full-model forward values and loss scratch. -/
structure ForwardF32 where
  embedding : Cuda.Buffer Float32
  layers : Array LayerForwardF32
  finalHidden : Cuda.Buffer Float32
  finalInverseRms : Cuda.Buffer Float32
  logits : Cuda.Buffer Float32
  rowLoss : Cuda.Buffer Float32
  loss : Cuda.Buffer Float32
  logitGradient : Cuda.Buffer Float32

/-- MLP VJP scratch and trainable gradients for one decoder layer. -/
structure MLPBackwardF32 where
  activationGradient : Cuda.Buffer Float32
  gateGradient : Cuda.Buffer Float32
  upGradient : Cuda.Buffer Float32
  normGradientGate : Cuda.Buffer Float32
  normGradientUp : Cuda.Buffer Float32
  normGradient : Cuda.Buffer Float32
  inputGradient : Cuda.Buffer Float32
  normWeightGradient : Cuda.Buffer Float32
  gateWeightGradient : Cuda.Buffer Float32
  upWeightGradient : Cuda.Buffer Float32
  downWeightGradient : Cuda.Buffer Float32

/-- Mixer VJP scratch and trainable gradients for one decoder layer. -/
inductive MixerBackwardF32 where
  | deltaNet (buffers : DeltaNet.StageBackwardF32)
  | attention (buffers : Attention.StageBackwardF32)

/-- Reverse values for one decoder layer. -/
structure LayerBackwardF32 where
  mlp : MLPBackwardF32
  mixerResidualGradient : Cuda.Buffer Float32
  mixer : MixerBackwardF32
  inputGradient : Cuda.Buffer Float32

/-- Full-model reverse scratch and trainable gradients. -/
structure BackwardF32 where
  finalHiddenGradient : Cuda.Buffer Float32
  lmHeadWeightGradient : Cuda.Buffer Float32
  decoderOutputGradient : Cuda.Buffer Float32
  finalNormWeightGradient : Cuda.Buffer Float32
  layers : Array LayerBackwardF32
  embeddingWeightGradient : Cuda.Buffer Float32

namespace Internal

@[always_inline]
partial def rowMaximumF32 (logits : Cuda.DevicePtr Float32) (row classes column stride : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < classes then
    let value ← Cuda.loadFloat32 logits (row * classes + column).toUSize
    rowMaximumF32 logits row classes (column + stride) stride (max accumulator value)
  else
    return accumulator

@[always_inline]
partial def rowExponentSumF32 (logits : Cuda.DevicePtr Float32) (row classes column stride :
    UInt32) (maximum accumulator : Float32) : Cuda.DeviceM Float32 := do
  if column < classes then
    let value ← Cuda.loadFloat32 logits (row * classes + column).toUSize
    rowExponentSumF32 logits row classes (column + stride) stride maximum
      (accumulator + Float32.exp (value - maximum))
  else
    return accumulator

@[always_inline]
partial def storeLogitGradientsF32 (logits gradient : Cuda.DevicePtr Float32)
    (row classes column stride label : UInt32) (maximum inverseSum reductionScale : Float32) :
    Cuda.DeviceM Unit := do
  if column < classes then
    let value ← Cuda.loadFloat32 logits (row * classes + column).toUSize
    let probability := Float32.exp (value - maximum) * inverseSum
    let target := if column == label then oneF32 else zeroF32
    Cuda.storeFloat32 gradient (row * classes + column).toUSize
      ((probability - target) * reductionScale)
    storeLogitGradientsF32 logits gradient row classes (column + stride) stride label maximum
      inverseSum reductionScale

@[always_inline]
partial def sumRowsF32 (rowLoss : Cuda.DevicePtr Float32) (rows row : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if row < rows then
    sumRowsF32 rowLoss rows (row + 1)
      (accumulator + (← Cuda.loadFloat32 rowLoss row.toUSize))
  else
    return accumulator

@[always_inline]
partial def zeroLogitGradientsF32 (gradient : Cuda.DevicePtr Float32)
    (row classes column stride : UInt32) : Cuda.DeviceM Unit := do
  if column < classes then
    Cuda.storeFloat32 gradient (row * classes + column).toUSize zeroF32
    zeroLogitGradientsF32 gradient row classes (column + stride) stride

@[always_inline]
partial def sumSequenceRowsF32 (rowLogProbability : Cuda.DevicePtr Float32)
    (start length offset : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if offset < length then
    sumSequenceRowsF32 rowLogProbability start length (offset + 1)
      (accumulator + (← Cuda.loadFloat32 rowLogProbability (start + offset).toUSize))
  else
    return accumulator

@[always_inline]
partial def accumulateDPOPairsF32 (policy reference coefficients : Cuda.DevicePtr Float32)
    (pairs pair : UInt32) (beta inversePairs accumulator : Float32) : Cuda.DeviceM Float32 := do
  if pair < pairs then
    let chosen := pair * 2
    let rejected := chosen + 1
    let policyChosen ← Cuda.loadFloat32 policy chosen.toUSize
    let policyRejected ← Cuda.loadFloat32 policy rejected.toUSize
    let referenceChosen ← Cuda.loadFloat32 reference chosen.toUSize
    let referenceRejected ← Cuda.loadFloat32 reference rejected.toUSize
    let margin := dpoMarginF32 beta policyChosen policyRejected referenceChosen referenceRejected
    let coefficient := dpoChosenCoefficientF32 beta margin * inversePairs
    Cuda.storeFloat32 coefficients chosen.toUSize coefficient
    Cuda.storeFloat32 coefficients rejected.toUSize (-coefficient)
    accumulateDPOPairsF32 policy reference coefficients pairs (pair + 1) beta inversePairs
      (accumulator + dpoLossFromMarginF32 margin * inversePairs)
  else
    return accumulator

structure GRPOAccumulatorF32 where
  loss : Float32
  meanKL : Float32
  clipped : Float32
  enabled : Float32
  deriving Cuda.POD

@[always_inline]
partial def sumGroupRewardsF32 (rewards : Cuda.DevicePtr Float32)
    (start count offset : UInt32) (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if offset < count then
    sumGroupRewardsF32 rewards start count (offset + 1)
      (accumulator + (← Cuda.loadFloat32 rewards (start + offset).toUSize))
  else
    return accumulator

@[always_inline]
partial def sumGroupSquaredDeviationsF32 (rewards : Cuda.DevicePtr Float32)
    (start count offset : UInt32) (mean accumulator : Float32) : Cuda.DeviceM Float32 := do
  if offset < count then
    let difference := (← Cuda.loadFloat32 rewards (start + offset).toUSize) - mean
    sumGroupSquaredDeviationsF32 rewards start count (offset + 1) mean
      (accumulator + difference * difference)
  else
    return accumulator

@[always_inline]
partial def countMaskedRowsF32 (masks : Cuda.DevicePtr UInt32)
    (start length offset count : UInt32) : Cuda.DeviceM UInt32 := do
  if offset < length then
    let enabled ← Cuda.loadUInt32 masks (start + offset).toUSize
    countMaskedRowsF32 masks start length (offset + 1)
      (if enabled == 0 then count else count + 1)
  else
    return count

@[always_inline]
partial def accumulateGRPORowsF32
    (policy old reference coefficients : Cuda.DevicePtr Float32)
    (masks : Cuda.DevicePtr UInt32) (start length offset : UInt32)
    (advantage clipEpsilon klBeta scale : Float32) (accumulator : GRPOAccumulatorF32) :
    Cuda.DeviceM GRPOAccumulatorF32 := do
  if offset < length then
    let row := start + offset
    let enabled ← Cuda.loadUInt32 masks row.toUSize
    if enabled == 0 then
      Cuda.storeFloat32 coefficients row.toUSize zeroF32
      accumulateGRPORowsF32 policy old reference coefficients masks start length (offset + 1)
        advantage clipEpsilon klBeta scale accumulator
    else
      let policyLogProbability ← Cuda.loadFloat32 policy row.toUSize
      let oldLogProbability ← Cuda.loadFloat32 old row.toUSize
      let referenceLogProbability ← Cuda.loadFloat32 reference row.toUSize
      let ratio := grpoRatioF32 policyLogProbability oldLogProbability
      let referenceRatio := Float32.exp (referenceLogProbability - policyLogProbability)
      let active := grpoRatioActiveF32 advantage ratio clipEpsilon
      let kl := grpoKLF32 policyLogProbability referenceLogProbability
      Cuda.storeFloat32 coefficients row.toUSize
        (grpoCoefficientF32 advantage ratio referenceRatio clipEpsilon klBeta * scale)
      accumulateGRPORowsF32 policy old reference coefficients masks start length (offset + 1)
        advantage clipEpsilon klBeta scale {
          loss := accumulator.loss +
            (-grpoSurrogateF32 advantage ratio clipEpsilon + klBeta * kl) * scale
          meanKL := accumulator.meanKL + kl * scale
          clipped := accumulator.clipped + if active then 0 else 1
          enabled := accumulator.enabled + 1
        }
  else
    return accumulator

@[always_inline]
partial def accumulateGRPOSequencesF32
    (policy old reference rewards advantages coefficients : Cuda.DevicePtr Float32)
    (masks : Cuda.DevicePtr UInt32) (sequences sequenceLength groupSize sequence : UInt32)
    (clipEpsilon klBeta advantageEpsilon inverseSequences : Float32)
    (accumulator : GRPOAccumulatorF32) : Cuda.DeviceM GRPOAccumulatorF32 := do
  if sequence < sequences then
    let groupStart := (sequence / groupSize) * groupSize
    let mean ← sumGroupRewardsF32 rewards groupStart groupSize 0 zeroF32
    let mean := mean / groupSize.toFloat32
    let variance ← sumGroupSquaredDeviationsF32 rewards groupStart groupSize 0 mean zeroF32
    let variance := variance / groupSize.toFloat32
    let reward ← Cuda.loadFloat32 rewards sequence.toUSize
    let advantage := grpoAdvantageF32 reward mean variance advantageEpsilon
    Cuda.storeFloat32 advantages sequence.toUSize advantage
    let start := sequence * sequenceLength
    let count ← countMaskedRowsF32 masks start sequenceLength 0 0
    let scale := inverseSequences / count.toFloat32
    let accumulator ← accumulateGRPORowsF32 policy old reference coefficients masks start
      sequenceLength 0 advantage clipEpsilon klBeta scale accumulator
    accumulateGRPOSequencesF32 policy old reference rewards advantages coefficients masks
      sequences sequenceLength groupSize (sequence + 1) clipEpsilon klBeta advantageEpsilon
      inverseSequences accumulator
  else
    return accumulator

@[always_inline]
partial def embeddingGradientElementF32 (ids : Cuda.DevicePtr UInt32)
    (outputGradient : Cuda.DevicePtr Float32) (tokens width vocabulary token id column : UInt32)
    (accumulator : Float32) : Cuda.DeviceM Float32 := do
  if token < tokens then
    let tokenId ← Cuda.loadUInt32 ids token.toUSize
    let accumulator ← if tokenId == id then
        pure (accumulator + (← Cuda.loadFloat32 outputGradient
          (token * width + column).toUSize))
      else
        pure accumulator
    embeddingGradientElementF32 ids outputGradient tokens width vocabulary (token + 1) id column
      accumulator
  else
    return accumulator

end Internal

/-- Stable mean cross-entropy and Float32 logit VJP, one 256-thread block per token row. -/
@[cuda_kernel]
def crossEntropyF32Kernel (logits : Cuda.DevicePtr Float32) (targets : Cuda.DevicePtr UInt32)
    (gradient rowLoss : Cuda.DevicePtr Float32) (rows classes : UInt32)
    (reductionScale : Float32) : Cuda.DeviceM Unit := do
  let row ← Cuda.blockIdxX
  if row < rows then
    let thread ← Cuda.threadIdxX
    let scratchRaw ← Cuda.dynamicShared (α := Float32) 8
    let scratch : Cuda.Collective.BlockScratch 8 := scratchRaw
    let localMaximum ← Internal.rowMaximumF32 logits row classes thread 256
      (Float32.ofBits 0xff800000)
    let maximum ← Cuda.Collective.blockMax scratch localMaximum
    let localSum ← Internal.rowExponentSumF32 logits row classes thread 256 maximum zeroF32
    let denominator ← Cuda.Collective.blockSum scratch localSum
    let inverseSum := oneF32 / denominator
    let label ← Cuda.loadUInt32 targets row.toUSize
    Internal.storeLogitGradientsF32 logits gradient row classes thread 256 label maximum inverseSum
      reductionScale
    if thread == 0 then
      let labelLogit ← Cuda.loadFloat32 logits (row * classes + label).toUSize
      Cuda.storeFloat32 rowLoss row.toUSize
        ((maximum + Float32.log denominator - labelLogit) * reductionScale)

/-- Deterministically reduce the scaled per-row loss into one scalar. -/
@[cuda_kernel]
def reduceLossF32Kernel (rowLoss loss : Cuda.DevicePtr Float32) (rows : UInt32) :
    Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear == 0 then
    Cuda.storeFloat32 loss 0 (← Internal.sumRowsF32 rowLoss rows 0 zeroF32)

/-- Stable target-token log probabilities with explicit completion/padding masking. -/
@[cuda_kernel]
def tokenLogProbabilityF32Kernel (logits : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (rowLogProbability : Cuda.DevicePtr Float32)
    (rows classes : UInt32) : Cuda.DeviceM Unit := do
  let row ← Cuda.blockIdxX
  if row < rows then
    let thread ← Cuda.threadIdxX
    let enabled ← Cuda.loadUInt32 masks row.toUSize
    if enabled == 0 then
      if thread == 0 then
        Cuda.storeFloat32 rowLogProbability row.toUSize zeroF32
    else
      let scratchRaw ← Cuda.dynamicShared (α := Float32) 8
      let scratch : Cuda.Collective.BlockScratch 8 := scratchRaw
      let localMaximum ← Internal.rowMaximumF32 logits row classes thread 256
        (Float32.ofBits 0xff800000)
      let maximum ← Cuda.Collective.blockMax scratch localMaximum
      let localSum ← Internal.rowExponentSumF32 logits row classes thread 256 maximum zeroF32
      let denominator ← Cuda.Collective.blockSum scratch localSum
      if thread == 0 then
        let label ← Cuda.loadUInt32 targets row.toUSize
        let labelLogit ← Cuda.loadFloat32 logits (row * classes + label).toUSize
        Cuda.storeFloat32 rowLogProbability row.toUSize
          (labelLogit - maximum - Float32.log denominator)

/-- Sum masked token log probabilities into one score per fixed-length sequence. -/
@[cuda_kernel]
def reduceSequenceLogProbabilityF32Kernel (rowLogProbability sequenceLogProbability :
    Cuda.DevicePtr Float32) (sequences sequenceLength : UInt32) : Cuda.DeviceM Unit := do
  let sequence ← elementIndex
  if sequence < sequences then
    Cuda.storeFloat32 sequenceLogProbability sequence.toUSize
      (← Internal.sumSequenceRowsF32 rowLogProbability (sequence * sequenceLength)
        sequenceLength 0 zeroF32)

/-- Mean DPO loss and signed chosen/rejected sequence coefficients. -/
@[cuda_kernel]
def dpoObjectiveF32Kernel (policy reference coefficients loss : Cuda.DevicePtr Float32)
    (pairs : UInt32) (beta : Float32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear == 0 then
    let inversePairs := oneF32 / pairs.toFloat32
    Cuda.storeFloat32 loss 0
      (← Internal.accumulateDPOPairsF32 policy reference coefficients pairs 0 beta inversePairs
        zeroF32)

/--
Evaluate the complete clipped outcome-GRPO objective from one logical device element. Keeping this
body independent of its ordinary kernel wrapper lets a dedicated training megakernel use exactly
the same reward normalization, clipping, sampled KL, and diagnostic equations as the modular
oracle.
-/
@[expose, cuda_device, always_inline]
def grpoObjectiveElementF32
    (policy old reference rewards advantages coefficients statistics loss : Cuda.DevicePtr Float32)
    (masks : Cuda.DevicePtr UInt32) (objective : GRPOObjectiveF32) (linear : UInt32) :
    Cuda.DeviceM Unit := do
  if linear == 0 then
    let result ← Internal.accumulateGRPOSequencesF32 policy old reference rewards advantages
      coefficients masks objective.sequences objective.sequenceLength objective.groupSize 0
      objective.clipEpsilon objective.klBeta objective.advantageEpsilon
      (oneF32 / objective.sequences.toFloat32) {
        loss := 0, meanKL := 0, clipped := 0, enabled := 0
      }
    Cuda.storeFloat32 loss 0 result.loss
    Cuda.storeFloat32 statistics 0 result.meanKL
    Cuda.storeFloat32 statistics 1
      (if result.enabled == 0 then 0 else result.clipped / result.enabled)

/--
Clipped outcome-GRPO objective. Rewards are normalized independently inside each adjacent group;
policy, old-policy, and reference values are masked target-token log probabilities.
-/
@[cuda_kernel]
def grpoObjectiveF32Kernel
    (policy old reference rewards advantages coefficients statistics loss : Cuda.DevicePtr Float32)
    (masks : Cuda.DevicePtr UInt32) (sequences sequenceLength groupSize : UInt32)
    (clipEpsilon klBeta advantageEpsilon : Float32) : Cuda.DeviceM Unit := do
  grpoObjectiveElementF32 policy old reference rewards advantages coefficients statistics loss
    masks { sequences, sequenceLength, groupSize, clipEpsilon, klBeta, advantageEpsilon }
    (← elementIndex)

/-- Logit VJP for DPO, with zero gradients on prompt and padding rows. -/
@[cuda_kernel]
def dpoGradientF32Kernel (logits : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (coefficients gradient : Cuda.DevicePtr Float32)
    (rows classes sequenceLength : UInt32) : Cuda.DeviceM Unit := do
  let row ← Cuda.blockIdxX
  if row < rows then
    let thread ← Cuda.threadIdxX
    let enabled ← Cuda.loadUInt32 masks row.toUSize
    if enabled == 0 then
      Internal.zeroLogitGradientsF32 gradient row classes thread 256
    else
      let scratchRaw ← Cuda.dynamicShared (α := Float32) 8
      let scratch : Cuda.Collective.BlockScratch 8 := scratchRaw
      let localMaximum ← Internal.rowMaximumF32 logits row classes thread 256
        (Float32.ofBits 0xff800000)
      let maximum ← Cuda.Collective.blockMax scratch localMaximum
      let localSum ← Internal.rowExponentSumF32 logits row classes thread 256 maximum zeroF32
      let denominator ← Cuda.Collective.blockSum scratch localSum
      let label ← Cuda.loadUInt32 targets row.toUSize
      let coefficient ← Cuda.loadFloat32 coefficients (row / sequenceLength).toUSize
      Internal.storeLogitGradientsF32 logits gradient row classes thread 256 label maximum
        (oneF32 / denominator) coefficient

/-- Completion-masked logit VJP using one precomputed cross-entropy coefficient per token row. -/
@[cuda_kernel]
def grpoGradientF32Kernel (logits : Cuda.DevicePtr Float32)
    (targets masks : Cuda.DevicePtr UInt32) (coefficients gradient : Cuda.DevicePtr Float32)
    (rows classes : UInt32) : Cuda.DeviceM Unit := do
  let row ← Cuda.blockIdxX
  if row < rows then
    let thread ← Cuda.threadIdxX
    let enabled ← Cuda.loadUInt32 masks row.toUSize
    if enabled == 0 then
      Internal.zeroLogitGradientsF32 gradient row classes thread 256
    else
      let scratchRaw ← Cuda.dynamicShared (α := Float32) 8
      let scratch : Cuda.Collective.BlockScratch 8 := scratchRaw
      let localMaximum ← Internal.rowMaximumF32 logits row classes thread 256
        (Float32.ofBits 0xff800000)
      let maximum ← Cuda.Collective.blockMax scratch localMaximum
      let localSum ← Internal.rowExponentSumF32 logits row classes thread 256 maximum zeroF32
      let denominator ← Cuda.Collective.blockSum scratch localSum
      let label ← Cuda.loadUInt32 targets row.toUSize
      let coefficient ← Cuda.loadFloat32 coefficients row.toUSize
      Internal.storeLogitGradientsF32 logits gradient row classes thread 256 label maximum
        (oneF32 / denominator) coefficient

/-- Deterministic embedding scatter-adjoint, one thread per `[vocabulary, width]` element. -/
@[cuda_kernel]
def embeddingBackwardF32Kernel (ids : Cuda.DevicePtr UInt32)
    (outputGradient weightGradient : Cuda.DevicePtr Float32)
    (tokens width vocabulary : UInt32) : Cuda.DeviceM Unit := do
  let linear ← elementIndex
  if linear < vocabulary * width then
    let id := linear / width
    let column := linear % width
    let value ← Internal.embeddingGradientElementF32 ids outputGradient tokens width vocabulary 0
      id column zeroF32
    Cuda.storeFloat32 weightGradient linear.toUSize value

/-- Launch stable mean cross-entropy and write both per-row loss and the Float32 logit VJP. -/
def crossEntropyF32 (stream : @& Cuda.Stream) (logits : @& Cuda.Buffer Float32)
    (targets : @& Cuda.Buffer UInt32) (gradient rowLoss : @& Cuda.Buffer Float32)
    (rows classes : UInt32) : IO Cuda.KernelHandle :=
  crossEntropyF32Kernel.launchOn stream {
    grid := { x := rows }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  } logits targets gradient rowLoss rows classes (oneF32 / rows.toFloat32)

/-- Reduce the already mean-scaled per-row cross-entropy values. -/
def reduceLossF32 (stream : @& Cuda.Stream) (rowLoss loss : @& Cuda.Buffer Float32)
    (rows : UInt32) : IO Cuda.KernelHandle :=
  reduceLossF32Kernel.launchOn stream (elementConfig 1) rowLoss loss rows

private def checkDPOGeometry (layout : Cuda.Qwen36.SequenceLayout) (beta : Float32) : IO Unit := do
  let layout ← match layout.check with
    | .ok layout => pure layout
    | .error message => throw <| IO.userError message
  unless layout.batchSize ≥ 2 && layout.batchSize % 2 == 0 do
    throw <| IO.userError "Qwen3.6 DPO requires an even sequence batch ordered chosen/rejected"
  unless beta > 0 && (beta.toBits &&& 0x7f800000) != 0x7f800000 do
    throw <| IO.userError "Qwen3.6 DPO beta must be positive and finite"

private def checkGRPOGeometry (layout : Cuda.Qwen36.SequenceLayout) (groupSize : UInt32)
    (clipEpsilon klBeta advantageEpsilon : Float32) : IO Unit := do
  let layout ← match layout.check with
    | .ok layout => pure layout
    | .error message => throw <| IO.userError message
  unless groupSize >= 2 && layout.batchSize % groupSize == 0 do
    throw <| IO.userError
      "Qwen3.6 GRPO requires adjacent groups of at least two sequences dividing the batch"
  unless clipEpsilon > 0 && clipEpsilon < 1 &&
      (clipEpsilon.toBits &&& 0x7f800000) != 0x7f800000 do
    throw <| IO.userError "Qwen3.6 GRPO clip epsilon must be finite and lie in (0, 1)"
  unless klBeta >= 0 && (klBeta.toBits &&& 0x7f800000) != 0x7f800000 do
    throw <| IO.userError "Qwen3.6 GRPO KL beta must be nonnegative and finite"
  unless advantageEpsilon > 0 &&
      (advantageEpsilon.toBits &&& 0x7f800000) != 0x7f800000 do
    throw <| IO.userError "Qwen3.6 GRPO advantage epsilon must be positive and finite"

/-- Launch completion-masked target-token log-probability scoring. -/
def tokenLogProbabilityF32 (stream : @& Cuda.Stream) (logits : @& Cuda.Buffer Float32)
    (targets masks : @& Cuda.Buffer UInt32) (rowLogProbability : @& Cuda.Buffer Float32)
    (rows classes : UInt32) : IO Cuda.KernelHandle :=
  tokenLogProbabilityF32Kernel.launchOn stream {
    grid := { x := rows }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  } logits targets masks rowLogProbability rows classes

/-- Launch fixed-sequence target-log-probability reduction. -/
def reduceSequenceLogProbabilityF32 (stream : @& Cuda.Stream)
    (rowLogProbability sequenceLogProbability : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) : IO Cuda.KernelHandle :=
  reduceSequenceLogProbabilityF32Kernel.launchOn stream (elementConfig layout.batchSize)
    rowLogProbability sequenceLogProbability layout.batchSize layout.sequenceLength

/-- Launch the mean DPO logistic objective for adjacent chosen/rejected pairs. -/
def dpoObjectiveF32 (stream : @& Cuda.Stream) (policy reference coefficients loss :
    @& Cuda.Buffer Float32) (layout : Cuda.Qwen36.SequenceLayout) (beta : Float32) :
    IO Cuda.KernelHandle := do
  checkDPOGeometry layout beta
  dpoObjectiveF32Kernel.launchOn stream (elementConfig 1) policy reference coefficients loss
    (layout.batchSize / 2) beta

/-- Launch the completion-masked DPO logit VJP. -/
def dpoGradientF32 (stream : @& Cuda.Stream) (logits : @& Cuda.Buffer Float32)
    (targets masks : @& Cuda.Buffer UInt32) (coefficients gradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (classes : UInt32) : IO Cuda.KernelHandle := do
  checkDPOGeometry layout 1
  dpoGradientF32Kernel.launchOn stream {
    grid := { x := layout.tokens }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  } logits targets masks coefficients gradient layout.tokens classes layout.sequenceLength

/-- Launch clipped outcome-GRPO and write advantages, per-row VJP coefficients, and metrics. -/
def grpoObjectiveF32 (stream : @& Cuda.Stream)
    (policy old reference rewards advantages coefficients statistics loss : @& Cuda.Buffer Float32)
    (masks : @& Cuda.Buffer UInt32) (layout : Cuda.Qwen36.SequenceLayout) (groupSize : UInt32)
    (clipEpsilon klBeta advantageEpsilon : Float32) : IO Cuda.KernelHandle := do
  checkGRPOGeometry layout groupSize clipEpsilon klBeta advantageEpsilon
  grpoObjectiveF32Kernel.launchOn stream (elementConfig 1) policy old reference rewards advantages
    coefficients statistics loss masks layout.batchSize layout.sequenceLength groupSize clipEpsilon
    klBeta advantageEpsilon

/-- Launch the completion-masked GRPO logit VJP. -/
def grpoGradientF32 (stream : @& Cuda.Stream) (logits : @& Cuda.Buffer Float32)
    (targets masks : @& Cuda.Buffer UInt32) (coefficients gradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (classes groupSize : UInt32) : IO Cuda.KernelHandle := do
  checkGRPOGeometry layout groupSize 0.2 0 1e-6
  grpoGradientF32Kernel.launchOn stream {
    grid := { x := layout.tokens }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  } logits targets masks coefficients gradient layout.tokens classes

/-- Deterministic embedding weight VJP. -/
def embeddingBackwardF32 (stream : @& Cuda.Stream) (ids : @& Cuda.Buffer UInt32)
    (outputGradient weightGradient : @& Cuda.Buffer Float32)
    (tokens width vocabulary : UInt32) : IO Cuda.KernelHandle :=
  embeddingBackwardF32Kernel.launchOn stream (elementConfig (vocabulary * width)) ids
    outputGradient weightGradient tokens width vocabulary

/-! ## Shared sequential/device-graph submissions -/

def submitCrossEntropyF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (logits : @& Cuda.Buffer Float32) (targets : @& Cuda.Buffer UInt32)
    (gradient rowLoss : @& Cuda.Buffer Float32) (rows classes : UInt32) : IO Unit :=
  let launch : Cuda.LaunchConfig := {
    grid := { x := rows }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  }
  let reductionScale := oneF32 / rows.toFloat32
  executor.submit label
    (fun stream => crossEntropyF32Kernel.launchOn stream launch logits targets gradient rowLoss rows
      classes reductionScale)
    (fun builder => crossEntropyF32Kernel.addToGraph builder launch logits targets gradient rowLoss
      rows classes reductionScale)

def submitReduceLossF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (rowLoss loss : @& Cuda.Buffer Float32) (rows : UInt32) : IO Unit :=
  let launch := elementConfig 1
  executor.submit label
    (fun stream => reduceLossF32Kernel.launchOn stream launch rowLoss loss rows)
    (fun builder => reduceLossF32Kernel.addToGraph builder launch rowLoss loss rows)

/-- Submit completion-masked target-token log-probability scoring. -/
def submitTokenLogProbabilityF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (logits : @& Cuda.Buffer Float32) (targets masks : @& Cuda.Buffer UInt32)
    (rowLogProbability : @& Cuda.Buffer Float32) (rows classes : UInt32) : IO Unit :=
  let launch : Cuda.LaunchConfig := {
    grid := { x := rows }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => tokenLogProbabilityF32Kernel.launchOn stream launch logits targets masks
      rowLogProbability rows classes)
    (fun builder => tokenLogProbabilityF32Kernel.addToGraph builder launch logits targets masks
      rowLogProbability rows classes)

/-- Submit one completion-log-probability reduction per fixed-length sequence. -/
def submitReduceSequenceLogProbabilityF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (rowLogProbability sequenceLogProbability : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) : IO Unit := do
  let layout ← match layout.check with
    | .ok layout => pure layout
    | .error message => throw <| IO.userError message
  let launch := elementConfig layout.batchSize
  executor.submit label
    (fun stream => reduceSequenceLogProbabilityF32Kernel.launchOn stream launch
      rowLogProbability sequenceLogProbability layout.batchSize layout.sequenceLength)
    (fun builder => reduceSequenceLogProbabilityF32Kernel.addToGraph builder launch
      rowLogProbability sequenceLogProbability layout.batchSize layout.sequenceLength)

/-- Submit the mean DPO logistic objective and its pairwise sequence coefficients. -/
def submitDPOObjectiveF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (policy reference coefficients loss : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (beta : Float32) : IO Unit := do
  checkDPOGeometry layout beta
  let launch := elementConfig 1
  executor.submit label
    (fun stream => dpoObjectiveF32Kernel.launchOn stream launch policy reference coefficients loss
      (layout.batchSize / 2) beta)
    (fun builder => dpoObjectiveF32Kernel.addToGraph builder launch policy reference coefficients
      loss (layout.batchSize / 2) beta)

/-- Submit the completion-masked DPO logit VJP. -/
def submitDPOGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (logits : @& Cuda.Buffer Float32) (targets masks : @& Cuda.Buffer UInt32)
    (coefficients gradient : @& Cuda.Buffer Float32) (layout : Cuda.Qwen36.SequenceLayout)
    (classes : UInt32) : IO Unit := do
  checkDPOGeometry layout 1
  let launch : Cuda.LaunchConfig := {
    grid := { x := layout.tokens }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => dpoGradientF32Kernel.launchOn stream launch logits targets masks coefficients
      gradient layout.tokens classes layout.sequenceLength)
    (fun builder => dpoGradientF32Kernel.addToGraph builder launch logits targets masks coefficients
      gradient layout.tokens classes layout.sequenceLength)

/-- Submit clipped outcome-GRPO and its per-row cross-entropy coefficients. -/
def submitGRPOObjectiveF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (policy old reference rewards advantages coefficients statistics loss : @& Cuda.Buffer Float32)
    (masks : @& Cuda.Buffer UInt32) (layout : Cuda.Qwen36.SequenceLayout) (groupSize : UInt32)
    (clipEpsilon klBeta advantageEpsilon : Float32) : IO Unit := do
  checkGRPOGeometry layout groupSize clipEpsilon klBeta advantageEpsilon
  let launch := elementConfig 1
  executor.submit label
    (fun stream => grpoObjectiveF32Kernel.launchOn stream launch policy old reference rewards
      advantages coefficients statistics loss masks layout.batchSize layout.sequenceLength groupSize
      clipEpsilon klBeta advantageEpsilon)
    (fun builder => grpoObjectiveF32Kernel.addToGraph builder launch policy old reference rewards
      advantages coefficients statistics loss masks layout.batchSize layout.sequenceLength groupSize
      clipEpsilon klBeta advantageEpsilon)

/-- Submit the completion-masked GRPO logit VJP. -/
def submitGRPOGradientF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (logits : @& Cuda.Buffer Float32) (targets masks : @& Cuda.Buffer UInt32)
    (coefficients gradient : @& Cuda.Buffer Float32) (layout : Cuda.Qwen36.SequenceLayout)
    (classes groupSize : UInt32) : IO Unit := do
  checkGRPOGeometry layout groupSize 0.2 0 1e-6
  let launch : Cuda.LaunchConfig := {
    grid := { x := layout.tokens }
    block := { x := 256 }
    sharedMemoryBytes := 32
    blockArenaBytes := 0
  }
  executor.submit label
    (fun stream => grpoGradientF32Kernel.launchOn stream launch logits targets masks coefficients
      gradient layout.tokens classes)
    (fun builder => grpoGradientF32Kernel.addToGraph builder launch logits targets masks coefficients
      gradient layout.tokens classes)

def submitEmbeddingBackwardF32 (executor : @& Cuda.Qwen36.Executor) (label : String)
    (ids : @& Cuda.Buffer UInt32) (outputGradient weightGradient : @& Cuda.Buffer Float32)
    (tokens width vocabulary : UInt32) : IO Unit :=
  let launch := elementConfig (vocabulary * width)
  executor.submit label
    (fun stream => embeddingBackwardF32Kernel.launchOn stream launch ids outputGradient
      weightGradient tokens width vocabulary)
    (fun builder => embeddingBackwardF32Kernel.addToGraph builder launch ids outputGradient
      weightGradient tokens width vocabulary)

private def MixerForwardF32.outputBuffer : MixerForwardF32 → Cuda.Buffer Float32
  | .deltaNet buffers => buffers.output
  | .attention buffers => buffers.output

private def MixerBackwardF32.inputGradientBuffer : MixerBackwardF32 → Cuda.Buffer Float32
  | .deltaNet buffers => buffers.inputGradient
  | .attention buffers => buffers.inputGradient

private def submitForwardMixerF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& MixerWeightsF32)
    (saved : @& MixerForwardF32) (input : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) : IO Unit :=
  match weights, saved with
  | .deltaNet weights, .deltaNet buffers =>
      DeltaNet.submitBatchedForwardStageF32 executor weights { buffers with input := input } layout
        shape.hidden shape.linearKeyHeads shape.linearValueHeads shape.linearKeyWidth
        shape.linearValueWidth shape.epsilon shape.queryScale
  | .attention weights, .attention buffers =>
      Attention.submitBatchedForwardStageF32 executor weights { buffers with input := input } layout
        shape.hidden shape.attentionQueryHeads shape.attentionKeyValueHeads
        shape.attentionHeadWidth shape.rotaryHalf shape.epsilon shape.attentionScale
  | _, _ =>
      throw <| IO.userError "Qwen3.6 model mixer weight/activation kind mismatch"

private def submitBackwardMixerF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& MixerWeightsF32)
    (forward : @& MixerForwardF32) (backward : @& MixerBackwardF32)
    (input outputGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) : IO Unit :=
  match weights, forward, backward with
  | .deltaNet weights, .deltaNet forward, .deltaNet backward =>
      DeltaNet.submitBatchedBackwardStageF32 executor weights { forward with input }
        { backward with outputGradient } layout shape.hidden shape.linearKeyHeads shape.linearValueHeads
        shape.linearKeyWidth shape.linearValueWidth shape.epsilon shape.queryScale
  | .attention weights, .attention forward, .attention backward =>
      Attention.submitBatchedBackwardStageF32 executor weights { forward with input }
        { backward with outputGradient } layout shape.hidden shape.attentionQueryHeads
        shape.attentionKeyValueHeads
        shape.attentionHeadWidth shape.rotaryHalf shape.attentionScale
  | _, _, _ =>
      throw <| IO.userError "Qwen3.6 model mixer forward/backward kind mismatch"

private def submitForwardLayerF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& LayerWeightsF32)
    (buffers : @& LayerForwardF32) (input : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) : IO Unit := do
  submitForwardMixerF32 executor weights.mixer buffers.mixer input layout shape
  Linear.submitAddF32 executor "Qwen3.6 mixer residual" input buffers.mixer.outputBuffer
    buffers.mixerResidual (shape.tokens * shape.hidden)
  MLP.submitForwardF32 executor buffers.mixerResidual weights.mlp.norm buffers.mlp.normalized
    buffers.mlp.inverseRms buffers.mlp.gate buffers.mlp.up buffers.mlp.activation
    buffers.mlp.output weights.mlp.gate weights.mlp.up weights.mlp.down shape.tokens shape.hidden
    shape.intermediate shape.epsilon
  Linear.submitAddF32 executor "Qwen3.6 MLP residual" buffers.mixerResidual buffers.mlp.output
    buffers.output (shape.tokens * shape.hidden)

private partial def submitForwardLayersF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& Array LayerWeightsF32) (buffers : @& Array LayerForwardF32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32)
    (index : Nat) (input : Cuda.Buffer Float32) :
    IO (Cuda.Buffer Float32) := do
  if index < weights.size then
    match weights[index]?, buffers[index]? with
    | some layerWeights, some layerBuffers =>
        submitForwardLayerF32 executor layerWeights layerBuffers input layout shape
        submitForwardLayersF32 executor weights buffers layout shape (index + 1)
          layerBuffers.output
    | _, _ =>
        throw <| IO.userError "Qwen3.6 model forward layer index mismatch"
  else
    return input

/-- Complete batch-isolated Float32 prefill/logit schedule for an arbitrary hybrid decoder stack. -/
def submitBatchedForwardF32 (executor : @& Cuda.Qwen36.Executor) (weights : @& WeightsF32)
    (buffers : @& ForwardF32) (tokenIds : @& Cuda.Buffer UInt32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) : IO Unit := do
  let layout ← match layout.check with
    | .ok layout => pure layout
    | .error message => throw <| IO.userError message
  unless weights.layers.size == buffers.layers.size do
    throw <| IO.userError "Qwen3.6 model forward layer-count mismatch"
  unless shape.tokens != 0 && shape.vocabulary != 0 do
    throw <| IO.userError "Qwen3.6 model requires positive token and vocabulary counts"
  unless layout.tokens == shape.tokens do
    throw <| IO.userError "Qwen3.6 model layout token product does not match its flat shape"
  match weights.embedding with
  | .f32 embedding =>
      submitEmbeddingGatherF32 executor "Qwen3.6 token embedding" embedding tokenIds
        buffers.embedding shape.tokens shape.hidden
  | .frozenBF16 embedding =>
      submitEmbeddingGatherFrozenBF16F32 executor "Qwen3.6 token embedding" embedding tokenIds
        buffers.embedding shape.tokens shape.hidden
  let decoderOutput ← submitForwardLayersF32 executor weights.layers buffers.layers layout shape 0
    buffers.embedding
  submitRmsNormForwardF32 executor "Qwen3.6 final RMSNorm" decoderOutput weights.finalNorm
    buffers.finalHidden buffers.finalInverseRms shape.tokens shape.hidden shape.epsilon
  Projection.submitForwardF32 executor "Qwen3.6 language-model head" buffers.finalHidden
    weights.lmHead buffers.logits shape.tokens shape.hidden shape.vocabulary

/-- Single-sequence convenience schedule containing all rows. -/
def submitForwardF32 (executor : @& Cuda.Qwen36.Executor) (weights : @& WeightsF32)
    (buffers : @& ForwardF32) (tokenIds : @& Cuda.Buffer UInt32) (shape : @& ShapeF32) :
    IO Unit :=
  submitBatchedForwardF32 executor weights buffers tokenIds (.single shape.tokens) shape

/-- Sequential oracle route for the common full-model forward schedule. -/
def forwardF32 (stream : @& Cuda.Stream) (weights : @& WeightsF32)
    (buffers : @& ForwardF32) (tokenIds : @& Cuda.Buffer UInt32) (shape : @& ShapeF32) :
    IO Unit :=
  submitForwardF32 (.sequential stream) weights buffers tokenIds shape

/-- Batch-isolated forward plus stable mean cross-entropy, saving the logit VJP. -/
def submitBatchedForwardLossF32 (executor : @& Cuda.Qwen36.Executor) (weights : @& WeightsF32)
    (buffers : @& ForwardF32) (tokenIds targets : @& Cuda.Buffer UInt32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) : IO Unit := do
  submitBatchedForwardF32 executor weights buffers tokenIds layout shape
  submitCrossEntropyF32 executor "Qwen3.6 mean cross-entropy" buffers.logits targets
    buffers.logitGradient buffers.rowLoss shape.tokens shape.vocabulary
  submitReduceLossF32 executor "Qwen3.6 loss reduction" buffers.rowLoss buffers.loss shape.tokens

/-- Forward and reduce completion-masked target log probabilities for every sequence. -/
def submitBatchedSequenceLogProbabilityF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& WeightsF32) (buffers : @& ForwardF32)
    (tokenIds targets masks : @& Cuda.Buffer UInt32)
    (sequenceLogProbability : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) : IO Unit := do
  submitBatchedForwardF32 executor weights buffers tokenIds layout shape
  submitTokenLogProbabilityF32 executor "Qwen3.6 masked target log probability" buffers.logits
    targets masks buffers.rowLoss shape.tokens shape.vocabulary
  submitReduceSequenceLogProbabilityF32 executor "Qwen3.6 sequence log-probability reduction"
    buffers.rowLoss sequenceLogProbability layout

/--
Forward the policy, compute mean DPO loss against cached reference scores, and save its logit VJP.
Sequences are adjacent `chosen, rejected` pairs and only rows selected by `masks` contribute.
-/
def submitBatchedDPOForwardF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& WeightsF32) (buffers : @& ForwardF32)
    (tokenIds targets masks : @& Cuda.Buffer UInt32)
    (referenceSequenceLogProbability policySequenceLogProbability sequenceCoefficients :
      @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) (beta : Float32) : IO Unit := do
  submitBatchedSequenceLogProbabilityF32 executor weights buffers tokenIds targets masks
    policySequenceLogProbability layout shape
  submitDPOObjectiveF32 executor "Qwen3.6 mean DPO objective" policySequenceLogProbability
    referenceSequenceLogProbability sequenceCoefficients buffers.loss layout beta
  submitDPOGradientF32 executor "Qwen3.6 DPO logit VJP" buffers.logits targets masks
    sequenceCoefficients buffers.logitGradient layout shape.vocabulary

/--
Forward the current policy, evaluate clipped outcome-GRPO against cached old/reference token
scores, and save its completion-only logit VJP.
-/
def submitBatchedGRPOForwardF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& WeightsF32) (buffers : @& ForwardF32)
    (tokenIds targets masks : @& Cuda.Buffer UInt32)
    (oldTokenLogProbability referenceTokenLogProbability policyTokenLogProbability rewards
      advantages rowCoefficients statistics : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) (groupSize : UInt32)
    (clipEpsilon klBeta advantageEpsilon : Float32) : IO Unit := do
  submitBatchedForwardF32 executor weights buffers tokenIds layout shape
  submitTokenLogProbabilityF32 executor "Qwen3.6 GRPO policy token log probability" buffers.logits
    targets masks policyTokenLogProbability shape.tokens shape.vocabulary
  submitGRPOObjectiveF32 executor "Qwen3.6 clipped outcome-GRPO objective"
    policyTokenLogProbability oldTokenLogProbability referenceTokenLogProbability rewards
    advantages rowCoefficients statistics buffers.loss masks layout groupSize clipEpsilon klBeta
    advantageEpsilon
  submitGRPOGradientF32 executor "Qwen3.6 GRPO logit VJP" buffers.logits targets masks
    rowCoefficients buffers.logitGradient layout shape.vocabulary groupSize

/-- Single-sequence convenience forward/loss schedule containing all rows. -/
def submitForwardLossF32 (executor : @& Cuda.Qwen36.Executor) (weights : @& WeightsF32)
    (buffers : @& ForwardF32) (tokenIds targets : @& Cuda.Buffer UInt32)
    (shape : @& ShapeF32) : IO Unit :=
  submitBatchedForwardLossF32 executor weights buffers tokenIds targets (.single shape.tokens) shape

/-- Sequential oracle route for the common full-model forward/loss schedule. -/
def forwardLossF32 (stream : @& Cuda.Stream) (weights : @& WeightsF32)
    (buffers : @& ForwardF32) (tokenIds targets : @& Cuda.Buffer UInt32)
    (shape : @& ShapeF32) : IO Unit :=
  submitForwardLossF32 (.sequential stream) weights buffers tokenIds targets shape

private def submitBackwardLayerF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& LayerWeightsF32)
    (forward : @& LayerForwardF32) (backward : @& LayerBackwardF32)
    (input outputGradient : @& Cuda.Buffer Float32)
    (layout : Cuda.Qwen36.SequenceLayout) (shape : @& ShapeF32) : IO Unit := do
  MLP.submitBackwardF32 executor forward.mixerResidual weights.mlp.norm forward.mlp.normalized
    forward.mlp.inverseRms forward.mlp.gate forward.mlp.up forward.mlp.activation outputGradient
    backward.mlp.activationGradient
    backward.mlp.gateGradient backward.mlp.upGradient backward.mlp.normGradientGate
    backward.mlp.normGradientUp backward.mlp.normGradient backward.mlp.inputGradient
    backward.mlp.normWeightGradient backward.mlp.gateWeightGradient
    backward.mlp.upWeightGradient backward.mlp.downWeightGradient weights.mlp.gate weights.mlp.up
    weights.mlp.down shape.tokens shape.hidden shape.intermediate
  Linear.submitAddF32 executor "Qwen3.6 MLP residual VJP" outputGradient
    backward.mlp.inputGradient backward.mixerResidualGradient (shape.tokens * shape.hidden)
  submitBackwardMixerF32 executor weights.mixer forward.mixer backward.mixer input
    backward.mixerResidualGradient layout shape
  Linear.submitAddF32 executor "Qwen3.6 mixer residual VJP" backward.mixerResidualGradient
    backward.mixer.inputGradientBuffer backward.inputGradient (shape.tokens * shape.hidden)

private partial def submitBackwardLayersF32 (executor : @& Cuda.Qwen36.Executor)
    (weights : @& Array LayerWeightsF32) (forward : @& Array LayerForwardF32)
    (backward : @& Array LayerBackwardF32) (layout : Cuda.Qwen36.SequenceLayout)
    (shape : @& ShapeF32) (remaining : Nat)
    (modelInput outputGradient : Cuda.Buffer Float32) : IO (Cuda.Buffer Float32) := do
  if remaining == 0 then
    return outputGradient
  else
    let index := remaining - 1
    match weights[index]?, forward[index]?, backward[index]? with
    | some layerWeights, some layerForward, some layerBackward =>
        let layerInput ← if index == 0 then
            pure modelInput
          else
            match forward[index - 1]? with
            | some previous => pure previous.output
            | none => throw <| IO.userError "Qwen3.6 model missing preceding layer activation"
        submitBackwardLayerF32 executor layerWeights layerForward layerBackward layerInput
          outputGradient layout shape
        submitBackwardLayersF32 executor weights forward backward layout shape index modelInput
          layerBackward.inputGradient
    | _, _, _ =>
        throw <| IO.userError "Qwen3.6 model backward layer index mismatch"

/-- Complete batch-isolated Float32 reverse schedule, including deterministic embedding VJP. -/
def submitBatchedBackwardF32 (executor : @& Cuda.Qwen36.Executor) (weights : @& WeightsF32)
    (forward : @& ForwardF32) (backward : @& BackwardF32)
    (tokenIds : @& Cuda.Buffer UInt32) (layout : Cuda.Qwen36.SequenceLayout)
    (shape : @& ShapeF32) : IO Unit := do
  let layout ← match layout.check with
    | .ok layout => pure layout
    | .error message => throw <| IO.userError message
  unless layout.tokens == shape.tokens do
    throw <| IO.userError "Qwen3.6 model layout token product does not match its flat shape"
  unless weights.layers.size == forward.layers.size &&
      weights.layers.size == backward.layers.size do
    throw <| IO.userError "Qwen3.6 model backward layer-count mismatch"
  Projection.submitBackwardF32 executor "Qwen3.6 language-model head" forward.finalHidden
    weights.lmHead forward.logitGradient backward.finalHiddenGradient backward.lmHeadWeightGradient
    shape.tokens shape.hidden shape.vocabulary
  let decoderOutput := match forward.layers.back? with
    | some layer => layer.output
    | none => forward.embedding
  submitRmsNormBackwardInputF32 executor "Qwen3.6 final RMSNorm input VJP" decoderOutput
    weights.finalNorm backward.finalHiddenGradient backward.decoderOutputGradient
    forward.finalInverseRms shape.tokens shape.hidden
  submitRmsNormBackwardWeightF32 executor "Qwen3.6 final RMSNorm weight VJP" decoderOutput
    backward.finalHiddenGradient forward.finalInverseRms backward.finalNormWeightGradient
    shape.tokens shape.hidden
  let embeddingGradient ← submitBackwardLayersF32 executor weights.layers forward.layers
    backward.layers layout shape weights.layers.size forward.embedding backward.decoderOutputGradient
  match weights.embedding with
  | .f32 _ =>
      submitEmbeddingBackwardF32 executor "Qwen3.6 embedding weight VJP" tokenIds embeddingGradient
        backward.embeddingWeightGradient shape.tokens shape.hidden shape.vocabulary
  | .frozenBF16 _ => pure ()

/-- Single-sequence convenience reverse schedule containing all rows. -/
def submitBackwardF32 (executor : @& Cuda.Qwen36.Executor) (weights : @& WeightsF32)
    (forward : @& ForwardF32) (backward : @& BackwardF32)
    (tokenIds : @& Cuda.Buffer UInt32) (shape : @& ShapeF32) : IO Unit :=
  submitBatchedBackwardF32 executor weights forward backward tokenIds (.single shape.tokens) shape

/-- Sequential oracle route for the common full-model reverse schedule. -/
def backwardF32 (stream : @& Cuda.Stream) (weights : @& WeightsF32)
    (forward : @& ForwardF32) (backward : @& BackwardF32)
    (tokenIds : @& Cuda.Buffer UInt32) (shape : @& ShapeF32) : IO Unit :=
  submitBackwardF32 (.sequential stream) weights forward backward tokenIds shape

end Cuda.Qwen36.Model
