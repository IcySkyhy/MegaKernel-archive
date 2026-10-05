/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import LeanCudaQwen.Qwen36

/-!
# Qwen3.8 HF config provider

Applies `hfconfig_type_provider` to the vendored Qwen3.8-27B `config.json` (RoPE fields nested
under `rope_parameters`) and to the tiny gate fixture `TinyConfig.json` (flat `rope_theta`), then
checks the emitted scalar constants, derived dimensions, `layerTypes` table, and
`isFullAttention` against the shapes in `docs/QWEN36_MEGAKERNEL.md`, cross-checking the built-in
`Config.qwen38_27B` and `Config.tiny`. `Malformed.lean` and `MalformedMissing.lean` check that
invalid configs fail elaboration.
-/

open Lean.Cuda.Qwen36

hfconfig_type_provider "config.json" as Real
hfconfig_type_provider "TinyConfig.json" as Tiny

-- 27B scalar fields.
example : Real.hiddenSize = 5120 := rfl
example : Real.numHiddenLayers = 64 := rfl
example : Real.linearNumKeyHeads = 16 := rfl
example : Real.linearKeyHeadDim = 128 := rfl
example : Real.linearNumValueHeads = 48 := rfl
example : Real.linearValueHeadDim = 128 := rfl
example : Real.linearConvKernelDim = 4 := rfl
example : Real.numAttentionHeads = 24 := rfl
example : Real.numKeyValueHeads = 4 := rfl
example : Real.headDim = 256 := rfl
example : Real.intermediateSize = 17408 := rfl
example : Real.vocabSize = 248320 := rfl
example : Real.tieWordEmbeddings = false := rfl

-- 27B derived dimensions.
example : Real.keyDim = 2048 := rfl
example : Real.valueDim = 6144 := rfl
example : Real.convDim = 10240 := rfl
example : Real.attnQDim = 6144 := rfl
example : Real.qProjDim = 12288 := rfl
example : Real.kvProjDim = 1024 := rfl
example : Real.gqaRatio = 6 := rfl
example : Real.linearRepeatFactor = 3 := rfl
example : Real.rotaryDim = 64 := rfl

-- 27B layer table: full attention iff `i % 4 == 3`.
example : Real.layerTypes.size = 64 := rfl
example : Real.isFullAttention 0 = false := rfl
example : Real.isFullAttention 2 = false := rfl
example : Real.isFullAttention 3 = true := rfl
example : Real.isFullAttention 62 = false := rfl
example : Real.isFullAttention 63 = true := rfl
example : Real.isFullAttention 64 = false := rfl
example : Real.config.check.isOk = true := rfl
example : Real.config.keyDim = Real.keyDim := rfl

-- Tiny scalar fields and derived dimensions.
example : Tiny.hiddenSize = 256 := rfl
example : Tiny.numHiddenLayers = 4 := rfl
example : Tiny.linearNumKeyHeads = 2 := rfl
example : Tiny.linearKeyHeadDim = 64 := rfl
example : Tiny.linearNumValueHeads = 4 := rfl
example : Tiny.linearValueHeadDim = 64 := rfl
example : Tiny.numAttentionHeads = 4 := rfl
example : Tiny.numKeyValueHeads = 2 := rfl
example : Tiny.headDim = 64 := rfl
example : Tiny.intermediateSize = 512 := rfl
example : Tiny.vocabSize = 512 := rfl
example : Tiny.keyDim = 128 := rfl
example : Tiny.valueDim = 256 := rfl
example : Tiny.convDim = 512 := rfl
example : Tiny.qProjDim = 512 := rfl
example : Tiny.kvProjDim = 128 := rfl
example : Tiny.gqaRatio = 2 := rfl
example : Tiny.linearRepeatFactor = 2 := rfl
example : Tiny.rotaryDim = 16 := rfl
example : Tiny.layerTypes.size = 4 := rfl
example : Tiny.isFullAttention 0 = false := rfl
example : Tiny.isFullAttention 2 = false := rfl
example : Tiny.isFullAttention 3 = true := rfl
example : Tiny.isFullAttention 4 = false := rfl
example : Tiny.config.check.isOk = true := rfl

-- The built-in constants agree with the vendored configs.
example : Real.hiddenSize = Config.qwen38_27B.hiddenSize := rfl
example : Real.layerTypes = Config.qwen38_27B.layerTypes := rfl
example : Real.config = Config.qwen38_27B := rfl
example : Tiny.config = Config.tiny := rfl
example : Config.tiny.check.isOk = true := rfl
example : Config.qwen38_27B.check.isOk = true := rfl

-- Config.check rejects the divisibility violations exercised by the malformed fixtures.
example : (Config.qwen38_27B.check).isOk = true := rfl
example : ({ Config.tiny with numKeyValueHeads := 3 : Config}).check.isOk = false := rfl
example : ({ Config.tiny with linearNumValueHeads := 5 : Config}).check.isOk = false := rfl
example : ({ Config.tiny with headDim := 24 : Config}).check.isOk = false := rfl
example : ({ Config.tiny with numHiddenLayers := 5 : Config}).check.isOk = false := rfl
example : ({ Config.tiny with partialRotaryFactor := 0.3 : Config}).check.isOk = false := rfl
example : ({ Config.tiny with vocabSize := 0 : Config}).check.isOk = false := rfl

@[test]
def emittedConfigurationMatchesBuiltins : IO Unit := do
  -- Float fields are checked at runtime; the rest is proved by `rfl` above.
  LeanTest.assertTrue (Real.rmsNormEps == 1e-6) "Real.rmsNormEps"
  LeanTest.assertTrue (Real.ropeTheta == 1e7) "Real.ropeTheta"
  LeanTest.assertTrue (Real.partialRotaryFactor == 0.25) "Real.partialRotaryFactor"
  LeanTest.assertTrue (Tiny.rmsNormEps == Config.tiny.rmsNormEps) "Tiny.rmsNormEps"
  LeanTest.assertTrue (Tiny.ropeTheta == Config.tiny.ropeTheta) "Tiny.ropeTheta"
  LeanTest.assertTrue (Tiny.partialRotaryFactor == Config.tiny.partialRotaryFactor)
    "Tiny.partialRotaryFactor"
  LeanTest.assertTrue (Real.config.rmsNormEps == Config.qwen38_27B.rmsNormEps)
    "Real.config.rmsNormEps"
  LeanTest.assertTrue (Real.config.ropeTheta == Config.qwen38_27B.ropeTheta)
    "Real.config.ropeTheta"
  LeanTest.assertTrue (Real.isFullAttention 3 && !Real.isFullAttention 0)
    "Real.isFullAttention"
