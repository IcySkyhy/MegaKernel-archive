/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-! LeanTest gates for batch isolation and mixed-adapter training-plan validation. -/

namespace Qwen36TrainingContract

open Cuda.Qwen36
open Cuda.Qwen36.Training

private def shape : Model.ShapeF32 := {
  tokens := 8
  hidden := 256
  intermediate := 512
  vocabulary := 512
  linearKeyHeads := 2
  linearValueHeads := 4
  linearKeyWidth := 64
  linearValueWidth := 64
  attentionQueryHeads := 4
  attentionKeyValueHeads := 2
  attentionHeadWidth := 64
  rotaryHalf := 8
  epsilon := 1e-6
  queryScale := 0.125
  attentionScale := 0.125
}

private def fullModel : Descriptor := {
  steps := 4
  batchSize := 1
  sequenceLength := 8
  model := shape
  profile := .fullModel
}

private def mixedLoRA : Descriptor := {
  steps := 4
  batchSize := 4
  sequenceLength := 2
  model := shape
  profile := .lora {
    adapterCount := 2
    rank := 16
    alpha := 16
    adapterIds := #[0, 1, 0, 1]
  }
}

private def requireAccepted (label : String) (descriptor : Descriptor) : IO Unit :=
  match descriptor.check with
  | .ok _ => pure ()
  | .error error => LeanTest.fail s!"{label} was rejected: {error}"

private def requireRejected (label : String) (descriptor : Descriptor) : IO Unit :=
  match descriptor.check with
  | .error _ => pure ()
  | .ok _ => LeanTest.fail s!"{label} was unexpectedly accepted"

@[test]
def acceptsFullModelSequence : IO Unit :=
  requireAccepted "full-model batch" fullModel

@[test]
def acceptsInterleavedMixedLoRA : IO Unit :=
  requireAccepted "mixed-adapter batch" mixedLoRA

@[test]
def acceptsNoAdapterRows : IO Unit :=
  requireAccepted "mixed no-adapter batch" {
    mixedLoRA with
    profile := .lora {
      adapterCount := 2
      rank := 16
      alpha := 16
      adapterIds := #[0, LoRA.noAdapter, 1, LoRA.noAdapter]
    }
  }

@[test]
def rejectsTokenProductMismatch : IO Unit :=
  requireRejected "token-product mismatch" { mixedLoRA with sequenceLength := 3 }

@[test]
def rejectsZeroSteps : IO Unit :=
  requireRejected "zero recurrent steps" { fullModel with steps := 0 }

@[test]
def rejectsNonTiledRank : IO Unit :=
  requireRejected "non-tiled LoRA rank" {
    mixedLoRA with
    profile := .lora {
      adapterCount := 2
      rank := 15
      alpha := 16
      adapterIds := #[0, 1, 0, 1]
    }
  }

@[test]
def rejectsOutOfRangeAdapter : IO Unit :=
  requireRejected "out-of-range adapter" {
    mixedLoRA with
    profile := .lora {
      adapterCount := 2
      rank := 16
      alpha := 16
      adapterIds := #[0, 2, 0, 1]
    }
  }

end Qwen36TrainingContract
