/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Cuda

/-! Host-side gates for the checked mixed-adapter LoRA descriptor. -/

namespace Qwen36LoRAContract

open Cuda.Qwen36.LoRA

private def valid : Descriptor := {
  batchSize := 4
  sequenceLength := 2
  adapterCount := 2
  rank := 16
  inputFeatures := 16
  outputFeatures := 16
  alpha := 16
  adapterIds := #[0, 1, 0, 1]
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
def acceptsInterleavedDescriptor : IO Unit :=
  requireAccepted "interleaved adapter descriptor" valid

@[test]
def acceptsNoAdapterSentinel : IO Unit :=
  requireAccepted "no-adapter sentinel" { valid with adapterIds := #[0, noAdapter, 1, noAdapter] }

@[test]
def rejectsNonTiledRank : IO Unit :=
  requireRejected "non-tiled rank" { valid with rank := 15 }

@[test]
def rejectsAdapterIdCountMismatch : IO Unit :=
  requireRejected "adapter-id count mismatch" { valid with adapterIds := #[0, 1] }

@[test]
def rejectsOutOfRangeAdapter : IO Unit :=
  requireRejected "out-of-range adapter id" { valid with adapterIds := #[0, 2, 0, 1] }

end Qwen36LoRAContract
