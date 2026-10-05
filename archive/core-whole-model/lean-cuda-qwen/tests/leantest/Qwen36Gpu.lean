/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest

namespace Qwen36Gpu

private def repositoryRoot : IO System.FilePath := do
  if let some configured ← IO.getEnv "LEAN_CUDA_QWEN_ROOT" then
    IO.FS.realPath configured
  else
    let cwd ← IO.currentDir
    IO.FS.realPath (cwd / "../..")

private def runExample (script expected : String) : IO Unit := do
  let root ← repositoryRoot
  let directory := root / "examples" / "qwen36_megakernel"
  let backend ←
    if let some configured ← IO.getEnv "LEAN_CUDA_ROOT" then
      IO.FS.realPath configured
    else
      IO.FS.realPath (root / ".lake/lean4-cuda-backend")
  let runner := directory / script
  LeanTest.assertTrue (← runner.pathExists) s!"missing Qwen3.6 runner: {runner}"
  let result ← IO.Process.output {
    cmd := "/usr/bin/env"
    args := #[
      "-u", "LD_LIBRARY_PATH",
      "-u", "LEAN_PATH",
      "-u", "LEAN_SRC_PATH",
      s!"LEAN_CUDA_ROOT={backend}", runner.toString
    ]
    cwd := some directory
  }
  LeanTest.assertEqual result.exitCode 0 (some <|
    s!"{script} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
  LeanTest.assertTrue (result.stdout.containsSubstr expected) <|
    s!"{script} did not report its correctness gate\nstdout:\n{result.stdout}"

@[test]
def recurrentDecodeMegakernel : IO Unit :=
  runExample "run.sh" "Lean Qwen3.6 recurrent decode megakernel ok"

@[test]
def recurrentTrainingVjpMegakernel : IO Unit :=
  runExample "run-training.sh" "Lean Qwen3.6 recurrent forward/backward megakernel VJP ok"

@[test]
def linearAttentionLayerCoreMegakernel : IO Unit :=
  runExample "run-linear-layer.sh"
    "Lean Qwen3.6 causal-conv/recurrent/gated-RMSNorm megakernel ok"

@[test]
def projectionMegakernel : IO Unit :=
  runExample "run-projection.sh" "Lean Qwen3.6 exact-shape decode projection megakernel ok"

@[test]
def mlpMegakernel : IO Unit :=
  runExample "run-mlp.sh" "Lean Qwen3.6 exact-shape RMSNorm/SwiGLU/residual megakernel ok"

@[test]
def fullAttentionMegakernel : IO Unit :=
  runExample "run-attention.sh" "Lean Qwen3.6 gated full-attention KV-cache megakernel ok"

@[test]
def tokenBoundaryMegakernel : IO Unit :=
  runExample "run-token.sh" "Lean Qwen3.6 embedding/RMSNorm/LM-head/loss/sampling megakernel ok"

@[test_ignore]
def realCheckpointInferenceMegakernel : IO Unit := do
  let expected := if (← IO.getEnv "QWEN36_COMPILE_ONLY") == some "1" then
      "Lean Qwen3.6 real-checkpoint 64-layer inference megakernel compiled"
    else
      "Lean Qwen3.6 real-checkpoint 64-layer inference megakernel ok"
  runExample "run-model.sh" expected

end Qwen36Gpu
