/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Lean.Util.Path
import Kernel

/-! LeanTest discovery driver for persistent Qwen3.8 inference. -/

unsafe def main : IO UInt32 := do
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let environment ← Lean.importModules #[
    { module := `LeanTest },
    { module := `Kernel }
  ] {}
  let includeRealCheckpoint := (← IO.getEnv "QWEN_MODEL_DIR").isSome
  LeanTest.runTestsAndExit environment {} {
    jobs := 1
    includeIgnored := includeRealCheckpoint
  }
