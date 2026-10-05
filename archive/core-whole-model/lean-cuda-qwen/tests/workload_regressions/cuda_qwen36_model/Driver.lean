/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Lean.Util.Path
import Batch
import Kernel

/-! LeanTest discovery driver for the complete Qwen3.6 numerical model suite. -/

unsafe def main : IO UInt32 := do
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let environment ← Lean.importModules #[
    { module := `LeanTest },
    { module := `Batch },
    { module := `Kernel }
  ] {}
  let filter ← IO.getEnv "QWEN36_MODEL_TEST_FILTER"
  LeanTest.runTestsAndExit environment {} { filter, jobs := 1 }
