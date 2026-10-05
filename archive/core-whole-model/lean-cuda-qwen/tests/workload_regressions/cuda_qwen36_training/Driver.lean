/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Lean.Util.Path
import Contract
import Kernel

/-! LeanTest discovery driver for the Qwen3.6 resident-training suite. -/

unsafe def main : IO UInt32 := do
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let environment ← Lean.importModules #[
    { module := `LeanTest },
    { module := `Contract },
    { module := `Kernel }
  ] {}
  LeanTest.runTestsAndExit environment {} { jobs := 1 }
