/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Lean.Util.Path
import Test

/-! LeanTest discovery driver for Qwen3.8 typed configuration providers. -/

unsafe def main : IO UInt32 := do
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let environment ← Lean.importModules #[
    { module := `LeanTest },
    { module := `Test }
  ] {}
  LeanTest.runTestsAndExit environment {} { jobs := 1 }
