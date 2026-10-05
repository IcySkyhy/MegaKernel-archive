/-
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Christian Pehle
-/
import LeanTest
import Qwen36Gpu
import Qwen36CheckpointTests
import Qwen36ArchitectureTests
import Qwen36PayloadTests

private def parseArgs (args : List String) : IO LeanTest.RunConfig := do
  let mut config : LeanTest.RunConfig := {}
  let mut remaining := args
  while _h : !remaining.isEmpty do
    match remaining with
    | "--filter" :: pattern :: rest =>
      config := { config with filter := some pattern }
      remaining := rest
    | "--ignored" :: rest =>
      config := { config with includeIgnored := true }
      remaining := rest
    | "--fail-fast" :: rest =>
      config := { config with failFast := true }
      remaining := rest
    | "--jobs" :: value :: rest | "-j" :: value :: rest =>
      let some jobs := value.toNat?
        | throw <| IO.userError s!"invalid job count: {value}"
      if jobs == 0 then
        throw <| IO.userError "job count must be positive"
      config := { config with jobs }
      remaining := rest
    | option :: _ =>
      throw <| IO.userError s!"unknown or incomplete option: {option}"
    | [] => remaining := []
  return config

unsafe def main (args : List String) : IO UInt32 := do
  let config ← parseArgs args
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let env ← Lean.importModules
    #[{ module := `LeanTest }, { module := `Qwen36Gpu }, { module := `Qwen36CheckpointTests }, { module := `Qwen36ArchitectureTests }, { module := `Qwen36PayloadTests }]
    {}
  LeanTest.runTestsAndExit env {} config
