/- Copyright (c) 2026 Ranvier Systems. Apache-2.0. -/
import Lean
import Lean.Compiler.LCNF.EmitCUDA
import Lean.Compiler.LCNF.CudaExternBody

open Lean

/-- Emit a compiler-provided device module from the binary's installed olean/IR files. -/
unsafe def main (args : List String) : IO Unit := do
  let [moduleName, output] := args
    | throw <| IO.userError "usage: EmitNightlyModule.lean MODULE OUTPUT.cu"
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let name := moduleName.toName
  let imported ← importModules #[{ module := name }] {} (loadExts := true)
  let mut env := imported.setMainModule name
  -- MapDeclarationExtension keeps imported entries lazy; promote only this module's
  -- exports into the local emission view, preserving their original names and ABI.
  for (declName, _) in imported.constants.toList do
    if let some entry := Compiler.Cuda.getDeviceExport? imported declName then
      if entry.moduleName == name then
        env := Compiler.Cuda.deviceExportExt.addEntry (asyncDecl := declName) env (declName, entry)
  let (source, _) ← (do
    for (declName, _) in Compiler.Cuda.deviceExportExt.getState (← getEnv) |>.toArray do
      discard <| Compiler.LCNF.compileCUDAExternBody declName
    Compiler.LCNF.emitCUDA name).toIO
    { fileName := "<nightly-device-module>", fileMap := default }
    { env }
  IO.FS.writeFile output source
