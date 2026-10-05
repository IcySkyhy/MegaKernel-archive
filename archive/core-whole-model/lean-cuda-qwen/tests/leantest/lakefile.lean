import Lake

open Lake DSL

package qwen36_cuda_tests

require LeanTest from git "https://github.com/cpehle/lean_test.git" @ "fbfea9210f4f2eb1fc7c3b9e3c86f57c5dfa6e28"

lean_lib Qwen36Model where
  srcDir := "../../examples/qwen36_megakernel"
  roots := #[`Qwen36Checkpoint, `Qwen36Architecture]

lean_lib Qwen36GpuTests where
  roots := #[`Qwen36Gpu, `Qwen36CheckpointTests, `Qwen36ArchitectureTests, `Qwen36PayloadTests]

@[test_driver]
lean_exe test where
  root := `TestDriver
  supportInterpreter := true
