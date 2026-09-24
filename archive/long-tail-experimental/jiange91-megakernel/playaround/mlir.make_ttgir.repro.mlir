module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c127_i32 = arith.constant 127 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<64> : tensor<128x64xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_3 = arith.constant dense<0> : tensor<128xi32>
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c128_i32 : i32
    %15 = arith.muli %13, %c128_i32 : i32
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %17 = tt.splat %14 : i32 -> tensor<128xi32>
    %18 = arith.addi %17, %16 : tensor<128xi32>
    %19 = tt.splat %15 : i32 -> tensor<128xi32>
    %20 = arith.addi %19, %16 : tensor<128xi32>
    %21 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %22 = arith.cmpi slt, %18, %21 : tensor<128xi32>
    %23 = arith.select %22, %18, %cst_3 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
    %24 = tt.splat %arg4 : i32 -> tensor<128xi32>
    %25 = arith.cmpi slt, %20, %24 : tensor<128xi32>
    %26 = arith.select %25, %20, %cst_3 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
    %27 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %28 = tt.expand_dims %23 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %29 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %30 = arith.muli %28, %29 : tensor<128x1xi32>
    %31 = tt.expand_dims %27 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %32 = tt.broadcast %30 : tensor<128x1xi32> -> tensor<128x64xi32>
    %33 = tt.broadcast %31 : tensor<1x64xi32> -> tensor<128x64xi32>
    %34 = arith.addi %32, %33 : tensor<128x64xi32>
    %35 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %36 = tt.addptr %35, %34 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    %37 = tt.expand_dims %27 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %38 = tt.splat %arg7 : i32 -> tensor<64x1xi32>
    %39 = arith.muli %37, %38 : tensor<64x1xi32>
    %40 = tt.expand_dims %26 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %41 = tt.broadcast %39 : tensor<64x1xi32> -> tensor<64x128xi32>
    %42 = tt.broadcast %40 : tensor<1x128xi32> -> tensor<64x128xi32>
    %43 = arith.addi %41, %42 : tensor<64x128xi32>
    %44 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
    %45 = tt.addptr %44, %43 : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
    %46 = arith.addi %arg5, %c63_i32 : i32
    %47 = arith.divsi %46, %c64_i32 : i32
    %48:3 = scf.for %arg9 = %c0_i32 to %47 step %c1_i32 iter_args(%arg10 = %36, %arg11 = %45, %arg12 = %cst) -> (tensor<128x64x!tt.ptr<f16>>, tensor<64x128x!tt.ptr<f16>>, tensor<128x128xf32>)  : i32 {
      %66 = arith.muli %arg9, %c64_i32 : i32
      %67 = arith.subi %arg5, %66 : i32
      %68 = tt.splat %67 : i32 -> tensor<1x64xi32>
      %69 = arith.cmpi slt, %31, %68 : tensor<1x64xi32>
      %70 = tt.broadcast %69 : tensor<1x64xi1> -> tensor<128x64xi1>
      %71 = tt.load %arg10, %70, %cst_1 : tensor<128x64x!tt.ptr<f16>>
      %72 = tt.splat %67 : i32 -> tensor<64x1xi32>
      %73 = arith.cmpi slt, %37, %72 : tensor<64x1xi32>
      %74 = tt.broadcast %73 : tensor<64x1xi1> -> tensor<64x128xi1>
      %75 = tt.load %arg11, %74, %cst_0 : tensor<64x128x!tt.ptr<f16>>
      %76 = tt.dot %71, %75, %arg12, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      %77 = tt.addptr %arg10, %cst_2 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %78 = arith.muli %arg7, %c64_i32 : i32
      %79 = tt.splat %78 : i32 -> tensor<64x128xi32>
      %80 = tt.addptr %arg11, %79 : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
      scf.yield %77, %80, %76 : tensor<128x64x!tt.ptr<f16>>, tensor<64x128x!tt.ptr<f16>>, tensor<128x128xf32>
    }
    %49 = arith.truncf %48#2 : tensor<128x128xf32> to tensor<128x128xf16>
    %50 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %51 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
    %52 = arith.muli %51, %50 : tensor<128x1xi32>
    %53 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %54 = tt.addptr %53, %52 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
    %55 = tt.expand_dims %20 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %56 = tt.broadcast %54 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
    %57 = tt.broadcast %55 : tensor<1x128xi32> -> tensor<128x128xi32>
    %58 = tt.addptr %56, %57 : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
    %59 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %60 = arith.cmpi slt, %50, %59 : tensor<128x1xi32>
    %61 = tt.splat %arg4 : i32 -> tensor<1x128xi32>
    %62 = arith.cmpi slt, %55, %61 : tensor<1x128xi32>
    %63 = tt.broadcast %60 : tensor<128x1xi1> -> tensor<128x128xi1>
    %64 = tt.broadcast %62 : tensor<1x128xi1> -> tensor<128x128xi1>
    %65 = arith.andi %63, %64 : tensor<128x128xi1>
    tt.store %58, %49, %65 : tensor<128x128x!tt.ptr<f16>>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(convert-triton-to-tritongpu{enable-source-remat=false num-ctas=1 num-warps=8 target=cuda:80 threads-per-warp=32},tritongpu-coalesce,tritongpu-F32DotTC{emu-tf32=true},triton-nvidia-gpu-plan-cta,tritongpu-remove-layout-conversions,tritongpu-optimize-thread-locality,tritongpu-accelerate-matmul,tritongpu-remove-layout-conversions,tritongpu-optimize-dot-operands{hoist-layout-conversion=true},triton-nvidia-optimize-descriptor-encoding,triton-loop-aware-cse,tritongpu-fuse-nested-loops,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},triton-licm,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},tritongpu-combine-tensor-select-and-if,nvgpu-warp-specialization{dump-intermediate-steps=true num-stages=4},tritongpu-assign-latencies{num-stages=4},tritongpu-schedule-loops,tritongpu-pipeline{dump-intermediate-steps=true num-stages=4},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},triton-loop-aware-cse,tritongpu-prefetch,tritongpu-optimize-dot-operands{hoist-layout-conversion=true},tritongpu-coalesce-async-copy,triton-nvidia-optimize-tmem-layouts,tritongpu-remove-layout-conversions,triton-nvidia-interleave-tmem,tritongpu-reduce-data-duplication,tritongpu-reorder-instructions,triton-loop-aware-cse,symbol-dce,triton-nvidia-gpu-fence-insertion{compute-capability=80},triton-nvidia-mma-lowering,sccp,cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}
