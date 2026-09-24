#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_2 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %cst_3 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_4 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
    %c-1_i32 = arith.constant -1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
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
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %20 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = arith.addi %19, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %22 = arith.addi %20, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %23 = tt.splat %15 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %24 = arith.addi %23, %18 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %25 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %26 = arith.cmpi slt, %21, %25 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.select %26, %21, %cst_2 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %28 = tt.splat %arg4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %29 = arith.cmpi slt, %24, %28 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %30 = arith.select %29, %24, %cst_3 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %31 = tt.expand_dims %27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %32 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %33 = arith.muli %31, %32 : tensor<128x1xi32, #blocked1>
    %34 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %36 = tt.broadcast %33 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %37 = tt.broadcast %35 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %38 = arith.addi %36, %37 : tensor<128x64xi32, #blocked1>
    %39 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %40 = tt.addptr %39, %38 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %41 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %43 = tt.splat %arg7 : i32 -> tensor<64x1xi32, #blocked>
    %44 = arith.muli %42, %43 : tensor<64x1xi32, #blocked>
    %45 = tt.expand_dims %30 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %46 = tt.broadcast %44 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %47 = tt.broadcast %45 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %48 = arith.addi %46, %47 : tensor<64x128xi32, #blocked>
    %49 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %50 = tt.addptr %49, %48 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %51 = arith.addi %arg5, %c63_i32 : i32
    %52 = arith.divsi %51, %c64_i32 : i32
    %53 = arith.muli %arg7, %c64_i32 : i32
    %54 = tt.splat %53 : i32 -> tensor<64x128xi32, #blocked>
    %55 = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %56 = ttg.local_alloc : () -> !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>
    %57 = arith.cmpi sgt, %52, %c0_i32 : i32
    %58 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %59 = arith.cmpi slt, %35, %58 : tensor<1x64xi32, #blocked1>
    %60 = tt.broadcast %59 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %61 = ttg.memdesc_index %55[%c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %62 = tt.splat %57 : i1 -> tensor<128x64xi1, #blocked1>
    %63 = arith.andi %62, %60 : tensor<128x64xi1, #blocked1>
    %64 = ttg.async_copy_global_to_local %40, %61 mask %63 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %65 = ttg.async_commit_group tokens %64
    %66 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked>
    %67 = arith.cmpi slt, %42, %66 : tensor<64x1xi32, #blocked>
    %68 = tt.broadcast %67 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %69 = ttg.memdesc_index %56[%c0_i32] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %70 = tt.splat %57 : i1 -> tensor<64x128xi1, #blocked>
    %71 = arith.andi %70, %68 : tensor<64x128xi1, #blocked>
    %72 = ttg.async_copy_global_to_local %50, %69 mask %71 other %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    %73 = ttg.async_commit_group tokens %72
    %74 = arith.cmpi sgt, %52, %c1_i32 : i32
    %75 = tt.addptr %40, %cst_4 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %76 = tt.addptr %50, %54 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %77 = arith.subi %arg5, %c64_i32 : i32
    %78 = tt.splat %77 : i32 -> tensor<1x64xi32, #blocked1>
    %79 = arith.cmpi slt, %35, %78 : tensor<1x64xi32, #blocked1>
    %80 = tt.broadcast %79 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %81 = ttg.memdesc_index %55[%c1_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %82 = tt.splat %74 : i1 -> tensor<128x64xi1, #blocked1>
    %83 = arith.andi %82, %80 : tensor<128x64xi1, #blocked1>
    %84 = ttg.async_copy_global_to_local %75, %81 mask %83 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %85 = ttg.async_commit_group tokens %84
    %86 = tt.splat %77 : i32 -> tensor<64x1xi32, #blocked>
    %87 = arith.cmpi slt, %42, %86 : tensor<64x1xi32, #blocked>
    %88 = tt.broadcast %87 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %89 = ttg.memdesc_index %56[%c1_i32] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %90 = tt.splat %74 : i1 -> tensor<64x128xi1, #blocked>
    %91 = arith.andi %90, %88 : tensor<64x128xi1, #blocked>
    %92 = ttg.async_copy_global_to_local %76, %89 mask %91 other %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    %93 = ttg.async_commit_group tokens %92
    %94 = arith.cmpi sgt, %52, %c2_i32 : i32
    %95 = tt.addptr %75, %cst_4 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %96 = tt.addptr %76, %54 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %97 = arith.subi %arg5, %c128_i32 : i32
    %98 = tt.splat %97 : i32 -> tensor<1x64xi32, #blocked1>
    %99 = arith.cmpi slt, %35, %98 : tensor<1x64xi32, #blocked1>
    %100 = tt.broadcast %99 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %101 = ttg.memdesc_index %55[%c2_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %102 = tt.splat %94 : i1 -> tensor<128x64xi1, #blocked1>
    %103 = arith.andi %102, %100 : tensor<128x64xi1, #blocked1>
    %104 = ttg.async_copy_global_to_local %95, %101 mask %103 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %105 = ttg.async_commit_group tokens %104
    %106 = tt.splat %97 : i32 -> tensor<64x1xi32, #blocked>
    %107 = arith.cmpi slt, %42, %106 : tensor<64x1xi32, #blocked>
    %108 = tt.broadcast %107 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %109 = ttg.memdesc_index %56[%c2_i32] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %110 = tt.splat %94 : i1 -> tensor<64x128xi1, #blocked>
    %111 = arith.andi %110, %108 : tensor<64x128xi1, #blocked>
    %112 = ttg.async_copy_global_to_local %96, %109 mask %111 other %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    %113 = ttg.async_commit_group tokens %112
    %114:11 = scf.for %arg9 = %c0_i32 to %52 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %95, %arg12 = %96, %arg13 = %c2_i32, %arg14 = %c-1_i32, %arg15 = %65, %arg16 = %85, %arg17 = %105, %arg18 = %73, %arg19 = %93, %arg20 = %113) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %134 = arith.subi %52, %c3_i32 : i32
      %135 = arith.cmpi slt, %arg9, %134 : i32
      %136 = arith.addi %arg14, %c1_i32 : i32
      %137 = arith.cmpi sge, %136, %c3_i32 : i32
      %138 = arith.select %137, %c0_i32, %136 : i32
      %139 = ttg.async_wait %arg15, %arg18 {num = 4 : i32}
      %140 = ttg.memdesc_index %55[%138] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %141 = ttg.local_load %140 token %139 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %142 = ttg.memdesc_index %56[%138] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %143 = ttg.local_load %142 token %139 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %144 = tt.dot %141, %143, %arg10, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %145 = tt.addptr %arg11, %cst_4 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %146 = tt.addptr %arg12, %54 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %147 = arith.addi %arg13, %c1_i32 : i32
      %148 = arith.cmpi sge, %147, %c3_i32 : i32
      %149 = arith.select %148, %c0_i32, %147 : i32
      %150 = arith.addi %arg9, %c3_i32 : i32
      %151 = arith.muli %150, %c64_i32 : i32
      %152 = arith.subi %arg5, %151 : i32
      %153 = tt.splat %152 : i32 -> tensor<1x64xi32, #blocked1>
      %154 = arith.cmpi slt, %35, %153 : tensor<1x64xi32, #blocked1>
      %155 = tt.broadcast %154 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
      %156 = ttg.memdesc_index %55[%149] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %157 = tt.splat %135 : i1 -> tensor<128x64xi1, #blocked1>
      %158 = arith.andi %157, %155 : tensor<128x64xi1, #blocked1>
      %159 = ttg.async_copy_global_to_local %145, %156 mask %158 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
      %160 = ttg.async_commit_group tokens %159
      %161 = tt.splat %152 : i32 -> tensor<64x1xi32, #blocked>
      %162 = arith.cmpi slt, %42, %161 : tensor<64x1xi32, #blocked>
      %163 = tt.broadcast %162 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %164 = ttg.memdesc_index %56[%149] : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %165 = tt.splat %135 : i1 -> tensor<64x128xi1, #blocked>
      %166 = arith.andi %165, %163 : tensor<64x128xi1, #blocked>
      %167 = ttg.async_copy_global_to_local %146, %164 mask %166 other %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
      %168 = ttg.async_commit_group tokens %167
      scf.yield %144, %145, %146, %149, %138, %arg16, %arg17, %160, %arg19, %arg20, %168 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %115 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %56 : !ttg.memdesc<3x64x128xf16, #shared, #smem, mutable>
    ttg.local_dealloc %55 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %116 = arith.truncf %114#0 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %117 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %118 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
    %119 = arith.muli %118, %117 : tensor<128x1xi32, #blocked>
    %120 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %121 = tt.addptr %120, %119 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %122 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %123 = tt.broadcast %121 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %124 = tt.broadcast %122 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %125 = tt.addptr %123, %124 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %126 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked>
    %127 = arith.cmpi slt, %117, %126 : tensor<128x1xi32, #blocked>
    %128 = tt.splat %arg4 : i32 -> tensor<1x128xi32, #blocked>
    %129 = arith.cmpi slt, %122, %128 : tensor<1x128xi32, #blocked>
    %130 = tt.broadcast %127 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %131 = tt.broadcast %129 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %132 = arith.andi %130, %131 : tensor<128x128xi1, #blocked>
    %133 = ttg.convert_layout %116 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked>
    tt.store %125, %133, %132 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(tritongpu-combine-tensor-select-and-if,tritongpu-allocate-warp-groups,convert-scf-to-cf,gluon-inline,allocate-shared-memory-nv{compute-capability=80 ptx-version=87},triton-tensor-memory-allocation,triton-nvidia-check-matmul-two-cta,tritongpu-global-scratch-memory-allocation,triton-nvidia-gpu-proxy-fence-insertion{compute-capability=80},convert-triton-gpu-to-llvm{compute-capability=80 ptx-version=87},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,convert-nv-gpu-to-llvm,convert-warp-specialize-to-llvm,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,symbol-dce,convert-nvvm-to-llvm,enable-line-info))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}
