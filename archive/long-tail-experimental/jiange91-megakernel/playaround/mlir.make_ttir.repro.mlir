module {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = tt.call @"triton.language.standard.cdiv__i32__(1,)cconstexpr_128_"(%arg3) : (i32) -> i32
    %2 = tt.call @"triton.language.standard.cdiv__i32__(1,)cconstexpr_128_"(%arg4) : (i32) -> i32
    %c8_i32 = arith.constant 8 : i32
    %c8_i32_0 = arith.constant 8 : i32
    %3 = arith.extsi %c8_i32_0 : i32 to i64
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.muli %3, %4 : i64
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %6 = arith.cmpi sle, %5, %c2147483647_i64 : i64
    %7 = arith.cmpi sge, %5, %c-2147483648_i64 : i64
    %8 = arith.andi %6, %7 : i1
    %9 = arith.muli %c8_i32_0, %2 : i32
    %10 = arith.divsi %0, %9 : i32
    %c8_i32_1 = arith.constant 8 : i32
    %c8_i32_2 = arith.constant 8 : i32
    %11 = arith.extsi %10 : i32 to i64
    %12 = arith.extsi %c8_i32_2 : i32 to i64
    %13 = arith.muli %11, %12 : i64
    %c2147483647_i64_3 = arith.constant 2147483647 : i64
    %c-2147483648_i64_4 = arith.constant -2147483648 : i64
    %14 = arith.cmpi sle, %13, %c2147483647_i64_3 : i64
    %15 = arith.cmpi sge, %13, %c-2147483648_i64_4 : i64
    %16 = arith.andi %14, %15 : i1
    %17 = arith.muli %10, %c8_i32_2 : i32
    %18 = arith.extsi %1 : i32 to i64
    %19 = arith.extsi %17 : i32 to i64
    %20 = arith.subi %18, %19 : i64
    %c2147483647_i64_5 = arith.constant 2147483647 : i64
    %c-2147483648_i64_6 = arith.constant -2147483648 : i64
    %21 = arith.cmpi sle, %20, %c2147483647_i64_5 : i64
    %22 = arith.cmpi sge, %20, %c-2147483648_i64_6 : i64
    %23 = arith.andi %21, %22 : i1
    %24 = arith.subi %1, %17 : i32
    %c8_i32_7 = arith.constant 8 : i32
    %25 = arith.minsi %24, %c8_i32_7 : i32
    %26 = arith.remsi %0, %25 : i32
    %27 = arith.extsi %17 : i32 to i64
    %28 = arith.extsi %26 : i32 to i64
    %29 = arith.addi %27, %28 : i64
    %c2147483647_i64_8 = arith.constant 2147483647 : i64
    %c-2147483648_i64_9 = arith.constant -2147483648 : i64
    %30 = arith.cmpi sle, %29, %c2147483647_i64_8 : i64
    %31 = arith.cmpi sge, %29, %c-2147483648_i64_9 : i64
    %32 = arith.andi %30, %31 : i1
    %33 = arith.addi %17, %26 : i32
    %34 = arith.remsi %0, %9 : i32
    %35 = arith.divsi %34, %25 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i32_10 = arith.constant 128 : i32
    %36 = arith.extsi %33 : i32 to i64
    %37 = arith.extsi %c128_i32_10 : i32 to i64
    %38 = arith.muli %36, %37 : i64
    %c2147483647_i64_11 = arith.constant 2147483647 : i64
    %c-2147483648_i64_12 = arith.constant -2147483648 : i64
    %39 = arith.cmpi sle, %38, %c2147483647_i64_11 : i64
    %40 = arith.cmpi sge, %38, %c-2147483648_i64_12 : i64
    %41 = arith.andi %39, %40 : i1
    %42 = arith.muli %33, %c128_i32_10 : i32
    %c128_i32_13 = arith.constant 128 : i32
    %c128_i32_14 = arith.constant 128 : i32
    %43 = arith.extsi %35 : i32 to i64
    %44 = arith.extsi %c128_i32_14 : i32 to i64
    %45 = arith.muli %43, %44 : i64
    %c2147483647_i64_15 = arith.constant 2147483647 : i64
    %c-2147483648_i64_16 = arith.constant -2147483648 : i64
    %46 = arith.cmpi sle, %45, %c2147483647_i64_15 : i64
    %47 = arith.cmpi sge, %45, %c-2147483648_i64_16 : i64
    %48 = arith.andi %46, %47 : i1
    %49 = arith.muli %35, %c128_i32_14 : i32
    %50 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %51 = tt.splat %42 : i32 -> tensor<128xi32>
    %52 = arith.extsi %51 : tensor<128xi32> to tensor<128xi64>
    %53 = arith.extsi %50 : tensor<128xi32> to tensor<128xi64>
    %54 = arith.addi %52, %53 : tensor<128xi64>
    %c2147483647_i64_17 = arith.constant 2147483647 : i64
    %c-2147483648_i64_18 = arith.constant -2147483648 : i64
    %cst = arith.constant dense<2147483647> : tensor<128xi64>
    %55 = arith.cmpi sle, %54, %cst : tensor<128xi64>
    %cst_19 = arith.constant dense<-2147483648> : tensor<128xi64>
    %56 = arith.cmpi sge, %54, %cst_19 : tensor<128xi64>
    %57 = arith.andi %55, %56 : tensor<128xi1>
    %58 = arith.addi %51, %50 : tensor<128xi32>
    %59 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %60 = tt.splat %49 : i32 -> tensor<128xi32>
    %61 = arith.extsi %60 : tensor<128xi32> to tensor<128xi64>
    %62 = arith.extsi %59 : tensor<128xi32> to tensor<128xi64>
    %63 = arith.addi %61, %62 : tensor<128xi64>
    %c2147483647_i64_20 = arith.constant 2147483647 : i64
    %c-2147483648_i64_21 = arith.constant -2147483648 : i64
    %cst_22 = arith.constant dense<2147483647> : tensor<128xi64>
    %64 = arith.cmpi sle, %63, %cst_22 : tensor<128xi64>
    %cst_23 = arith.constant dense<-2147483648> : tensor<128xi64>
    %65 = arith.cmpi sge, %63, %cst_23 : tensor<128xi64>
    %66 = arith.andi %64, %65 : tensor<128xi1>
    %67 = arith.addi %60, %59 : tensor<128xi32>
    %68 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %69 = arith.cmpi slt, %58, %68 : tensor<128xi32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_24 = arith.constant 0 : i32
    %cst_25 = arith.constant dense<0> : tensor<128xi32>
    %70 = arith.select %69, %58, %cst_25 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
    %71 = tt.splat %arg4 : i32 -> tensor<128xi32>
    %72 = arith.cmpi slt, %67, %71 : tensor<128xi32>
    %c0_i32_26 = arith.constant 0 : i32
    %c0_i32_27 = arith.constant 0 : i32
    %cst_28 = arith.constant dense<0> : tensor<128xi32>
    %73 = arith.select %72, %67, %cst_28 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
    %74 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %75 = tt.expand_dims %70 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %76 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %77 = arith.extsi %75 : tensor<128x1xi32> to tensor<128x1xi64>
    %78 = arith.extsi %76 : tensor<128x1xi32> to tensor<128x1xi64>
    %79 = arith.muli %77, %78 : tensor<128x1xi64>
    %c2147483647_i64_29 = arith.constant 2147483647 : i64
    %c-2147483648_i64_30 = arith.constant -2147483648 : i64
    %cst_31 = arith.constant dense<2147483647> : tensor<128x1xi64>
    %80 = arith.cmpi sle, %79, %cst_31 : tensor<128x1xi64>
    %cst_32 = arith.constant dense<-2147483648> : tensor<128x1xi64>
    %81 = arith.cmpi sge, %79, %cst_32 : tensor<128x1xi64>
    %82 = arith.andi %80, %81 : tensor<128x1xi1>
    %83 = arith.muli %75, %76 : tensor<128x1xi32>
    %84 = tt.expand_dims %74 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_33 = arith.constant 1 : i32
    %cst_34 = arith.constant dense<1> : tensor<1x64xi32>
    %85 = arith.extsi %84 : tensor<1x64xi32> to tensor<1x64xi64>
    %86 = arith.extsi %cst_34 : tensor<1x64xi32> to tensor<1x64xi64>
    %87 = arith.muli %85, %86 : tensor<1x64xi64>
    %c2147483647_i64_35 = arith.constant 2147483647 : i64
    %c-2147483648_i64_36 = arith.constant -2147483648 : i64
    %cst_37 = arith.constant dense<2147483647> : tensor<1x64xi64>
    %88 = arith.cmpi sle, %87, %cst_37 : tensor<1x64xi64>
    %cst_38 = arith.constant dense<-2147483648> : tensor<1x64xi64>
    %89 = arith.cmpi sge, %87, %cst_38 : tensor<1x64xi64>
    %90 = arith.andi %88, %89 : tensor<1x64xi1>
    %91 = arith.muli %84, %cst_34 : tensor<1x64xi32>
    %92 = tt.broadcast %83 : tensor<128x1xi32> -> tensor<128x64xi32>
    %93 = tt.broadcast %91 : tensor<1x64xi32> -> tensor<128x64xi32>
    %94 = arith.extsi %92 : tensor<128x64xi32> to tensor<128x64xi64>
    %95 = arith.extsi %93 : tensor<128x64xi32> to tensor<128x64xi64>
    %96 = arith.addi %94, %95 : tensor<128x64xi64>
    %c2147483647_i64_39 = arith.constant 2147483647 : i64
    %c-2147483648_i64_40 = arith.constant -2147483648 : i64
    %cst_41 = arith.constant dense<2147483647> : tensor<128x64xi64>
    %97 = arith.cmpi sle, %96, %cst_41 : tensor<128x64xi64>
    %cst_42 = arith.constant dense<-2147483648> : tensor<128x64xi64>
    %98 = arith.cmpi sge, %96, %cst_42 : tensor<128x64xi64>
    %99 = arith.andi %97, %98 : tensor<128x64xi1>
    %100 = arith.addi %92, %93 : tensor<128x64xi32>
    %101 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %102 = tt.addptr %101, %100 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    %103 = tt.expand_dims %74 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %104 = tt.splat %arg7 : i32 -> tensor<64x1xi32>
    %105 = arith.extsi %103 : tensor<64x1xi32> to tensor<64x1xi64>
    %106 = arith.extsi %104 : tensor<64x1xi32> to tensor<64x1xi64>
    %107 = arith.muli %105, %106 : tensor<64x1xi64>
    %c2147483647_i64_43 = arith.constant 2147483647 : i64
    %c-2147483648_i64_44 = arith.constant -2147483648 : i64
    %cst_45 = arith.constant dense<2147483647> : tensor<64x1xi64>
    %108 = arith.cmpi sle, %107, %cst_45 : tensor<64x1xi64>
    %cst_46 = arith.constant dense<-2147483648> : tensor<64x1xi64>
    %109 = arith.cmpi sge, %107, %cst_46 : tensor<64x1xi64>
    %110 = arith.andi %108, %109 : tensor<64x1xi1>
    %111 = arith.muli %103, %104 : tensor<64x1xi32>
    %112 = tt.expand_dims %73 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %c1_i32_47 = arith.constant 1 : i32
    %c1_i32_48 = arith.constant 1 : i32
    %cst_49 = arith.constant dense<1> : tensor<1x128xi32>
    %113 = arith.extsi %112 : tensor<1x128xi32> to tensor<1x128xi64>
    %114 = arith.extsi %cst_49 : tensor<1x128xi32> to tensor<1x128xi64>
    %115 = arith.muli %113, %114 : tensor<1x128xi64>
    %c2147483647_i64_50 = arith.constant 2147483647 : i64
    %c-2147483648_i64_51 = arith.constant -2147483648 : i64
    %cst_52 = arith.constant dense<2147483647> : tensor<1x128xi64>
    %116 = arith.cmpi sle, %115, %cst_52 : tensor<1x128xi64>
    %cst_53 = arith.constant dense<-2147483648> : tensor<1x128xi64>
    %117 = arith.cmpi sge, %115, %cst_53 : tensor<1x128xi64>
    %118 = arith.andi %116, %117 : tensor<1x128xi1>
    %119 = arith.muli %112, %cst_49 : tensor<1x128xi32>
    %120 = tt.broadcast %111 : tensor<64x1xi32> -> tensor<64x128xi32>
    %121 = tt.broadcast %119 : tensor<1x128xi32> -> tensor<64x128xi32>
    %122 = arith.extsi %120 : tensor<64x128xi32> to tensor<64x128xi64>
    %123 = arith.extsi %121 : tensor<64x128xi32> to tensor<64x128xi64>
    %124 = arith.addi %122, %123 : tensor<64x128xi64>
    %c2147483647_i64_54 = arith.constant 2147483647 : i64
    %c-2147483648_i64_55 = arith.constant -2147483648 : i64
    %cst_56 = arith.constant dense<2147483647> : tensor<64x128xi64>
    %125 = arith.cmpi sle, %124, %cst_56 : tensor<64x128xi64>
    %cst_57 = arith.constant dense<-2147483648> : tensor<64x128xi64>
    %126 = arith.cmpi sge, %124, %cst_57 : tensor<64x128xi64>
    %127 = arith.andi %125, %126 : tensor<64x128xi1>
    %128 = arith.addi %120, %121 : tensor<64x128xi32>
    %129 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
    %130 = tt.addptr %129, %128 : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
    %131 = tt.call @"triton.language.standard.zeros____(0, 0)cconstexpr_128__(0, 1)cconstexpr_128__(1,)cconstexpr_fp32_"() : () -> tensor<128x128xf32>
    %132 = tt.call @"triton.language.standard.cdiv__i32__(1,)cconstexpr_64_"(%arg5) : (i32) -> i32
    %c0_i32_58 = arith.constant 0 : i32
    %c1_i32_59 = arith.constant 1 : i32
    %133 = arith.bitcast %c0_i32_58 : i32 to i32
    %134 = arith.bitcast %132 : i32 to i32
    %135 = arith.bitcast %c1_i32_59 : i32 to i32
    %136 = ub.poison : i32
    %137:3 = scf.for %arg9 = %133 to %134 step %135 iter_args(%arg10 = %102, %arg11 = %130, %arg12 = %131) -> (tensor<128x64x!tt.ptr<f16>>, tensor<64x128x!tt.ptr<f16>>, tensor<128x128xf32>)  : i32 {
      %202 = tt.expand_dims %74 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %c64_i32 = arith.constant 64 : i32
      %c64_i32_87 = arith.constant 64 : i32
      %203 = arith.extsi %arg9 : i32 to i64
      %204 = arith.extsi %c64_i32_87 : i32 to i64
      %205 = arith.muli %203, %204 : i64
      %c2147483647_i64_88 = arith.constant 2147483647 : i64
      %c-2147483648_i64_89 = arith.constant -2147483648 : i64
      %206 = arith.cmpi sle, %205, %c2147483647_i64_88 : i64
      %207 = arith.cmpi sge, %205, %c-2147483648_i64_89 : i64
      %208 = arith.andi %206, %207 : i1
      %209 = arith.muli %arg9, %c64_i32_87 : i32
      %210 = arith.extsi %arg5 : i32 to i64
      %211 = arith.extsi %209 : i32 to i64
      %212 = arith.subi %210, %211 : i64
      %c2147483647_i64_90 = arith.constant 2147483647 : i64
      %c-2147483648_i64_91 = arith.constant -2147483648 : i64
      %213 = arith.cmpi sle, %212, %c2147483647_i64_90 : i64
      %214 = arith.cmpi sge, %212, %c-2147483648_i64_91 : i64
      %215 = arith.andi %213, %214 : i1
      %216 = arith.subi %arg5, %209 : i32
      %217 = tt.splat %216 : i32 -> tensor<1x64xi32>
      %218 = arith.cmpi slt, %202, %217 : tensor<1x64xi32>
      %cst_92 = arith.constant 0.000000e+00 : f32
      %219 = tt.broadcast %218 : tensor<1x64xi1> -> tensor<128x64xi1>
      %cst_93 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
      %220 = arith.truncf %cst_93 : tensor<128x64xf32> to tensor<128x64xf16>
      %221 = tt.load %arg10, %219, %220 : tensor<128x64x!tt.ptr<f16>>
      %222 = tt.expand_dims %74 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %c64_i32_94 = arith.constant 64 : i32
      %c64_i32_95 = arith.constant 64 : i32
      %223 = arith.extsi %arg9 : i32 to i64
      %224 = arith.extsi %c64_i32_95 : i32 to i64
      %225 = arith.muli %223, %224 : i64
      %c2147483647_i64_96 = arith.constant 2147483647 : i64
      %c-2147483648_i64_97 = arith.constant -2147483648 : i64
      %226 = arith.cmpi sle, %225, %c2147483647_i64_96 : i64
      %227 = arith.cmpi sge, %225, %c-2147483648_i64_97 : i64
      %228 = arith.andi %226, %227 : i1
      %229 = arith.muli %arg9, %c64_i32_95 : i32
      %230 = arith.extsi %arg5 : i32 to i64
      %231 = arith.extsi %229 : i32 to i64
      %232 = arith.subi %230, %231 : i64
      %c2147483647_i64_98 = arith.constant 2147483647 : i64
      %c-2147483648_i64_99 = arith.constant -2147483648 : i64
      %233 = arith.cmpi sle, %232, %c2147483647_i64_98 : i64
      %234 = arith.cmpi sge, %232, %c-2147483648_i64_99 : i64
      %235 = arith.andi %233, %234 : i1
      %236 = arith.subi %arg5, %229 : i32
      %237 = tt.splat %236 : i32 -> tensor<64x1xi32>
      %238 = arith.cmpi slt, %222, %237 : tensor<64x1xi32>
      %cst_100 = arith.constant 0.000000e+00 : f32
      %239 = tt.broadcast %238 : tensor<64x1xi1> -> tensor<64x128xi1>
      %cst_101 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
      %240 = arith.truncf %cst_101 : tensor<64x128xf32> to tensor<64x128xf16>
      %241 = tt.load %arg11, %239, %240 : tensor<64x128x!tt.ptr<f16>>
      %cst_102 = arith.constant 0.000000e+00 : f32
      %242 = tt.dot %221, %241, %arg12, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      %c64_i32_103 = arith.constant 64 : i32
      %cst_104 = arith.constant dense<64> : tensor<128x64xi32>
      %243 = tt.addptr %arg10, %cst_104 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %c64_i32_105 = arith.constant 64 : i32
      %c64_i32_106 = arith.constant 64 : i32
      %244 = arith.extsi %c64_i32_106 : i32 to i64
      %245 = arith.extsi %arg7 : i32 to i64
      %246 = arith.muli %244, %245 : i64
      %c2147483647_i64_107 = arith.constant 2147483647 : i64
      %c-2147483648_i64_108 = arith.constant -2147483648 : i64
      %247 = arith.cmpi sle, %246, %c2147483647_i64_107 : i64
      %248 = arith.cmpi sge, %246, %c-2147483648_i64_108 : i64
      %249 = arith.andi %247, %248 : i1
      %250 = arith.muli %c64_i32_106, %arg7 : i32
      %251 = tt.splat %250 : i32 -> tensor<64x128xi32>
      %252 = tt.addptr %arg11, %251 : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
      scf.yield %243, %252, %242 : tensor<128x64x!tt.ptr<f16>>, tensor<64x128x!tt.ptr<f16>>, tensor<128x128xf32>
    }
    %138 = arith.truncf %137#2 : tensor<128x128xf32> to tensor<128x128xf16>
    %c128_i32_60 = arith.constant 128 : i32
    %c128_i32_61 = arith.constant 128 : i32
    %139 = arith.extsi %33 : i32 to i64
    %140 = arith.extsi %c128_i32_61 : i32 to i64
    %141 = arith.muli %139, %140 : i64
    %c2147483647_i64_62 = arith.constant 2147483647 : i64
    %c-2147483648_i64_63 = arith.constant -2147483648 : i64
    %142 = arith.cmpi sle, %141, %c2147483647_i64_62 : i64
    %143 = arith.cmpi sge, %141, %c-2147483648_i64_63 : i64
    %144 = arith.andi %142, %143 : i1
    %145 = arith.muli %33, %c128_i32_61 : i32
    %146 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %147 = tt.splat %145 : i32 -> tensor<128xi32>
    %148 = arith.extsi %147 : tensor<128xi32> to tensor<128xi64>
    %149 = arith.extsi %146 : tensor<128xi32> to tensor<128xi64>
    %150 = arith.addi %148, %149 : tensor<128xi64>
    %c2147483647_i64_64 = arith.constant 2147483647 : i64
    %c-2147483648_i64_65 = arith.constant -2147483648 : i64
    %cst_66 = arith.constant dense<2147483647> : tensor<128xi64>
    %151 = arith.cmpi sle, %150, %cst_66 : tensor<128xi64>
    %cst_67 = arith.constant dense<-2147483648> : tensor<128xi64>
    %152 = arith.cmpi sge, %150, %cst_67 : tensor<128xi64>
    %153 = arith.andi %151, %152 : tensor<128xi1>
    %154 = arith.addi %147, %146 : tensor<128xi32>
    %c128_i32_68 = arith.constant 128 : i32
    %c128_i32_69 = arith.constant 128 : i32
    %155 = arith.extsi %35 : i32 to i64
    %156 = arith.extsi %c128_i32_69 : i32 to i64
    %157 = arith.muli %155, %156 : i64
    %c2147483647_i64_70 = arith.constant 2147483647 : i64
    %c-2147483648_i64_71 = arith.constant -2147483648 : i64
    %158 = arith.cmpi sle, %157, %c2147483647_i64_70 : i64
    %159 = arith.cmpi sge, %157, %c-2147483648_i64_71 : i64
    %160 = arith.andi %158, %159 : i1
    %161 = arith.muli %35, %c128_i32_69 : i32
    %162 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %163 = tt.splat %161 : i32 -> tensor<128xi32>
    %164 = arith.extsi %163 : tensor<128xi32> to tensor<128xi64>
    %165 = arith.extsi %162 : tensor<128xi32> to tensor<128xi64>
    %166 = arith.addi %164, %165 : tensor<128xi64>
    %c2147483647_i64_72 = arith.constant 2147483647 : i64
    %c-2147483648_i64_73 = arith.constant -2147483648 : i64
    %cst_74 = arith.constant dense<2147483647> : tensor<128xi64>
    %167 = arith.cmpi sle, %166, %cst_74 : tensor<128xi64>
    %cst_75 = arith.constant dense<-2147483648> : tensor<128xi64>
    %168 = arith.cmpi sge, %166, %cst_75 : tensor<128xi64>
    %169 = arith.andi %167, %168 : tensor<128xi1>
    %170 = arith.addi %163, %162 : tensor<128xi32>
    %171 = tt.expand_dims %154 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %172 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
    %173 = arith.extsi %172 : tensor<128x1xi32> to tensor<128x1xi64>
    %174 = arith.extsi %171 : tensor<128x1xi32> to tensor<128x1xi64>
    %175 = arith.muli %173, %174 : tensor<128x1xi64>
    %c2147483647_i64_76 = arith.constant 2147483647 : i64
    %c-2147483648_i64_77 = arith.constant -2147483648 : i64
    %cst_78 = arith.constant dense<2147483647> : tensor<128x1xi64>
    %176 = arith.cmpi sle, %175, %cst_78 : tensor<128x1xi64>
    %cst_79 = arith.constant dense<-2147483648> : tensor<128x1xi64>
    %177 = arith.cmpi sge, %175, %cst_79 : tensor<128x1xi64>
    %178 = arith.andi %176, %177 : tensor<128x1xi1>
    %179 = arith.muli %172, %171 : tensor<128x1xi32>
    %180 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %181 = tt.addptr %180, %179 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
    %182 = tt.expand_dims %170 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %c1_i32_80 = arith.constant 1 : i32
    %c1_i32_81 = arith.constant 1 : i32
    %cst_82 = arith.constant dense<1> : tensor<1x128xi32>
    %183 = arith.extsi %cst_82 : tensor<1x128xi32> to tensor<1x128xi64>
    %184 = arith.extsi %182 : tensor<1x128xi32> to tensor<1x128xi64>
    %185 = arith.muli %183, %184 : tensor<1x128xi64>
    %c2147483647_i64_83 = arith.constant 2147483647 : i64
    %c-2147483648_i64_84 = arith.constant -2147483648 : i64
    %cst_85 = arith.constant dense<2147483647> : tensor<1x128xi64>
    %186 = arith.cmpi sle, %185, %cst_85 : tensor<1x128xi64>
    %cst_86 = arith.constant dense<-2147483648> : tensor<1x128xi64>
    %187 = arith.cmpi sge, %185, %cst_86 : tensor<1x128xi64>
    %188 = arith.andi %186, %187 : tensor<1x128xi1>
    %189 = arith.muli %cst_82, %182 : tensor<1x128xi32>
    %190 = tt.broadcast %181 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
    %191 = tt.broadcast %189 : tensor<1x128xi32> -> tensor<128x128xi32>
    %192 = tt.addptr %190, %191 : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
    %193 = tt.expand_dims %154 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %194 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %195 = arith.cmpi slt, %193, %194 : tensor<128x1xi32>
    %196 = tt.expand_dims %170 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %197 = tt.splat %arg4 : i32 -> tensor<1x128xi32>
    %198 = arith.cmpi slt, %196, %197 : tensor<1x128xi32>
    %199 = tt.broadcast %195 : tensor<128x1xi1> -> tensor<128x128xi1>
    %200 = tt.broadcast %198 : tensor<1x128xi1> -> tensor<128x128xi1>
    %201 = arith.andi %199, %200 : tensor<128x128xi1>
    tt.store %192, %138, %201 : tensor<128x128x!tt.ptr<f16>>
    tt.return
  }
  tt.func private @"triton.language.standard.cdiv__i32__(1,)cconstexpr_128_"(%arg0: i32) -> i32 attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c127_i32_0 = arith.constant 127 : i32
    %0 = arith.extsi %arg0 : i32 to i64
    %1 = arith.extsi %c127_i32_0 : i32 to i64
    %2 = arith.addi %0, %1 : i64
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %3 = arith.cmpi sle, %2, %c2147483647_i64 : i64
    %4 = arith.cmpi sge, %2, %c-2147483648_i64 : i64
    %5 = arith.andi %3, %4 : i1
    %6 = arith.addi %arg0, %c127_i32_0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i32_1 = arith.constant 128 : i32
    %7 = arith.divsi %6, %c128_i32_1 : i32
    tt.return %7 : i32
  ^bb1:  // no predecessors
    %8 = ub.poison : i32
    tt.return %8 : i32
  }
  tt.func private @"triton.language.standard.zeros____(0, 0)cconstexpr_128__(0, 1)cconstexpr_128__(1,)cconstexpr_fp32_"() -> tensor<128x128xf32> attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    tt.return %cst_0 : tensor<128x128xf32>
  ^bb1:  // no predecessors
    %0 = ub.poison : tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
  }
  tt.func private @"triton.language.standard.cdiv__i32__(1,)cconstexpr_64_"(%arg0: i32) -> i32 attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c63_i32_0 = arith.constant 63 : i32
    %0 = arith.extsi %arg0 : i32 to i64
    %1 = arith.extsi %c63_i32_0 : i32 to i64
    %2 = arith.addi %0, %1 : i64
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %3 = arith.cmpi sle, %2, %c2147483647_i64 : i64
    %4 = arith.cmpi sge, %2, %c-2147483648_i64 : i64
    %5 = arith.andi %3, %4 : i1
    %6 = arith.addi %arg0, %c63_i32_0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i32_1 = arith.constant 64 : i32
    %7 = arith.divsi %6, %c64_i32_1 : i32
    tt.return %7 : i32
  ^bb1:  // no predecessors
    %8 = ub.poison : i32
    tt.return %8 : i32
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },triton-rewrite-tensor-pointer,triton-rewrite-tensor-descriptor-to-pointer,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},triton-combine,triton-reorder-broadcast,cse,symbol-dce,triton-loop-unroll))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}
