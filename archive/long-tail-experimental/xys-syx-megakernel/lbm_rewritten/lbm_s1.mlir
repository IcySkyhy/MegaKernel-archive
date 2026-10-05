#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "_Z27performStreamCollide_kernelPvPf">
#alias_scope_domain1 = #llvm.alias_scope_domain<id = distinct[1]<>, description = "_Z27performStreamCollide_kernelPvPf">
#alias_scope_domain2 = #llvm.alias_scope_domain<id = distinct[2]<>, description = "_Z27performStreamCollide_kernelPvPf">
#alias_scope_domain3 = #llvm.alias_scope_domain<id = distinct[3]<>, description = "_Z27performStreamCollide_kernelPvPf">
#set = affine_set<()[s0] : (s0 == 0)>
#alias_scope = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain, description = "_Z27performStreamCollide_kernelPvPf: argument 0">
#alias_scope1 = #llvm.alias_scope<id = distinct[5]<>, domain = #alias_scope_domain, description = "_Z27performStreamCollide_kernelPvPf: argument 1">
#alias_scope2 = #llvm.alias_scope<id = distinct[6]<>, domain = #alias_scope_domain1, description = "_Z27performStreamCollide_kernelPvPf: argument 0">
#alias_scope3 = #llvm.alias_scope<id = distinct[7]<>, domain = #alias_scope_domain1, description = "_Z27performStreamCollide_kernelPvPf: argument 1">
#alias_scope4 = #llvm.alias_scope<id = distinct[8]<>, domain = #alias_scope_domain2, description = "_Z27performStreamCollide_kernelPvPf: argument 0">
#alias_scope5 = #llvm.alias_scope<id = distinct[9]<>, domain = #alias_scope_domain2, description = "_Z27performStreamCollide_kernelPvPf: argument 1">
#alias_scope6 = #llvm.alias_scope<id = distinct[10]<>, domain = #alias_scope_domain3, description = "_Z27performStreamCollide_kernelPvPf: argument 0">
#alias_scope7 = #llvm.alias_scope<id = distinct[11]<>, domain = #alias_scope_domain3, description = "_Z27performStreamCollide_kernelPvPf: argument 1">
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.module_flags [#llvm.mlir.module_flag<min, "PIC Level", 0 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.mlir.global external local_unnamed_addr @A() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global private unnamed_addr constant @".str"("LBM_allocateGrid: could not allocate %.1f MByte\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.1"("LBM_allocateGrid: allocated %.1f MByte\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @stderr() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
  llvm.mlir.global private unnamed_addr constant @".str.2"("CUDA error on line %d: %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.3"("rb\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.4"("LBM_showGridStatistics:\0A\09nObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\0A\09minRho: %8.4f maxRho: %8.4f mass: %e\0A\09minU: %e maxU: %e\0A\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.5"("wb\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.6"("w\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("can't open %s: %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("%e %e %e\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i32 = arith.constant -1 : i32
    %c51_i32 = arith.constant 51 : i32
    %0 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %false = arith.constant false
    %c2 = arith.constant 2 : index
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = ub.poison : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.0541666672 : f32
    %cst_0 = arith.constant -3.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %cst_2 = arith.constant 4.500000e+00 : f32
    %cst_3 = arith.constant 0.950000047 : f32
    %cst_4 = arith.constant 0.108333334 : f32
    %cst_5 = arith.constant -0.950000047 : f32
    %cst_6 = arith.constant 0.650000035 : f32
    %cst_7 = arith.constant -1.000000e+00 : f32
    %cst_8 = arith.constant 1.500000e+00 : f32
    %cst_9 = arith.constant 0.000000e+00 : f32
    %cst_10 = arith.constant 2.000000e-03 : f32
    %cst_11 = arith.constant 5.000000e-03 : f32
    %c1 = arith.constant 1 : index
    %c150 = arith.constant 150 : index
    %c120 = arith.constant 120 : index
    %2 = llvm.mlir.addressof @stderr : !llvm.ptr
    %3 = arith.addi %arg0, %c1_i32 : i32
    %4 = arith.cmpi ult, %3, %c3_i32 : i32
    %5 = scf.if %4 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %6 = arith.divsi %arg0, %c2_i32 : i32
      %7 = arith.maxui %6, %c1_i32 : i32
      %c5 = arith.constant 5 : index
      %sl_tc = arith.index_castui %7 : i32 to index
      %sl_H = arith.muli %sl_tc, %c2 : index
      %sl_k = arith.constant 4 : index
      %sl_ns = arith.constant 30 : index
      %sl_nw = arith.constant 33 : index
      %sl_z = arith.constant 0 : index
      %sl_rows = arith.constant 76800 : index
      scf.for %sl_b = %sl_z to %sl_H step %sl_k {
        %sl_rem = arith.subi %sl_H, %sl_b : index
        %sl_kk = arith.minui %sl_rem, %sl_k : index
        scf.for %sl_w = %sl_z to %sl_nw step %c1 {
          scf.for %sl_j = %sl_z to %sl_kk step %c1 {
            %sl_i = arith.subi %sl_w, %sl_j : index
            %sl_ge = arith.cmpi sge, %sl_i, %sl_z : index
            %sl_lt = arith.cmpi slt, %sl_i, %sl_ns : index
            %sl_ok = arith.andi %sl_ge, %sl_lt : i1
            scf.if %sl_ok {
              %sl_s = arith.addi %sl_b, %sl_j : index
              %sl_par = arith.remui %sl_s, %c2 : index
              %sl_even = arith.cmpi eq, %sl_par, %sl_z : index
              %sl_src = llvm.select %sl_even, %arg1, %arg2 : i1, !llvm.ptr
              %sl_dst = llvm.select %sl_even, %arg2, %arg1 : i1, !llvm.ptr
              %sl_off = arith.muli %sl_i, %sl_rows : index
              %sl_o64 = arith.index_cast %sl_off : index to i64
              %sl_srcS = llvm.getelementptr %sl_src[%sl_o64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %sl_dstS = llvm.getelementptr %sl_dst[%sl_o64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %sl_wr = "enzymexla.gpu_wrapper"(%c120, %c5, %c1, %c120, %c1, %c1) ({
                  affine.parallel (%arg4, %arg5) = (0, 0) to (600, 120) {
                    llvm.intr.experimental.noalias.scope.decl #alias_scope
                    llvm.intr.experimental.noalias.scope.decl #alias_scope1
                    %w17 = "enzymexla.pointer2memref"(%sl_srcS) : (!llvm.ptr) -> memref<?xf32>
                    %w18 = affine.load %w17[%arg5 + %arg4 * 128] : memref<?xf32>
                    %w19 = affine.load %w17[%arg5 + %arg4 * 128 + 2365440] : memref<?xf32>
                    %w20 = affine.load %w17[%arg5 + %arg4 * 128 + 4730880] : memref<?xf32>
                    %w21 = affine.load %w17[%arg5 + %arg4 * 128 + 7096320] : memref<?xf32>
                    %w22 = affine.load %w17[%arg5 + %arg4 * 128 + 9461760] : memref<?xf32>
                    %w23 = affine.load %w17[%arg5 + %arg4 * 128 + 11827200] : memref<?xf32>
                    %w24 = affine.load %w17[%arg5 + %arg4 * 128 + 14192640] : memref<?xf32>
                    %w25 = affine.load %w17[%arg5 + %arg4 * 128 + 16558080] : memref<?xf32>
                    %w26 = affine.load %w17[%arg5 + %arg4 * 128 + 18923520] : memref<?xf32>
                    %w27 = affine.load %w17[%arg5 + %arg4 * 128 + 21288960] : memref<?xf32>
                    %w28 = affine.load %w17[%arg5 + %arg4 * 128 + 23654400] : memref<?xf32>
                    %w29 = affine.load %w17[%arg5 + %arg4 * 128 + 26019840] : memref<?xf32>
                    %w30 = affine.load %w17[%arg5 + %arg4 * 128 + 28385280] : memref<?xf32>
                    %w31 = affine.load %w17[%arg5 + %arg4 * 128 + 30750720] : memref<?xf32>
                    %w32 = affine.load %w17[%arg5 + %arg4 * 128 + 33116160] : memref<?xf32>
                    %w33 = affine.load %w17[%arg5 + %arg4 * 128 + 35481600] : memref<?xf32>
                    %w34 = affine.load %w17[%arg5 + %arg4 * 128 + 37847040] : memref<?xf32>
                    %w35 = affine.load %w17[%arg5 + %arg4 * 128 + 40212480] : memref<?xf32>
                    %w36 = affine.load %w17[%arg5 + %arg4 * 128 + 42577920] : memref<?xf32>
                    %w37 = "enzymexla.pointer2memref"(%sl_srcS) : (!llvm.ptr) -> memref<?xi8>
                    %w38 = affine.load %w37[%arg5 * 4 + %arg4 * 512 + 179773440] : memref<?xi8>
                    %w39 = arith.extui %w38 : i8 to i32
                    %w40 = arith.andi %w39, %c1_i32 : i32
                    %w41 = arith.cmpi eq, %w40, %c0_i32 : i32
                    %w42:19 = scf.if %w41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                      %w44 = arith.addf %w19, %w18 fastmath<fast> : f32
                      %w45 = arith.addf %w44, %w20 fastmath<fast> : f32
                      %w46 = arith.addf %w45, %w21 fastmath<fast> : f32
                      %w47 = arith.addf %w46, %w22 fastmath<fast> : f32
                      %w48 = arith.addf %w47, %w23 fastmath<fast> : f32
                      %w49 = arith.addf %w48, %w24 fastmath<fast> : f32
                      %w50 = arith.addf %w49, %w25 fastmath<fast> : f32
                      %w51 = arith.addf %w50, %w26 fastmath<fast> : f32
                      %w52 = arith.addf %w51, %w27 fastmath<fast> : f32
                      %w53 = arith.addf %w52, %w28 fastmath<fast> : f32
                      %w54 = arith.addf %w53, %w29 fastmath<fast> : f32
                      %w55 = arith.addf %w54, %w30 fastmath<fast> : f32
                      %w56 = arith.addf %w55, %w31 fastmath<fast> : f32
                      %w57 = arith.addf %w56, %w32 fastmath<fast> : f32
                      %w58 = arith.addf %w57, %w33 fastmath<fast> : f32
                      %w59 = arith.addf %w58, %w34 fastmath<fast> : f32
                      %w60 = arith.addf %w59, %w35 fastmath<fast> : f32
                      %w61 = arith.addf %w60, %w36 fastmath<fast> : f32
                      %w62 = arith.addf %w21, %w25 fastmath<fast> : f32
                      %w63 = arith.addf %w22, %w26 fastmath<fast> : f32
                      %w64 = arith.addf %w62, %w27 fastmath<fast> : f32
                      %w65 = arith.addf %w63, %w28 fastmath<fast> : f32
                      %w66 = arith.subf %w64, %w65 fastmath<fast> : f32
                      %w67 = arith.addf %w66, %w33 fastmath<fast> : f32
                      %w68 = arith.addf %w67, %w34 fastmath<fast> : f32
                      %w69 = arith.addf %w35, %w36 fastmath<fast> : f32
                      %w70 = arith.subf %w68, %w69 fastmath<fast> : f32
                      %w71 = arith.subf %w19, %w20 fastmath<fast> : f32
                      %w72 = arith.addf %w71, %w25 fastmath<fast> : f32
                      %w73 = arith.addf %w72, %w26 fastmath<fast> : f32
                      %w74 = arith.addf %w27, %w28 fastmath<fast> : f32
                      %w75 = arith.subf %w73, %w74 fastmath<fast> : f32
                      %w76 = arith.addf %w75, %w29 fastmath<fast> : f32
                      %w77 = arith.addf %w76, %w30 fastmath<fast> : f32
                      %w78 = arith.addf %w31, %w32 fastmath<fast> : f32
                      %w79 = arith.subf %w77, %w78 fastmath<fast> : f32
                      %w80 = arith.addf %w23, %w29 fastmath<fast> : f32
                      %w81 = arith.addf %w24, %w30 fastmath<fast> : f32
                      %w82 = arith.addf %w80, %w31 fastmath<fast> : f32
                      %w83 = arith.addf %w81, %w32 fastmath<fast> : f32
                      %w84 = arith.addf %w82, %w33 fastmath<fast> : f32
                      %w85 = arith.addf %w83, %w34 fastmath<fast> : f32
                      %w86 = arith.addf %w84, %w35 fastmath<fast> : f32
                      %w87 = arith.addf %w85, %w36 fastmath<fast> : f32
                      %w88 = arith.subf %w86, %w87 fastmath<fast> : f32
                      %w89 = arith.divf %w70, %w61 fastmath<fast> : f32
                      %w90 = arith.divf %w79, %w61 fastmath<fast> : f32
                      %w91 = arith.divf %w88, %w61 fastmath<fast> : f32
                      %w92 = arith.andi %w39, %c2_i32 : i32
                      %w93 = arith.cmpi eq, %w92, %c0_i32 : i32
                      %w94 = arith.select %w93, %w89, %cst_11 : f32
                      %w95 = arith.select %w93, %w90, %cst_10 : f32
                      %w96 = arith.select %w93, %w91, %cst_9 : f32
                      %w97 = arith.mulf %w94, %w94 fastmath<fast> : f32
                      %w98 = arith.mulf %w95, %w95 fastmath<fast> : f32
                      %w99 = arith.addf %w97, %w98 fastmath<fast> : f32
                      %w100 = arith.mulf %w96, %w96 fastmath<fast> : f32
                      %w101 = arith.addf %w99, %w100 fastmath<fast> : f32
                      %w102 = arith.mulf %w101, %cst_8 fastmath<fast> : f32
                      %w103 = arith.addf %w102, %cst_7 fastmath<fast> : f32
                      %w104 = arith.mulf %w61, %cst_6 fastmath<fast> : f32
                      %w105 = arith.mulf %w18, %cst_5 fastmath<fast> : f32
                      %w106 = arith.mulf %w104, %w103 fastmath<fast> : f32
                      %w107 = arith.subf %w105, %w106 fastmath<fast> : f32
                      %w108 = arith.mulf %w61, %cst_4 fastmath<fast> : f32
                      %w109 = arith.mulf %w19, %cst_3 fastmath<fast> : f32
                      %w110 = arith.mulf %w95, %cst_2 fastmath<fast> : f32
                      %w111 = arith.addf %w110, %cst_1 fastmath<fast> : f32
                      %w112 = arith.mulf %w111, %w95 fastmath<fast> : f32
                      %w113 = arith.subf %w112, %w103 fastmath<fast> : f32
                      %w114 = arith.mulf %w113, %w108 fastmath<fast> : f32
                      %w115 = arith.subf %w114, %w109 fastmath<fast> : f32
                      %w116 = arith.mulf %w20, %cst_3 fastmath<fast> : f32
                      %w117 = arith.addf %w110, %cst_0 fastmath<fast> : f32
                      %w118 = arith.mulf %w117, %w95 fastmath<fast> : f32
                      %w119 = arith.subf %w118, %w103 fastmath<fast> : f32
                      %w120 = arith.mulf %w119, %w108 fastmath<fast> : f32
                      %w121 = arith.subf %w120, %w116 fastmath<fast> : f32
                      %w122 = arith.mulf %w23, %cst_3 fastmath<fast> : f32
                      %w123 = arith.mulf %w96, %cst_2 fastmath<fast> : f32
                      %w124 = arith.addf %w123, %cst_1 fastmath<fast> : f32
                      %w125 = arith.mulf %w124, %w96 fastmath<fast> : f32
                      %w126 = arith.subf %w125, %w103 fastmath<fast> : f32
                      %w127 = arith.mulf %w126, %w108 fastmath<fast> : f32
                      %w128 = arith.subf %w127, %w122 fastmath<fast> : f32
                      %w129 = arith.mulf %w24, %cst_3 fastmath<fast> : f32
                      %w130 = arith.addf %w123, %cst_0 fastmath<fast> : f32
                      %w131 = arith.mulf %w130, %w96 fastmath<fast> : f32
                      %w132 = arith.subf %w131, %w103 fastmath<fast> : f32
                      %w133 = arith.mulf %w132, %w108 fastmath<fast> : f32
                      %w134 = arith.subf %w133, %w129 fastmath<fast> : f32
                      %w135 = arith.mulf %w21, %cst_3 fastmath<fast> : f32
                      %w136 = arith.mulf %w94, %cst_2 fastmath<fast> : f32
                      %w137 = arith.addf %w136, %cst_1 fastmath<fast> : f32
                      %w138 = arith.mulf %w137, %w94 fastmath<fast> : f32
                      %w139 = arith.subf %w138, %w103 fastmath<fast> : f32
                      %w140 = arith.mulf %w139, %w108 fastmath<fast> : f32
                      %w141 = arith.subf %w140, %w135 fastmath<fast> : f32
                      %w142 = arith.mulf %w22, %cst_3 fastmath<fast> : f32
                      %w143 = arith.addf %w136, %cst_0 fastmath<fast> : f32
                      %w144 = arith.mulf %w143, %w94 fastmath<fast> : f32
                      %w145 = arith.subf %w144, %w103 fastmath<fast> : f32
                      %w146 = arith.mulf %w145, %w108 fastmath<fast> : f32
                      %w147 = arith.subf %w146, %w142 fastmath<fast> : f32
                      %w148 = arith.mulf %w61, %cst fastmath<fast> : f32
                      %w149 = arith.mulf %w29, %cst_3 fastmath<fast> : f32
                      %w150 = arith.addf %w95, %w96 fastmath<fast> : f32
                      %w151 = arith.mulf %w150, %cst_2 fastmath<fast> : f32
                      %w152 = arith.addf %w151, %cst_1 fastmath<fast> : f32
                      %w153 = arith.mulf %w152, %w150 fastmath<fast> : f32
                      %w154 = arith.subf %w153, %w103 fastmath<fast> : f32
                      %w155 = arith.mulf %w154, %w148 fastmath<fast> : f32
                      %w156 = arith.subf %w155, %w149 fastmath<fast> : f32
                      %w157 = arith.mulf %w30, %cst_3 fastmath<fast> : f32
                      %w158 = arith.subf %w95, %w96 fastmath<fast> : f32
                      %w159 = arith.mulf %w158, %cst_2 fastmath<fast> : f32
                      %w160 = arith.addf %w159, %cst_1 fastmath<fast> : f32
                      %w161 = arith.mulf %w160, %w158 fastmath<fast> : f32
                      %w162 = arith.subf %w161, %w103 fastmath<fast> : f32
                      %w163 = arith.mulf %w162, %w148 fastmath<fast> : f32
                      %w164 = arith.subf %w163, %w157 fastmath<fast> : f32
                      %w165 = arith.mulf %w31, %cst_3 fastmath<fast> : f32
                      %w166 = arith.subf %w96, %w95 fastmath<fast> : f32
                      %w167 = arith.mulf %w166, %cst_2 fastmath<fast> : f32
                      %w168 = arith.addf %w167, %cst_1 fastmath<fast> : f32
                      %w169 = arith.mulf %w168, %w166 fastmath<fast> : f32
                      %w170 = arith.subf %w169, %w103 fastmath<fast> : f32
                      %w171 = arith.mulf %w170, %w148 fastmath<fast> : f32
                      %w172 = arith.subf %w171, %w165 fastmath<fast> : f32
                      %w173 = arith.mulf %w32, %cst_3 fastmath<fast> : f32
                      %w174 = arith.negf %w150 fastmath<fast> : f32
                      %w175 = arith.subf %cst_1, %w151 fastmath<fast> : f32
                      %w176 = arith.mulf %w175, %w174 fastmath<fast> : f32
                      %w177 = arith.subf %w176, %w103 fastmath<fast> : f32
                      %w178 = arith.mulf %w177, %w148 fastmath<fast> : f32
                      %w179 = arith.subf %w178, %w173 fastmath<fast> : f32
                      %w180 = arith.mulf %w25, %cst_3 fastmath<fast> : f32
                      %w181 = arith.addf %w94, %w95 fastmath<fast> : f32
                      %w182 = arith.mulf %w181, %cst_2 fastmath<fast> : f32
                      %w183 = arith.addf %w182, %cst_1 fastmath<fast> : f32
                      %w184 = arith.mulf %w183, %w181 fastmath<fast> : f32
                      %w185 = arith.subf %w184, %w103 fastmath<fast> : f32
                      %w186 = arith.mulf %w185, %w148 fastmath<fast> : f32
                      %w187 = arith.subf %w186, %w180 fastmath<fast> : f32
                      %w188 = arith.mulf %w27, %cst_3 fastmath<fast> : f32
                      %w189 = arith.subf %w94, %w95 fastmath<fast> : f32
                      %w190 = arith.mulf %w189, %cst_2 fastmath<fast> : f32
                      %w191 = arith.addf %w190, %cst_1 fastmath<fast> : f32
                      %w192 = arith.mulf %w191, %w189 fastmath<fast> : f32
                      %w193 = arith.subf %w192, %w103 fastmath<fast> : f32
                      %w194 = arith.mulf %w193, %w148 fastmath<fast> : f32
                      %w195 = arith.subf %w194, %w188 fastmath<fast> : f32
                      %w196 = arith.mulf %w33, %cst_3 fastmath<fast> : f32
                      %w197 = arith.addf %w94, %w96 fastmath<fast> : f32
                      %w198 = arith.mulf %w197, %cst_2 fastmath<fast> : f32
                      %w199 = arith.addf %w198, %cst_1 fastmath<fast> : f32
                      %w200 = arith.mulf %w199, %w197 fastmath<fast> : f32
                      %w201 = arith.subf %w200, %w103 fastmath<fast> : f32
                      %w202 = arith.mulf %w201, %w148 fastmath<fast> : f32
                      %w203 = arith.subf %w202, %w196 fastmath<fast> : f32
                      %w204 = arith.mulf %w34, %cst_3 fastmath<fast> : f32
                      %w205 = arith.subf %w94, %w96 fastmath<fast> : f32
                      %w206 = arith.mulf %w205, %cst_2 fastmath<fast> : f32
                      %w207 = arith.addf %w206, %cst_1 fastmath<fast> : f32
                      %w208 = arith.mulf %w207, %w205 fastmath<fast> : f32
                      %w209 = arith.subf %w208, %w103 fastmath<fast> : f32
                      %w210 = arith.mulf %w209, %w148 fastmath<fast> : f32
                      %w211 = arith.subf %w210, %w204 fastmath<fast> : f32
                      %w212 = arith.mulf %w26, %cst_3 fastmath<fast> : f32
                      %w213 = arith.negf %w94 fastmath<fast> : f32
                      %w214 = arith.subf %w95, %w94 fastmath<fast> : f32
                      %w215 = arith.mulf %w214, %cst_2 fastmath<fast> : f32
                      %w216 = arith.addf %w215, %cst_1 fastmath<fast> : f32
                      %w217 = arith.mulf %w216, %w214 fastmath<fast> : f32
                      %w218 = arith.subf %w217, %w103 fastmath<fast> : f32
                      %w219 = arith.mulf %w218, %w148 fastmath<fast> : f32
                      %w220 = arith.subf %w219, %w212 fastmath<fast> : f32
                      %w221 = arith.mulf %w28, %cst_3 fastmath<fast> : f32
                      %w222 = arith.subf %w213, %w95 fastmath<fast> : f32
                      %w223 = arith.mulf %w222, %cst_2 fastmath<fast> : f32
                      %w224 = arith.addf %w223, %cst_1 fastmath<fast> : f32
                      %w225 = arith.mulf %w224, %w222 fastmath<fast> : f32
                      %w226 = arith.subf %w225, %w103 fastmath<fast> : f32
                      %w227 = arith.mulf %w226, %w148 fastmath<fast> : f32
                      %w228 = arith.subf %w227, %w221 fastmath<fast> : f32
                      %w229 = arith.mulf %w35, %cst_3 fastmath<fast> : f32
                      %w230 = arith.subf %w96, %w94 fastmath<fast> : f32
                      %w231 = arith.mulf %w230, %cst_2 fastmath<fast> : f32
                      %w232 = arith.addf %w231, %cst_1 fastmath<fast> : f32
                      %w233 = arith.mulf %w232, %w230 fastmath<fast> : f32
                      %w234 = arith.subf %w233, %w103 fastmath<fast> : f32
                      %w235 = arith.mulf %w234, %w148 fastmath<fast> : f32
                      %w236 = arith.subf %w235, %w229 fastmath<fast> : f32
                      %w237 = arith.mulf %w36, %cst_3 fastmath<fast> : f32
                      %w238 = arith.subf %w213, %w96 fastmath<fast> : f32
                      %w239 = arith.mulf %w238, %cst_2 fastmath<fast> : f32
                      %w240 = arith.addf %w239, %cst_1 fastmath<fast> : f32
                      %w241 = arith.mulf %w240, %w238 fastmath<fast> : f32
                      %w242 = arith.subf %w241, %w103 fastmath<fast> : f32
                      %w243 = arith.mulf %w242, %w148 fastmath<fast> : f32
                      %w244 = arith.subf %w243, %w237 fastmath<fast> : f32
                      scf.yield %w107, %w115, %w121, %w141, %w147, %w128, %w134, %w187, %w220, %w195, %w228, %w156, %w164, %w172, %w179, %w203, %w211, %w236, %w244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                    } else {
                      scf.yield %w18, %w20, %w19, %w22, %w21, %w24, %w23, %w28, %w27, %w26, %w25, %w32, %w31, %w30, %w29, %w36, %w35, %w34, %w33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                    }
                    %w43 = "enzymexla.pointer2memref"(%sl_dstS) : (!llvm.ptr) -> memref<?xf32>
                    affine.store %w42#0, %w43[%arg5 + %arg4 * 128] : memref<?xf32>
                    affine.store %w42#1, %w43[%arg5 + %arg4 * 128 + 2365568] : memref<?xf32>
                    affine.store %w42#2, %w43[%arg5 + %arg4 * 128 + 4730752] : memref<?xf32>
                    affine.store %w42#3, %w43[%arg5 + %arg4 * 128 + 7096321] : memref<?xf32>
                    affine.store %w42#4, %w43[%arg5 + %arg4 * 128 + 9461759] : memref<?xf32>
                    affine.store %w42#5, %w43[%arg5 + %arg4 * 128 + 11842560] : memref<?xf32>
                    affine.store %w42#6, %w43[%arg5 + %arg4 * 128 + 14177280] : memref<?xf32>
                    affine.store %w42#7, %w43[%arg5 + %arg4 * 128 + 16558209] : memref<?xf32>
                    affine.store %w42#8, %w43[%arg5 + %arg4 * 128 + 18923647] : memref<?xf32>
                    affine.store %w42#9, %w43[%arg5 + %arg4 * 128 + 21288833] : memref<?xf32>
                    affine.store %w42#10, %w43[%arg5 + %arg4 * 128 + 23654271] : memref<?xf32>
                    affine.store %w42#11, %w43[%arg5 + %arg4 * 128 + 26035328] : memref<?xf32>
                    affine.store %w42#12, %w43[%arg5 + %arg4 * 128 + 28370048] : memref<?xf32>
                    affine.store %w42#13, %w43[%arg5 + %arg4 * 128 + 30765952] : memref<?xf32>
                    affine.store %w42#14, %w43[%arg5 + %arg4 * 128 + 33100672] : memref<?xf32>
                    affine.store %w42#15, %w43[%arg5 + %arg4 * 128 + 35496961] : memref<?xf32>
                    affine.store %w42#16, %w43[%arg5 + %arg4 * 128 + 37831681] : memref<?xf32>
                    affine.store %w42#17, %w43[%arg5 + %arg4 * 128 + 40227839] : memref<?xf32>
                    affine.store %w42#18, %w43[%arg5 + %arg4 * 128 + 42562559] : memref<?xf32>
                  }
                  "enzymexla.polygeist_yield"() : () -> ()
                }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
            }
          }
        }
      }
      %sl_err = llvm.call @cudaGetLastError() {no_unwind} : () -> i32
      %sl_bad = arith.cmpi ne, %sl_err, %c0_i32 : i32
      scf.if %sl_bad {
        %sl_e13 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
        %sl_e14 = affine.load %sl_e13[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
        %sl_e15 = llvm.call @cudaGetErrorString(%sl_err) {no_unwind} : (i32 {llvm.noundef}) -> !llvm.ptr
        %sl_e16 = llvm.call @fprintf(%sl_e14, %0, %c51_i32, %sl_e15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
        llvm.call @exit(%c-1_i32) {cold, no_unwind, noreturn} : (i32 {llvm.noundef}) -> ()
      }
      scf.yield %c0_i32 : i32
    }
    cf.switch %5 : i32, [
      default: ^bb2,
      0: ^bb1
    ]
  ^bb1:  // pred: ^bb0
    llvm.return
  ^bb2:  // pred: ^bb0
    llvm.unreachable
  }
  llvm.func local_unnamed_addr @CUDA_LBM_kernel_loop(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readnone}, %arg3: !llvm.ptr {llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readnone}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i32 = arith.constant -1 : i32
    %c51_i32 = arith.constant 51 : i32
    %0 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %false = arith.constant false
    %c2 = arith.constant 2 : index
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = ub.poison : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.0541666672 : f32
    %cst_0 = arith.constant -3.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %cst_2 = arith.constant 4.500000e+00 : f32
    %cst_3 = arith.constant 0.950000047 : f32
    %cst_4 = arith.constant 0.108333334 : f32
    %cst_5 = arith.constant -0.950000047 : f32
    %cst_6 = arith.constant 0.650000035 : f32
    %cst_7 = arith.constant -1.000000e+00 : f32
    %cst_8 = arith.constant 1.500000e+00 : f32
    %cst_9 = arith.constant 0.000000e+00 : f32
    %cst_10 = arith.constant 2.000000e-03 : f32
    %cst_11 = arith.constant 5.000000e-03 : f32
    %c1 = arith.constant 1 : index
    %c150 = arith.constant 150 : index
    %c120 = arith.constant 120 : index
    %2 = llvm.mlir.addressof @stderr : !llvm.ptr
    %3 = arith.addi %arg0, %c1_i32 : i32
    %4 = arith.cmpi ult, %3, %c3_i32 : i32
    %5 = scf.if %4 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %6 = arith.divsi %arg0, %c2_i32 : i32
      %7 = arith.maxui %6, %c1_i32 : i32
      %c5 = arith.constant 5 : index
      %sl_tc = arith.index_castui %7 : i32 to index
      %sl_H = arith.muli %sl_tc, %c2 : index
      %sl_k = arith.constant 4 : index
      %sl_ns = arith.constant 30 : index
      %sl_nw = arith.constant 33 : index
      %sl_z = arith.constant 0 : index
      %sl_rows = arith.constant 76800 : index
      scf.for %sl_b = %sl_z to %sl_H step %sl_k {
        %sl_rem = arith.subi %sl_H, %sl_b : index
        %sl_kk = arith.minui %sl_rem, %sl_k : index
        scf.for %sl_w = %sl_z to %sl_nw step %c1 {
          scf.for %sl_j = %sl_z to %sl_kk step %c1 {
            %sl_i = arith.subi %sl_w, %sl_j : index
            %sl_ge = arith.cmpi sge, %sl_i, %sl_z : index
            %sl_lt = arith.cmpi slt, %sl_i, %sl_ns : index
            %sl_ok = arith.andi %sl_ge, %sl_lt : i1
            scf.if %sl_ok {
              %sl_s = arith.addi %sl_b, %sl_j : index
              %sl_par = arith.remui %sl_s, %c2 : index
              %sl_even = arith.cmpi eq, %sl_par, %sl_z : index
              %sl_src = llvm.select %sl_even, %arg1, %arg3 : i1, !llvm.ptr
              %sl_dst = llvm.select %sl_even, %arg3, %arg1 : i1, !llvm.ptr
              %sl_off = arith.muli %sl_i, %sl_rows : index
              %sl_o64 = arith.index_cast %sl_off : index to i64
              %sl_srcS = llvm.getelementptr %sl_src[%sl_o64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %sl_dstS = llvm.getelementptr %sl_dst[%sl_o64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %sl_wr = "enzymexla.gpu_wrapper"(%c120, %c5, %c1, %c120, %c1, %c1) ({
                  affine.parallel (%arg6, %arg7) = (0, 0) to (600, 120) {
                    llvm.intr.experimental.noalias.scope.decl #alias_scope4
                    llvm.intr.experimental.noalias.scope.decl #alias_scope5
                    %w17 = "enzymexla.pointer2memref"(%sl_srcS) : (!llvm.ptr) -> memref<?xf32>
                    %w18 = affine.load %w17[%arg7 + %arg6 * 128] : memref<?xf32>
                    %w19 = affine.load %w17[%arg7 + %arg6 * 128 + 2365440] : memref<?xf32>
                    %w20 = affine.load %w17[%arg7 + %arg6 * 128 + 4730880] : memref<?xf32>
                    %w21 = affine.load %w17[%arg7 + %arg6 * 128 + 7096320] : memref<?xf32>
                    %w22 = affine.load %w17[%arg7 + %arg6 * 128 + 9461760] : memref<?xf32>
                    %w23 = affine.load %w17[%arg7 + %arg6 * 128 + 11827200] : memref<?xf32>
                    %w24 = affine.load %w17[%arg7 + %arg6 * 128 + 14192640] : memref<?xf32>
                    %w25 = affine.load %w17[%arg7 + %arg6 * 128 + 16558080] : memref<?xf32>
                    %w26 = affine.load %w17[%arg7 + %arg6 * 128 + 18923520] : memref<?xf32>
                    %w27 = affine.load %w17[%arg7 + %arg6 * 128 + 21288960] : memref<?xf32>
                    %w28 = affine.load %w17[%arg7 + %arg6 * 128 + 23654400] : memref<?xf32>
                    %w29 = affine.load %w17[%arg7 + %arg6 * 128 + 26019840] : memref<?xf32>
                    %w30 = affine.load %w17[%arg7 + %arg6 * 128 + 28385280] : memref<?xf32>
                    %w31 = affine.load %w17[%arg7 + %arg6 * 128 + 30750720] : memref<?xf32>
                    %w32 = affine.load %w17[%arg7 + %arg6 * 128 + 33116160] : memref<?xf32>
                    %w33 = affine.load %w17[%arg7 + %arg6 * 128 + 35481600] : memref<?xf32>
                    %w34 = affine.load %w17[%arg7 + %arg6 * 128 + 37847040] : memref<?xf32>
                    %w35 = affine.load %w17[%arg7 + %arg6 * 128 + 40212480] : memref<?xf32>
                    %w36 = affine.load %w17[%arg7 + %arg6 * 128 + 42577920] : memref<?xf32>
                    %w37 = "enzymexla.pointer2memref"(%sl_srcS) : (!llvm.ptr) -> memref<?xi8>
                    %w38 = affine.load %w37[%arg7 * 4 + %arg6 * 512 + 179773440] : memref<?xi8>
                    %w39 = arith.extui %w38 : i8 to i32
                    %w40 = arith.andi %w39, %c1_i32 : i32
                    %w41 = arith.cmpi eq, %w40, %c0_i32 : i32
                    %w42:19 = scf.if %w41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                      %w44 = arith.addf %w19, %w18 fastmath<fast> : f32
                      %w45 = arith.addf %w44, %w20 fastmath<fast> : f32
                      %w46 = arith.addf %w45, %w21 fastmath<fast> : f32
                      %w47 = arith.addf %w46, %w22 fastmath<fast> : f32
                      %w48 = arith.addf %w47, %w23 fastmath<fast> : f32
                      %w49 = arith.addf %w48, %w24 fastmath<fast> : f32
                      %w50 = arith.addf %w49, %w25 fastmath<fast> : f32
                      %w51 = arith.addf %w50, %w26 fastmath<fast> : f32
                      %w52 = arith.addf %w51, %w27 fastmath<fast> : f32
                      %w53 = arith.addf %w52, %w28 fastmath<fast> : f32
                      %w54 = arith.addf %w53, %w29 fastmath<fast> : f32
                      %w55 = arith.addf %w54, %w30 fastmath<fast> : f32
                      %w56 = arith.addf %w55, %w31 fastmath<fast> : f32
                      %w57 = arith.addf %w56, %w32 fastmath<fast> : f32
                      %w58 = arith.addf %w57, %w33 fastmath<fast> : f32
                      %w59 = arith.addf %w58, %w34 fastmath<fast> : f32
                      %w60 = arith.addf %w59, %w35 fastmath<fast> : f32
                      %w61 = arith.addf %w60, %w36 fastmath<fast> : f32
                      %w62 = arith.addf %w21, %w25 fastmath<fast> : f32
                      %w63 = arith.addf %w22, %w26 fastmath<fast> : f32
                      %w64 = arith.addf %w62, %w27 fastmath<fast> : f32
                      %w65 = arith.addf %w63, %w28 fastmath<fast> : f32
                      %w66 = arith.subf %w64, %w65 fastmath<fast> : f32
                      %w67 = arith.addf %w66, %w33 fastmath<fast> : f32
                      %w68 = arith.addf %w67, %w34 fastmath<fast> : f32
                      %w69 = arith.addf %w35, %w36 fastmath<fast> : f32
                      %w70 = arith.subf %w68, %w69 fastmath<fast> : f32
                      %w71 = arith.subf %w19, %w20 fastmath<fast> : f32
                      %w72 = arith.addf %w71, %w25 fastmath<fast> : f32
                      %w73 = arith.addf %w72, %w26 fastmath<fast> : f32
                      %w74 = arith.addf %w27, %w28 fastmath<fast> : f32
                      %w75 = arith.subf %w73, %w74 fastmath<fast> : f32
                      %w76 = arith.addf %w75, %w29 fastmath<fast> : f32
                      %w77 = arith.addf %w76, %w30 fastmath<fast> : f32
                      %w78 = arith.addf %w31, %w32 fastmath<fast> : f32
                      %w79 = arith.subf %w77, %w78 fastmath<fast> : f32
                      %w80 = arith.addf %w23, %w29 fastmath<fast> : f32
                      %w81 = arith.addf %w24, %w30 fastmath<fast> : f32
                      %w82 = arith.addf %w80, %w31 fastmath<fast> : f32
                      %w83 = arith.addf %w81, %w32 fastmath<fast> : f32
                      %w84 = arith.addf %w82, %w33 fastmath<fast> : f32
                      %w85 = arith.addf %w83, %w34 fastmath<fast> : f32
                      %w86 = arith.addf %w84, %w35 fastmath<fast> : f32
                      %w87 = arith.addf %w85, %w36 fastmath<fast> : f32
                      %w88 = arith.subf %w86, %w87 fastmath<fast> : f32
                      %w89 = arith.divf %w70, %w61 fastmath<fast> : f32
                      %w90 = arith.divf %w79, %w61 fastmath<fast> : f32
                      %w91 = arith.divf %w88, %w61 fastmath<fast> : f32
                      %w92 = arith.andi %w39, %c2_i32 : i32
                      %w93 = arith.cmpi eq, %w92, %c0_i32 : i32
                      %w94 = arith.select %w93, %w89, %cst_11 : f32
                      %w95 = arith.select %w93, %w90, %cst_10 : f32
                      %w96 = arith.select %w93, %w91, %cst_9 : f32
                      %w97 = arith.mulf %w94, %w94 fastmath<fast> : f32
                      %w98 = arith.mulf %w95, %w95 fastmath<fast> : f32
                      %w99 = arith.addf %w97, %w98 fastmath<fast> : f32
                      %w100 = arith.mulf %w96, %w96 fastmath<fast> : f32
                      %w101 = arith.addf %w99, %w100 fastmath<fast> : f32
                      %w102 = arith.mulf %w101, %cst_8 fastmath<fast> : f32
                      %w103 = arith.addf %w102, %cst_7 fastmath<fast> : f32
                      %w104 = arith.mulf %w61, %cst_6 fastmath<fast> : f32
                      %w105 = arith.mulf %w18, %cst_5 fastmath<fast> : f32
                      %w106 = arith.mulf %w104, %w103 fastmath<fast> : f32
                      %w107 = arith.subf %w105, %w106 fastmath<fast> : f32
                      %w108 = arith.mulf %w61, %cst_4 fastmath<fast> : f32
                      %w109 = arith.mulf %w19, %cst_3 fastmath<fast> : f32
                      %w110 = arith.mulf %w95, %cst_2 fastmath<fast> : f32
                      %w111 = arith.addf %w110, %cst_1 fastmath<fast> : f32
                      %w112 = arith.mulf %w111, %w95 fastmath<fast> : f32
                      %w113 = arith.subf %w112, %w103 fastmath<fast> : f32
                      %w114 = arith.mulf %w113, %w108 fastmath<fast> : f32
                      %w115 = arith.subf %w114, %w109 fastmath<fast> : f32
                      %w116 = arith.mulf %w20, %cst_3 fastmath<fast> : f32
                      %w117 = arith.addf %w110, %cst_0 fastmath<fast> : f32
                      %w118 = arith.mulf %w117, %w95 fastmath<fast> : f32
                      %w119 = arith.subf %w118, %w103 fastmath<fast> : f32
                      %w120 = arith.mulf %w119, %w108 fastmath<fast> : f32
                      %w121 = arith.subf %w120, %w116 fastmath<fast> : f32
                      %w122 = arith.mulf %w23, %cst_3 fastmath<fast> : f32
                      %w123 = arith.mulf %w96, %cst_2 fastmath<fast> : f32
                      %w124 = arith.addf %w123, %cst_1 fastmath<fast> : f32
                      %w125 = arith.mulf %w124, %w96 fastmath<fast> : f32
                      %w126 = arith.subf %w125, %w103 fastmath<fast> : f32
                      %w127 = arith.mulf %w126, %w108 fastmath<fast> : f32
                      %w128 = arith.subf %w127, %w122 fastmath<fast> : f32
                      %w129 = arith.mulf %w24, %cst_3 fastmath<fast> : f32
                      %w130 = arith.addf %w123, %cst_0 fastmath<fast> : f32
                      %w131 = arith.mulf %w130, %w96 fastmath<fast> : f32
                      %w132 = arith.subf %w131, %w103 fastmath<fast> : f32
                      %w133 = arith.mulf %w132, %w108 fastmath<fast> : f32
                      %w134 = arith.subf %w133, %w129 fastmath<fast> : f32
                      %w135 = arith.mulf %w21, %cst_3 fastmath<fast> : f32
                      %w136 = arith.mulf %w94, %cst_2 fastmath<fast> : f32
                      %w137 = arith.addf %w136, %cst_1 fastmath<fast> : f32
                      %w138 = arith.mulf %w137, %w94 fastmath<fast> : f32
                      %w139 = arith.subf %w138, %w103 fastmath<fast> : f32
                      %w140 = arith.mulf %w139, %w108 fastmath<fast> : f32
                      %w141 = arith.subf %w140, %w135 fastmath<fast> : f32
                      %w142 = arith.mulf %w22, %cst_3 fastmath<fast> : f32
                      %w143 = arith.addf %w136, %cst_0 fastmath<fast> : f32
                      %w144 = arith.mulf %w143, %w94 fastmath<fast> : f32
                      %w145 = arith.subf %w144, %w103 fastmath<fast> : f32
                      %w146 = arith.mulf %w145, %w108 fastmath<fast> : f32
                      %w147 = arith.subf %w146, %w142 fastmath<fast> : f32
                      %w148 = arith.mulf %w61, %cst fastmath<fast> : f32
                      %w149 = arith.mulf %w29, %cst_3 fastmath<fast> : f32
                      %w150 = arith.addf %w95, %w96 fastmath<fast> : f32
                      %w151 = arith.mulf %w150, %cst_2 fastmath<fast> : f32
                      %w152 = arith.addf %w151, %cst_1 fastmath<fast> : f32
                      %w153 = arith.mulf %w152, %w150 fastmath<fast> : f32
                      %w154 = arith.subf %w153, %w103 fastmath<fast> : f32
                      %w155 = arith.mulf %w154, %w148 fastmath<fast> : f32
                      %w156 = arith.subf %w155, %w149 fastmath<fast> : f32
                      %w157 = arith.mulf %w30, %cst_3 fastmath<fast> : f32
                      %w158 = arith.subf %w95, %w96 fastmath<fast> : f32
                      %w159 = arith.mulf %w158, %cst_2 fastmath<fast> : f32
                      %w160 = arith.addf %w159, %cst_1 fastmath<fast> : f32
                      %w161 = arith.mulf %w160, %w158 fastmath<fast> : f32
                      %w162 = arith.subf %w161, %w103 fastmath<fast> : f32
                      %w163 = arith.mulf %w162, %w148 fastmath<fast> : f32
                      %w164 = arith.subf %w163, %w157 fastmath<fast> : f32
                      %w165 = arith.mulf %w31, %cst_3 fastmath<fast> : f32
                      %w166 = arith.subf %w96, %w95 fastmath<fast> : f32
                      %w167 = arith.mulf %w166, %cst_2 fastmath<fast> : f32
                      %w168 = arith.addf %w167, %cst_1 fastmath<fast> : f32
                      %w169 = arith.mulf %w168, %w166 fastmath<fast> : f32
                      %w170 = arith.subf %w169, %w103 fastmath<fast> : f32
                      %w171 = arith.mulf %w170, %w148 fastmath<fast> : f32
                      %w172 = arith.subf %w171, %w165 fastmath<fast> : f32
                      %w173 = arith.mulf %w32, %cst_3 fastmath<fast> : f32
                      %w174 = arith.negf %w150 fastmath<fast> : f32
                      %w175 = arith.subf %cst_1, %w151 fastmath<fast> : f32
                      %w176 = arith.mulf %w175, %w174 fastmath<fast> : f32
                      %w177 = arith.subf %w176, %w103 fastmath<fast> : f32
                      %w178 = arith.mulf %w177, %w148 fastmath<fast> : f32
                      %w179 = arith.subf %w178, %w173 fastmath<fast> : f32
                      %w180 = arith.mulf %w25, %cst_3 fastmath<fast> : f32
                      %w181 = arith.addf %w94, %w95 fastmath<fast> : f32
                      %w182 = arith.mulf %w181, %cst_2 fastmath<fast> : f32
                      %w183 = arith.addf %w182, %cst_1 fastmath<fast> : f32
                      %w184 = arith.mulf %w183, %w181 fastmath<fast> : f32
                      %w185 = arith.subf %w184, %w103 fastmath<fast> : f32
                      %w186 = arith.mulf %w185, %w148 fastmath<fast> : f32
                      %w187 = arith.subf %w186, %w180 fastmath<fast> : f32
                      %w188 = arith.mulf %w27, %cst_3 fastmath<fast> : f32
                      %w189 = arith.subf %w94, %w95 fastmath<fast> : f32
                      %w190 = arith.mulf %w189, %cst_2 fastmath<fast> : f32
                      %w191 = arith.addf %w190, %cst_1 fastmath<fast> : f32
                      %w192 = arith.mulf %w191, %w189 fastmath<fast> : f32
                      %w193 = arith.subf %w192, %w103 fastmath<fast> : f32
                      %w194 = arith.mulf %w193, %w148 fastmath<fast> : f32
                      %w195 = arith.subf %w194, %w188 fastmath<fast> : f32
                      %w196 = arith.mulf %w33, %cst_3 fastmath<fast> : f32
                      %w197 = arith.addf %w94, %w96 fastmath<fast> : f32
                      %w198 = arith.mulf %w197, %cst_2 fastmath<fast> : f32
                      %w199 = arith.addf %w198, %cst_1 fastmath<fast> : f32
                      %w200 = arith.mulf %w199, %w197 fastmath<fast> : f32
                      %w201 = arith.subf %w200, %w103 fastmath<fast> : f32
                      %w202 = arith.mulf %w201, %w148 fastmath<fast> : f32
                      %w203 = arith.subf %w202, %w196 fastmath<fast> : f32
                      %w204 = arith.mulf %w34, %cst_3 fastmath<fast> : f32
                      %w205 = arith.subf %w94, %w96 fastmath<fast> : f32
                      %w206 = arith.mulf %w205, %cst_2 fastmath<fast> : f32
                      %w207 = arith.addf %w206, %cst_1 fastmath<fast> : f32
                      %w208 = arith.mulf %w207, %w205 fastmath<fast> : f32
                      %w209 = arith.subf %w208, %w103 fastmath<fast> : f32
                      %w210 = arith.mulf %w209, %w148 fastmath<fast> : f32
                      %w211 = arith.subf %w210, %w204 fastmath<fast> : f32
                      %w212 = arith.mulf %w26, %cst_3 fastmath<fast> : f32
                      %w213 = arith.negf %w94 fastmath<fast> : f32
                      %w214 = arith.subf %w95, %w94 fastmath<fast> : f32
                      %w215 = arith.mulf %w214, %cst_2 fastmath<fast> : f32
                      %w216 = arith.addf %w215, %cst_1 fastmath<fast> : f32
                      %w217 = arith.mulf %w216, %w214 fastmath<fast> : f32
                      %w218 = arith.subf %w217, %w103 fastmath<fast> : f32
                      %w219 = arith.mulf %w218, %w148 fastmath<fast> : f32
                      %w220 = arith.subf %w219, %w212 fastmath<fast> : f32
                      %w221 = arith.mulf %w28, %cst_3 fastmath<fast> : f32
                      %w222 = arith.subf %w213, %w95 fastmath<fast> : f32
                      %w223 = arith.mulf %w222, %cst_2 fastmath<fast> : f32
                      %w224 = arith.addf %w223, %cst_1 fastmath<fast> : f32
                      %w225 = arith.mulf %w224, %w222 fastmath<fast> : f32
                      %w226 = arith.subf %w225, %w103 fastmath<fast> : f32
                      %w227 = arith.mulf %w226, %w148 fastmath<fast> : f32
                      %w228 = arith.subf %w227, %w221 fastmath<fast> : f32
                      %w229 = arith.mulf %w35, %cst_3 fastmath<fast> : f32
                      %w230 = arith.subf %w96, %w94 fastmath<fast> : f32
                      %w231 = arith.mulf %w230, %cst_2 fastmath<fast> : f32
                      %w232 = arith.addf %w231, %cst_1 fastmath<fast> : f32
                      %w233 = arith.mulf %w232, %w230 fastmath<fast> : f32
                      %w234 = arith.subf %w233, %w103 fastmath<fast> : f32
                      %w235 = arith.mulf %w234, %w148 fastmath<fast> : f32
                      %w236 = arith.subf %w235, %w229 fastmath<fast> : f32
                      %w237 = arith.mulf %w36, %cst_3 fastmath<fast> : f32
                      %w238 = arith.subf %w213, %w96 fastmath<fast> : f32
                      %w239 = arith.mulf %w238, %cst_2 fastmath<fast> : f32
                      %w240 = arith.addf %w239, %cst_1 fastmath<fast> : f32
                      %w241 = arith.mulf %w240, %w238 fastmath<fast> : f32
                      %w242 = arith.subf %w241, %w103 fastmath<fast> : f32
                      %w243 = arith.mulf %w242, %w148 fastmath<fast> : f32
                      %w244 = arith.subf %w243, %w237 fastmath<fast> : f32
                      scf.yield %w107, %w115, %w121, %w141, %w147, %w128, %w134, %w187, %w220, %w195, %w228, %w156, %w164, %w172, %w179, %w203, %w211, %w236, %w244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                    } else {
                      scf.yield %w18, %w20, %w19, %w22, %w21, %w24, %w23, %w28, %w27, %w26, %w25, %w32, %w31, %w30, %w29, %w36, %w35, %w34, %w33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                    }
                    %w43 = "enzymexla.pointer2memref"(%sl_dstS) : (!llvm.ptr) -> memref<?xf32>
                    affine.store %w42#0, %w43[%arg7 + %arg6 * 128] : memref<?xf32>
                    affine.store %w42#1, %w43[%arg7 + %arg6 * 128 + 2365568] : memref<?xf32>
                    affine.store %w42#2, %w43[%arg7 + %arg6 * 128 + 4730752] : memref<?xf32>
                    affine.store %w42#3, %w43[%arg7 + %arg6 * 128 + 7096321] : memref<?xf32>
                    affine.store %w42#4, %w43[%arg7 + %arg6 * 128 + 9461759] : memref<?xf32>
                    affine.store %w42#5, %w43[%arg7 + %arg6 * 128 + 11842560] : memref<?xf32>
                    affine.store %w42#6, %w43[%arg7 + %arg6 * 128 + 14177280] : memref<?xf32>
                    affine.store %w42#7, %w43[%arg7 + %arg6 * 128 + 16558209] : memref<?xf32>
                    affine.store %w42#8, %w43[%arg7 + %arg6 * 128 + 18923647] : memref<?xf32>
                    affine.store %w42#9, %w43[%arg7 + %arg6 * 128 + 21288833] : memref<?xf32>
                    affine.store %w42#10, %w43[%arg7 + %arg6 * 128 + 23654271] : memref<?xf32>
                    affine.store %w42#11, %w43[%arg7 + %arg6 * 128 + 26035328] : memref<?xf32>
                    affine.store %w42#12, %w43[%arg7 + %arg6 * 128 + 28370048] : memref<?xf32>
                    affine.store %w42#13, %w43[%arg7 + %arg6 * 128 + 30765952] : memref<?xf32>
                    affine.store %w42#14, %w43[%arg7 + %arg6 * 128 + 33100672] : memref<?xf32>
                    affine.store %w42#15, %w43[%arg7 + %arg6 * 128 + 35496961] : memref<?xf32>
                    affine.store %w42#16, %w43[%arg7 + %arg6 * 128 + 37831681] : memref<?xf32>
                    affine.store %w42#17, %w43[%arg7 + %arg6 * 128 + 40227839] : memref<?xf32>
                    affine.store %w42#18, %w43[%arg7 + %arg6 * 128 + 42562559] : memref<?xf32>
                  }
                  "enzymexla.polygeist_yield"() : () -> ()
                }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
            }
          }
        }
      }
      %sl_err = llvm.call @cudaGetLastError() {no_unwind} : () -> i32
      %sl_bad = arith.cmpi ne, %sl_err, %c0_i32 : i32
      scf.if %sl_bad {
        %sl_e13 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
        %sl_e14 = affine.load %sl_e13[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
        %sl_e15 = llvm.call @cudaGetErrorString(%sl_err) {no_unwind} : (i32 {llvm.noundef}) -> !llvm.ptr
        %sl_e16 = llvm.call @fprintf(%sl_e14, %0, %c51_i32, %sl_e15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
        llvm.call @exit(%c-1_i32) {cold, no_unwind, noreturn} : (i32 {llvm.noundef}) -> ()
      }
      scf.yield %c0_i32 : i32
    }
    cf.switch %5 : i32, [
      default: ^bb2,
      0: ^bb1
    ]
  ^bb1:  // pred: ^bb0
    llvm.return
  ^bb2:  // pred: ^bb0
    llvm.unreachable
  }
  llvm.func local_unnamed_addr @LBM_allocateGrid(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    %cst = arith.constant 185.15625 : f64
    %1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %c194150400_i64 = arith.constant 194150400 : i64
    %c1_i64 = arith.constant 1 : i64
    %3 = llvm.call @calloc(%c1_i64, %c194150400_i64) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite, errnoMem = write, targetMem0 = none, targetMem1 = none>} : (i64, i64) -> (!llvm.ptr {llvm.dereferenceable_or_null = 194150400 : i64})
    %4 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    affine.store %3, %4[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %5 = llvm.icmp "eq" %3, %2 : !llvm.ptr
    cf.cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %6 = llvm.call tail @printf(%0, %cst) vararg(!llvm.func<i32 (ptr, ...)>) {uniform_work_group_size} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}) -> i32
    llvm.call tail @exit(%c1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    %7 = llvm.call tail @printf(%1, %cst) vararg(!llvm.func<i32 (ptr, ...)>) {uniform_work_group_size} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}) -> i32
    %8 = affine.load %4[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %9 = llvm.getelementptr inbounds|nuw %8[122880] : (!llvm.ptr) -> !llvm.ptr, i8
    affine.store %9, %4[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    llvm.return
  }
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @exit(i32 {llvm.noundef}) attributes {no_unwind, noreturn, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @CUDA_LBM_allocateGrid(%arg0: !llvm.ptr {llvm.noundef}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i32 = arith.constant -1 : i32
    %c229_i32 = arith.constant 229 : i32
    %0 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %1 = llvm.mlir.addressof @stderr : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %memref = gpu.alloc  () : memref<194150400xi8, 1>
    %2 = "enzymexla.memref2pointer"(%memref) : (memref<194150400xi8, 1>) -> !llvm.ptr
    %3 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    affine.store %2, %3[0] : memref<?x!llvm.ptr>
    %4 = llvm.call tail @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
    %5 = arith.cmpi eq, %4, %c0_i32 : i32
    cf.cond_br %5, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %6 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %7 = affine.load %6[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %8 = llvm.call tail @cudaGetErrorString(%4) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
    %9 = llvm.call tail @fprintf(%7, %0, %c229_i32, %8) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    llvm.call tail @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    %10 = affine.load %3[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %11 = llvm.getelementptr inbounds|nuw %10[122880] : (!llvm.ptr) -> !llvm.ptr, i8
    affine.store %11, %3[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    llvm.return
  }
  llvm.func local_unnamed_addr @cudaGetLastError() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @fprintf(!llvm.ptr {llvm.nocapture, llvm.noundef}, !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @cudaGetErrorString(i32 {llvm.noundef}) -> !llvm.ptr attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @LBM_freeGrid(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}) attributes {dso_local, memory_effects = #llvm.memory_effects<other = readwrite, argMem = readwrite, inaccessibleMem = readwrite, errnoMem = readwrite, targetMem0 = none, targetMem1 = none>, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>, will_return} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %2 = affine.load %1[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %3 = llvm.getelementptr inbounds %2[-122880] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.call tail @free(%3) {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
    affine.store %0, %1[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    llvm.return
  }
  llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["mustprogress", ["allockind", "4"], ["alloc-family", "malloc"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size, will_return}
  llvm.func local_unnamed_addr @CUDA_LBM_freeGrid(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %2 = affine.load %1[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %3 = llvm.getelementptr inbounds %2[-122880] : (!llvm.ptr) -> !llvm.ptr, i8
    %4 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<?xi8, 1>
    gpu.dealloc  %4 : memref<?xi8, 1>
    affine.store %0, %1[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    llvm.return
  }
  llvm.func local_unnamed_addr @LBM_initializeGrid(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.writeonly}) attributes {dso_local, memory_effects = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %cst = arith.constant 0.333333343 : f32
    %cst_0 = arith.constant 0.055555556 : f32
    %cst_1 = arith.constant 0.027777778 : f32
    %c0_i8 = arith.constant 0 : i8
    affine.for %arg1 = 0 to 150 {
      affine.for %arg2 = 0 to 120 {
        affine.parallel (%arg3) = (0) to (120) {
          %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
          %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
          affine.store %cst, %0[%arg3 + %arg2 * 128 + %arg1 * 15360] : memref<?xf32>
          affine.store %cst_0, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 2365440] : memref<?xf32>
          affine.store %cst_0, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 4730880] : memref<?xf32>
          affine.store %cst_0, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 7096320] : memref<?xf32>
          affine.store %cst_0, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 9461760] : memref<?xf32>
          affine.store %cst_0, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 11827200] : memref<?xf32>
          affine.store %cst_0, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 14192640] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 16558080] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 18923520] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 21288960] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 23654400] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 26019840] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 28385280] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 30750720] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 33116160] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 35481600] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 37847040] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 40212480] : memref<?xf32>
          affine.store %cst_1, %0[%arg3 + %arg2 * 128 + %arg1 * 15360 + 42577920] : memref<?xf32>
          affine.store %c0_i8, %1[%arg2 * 512 + %arg3 * 4 + %arg1 * 61440 + 179773440] : memref<?xi8>
        }
      }
    }
    llvm.return
  }
  llvm.func local_unnamed_addr @CUDA_LBM_initializeGrid(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i32 = arith.constant -1 : i32
    %c283_i32 = arith.constant 283 : i32
    %0 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %1 = llvm.mlir.addressof @stderr : !llvm.ptr
    %c194150400 = arith.constant 194150400 : index
    %c0_i32 = arith.constant 0 : i32
    %2 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %3 = affine.load %2[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %4 = llvm.getelementptr inbounds %3[-122880] : (!llvm.ptr) -> !llvm.ptr, i8
    %5 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %6 = affine.load %5[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %7 = llvm.getelementptr inbounds %6[-122880] : (!llvm.ptr) -> !llvm.ptr, i8
    %8 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<?xi8, 1>
    %9 = "enzymexla.pointer2memref"(%7) : (!llvm.ptr) -> memref<?xi8>
    enzymexla.memcpy  %8, %9, %c194150400 : memref<?xi8, 1>, memref<?xi8>
    %10 = llvm.call tail @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
    %11 = arith.cmpi eq, %10, %c0_i32 : i32
    cf.cond_br %11, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %12 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %13 = affine.load %12[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %14 = llvm.call tail @cudaGetErrorString(%10) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
    %15 = llvm.call tail @fprintf(%13, %0, %c283_i32, %14) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    llvm.call tail @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    llvm.return
  }
  llvm.func local_unnamed_addr @CUDA_LBM_getDeviceGrid(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c291_i32 = arith.constant 291 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c289_i32 = arith.constant 289 : i32
    %0 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %1 = llvm.mlir.addressof @stderr : !llvm.ptr
    %c194150400 = arith.constant 194150400 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %2 = llvm.call tail @cudaThreadSynchronize() {no_unwind, uniform_work_group_size} : () -> i32
    %3 = llvm.call tail @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
    %4 = arith.cmpi eq, %3, %c0_i32 : i32
    %5 = scf.if %4 -> (i32) {
      %6 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      %7 = affine.load %6[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
      %8 = llvm.getelementptr inbounds %7[-122880] : (!llvm.ptr) -> !llvm.ptr, i8
      %9 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      %10 = affine.load %9[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
      %11 = llvm.getelementptr inbounds %10[-122880] : (!llvm.ptr) -> !llvm.ptr, i8
      %12 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<?xi8>
      %13 = "enzymexla.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi8, 1>
      enzymexla.memcpy  %12, %13, %c194150400 : memref<?xi8>, memref<?xi8, 1>
      %14 = llvm.call tail @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
      %15 = arith.cmpi ne, %14, %c0_i32 : i32
      %16 = arith.extui %15 : i1 to i32
      scf.if %15 {
        %17 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
        %18 = affine.load %17[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
        %19 = llvm.call tail @cudaGetErrorString(%14) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
        %20 = llvm.call tail @fprintf(%18, %0, %c291_i32, %19) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
        llvm.call tail @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
      }
      scf.yield %16 : i32
    } else {
      %6 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      %7 = affine.load %6[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
      %8 = llvm.call tail @cudaGetErrorString(%3) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
      %9 = llvm.call tail @fprintf(%7, %0, %c289_i32, %8) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
      llvm.call tail @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
      scf.yield %c1_i32 : i32
    }
    cf.switch %5 : i32, [
      default: ^bb1,
      0: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    llvm.return
  }
  llvm.func local_unnamed_addr @cudaThreadSynchronize() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @LBM_swapGrids(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}) attributes {dso_local, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>, will_return} {
    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %1 = affine.load %0[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %3 = affine.load %2[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    affine.store %3, %0[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    affine.store %1, %2[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    llvm.return
  }
  llvm.func local_unnamed_addr @LBM_loadObstacleFile(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %c46_i32 = arith.constant 46 : i32
    %c1_i8 = arith.constant 1 : i8
    %1 = llvm.call tail @fopen(%arg1, %0) {uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noalias})
    affine.for %arg2 = 0 to 150 {
      affine.for %arg3 = 0 to 120 {
        affine.for %arg4 = 0 to 120 {
          %5 = llvm.call tail @fgetc(%1) {uniform_work_group_size} : (!llvm.ptr {llvm.noundef}) -> i32
          %6 = arith.cmpi ne, %5, %c46_i32 : i32
          scf.if %6 {
            %7 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
            %8 = affine.load %7[%arg3 * 512 + %arg4 * 4 + %arg2 * 61440 + 179773440] : memref<?xi8>
            %9 = arith.ori %8, %c1_i8 : i8
            affine.store %9, %7[%arg3 * 512 + %arg4 * 4 + %arg2 * 61440 + 179773440] : memref<?xi8>
          }
        }
        %4 = llvm.call tail @fgetc(%1) {uniform_work_group_size} : (!llvm.ptr {llvm.noundef}) -> i32
      }
      %3 = llvm.call tail @fgetc(%1) {uniform_work_group_size} : (!llvm.ptr {llvm.noundef}) -> i32
    }
    %2 = llvm.call tail @fclose(%1) {uniform_work_group_size} : (!llvm.ptr {llvm.noundef}) -> i32
    llvm.return
  }
  llvm.func local_unnamed_addr @fopen(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @fgetc(!llvm.ptr {llvm.nocapture, llvm.noundef}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @fclose(!llvm.ptr {llvm.nocapture, llvm.noundef}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @LBM_initializeSpecialCellsForLDC(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}) attributes {dso_local, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c-2_i64 = arith.constant -2 : i64
    %c0 = arith.constant 0 : index
    %c119 = arith.constant 119 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c149_i64 = arith.constant 149 : i64
    %c1_i64 = arith.constant 1 : i64
    %c148_i64 = arith.constant 148 : i64
    %c119_i64 = arith.constant 119 : i64
    %c1_i8 = arith.constant 1 : i8
    %c-2_i32 = arith.constant -2 : i32
    %c116_i32 = arith.constant 116 : i32
    %c2_i8 = arith.constant 2 : i8
    affine.for %arg1 = 0 to 154 {
      %0 = arith.index_cast %arg1 : index to i64
      %1 = arith.addi %0, %c-2_i64 : i64
      %2 = arith.cmpi eq, %1, %c0_i64 : i64
      %3 = arith.cmpi eq, %1, %c149_i64 : i64
      %4 = arith.ori %2, %3 : i1
      %5 = arith.cmpi eq, %1, %c1_i64 : i64
      %6 = arith.cmpi eq, %1, %c148_i64 : i64
      %7 = arith.ori %5, %6 : i1
      scf.if %7 {
        affine.for %arg2 = 0 to 120 {
          %8 = arith.index_cast %arg2 : index to i64
          %9 = arith.cmpi eq, %8, %c0_i64 : i64
          %10 = arith.cmpi eq, %8, %c119_i64 : i64
          %11 = arith.ori %9, %10 : i1
          %12 = arith.ori %11, %4 : i1
          scf.if %12 {
            affine.parallel (%arg3) = (0) to (120) {
              %13 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
              %14 = affine.load %13[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
              %15 = arith.ori %14, %c1_i8 : i8
              affine.store %15, %13[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
            }
          } else {
            %13 = arith.trunci %8 : i64 to i32
            %14 = arith.addi %13, %c-2_i32 : i32
            %15 = arith.cmpi ult, %14, %c116_i32 : i32
            scf.if %15 {
              affine.for %arg3 = 0 to 120 {
                %16 = arith.index_cast %arg3 : index to i64
                %17 = arith.trunci %16 : i64 to i32
                %18 = arith.index_castui %17 : i32 to index
                %19 = arith.cmpi eq, %18, %c119 : index
                %20 = arith.cmpi eq, %18, %c0 : index
                %21 = arith.select %20, %c1_i8, %c2_i8 : i8
                %22 = arith.select %19, %c1_i8, %21 : i8
                %23 = arith.addi %17, %c-2_i32 : i32
                %24 = arith.cmpi uge, %23, %c116_i32 : i32
                %25 = arith.extui %24 : i1 to i32
                %26 = arith.select %20, %c0_i32, %25 : i32
                %27 = arith.select %19, %c0_i32, %26 : i32
                %28 = arith.index_castui %27 : i32 to index
                %29 = arith.cmpi eq, %28, %c0 : index
                scf.if %29 {
                  %30 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
                  %31 = affine.load %30[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
                  %32 = arith.ori %31, %22 : i8
                  affine.store %32, %30[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
                }
              }
            } else {
              affine.for %arg3 = 0 to 120 {
                %16 = arith.index_cast %arg3 : index to i64
                %17 = arith.trunci %16 : i64 to i32
                %18 = arith.index_castui %17 : i32 to index
                %19 = arith.cmpi eq, %18, %c119 : index
                %20 = arith.cmpi eq, %18, %c0 : index
                %21 = arith.extui %20 : i1 to i32
                %22 = arith.select %19, %c1_i32, %21 : i32
                %23 = arith.index_castui %22 : i32 to index
                %24 = arith.cmpi ne, %23, %c0 : index
                scf.if %24 {
                  %25 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
                  %26 = affine.load %25[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
                  %27 = arith.ori %26, %c1_i8 : i8
                  affine.store %27, %25[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
                }
              }
            }
          }
        }
      } else {
        affine.for %arg2 = 0 to 120 {
          %8 = arith.index_cast %arg2 : index to i64
          %9 = arith.cmpi eq, %8, %c0_i64 : i64
          %10 = arith.cmpi eq, %8, %c119_i64 : i64
          %11 = arith.ori %9, %10 : i1
          %12 = arith.ori %11, %4 : i1
          scf.if %12 {
            affine.parallel (%arg3) = (0) to (120) {
              %13 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
              %14 = affine.load %13[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
              %15 = arith.ori %14, %c1_i8 : i8
              affine.store %15, %13[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
            }
          } else {
            affine.for %arg3 = 0 to 120 {
              %13 = arith.index_cast %arg3 : index to i64
              %14 = arith.trunci %13 : i64 to i32
              %15 = arith.index_castui %14 : i32 to index
              %16 = arith.cmpi eq, %15, %c119 : index
              %17 = arith.cmpi eq, %15, %c0 : index
              %18 = arith.extui %17 : i1 to i32
              %19 = arith.select %16, %c1_i32, %18 : i32
              %20 = arith.index_castui %19 : i32 to index
              %21 = arith.cmpi ne, %20, %c0 : index
              scf.if %21 {
                %22 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
                %23 = affine.load %22[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
                %24 = arith.ori %23, %c1_i8 : i8
                affine.store %24, %22[%arg1 * 61440 + %arg3 * 4 + %arg2 * 512 + 179650560] : memref<?xi8>
              }
            }
          }
        }
      }
    }
    llvm.return
  }
  llvm.func local_unnamed_addr @LBM_showGridStatistics(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -1.000000e+30 : f32
    %cst_1 = arith.constant 1.000000e+30 : f32
    %0 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %1:8 = affine.for %arg1 = 0 to 150 iter_args(%arg2 = %cst, %arg3 = %cst_0, %arg4 = %cst_1, %arg5 = %cst_0, %arg6 = %cst_1, %arg7 = %c0_i32, %arg8 = %c0_i32, %arg9 = %c0_i32) -> (f32, f32, f32, f32, f32, i32, i32, i32) {
      %10:8 = affine.for %arg10 = 0 to 120 iter_args(%arg11 = %arg2, %arg12 = %arg3, %arg13 = %arg4, %arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9) -> (f32, f32, f32, f32, f32, i32, i32, i32) {
        %11:8 = affine.for %arg19 = 0 to 120 iter_args(%arg20 = %arg11, %arg21 = %arg12, %arg22 = %arg13, %arg23 = %arg14, %arg24 = %arg15, %arg25 = %arg16, %arg26 = %arg17, %arg27 = %arg18) -> (f32, f32, f32, f32, f32, i32, i32, i32) {
          %12 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
          %13 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360] : memref<?xf32>
          %14 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 2365440] : memref<?xf32>
          %15 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 4730880] : memref<?xf32>
          %16 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 7096320] : memref<?xf32>
          %17 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 9461760] : memref<?xf32>
          %18 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 11827200] : memref<?xf32>
          %19 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 14192640] : memref<?xf32>
          %20 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 16558080] : memref<?xf32>
          %21 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 18923520] : memref<?xf32>
          %22 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 21288960] : memref<?xf32>
          %23 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 23654400] : memref<?xf32>
          %24 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 26019840] : memref<?xf32>
          %25 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 28385280] : memref<?xf32>
          %26 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 30750720] : memref<?xf32>
          %27 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 33116160] : memref<?xf32>
          %28 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 35481600] : memref<?xf32>
          %29 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 37847040] : memref<?xf32>
          %30 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 40212480] : memref<?xf32>
          %31 = affine.load %12[%arg19 + %arg10 * 128 + %arg1 * 15360 + 42577920] : memref<?xf32>
          %32 = arith.addf %14, %13 fastmath<fast> : f32
          %33 = arith.addf %32, %15 fastmath<fast> : f32
          %34 = arith.addf %33, %16 fastmath<fast> : f32
          %35 = arith.addf %34, %17 fastmath<fast> : f32
          %36 = arith.addf %35, %18 fastmath<fast> : f32
          %37 = arith.addf %36, %19 fastmath<fast> : f32
          %38 = arith.addf %37, %20 fastmath<fast> : f32
          %39 = arith.addf %38, %21 fastmath<fast> : f32
          %40 = arith.addf %39, %22 fastmath<fast> : f32
          %41 = arith.addf %40, %23 fastmath<fast> : f32
          %42 = arith.addf %41, %24 fastmath<fast> : f32
          %43 = arith.addf %42, %25 fastmath<fast> : f32
          %44 = arith.addf %43, %26 fastmath<fast> : f32
          %45 = arith.addf %44, %27 fastmath<fast> : f32
          %46 = arith.addf %45, %28 fastmath<fast> : f32
          %47 = arith.addf %46, %29 fastmath<fast> : f32
          %48 = arith.addf %47, %30 fastmath<fast> : f32
          %49 = arith.addf %48, %31 fastmath<fast> : f32
          %50 = arith.minnumf %49, %arg22 fastmath<nnan,ninf,nsz> : f32
          %51 = arith.maxnumf %49, %arg21 fastmath<nnan,ninf,nsz> : f32
          %52 = arith.addf %49, %arg20 fastmath<fast> : f32
          %53 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
          %54 = affine.load %53[%arg10 * 512 + %arg19 * 4 + %arg1 * 61440 + 179773440] : memref<?xi8>
          %55 = arith.extui %54 : i8 to i32
          %56 = arith.andi %55, %c1_i32 : i32
          %57 = arith.cmpi eq, %56, %c0_i32 : i32
          %58 = arith.addi %arg27, %c1_i32 overflow<nsw> : i32
          %59 = arith.select %57, %arg27, %58 : i32
          %60 = arith.shrui %55, %c1_i32 : i32
          %61 = arith.andi %60, %c1_i32 : i32
          %62 = arith.addi %61, %arg26 overflow<nsw> : i32
          %63 = arith.select %57, %62, %arg26 : i32
          %64 = arith.xori %61, %c1_i32 : i32
          %65 = arith.addi %64, %arg25 overflow<nsw> : i32
          %66 = arith.select %57, %65, %arg25 : i32
          %67:2 = scf.if %57 -> (f32, f32) {
            %68 = arith.addf %16, %20 fastmath<fast> : f32
            %69 = arith.addf %17, %21 fastmath<fast> : f32
            %70 = arith.addf %68, %22 fastmath<fast> : f32
            %71 = arith.addf %69, %23 fastmath<fast> : f32
            %72 = arith.subf %70, %71 fastmath<fast> : f32
            %73 = arith.addf %72, %28 fastmath<fast> : f32
            %74 = arith.addf %73, %29 fastmath<fast> : f32
            %75 = arith.addf %30, %31 fastmath<fast> : f32
            %76 = arith.subf %74, %75 fastmath<fast> : f32
            %77 = arith.subf %14, %15 fastmath<fast> : f32
            %78 = arith.addf %77, %20 fastmath<fast> : f32
            %79 = arith.addf %78, %21 fastmath<fast> : f32
            %80 = arith.addf %22, %23 fastmath<fast> : f32
            %81 = arith.subf %79, %80 fastmath<fast> : f32
            %82 = arith.addf %81, %24 fastmath<fast> : f32
            %83 = arith.addf %82, %25 fastmath<fast> : f32
            %84 = arith.addf %26, %27 fastmath<fast> : f32
            %85 = arith.subf %83, %84 fastmath<fast> : f32
            %86 = arith.addf %18, %24 fastmath<fast> : f32
            %87 = arith.addf %19, %25 fastmath<fast> : f32
            %88 = arith.addf %86, %26 fastmath<fast> : f32
            %89 = arith.addf %87, %27 fastmath<fast> : f32
            %90 = arith.addf %88, %28 fastmath<fast> : f32
            %91 = arith.addf %89, %29 fastmath<fast> : f32
            %92 = arith.addf %90, %30 fastmath<fast> : f32
            %93 = arith.addf %91, %31 fastmath<fast> : f32
            %94 = arith.subf %92, %93 fastmath<fast> : f32
            %95 = arith.mulf %76, %76 fastmath<fast> : f32
            %96 = arith.mulf %85, %85 fastmath<fast> : f32
            %97 = arith.addf %95, %96 fastmath<fast> : f32
            %98 = arith.mulf %94, %94 fastmath<fast> : f32
            %99 = arith.addf %97, %98 fastmath<fast> : f32
            %100 = arith.mulf %49, %49 fastmath<fast> : f32
            %101 = arith.divf %99, %100 fastmath<fast> : f32
            %102 = arith.minnumf %101, %arg24 fastmath<nnan,ninf,nsz> : f32
            %103 = arith.cmpf ogt, %101, %arg23 fastmath<fast> : f32
            %104 = arith.select %103, %101, %arg23 : f32
            scf.yield %102, %104 : f32, f32
          } else {
            scf.yield %arg24, %arg23 : f32, f32
          }
          affine.yield %52, %51, %50, %67#1, %67#0, %66, %63, %59 : f32, f32, f32, f32, f32, i32, i32, i32
        }
        affine.yield %11#0, %11#1, %11#2, %11#3, %11#4, %11#5, %11#6, %11#7 : f32, f32, f32, f32, f32, i32, i32, i32
      }
      affine.yield %10#0, %10#1, %10#2, %10#3, %10#4, %10#5, %10#6, %10#7 : f32, f32, f32, f32, f32, i32, i32, i32
    }
    %2 = arith.extf %1#2 : f32 to f64
    %3 = arith.extf %1#1 : f32 to f64
    %4 = arith.extf %1#0 : f32 to f64
    %5 = math.sqrt %1#4 fastmath<fast> : f32
    %6 = arith.extf %5 : f32 to f64
    %7 = math.sqrt %1#3 fastmath<fast> : f32
    %8 = arith.extf %7 : f32 to f64
    %9 = llvm.call tail @printf(%0, %1#7, %1#6, %1#5, %2, %3, %4, %6, %8) vararg(!llvm.func<i32 (ptr, ...)>) {uniform_work_group_size} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}) -> i32
    llvm.return
  }
  llvm.func local_unnamed_addr @LBM_storeVelocityField(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: i32 {llvm.noundef}) attributes {dso_local, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uniform_work_group_size, uwtable_kind = #llvm.uwtableKind<async>} {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %1 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %c0_i64 = arith.constant 0 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %3 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %c150_i64 = arith.constant 150 : i64
    %4 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %5 = llvm.mlir.addressof @stderr : !llvm.ptr
    %6 = arith.index_cast %arg2 : i32 to index
    %7 = llvm.alloca %c1_i32 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %c1_i32 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %c1_i32 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start %7 : !llvm.ptr
    llvm.intr.lifetime.start %8 : !llvm.ptr
    llvm.intr.lifetime.start %9 : !llvm.ptr
    %10 = arith.cmpi eq, %arg2, %c0_i32 : i32
    %11 = arith.select %10, %0, %1 : !llvm.ptr
    %12 = llvm.call tail @fopen(%arg1, %11) {uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noalias})
    %13 = llvm.icmp "eq" %12, %2 : !llvm.ptr
    cf.cond_br %13, ^bb1, ^bb2(%c0_i64 : i64)
  ^bb1:  // pred: ^bb0
    %14 = "enzymexla.pointer2memref"(%5) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %15 = affine.load %14[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
    %16 = llvm.call tail @__errno_location() {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, uniform_work_group_size, will_return} : () -> !llvm.ptr
    %17 = "enzymexla.pointer2memref"(%16) : (!llvm.ptr) -> memref<?xi32>
    %18 = affine.load %17[0] {alignment = 4 : i64} : memref<?xi32>
    %19 = llvm.call tail @strerror(%18) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
    %20 = llvm.call tail @fprintf(%15, %4, %arg1, %19) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    llvm.call tail @exit(%c1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
    llvm.unreachable
  ^bb2(%21: i64):  // pred: ^bb0
    %22 = arith.index_cast %21 : i64 to index
    %23 = arith.addi %21, %c1_i64 : i64
    %24 = arith.maxsi %23, %c150_i64 : i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = arith.addi %25, %c1 : index
    %27 = arith.addi %22, %c1 : index
    %28 = arith.subi %26, %27 : index
    affine.for %arg3 = 0 to %28 {
      affine.for %arg4 = 0 to 120 {
        affine.for %arg5 = 0 to 120 {
          %30 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
          %31 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128] : memref<?xf32>
          %32 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 2365440] : memref<?xf32>
          %33 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 4730880] : memref<?xf32>
          %34 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 7096320] : memref<?xf32>
          %35 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 9461760] : memref<?xf32>
          %36 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 11827200] : memref<?xf32>
          %37 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 14192640] : memref<?xf32>
          %38 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 16558080] : memref<?xf32>
          %39 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 18923520] : memref<?xf32>
          %40 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 21288960] : memref<?xf32>
          %41 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 23654400] : memref<?xf32>
          %42 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 26019840] : memref<?xf32>
          %43 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 28385280] : memref<?xf32>
          %44 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 30750720] : memref<?xf32>
          %45 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 33116160] : memref<?xf32>
          %46 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 35481600] : memref<?xf32>
          %47 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 37847040] : memref<?xf32>
          %48 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 40212480] : memref<?xf32>
          %49 = affine.load %30[%arg5 + symbol(%22) * 15360 + %arg3 * 15360 + %arg4 * 128 + 42577920] : memref<?xf32>
          %50 = arith.addf %32, %31 fastmath<fast> : f32
          %51 = arith.addf %50, %33 fastmath<fast> : f32
          %52 = arith.addf %51, %34 fastmath<fast> : f32
          %53 = arith.addf %52, %35 fastmath<fast> : f32
          %54 = arith.addf %53, %36 fastmath<fast> : f32
          %55 = arith.addf %54, %37 fastmath<fast> : f32
          %56 = arith.addf %55, %38 fastmath<fast> : f32
          %57 = arith.addf %56, %39 fastmath<fast> : f32
          %58 = arith.addf %57, %40 fastmath<fast> : f32
          %59 = arith.addf %58, %41 fastmath<fast> : f32
          %60 = arith.addf %59, %42 fastmath<fast> : f32
          %61 = arith.addf %60, %43 fastmath<fast> : f32
          %62 = arith.addf %61, %44 fastmath<fast> : f32
          %63 = arith.addf %62, %45 fastmath<fast> : f32
          %64 = arith.addf %63, %46 fastmath<fast> : f32
          %65 = arith.addf %64, %47 fastmath<fast> : f32
          %66 = arith.addf %65, %48 fastmath<fast> : f32
          %67 = arith.addf %66, %49 fastmath<fast> : f32
          %68 = arith.addf %34, %38 fastmath<fast> : f32
          %69 = arith.addf %35, %39 fastmath<fast> : f32
          %70 = arith.addf %68, %40 fastmath<fast> : f32
          %71 = arith.addf %69, %41 fastmath<fast> : f32
          %72 = arith.subf %70, %71 fastmath<fast> : f32
          %73 = arith.addf %72, %46 fastmath<fast> : f32
          %74 = arith.addf %73, %47 fastmath<fast> : f32
          %75 = arith.addf %48, %49 fastmath<fast> : f32
          %76 = arith.subf %74, %75 fastmath<fast> : f32
          %77 = arith.subf %32, %33 fastmath<fast> : f32
          %78 = arith.addf %77, %38 fastmath<fast> : f32
          %79 = arith.addf %78, %39 fastmath<fast> : f32
          %80 = arith.addf %40, %41 fastmath<fast> : f32
          %81 = arith.subf %79, %80 fastmath<fast> : f32
          %82 = arith.addf %81, %42 fastmath<fast> : f32
          %83 = arith.addf %82, %43 fastmath<fast> : f32
          %84 = arith.addf %44, %45 fastmath<fast> : f32
          %85 = arith.subf %83, %84 fastmath<fast> : f32
          %86 = arith.addf %36, %42 fastmath<fast> : f32
          %87 = arith.addf %37, %43 fastmath<fast> : f32
          %88 = arith.addf %86, %44 fastmath<fast> : f32
          %89 = arith.addf %87, %45 fastmath<fast> : f32
          %90 = arith.addf %88, %46 fastmath<fast> : f32
          %91 = arith.addf %89, %47 fastmath<fast> : f32
          %92 = arith.addf %90, %48 fastmath<fast> : f32
          %93 = arith.addf %91, %49 fastmath<fast> : f32
          %94 = arith.subf %92, %93 fastmath<fast> : f32
          %95 = arith.divf %76, %67 fastmath<fast> : f32
          %96 = "enzymexla.pointer2memref"(%7) : (!llvm.ptr) -> memref<?xf32>
          %97 = arith.divf %85, %67 fastmath<fast> : f32
          %98 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<?xf32>
          %99 = arith.divf %94, %67 fastmath<fast> : f32
          %100 = "enzymexla.pointer2memref"(%9) : (!llvm.ptr) -> memref<?xf32>
          affine.store %95, %96[0] {alignment = 4 : i64} : memref<?xf32>
          affine.store %97, %98[0] {alignment = 4 : i64} : memref<?xf32>
          affine.store %99, %100[0] {alignment = 4 : i64} : memref<?xf32>
          affine.if #set()[%6] {
            %101 = arith.extf %95 : f32 to f64
            %102 = arith.extf %97 : f32 to f64
            %103 = arith.extf %99 : f32 to f64
            %104 = llvm.call tail @fprintf(%12, %3, %101, %102, %103) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}, f64 {llvm.nofpclass = 519 : i64, llvm.noundef}) -> i32
          } else {
            %101 = llvm.call @fwrite(%7, %c4_i64, %c1_i64, %12) {uniform_work_group_size} : (!llvm.ptr {llvm.nonnull, llvm.noundef, llvm.readonly}, i64 {llvm.noundef}, i64 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> i64
            %102 = llvm.call @fwrite(%8, %c4_i64, %c1_i64, %12) {uniform_work_group_size} : (!llvm.ptr {llvm.nonnull, llvm.noundef, llvm.readonly}, i64 {llvm.noundef}, i64 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> i64
            %103 = llvm.call @fwrite(%9, %c4_i64, %c1_i64, %12) {uniform_work_group_size} : (!llvm.ptr {llvm.nonnull, llvm.noundef, llvm.readonly}, i64 {llvm.noundef}, i64 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}) -> i64
          }
        }
      }
    }
    %29 = llvm.call tail @fclose(%12) {uniform_work_group_size} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> i32
    llvm.intr.lifetime.end %9 : !llvm.ptr
    llvm.intr.lifetime.end %8 : !llvm.ptr
    llvm.intr.lifetime.end %7 : !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @strerror(i32 {llvm.noundef}) -> !llvm.ptr attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @__errno_location() -> !llvm.ptr attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["mustprogress", "nofree", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size, will_return}
  llvm.func local_unnamed_addr @fwrite(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, i64 {llvm.noundef}, i64 {llvm.noundef}, !llvm.ptr {llvm.nocapture, llvm.noundef}) -> (i64 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uniform_work_group_size}
  llvm.func local_unnamed_addr @calloc(i64 {llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {allocsize = array<i32: 0, 1>, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite, errnoMem = write, targetMem0 = none, targetMem1 = none>, no_unwind, passthrough = ["nofree", ["allockind", "17"], ["alloc-family", "malloc"]], sym_visibility = "private", will_return}
}
