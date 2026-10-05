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
      %sl_z = arith.constant 0 : index
      %sl_rows = arith.constant 76800 : index
      %sl_one = llvm.mlir.constant(1 : i64) : i64
      %sl_zero64 = llvm.mlir.constant(0 : i64) : i64
      %sl_stp = llvm.alloca %sl_one x !llvm.ptr : (i64) -> !llvm.ptr
      %sl_gp = llvm.alloca %sl_one x !llvm.ptr : (i64) -> !llvm.ptr
      %sl_ep = llvm.alloca %sl_one x !llvm.ptr : (i64) -> !llvm.ptr
      %sl_r0 = llvm.call @cudaStreamCreateWithFlags(%sl_stp, %c1_i32) : (!llvm.ptr, i32) -> i32
      %sl_st = llvm.load %sl_stp : !llvm.ptr -> !llvm.ptr
      %sl_r1 = llvm.call @cudaDeviceSynchronize() : () -> i32
      %sl_nfull = arith.divui %sl_H, %sl_k : index
      %sl_Hfull = arith.muli %sl_nfull, %sl_k : index
      %sl_hasfull = arith.cmpi ugt, %sl_nfull, %sl_z : index
      scf.if %sl_hasfull {
        %sl_r2 = llvm.call @cudaStreamBeginCapture(%sl_st, %c2_i32) : (!llvm.ptr, i32) -> i32
        %sfnw0 = arith.addi %sl_ns, %sl_k : index
        %sfnw = arith.subi %sfnw0, %c1 : index
        scf.for %sfwv = %sl_z to %sfnw step %c1 {
          scf.for %sfj = %sl_z to %sl_k step %c1 {
            %sfi = arith.subi %sfwv, %sfj : index
            %sfge = arith.cmpi sge, %sfi, %sl_z : index
            %sflt = arith.cmpi slt, %sfi, %sl_ns : index
            %sfok = arith.andi %sfge, %sflt : i1
            scf.if %sfok {
              %sfs = arith.addi %sl_z, %sfj : index
              %sfpar = arith.remui %sfs, %c2 : index
              %sfeven = arith.cmpi eq, %sfpar, %sl_z : index
              %sfsrc = llvm.select %sfeven, %arg1, %arg2 : i1, !llvm.ptr
              %sfdst = llvm.select %sfeven, %arg2, %arg1 : i1, !llvm.ptr
              %sfoff = arith.muli %sfi, %sl_rows : index
              %sfo64 = arith.index_cast %sfoff : index to i64
              %sfsrcS = llvm.getelementptr %sfsrc[%sfo64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %sfdstS = llvm.getelementptr %sfdst[%sfo64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %sftok = "enzymexla.stream2token"(%sl_st) : (!llvm.ptr) -> !async.token
              %sfdone = async.execute [%sftok] {
                %sfwr = "enzymexla.gpu_wrapper"(%c120, %c5, %c1, %c120, %c1, %c1) ({
                            affine.parallel (%arg4, %arg5) = (0, 0) to (600, 120) {
                              llvm.intr.experimental.noalias.scope.decl #alias_scope
                              llvm.intr.experimental.noalias.scope.decl #alias_scope1
                              %sfw17 = "enzymexla.pointer2memref"(%sfsrcS) : (!llvm.ptr) -> memref<?xf32>
                              %sfw18 = affine.load %sfw17[%arg5 + %arg4 * 128] : memref<?xf32>
                              %sfw19 = affine.load %sfw17[%arg5 + %arg4 * 128 + 2365440] : memref<?xf32>
                              %sfw20 = affine.load %sfw17[%arg5 + %arg4 * 128 + 4730880] : memref<?xf32>
                              %sfw21 = affine.load %sfw17[%arg5 + %arg4 * 128 + 7096320] : memref<?xf32>
                              %sfw22 = affine.load %sfw17[%arg5 + %arg4 * 128 + 9461760] : memref<?xf32>
                              %sfw23 = affine.load %sfw17[%arg5 + %arg4 * 128 + 11827200] : memref<?xf32>
                              %sfw24 = affine.load %sfw17[%arg5 + %arg4 * 128 + 14192640] : memref<?xf32>
                              %sfw25 = affine.load %sfw17[%arg5 + %arg4 * 128 + 16558080] : memref<?xf32>
                              %sfw26 = affine.load %sfw17[%arg5 + %arg4 * 128 + 18923520] : memref<?xf32>
                              %sfw27 = affine.load %sfw17[%arg5 + %arg4 * 128 + 21288960] : memref<?xf32>
                              %sfw28 = affine.load %sfw17[%arg5 + %arg4 * 128 + 23654400] : memref<?xf32>
                              %sfw29 = affine.load %sfw17[%arg5 + %arg4 * 128 + 26019840] : memref<?xf32>
                              %sfw30 = affine.load %sfw17[%arg5 + %arg4 * 128 + 28385280] : memref<?xf32>
                              %sfw31 = affine.load %sfw17[%arg5 + %arg4 * 128 + 30750720] : memref<?xf32>
                              %sfw32 = affine.load %sfw17[%arg5 + %arg4 * 128 + 33116160] : memref<?xf32>
                              %sfw33 = affine.load %sfw17[%arg5 + %arg4 * 128 + 35481600] : memref<?xf32>
                              %sfw34 = affine.load %sfw17[%arg5 + %arg4 * 128 + 37847040] : memref<?xf32>
                              %sfw35 = affine.load %sfw17[%arg5 + %arg4 * 128 + 40212480] : memref<?xf32>
                              %sfw36 = affine.load %sfw17[%arg5 + %arg4 * 128 + 42577920] : memref<?xf32>
                              %sfw37 = "enzymexla.pointer2memref"(%sfsrcS) : (!llvm.ptr) -> memref<?xi8>
                              %sfw38 = affine.load %sfw37[%arg5 * 4 + %arg4 * 512 + 179773440] : memref<?xi8>
                              %sfw39 = arith.extui %sfw38 : i8 to i32
                              %sfw40 = arith.andi %sfw39, %c1_i32 : i32
                              %sfw41 = arith.cmpi eq, %sfw40, %c0_i32 : i32
                              %sfw42:19 = scf.if %sfw41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                                %sfw44 = arith.addf %sfw19, %sfw18 fastmath<fast> : f32
                                %sfw45 = arith.addf %sfw44, %sfw20 fastmath<fast> : f32
                                %sfw46 = arith.addf %sfw45, %sfw21 fastmath<fast> : f32
                                %sfw47 = arith.addf %sfw46, %sfw22 fastmath<fast> : f32
                                %sfw48 = arith.addf %sfw47, %sfw23 fastmath<fast> : f32
                                %sfw49 = arith.addf %sfw48, %sfw24 fastmath<fast> : f32
                                %sfw50 = arith.addf %sfw49, %sfw25 fastmath<fast> : f32
                                %sfw51 = arith.addf %sfw50, %sfw26 fastmath<fast> : f32
                                %sfw52 = arith.addf %sfw51, %sfw27 fastmath<fast> : f32
                                %sfw53 = arith.addf %sfw52, %sfw28 fastmath<fast> : f32
                                %sfw54 = arith.addf %sfw53, %sfw29 fastmath<fast> : f32
                                %sfw55 = arith.addf %sfw54, %sfw30 fastmath<fast> : f32
                                %sfw56 = arith.addf %sfw55, %sfw31 fastmath<fast> : f32
                                %sfw57 = arith.addf %sfw56, %sfw32 fastmath<fast> : f32
                                %sfw58 = arith.addf %sfw57, %sfw33 fastmath<fast> : f32
                                %sfw59 = arith.addf %sfw58, %sfw34 fastmath<fast> : f32
                                %sfw60 = arith.addf %sfw59, %sfw35 fastmath<fast> : f32
                                %sfw61 = arith.addf %sfw60, %sfw36 fastmath<fast> : f32
                                %sfw62 = arith.addf %sfw21, %sfw25 fastmath<fast> : f32
                                %sfw63 = arith.addf %sfw22, %sfw26 fastmath<fast> : f32
                                %sfw64 = arith.addf %sfw62, %sfw27 fastmath<fast> : f32
                                %sfw65 = arith.addf %sfw63, %sfw28 fastmath<fast> : f32
                                %sfw66 = arith.subf %sfw64, %sfw65 fastmath<fast> : f32
                                %sfw67 = arith.addf %sfw66, %sfw33 fastmath<fast> : f32
                                %sfw68 = arith.addf %sfw67, %sfw34 fastmath<fast> : f32
                                %sfw69 = arith.addf %sfw35, %sfw36 fastmath<fast> : f32
                                %sfw70 = arith.subf %sfw68, %sfw69 fastmath<fast> : f32
                                %sfw71 = arith.subf %sfw19, %sfw20 fastmath<fast> : f32
                                %sfw72 = arith.addf %sfw71, %sfw25 fastmath<fast> : f32
                                %sfw73 = arith.addf %sfw72, %sfw26 fastmath<fast> : f32
                                %sfw74 = arith.addf %sfw27, %sfw28 fastmath<fast> : f32
                                %sfw75 = arith.subf %sfw73, %sfw74 fastmath<fast> : f32
                                %sfw76 = arith.addf %sfw75, %sfw29 fastmath<fast> : f32
                                %sfw77 = arith.addf %sfw76, %sfw30 fastmath<fast> : f32
                                %sfw78 = arith.addf %sfw31, %sfw32 fastmath<fast> : f32
                                %sfw79 = arith.subf %sfw77, %sfw78 fastmath<fast> : f32
                                %sfw80 = arith.addf %sfw23, %sfw29 fastmath<fast> : f32
                                %sfw81 = arith.addf %sfw24, %sfw30 fastmath<fast> : f32
                                %sfw82 = arith.addf %sfw80, %sfw31 fastmath<fast> : f32
                                %sfw83 = arith.addf %sfw81, %sfw32 fastmath<fast> : f32
                                %sfw84 = arith.addf %sfw82, %sfw33 fastmath<fast> : f32
                                %sfw85 = arith.addf %sfw83, %sfw34 fastmath<fast> : f32
                                %sfw86 = arith.addf %sfw84, %sfw35 fastmath<fast> : f32
                                %sfw87 = arith.addf %sfw85, %sfw36 fastmath<fast> : f32
                                %sfw88 = arith.subf %sfw86, %sfw87 fastmath<fast> : f32
                                %sfw89 = arith.divf %sfw70, %sfw61 fastmath<fast> : f32
                                %sfw90 = arith.divf %sfw79, %sfw61 fastmath<fast> : f32
                                %sfw91 = arith.divf %sfw88, %sfw61 fastmath<fast> : f32
                                %sfw92 = arith.andi %sfw39, %c2_i32 : i32
                                %sfw93 = arith.cmpi eq, %sfw92, %c0_i32 : i32
                                %sfw94 = arith.select %sfw93, %sfw89, %cst_11 : f32
                                %sfw95 = arith.select %sfw93, %sfw90, %cst_10 : f32
                                %sfw96 = arith.select %sfw93, %sfw91, %cst_9 : f32
                                %sfw97 = arith.mulf %sfw94, %sfw94 fastmath<fast> : f32
                                %sfw98 = arith.mulf %sfw95, %sfw95 fastmath<fast> : f32
                                %sfw99 = arith.addf %sfw97, %sfw98 fastmath<fast> : f32
                                %sfw100 = arith.mulf %sfw96, %sfw96 fastmath<fast> : f32
                                %sfw101 = arith.addf %sfw99, %sfw100 fastmath<fast> : f32
                                %sfw102 = arith.mulf %sfw101, %cst_8 fastmath<fast> : f32
                                %sfw103 = arith.addf %sfw102, %cst_7 fastmath<fast> : f32
                                %sfw104 = arith.mulf %sfw61, %cst_6 fastmath<fast> : f32
                                %sfw105 = arith.mulf %sfw18, %cst_5 fastmath<fast> : f32
                                %sfw106 = arith.mulf %sfw104, %sfw103 fastmath<fast> : f32
                                %sfw107 = arith.subf %sfw105, %sfw106 fastmath<fast> : f32
                                %sfw108 = arith.mulf %sfw61, %cst_4 fastmath<fast> : f32
                                %sfw109 = arith.mulf %sfw19, %cst_3 fastmath<fast> : f32
                                %sfw110 = arith.mulf %sfw95, %cst_2 fastmath<fast> : f32
                                %sfw111 = arith.addf %sfw110, %cst_1 fastmath<fast> : f32
                                %sfw112 = arith.mulf %sfw111, %sfw95 fastmath<fast> : f32
                                %sfw113 = arith.subf %sfw112, %sfw103 fastmath<fast> : f32
                                %sfw114 = arith.mulf %sfw113, %sfw108 fastmath<fast> : f32
                                %sfw115 = arith.subf %sfw114, %sfw109 fastmath<fast> : f32
                                %sfw116 = arith.mulf %sfw20, %cst_3 fastmath<fast> : f32
                                %sfw117 = arith.addf %sfw110, %cst_0 fastmath<fast> : f32
                                %sfw118 = arith.mulf %sfw117, %sfw95 fastmath<fast> : f32
                                %sfw119 = arith.subf %sfw118, %sfw103 fastmath<fast> : f32
                                %sfw120 = arith.mulf %sfw119, %sfw108 fastmath<fast> : f32
                                %sfw121 = arith.subf %sfw120, %sfw116 fastmath<fast> : f32
                                %sfw122 = arith.mulf %sfw23, %cst_3 fastmath<fast> : f32
                                %sfw123 = arith.mulf %sfw96, %cst_2 fastmath<fast> : f32
                                %sfw124 = arith.addf %sfw123, %cst_1 fastmath<fast> : f32
                                %sfw125 = arith.mulf %sfw124, %sfw96 fastmath<fast> : f32
                                %sfw126 = arith.subf %sfw125, %sfw103 fastmath<fast> : f32
                                %sfw127 = arith.mulf %sfw126, %sfw108 fastmath<fast> : f32
                                %sfw128 = arith.subf %sfw127, %sfw122 fastmath<fast> : f32
                                %sfw129 = arith.mulf %sfw24, %cst_3 fastmath<fast> : f32
                                %sfw130 = arith.addf %sfw123, %cst_0 fastmath<fast> : f32
                                %sfw131 = arith.mulf %sfw130, %sfw96 fastmath<fast> : f32
                                %sfw132 = arith.subf %sfw131, %sfw103 fastmath<fast> : f32
                                %sfw133 = arith.mulf %sfw132, %sfw108 fastmath<fast> : f32
                                %sfw134 = arith.subf %sfw133, %sfw129 fastmath<fast> : f32
                                %sfw135 = arith.mulf %sfw21, %cst_3 fastmath<fast> : f32
                                %sfw136 = arith.mulf %sfw94, %cst_2 fastmath<fast> : f32
                                %sfw137 = arith.addf %sfw136, %cst_1 fastmath<fast> : f32
                                %sfw138 = arith.mulf %sfw137, %sfw94 fastmath<fast> : f32
                                %sfw139 = arith.subf %sfw138, %sfw103 fastmath<fast> : f32
                                %sfw140 = arith.mulf %sfw139, %sfw108 fastmath<fast> : f32
                                %sfw141 = arith.subf %sfw140, %sfw135 fastmath<fast> : f32
                                %sfw142 = arith.mulf %sfw22, %cst_3 fastmath<fast> : f32
                                %sfw143 = arith.addf %sfw136, %cst_0 fastmath<fast> : f32
                                %sfw144 = arith.mulf %sfw143, %sfw94 fastmath<fast> : f32
                                %sfw145 = arith.subf %sfw144, %sfw103 fastmath<fast> : f32
                                %sfw146 = arith.mulf %sfw145, %sfw108 fastmath<fast> : f32
                                %sfw147 = arith.subf %sfw146, %sfw142 fastmath<fast> : f32
                                %sfw148 = arith.mulf %sfw61, %cst fastmath<fast> : f32
                                %sfw149 = arith.mulf %sfw29, %cst_3 fastmath<fast> : f32
                                %sfw150 = arith.addf %sfw95, %sfw96 fastmath<fast> : f32
                                %sfw151 = arith.mulf %sfw150, %cst_2 fastmath<fast> : f32
                                %sfw152 = arith.addf %sfw151, %cst_1 fastmath<fast> : f32
                                %sfw153 = arith.mulf %sfw152, %sfw150 fastmath<fast> : f32
                                %sfw154 = arith.subf %sfw153, %sfw103 fastmath<fast> : f32
                                %sfw155 = arith.mulf %sfw154, %sfw148 fastmath<fast> : f32
                                %sfw156 = arith.subf %sfw155, %sfw149 fastmath<fast> : f32
                                %sfw157 = arith.mulf %sfw30, %cst_3 fastmath<fast> : f32
                                %sfw158 = arith.subf %sfw95, %sfw96 fastmath<fast> : f32
                                %sfw159 = arith.mulf %sfw158, %cst_2 fastmath<fast> : f32
                                %sfw160 = arith.addf %sfw159, %cst_1 fastmath<fast> : f32
                                %sfw161 = arith.mulf %sfw160, %sfw158 fastmath<fast> : f32
                                %sfw162 = arith.subf %sfw161, %sfw103 fastmath<fast> : f32
                                %sfw163 = arith.mulf %sfw162, %sfw148 fastmath<fast> : f32
                                %sfw164 = arith.subf %sfw163, %sfw157 fastmath<fast> : f32
                                %sfw165 = arith.mulf %sfw31, %cst_3 fastmath<fast> : f32
                                %sfw166 = arith.subf %sfw96, %sfw95 fastmath<fast> : f32
                                %sfw167 = arith.mulf %sfw166, %cst_2 fastmath<fast> : f32
                                %sfw168 = arith.addf %sfw167, %cst_1 fastmath<fast> : f32
                                %sfw169 = arith.mulf %sfw168, %sfw166 fastmath<fast> : f32
                                %sfw170 = arith.subf %sfw169, %sfw103 fastmath<fast> : f32
                                %sfw171 = arith.mulf %sfw170, %sfw148 fastmath<fast> : f32
                                %sfw172 = arith.subf %sfw171, %sfw165 fastmath<fast> : f32
                                %sfw173 = arith.mulf %sfw32, %cst_3 fastmath<fast> : f32
                                %sfw174 = arith.negf %sfw150 fastmath<fast> : f32
                                %sfw175 = arith.subf %cst_1, %sfw151 fastmath<fast> : f32
                                %sfw176 = arith.mulf %sfw175, %sfw174 fastmath<fast> : f32
                                %sfw177 = arith.subf %sfw176, %sfw103 fastmath<fast> : f32
                                %sfw178 = arith.mulf %sfw177, %sfw148 fastmath<fast> : f32
                                %sfw179 = arith.subf %sfw178, %sfw173 fastmath<fast> : f32
                                %sfw180 = arith.mulf %sfw25, %cst_3 fastmath<fast> : f32
                                %sfw181 = arith.addf %sfw94, %sfw95 fastmath<fast> : f32
                                %sfw182 = arith.mulf %sfw181, %cst_2 fastmath<fast> : f32
                                %sfw183 = arith.addf %sfw182, %cst_1 fastmath<fast> : f32
                                %sfw184 = arith.mulf %sfw183, %sfw181 fastmath<fast> : f32
                                %sfw185 = arith.subf %sfw184, %sfw103 fastmath<fast> : f32
                                %sfw186 = arith.mulf %sfw185, %sfw148 fastmath<fast> : f32
                                %sfw187 = arith.subf %sfw186, %sfw180 fastmath<fast> : f32
                                %sfw188 = arith.mulf %sfw27, %cst_3 fastmath<fast> : f32
                                %sfw189 = arith.subf %sfw94, %sfw95 fastmath<fast> : f32
                                %sfw190 = arith.mulf %sfw189, %cst_2 fastmath<fast> : f32
                                %sfw191 = arith.addf %sfw190, %cst_1 fastmath<fast> : f32
                                %sfw192 = arith.mulf %sfw191, %sfw189 fastmath<fast> : f32
                                %sfw193 = arith.subf %sfw192, %sfw103 fastmath<fast> : f32
                                %sfw194 = arith.mulf %sfw193, %sfw148 fastmath<fast> : f32
                                %sfw195 = arith.subf %sfw194, %sfw188 fastmath<fast> : f32
                                %sfw196 = arith.mulf %sfw33, %cst_3 fastmath<fast> : f32
                                %sfw197 = arith.addf %sfw94, %sfw96 fastmath<fast> : f32
                                %sfw198 = arith.mulf %sfw197, %cst_2 fastmath<fast> : f32
                                %sfw199 = arith.addf %sfw198, %cst_1 fastmath<fast> : f32
                                %sfw200 = arith.mulf %sfw199, %sfw197 fastmath<fast> : f32
                                %sfw201 = arith.subf %sfw200, %sfw103 fastmath<fast> : f32
                                %sfw202 = arith.mulf %sfw201, %sfw148 fastmath<fast> : f32
                                %sfw203 = arith.subf %sfw202, %sfw196 fastmath<fast> : f32
                                %sfw204 = arith.mulf %sfw34, %cst_3 fastmath<fast> : f32
                                %sfw205 = arith.subf %sfw94, %sfw96 fastmath<fast> : f32
                                %sfw206 = arith.mulf %sfw205, %cst_2 fastmath<fast> : f32
                                %sfw207 = arith.addf %sfw206, %cst_1 fastmath<fast> : f32
                                %sfw208 = arith.mulf %sfw207, %sfw205 fastmath<fast> : f32
                                %sfw209 = arith.subf %sfw208, %sfw103 fastmath<fast> : f32
                                %sfw210 = arith.mulf %sfw209, %sfw148 fastmath<fast> : f32
                                %sfw211 = arith.subf %sfw210, %sfw204 fastmath<fast> : f32
                                %sfw212 = arith.mulf %sfw26, %cst_3 fastmath<fast> : f32
                                %sfw213 = arith.negf %sfw94 fastmath<fast> : f32
                                %sfw214 = arith.subf %sfw95, %sfw94 fastmath<fast> : f32
                                %sfw215 = arith.mulf %sfw214, %cst_2 fastmath<fast> : f32
                                %sfw216 = arith.addf %sfw215, %cst_1 fastmath<fast> : f32
                                %sfw217 = arith.mulf %sfw216, %sfw214 fastmath<fast> : f32
                                %sfw218 = arith.subf %sfw217, %sfw103 fastmath<fast> : f32
                                %sfw219 = arith.mulf %sfw218, %sfw148 fastmath<fast> : f32
                                %sfw220 = arith.subf %sfw219, %sfw212 fastmath<fast> : f32
                                %sfw221 = arith.mulf %sfw28, %cst_3 fastmath<fast> : f32
                                %sfw222 = arith.subf %sfw213, %sfw95 fastmath<fast> : f32
                                %sfw223 = arith.mulf %sfw222, %cst_2 fastmath<fast> : f32
                                %sfw224 = arith.addf %sfw223, %cst_1 fastmath<fast> : f32
                                %sfw225 = arith.mulf %sfw224, %sfw222 fastmath<fast> : f32
                                %sfw226 = arith.subf %sfw225, %sfw103 fastmath<fast> : f32
                                %sfw227 = arith.mulf %sfw226, %sfw148 fastmath<fast> : f32
                                %sfw228 = arith.subf %sfw227, %sfw221 fastmath<fast> : f32
                                %sfw229 = arith.mulf %sfw35, %cst_3 fastmath<fast> : f32
                                %sfw230 = arith.subf %sfw96, %sfw94 fastmath<fast> : f32
                                %sfw231 = arith.mulf %sfw230, %cst_2 fastmath<fast> : f32
                                %sfw232 = arith.addf %sfw231, %cst_1 fastmath<fast> : f32
                                %sfw233 = arith.mulf %sfw232, %sfw230 fastmath<fast> : f32
                                %sfw234 = arith.subf %sfw233, %sfw103 fastmath<fast> : f32
                                %sfw235 = arith.mulf %sfw234, %sfw148 fastmath<fast> : f32
                                %sfw236 = arith.subf %sfw235, %sfw229 fastmath<fast> : f32
                                %sfw237 = arith.mulf %sfw36, %cst_3 fastmath<fast> : f32
                                %sfw238 = arith.subf %sfw213, %sfw96 fastmath<fast> : f32
                                %sfw239 = arith.mulf %sfw238, %cst_2 fastmath<fast> : f32
                                %sfw240 = arith.addf %sfw239, %cst_1 fastmath<fast> : f32
                                %sfw241 = arith.mulf %sfw240, %sfw238 fastmath<fast> : f32
                                %sfw242 = arith.subf %sfw241, %sfw103 fastmath<fast> : f32
                                %sfw243 = arith.mulf %sfw242, %sfw148 fastmath<fast> : f32
                                %sfw244 = arith.subf %sfw243, %sfw237 fastmath<fast> : f32
                                scf.yield %sfw107, %sfw115, %sfw121, %sfw141, %sfw147, %sfw128, %sfw134, %sfw187, %sfw220, %sfw195, %sfw228, %sfw156, %sfw164, %sfw172, %sfw179, %sfw203, %sfw211, %sfw236, %sfw244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              } else {
                                scf.yield %sfw18, %sfw20, %sfw19, %sfw22, %sfw21, %sfw24, %sfw23, %sfw28, %sfw27, %sfw26, %sfw25, %sfw32, %sfw31, %sfw30, %sfw29, %sfw36, %sfw35, %sfw34, %sfw33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              }
                              %sfw43 = "enzymexla.pointer2memref"(%sfdstS) : (!llvm.ptr) -> memref<?xf32>
                              affine.store %sfw42#0, %sfw43[%arg5 + %arg4 * 128] : memref<?xf32>
                              affine.store %sfw42#1, %sfw43[%arg5 + %arg4 * 128 + 2365568] : memref<?xf32>
                              affine.store %sfw42#2, %sfw43[%arg5 + %arg4 * 128 + 4730752] : memref<?xf32>
                              affine.store %sfw42#3, %sfw43[%arg5 + %arg4 * 128 + 7096321] : memref<?xf32>
                              affine.store %sfw42#4, %sfw43[%arg5 + %arg4 * 128 + 9461759] : memref<?xf32>
                              affine.store %sfw42#5, %sfw43[%arg5 + %arg4 * 128 + 11842560] : memref<?xf32>
                              affine.store %sfw42#6, %sfw43[%arg5 + %arg4 * 128 + 14177280] : memref<?xf32>
                              affine.store %sfw42#7, %sfw43[%arg5 + %arg4 * 128 + 16558209] : memref<?xf32>
                              affine.store %sfw42#8, %sfw43[%arg5 + %arg4 * 128 + 18923647] : memref<?xf32>
                              affine.store %sfw42#9, %sfw43[%arg5 + %arg4 * 128 + 21288833] : memref<?xf32>
                              affine.store %sfw42#10, %sfw43[%arg5 + %arg4 * 128 + 23654271] : memref<?xf32>
                              affine.store %sfw42#11, %sfw43[%arg5 + %arg4 * 128 + 26035328] : memref<?xf32>
                              affine.store %sfw42#12, %sfw43[%arg5 + %arg4 * 128 + 28370048] : memref<?xf32>
                              affine.store %sfw42#13, %sfw43[%arg5 + %arg4 * 128 + 30765952] : memref<?xf32>
                              affine.store %sfw42#14, %sfw43[%arg5 + %arg4 * 128 + 33100672] : memref<?xf32>
                              affine.store %sfw42#15, %sfw43[%arg5 + %arg4 * 128 + 35496961] : memref<?xf32>
                              affine.store %sfw42#16, %sfw43[%arg5 + %arg4 * 128 + 37831681] : memref<?xf32>
                              affine.store %sfw42#17, %sfw43[%arg5 + %arg4 * 128 + 40227839] : memref<?xf32>
                              affine.store %sfw42#18, %sfw43[%arg5 + %arg4 * 128 + 42562559] : memref<?xf32>
                            }
                            "enzymexla.polygeist_yield"() : () -> ()
                }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
                async.yield
              }
            }
          }
        }
        %sl_r3 = llvm.call @cudaStreamEndCapture(%sl_st, %sl_gp) : (!llvm.ptr, !llvm.ptr) -> i32
        %sl_g = llvm.load %sl_gp : !llvm.ptr -> !llvm.ptr
        %sl_r4 = llvm.call @cudaGraphInstantiate(%sl_ep, %sl_g, %sl_zero64) : (!llvm.ptr, !llvm.ptr, i64) -> i32
        %sl_e = llvm.load %sl_ep : !llvm.ptr -> !llvm.ptr
        scf.for %sl_b = %sl_z to %sl_nfull step %c1 {
          %sl_r5 = llvm.call @cudaGraphLaunch(%sl_e, %sl_st) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        %sl_r6 = llvm.call @cudaGraphExecDestroy(%sl_e) : (!llvm.ptr) -> i32
        %sl_r7 = llvm.call @cudaGraphDestroy(%sl_g) : (!llvm.ptr) -> i32
      }
      %sl_rem = arith.subi %sl_H, %sl_Hfull : index
      %sl_hasrem = arith.cmpi ugt, %sl_rem, %sl_z : index
      scf.if %sl_hasrem {
        %srnw0 = arith.addi %sl_ns, %sl_rem : index
        %srnw = arith.subi %srnw0, %c1 : index
        scf.for %srwv = %sl_z to %srnw step %c1 {
          scf.for %srj = %sl_z to %sl_rem step %c1 {
            %sri = arith.subi %srwv, %srj : index
            %srge = arith.cmpi sge, %sri, %sl_z : index
            %srlt = arith.cmpi slt, %sri, %sl_ns : index
            %srok = arith.andi %srge, %srlt : i1
            scf.if %srok {
              %srs = arith.addi %sl_Hfull, %srj : index
              %srpar = arith.remui %srs, %c2 : index
              %sreven = arith.cmpi eq, %srpar, %sl_z : index
              %srsrc = llvm.select %sreven, %arg1, %arg2 : i1, !llvm.ptr
              %srdst = llvm.select %sreven, %arg2, %arg1 : i1, !llvm.ptr
              %sroff = arith.muli %sri, %sl_rows : index
              %sro64 = arith.index_cast %sroff : index to i64
              %srsrcS = llvm.getelementptr %srsrc[%sro64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %srdstS = llvm.getelementptr %srdst[%sro64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %srtok = "enzymexla.stream2token"(%sl_st) : (!llvm.ptr) -> !async.token
              %srdone = async.execute [%srtok] {
                %srwr = "enzymexla.gpu_wrapper"(%c120, %c5, %c1, %c120, %c1, %c1) ({
                            affine.parallel (%arg4, %arg5) = (0, 0) to (600, 120) {
                              llvm.intr.experimental.noalias.scope.decl #alias_scope
                              llvm.intr.experimental.noalias.scope.decl #alias_scope1
                              %srw17 = "enzymexla.pointer2memref"(%srsrcS) : (!llvm.ptr) -> memref<?xf32>
                              %srw18 = affine.load %srw17[%arg5 + %arg4 * 128] : memref<?xf32>
                              %srw19 = affine.load %srw17[%arg5 + %arg4 * 128 + 2365440] : memref<?xf32>
                              %srw20 = affine.load %srw17[%arg5 + %arg4 * 128 + 4730880] : memref<?xf32>
                              %srw21 = affine.load %srw17[%arg5 + %arg4 * 128 + 7096320] : memref<?xf32>
                              %srw22 = affine.load %srw17[%arg5 + %arg4 * 128 + 9461760] : memref<?xf32>
                              %srw23 = affine.load %srw17[%arg5 + %arg4 * 128 + 11827200] : memref<?xf32>
                              %srw24 = affine.load %srw17[%arg5 + %arg4 * 128 + 14192640] : memref<?xf32>
                              %srw25 = affine.load %srw17[%arg5 + %arg4 * 128 + 16558080] : memref<?xf32>
                              %srw26 = affine.load %srw17[%arg5 + %arg4 * 128 + 18923520] : memref<?xf32>
                              %srw27 = affine.load %srw17[%arg5 + %arg4 * 128 + 21288960] : memref<?xf32>
                              %srw28 = affine.load %srw17[%arg5 + %arg4 * 128 + 23654400] : memref<?xf32>
                              %srw29 = affine.load %srw17[%arg5 + %arg4 * 128 + 26019840] : memref<?xf32>
                              %srw30 = affine.load %srw17[%arg5 + %arg4 * 128 + 28385280] : memref<?xf32>
                              %srw31 = affine.load %srw17[%arg5 + %arg4 * 128 + 30750720] : memref<?xf32>
                              %srw32 = affine.load %srw17[%arg5 + %arg4 * 128 + 33116160] : memref<?xf32>
                              %srw33 = affine.load %srw17[%arg5 + %arg4 * 128 + 35481600] : memref<?xf32>
                              %srw34 = affine.load %srw17[%arg5 + %arg4 * 128 + 37847040] : memref<?xf32>
                              %srw35 = affine.load %srw17[%arg5 + %arg4 * 128 + 40212480] : memref<?xf32>
                              %srw36 = affine.load %srw17[%arg5 + %arg4 * 128 + 42577920] : memref<?xf32>
                              %srw37 = "enzymexla.pointer2memref"(%srsrcS) : (!llvm.ptr) -> memref<?xi8>
                              %srw38 = affine.load %srw37[%arg5 * 4 + %arg4 * 512 + 179773440] : memref<?xi8>
                              %srw39 = arith.extui %srw38 : i8 to i32
                              %srw40 = arith.andi %srw39, %c1_i32 : i32
                              %srw41 = arith.cmpi eq, %srw40, %c0_i32 : i32
                              %srw42:19 = scf.if %srw41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                                %srw44 = arith.addf %srw19, %srw18 fastmath<fast> : f32
                                %srw45 = arith.addf %srw44, %srw20 fastmath<fast> : f32
                                %srw46 = arith.addf %srw45, %srw21 fastmath<fast> : f32
                                %srw47 = arith.addf %srw46, %srw22 fastmath<fast> : f32
                                %srw48 = arith.addf %srw47, %srw23 fastmath<fast> : f32
                                %srw49 = arith.addf %srw48, %srw24 fastmath<fast> : f32
                                %srw50 = arith.addf %srw49, %srw25 fastmath<fast> : f32
                                %srw51 = arith.addf %srw50, %srw26 fastmath<fast> : f32
                                %srw52 = arith.addf %srw51, %srw27 fastmath<fast> : f32
                                %srw53 = arith.addf %srw52, %srw28 fastmath<fast> : f32
                                %srw54 = arith.addf %srw53, %srw29 fastmath<fast> : f32
                                %srw55 = arith.addf %srw54, %srw30 fastmath<fast> : f32
                                %srw56 = arith.addf %srw55, %srw31 fastmath<fast> : f32
                                %srw57 = arith.addf %srw56, %srw32 fastmath<fast> : f32
                                %srw58 = arith.addf %srw57, %srw33 fastmath<fast> : f32
                                %srw59 = arith.addf %srw58, %srw34 fastmath<fast> : f32
                                %srw60 = arith.addf %srw59, %srw35 fastmath<fast> : f32
                                %srw61 = arith.addf %srw60, %srw36 fastmath<fast> : f32
                                %srw62 = arith.addf %srw21, %srw25 fastmath<fast> : f32
                                %srw63 = arith.addf %srw22, %srw26 fastmath<fast> : f32
                                %srw64 = arith.addf %srw62, %srw27 fastmath<fast> : f32
                                %srw65 = arith.addf %srw63, %srw28 fastmath<fast> : f32
                                %srw66 = arith.subf %srw64, %srw65 fastmath<fast> : f32
                                %srw67 = arith.addf %srw66, %srw33 fastmath<fast> : f32
                                %srw68 = arith.addf %srw67, %srw34 fastmath<fast> : f32
                                %srw69 = arith.addf %srw35, %srw36 fastmath<fast> : f32
                                %srw70 = arith.subf %srw68, %srw69 fastmath<fast> : f32
                                %srw71 = arith.subf %srw19, %srw20 fastmath<fast> : f32
                                %srw72 = arith.addf %srw71, %srw25 fastmath<fast> : f32
                                %srw73 = arith.addf %srw72, %srw26 fastmath<fast> : f32
                                %srw74 = arith.addf %srw27, %srw28 fastmath<fast> : f32
                                %srw75 = arith.subf %srw73, %srw74 fastmath<fast> : f32
                                %srw76 = arith.addf %srw75, %srw29 fastmath<fast> : f32
                                %srw77 = arith.addf %srw76, %srw30 fastmath<fast> : f32
                                %srw78 = arith.addf %srw31, %srw32 fastmath<fast> : f32
                                %srw79 = arith.subf %srw77, %srw78 fastmath<fast> : f32
                                %srw80 = arith.addf %srw23, %srw29 fastmath<fast> : f32
                                %srw81 = arith.addf %srw24, %srw30 fastmath<fast> : f32
                                %srw82 = arith.addf %srw80, %srw31 fastmath<fast> : f32
                                %srw83 = arith.addf %srw81, %srw32 fastmath<fast> : f32
                                %srw84 = arith.addf %srw82, %srw33 fastmath<fast> : f32
                                %srw85 = arith.addf %srw83, %srw34 fastmath<fast> : f32
                                %srw86 = arith.addf %srw84, %srw35 fastmath<fast> : f32
                                %srw87 = arith.addf %srw85, %srw36 fastmath<fast> : f32
                                %srw88 = arith.subf %srw86, %srw87 fastmath<fast> : f32
                                %srw89 = arith.divf %srw70, %srw61 fastmath<fast> : f32
                                %srw90 = arith.divf %srw79, %srw61 fastmath<fast> : f32
                                %srw91 = arith.divf %srw88, %srw61 fastmath<fast> : f32
                                %srw92 = arith.andi %srw39, %c2_i32 : i32
                                %srw93 = arith.cmpi eq, %srw92, %c0_i32 : i32
                                %srw94 = arith.select %srw93, %srw89, %cst_11 : f32
                                %srw95 = arith.select %srw93, %srw90, %cst_10 : f32
                                %srw96 = arith.select %srw93, %srw91, %cst_9 : f32
                                %srw97 = arith.mulf %srw94, %srw94 fastmath<fast> : f32
                                %srw98 = arith.mulf %srw95, %srw95 fastmath<fast> : f32
                                %srw99 = arith.addf %srw97, %srw98 fastmath<fast> : f32
                                %srw100 = arith.mulf %srw96, %srw96 fastmath<fast> : f32
                                %srw101 = arith.addf %srw99, %srw100 fastmath<fast> : f32
                                %srw102 = arith.mulf %srw101, %cst_8 fastmath<fast> : f32
                                %srw103 = arith.addf %srw102, %cst_7 fastmath<fast> : f32
                                %srw104 = arith.mulf %srw61, %cst_6 fastmath<fast> : f32
                                %srw105 = arith.mulf %srw18, %cst_5 fastmath<fast> : f32
                                %srw106 = arith.mulf %srw104, %srw103 fastmath<fast> : f32
                                %srw107 = arith.subf %srw105, %srw106 fastmath<fast> : f32
                                %srw108 = arith.mulf %srw61, %cst_4 fastmath<fast> : f32
                                %srw109 = arith.mulf %srw19, %cst_3 fastmath<fast> : f32
                                %srw110 = arith.mulf %srw95, %cst_2 fastmath<fast> : f32
                                %srw111 = arith.addf %srw110, %cst_1 fastmath<fast> : f32
                                %srw112 = arith.mulf %srw111, %srw95 fastmath<fast> : f32
                                %srw113 = arith.subf %srw112, %srw103 fastmath<fast> : f32
                                %srw114 = arith.mulf %srw113, %srw108 fastmath<fast> : f32
                                %srw115 = arith.subf %srw114, %srw109 fastmath<fast> : f32
                                %srw116 = arith.mulf %srw20, %cst_3 fastmath<fast> : f32
                                %srw117 = arith.addf %srw110, %cst_0 fastmath<fast> : f32
                                %srw118 = arith.mulf %srw117, %srw95 fastmath<fast> : f32
                                %srw119 = arith.subf %srw118, %srw103 fastmath<fast> : f32
                                %srw120 = arith.mulf %srw119, %srw108 fastmath<fast> : f32
                                %srw121 = arith.subf %srw120, %srw116 fastmath<fast> : f32
                                %srw122 = arith.mulf %srw23, %cst_3 fastmath<fast> : f32
                                %srw123 = arith.mulf %srw96, %cst_2 fastmath<fast> : f32
                                %srw124 = arith.addf %srw123, %cst_1 fastmath<fast> : f32
                                %srw125 = arith.mulf %srw124, %srw96 fastmath<fast> : f32
                                %srw126 = arith.subf %srw125, %srw103 fastmath<fast> : f32
                                %srw127 = arith.mulf %srw126, %srw108 fastmath<fast> : f32
                                %srw128 = arith.subf %srw127, %srw122 fastmath<fast> : f32
                                %srw129 = arith.mulf %srw24, %cst_3 fastmath<fast> : f32
                                %srw130 = arith.addf %srw123, %cst_0 fastmath<fast> : f32
                                %srw131 = arith.mulf %srw130, %srw96 fastmath<fast> : f32
                                %srw132 = arith.subf %srw131, %srw103 fastmath<fast> : f32
                                %srw133 = arith.mulf %srw132, %srw108 fastmath<fast> : f32
                                %srw134 = arith.subf %srw133, %srw129 fastmath<fast> : f32
                                %srw135 = arith.mulf %srw21, %cst_3 fastmath<fast> : f32
                                %srw136 = arith.mulf %srw94, %cst_2 fastmath<fast> : f32
                                %srw137 = arith.addf %srw136, %cst_1 fastmath<fast> : f32
                                %srw138 = arith.mulf %srw137, %srw94 fastmath<fast> : f32
                                %srw139 = arith.subf %srw138, %srw103 fastmath<fast> : f32
                                %srw140 = arith.mulf %srw139, %srw108 fastmath<fast> : f32
                                %srw141 = arith.subf %srw140, %srw135 fastmath<fast> : f32
                                %srw142 = arith.mulf %srw22, %cst_3 fastmath<fast> : f32
                                %srw143 = arith.addf %srw136, %cst_0 fastmath<fast> : f32
                                %srw144 = arith.mulf %srw143, %srw94 fastmath<fast> : f32
                                %srw145 = arith.subf %srw144, %srw103 fastmath<fast> : f32
                                %srw146 = arith.mulf %srw145, %srw108 fastmath<fast> : f32
                                %srw147 = arith.subf %srw146, %srw142 fastmath<fast> : f32
                                %srw148 = arith.mulf %srw61, %cst fastmath<fast> : f32
                                %srw149 = arith.mulf %srw29, %cst_3 fastmath<fast> : f32
                                %srw150 = arith.addf %srw95, %srw96 fastmath<fast> : f32
                                %srw151 = arith.mulf %srw150, %cst_2 fastmath<fast> : f32
                                %srw152 = arith.addf %srw151, %cst_1 fastmath<fast> : f32
                                %srw153 = arith.mulf %srw152, %srw150 fastmath<fast> : f32
                                %srw154 = arith.subf %srw153, %srw103 fastmath<fast> : f32
                                %srw155 = arith.mulf %srw154, %srw148 fastmath<fast> : f32
                                %srw156 = arith.subf %srw155, %srw149 fastmath<fast> : f32
                                %srw157 = arith.mulf %srw30, %cst_3 fastmath<fast> : f32
                                %srw158 = arith.subf %srw95, %srw96 fastmath<fast> : f32
                                %srw159 = arith.mulf %srw158, %cst_2 fastmath<fast> : f32
                                %srw160 = arith.addf %srw159, %cst_1 fastmath<fast> : f32
                                %srw161 = arith.mulf %srw160, %srw158 fastmath<fast> : f32
                                %srw162 = arith.subf %srw161, %srw103 fastmath<fast> : f32
                                %srw163 = arith.mulf %srw162, %srw148 fastmath<fast> : f32
                                %srw164 = arith.subf %srw163, %srw157 fastmath<fast> : f32
                                %srw165 = arith.mulf %srw31, %cst_3 fastmath<fast> : f32
                                %srw166 = arith.subf %srw96, %srw95 fastmath<fast> : f32
                                %srw167 = arith.mulf %srw166, %cst_2 fastmath<fast> : f32
                                %srw168 = arith.addf %srw167, %cst_1 fastmath<fast> : f32
                                %srw169 = arith.mulf %srw168, %srw166 fastmath<fast> : f32
                                %srw170 = arith.subf %srw169, %srw103 fastmath<fast> : f32
                                %srw171 = arith.mulf %srw170, %srw148 fastmath<fast> : f32
                                %srw172 = arith.subf %srw171, %srw165 fastmath<fast> : f32
                                %srw173 = arith.mulf %srw32, %cst_3 fastmath<fast> : f32
                                %srw174 = arith.negf %srw150 fastmath<fast> : f32
                                %srw175 = arith.subf %cst_1, %srw151 fastmath<fast> : f32
                                %srw176 = arith.mulf %srw175, %srw174 fastmath<fast> : f32
                                %srw177 = arith.subf %srw176, %srw103 fastmath<fast> : f32
                                %srw178 = arith.mulf %srw177, %srw148 fastmath<fast> : f32
                                %srw179 = arith.subf %srw178, %srw173 fastmath<fast> : f32
                                %srw180 = arith.mulf %srw25, %cst_3 fastmath<fast> : f32
                                %srw181 = arith.addf %srw94, %srw95 fastmath<fast> : f32
                                %srw182 = arith.mulf %srw181, %cst_2 fastmath<fast> : f32
                                %srw183 = arith.addf %srw182, %cst_1 fastmath<fast> : f32
                                %srw184 = arith.mulf %srw183, %srw181 fastmath<fast> : f32
                                %srw185 = arith.subf %srw184, %srw103 fastmath<fast> : f32
                                %srw186 = arith.mulf %srw185, %srw148 fastmath<fast> : f32
                                %srw187 = arith.subf %srw186, %srw180 fastmath<fast> : f32
                                %srw188 = arith.mulf %srw27, %cst_3 fastmath<fast> : f32
                                %srw189 = arith.subf %srw94, %srw95 fastmath<fast> : f32
                                %srw190 = arith.mulf %srw189, %cst_2 fastmath<fast> : f32
                                %srw191 = arith.addf %srw190, %cst_1 fastmath<fast> : f32
                                %srw192 = arith.mulf %srw191, %srw189 fastmath<fast> : f32
                                %srw193 = arith.subf %srw192, %srw103 fastmath<fast> : f32
                                %srw194 = arith.mulf %srw193, %srw148 fastmath<fast> : f32
                                %srw195 = arith.subf %srw194, %srw188 fastmath<fast> : f32
                                %srw196 = arith.mulf %srw33, %cst_3 fastmath<fast> : f32
                                %srw197 = arith.addf %srw94, %srw96 fastmath<fast> : f32
                                %srw198 = arith.mulf %srw197, %cst_2 fastmath<fast> : f32
                                %srw199 = arith.addf %srw198, %cst_1 fastmath<fast> : f32
                                %srw200 = arith.mulf %srw199, %srw197 fastmath<fast> : f32
                                %srw201 = arith.subf %srw200, %srw103 fastmath<fast> : f32
                                %srw202 = arith.mulf %srw201, %srw148 fastmath<fast> : f32
                                %srw203 = arith.subf %srw202, %srw196 fastmath<fast> : f32
                                %srw204 = arith.mulf %srw34, %cst_3 fastmath<fast> : f32
                                %srw205 = arith.subf %srw94, %srw96 fastmath<fast> : f32
                                %srw206 = arith.mulf %srw205, %cst_2 fastmath<fast> : f32
                                %srw207 = arith.addf %srw206, %cst_1 fastmath<fast> : f32
                                %srw208 = arith.mulf %srw207, %srw205 fastmath<fast> : f32
                                %srw209 = arith.subf %srw208, %srw103 fastmath<fast> : f32
                                %srw210 = arith.mulf %srw209, %srw148 fastmath<fast> : f32
                                %srw211 = arith.subf %srw210, %srw204 fastmath<fast> : f32
                                %srw212 = arith.mulf %srw26, %cst_3 fastmath<fast> : f32
                                %srw213 = arith.negf %srw94 fastmath<fast> : f32
                                %srw214 = arith.subf %srw95, %srw94 fastmath<fast> : f32
                                %srw215 = arith.mulf %srw214, %cst_2 fastmath<fast> : f32
                                %srw216 = arith.addf %srw215, %cst_1 fastmath<fast> : f32
                                %srw217 = arith.mulf %srw216, %srw214 fastmath<fast> : f32
                                %srw218 = arith.subf %srw217, %srw103 fastmath<fast> : f32
                                %srw219 = arith.mulf %srw218, %srw148 fastmath<fast> : f32
                                %srw220 = arith.subf %srw219, %srw212 fastmath<fast> : f32
                                %srw221 = arith.mulf %srw28, %cst_3 fastmath<fast> : f32
                                %srw222 = arith.subf %srw213, %srw95 fastmath<fast> : f32
                                %srw223 = arith.mulf %srw222, %cst_2 fastmath<fast> : f32
                                %srw224 = arith.addf %srw223, %cst_1 fastmath<fast> : f32
                                %srw225 = arith.mulf %srw224, %srw222 fastmath<fast> : f32
                                %srw226 = arith.subf %srw225, %srw103 fastmath<fast> : f32
                                %srw227 = arith.mulf %srw226, %srw148 fastmath<fast> : f32
                                %srw228 = arith.subf %srw227, %srw221 fastmath<fast> : f32
                                %srw229 = arith.mulf %srw35, %cst_3 fastmath<fast> : f32
                                %srw230 = arith.subf %srw96, %srw94 fastmath<fast> : f32
                                %srw231 = arith.mulf %srw230, %cst_2 fastmath<fast> : f32
                                %srw232 = arith.addf %srw231, %cst_1 fastmath<fast> : f32
                                %srw233 = arith.mulf %srw232, %srw230 fastmath<fast> : f32
                                %srw234 = arith.subf %srw233, %srw103 fastmath<fast> : f32
                                %srw235 = arith.mulf %srw234, %srw148 fastmath<fast> : f32
                                %srw236 = arith.subf %srw235, %srw229 fastmath<fast> : f32
                                %srw237 = arith.mulf %srw36, %cst_3 fastmath<fast> : f32
                                %srw238 = arith.subf %srw213, %srw96 fastmath<fast> : f32
                                %srw239 = arith.mulf %srw238, %cst_2 fastmath<fast> : f32
                                %srw240 = arith.addf %srw239, %cst_1 fastmath<fast> : f32
                                %srw241 = arith.mulf %srw240, %srw238 fastmath<fast> : f32
                                %srw242 = arith.subf %srw241, %srw103 fastmath<fast> : f32
                                %srw243 = arith.mulf %srw242, %srw148 fastmath<fast> : f32
                                %srw244 = arith.subf %srw243, %srw237 fastmath<fast> : f32
                                scf.yield %srw107, %srw115, %srw121, %srw141, %srw147, %srw128, %srw134, %srw187, %srw220, %srw195, %srw228, %srw156, %srw164, %srw172, %srw179, %srw203, %srw211, %srw236, %srw244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              } else {
                                scf.yield %srw18, %srw20, %srw19, %srw22, %srw21, %srw24, %srw23, %srw28, %srw27, %srw26, %srw25, %srw32, %srw31, %srw30, %srw29, %srw36, %srw35, %srw34, %srw33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              }
                              %srw43 = "enzymexla.pointer2memref"(%srdstS) : (!llvm.ptr) -> memref<?xf32>
                              affine.store %srw42#0, %srw43[%arg5 + %arg4 * 128] : memref<?xf32>
                              affine.store %srw42#1, %srw43[%arg5 + %arg4 * 128 + 2365568] : memref<?xf32>
                              affine.store %srw42#2, %srw43[%arg5 + %arg4 * 128 + 4730752] : memref<?xf32>
                              affine.store %srw42#3, %srw43[%arg5 + %arg4 * 128 + 7096321] : memref<?xf32>
                              affine.store %srw42#4, %srw43[%arg5 + %arg4 * 128 + 9461759] : memref<?xf32>
                              affine.store %srw42#5, %srw43[%arg5 + %arg4 * 128 + 11842560] : memref<?xf32>
                              affine.store %srw42#6, %srw43[%arg5 + %arg4 * 128 + 14177280] : memref<?xf32>
                              affine.store %srw42#7, %srw43[%arg5 + %arg4 * 128 + 16558209] : memref<?xf32>
                              affine.store %srw42#8, %srw43[%arg5 + %arg4 * 128 + 18923647] : memref<?xf32>
                              affine.store %srw42#9, %srw43[%arg5 + %arg4 * 128 + 21288833] : memref<?xf32>
                              affine.store %srw42#10, %srw43[%arg5 + %arg4 * 128 + 23654271] : memref<?xf32>
                              affine.store %srw42#11, %srw43[%arg5 + %arg4 * 128 + 26035328] : memref<?xf32>
                              affine.store %srw42#12, %srw43[%arg5 + %arg4 * 128 + 28370048] : memref<?xf32>
                              affine.store %srw42#13, %srw43[%arg5 + %arg4 * 128 + 30765952] : memref<?xf32>
                              affine.store %srw42#14, %srw43[%arg5 + %arg4 * 128 + 33100672] : memref<?xf32>
                              affine.store %srw42#15, %srw43[%arg5 + %arg4 * 128 + 35496961] : memref<?xf32>
                              affine.store %srw42#16, %srw43[%arg5 + %arg4 * 128 + 37831681] : memref<?xf32>
                              affine.store %srw42#17, %srw43[%arg5 + %arg4 * 128 + 40227839] : memref<?xf32>
                              affine.store %srw42#18, %srw43[%arg5 + %arg4 * 128 + 42562559] : memref<?xf32>
                            }
                            "enzymexla.polygeist_yield"() : () -> ()
                }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
                async.yield
              }
            }
          }
        }
      }
      %sl_r8 = llvm.call @cudaStreamSynchronize(%sl_st) : (!llvm.ptr) -> i32
      %sl_r9 = llvm.call @cudaStreamDestroy(%sl_st) : (!llvm.ptr) -> i32
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
      %sl_z = arith.constant 0 : index
      %sl_rows = arith.constant 76800 : index
      %sl_one = llvm.mlir.constant(1 : i64) : i64
      %sl_zero64 = llvm.mlir.constant(0 : i64) : i64
      %sl_stp = llvm.alloca %sl_one x !llvm.ptr : (i64) -> !llvm.ptr
      %sl_gp = llvm.alloca %sl_one x !llvm.ptr : (i64) -> !llvm.ptr
      %sl_ep = llvm.alloca %sl_one x !llvm.ptr : (i64) -> !llvm.ptr
      %sl_r0 = llvm.call @cudaStreamCreateWithFlags(%sl_stp, %c1_i32) : (!llvm.ptr, i32) -> i32
      %sl_st = llvm.load %sl_stp : !llvm.ptr -> !llvm.ptr
      %sl_r1 = llvm.call @cudaDeviceSynchronize() : () -> i32
      %sl_nfull = arith.divui %sl_H, %sl_k : index
      %sl_Hfull = arith.muli %sl_nfull, %sl_k : index
      %sl_hasfull = arith.cmpi ugt, %sl_nfull, %sl_z : index
      scf.if %sl_hasfull {
        %sl_r2 = llvm.call @cudaStreamBeginCapture(%sl_st, %c2_i32) : (!llvm.ptr, i32) -> i32
        %sfnw0 = arith.addi %sl_ns, %sl_k : index
        %sfnw = arith.subi %sfnw0, %c1 : index
        scf.for %sfwv = %sl_z to %sfnw step %c1 {
          scf.for %sfj = %sl_z to %sl_k step %c1 {
            %sfi = arith.subi %sfwv, %sfj : index
            %sfge = arith.cmpi sge, %sfi, %sl_z : index
            %sflt = arith.cmpi slt, %sfi, %sl_ns : index
            %sfok = arith.andi %sfge, %sflt : i1
            scf.if %sfok {
              %sfs = arith.addi %sl_z, %sfj : index
              %sfpar = arith.remui %sfs, %c2 : index
              %sfeven = arith.cmpi eq, %sfpar, %sl_z : index
              %sfsrc = llvm.select %sfeven, %arg1, %arg3 : i1, !llvm.ptr
              %sfdst = llvm.select %sfeven, %arg3, %arg1 : i1, !llvm.ptr
              %sfoff = arith.muli %sfi, %sl_rows : index
              %sfo64 = arith.index_cast %sfoff : index to i64
              %sfsrcS = llvm.getelementptr %sfsrc[%sfo64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %sfdstS = llvm.getelementptr %sfdst[%sfo64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %sftok = "enzymexla.stream2token"(%sl_st) : (!llvm.ptr) -> !async.token
              %sfdone = async.execute [%sftok] {
                %sfwr = "enzymexla.gpu_wrapper"(%c120, %c5, %c1, %c120, %c1, %c1) ({
                            affine.parallel (%arg6, %arg7) = (0, 0) to (600, 120) {
                              llvm.intr.experimental.noalias.scope.decl #alias_scope4
                              llvm.intr.experimental.noalias.scope.decl #alias_scope5
                              %sfw17 = "enzymexla.pointer2memref"(%sfsrcS) : (!llvm.ptr) -> memref<?xf32>
                              %sfw18 = affine.load %sfw17[%arg7 + %arg6 * 128] : memref<?xf32>
                              %sfw19 = affine.load %sfw17[%arg7 + %arg6 * 128 + 2365440] : memref<?xf32>
                              %sfw20 = affine.load %sfw17[%arg7 + %arg6 * 128 + 4730880] : memref<?xf32>
                              %sfw21 = affine.load %sfw17[%arg7 + %arg6 * 128 + 7096320] : memref<?xf32>
                              %sfw22 = affine.load %sfw17[%arg7 + %arg6 * 128 + 9461760] : memref<?xf32>
                              %sfw23 = affine.load %sfw17[%arg7 + %arg6 * 128 + 11827200] : memref<?xf32>
                              %sfw24 = affine.load %sfw17[%arg7 + %arg6 * 128 + 14192640] : memref<?xf32>
                              %sfw25 = affine.load %sfw17[%arg7 + %arg6 * 128 + 16558080] : memref<?xf32>
                              %sfw26 = affine.load %sfw17[%arg7 + %arg6 * 128 + 18923520] : memref<?xf32>
                              %sfw27 = affine.load %sfw17[%arg7 + %arg6 * 128 + 21288960] : memref<?xf32>
                              %sfw28 = affine.load %sfw17[%arg7 + %arg6 * 128 + 23654400] : memref<?xf32>
                              %sfw29 = affine.load %sfw17[%arg7 + %arg6 * 128 + 26019840] : memref<?xf32>
                              %sfw30 = affine.load %sfw17[%arg7 + %arg6 * 128 + 28385280] : memref<?xf32>
                              %sfw31 = affine.load %sfw17[%arg7 + %arg6 * 128 + 30750720] : memref<?xf32>
                              %sfw32 = affine.load %sfw17[%arg7 + %arg6 * 128 + 33116160] : memref<?xf32>
                              %sfw33 = affine.load %sfw17[%arg7 + %arg6 * 128 + 35481600] : memref<?xf32>
                              %sfw34 = affine.load %sfw17[%arg7 + %arg6 * 128 + 37847040] : memref<?xf32>
                              %sfw35 = affine.load %sfw17[%arg7 + %arg6 * 128 + 40212480] : memref<?xf32>
                              %sfw36 = affine.load %sfw17[%arg7 + %arg6 * 128 + 42577920] : memref<?xf32>
                              %sfw37 = "enzymexla.pointer2memref"(%sfsrcS) : (!llvm.ptr) -> memref<?xi8>
                              %sfw38 = affine.load %sfw37[%arg7 * 4 + %arg6 * 512 + 179773440] : memref<?xi8>
                              %sfw39 = arith.extui %sfw38 : i8 to i32
                              %sfw40 = arith.andi %sfw39, %c1_i32 : i32
                              %sfw41 = arith.cmpi eq, %sfw40, %c0_i32 : i32
                              %sfw42:19 = scf.if %sfw41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                                %sfw44 = arith.addf %sfw19, %sfw18 fastmath<fast> : f32
                                %sfw45 = arith.addf %sfw44, %sfw20 fastmath<fast> : f32
                                %sfw46 = arith.addf %sfw45, %sfw21 fastmath<fast> : f32
                                %sfw47 = arith.addf %sfw46, %sfw22 fastmath<fast> : f32
                                %sfw48 = arith.addf %sfw47, %sfw23 fastmath<fast> : f32
                                %sfw49 = arith.addf %sfw48, %sfw24 fastmath<fast> : f32
                                %sfw50 = arith.addf %sfw49, %sfw25 fastmath<fast> : f32
                                %sfw51 = arith.addf %sfw50, %sfw26 fastmath<fast> : f32
                                %sfw52 = arith.addf %sfw51, %sfw27 fastmath<fast> : f32
                                %sfw53 = arith.addf %sfw52, %sfw28 fastmath<fast> : f32
                                %sfw54 = arith.addf %sfw53, %sfw29 fastmath<fast> : f32
                                %sfw55 = arith.addf %sfw54, %sfw30 fastmath<fast> : f32
                                %sfw56 = arith.addf %sfw55, %sfw31 fastmath<fast> : f32
                                %sfw57 = arith.addf %sfw56, %sfw32 fastmath<fast> : f32
                                %sfw58 = arith.addf %sfw57, %sfw33 fastmath<fast> : f32
                                %sfw59 = arith.addf %sfw58, %sfw34 fastmath<fast> : f32
                                %sfw60 = arith.addf %sfw59, %sfw35 fastmath<fast> : f32
                                %sfw61 = arith.addf %sfw60, %sfw36 fastmath<fast> : f32
                                %sfw62 = arith.addf %sfw21, %sfw25 fastmath<fast> : f32
                                %sfw63 = arith.addf %sfw22, %sfw26 fastmath<fast> : f32
                                %sfw64 = arith.addf %sfw62, %sfw27 fastmath<fast> : f32
                                %sfw65 = arith.addf %sfw63, %sfw28 fastmath<fast> : f32
                                %sfw66 = arith.subf %sfw64, %sfw65 fastmath<fast> : f32
                                %sfw67 = arith.addf %sfw66, %sfw33 fastmath<fast> : f32
                                %sfw68 = arith.addf %sfw67, %sfw34 fastmath<fast> : f32
                                %sfw69 = arith.addf %sfw35, %sfw36 fastmath<fast> : f32
                                %sfw70 = arith.subf %sfw68, %sfw69 fastmath<fast> : f32
                                %sfw71 = arith.subf %sfw19, %sfw20 fastmath<fast> : f32
                                %sfw72 = arith.addf %sfw71, %sfw25 fastmath<fast> : f32
                                %sfw73 = arith.addf %sfw72, %sfw26 fastmath<fast> : f32
                                %sfw74 = arith.addf %sfw27, %sfw28 fastmath<fast> : f32
                                %sfw75 = arith.subf %sfw73, %sfw74 fastmath<fast> : f32
                                %sfw76 = arith.addf %sfw75, %sfw29 fastmath<fast> : f32
                                %sfw77 = arith.addf %sfw76, %sfw30 fastmath<fast> : f32
                                %sfw78 = arith.addf %sfw31, %sfw32 fastmath<fast> : f32
                                %sfw79 = arith.subf %sfw77, %sfw78 fastmath<fast> : f32
                                %sfw80 = arith.addf %sfw23, %sfw29 fastmath<fast> : f32
                                %sfw81 = arith.addf %sfw24, %sfw30 fastmath<fast> : f32
                                %sfw82 = arith.addf %sfw80, %sfw31 fastmath<fast> : f32
                                %sfw83 = arith.addf %sfw81, %sfw32 fastmath<fast> : f32
                                %sfw84 = arith.addf %sfw82, %sfw33 fastmath<fast> : f32
                                %sfw85 = arith.addf %sfw83, %sfw34 fastmath<fast> : f32
                                %sfw86 = arith.addf %sfw84, %sfw35 fastmath<fast> : f32
                                %sfw87 = arith.addf %sfw85, %sfw36 fastmath<fast> : f32
                                %sfw88 = arith.subf %sfw86, %sfw87 fastmath<fast> : f32
                                %sfw89 = arith.divf %sfw70, %sfw61 fastmath<fast> : f32
                                %sfw90 = arith.divf %sfw79, %sfw61 fastmath<fast> : f32
                                %sfw91 = arith.divf %sfw88, %sfw61 fastmath<fast> : f32
                                %sfw92 = arith.andi %sfw39, %c2_i32 : i32
                                %sfw93 = arith.cmpi eq, %sfw92, %c0_i32 : i32
                                %sfw94 = arith.select %sfw93, %sfw89, %cst_11 : f32
                                %sfw95 = arith.select %sfw93, %sfw90, %cst_10 : f32
                                %sfw96 = arith.select %sfw93, %sfw91, %cst_9 : f32
                                %sfw97 = arith.mulf %sfw94, %sfw94 fastmath<fast> : f32
                                %sfw98 = arith.mulf %sfw95, %sfw95 fastmath<fast> : f32
                                %sfw99 = arith.addf %sfw97, %sfw98 fastmath<fast> : f32
                                %sfw100 = arith.mulf %sfw96, %sfw96 fastmath<fast> : f32
                                %sfw101 = arith.addf %sfw99, %sfw100 fastmath<fast> : f32
                                %sfw102 = arith.mulf %sfw101, %cst_8 fastmath<fast> : f32
                                %sfw103 = arith.addf %sfw102, %cst_7 fastmath<fast> : f32
                                %sfw104 = arith.mulf %sfw61, %cst_6 fastmath<fast> : f32
                                %sfw105 = arith.mulf %sfw18, %cst_5 fastmath<fast> : f32
                                %sfw106 = arith.mulf %sfw104, %sfw103 fastmath<fast> : f32
                                %sfw107 = arith.subf %sfw105, %sfw106 fastmath<fast> : f32
                                %sfw108 = arith.mulf %sfw61, %cst_4 fastmath<fast> : f32
                                %sfw109 = arith.mulf %sfw19, %cst_3 fastmath<fast> : f32
                                %sfw110 = arith.mulf %sfw95, %cst_2 fastmath<fast> : f32
                                %sfw111 = arith.addf %sfw110, %cst_1 fastmath<fast> : f32
                                %sfw112 = arith.mulf %sfw111, %sfw95 fastmath<fast> : f32
                                %sfw113 = arith.subf %sfw112, %sfw103 fastmath<fast> : f32
                                %sfw114 = arith.mulf %sfw113, %sfw108 fastmath<fast> : f32
                                %sfw115 = arith.subf %sfw114, %sfw109 fastmath<fast> : f32
                                %sfw116 = arith.mulf %sfw20, %cst_3 fastmath<fast> : f32
                                %sfw117 = arith.addf %sfw110, %cst_0 fastmath<fast> : f32
                                %sfw118 = arith.mulf %sfw117, %sfw95 fastmath<fast> : f32
                                %sfw119 = arith.subf %sfw118, %sfw103 fastmath<fast> : f32
                                %sfw120 = arith.mulf %sfw119, %sfw108 fastmath<fast> : f32
                                %sfw121 = arith.subf %sfw120, %sfw116 fastmath<fast> : f32
                                %sfw122 = arith.mulf %sfw23, %cst_3 fastmath<fast> : f32
                                %sfw123 = arith.mulf %sfw96, %cst_2 fastmath<fast> : f32
                                %sfw124 = arith.addf %sfw123, %cst_1 fastmath<fast> : f32
                                %sfw125 = arith.mulf %sfw124, %sfw96 fastmath<fast> : f32
                                %sfw126 = arith.subf %sfw125, %sfw103 fastmath<fast> : f32
                                %sfw127 = arith.mulf %sfw126, %sfw108 fastmath<fast> : f32
                                %sfw128 = arith.subf %sfw127, %sfw122 fastmath<fast> : f32
                                %sfw129 = arith.mulf %sfw24, %cst_3 fastmath<fast> : f32
                                %sfw130 = arith.addf %sfw123, %cst_0 fastmath<fast> : f32
                                %sfw131 = arith.mulf %sfw130, %sfw96 fastmath<fast> : f32
                                %sfw132 = arith.subf %sfw131, %sfw103 fastmath<fast> : f32
                                %sfw133 = arith.mulf %sfw132, %sfw108 fastmath<fast> : f32
                                %sfw134 = arith.subf %sfw133, %sfw129 fastmath<fast> : f32
                                %sfw135 = arith.mulf %sfw21, %cst_3 fastmath<fast> : f32
                                %sfw136 = arith.mulf %sfw94, %cst_2 fastmath<fast> : f32
                                %sfw137 = arith.addf %sfw136, %cst_1 fastmath<fast> : f32
                                %sfw138 = arith.mulf %sfw137, %sfw94 fastmath<fast> : f32
                                %sfw139 = arith.subf %sfw138, %sfw103 fastmath<fast> : f32
                                %sfw140 = arith.mulf %sfw139, %sfw108 fastmath<fast> : f32
                                %sfw141 = arith.subf %sfw140, %sfw135 fastmath<fast> : f32
                                %sfw142 = arith.mulf %sfw22, %cst_3 fastmath<fast> : f32
                                %sfw143 = arith.addf %sfw136, %cst_0 fastmath<fast> : f32
                                %sfw144 = arith.mulf %sfw143, %sfw94 fastmath<fast> : f32
                                %sfw145 = arith.subf %sfw144, %sfw103 fastmath<fast> : f32
                                %sfw146 = arith.mulf %sfw145, %sfw108 fastmath<fast> : f32
                                %sfw147 = arith.subf %sfw146, %sfw142 fastmath<fast> : f32
                                %sfw148 = arith.mulf %sfw61, %cst fastmath<fast> : f32
                                %sfw149 = arith.mulf %sfw29, %cst_3 fastmath<fast> : f32
                                %sfw150 = arith.addf %sfw95, %sfw96 fastmath<fast> : f32
                                %sfw151 = arith.mulf %sfw150, %cst_2 fastmath<fast> : f32
                                %sfw152 = arith.addf %sfw151, %cst_1 fastmath<fast> : f32
                                %sfw153 = arith.mulf %sfw152, %sfw150 fastmath<fast> : f32
                                %sfw154 = arith.subf %sfw153, %sfw103 fastmath<fast> : f32
                                %sfw155 = arith.mulf %sfw154, %sfw148 fastmath<fast> : f32
                                %sfw156 = arith.subf %sfw155, %sfw149 fastmath<fast> : f32
                                %sfw157 = arith.mulf %sfw30, %cst_3 fastmath<fast> : f32
                                %sfw158 = arith.subf %sfw95, %sfw96 fastmath<fast> : f32
                                %sfw159 = arith.mulf %sfw158, %cst_2 fastmath<fast> : f32
                                %sfw160 = arith.addf %sfw159, %cst_1 fastmath<fast> : f32
                                %sfw161 = arith.mulf %sfw160, %sfw158 fastmath<fast> : f32
                                %sfw162 = arith.subf %sfw161, %sfw103 fastmath<fast> : f32
                                %sfw163 = arith.mulf %sfw162, %sfw148 fastmath<fast> : f32
                                %sfw164 = arith.subf %sfw163, %sfw157 fastmath<fast> : f32
                                %sfw165 = arith.mulf %sfw31, %cst_3 fastmath<fast> : f32
                                %sfw166 = arith.subf %sfw96, %sfw95 fastmath<fast> : f32
                                %sfw167 = arith.mulf %sfw166, %cst_2 fastmath<fast> : f32
                                %sfw168 = arith.addf %sfw167, %cst_1 fastmath<fast> : f32
                                %sfw169 = arith.mulf %sfw168, %sfw166 fastmath<fast> : f32
                                %sfw170 = arith.subf %sfw169, %sfw103 fastmath<fast> : f32
                                %sfw171 = arith.mulf %sfw170, %sfw148 fastmath<fast> : f32
                                %sfw172 = arith.subf %sfw171, %sfw165 fastmath<fast> : f32
                                %sfw173 = arith.mulf %sfw32, %cst_3 fastmath<fast> : f32
                                %sfw174 = arith.negf %sfw150 fastmath<fast> : f32
                                %sfw175 = arith.subf %cst_1, %sfw151 fastmath<fast> : f32
                                %sfw176 = arith.mulf %sfw175, %sfw174 fastmath<fast> : f32
                                %sfw177 = arith.subf %sfw176, %sfw103 fastmath<fast> : f32
                                %sfw178 = arith.mulf %sfw177, %sfw148 fastmath<fast> : f32
                                %sfw179 = arith.subf %sfw178, %sfw173 fastmath<fast> : f32
                                %sfw180 = arith.mulf %sfw25, %cst_3 fastmath<fast> : f32
                                %sfw181 = arith.addf %sfw94, %sfw95 fastmath<fast> : f32
                                %sfw182 = arith.mulf %sfw181, %cst_2 fastmath<fast> : f32
                                %sfw183 = arith.addf %sfw182, %cst_1 fastmath<fast> : f32
                                %sfw184 = arith.mulf %sfw183, %sfw181 fastmath<fast> : f32
                                %sfw185 = arith.subf %sfw184, %sfw103 fastmath<fast> : f32
                                %sfw186 = arith.mulf %sfw185, %sfw148 fastmath<fast> : f32
                                %sfw187 = arith.subf %sfw186, %sfw180 fastmath<fast> : f32
                                %sfw188 = arith.mulf %sfw27, %cst_3 fastmath<fast> : f32
                                %sfw189 = arith.subf %sfw94, %sfw95 fastmath<fast> : f32
                                %sfw190 = arith.mulf %sfw189, %cst_2 fastmath<fast> : f32
                                %sfw191 = arith.addf %sfw190, %cst_1 fastmath<fast> : f32
                                %sfw192 = arith.mulf %sfw191, %sfw189 fastmath<fast> : f32
                                %sfw193 = arith.subf %sfw192, %sfw103 fastmath<fast> : f32
                                %sfw194 = arith.mulf %sfw193, %sfw148 fastmath<fast> : f32
                                %sfw195 = arith.subf %sfw194, %sfw188 fastmath<fast> : f32
                                %sfw196 = arith.mulf %sfw33, %cst_3 fastmath<fast> : f32
                                %sfw197 = arith.addf %sfw94, %sfw96 fastmath<fast> : f32
                                %sfw198 = arith.mulf %sfw197, %cst_2 fastmath<fast> : f32
                                %sfw199 = arith.addf %sfw198, %cst_1 fastmath<fast> : f32
                                %sfw200 = arith.mulf %sfw199, %sfw197 fastmath<fast> : f32
                                %sfw201 = arith.subf %sfw200, %sfw103 fastmath<fast> : f32
                                %sfw202 = arith.mulf %sfw201, %sfw148 fastmath<fast> : f32
                                %sfw203 = arith.subf %sfw202, %sfw196 fastmath<fast> : f32
                                %sfw204 = arith.mulf %sfw34, %cst_3 fastmath<fast> : f32
                                %sfw205 = arith.subf %sfw94, %sfw96 fastmath<fast> : f32
                                %sfw206 = arith.mulf %sfw205, %cst_2 fastmath<fast> : f32
                                %sfw207 = arith.addf %sfw206, %cst_1 fastmath<fast> : f32
                                %sfw208 = arith.mulf %sfw207, %sfw205 fastmath<fast> : f32
                                %sfw209 = arith.subf %sfw208, %sfw103 fastmath<fast> : f32
                                %sfw210 = arith.mulf %sfw209, %sfw148 fastmath<fast> : f32
                                %sfw211 = arith.subf %sfw210, %sfw204 fastmath<fast> : f32
                                %sfw212 = arith.mulf %sfw26, %cst_3 fastmath<fast> : f32
                                %sfw213 = arith.negf %sfw94 fastmath<fast> : f32
                                %sfw214 = arith.subf %sfw95, %sfw94 fastmath<fast> : f32
                                %sfw215 = arith.mulf %sfw214, %cst_2 fastmath<fast> : f32
                                %sfw216 = arith.addf %sfw215, %cst_1 fastmath<fast> : f32
                                %sfw217 = arith.mulf %sfw216, %sfw214 fastmath<fast> : f32
                                %sfw218 = arith.subf %sfw217, %sfw103 fastmath<fast> : f32
                                %sfw219 = arith.mulf %sfw218, %sfw148 fastmath<fast> : f32
                                %sfw220 = arith.subf %sfw219, %sfw212 fastmath<fast> : f32
                                %sfw221 = arith.mulf %sfw28, %cst_3 fastmath<fast> : f32
                                %sfw222 = arith.subf %sfw213, %sfw95 fastmath<fast> : f32
                                %sfw223 = arith.mulf %sfw222, %cst_2 fastmath<fast> : f32
                                %sfw224 = arith.addf %sfw223, %cst_1 fastmath<fast> : f32
                                %sfw225 = arith.mulf %sfw224, %sfw222 fastmath<fast> : f32
                                %sfw226 = arith.subf %sfw225, %sfw103 fastmath<fast> : f32
                                %sfw227 = arith.mulf %sfw226, %sfw148 fastmath<fast> : f32
                                %sfw228 = arith.subf %sfw227, %sfw221 fastmath<fast> : f32
                                %sfw229 = arith.mulf %sfw35, %cst_3 fastmath<fast> : f32
                                %sfw230 = arith.subf %sfw96, %sfw94 fastmath<fast> : f32
                                %sfw231 = arith.mulf %sfw230, %cst_2 fastmath<fast> : f32
                                %sfw232 = arith.addf %sfw231, %cst_1 fastmath<fast> : f32
                                %sfw233 = arith.mulf %sfw232, %sfw230 fastmath<fast> : f32
                                %sfw234 = arith.subf %sfw233, %sfw103 fastmath<fast> : f32
                                %sfw235 = arith.mulf %sfw234, %sfw148 fastmath<fast> : f32
                                %sfw236 = arith.subf %sfw235, %sfw229 fastmath<fast> : f32
                                %sfw237 = arith.mulf %sfw36, %cst_3 fastmath<fast> : f32
                                %sfw238 = arith.subf %sfw213, %sfw96 fastmath<fast> : f32
                                %sfw239 = arith.mulf %sfw238, %cst_2 fastmath<fast> : f32
                                %sfw240 = arith.addf %sfw239, %cst_1 fastmath<fast> : f32
                                %sfw241 = arith.mulf %sfw240, %sfw238 fastmath<fast> : f32
                                %sfw242 = arith.subf %sfw241, %sfw103 fastmath<fast> : f32
                                %sfw243 = arith.mulf %sfw242, %sfw148 fastmath<fast> : f32
                                %sfw244 = arith.subf %sfw243, %sfw237 fastmath<fast> : f32
                                scf.yield %sfw107, %sfw115, %sfw121, %sfw141, %sfw147, %sfw128, %sfw134, %sfw187, %sfw220, %sfw195, %sfw228, %sfw156, %sfw164, %sfw172, %sfw179, %sfw203, %sfw211, %sfw236, %sfw244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              } else {
                                scf.yield %sfw18, %sfw20, %sfw19, %sfw22, %sfw21, %sfw24, %sfw23, %sfw28, %sfw27, %sfw26, %sfw25, %sfw32, %sfw31, %sfw30, %sfw29, %sfw36, %sfw35, %sfw34, %sfw33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              }
                              %sfw43 = "enzymexla.pointer2memref"(%sfdstS) : (!llvm.ptr) -> memref<?xf32>
                              affine.store %sfw42#0, %sfw43[%arg7 + %arg6 * 128] : memref<?xf32>
                              affine.store %sfw42#1, %sfw43[%arg7 + %arg6 * 128 + 2365568] : memref<?xf32>
                              affine.store %sfw42#2, %sfw43[%arg7 + %arg6 * 128 + 4730752] : memref<?xf32>
                              affine.store %sfw42#3, %sfw43[%arg7 + %arg6 * 128 + 7096321] : memref<?xf32>
                              affine.store %sfw42#4, %sfw43[%arg7 + %arg6 * 128 + 9461759] : memref<?xf32>
                              affine.store %sfw42#5, %sfw43[%arg7 + %arg6 * 128 + 11842560] : memref<?xf32>
                              affine.store %sfw42#6, %sfw43[%arg7 + %arg6 * 128 + 14177280] : memref<?xf32>
                              affine.store %sfw42#7, %sfw43[%arg7 + %arg6 * 128 + 16558209] : memref<?xf32>
                              affine.store %sfw42#8, %sfw43[%arg7 + %arg6 * 128 + 18923647] : memref<?xf32>
                              affine.store %sfw42#9, %sfw43[%arg7 + %arg6 * 128 + 21288833] : memref<?xf32>
                              affine.store %sfw42#10, %sfw43[%arg7 + %arg6 * 128 + 23654271] : memref<?xf32>
                              affine.store %sfw42#11, %sfw43[%arg7 + %arg6 * 128 + 26035328] : memref<?xf32>
                              affine.store %sfw42#12, %sfw43[%arg7 + %arg6 * 128 + 28370048] : memref<?xf32>
                              affine.store %sfw42#13, %sfw43[%arg7 + %arg6 * 128 + 30765952] : memref<?xf32>
                              affine.store %sfw42#14, %sfw43[%arg7 + %arg6 * 128 + 33100672] : memref<?xf32>
                              affine.store %sfw42#15, %sfw43[%arg7 + %arg6 * 128 + 35496961] : memref<?xf32>
                              affine.store %sfw42#16, %sfw43[%arg7 + %arg6 * 128 + 37831681] : memref<?xf32>
                              affine.store %sfw42#17, %sfw43[%arg7 + %arg6 * 128 + 40227839] : memref<?xf32>
                              affine.store %sfw42#18, %sfw43[%arg7 + %arg6 * 128 + 42562559] : memref<?xf32>
                            }
                            "enzymexla.polygeist_yield"() : () -> ()
                }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
                async.yield
              }
            }
          }
        }
        %sl_r3 = llvm.call @cudaStreamEndCapture(%sl_st, %sl_gp) : (!llvm.ptr, !llvm.ptr) -> i32
        %sl_g = llvm.load %sl_gp : !llvm.ptr -> !llvm.ptr
        %sl_r4 = llvm.call @cudaGraphInstantiate(%sl_ep, %sl_g, %sl_zero64) : (!llvm.ptr, !llvm.ptr, i64) -> i32
        %sl_e = llvm.load %sl_ep : !llvm.ptr -> !llvm.ptr
        scf.for %sl_b = %sl_z to %sl_nfull step %c1 {
          %sl_r5 = llvm.call @cudaGraphLaunch(%sl_e, %sl_st) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        %sl_r6 = llvm.call @cudaGraphExecDestroy(%sl_e) : (!llvm.ptr) -> i32
        %sl_r7 = llvm.call @cudaGraphDestroy(%sl_g) : (!llvm.ptr) -> i32
      }
      %sl_rem = arith.subi %sl_H, %sl_Hfull : index
      %sl_hasrem = arith.cmpi ugt, %sl_rem, %sl_z : index
      scf.if %sl_hasrem {
        %srnw0 = arith.addi %sl_ns, %sl_rem : index
        %srnw = arith.subi %srnw0, %c1 : index
        scf.for %srwv = %sl_z to %srnw step %c1 {
          scf.for %srj = %sl_z to %sl_rem step %c1 {
            %sri = arith.subi %srwv, %srj : index
            %srge = arith.cmpi sge, %sri, %sl_z : index
            %srlt = arith.cmpi slt, %sri, %sl_ns : index
            %srok = arith.andi %srge, %srlt : i1
            scf.if %srok {
              %srs = arith.addi %sl_Hfull, %srj : index
              %srpar = arith.remui %srs, %c2 : index
              %sreven = arith.cmpi eq, %srpar, %sl_z : index
              %srsrc = llvm.select %sreven, %arg1, %arg3 : i1, !llvm.ptr
              %srdst = llvm.select %sreven, %arg3, %arg1 : i1, !llvm.ptr
              %sroff = arith.muli %sri, %sl_rows : index
              %sro64 = arith.index_cast %sroff : index to i64
              %srsrcS = llvm.getelementptr %srsrc[%sro64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %srdstS = llvm.getelementptr %srdst[%sro64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
              %srtok = "enzymexla.stream2token"(%sl_st) : (!llvm.ptr) -> !async.token
              %srdone = async.execute [%srtok] {
                %srwr = "enzymexla.gpu_wrapper"(%c120, %c5, %c1, %c120, %c1, %c1) ({
                            affine.parallel (%arg6, %arg7) = (0, 0) to (600, 120) {
                              llvm.intr.experimental.noalias.scope.decl #alias_scope4
                              llvm.intr.experimental.noalias.scope.decl #alias_scope5
                              %srw17 = "enzymexla.pointer2memref"(%srsrcS) : (!llvm.ptr) -> memref<?xf32>
                              %srw18 = affine.load %srw17[%arg7 + %arg6 * 128] : memref<?xf32>
                              %srw19 = affine.load %srw17[%arg7 + %arg6 * 128 + 2365440] : memref<?xf32>
                              %srw20 = affine.load %srw17[%arg7 + %arg6 * 128 + 4730880] : memref<?xf32>
                              %srw21 = affine.load %srw17[%arg7 + %arg6 * 128 + 7096320] : memref<?xf32>
                              %srw22 = affine.load %srw17[%arg7 + %arg6 * 128 + 9461760] : memref<?xf32>
                              %srw23 = affine.load %srw17[%arg7 + %arg6 * 128 + 11827200] : memref<?xf32>
                              %srw24 = affine.load %srw17[%arg7 + %arg6 * 128 + 14192640] : memref<?xf32>
                              %srw25 = affine.load %srw17[%arg7 + %arg6 * 128 + 16558080] : memref<?xf32>
                              %srw26 = affine.load %srw17[%arg7 + %arg6 * 128 + 18923520] : memref<?xf32>
                              %srw27 = affine.load %srw17[%arg7 + %arg6 * 128 + 21288960] : memref<?xf32>
                              %srw28 = affine.load %srw17[%arg7 + %arg6 * 128 + 23654400] : memref<?xf32>
                              %srw29 = affine.load %srw17[%arg7 + %arg6 * 128 + 26019840] : memref<?xf32>
                              %srw30 = affine.load %srw17[%arg7 + %arg6 * 128 + 28385280] : memref<?xf32>
                              %srw31 = affine.load %srw17[%arg7 + %arg6 * 128 + 30750720] : memref<?xf32>
                              %srw32 = affine.load %srw17[%arg7 + %arg6 * 128 + 33116160] : memref<?xf32>
                              %srw33 = affine.load %srw17[%arg7 + %arg6 * 128 + 35481600] : memref<?xf32>
                              %srw34 = affine.load %srw17[%arg7 + %arg6 * 128 + 37847040] : memref<?xf32>
                              %srw35 = affine.load %srw17[%arg7 + %arg6 * 128 + 40212480] : memref<?xf32>
                              %srw36 = affine.load %srw17[%arg7 + %arg6 * 128 + 42577920] : memref<?xf32>
                              %srw37 = "enzymexla.pointer2memref"(%srsrcS) : (!llvm.ptr) -> memref<?xi8>
                              %srw38 = affine.load %srw37[%arg7 * 4 + %arg6 * 512 + 179773440] : memref<?xi8>
                              %srw39 = arith.extui %srw38 : i8 to i32
                              %srw40 = arith.andi %srw39, %c1_i32 : i32
                              %srw41 = arith.cmpi eq, %srw40, %c0_i32 : i32
                              %srw42:19 = scf.if %srw41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                                %srw44 = arith.addf %srw19, %srw18 fastmath<fast> : f32
                                %srw45 = arith.addf %srw44, %srw20 fastmath<fast> : f32
                                %srw46 = arith.addf %srw45, %srw21 fastmath<fast> : f32
                                %srw47 = arith.addf %srw46, %srw22 fastmath<fast> : f32
                                %srw48 = arith.addf %srw47, %srw23 fastmath<fast> : f32
                                %srw49 = arith.addf %srw48, %srw24 fastmath<fast> : f32
                                %srw50 = arith.addf %srw49, %srw25 fastmath<fast> : f32
                                %srw51 = arith.addf %srw50, %srw26 fastmath<fast> : f32
                                %srw52 = arith.addf %srw51, %srw27 fastmath<fast> : f32
                                %srw53 = arith.addf %srw52, %srw28 fastmath<fast> : f32
                                %srw54 = arith.addf %srw53, %srw29 fastmath<fast> : f32
                                %srw55 = arith.addf %srw54, %srw30 fastmath<fast> : f32
                                %srw56 = arith.addf %srw55, %srw31 fastmath<fast> : f32
                                %srw57 = arith.addf %srw56, %srw32 fastmath<fast> : f32
                                %srw58 = arith.addf %srw57, %srw33 fastmath<fast> : f32
                                %srw59 = arith.addf %srw58, %srw34 fastmath<fast> : f32
                                %srw60 = arith.addf %srw59, %srw35 fastmath<fast> : f32
                                %srw61 = arith.addf %srw60, %srw36 fastmath<fast> : f32
                                %srw62 = arith.addf %srw21, %srw25 fastmath<fast> : f32
                                %srw63 = arith.addf %srw22, %srw26 fastmath<fast> : f32
                                %srw64 = arith.addf %srw62, %srw27 fastmath<fast> : f32
                                %srw65 = arith.addf %srw63, %srw28 fastmath<fast> : f32
                                %srw66 = arith.subf %srw64, %srw65 fastmath<fast> : f32
                                %srw67 = arith.addf %srw66, %srw33 fastmath<fast> : f32
                                %srw68 = arith.addf %srw67, %srw34 fastmath<fast> : f32
                                %srw69 = arith.addf %srw35, %srw36 fastmath<fast> : f32
                                %srw70 = arith.subf %srw68, %srw69 fastmath<fast> : f32
                                %srw71 = arith.subf %srw19, %srw20 fastmath<fast> : f32
                                %srw72 = arith.addf %srw71, %srw25 fastmath<fast> : f32
                                %srw73 = arith.addf %srw72, %srw26 fastmath<fast> : f32
                                %srw74 = arith.addf %srw27, %srw28 fastmath<fast> : f32
                                %srw75 = arith.subf %srw73, %srw74 fastmath<fast> : f32
                                %srw76 = arith.addf %srw75, %srw29 fastmath<fast> : f32
                                %srw77 = arith.addf %srw76, %srw30 fastmath<fast> : f32
                                %srw78 = arith.addf %srw31, %srw32 fastmath<fast> : f32
                                %srw79 = arith.subf %srw77, %srw78 fastmath<fast> : f32
                                %srw80 = arith.addf %srw23, %srw29 fastmath<fast> : f32
                                %srw81 = arith.addf %srw24, %srw30 fastmath<fast> : f32
                                %srw82 = arith.addf %srw80, %srw31 fastmath<fast> : f32
                                %srw83 = arith.addf %srw81, %srw32 fastmath<fast> : f32
                                %srw84 = arith.addf %srw82, %srw33 fastmath<fast> : f32
                                %srw85 = arith.addf %srw83, %srw34 fastmath<fast> : f32
                                %srw86 = arith.addf %srw84, %srw35 fastmath<fast> : f32
                                %srw87 = arith.addf %srw85, %srw36 fastmath<fast> : f32
                                %srw88 = arith.subf %srw86, %srw87 fastmath<fast> : f32
                                %srw89 = arith.divf %srw70, %srw61 fastmath<fast> : f32
                                %srw90 = arith.divf %srw79, %srw61 fastmath<fast> : f32
                                %srw91 = arith.divf %srw88, %srw61 fastmath<fast> : f32
                                %srw92 = arith.andi %srw39, %c2_i32 : i32
                                %srw93 = arith.cmpi eq, %srw92, %c0_i32 : i32
                                %srw94 = arith.select %srw93, %srw89, %cst_11 : f32
                                %srw95 = arith.select %srw93, %srw90, %cst_10 : f32
                                %srw96 = arith.select %srw93, %srw91, %cst_9 : f32
                                %srw97 = arith.mulf %srw94, %srw94 fastmath<fast> : f32
                                %srw98 = arith.mulf %srw95, %srw95 fastmath<fast> : f32
                                %srw99 = arith.addf %srw97, %srw98 fastmath<fast> : f32
                                %srw100 = arith.mulf %srw96, %srw96 fastmath<fast> : f32
                                %srw101 = arith.addf %srw99, %srw100 fastmath<fast> : f32
                                %srw102 = arith.mulf %srw101, %cst_8 fastmath<fast> : f32
                                %srw103 = arith.addf %srw102, %cst_7 fastmath<fast> : f32
                                %srw104 = arith.mulf %srw61, %cst_6 fastmath<fast> : f32
                                %srw105 = arith.mulf %srw18, %cst_5 fastmath<fast> : f32
                                %srw106 = arith.mulf %srw104, %srw103 fastmath<fast> : f32
                                %srw107 = arith.subf %srw105, %srw106 fastmath<fast> : f32
                                %srw108 = arith.mulf %srw61, %cst_4 fastmath<fast> : f32
                                %srw109 = arith.mulf %srw19, %cst_3 fastmath<fast> : f32
                                %srw110 = arith.mulf %srw95, %cst_2 fastmath<fast> : f32
                                %srw111 = arith.addf %srw110, %cst_1 fastmath<fast> : f32
                                %srw112 = arith.mulf %srw111, %srw95 fastmath<fast> : f32
                                %srw113 = arith.subf %srw112, %srw103 fastmath<fast> : f32
                                %srw114 = arith.mulf %srw113, %srw108 fastmath<fast> : f32
                                %srw115 = arith.subf %srw114, %srw109 fastmath<fast> : f32
                                %srw116 = arith.mulf %srw20, %cst_3 fastmath<fast> : f32
                                %srw117 = arith.addf %srw110, %cst_0 fastmath<fast> : f32
                                %srw118 = arith.mulf %srw117, %srw95 fastmath<fast> : f32
                                %srw119 = arith.subf %srw118, %srw103 fastmath<fast> : f32
                                %srw120 = arith.mulf %srw119, %srw108 fastmath<fast> : f32
                                %srw121 = arith.subf %srw120, %srw116 fastmath<fast> : f32
                                %srw122 = arith.mulf %srw23, %cst_3 fastmath<fast> : f32
                                %srw123 = arith.mulf %srw96, %cst_2 fastmath<fast> : f32
                                %srw124 = arith.addf %srw123, %cst_1 fastmath<fast> : f32
                                %srw125 = arith.mulf %srw124, %srw96 fastmath<fast> : f32
                                %srw126 = arith.subf %srw125, %srw103 fastmath<fast> : f32
                                %srw127 = arith.mulf %srw126, %srw108 fastmath<fast> : f32
                                %srw128 = arith.subf %srw127, %srw122 fastmath<fast> : f32
                                %srw129 = arith.mulf %srw24, %cst_3 fastmath<fast> : f32
                                %srw130 = arith.addf %srw123, %cst_0 fastmath<fast> : f32
                                %srw131 = arith.mulf %srw130, %srw96 fastmath<fast> : f32
                                %srw132 = arith.subf %srw131, %srw103 fastmath<fast> : f32
                                %srw133 = arith.mulf %srw132, %srw108 fastmath<fast> : f32
                                %srw134 = arith.subf %srw133, %srw129 fastmath<fast> : f32
                                %srw135 = arith.mulf %srw21, %cst_3 fastmath<fast> : f32
                                %srw136 = arith.mulf %srw94, %cst_2 fastmath<fast> : f32
                                %srw137 = arith.addf %srw136, %cst_1 fastmath<fast> : f32
                                %srw138 = arith.mulf %srw137, %srw94 fastmath<fast> : f32
                                %srw139 = arith.subf %srw138, %srw103 fastmath<fast> : f32
                                %srw140 = arith.mulf %srw139, %srw108 fastmath<fast> : f32
                                %srw141 = arith.subf %srw140, %srw135 fastmath<fast> : f32
                                %srw142 = arith.mulf %srw22, %cst_3 fastmath<fast> : f32
                                %srw143 = arith.addf %srw136, %cst_0 fastmath<fast> : f32
                                %srw144 = arith.mulf %srw143, %srw94 fastmath<fast> : f32
                                %srw145 = arith.subf %srw144, %srw103 fastmath<fast> : f32
                                %srw146 = arith.mulf %srw145, %srw108 fastmath<fast> : f32
                                %srw147 = arith.subf %srw146, %srw142 fastmath<fast> : f32
                                %srw148 = arith.mulf %srw61, %cst fastmath<fast> : f32
                                %srw149 = arith.mulf %srw29, %cst_3 fastmath<fast> : f32
                                %srw150 = arith.addf %srw95, %srw96 fastmath<fast> : f32
                                %srw151 = arith.mulf %srw150, %cst_2 fastmath<fast> : f32
                                %srw152 = arith.addf %srw151, %cst_1 fastmath<fast> : f32
                                %srw153 = arith.mulf %srw152, %srw150 fastmath<fast> : f32
                                %srw154 = arith.subf %srw153, %srw103 fastmath<fast> : f32
                                %srw155 = arith.mulf %srw154, %srw148 fastmath<fast> : f32
                                %srw156 = arith.subf %srw155, %srw149 fastmath<fast> : f32
                                %srw157 = arith.mulf %srw30, %cst_3 fastmath<fast> : f32
                                %srw158 = arith.subf %srw95, %srw96 fastmath<fast> : f32
                                %srw159 = arith.mulf %srw158, %cst_2 fastmath<fast> : f32
                                %srw160 = arith.addf %srw159, %cst_1 fastmath<fast> : f32
                                %srw161 = arith.mulf %srw160, %srw158 fastmath<fast> : f32
                                %srw162 = arith.subf %srw161, %srw103 fastmath<fast> : f32
                                %srw163 = arith.mulf %srw162, %srw148 fastmath<fast> : f32
                                %srw164 = arith.subf %srw163, %srw157 fastmath<fast> : f32
                                %srw165 = arith.mulf %srw31, %cst_3 fastmath<fast> : f32
                                %srw166 = arith.subf %srw96, %srw95 fastmath<fast> : f32
                                %srw167 = arith.mulf %srw166, %cst_2 fastmath<fast> : f32
                                %srw168 = arith.addf %srw167, %cst_1 fastmath<fast> : f32
                                %srw169 = arith.mulf %srw168, %srw166 fastmath<fast> : f32
                                %srw170 = arith.subf %srw169, %srw103 fastmath<fast> : f32
                                %srw171 = arith.mulf %srw170, %srw148 fastmath<fast> : f32
                                %srw172 = arith.subf %srw171, %srw165 fastmath<fast> : f32
                                %srw173 = arith.mulf %srw32, %cst_3 fastmath<fast> : f32
                                %srw174 = arith.negf %srw150 fastmath<fast> : f32
                                %srw175 = arith.subf %cst_1, %srw151 fastmath<fast> : f32
                                %srw176 = arith.mulf %srw175, %srw174 fastmath<fast> : f32
                                %srw177 = arith.subf %srw176, %srw103 fastmath<fast> : f32
                                %srw178 = arith.mulf %srw177, %srw148 fastmath<fast> : f32
                                %srw179 = arith.subf %srw178, %srw173 fastmath<fast> : f32
                                %srw180 = arith.mulf %srw25, %cst_3 fastmath<fast> : f32
                                %srw181 = arith.addf %srw94, %srw95 fastmath<fast> : f32
                                %srw182 = arith.mulf %srw181, %cst_2 fastmath<fast> : f32
                                %srw183 = arith.addf %srw182, %cst_1 fastmath<fast> : f32
                                %srw184 = arith.mulf %srw183, %srw181 fastmath<fast> : f32
                                %srw185 = arith.subf %srw184, %srw103 fastmath<fast> : f32
                                %srw186 = arith.mulf %srw185, %srw148 fastmath<fast> : f32
                                %srw187 = arith.subf %srw186, %srw180 fastmath<fast> : f32
                                %srw188 = arith.mulf %srw27, %cst_3 fastmath<fast> : f32
                                %srw189 = arith.subf %srw94, %srw95 fastmath<fast> : f32
                                %srw190 = arith.mulf %srw189, %cst_2 fastmath<fast> : f32
                                %srw191 = arith.addf %srw190, %cst_1 fastmath<fast> : f32
                                %srw192 = arith.mulf %srw191, %srw189 fastmath<fast> : f32
                                %srw193 = arith.subf %srw192, %srw103 fastmath<fast> : f32
                                %srw194 = arith.mulf %srw193, %srw148 fastmath<fast> : f32
                                %srw195 = arith.subf %srw194, %srw188 fastmath<fast> : f32
                                %srw196 = arith.mulf %srw33, %cst_3 fastmath<fast> : f32
                                %srw197 = arith.addf %srw94, %srw96 fastmath<fast> : f32
                                %srw198 = arith.mulf %srw197, %cst_2 fastmath<fast> : f32
                                %srw199 = arith.addf %srw198, %cst_1 fastmath<fast> : f32
                                %srw200 = arith.mulf %srw199, %srw197 fastmath<fast> : f32
                                %srw201 = arith.subf %srw200, %srw103 fastmath<fast> : f32
                                %srw202 = arith.mulf %srw201, %srw148 fastmath<fast> : f32
                                %srw203 = arith.subf %srw202, %srw196 fastmath<fast> : f32
                                %srw204 = arith.mulf %srw34, %cst_3 fastmath<fast> : f32
                                %srw205 = arith.subf %srw94, %srw96 fastmath<fast> : f32
                                %srw206 = arith.mulf %srw205, %cst_2 fastmath<fast> : f32
                                %srw207 = arith.addf %srw206, %cst_1 fastmath<fast> : f32
                                %srw208 = arith.mulf %srw207, %srw205 fastmath<fast> : f32
                                %srw209 = arith.subf %srw208, %srw103 fastmath<fast> : f32
                                %srw210 = arith.mulf %srw209, %srw148 fastmath<fast> : f32
                                %srw211 = arith.subf %srw210, %srw204 fastmath<fast> : f32
                                %srw212 = arith.mulf %srw26, %cst_3 fastmath<fast> : f32
                                %srw213 = arith.negf %srw94 fastmath<fast> : f32
                                %srw214 = arith.subf %srw95, %srw94 fastmath<fast> : f32
                                %srw215 = arith.mulf %srw214, %cst_2 fastmath<fast> : f32
                                %srw216 = arith.addf %srw215, %cst_1 fastmath<fast> : f32
                                %srw217 = arith.mulf %srw216, %srw214 fastmath<fast> : f32
                                %srw218 = arith.subf %srw217, %srw103 fastmath<fast> : f32
                                %srw219 = arith.mulf %srw218, %srw148 fastmath<fast> : f32
                                %srw220 = arith.subf %srw219, %srw212 fastmath<fast> : f32
                                %srw221 = arith.mulf %srw28, %cst_3 fastmath<fast> : f32
                                %srw222 = arith.subf %srw213, %srw95 fastmath<fast> : f32
                                %srw223 = arith.mulf %srw222, %cst_2 fastmath<fast> : f32
                                %srw224 = arith.addf %srw223, %cst_1 fastmath<fast> : f32
                                %srw225 = arith.mulf %srw224, %srw222 fastmath<fast> : f32
                                %srw226 = arith.subf %srw225, %srw103 fastmath<fast> : f32
                                %srw227 = arith.mulf %srw226, %srw148 fastmath<fast> : f32
                                %srw228 = arith.subf %srw227, %srw221 fastmath<fast> : f32
                                %srw229 = arith.mulf %srw35, %cst_3 fastmath<fast> : f32
                                %srw230 = arith.subf %srw96, %srw94 fastmath<fast> : f32
                                %srw231 = arith.mulf %srw230, %cst_2 fastmath<fast> : f32
                                %srw232 = arith.addf %srw231, %cst_1 fastmath<fast> : f32
                                %srw233 = arith.mulf %srw232, %srw230 fastmath<fast> : f32
                                %srw234 = arith.subf %srw233, %srw103 fastmath<fast> : f32
                                %srw235 = arith.mulf %srw234, %srw148 fastmath<fast> : f32
                                %srw236 = arith.subf %srw235, %srw229 fastmath<fast> : f32
                                %srw237 = arith.mulf %srw36, %cst_3 fastmath<fast> : f32
                                %srw238 = arith.subf %srw213, %srw96 fastmath<fast> : f32
                                %srw239 = arith.mulf %srw238, %cst_2 fastmath<fast> : f32
                                %srw240 = arith.addf %srw239, %cst_1 fastmath<fast> : f32
                                %srw241 = arith.mulf %srw240, %srw238 fastmath<fast> : f32
                                %srw242 = arith.subf %srw241, %srw103 fastmath<fast> : f32
                                %srw243 = arith.mulf %srw242, %srw148 fastmath<fast> : f32
                                %srw244 = arith.subf %srw243, %srw237 fastmath<fast> : f32
                                scf.yield %srw107, %srw115, %srw121, %srw141, %srw147, %srw128, %srw134, %srw187, %srw220, %srw195, %srw228, %srw156, %srw164, %srw172, %srw179, %srw203, %srw211, %srw236, %srw244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              } else {
                                scf.yield %srw18, %srw20, %srw19, %srw22, %srw21, %srw24, %srw23, %srw28, %srw27, %srw26, %srw25, %srw32, %srw31, %srw30, %srw29, %srw36, %srw35, %srw34, %srw33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
                              }
                              %srw43 = "enzymexla.pointer2memref"(%srdstS) : (!llvm.ptr) -> memref<?xf32>
                              affine.store %srw42#0, %srw43[%arg7 + %arg6 * 128] : memref<?xf32>
                              affine.store %srw42#1, %srw43[%arg7 + %arg6 * 128 + 2365568] : memref<?xf32>
                              affine.store %srw42#2, %srw43[%arg7 + %arg6 * 128 + 4730752] : memref<?xf32>
                              affine.store %srw42#3, %srw43[%arg7 + %arg6 * 128 + 7096321] : memref<?xf32>
                              affine.store %srw42#4, %srw43[%arg7 + %arg6 * 128 + 9461759] : memref<?xf32>
                              affine.store %srw42#5, %srw43[%arg7 + %arg6 * 128 + 11842560] : memref<?xf32>
                              affine.store %srw42#6, %srw43[%arg7 + %arg6 * 128 + 14177280] : memref<?xf32>
                              affine.store %srw42#7, %srw43[%arg7 + %arg6 * 128 + 16558209] : memref<?xf32>
                              affine.store %srw42#8, %srw43[%arg7 + %arg6 * 128 + 18923647] : memref<?xf32>
                              affine.store %srw42#9, %srw43[%arg7 + %arg6 * 128 + 21288833] : memref<?xf32>
                              affine.store %srw42#10, %srw43[%arg7 + %arg6 * 128 + 23654271] : memref<?xf32>
                              affine.store %srw42#11, %srw43[%arg7 + %arg6 * 128 + 26035328] : memref<?xf32>
                              affine.store %srw42#12, %srw43[%arg7 + %arg6 * 128 + 28370048] : memref<?xf32>
                              affine.store %srw42#13, %srw43[%arg7 + %arg6 * 128 + 30765952] : memref<?xf32>
                              affine.store %srw42#14, %srw43[%arg7 + %arg6 * 128 + 33100672] : memref<?xf32>
                              affine.store %srw42#15, %srw43[%arg7 + %arg6 * 128 + 35496961] : memref<?xf32>
                              affine.store %srw42#16, %srw43[%arg7 + %arg6 * 128 + 37831681] : memref<?xf32>
                              affine.store %srw42#17, %srw43[%arg7 + %arg6 * 128 + 40227839] : memref<?xf32>
                              affine.store %srw42#18, %srw43[%arg7 + %arg6 * 128 + 42562559] : memref<?xf32>
                            }
                            "enzymexla.polygeist_yield"() : () -> ()
                }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
                async.yield
              }
            }
          }
        }
      }
      %sl_r8 = llvm.call @cudaStreamSynchronize(%sl_st) : (!llvm.ptr) -> i32
      %sl_r9 = llvm.call @cudaStreamDestroy(%sl_st) : (!llvm.ptr) -> i32
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
  llvm.func @cudaStreamCreateWithFlags(!llvm.ptr, i32) -> i32
  llvm.func @cudaStreamBeginCapture(!llvm.ptr, i32) -> i32
  llvm.func @cudaStreamEndCapture(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @cudaGraphInstantiate(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.func @cudaGraphLaunch(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @cudaStreamSynchronize(!llvm.ptr) -> i32
  llvm.func @cudaDeviceSynchronize() -> i32
  llvm.func @cudaGraphExecDestroy(!llvm.ptr) -> i32
  llvm.func @cudaGraphDestroy(!llvm.ptr) -> i32
  llvm.func @cudaStreamDestroy(!llvm.ptr) -> i32
}
