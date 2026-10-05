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
      %8:4 = scf.while (%arg3 = %c0_i32) : (i32) -> (i32, i32, i32, i32) {
        %12 = "enzymexla.gpu_wrapper"(%c120, %c150, %c1, %c120, %c1, %c1) ({
          affine.parallel (%arg4, %arg5) = (0, 0) to (18000, 120) {
            llvm.intr.experimental.noalias.scope.decl #alias_scope
            llvm.intr.experimental.noalias.scope.decl #alias_scope1
            %17 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
            %18 = affine.load %17[%arg5 + %arg4 * 128] : memref<?xf32>
            %19 = affine.load %17[%arg5 + %arg4 * 128 + 2365440] : memref<?xf32>
            %20 = affine.load %17[%arg5 + %arg4 * 128 + 4730880] : memref<?xf32>
            %21 = affine.load %17[%arg5 + %arg4 * 128 + 7096320] : memref<?xf32>
            %22 = affine.load %17[%arg5 + %arg4 * 128 + 9461760] : memref<?xf32>
            %23 = affine.load %17[%arg5 + %arg4 * 128 + 11827200] : memref<?xf32>
            %24 = affine.load %17[%arg5 + %arg4 * 128 + 14192640] : memref<?xf32>
            %25 = affine.load %17[%arg5 + %arg4 * 128 + 16558080] : memref<?xf32>
            %26 = affine.load %17[%arg5 + %arg4 * 128 + 18923520] : memref<?xf32>
            %27 = affine.load %17[%arg5 + %arg4 * 128 + 21288960] : memref<?xf32>
            %28 = affine.load %17[%arg5 + %arg4 * 128 + 23654400] : memref<?xf32>
            %29 = affine.load %17[%arg5 + %arg4 * 128 + 26019840] : memref<?xf32>
            %30 = affine.load %17[%arg5 + %arg4 * 128 + 28385280] : memref<?xf32>
            %31 = affine.load %17[%arg5 + %arg4 * 128 + 30750720] : memref<?xf32>
            %32 = affine.load %17[%arg5 + %arg4 * 128 + 33116160] : memref<?xf32>
            %33 = affine.load %17[%arg5 + %arg4 * 128 + 35481600] : memref<?xf32>
            %34 = affine.load %17[%arg5 + %arg4 * 128 + 37847040] : memref<?xf32>
            %35 = affine.load %17[%arg5 + %arg4 * 128 + 40212480] : memref<?xf32>
            %36 = affine.load %17[%arg5 + %arg4 * 128 + 42577920] : memref<?xf32>
            %37 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi8>
            %38 = affine.load %37[%arg5 * 4 + %arg4 * 512 + 179773440] : memref<?xi8>
            %39 = arith.extui %38 : i8 to i32
            %40 = arith.andi %39, %c1_i32 : i32
            %41 = arith.cmpi eq, %40, %c0_i32 : i32
            %42:19 = scf.if %41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
              %44 = arith.addf %19, %18 fastmath<fast> : f32
              %45 = arith.addf %44, %20 fastmath<fast> : f32
              %46 = arith.addf %45, %21 fastmath<fast> : f32
              %47 = arith.addf %46, %22 fastmath<fast> : f32
              %48 = arith.addf %47, %23 fastmath<fast> : f32
              %49 = arith.addf %48, %24 fastmath<fast> : f32
              %50 = arith.addf %49, %25 fastmath<fast> : f32
              %51 = arith.addf %50, %26 fastmath<fast> : f32
              %52 = arith.addf %51, %27 fastmath<fast> : f32
              %53 = arith.addf %52, %28 fastmath<fast> : f32
              %54 = arith.addf %53, %29 fastmath<fast> : f32
              %55 = arith.addf %54, %30 fastmath<fast> : f32
              %56 = arith.addf %55, %31 fastmath<fast> : f32
              %57 = arith.addf %56, %32 fastmath<fast> : f32
              %58 = arith.addf %57, %33 fastmath<fast> : f32
              %59 = arith.addf %58, %34 fastmath<fast> : f32
              %60 = arith.addf %59, %35 fastmath<fast> : f32
              %61 = arith.addf %60, %36 fastmath<fast> : f32
              %62 = arith.addf %21, %25 fastmath<fast> : f32
              %63 = arith.addf %22, %26 fastmath<fast> : f32
              %64 = arith.addf %62, %27 fastmath<fast> : f32
              %65 = arith.addf %63, %28 fastmath<fast> : f32
              %66 = arith.subf %64, %65 fastmath<fast> : f32
              %67 = arith.addf %66, %33 fastmath<fast> : f32
              %68 = arith.addf %67, %34 fastmath<fast> : f32
              %69 = arith.addf %35, %36 fastmath<fast> : f32
              %70 = arith.subf %68, %69 fastmath<fast> : f32
              %71 = arith.subf %19, %20 fastmath<fast> : f32
              %72 = arith.addf %71, %25 fastmath<fast> : f32
              %73 = arith.addf %72, %26 fastmath<fast> : f32
              %74 = arith.addf %27, %28 fastmath<fast> : f32
              %75 = arith.subf %73, %74 fastmath<fast> : f32
              %76 = arith.addf %75, %29 fastmath<fast> : f32
              %77 = arith.addf %76, %30 fastmath<fast> : f32
              %78 = arith.addf %31, %32 fastmath<fast> : f32
              %79 = arith.subf %77, %78 fastmath<fast> : f32
              %80 = arith.addf %23, %29 fastmath<fast> : f32
              %81 = arith.addf %24, %30 fastmath<fast> : f32
              %82 = arith.addf %80, %31 fastmath<fast> : f32
              %83 = arith.addf %81, %32 fastmath<fast> : f32
              %84 = arith.addf %82, %33 fastmath<fast> : f32
              %85 = arith.addf %83, %34 fastmath<fast> : f32
              %86 = arith.addf %84, %35 fastmath<fast> : f32
              %87 = arith.addf %85, %36 fastmath<fast> : f32
              %88 = arith.subf %86, %87 fastmath<fast> : f32
              %89 = arith.divf %70, %61 fastmath<fast> : f32
              %90 = arith.divf %79, %61 fastmath<fast> : f32
              %91 = arith.divf %88, %61 fastmath<fast> : f32
              %92 = arith.andi %39, %c2_i32 : i32
              %93 = arith.cmpi eq, %92, %c0_i32 : i32
              %94 = arith.select %93, %89, %cst_11 : f32
              %95 = arith.select %93, %90, %cst_10 : f32
              %96 = arith.select %93, %91, %cst_9 : f32
              %97 = arith.mulf %94, %94 fastmath<fast> : f32
              %98 = arith.mulf %95, %95 fastmath<fast> : f32
              %99 = arith.addf %97, %98 fastmath<fast> : f32
              %100 = arith.mulf %96, %96 fastmath<fast> : f32
              %101 = arith.addf %99, %100 fastmath<fast> : f32
              %102 = arith.mulf %101, %cst_8 fastmath<fast> : f32
              %103 = arith.addf %102, %cst_7 fastmath<fast> : f32
              %104 = arith.mulf %61, %cst_6 fastmath<fast> : f32
              %105 = arith.mulf %18, %cst_5 fastmath<fast> : f32
              %106 = arith.mulf %104, %103 fastmath<fast> : f32
              %107 = arith.subf %105, %106 fastmath<fast> : f32
              %108 = arith.mulf %61, %cst_4 fastmath<fast> : f32
              %109 = arith.mulf %19, %cst_3 fastmath<fast> : f32
              %110 = arith.mulf %95, %cst_2 fastmath<fast> : f32
              %111 = arith.addf %110, %cst_1 fastmath<fast> : f32
              %112 = arith.mulf %111, %95 fastmath<fast> : f32
              %113 = arith.subf %112, %103 fastmath<fast> : f32
              %114 = arith.mulf %113, %108 fastmath<fast> : f32
              %115 = arith.subf %114, %109 fastmath<fast> : f32
              %116 = arith.mulf %20, %cst_3 fastmath<fast> : f32
              %117 = arith.addf %110, %cst_0 fastmath<fast> : f32
              %118 = arith.mulf %117, %95 fastmath<fast> : f32
              %119 = arith.subf %118, %103 fastmath<fast> : f32
              %120 = arith.mulf %119, %108 fastmath<fast> : f32
              %121 = arith.subf %120, %116 fastmath<fast> : f32
              %122 = arith.mulf %23, %cst_3 fastmath<fast> : f32
              %123 = arith.mulf %96, %cst_2 fastmath<fast> : f32
              %124 = arith.addf %123, %cst_1 fastmath<fast> : f32
              %125 = arith.mulf %124, %96 fastmath<fast> : f32
              %126 = arith.subf %125, %103 fastmath<fast> : f32
              %127 = arith.mulf %126, %108 fastmath<fast> : f32
              %128 = arith.subf %127, %122 fastmath<fast> : f32
              %129 = arith.mulf %24, %cst_3 fastmath<fast> : f32
              %130 = arith.addf %123, %cst_0 fastmath<fast> : f32
              %131 = arith.mulf %130, %96 fastmath<fast> : f32
              %132 = arith.subf %131, %103 fastmath<fast> : f32
              %133 = arith.mulf %132, %108 fastmath<fast> : f32
              %134 = arith.subf %133, %129 fastmath<fast> : f32
              %135 = arith.mulf %21, %cst_3 fastmath<fast> : f32
              %136 = arith.mulf %94, %cst_2 fastmath<fast> : f32
              %137 = arith.addf %136, %cst_1 fastmath<fast> : f32
              %138 = arith.mulf %137, %94 fastmath<fast> : f32
              %139 = arith.subf %138, %103 fastmath<fast> : f32
              %140 = arith.mulf %139, %108 fastmath<fast> : f32
              %141 = arith.subf %140, %135 fastmath<fast> : f32
              %142 = arith.mulf %22, %cst_3 fastmath<fast> : f32
              %143 = arith.addf %136, %cst_0 fastmath<fast> : f32
              %144 = arith.mulf %143, %94 fastmath<fast> : f32
              %145 = arith.subf %144, %103 fastmath<fast> : f32
              %146 = arith.mulf %145, %108 fastmath<fast> : f32
              %147 = arith.subf %146, %142 fastmath<fast> : f32
              %148 = arith.mulf %61, %cst fastmath<fast> : f32
              %149 = arith.mulf %29, %cst_3 fastmath<fast> : f32
              %150 = arith.addf %95, %96 fastmath<fast> : f32
              %151 = arith.mulf %150, %cst_2 fastmath<fast> : f32
              %152 = arith.addf %151, %cst_1 fastmath<fast> : f32
              %153 = arith.mulf %152, %150 fastmath<fast> : f32
              %154 = arith.subf %153, %103 fastmath<fast> : f32
              %155 = arith.mulf %154, %148 fastmath<fast> : f32
              %156 = arith.subf %155, %149 fastmath<fast> : f32
              %157 = arith.mulf %30, %cst_3 fastmath<fast> : f32
              %158 = arith.subf %95, %96 fastmath<fast> : f32
              %159 = arith.mulf %158, %cst_2 fastmath<fast> : f32
              %160 = arith.addf %159, %cst_1 fastmath<fast> : f32
              %161 = arith.mulf %160, %158 fastmath<fast> : f32
              %162 = arith.subf %161, %103 fastmath<fast> : f32
              %163 = arith.mulf %162, %148 fastmath<fast> : f32
              %164 = arith.subf %163, %157 fastmath<fast> : f32
              %165 = arith.mulf %31, %cst_3 fastmath<fast> : f32
              %166 = arith.subf %96, %95 fastmath<fast> : f32
              %167 = arith.mulf %166, %cst_2 fastmath<fast> : f32
              %168 = arith.addf %167, %cst_1 fastmath<fast> : f32
              %169 = arith.mulf %168, %166 fastmath<fast> : f32
              %170 = arith.subf %169, %103 fastmath<fast> : f32
              %171 = arith.mulf %170, %148 fastmath<fast> : f32
              %172 = arith.subf %171, %165 fastmath<fast> : f32
              %173 = arith.mulf %32, %cst_3 fastmath<fast> : f32
              %174 = arith.negf %150 fastmath<fast> : f32
              %175 = arith.subf %cst_1, %151 fastmath<fast> : f32
              %176 = arith.mulf %175, %174 fastmath<fast> : f32
              %177 = arith.subf %176, %103 fastmath<fast> : f32
              %178 = arith.mulf %177, %148 fastmath<fast> : f32
              %179 = arith.subf %178, %173 fastmath<fast> : f32
              %180 = arith.mulf %25, %cst_3 fastmath<fast> : f32
              %181 = arith.addf %94, %95 fastmath<fast> : f32
              %182 = arith.mulf %181, %cst_2 fastmath<fast> : f32
              %183 = arith.addf %182, %cst_1 fastmath<fast> : f32
              %184 = arith.mulf %183, %181 fastmath<fast> : f32
              %185 = arith.subf %184, %103 fastmath<fast> : f32
              %186 = arith.mulf %185, %148 fastmath<fast> : f32
              %187 = arith.subf %186, %180 fastmath<fast> : f32
              %188 = arith.mulf %27, %cst_3 fastmath<fast> : f32
              %189 = arith.subf %94, %95 fastmath<fast> : f32
              %190 = arith.mulf %189, %cst_2 fastmath<fast> : f32
              %191 = arith.addf %190, %cst_1 fastmath<fast> : f32
              %192 = arith.mulf %191, %189 fastmath<fast> : f32
              %193 = arith.subf %192, %103 fastmath<fast> : f32
              %194 = arith.mulf %193, %148 fastmath<fast> : f32
              %195 = arith.subf %194, %188 fastmath<fast> : f32
              %196 = arith.mulf %33, %cst_3 fastmath<fast> : f32
              %197 = arith.addf %94, %96 fastmath<fast> : f32
              %198 = arith.mulf %197, %cst_2 fastmath<fast> : f32
              %199 = arith.addf %198, %cst_1 fastmath<fast> : f32
              %200 = arith.mulf %199, %197 fastmath<fast> : f32
              %201 = arith.subf %200, %103 fastmath<fast> : f32
              %202 = arith.mulf %201, %148 fastmath<fast> : f32
              %203 = arith.subf %202, %196 fastmath<fast> : f32
              %204 = arith.mulf %34, %cst_3 fastmath<fast> : f32
              %205 = arith.subf %94, %96 fastmath<fast> : f32
              %206 = arith.mulf %205, %cst_2 fastmath<fast> : f32
              %207 = arith.addf %206, %cst_1 fastmath<fast> : f32
              %208 = arith.mulf %207, %205 fastmath<fast> : f32
              %209 = arith.subf %208, %103 fastmath<fast> : f32
              %210 = arith.mulf %209, %148 fastmath<fast> : f32
              %211 = arith.subf %210, %204 fastmath<fast> : f32
              %212 = arith.mulf %26, %cst_3 fastmath<fast> : f32
              %213 = arith.negf %94 fastmath<fast> : f32
              %214 = arith.subf %95, %94 fastmath<fast> : f32
              %215 = arith.mulf %214, %cst_2 fastmath<fast> : f32
              %216 = arith.addf %215, %cst_1 fastmath<fast> : f32
              %217 = arith.mulf %216, %214 fastmath<fast> : f32
              %218 = arith.subf %217, %103 fastmath<fast> : f32
              %219 = arith.mulf %218, %148 fastmath<fast> : f32
              %220 = arith.subf %219, %212 fastmath<fast> : f32
              %221 = arith.mulf %28, %cst_3 fastmath<fast> : f32
              %222 = arith.subf %213, %95 fastmath<fast> : f32
              %223 = arith.mulf %222, %cst_2 fastmath<fast> : f32
              %224 = arith.addf %223, %cst_1 fastmath<fast> : f32
              %225 = arith.mulf %224, %222 fastmath<fast> : f32
              %226 = arith.subf %225, %103 fastmath<fast> : f32
              %227 = arith.mulf %226, %148 fastmath<fast> : f32
              %228 = arith.subf %227, %221 fastmath<fast> : f32
              %229 = arith.mulf %35, %cst_3 fastmath<fast> : f32
              %230 = arith.subf %96, %94 fastmath<fast> : f32
              %231 = arith.mulf %230, %cst_2 fastmath<fast> : f32
              %232 = arith.addf %231, %cst_1 fastmath<fast> : f32
              %233 = arith.mulf %232, %230 fastmath<fast> : f32
              %234 = arith.subf %233, %103 fastmath<fast> : f32
              %235 = arith.mulf %234, %148 fastmath<fast> : f32
              %236 = arith.subf %235, %229 fastmath<fast> : f32
              %237 = arith.mulf %36, %cst_3 fastmath<fast> : f32
              %238 = arith.subf %213, %96 fastmath<fast> : f32
              %239 = arith.mulf %238, %cst_2 fastmath<fast> : f32
              %240 = arith.addf %239, %cst_1 fastmath<fast> : f32
              %241 = arith.mulf %240, %238 fastmath<fast> : f32
              %242 = arith.subf %241, %103 fastmath<fast> : f32
              %243 = arith.mulf %242, %148 fastmath<fast> : f32
              %244 = arith.subf %243, %237 fastmath<fast> : f32
              scf.yield %107, %115, %121, %141, %147, %128, %134, %187, %220, %195, %228, %156, %164, %172, %179, %203, %211, %236, %244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
            } else {
              scf.yield %18, %20, %19, %22, %21, %24, %23, %28, %27, %26, %25, %32, %31, %30, %29, %36, %35, %34, %33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
            }
            %43 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xf32>
            affine.store %42#0, %43[%arg5 + %arg4 * 128] : memref<?xf32>
            affine.store %42#1, %43[%arg5 + %arg4 * 128 + 2365568] : memref<?xf32>
            affine.store %42#2, %43[%arg5 + %arg4 * 128 + 4730752] : memref<?xf32>
            affine.store %42#3, %43[%arg5 + %arg4 * 128 + 7096321] : memref<?xf32>
            affine.store %42#4, %43[%arg5 + %arg4 * 128 + 9461759] : memref<?xf32>
            affine.store %42#5, %43[%arg5 + %arg4 * 128 + 11842560] : memref<?xf32>
            affine.store %42#6, %43[%arg5 + %arg4 * 128 + 14177280] : memref<?xf32>
            affine.store %42#7, %43[%arg5 + %arg4 * 128 + 16558209] : memref<?xf32>
            affine.store %42#8, %43[%arg5 + %arg4 * 128 + 18923647] : memref<?xf32>
            affine.store %42#9, %43[%arg5 + %arg4 * 128 + 21288833] : memref<?xf32>
            affine.store %42#10, %43[%arg5 + %arg4 * 128 + 23654271] : memref<?xf32>
            affine.store %42#11, %43[%arg5 + %arg4 * 128 + 26035328] : memref<?xf32>
            affine.store %42#12, %43[%arg5 + %arg4 * 128 + 28370048] : memref<?xf32>
            affine.store %42#13, %43[%arg5 + %arg4 * 128 + 30765952] : memref<?xf32>
            affine.store %42#14, %43[%arg5 + %arg4 * 128 + 33100672] : memref<?xf32>
            affine.store %42#15, %43[%arg5 + %arg4 * 128 + 35496961] : memref<?xf32>
            affine.store %42#16, %43[%arg5 + %arg4 * 128 + 37831681] : memref<?xf32>
            affine.store %42#17, %43[%arg5 + %arg4 * 128 + 40227839] : memref<?xf32>
            affine.store %42#18, %43[%arg5 + %arg4 * 128 + 42562559] : memref<?xf32>
          }
          "enzymexla.polygeist_yield"() : () -> ()
        }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
        %13 = llvm.call @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
        %14 = arith.cmpi eq, %13, %c0_i32 : i32
        %15 = arith.addi %arg3, %c1_i32 overflow<nuw> : i32
        %16:3 = scf.if %14 -> (i32, i32, i1) {
          %17 = "enzymexla.gpu_wrapper"(%c120, %c150, %c1, %c120, %c1, %c1) ({
            affine.parallel (%arg4, %arg5) = (0, 0) to (18000, 120) {
              llvm.intr.experimental.noalias.scope.decl #alias_scope2
              llvm.intr.experimental.noalias.scope.decl #alias_scope3
              %25 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xf32>
              %26 = affine.load %25[%arg5 + %arg4 * 128] : memref<?xf32>
              %27 = affine.load %25[%arg5 + %arg4 * 128 + 2365440] : memref<?xf32>
              %28 = affine.load %25[%arg5 + %arg4 * 128 + 4730880] : memref<?xf32>
              %29 = affine.load %25[%arg5 + %arg4 * 128 + 7096320] : memref<?xf32>
              %30 = affine.load %25[%arg5 + %arg4 * 128 + 9461760] : memref<?xf32>
              %31 = affine.load %25[%arg5 + %arg4 * 128 + 11827200] : memref<?xf32>
              %32 = affine.load %25[%arg5 + %arg4 * 128 + 14192640] : memref<?xf32>
              %33 = affine.load %25[%arg5 + %arg4 * 128 + 16558080] : memref<?xf32>
              %34 = affine.load %25[%arg5 + %arg4 * 128 + 18923520] : memref<?xf32>
              %35 = affine.load %25[%arg5 + %arg4 * 128 + 21288960] : memref<?xf32>
              %36 = affine.load %25[%arg5 + %arg4 * 128 + 23654400] : memref<?xf32>
              %37 = affine.load %25[%arg5 + %arg4 * 128 + 26019840] : memref<?xf32>
              %38 = affine.load %25[%arg5 + %arg4 * 128 + 28385280] : memref<?xf32>
              %39 = affine.load %25[%arg5 + %arg4 * 128 + 30750720] : memref<?xf32>
              %40 = affine.load %25[%arg5 + %arg4 * 128 + 33116160] : memref<?xf32>
              %41 = affine.load %25[%arg5 + %arg4 * 128 + 35481600] : memref<?xf32>
              %42 = affine.load %25[%arg5 + %arg4 * 128 + 37847040] : memref<?xf32>
              %43 = affine.load %25[%arg5 + %arg4 * 128 + 40212480] : memref<?xf32>
              %44 = affine.load %25[%arg5 + %arg4 * 128 + 42577920] : memref<?xf32>
              %45 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi8>
              %46 = affine.load %45[%arg5 * 4 + %arg4 * 512 + 179773440] : memref<?xi8>
              %47 = arith.extui %46 : i8 to i32
              %48 = arith.andi %47, %c1_i32 : i32
              %49 = arith.cmpi eq, %48, %c0_i32 : i32
              %50:19 = scf.if %49 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                %52 = arith.addf %27, %26 fastmath<fast> : f32
                %53 = arith.addf %52, %28 fastmath<fast> : f32
                %54 = arith.addf %53, %29 fastmath<fast> : f32
                %55 = arith.addf %54, %30 fastmath<fast> : f32
                %56 = arith.addf %55, %31 fastmath<fast> : f32
                %57 = arith.addf %56, %32 fastmath<fast> : f32
                %58 = arith.addf %57, %33 fastmath<fast> : f32
                %59 = arith.addf %58, %34 fastmath<fast> : f32
                %60 = arith.addf %59, %35 fastmath<fast> : f32
                %61 = arith.addf %60, %36 fastmath<fast> : f32
                %62 = arith.addf %61, %37 fastmath<fast> : f32
                %63 = arith.addf %62, %38 fastmath<fast> : f32
                %64 = arith.addf %63, %39 fastmath<fast> : f32
                %65 = arith.addf %64, %40 fastmath<fast> : f32
                %66 = arith.addf %65, %41 fastmath<fast> : f32
                %67 = arith.addf %66, %42 fastmath<fast> : f32
                %68 = arith.addf %67, %43 fastmath<fast> : f32
                %69 = arith.addf %68, %44 fastmath<fast> : f32
                %70 = arith.addf %29, %33 fastmath<fast> : f32
                %71 = arith.addf %30, %34 fastmath<fast> : f32
                %72 = arith.addf %70, %35 fastmath<fast> : f32
                %73 = arith.addf %71, %36 fastmath<fast> : f32
                %74 = arith.subf %72, %73 fastmath<fast> : f32
                %75 = arith.addf %74, %41 fastmath<fast> : f32
                %76 = arith.addf %75, %42 fastmath<fast> : f32
                %77 = arith.addf %43, %44 fastmath<fast> : f32
                %78 = arith.subf %76, %77 fastmath<fast> : f32
                %79 = arith.subf %27, %28 fastmath<fast> : f32
                %80 = arith.addf %79, %33 fastmath<fast> : f32
                %81 = arith.addf %80, %34 fastmath<fast> : f32
                %82 = arith.addf %35, %36 fastmath<fast> : f32
                %83 = arith.subf %81, %82 fastmath<fast> : f32
                %84 = arith.addf %83, %37 fastmath<fast> : f32
                %85 = arith.addf %84, %38 fastmath<fast> : f32
                %86 = arith.addf %39, %40 fastmath<fast> : f32
                %87 = arith.subf %85, %86 fastmath<fast> : f32
                %88 = arith.addf %31, %37 fastmath<fast> : f32
                %89 = arith.addf %32, %38 fastmath<fast> : f32
                %90 = arith.addf %88, %39 fastmath<fast> : f32
                %91 = arith.addf %89, %40 fastmath<fast> : f32
                %92 = arith.addf %90, %41 fastmath<fast> : f32
                %93 = arith.addf %91, %42 fastmath<fast> : f32
                %94 = arith.addf %92, %43 fastmath<fast> : f32
                %95 = arith.addf %93, %44 fastmath<fast> : f32
                %96 = arith.subf %94, %95 fastmath<fast> : f32
                %97 = arith.divf %78, %69 fastmath<fast> : f32
                %98 = arith.divf %87, %69 fastmath<fast> : f32
                %99 = arith.divf %96, %69 fastmath<fast> : f32
                %100 = arith.andi %47, %c2_i32 : i32
                %101 = arith.cmpi eq, %100, %c0_i32 : i32
                %102 = arith.select %101, %97, %cst_11 : f32
                %103 = arith.select %101, %98, %cst_10 : f32
                %104 = arith.select %101, %99, %cst_9 : f32
                %105 = arith.mulf %102, %102 fastmath<fast> : f32
                %106 = arith.mulf %103, %103 fastmath<fast> : f32
                %107 = arith.addf %105, %106 fastmath<fast> : f32
                %108 = arith.mulf %104, %104 fastmath<fast> : f32
                %109 = arith.addf %107, %108 fastmath<fast> : f32
                %110 = arith.mulf %109, %cst_8 fastmath<fast> : f32
                %111 = arith.addf %110, %cst_7 fastmath<fast> : f32
                %112 = arith.mulf %69, %cst_6 fastmath<fast> : f32
                %113 = arith.mulf %26, %cst_5 fastmath<fast> : f32
                %114 = arith.mulf %112, %111 fastmath<fast> : f32
                %115 = arith.subf %113, %114 fastmath<fast> : f32
                %116 = arith.mulf %69, %cst_4 fastmath<fast> : f32
                %117 = arith.mulf %27, %cst_3 fastmath<fast> : f32
                %118 = arith.mulf %103, %cst_2 fastmath<fast> : f32
                %119 = arith.addf %118, %cst_1 fastmath<fast> : f32
                %120 = arith.mulf %119, %103 fastmath<fast> : f32
                %121 = arith.subf %120, %111 fastmath<fast> : f32
                %122 = arith.mulf %121, %116 fastmath<fast> : f32
                %123 = arith.subf %122, %117 fastmath<fast> : f32
                %124 = arith.mulf %28, %cst_3 fastmath<fast> : f32
                %125 = arith.addf %118, %cst_0 fastmath<fast> : f32
                %126 = arith.mulf %125, %103 fastmath<fast> : f32
                %127 = arith.subf %126, %111 fastmath<fast> : f32
                %128 = arith.mulf %127, %116 fastmath<fast> : f32
                %129 = arith.subf %128, %124 fastmath<fast> : f32
                %130 = arith.mulf %31, %cst_3 fastmath<fast> : f32
                %131 = arith.mulf %104, %cst_2 fastmath<fast> : f32
                %132 = arith.addf %131, %cst_1 fastmath<fast> : f32
                %133 = arith.mulf %132, %104 fastmath<fast> : f32
                %134 = arith.subf %133, %111 fastmath<fast> : f32
                %135 = arith.mulf %134, %116 fastmath<fast> : f32
                %136 = arith.subf %135, %130 fastmath<fast> : f32
                %137 = arith.mulf %32, %cst_3 fastmath<fast> : f32
                %138 = arith.addf %131, %cst_0 fastmath<fast> : f32
                %139 = arith.mulf %138, %104 fastmath<fast> : f32
                %140 = arith.subf %139, %111 fastmath<fast> : f32
                %141 = arith.mulf %140, %116 fastmath<fast> : f32
                %142 = arith.subf %141, %137 fastmath<fast> : f32
                %143 = arith.mulf %29, %cst_3 fastmath<fast> : f32
                %144 = arith.mulf %102, %cst_2 fastmath<fast> : f32
                %145 = arith.addf %144, %cst_1 fastmath<fast> : f32
                %146 = arith.mulf %145, %102 fastmath<fast> : f32
                %147 = arith.subf %146, %111 fastmath<fast> : f32
                %148 = arith.mulf %147, %116 fastmath<fast> : f32
                %149 = arith.subf %148, %143 fastmath<fast> : f32
                %150 = arith.mulf %30, %cst_3 fastmath<fast> : f32
                %151 = arith.addf %144, %cst_0 fastmath<fast> : f32
                %152 = arith.mulf %151, %102 fastmath<fast> : f32
                %153 = arith.subf %152, %111 fastmath<fast> : f32
                %154 = arith.mulf %153, %116 fastmath<fast> : f32
                %155 = arith.subf %154, %150 fastmath<fast> : f32
                %156 = arith.mulf %69, %cst fastmath<fast> : f32
                %157 = arith.mulf %37, %cst_3 fastmath<fast> : f32
                %158 = arith.addf %103, %104 fastmath<fast> : f32
                %159 = arith.mulf %158, %cst_2 fastmath<fast> : f32
                %160 = arith.addf %159, %cst_1 fastmath<fast> : f32
                %161 = arith.mulf %160, %158 fastmath<fast> : f32
                %162 = arith.subf %161, %111 fastmath<fast> : f32
                %163 = arith.mulf %162, %156 fastmath<fast> : f32
                %164 = arith.subf %163, %157 fastmath<fast> : f32
                %165 = arith.mulf %38, %cst_3 fastmath<fast> : f32
                %166 = arith.subf %103, %104 fastmath<fast> : f32
                %167 = arith.mulf %166, %cst_2 fastmath<fast> : f32
                %168 = arith.addf %167, %cst_1 fastmath<fast> : f32
                %169 = arith.mulf %168, %166 fastmath<fast> : f32
                %170 = arith.subf %169, %111 fastmath<fast> : f32
                %171 = arith.mulf %170, %156 fastmath<fast> : f32
                %172 = arith.subf %171, %165 fastmath<fast> : f32
                %173 = arith.mulf %39, %cst_3 fastmath<fast> : f32
                %174 = arith.subf %104, %103 fastmath<fast> : f32
                %175 = arith.mulf %174, %cst_2 fastmath<fast> : f32
                %176 = arith.addf %175, %cst_1 fastmath<fast> : f32
                %177 = arith.mulf %176, %174 fastmath<fast> : f32
                %178 = arith.subf %177, %111 fastmath<fast> : f32
                %179 = arith.mulf %178, %156 fastmath<fast> : f32
                %180 = arith.subf %179, %173 fastmath<fast> : f32
                %181 = arith.mulf %40, %cst_3 fastmath<fast> : f32
                %182 = arith.negf %158 fastmath<fast> : f32
                %183 = arith.subf %cst_1, %159 fastmath<fast> : f32
                %184 = arith.mulf %183, %182 fastmath<fast> : f32
                %185 = arith.subf %184, %111 fastmath<fast> : f32
                %186 = arith.mulf %185, %156 fastmath<fast> : f32
                %187 = arith.subf %186, %181 fastmath<fast> : f32
                %188 = arith.mulf %33, %cst_3 fastmath<fast> : f32
                %189 = arith.addf %102, %103 fastmath<fast> : f32
                %190 = arith.mulf %189, %cst_2 fastmath<fast> : f32
                %191 = arith.addf %190, %cst_1 fastmath<fast> : f32
                %192 = arith.mulf %191, %189 fastmath<fast> : f32
                %193 = arith.subf %192, %111 fastmath<fast> : f32
                %194 = arith.mulf %193, %156 fastmath<fast> : f32
                %195 = arith.subf %194, %188 fastmath<fast> : f32
                %196 = arith.mulf %35, %cst_3 fastmath<fast> : f32
                %197 = arith.subf %102, %103 fastmath<fast> : f32
                %198 = arith.mulf %197, %cst_2 fastmath<fast> : f32
                %199 = arith.addf %198, %cst_1 fastmath<fast> : f32
                %200 = arith.mulf %199, %197 fastmath<fast> : f32
                %201 = arith.subf %200, %111 fastmath<fast> : f32
                %202 = arith.mulf %201, %156 fastmath<fast> : f32
                %203 = arith.subf %202, %196 fastmath<fast> : f32
                %204 = arith.mulf %41, %cst_3 fastmath<fast> : f32
                %205 = arith.addf %102, %104 fastmath<fast> : f32
                %206 = arith.mulf %205, %cst_2 fastmath<fast> : f32
                %207 = arith.addf %206, %cst_1 fastmath<fast> : f32
                %208 = arith.mulf %207, %205 fastmath<fast> : f32
                %209 = arith.subf %208, %111 fastmath<fast> : f32
                %210 = arith.mulf %209, %156 fastmath<fast> : f32
                %211 = arith.subf %210, %204 fastmath<fast> : f32
                %212 = arith.mulf %42, %cst_3 fastmath<fast> : f32
                %213 = arith.subf %102, %104 fastmath<fast> : f32
                %214 = arith.mulf %213, %cst_2 fastmath<fast> : f32
                %215 = arith.addf %214, %cst_1 fastmath<fast> : f32
                %216 = arith.mulf %215, %213 fastmath<fast> : f32
                %217 = arith.subf %216, %111 fastmath<fast> : f32
                %218 = arith.mulf %217, %156 fastmath<fast> : f32
                %219 = arith.subf %218, %212 fastmath<fast> : f32
                %220 = arith.mulf %34, %cst_3 fastmath<fast> : f32
                %221 = arith.negf %102 fastmath<fast> : f32
                %222 = arith.subf %103, %102 fastmath<fast> : f32
                %223 = arith.mulf %222, %cst_2 fastmath<fast> : f32
                %224 = arith.addf %223, %cst_1 fastmath<fast> : f32
                %225 = arith.mulf %224, %222 fastmath<fast> : f32
                %226 = arith.subf %225, %111 fastmath<fast> : f32
                %227 = arith.mulf %226, %156 fastmath<fast> : f32
                %228 = arith.subf %227, %220 fastmath<fast> : f32
                %229 = arith.mulf %36, %cst_3 fastmath<fast> : f32
                %230 = arith.subf %221, %103 fastmath<fast> : f32
                %231 = arith.mulf %230, %cst_2 fastmath<fast> : f32
                %232 = arith.addf %231, %cst_1 fastmath<fast> : f32
                %233 = arith.mulf %232, %230 fastmath<fast> : f32
                %234 = arith.subf %233, %111 fastmath<fast> : f32
                %235 = arith.mulf %234, %156 fastmath<fast> : f32
                %236 = arith.subf %235, %229 fastmath<fast> : f32
                %237 = arith.mulf %43, %cst_3 fastmath<fast> : f32
                %238 = arith.subf %104, %102 fastmath<fast> : f32
                %239 = arith.mulf %238, %cst_2 fastmath<fast> : f32
                %240 = arith.addf %239, %cst_1 fastmath<fast> : f32
                %241 = arith.mulf %240, %238 fastmath<fast> : f32
                %242 = arith.subf %241, %111 fastmath<fast> : f32
                %243 = arith.mulf %242, %156 fastmath<fast> : f32
                %244 = arith.subf %243, %237 fastmath<fast> : f32
                %245 = arith.mulf %44, %cst_3 fastmath<fast> : f32
                %246 = arith.subf %221, %104 fastmath<fast> : f32
                %247 = arith.mulf %246, %cst_2 fastmath<fast> : f32
                %248 = arith.addf %247, %cst_1 fastmath<fast> : f32
                %249 = arith.mulf %248, %246 fastmath<fast> : f32
                %250 = arith.subf %249, %111 fastmath<fast> : f32
                %251 = arith.mulf %250, %156 fastmath<fast> : f32
                %252 = arith.subf %251, %245 fastmath<fast> : f32
                scf.yield %115, %123, %129, %149, %155, %136, %142, %195, %228, %203, %236, %164, %172, %180, %187, %211, %219, %244, %252 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
              } else {
                scf.yield %26, %28, %27, %30, %29, %32, %31, %36, %35, %34, %33, %40, %39, %38, %37, %44, %43, %42, %41 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
              }
              %51 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
              affine.store %50#0, %51[%arg5 + %arg4 * 128] : memref<?xf32>
              affine.store %50#1, %51[%arg5 + %arg4 * 128 + 2365568] : memref<?xf32>
              affine.store %50#2, %51[%arg5 + %arg4 * 128 + 4730752] : memref<?xf32>
              affine.store %50#3, %51[%arg5 + %arg4 * 128 + 7096321] : memref<?xf32>
              affine.store %50#4, %51[%arg5 + %arg4 * 128 + 9461759] : memref<?xf32>
              affine.store %50#5, %51[%arg5 + %arg4 * 128 + 11842560] : memref<?xf32>
              affine.store %50#6, %51[%arg5 + %arg4 * 128 + 14177280] : memref<?xf32>
              affine.store %50#7, %51[%arg5 + %arg4 * 128 + 16558209] : memref<?xf32>
              affine.store %50#8, %51[%arg5 + %arg4 * 128 + 18923647] : memref<?xf32>
              affine.store %50#9, %51[%arg5 + %arg4 * 128 + 21288833] : memref<?xf32>
              affine.store %50#10, %51[%arg5 + %arg4 * 128 + 23654271] : memref<?xf32>
              affine.store %50#11, %51[%arg5 + %arg4 * 128 + 26035328] : memref<?xf32>
              affine.store %50#12, %51[%arg5 + %arg4 * 128 + 28370048] : memref<?xf32>
              affine.store %50#13, %51[%arg5 + %arg4 * 128 + 30765952] : memref<?xf32>
              affine.store %50#14, %51[%arg5 + %arg4 * 128 + 33100672] : memref<?xf32>
              affine.store %50#15, %51[%arg5 + %arg4 * 128 + 35496961] : memref<?xf32>
              affine.store %50#16, %51[%arg5 + %arg4 * 128 + 37831681] : memref<?xf32>
              affine.store %50#17, %51[%arg5 + %arg4 * 128 + 40227839] : memref<?xf32>
              affine.store %50#18, %51[%arg5 + %arg4 * 128 + 42562559] : memref<?xf32>
            }
            "enzymexla.polygeist_yield"() : () -> ()
          }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
          %18 = llvm.call @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
          %19 = arith.cmpi eq, %18, %c0_i32 : i32
          %20 = arith.cmpi eq, %15, %7 : i32
          %21 = arith.extui %20 : i1 to i32
          %22 = arith.select %19, %21, %c2_i32 : i32
          %23 = arith.cmpi ne, %15, %7 : i32
          %24 = arith.andi %19, %23 : i1
          scf.yield %22, %18, %24 : i32, i32, i1
        } else {
          scf.yield %c3_i32, %1, %false : i32, i32, i1
        }
        scf.condition(%16#2) %15, %16#1, %13, %16#0 : i32, i32, i32, i32
      } do {
      ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32):
        scf.yield %arg3 : i32
      }
      %9 = arith.index_castui %8#3 : i32 to index
      %10 = arith.cmpi ne, %9, %c1 : index
      %11 = arith.extui %10 : i1 to i32
      scf.if %10 {
        %12 = arith.cmpi eq, %9, %c2 : index
        scf.if %12 {
          %13 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
          %14 = affine.load %13[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
          %15 = llvm.call @cudaGetErrorString(%8#1) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
          %16 = llvm.call @fprintf(%14, %0, %c51_i32, %15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
          llvm.call @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
        } else {
          %13 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
          %14 = affine.load %13[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
          %15 = llvm.call @cudaGetErrorString(%8#2) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
          %16 = llvm.call @fprintf(%14, %0, %c51_i32, %15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
          llvm.call @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
        }
      }
      scf.yield %11 : i32
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
      %8:4 = scf.while (%arg5 = %c0_i32) : (i32) -> (i32, i32, i32, i32) {
        %12 = "enzymexla.gpu_wrapper"(%c120, %c150, %c1, %c120, %c1, %c1) ({
          affine.parallel (%arg6, %arg7) = (0, 0) to (18000, 120) {
            llvm.intr.experimental.noalias.scope.decl #alias_scope4
            llvm.intr.experimental.noalias.scope.decl #alias_scope5
            %17 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
            %18 = affine.load %17[%arg7 + %arg6 * 128] : memref<?xf32>
            %19 = affine.load %17[%arg7 + %arg6 * 128 + 2365440] : memref<?xf32>
            %20 = affine.load %17[%arg7 + %arg6 * 128 + 4730880] : memref<?xf32>
            %21 = affine.load %17[%arg7 + %arg6 * 128 + 7096320] : memref<?xf32>
            %22 = affine.load %17[%arg7 + %arg6 * 128 + 9461760] : memref<?xf32>
            %23 = affine.load %17[%arg7 + %arg6 * 128 + 11827200] : memref<?xf32>
            %24 = affine.load %17[%arg7 + %arg6 * 128 + 14192640] : memref<?xf32>
            %25 = affine.load %17[%arg7 + %arg6 * 128 + 16558080] : memref<?xf32>
            %26 = affine.load %17[%arg7 + %arg6 * 128 + 18923520] : memref<?xf32>
            %27 = affine.load %17[%arg7 + %arg6 * 128 + 21288960] : memref<?xf32>
            %28 = affine.load %17[%arg7 + %arg6 * 128 + 23654400] : memref<?xf32>
            %29 = affine.load %17[%arg7 + %arg6 * 128 + 26019840] : memref<?xf32>
            %30 = affine.load %17[%arg7 + %arg6 * 128 + 28385280] : memref<?xf32>
            %31 = affine.load %17[%arg7 + %arg6 * 128 + 30750720] : memref<?xf32>
            %32 = affine.load %17[%arg7 + %arg6 * 128 + 33116160] : memref<?xf32>
            %33 = affine.load %17[%arg7 + %arg6 * 128 + 35481600] : memref<?xf32>
            %34 = affine.load %17[%arg7 + %arg6 * 128 + 37847040] : memref<?xf32>
            %35 = affine.load %17[%arg7 + %arg6 * 128 + 40212480] : memref<?xf32>
            %36 = affine.load %17[%arg7 + %arg6 * 128 + 42577920] : memref<?xf32>
            %37 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi8>
            %38 = affine.load %37[%arg7 * 4 + %arg6 * 512 + 179773440] : memref<?xi8>
            %39 = arith.extui %38 : i8 to i32
            %40 = arith.andi %39, %c1_i32 : i32
            %41 = arith.cmpi eq, %40, %c0_i32 : i32
            %42:19 = scf.if %41 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
              %44 = arith.addf %19, %18 fastmath<fast> : f32
              %45 = arith.addf %44, %20 fastmath<fast> : f32
              %46 = arith.addf %45, %21 fastmath<fast> : f32
              %47 = arith.addf %46, %22 fastmath<fast> : f32
              %48 = arith.addf %47, %23 fastmath<fast> : f32
              %49 = arith.addf %48, %24 fastmath<fast> : f32
              %50 = arith.addf %49, %25 fastmath<fast> : f32
              %51 = arith.addf %50, %26 fastmath<fast> : f32
              %52 = arith.addf %51, %27 fastmath<fast> : f32
              %53 = arith.addf %52, %28 fastmath<fast> : f32
              %54 = arith.addf %53, %29 fastmath<fast> : f32
              %55 = arith.addf %54, %30 fastmath<fast> : f32
              %56 = arith.addf %55, %31 fastmath<fast> : f32
              %57 = arith.addf %56, %32 fastmath<fast> : f32
              %58 = arith.addf %57, %33 fastmath<fast> : f32
              %59 = arith.addf %58, %34 fastmath<fast> : f32
              %60 = arith.addf %59, %35 fastmath<fast> : f32
              %61 = arith.addf %60, %36 fastmath<fast> : f32
              %62 = arith.addf %21, %25 fastmath<fast> : f32
              %63 = arith.addf %22, %26 fastmath<fast> : f32
              %64 = arith.addf %62, %27 fastmath<fast> : f32
              %65 = arith.addf %63, %28 fastmath<fast> : f32
              %66 = arith.subf %64, %65 fastmath<fast> : f32
              %67 = arith.addf %66, %33 fastmath<fast> : f32
              %68 = arith.addf %67, %34 fastmath<fast> : f32
              %69 = arith.addf %35, %36 fastmath<fast> : f32
              %70 = arith.subf %68, %69 fastmath<fast> : f32
              %71 = arith.subf %19, %20 fastmath<fast> : f32
              %72 = arith.addf %71, %25 fastmath<fast> : f32
              %73 = arith.addf %72, %26 fastmath<fast> : f32
              %74 = arith.addf %27, %28 fastmath<fast> : f32
              %75 = arith.subf %73, %74 fastmath<fast> : f32
              %76 = arith.addf %75, %29 fastmath<fast> : f32
              %77 = arith.addf %76, %30 fastmath<fast> : f32
              %78 = arith.addf %31, %32 fastmath<fast> : f32
              %79 = arith.subf %77, %78 fastmath<fast> : f32
              %80 = arith.addf %23, %29 fastmath<fast> : f32
              %81 = arith.addf %24, %30 fastmath<fast> : f32
              %82 = arith.addf %80, %31 fastmath<fast> : f32
              %83 = arith.addf %81, %32 fastmath<fast> : f32
              %84 = arith.addf %82, %33 fastmath<fast> : f32
              %85 = arith.addf %83, %34 fastmath<fast> : f32
              %86 = arith.addf %84, %35 fastmath<fast> : f32
              %87 = arith.addf %85, %36 fastmath<fast> : f32
              %88 = arith.subf %86, %87 fastmath<fast> : f32
              %89 = arith.divf %70, %61 fastmath<fast> : f32
              %90 = arith.divf %79, %61 fastmath<fast> : f32
              %91 = arith.divf %88, %61 fastmath<fast> : f32
              %92 = arith.andi %39, %c2_i32 : i32
              %93 = arith.cmpi eq, %92, %c0_i32 : i32
              %94 = arith.select %93, %89, %cst_11 : f32
              %95 = arith.select %93, %90, %cst_10 : f32
              %96 = arith.select %93, %91, %cst_9 : f32
              %97 = arith.mulf %94, %94 fastmath<fast> : f32
              %98 = arith.mulf %95, %95 fastmath<fast> : f32
              %99 = arith.addf %97, %98 fastmath<fast> : f32
              %100 = arith.mulf %96, %96 fastmath<fast> : f32
              %101 = arith.addf %99, %100 fastmath<fast> : f32
              %102 = arith.mulf %101, %cst_8 fastmath<fast> : f32
              %103 = arith.addf %102, %cst_7 fastmath<fast> : f32
              %104 = arith.mulf %61, %cst_6 fastmath<fast> : f32
              %105 = arith.mulf %18, %cst_5 fastmath<fast> : f32
              %106 = arith.mulf %104, %103 fastmath<fast> : f32
              %107 = arith.subf %105, %106 fastmath<fast> : f32
              %108 = arith.mulf %61, %cst_4 fastmath<fast> : f32
              %109 = arith.mulf %19, %cst_3 fastmath<fast> : f32
              %110 = arith.mulf %95, %cst_2 fastmath<fast> : f32
              %111 = arith.addf %110, %cst_1 fastmath<fast> : f32
              %112 = arith.mulf %111, %95 fastmath<fast> : f32
              %113 = arith.subf %112, %103 fastmath<fast> : f32
              %114 = arith.mulf %113, %108 fastmath<fast> : f32
              %115 = arith.subf %114, %109 fastmath<fast> : f32
              %116 = arith.mulf %20, %cst_3 fastmath<fast> : f32
              %117 = arith.addf %110, %cst_0 fastmath<fast> : f32
              %118 = arith.mulf %117, %95 fastmath<fast> : f32
              %119 = arith.subf %118, %103 fastmath<fast> : f32
              %120 = arith.mulf %119, %108 fastmath<fast> : f32
              %121 = arith.subf %120, %116 fastmath<fast> : f32
              %122 = arith.mulf %23, %cst_3 fastmath<fast> : f32
              %123 = arith.mulf %96, %cst_2 fastmath<fast> : f32
              %124 = arith.addf %123, %cst_1 fastmath<fast> : f32
              %125 = arith.mulf %124, %96 fastmath<fast> : f32
              %126 = arith.subf %125, %103 fastmath<fast> : f32
              %127 = arith.mulf %126, %108 fastmath<fast> : f32
              %128 = arith.subf %127, %122 fastmath<fast> : f32
              %129 = arith.mulf %24, %cst_3 fastmath<fast> : f32
              %130 = arith.addf %123, %cst_0 fastmath<fast> : f32
              %131 = arith.mulf %130, %96 fastmath<fast> : f32
              %132 = arith.subf %131, %103 fastmath<fast> : f32
              %133 = arith.mulf %132, %108 fastmath<fast> : f32
              %134 = arith.subf %133, %129 fastmath<fast> : f32
              %135 = arith.mulf %21, %cst_3 fastmath<fast> : f32
              %136 = arith.mulf %94, %cst_2 fastmath<fast> : f32
              %137 = arith.addf %136, %cst_1 fastmath<fast> : f32
              %138 = arith.mulf %137, %94 fastmath<fast> : f32
              %139 = arith.subf %138, %103 fastmath<fast> : f32
              %140 = arith.mulf %139, %108 fastmath<fast> : f32
              %141 = arith.subf %140, %135 fastmath<fast> : f32
              %142 = arith.mulf %22, %cst_3 fastmath<fast> : f32
              %143 = arith.addf %136, %cst_0 fastmath<fast> : f32
              %144 = arith.mulf %143, %94 fastmath<fast> : f32
              %145 = arith.subf %144, %103 fastmath<fast> : f32
              %146 = arith.mulf %145, %108 fastmath<fast> : f32
              %147 = arith.subf %146, %142 fastmath<fast> : f32
              %148 = arith.mulf %61, %cst fastmath<fast> : f32
              %149 = arith.mulf %29, %cst_3 fastmath<fast> : f32
              %150 = arith.addf %95, %96 fastmath<fast> : f32
              %151 = arith.mulf %150, %cst_2 fastmath<fast> : f32
              %152 = arith.addf %151, %cst_1 fastmath<fast> : f32
              %153 = arith.mulf %152, %150 fastmath<fast> : f32
              %154 = arith.subf %153, %103 fastmath<fast> : f32
              %155 = arith.mulf %154, %148 fastmath<fast> : f32
              %156 = arith.subf %155, %149 fastmath<fast> : f32
              %157 = arith.mulf %30, %cst_3 fastmath<fast> : f32
              %158 = arith.subf %95, %96 fastmath<fast> : f32
              %159 = arith.mulf %158, %cst_2 fastmath<fast> : f32
              %160 = arith.addf %159, %cst_1 fastmath<fast> : f32
              %161 = arith.mulf %160, %158 fastmath<fast> : f32
              %162 = arith.subf %161, %103 fastmath<fast> : f32
              %163 = arith.mulf %162, %148 fastmath<fast> : f32
              %164 = arith.subf %163, %157 fastmath<fast> : f32
              %165 = arith.mulf %31, %cst_3 fastmath<fast> : f32
              %166 = arith.subf %96, %95 fastmath<fast> : f32
              %167 = arith.mulf %166, %cst_2 fastmath<fast> : f32
              %168 = arith.addf %167, %cst_1 fastmath<fast> : f32
              %169 = arith.mulf %168, %166 fastmath<fast> : f32
              %170 = arith.subf %169, %103 fastmath<fast> : f32
              %171 = arith.mulf %170, %148 fastmath<fast> : f32
              %172 = arith.subf %171, %165 fastmath<fast> : f32
              %173 = arith.mulf %32, %cst_3 fastmath<fast> : f32
              %174 = arith.negf %150 fastmath<fast> : f32
              %175 = arith.subf %cst_1, %151 fastmath<fast> : f32
              %176 = arith.mulf %175, %174 fastmath<fast> : f32
              %177 = arith.subf %176, %103 fastmath<fast> : f32
              %178 = arith.mulf %177, %148 fastmath<fast> : f32
              %179 = arith.subf %178, %173 fastmath<fast> : f32
              %180 = arith.mulf %25, %cst_3 fastmath<fast> : f32
              %181 = arith.addf %94, %95 fastmath<fast> : f32
              %182 = arith.mulf %181, %cst_2 fastmath<fast> : f32
              %183 = arith.addf %182, %cst_1 fastmath<fast> : f32
              %184 = arith.mulf %183, %181 fastmath<fast> : f32
              %185 = arith.subf %184, %103 fastmath<fast> : f32
              %186 = arith.mulf %185, %148 fastmath<fast> : f32
              %187 = arith.subf %186, %180 fastmath<fast> : f32
              %188 = arith.mulf %27, %cst_3 fastmath<fast> : f32
              %189 = arith.subf %94, %95 fastmath<fast> : f32
              %190 = arith.mulf %189, %cst_2 fastmath<fast> : f32
              %191 = arith.addf %190, %cst_1 fastmath<fast> : f32
              %192 = arith.mulf %191, %189 fastmath<fast> : f32
              %193 = arith.subf %192, %103 fastmath<fast> : f32
              %194 = arith.mulf %193, %148 fastmath<fast> : f32
              %195 = arith.subf %194, %188 fastmath<fast> : f32
              %196 = arith.mulf %33, %cst_3 fastmath<fast> : f32
              %197 = arith.addf %94, %96 fastmath<fast> : f32
              %198 = arith.mulf %197, %cst_2 fastmath<fast> : f32
              %199 = arith.addf %198, %cst_1 fastmath<fast> : f32
              %200 = arith.mulf %199, %197 fastmath<fast> : f32
              %201 = arith.subf %200, %103 fastmath<fast> : f32
              %202 = arith.mulf %201, %148 fastmath<fast> : f32
              %203 = arith.subf %202, %196 fastmath<fast> : f32
              %204 = arith.mulf %34, %cst_3 fastmath<fast> : f32
              %205 = arith.subf %94, %96 fastmath<fast> : f32
              %206 = arith.mulf %205, %cst_2 fastmath<fast> : f32
              %207 = arith.addf %206, %cst_1 fastmath<fast> : f32
              %208 = arith.mulf %207, %205 fastmath<fast> : f32
              %209 = arith.subf %208, %103 fastmath<fast> : f32
              %210 = arith.mulf %209, %148 fastmath<fast> : f32
              %211 = arith.subf %210, %204 fastmath<fast> : f32
              %212 = arith.mulf %26, %cst_3 fastmath<fast> : f32
              %213 = arith.negf %94 fastmath<fast> : f32
              %214 = arith.subf %95, %94 fastmath<fast> : f32
              %215 = arith.mulf %214, %cst_2 fastmath<fast> : f32
              %216 = arith.addf %215, %cst_1 fastmath<fast> : f32
              %217 = arith.mulf %216, %214 fastmath<fast> : f32
              %218 = arith.subf %217, %103 fastmath<fast> : f32
              %219 = arith.mulf %218, %148 fastmath<fast> : f32
              %220 = arith.subf %219, %212 fastmath<fast> : f32
              %221 = arith.mulf %28, %cst_3 fastmath<fast> : f32
              %222 = arith.subf %213, %95 fastmath<fast> : f32
              %223 = arith.mulf %222, %cst_2 fastmath<fast> : f32
              %224 = arith.addf %223, %cst_1 fastmath<fast> : f32
              %225 = arith.mulf %224, %222 fastmath<fast> : f32
              %226 = arith.subf %225, %103 fastmath<fast> : f32
              %227 = arith.mulf %226, %148 fastmath<fast> : f32
              %228 = arith.subf %227, %221 fastmath<fast> : f32
              %229 = arith.mulf %35, %cst_3 fastmath<fast> : f32
              %230 = arith.subf %96, %94 fastmath<fast> : f32
              %231 = arith.mulf %230, %cst_2 fastmath<fast> : f32
              %232 = arith.addf %231, %cst_1 fastmath<fast> : f32
              %233 = arith.mulf %232, %230 fastmath<fast> : f32
              %234 = arith.subf %233, %103 fastmath<fast> : f32
              %235 = arith.mulf %234, %148 fastmath<fast> : f32
              %236 = arith.subf %235, %229 fastmath<fast> : f32
              %237 = arith.mulf %36, %cst_3 fastmath<fast> : f32
              %238 = arith.subf %213, %96 fastmath<fast> : f32
              %239 = arith.mulf %238, %cst_2 fastmath<fast> : f32
              %240 = arith.addf %239, %cst_1 fastmath<fast> : f32
              %241 = arith.mulf %240, %238 fastmath<fast> : f32
              %242 = arith.subf %241, %103 fastmath<fast> : f32
              %243 = arith.mulf %242, %148 fastmath<fast> : f32
              %244 = arith.subf %243, %237 fastmath<fast> : f32
              scf.yield %107, %115, %121, %141, %147, %128, %134, %187, %220, %195, %228, %156, %164, %172, %179, %203, %211, %236, %244 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
            } else {
              scf.yield %18, %20, %19, %22, %21, %24, %23, %28, %27, %26, %25, %32, %31, %30, %29, %36, %35, %34, %33 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
            }
            %43 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xf32>
            affine.store %42#0, %43[%arg7 + %arg6 * 128] : memref<?xf32>
            affine.store %42#1, %43[%arg7 + %arg6 * 128 + 2365568] : memref<?xf32>
            affine.store %42#2, %43[%arg7 + %arg6 * 128 + 4730752] : memref<?xf32>
            affine.store %42#3, %43[%arg7 + %arg6 * 128 + 7096321] : memref<?xf32>
            affine.store %42#4, %43[%arg7 + %arg6 * 128 + 9461759] : memref<?xf32>
            affine.store %42#5, %43[%arg7 + %arg6 * 128 + 11842560] : memref<?xf32>
            affine.store %42#6, %43[%arg7 + %arg6 * 128 + 14177280] : memref<?xf32>
            affine.store %42#7, %43[%arg7 + %arg6 * 128 + 16558209] : memref<?xf32>
            affine.store %42#8, %43[%arg7 + %arg6 * 128 + 18923647] : memref<?xf32>
            affine.store %42#9, %43[%arg7 + %arg6 * 128 + 21288833] : memref<?xf32>
            affine.store %42#10, %43[%arg7 + %arg6 * 128 + 23654271] : memref<?xf32>
            affine.store %42#11, %43[%arg7 + %arg6 * 128 + 26035328] : memref<?xf32>
            affine.store %42#12, %43[%arg7 + %arg6 * 128 + 28370048] : memref<?xf32>
            affine.store %42#13, %43[%arg7 + %arg6 * 128 + 30765952] : memref<?xf32>
            affine.store %42#14, %43[%arg7 + %arg6 * 128 + 33100672] : memref<?xf32>
            affine.store %42#15, %43[%arg7 + %arg6 * 128 + 35496961] : memref<?xf32>
            affine.store %42#16, %43[%arg7 + %arg6 * 128 + 37831681] : memref<?xf32>
            affine.store %42#17, %43[%arg7 + %arg6 * 128 + 40227839] : memref<?xf32>
            affine.store %42#18, %43[%arg7 + %arg6 * 128 + 42562559] : memref<?xf32>
          }
          "enzymexla.polygeist_yield"() : () -> ()
        }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
        %13 = llvm.call @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
        %14 = arith.cmpi eq, %13, %c0_i32 : i32
        %15 = arith.addi %arg5, %c1_i32 overflow<nuw> : i32
        %16:3 = scf.if %14 -> (i32, i32, i1) {
          %17 = "enzymexla.gpu_wrapper"(%c120, %c150, %c1, %c120, %c1, %c1) ({
            affine.parallel (%arg6, %arg7) = (0, 0) to (18000, 120) {
              llvm.intr.experimental.noalias.scope.decl #alias_scope6
              llvm.intr.experimental.noalias.scope.decl #alias_scope7
              %25 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xf32>
              %26 = affine.load %25[%arg7 + %arg6 * 128] : memref<?xf32>
              %27 = affine.load %25[%arg7 + %arg6 * 128 + 2365440] : memref<?xf32>
              %28 = affine.load %25[%arg7 + %arg6 * 128 + 4730880] : memref<?xf32>
              %29 = affine.load %25[%arg7 + %arg6 * 128 + 7096320] : memref<?xf32>
              %30 = affine.load %25[%arg7 + %arg6 * 128 + 9461760] : memref<?xf32>
              %31 = affine.load %25[%arg7 + %arg6 * 128 + 11827200] : memref<?xf32>
              %32 = affine.load %25[%arg7 + %arg6 * 128 + 14192640] : memref<?xf32>
              %33 = affine.load %25[%arg7 + %arg6 * 128 + 16558080] : memref<?xf32>
              %34 = affine.load %25[%arg7 + %arg6 * 128 + 18923520] : memref<?xf32>
              %35 = affine.load %25[%arg7 + %arg6 * 128 + 21288960] : memref<?xf32>
              %36 = affine.load %25[%arg7 + %arg6 * 128 + 23654400] : memref<?xf32>
              %37 = affine.load %25[%arg7 + %arg6 * 128 + 26019840] : memref<?xf32>
              %38 = affine.load %25[%arg7 + %arg6 * 128 + 28385280] : memref<?xf32>
              %39 = affine.load %25[%arg7 + %arg6 * 128 + 30750720] : memref<?xf32>
              %40 = affine.load %25[%arg7 + %arg6 * 128 + 33116160] : memref<?xf32>
              %41 = affine.load %25[%arg7 + %arg6 * 128 + 35481600] : memref<?xf32>
              %42 = affine.load %25[%arg7 + %arg6 * 128 + 37847040] : memref<?xf32>
              %43 = affine.load %25[%arg7 + %arg6 * 128 + 40212480] : memref<?xf32>
              %44 = affine.load %25[%arg7 + %arg6 * 128 + 42577920] : memref<?xf32>
              %45 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi8>
              %46 = affine.load %45[%arg7 * 4 + %arg6 * 512 + 179773440] : memref<?xi8>
              %47 = arith.extui %46 : i8 to i32
              %48 = arith.andi %47, %c1_i32 : i32
              %49 = arith.cmpi eq, %48, %c0_i32 : i32
              %50:19 = scf.if %49 -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
                %52 = arith.addf %27, %26 fastmath<fast> : f32
                %53 = arith.addf %52, %28 fastmath<fast> : f32
                %54 = arith.addf %53, %29 fastmath<fast> : f32
                %55 = arith.addf %54, %30 fastmath<fast> : f32
                %56 = arith.addf %55, %31 fastmath<fast> : f32
                %57 = arith.addf %56, %32 fastmath<fast> : f32
                %58 = arith.addf %57, %33 fastmath<fast> : f32
                %59 = arith.addf %58, %34 fastmath<fast> : f32
                %60 = arith.addf %59, %35 fastmath<fast> : f32
                %61 = arith.addf %60, %36 fastmath<fast> : f32
                %62 = arith.addf %61, %37 fastmath<fast> : f32
                %63 = arith.addf %62, %38 fastmath<fast> : f32
                %64 = arith.addf %63, %39 fastmath<fast> : f32
                %65 = arith.addf %64, %40 fastmath<fast> : f32
                %66 = arith.addf %65, %41 fastmath<fast> : f32
                %67 = arith.addf %66, %42 fastmath<fast> : f32
                %68 = arith.addf %67, %43 fastmath<fast> : f32
                %69 = arith.addf %68, %44 fastmath<fast> : f32
                %70 = arith.addf %29, %33 fastmath<fast> : f32
                %71 = arith.addf %30, %34 fastmath<fast> : f32
                %72 = arith.addf %70, %35 fastmath<fast> : f32
                %73 = arith.addf %71, %36 fastmath<fast> : f32
                %74 = arith.subf %72, %73 fastmath<fast> : f32
                %75 = arith.addf %74, %41 fastmath<fast> : f32
                %76 = arith.addf %75, %42 fastmath<fast> : f32
                %77 = arith.addf %43, %44 fastmath<fast> : f32
                %78 = arith.subf %76, %77 fastmath<fast> : f32
                %79 = arith.subf %27, %28 fastmath<fast> : f32
                %80 = arith.addf %79, %33 fastmath<fast> : f32
                %81 = arith.addf %80, %34 fastmath<fast> : f32
                %82 = arith.addf %35, %36 fastmath<fast> : f32
                %83 = arith.subf %81, %82 fastmath<fast> : f32
                %84 = arith.addf %83, %37 fastmath<fast> : f32
                %85 = arith.addf %84, %38 fastmath<fast> : f32
                %86 = arith.addf %39, %40 fastmath<fast> : f32
                %87 = arith.subf %85, %86 fastmath<fast> : f32
                %88 = arith.addf %31, %37 fastmath<fast> : f32
                %89 = arith.addf %32, %38 fastmath<fast> : f32
                %90 = arith.addf %88, %39 fastmath<fast> : f32
                %91 = arith.addf %89, %40 fastmath<fast> : f32
                %92 = arith.addf %90, %41 fastmath<fast> : f32
                %93 = arith.addf %91, %42 fastmath<fast> : f32
                %94 = arith.addf %92, %43 fastmath<fast> : f32
                %95 = arith.addf %93, %44 fastmath<fast> : f32
                %96 = arith.subf %94, %95 fastmath<fast> : f32
                %97 = arith.divf %78, %69 fastmath<fast> : f32
                %98 = arith.divf %87, %69 fastmath<fast> : f32
                %99 = arith.divf %96, %69 fastmath<fast> : f32
                %100 = arith.andi %47, %c2_i32 : i32
                %101 = arith.cmpi eq, %100, %c0_i32 : i32
                %102 = arith.select %101, %97, %cst_11 : f32
                %103 = arith.select %101, %98, %cst_10 : f32
                %104 = arith.select %101, %99, %cst_9 : f32
                %105 = arith.mulf %102, %102 fastmath<fast> : f32
                %106 = arith.mulf %103, %103 fastmath<fast> : f32
                %107 = arith.addf %105, %106 fastmath<fast> : f32
                %108 = arith.mulf %104, %104 fastmath<fast> : f32
                %109 = arith.addf %107, %108 fastmath<fast> : f32
                %110 = arith.mulf %109, %cst_8 fastmath<fast> : f32
                %111 = arith.addf %110, %cst_7 fastmath<fast> : f32
                %112 = arith.mulf %69, %cst_6 fastmath<fast> : f32
                %113 = arith.mulf %26, %cst_5 fastmath<fast> : f32
                %114 = arith.mulf %112, %111 fastmath<fast> : f32
                %115 = arith.subf %113, %114 fastmath<fast> : f32
                %116 = arith.mulf %69, %cst_4 fastmath<fast> : f32
                %117 = arith.mulf %27, %cst_3 fastmath<fast> : f32
                %118 = arith.mulf %103, %cst_2 fastmath<fast> : f32
                %119 = arith.addf %118, %cst_1 fastmath<fast> : f32
                %120 = arith.mulf %119, %103 fastmath<fast> : f32
                %121 = arith.subf %120, %111 fastmath<fast> : f32
                %122 = arith.mulf %121, %116 fastmath<fast> : f32
                %123 = arith.subf %122, %117 fastmath<fast> : f32
                %124 = arith.mulf %28, %cst_3 fastmath<fast> : f32
                %125 = arith.addf %118, %cst_0 fastmath<fast> : f32
                %126 = arith.mulf %125, %103 fastmath<fast> : f32
                %127 = arith.subf %126, %111 fastmath<fast> : f32
                %128 = arith.mulf %127, %116 fastmath<fast> : f32
                %129 = arith.subf %128, %124 fastmath<fast> : f32
                %130 = arith.mulf %31, %cst_3 fastmath<fast> : f32
                %131 = arith.mulf %104, %cst_2 fastmath<fast> : f32
                %132 = arith.addf %131, %cst_1 fastmath<fast> : f32
                %133 = arith.mulf %132, %104 fastmath<fast> : f32
                %134 = arith.subf %133, %111 fastmath<fast> : f32
                %135 = arith.mulf %134, %116 fastmath<fast> : f32
                %136 = arith.subf %135, %130 fastmath<fast> : f32
                %137 = arith.mulf %32, %cst_3 fastmath<fast> : f32
                %138 = arith.addf %131, %cst_0 fastmath<fast> : f32
                %139 = arith.mulf %138, %104 fastmath<fast> : f32
                %140 = arith.subf %139, %111 fastmath<fast> : f32
                %141 = arith.mulf %140, %116 fastmath<fast> : f32
                %142 = arith.subf %141, %137 fastmath<fast> : f32
                %143 = arith.mulf %29, %cst_3 fastmath<fast> : f32
                %144 = arith.mulf %102, %cst_2 fastmath<fast> : f32
                %145 = arith.addf %144, %cst_1 fastmath<fast> : f32
                %146 = arith.mulf %145, %102 fastmath<fast> : f32
                %147 = arith.subf %146, %111 fastmath<fast> : f32
                %148 = arith.mulf %147, %116 fastmath<fast> : f32
                %149 = arith.subf %148, %143 fastmath<fast> : f32
                %150 = arith.mulf %30, %cst_3 fastmath<fast> : f32
                %151 = arith.addf %144, %cst_0 fastmath<fast> : f32
                %152 = arith.mulf %151, %102 fastmath<fast> : f32
                %153 = arith.subf %152, %111 fastmath<fast> : f32
                %154 = arith.mulf %153, %116 fastmath<fast> : f32
                %155 = arith.subf %154, %150 fastmath<fast> : f32
                %156 = arith.mulf %69, %cst fastmath<fast> : f32
                %157 = arith.mulf %37, %cst_3 fastmath<fast> : f32
                %158 = arith.addf %103, %104 fastmath<fast> : f32
                %159 = arith.mulf %158, %cst_2 fastmath<fast> : f32
                %160 = arith.addf %159, %cst_1 fastmath<fast> : f32
                %161 = arith.mulf %160, %158 fastmath<fast> : f32
                %162 = arith.subf %161, %111 fastmath<fast> : f32
                %163 = arith.mulf %162, %156 fastmath<fast> : f32
                %164 = arith.subf %163, %157 fastmath<fast> : f32
                %165 = arith.mulf %38, %cst_3 fastmath<fast> : f32
                %166 = arith.subf %103, %104 fastmath<fast> : f32
                %167 = arith.mulf %166, %cst_2 fastmath<fast> : f32
                %168 = arith.addf %167, %cst_1 fastmath<fast> : f32
                %169 = arith.mulf %168, %166 fastmath<fast> : f32
                %170 = arith.subf %169, %111 fastmath<fast> : f32
                %171 = arith.mulf %170, %156 fastmath<fast> : f32
                %172 = arith.subf %171, %165 fastmath<fast> : f32
                %173 = arith.mulf %39, %cst_3 fastmath<fast> : f32
                %174 = arith.subf %104, %103 fastmath<fast> : f32
                %175 = arith.mulf %174, %cst_2 fastmath<fast> : f32
                %176 = arith.addf %175, %cst_1 fastmath<fast> : f32
                %177 = arith.mulf %176, %174 fastmath<fast> : f32
                %178 = arith.subf %177, %111 fastmath<fast> : f32
                %179 = arith.mulf %178, %156 fastmath<fast> : f32
                %180 = arith.subf %179, %173 fastmath<fast> : f32
                %181 = arith.mulf %40, %cst_3 fastmath<fast> : f32
                %182 = arith.negf %158 fastmath<fast> : f32
                %183 = arith.subf %cst_1, %159 fastmath<fast> : f32
                %184 = arith.mulf %183, %182 fastmath<fast> : f32
                %185 = arith.subf %184, %111 fastmath<fast> : f32
                %186 = arith.mulf %185, %156 fastmath<fast> : f32
                %187 = arith.subf %186, %181 fastmath<fast> : f32
                %188 = arith.mulf %33, %cst_3 fastmath<fast> : f32
                %189 = arith.addf %102, %103 fastmath<fast> : f32
                %190 = arith.mulf %189, %cst_2 fastmath<fast> : f32
                %191 = arith.addf %190, %cst_1 fastmath<fast> : f32
                %192 = arith.mulf %191, %189 fastmath<fast> : f32
                %193 = arith.subf %192, %111 fastmath<fast> : f32
                %194 = arith.mulf %193, %156 fastmath<fast> : f32
                %195 = arith.subf %194, %188 fastmath<fast> : f32
                %196 = arith.mulf %35, %cst_3 fastmath<fast> : f32
                %197 = arith.subf %102, %103 fastmath<fast> : f32
                %198 = arith.mulf %197, %cst_2 fastmath<fast> : f32
                %199 = arith.addf %198, %cst_1 fastmath<fast> : f32
                %200 = arith.mulf %199, %197 fastmath<fast> : f32
                %201 = arith.subf %200, %111 fastmath<fast> : f32
                %202 = arith.mulf %201, %156 fastmath<fast> : f32
                %203 = arith.subf %202, %196 fastmath<fast> : f32
                %204 = arith.mulf %41, %cst_3 fastmath<fast> : f32
                %205 = arith.addf %102, %104 fastmath<fast> : f32
                %206 = arith.mulf %205, %cst_2 fastmath<fast> : f32
                %207 = arith.addf %206, %cst_1 fastmath<fast> : f32
                %208 = arith.mulf %207, %205 fastmath<fast> : f32
                %209 = arith.subf %208, %111 fastmath<fast> : f32
                %210 = arith.mulf %209, %156 fastmath<fast> : f32
                %211 = arith.subf %210, %204 fastmath<fast> : f32
                %212 = arith.mulf %42, %cst_3 fastmath<fast> : f32
                %213 = arith.subf %102, %104 fastmath<fast> : f32
                %214 = arith.mulf %213, %cst_2 fastmath<fast> : f32
                %215 = arith.addf %214, %cst_1 fastmath<fast> : f32
                %216 = arith.mulf %215, %213 fastmath<fast> : f32
                %217 = arith.subf %216, %111 fastmath<fast> : f32
                %218 = arith.mulf %217, %156 fastmath<fast> : f32
                %219 = arith.subf %218, %212 fastmath<fast> : f32
                %220 = arith.mulf %34, %cst_3 fastmath<fast> : f32
                %221 = arith.negf %102 fastmath<fast> : f32
                %222 = arith.subf %103, %102 fastmath<fast> : f32
                %223 = arith.mulf %222, %cst_2 fastmath<fast> : f32
                %224 = arith.addf %223, %cst_1 fastmath<fast> : f32
                %225 = arith.mulf %224, %222 fastmath<fast> : f32
                %226 = arith.subf %225, %111 fastmath<fast> : f32
                %227 = arith.mulf %226, %156 fastmath<fast> : f32
                %228 = arith.subf %227, %220 fastmath<fast> : f32
                %229 = arith.mulf %36, %cst_3 fastmath<fast> : f32
                %230 = arith.subf %221, %103 fastmath<fast> : f32
                %231 = arith.mulf %230, %cst_2 fastmath<fast> : f32
                %232 = arith.addf %231, %cst_1 fastmath<fast> : f32
                %233 = arith.mulf %232, %230 fastmath<fast> : f32
                %234 = arith.subf %233, %111 fastmath<fast> : f32
                %235 = arith.mulf %234, %156 fastmath<fast> : f32
                %236 = arith.subf %235, %229 fastmath<fast> : f32
                %237 = arith.mulf %43, %cst_3 fastmath<fast> : f32
                %238 = arith.subf %104, %102 fastmath<fast> : f32
                %239 = arith.mulf %238, %cst_2 fastmath<fast> : f32
                %240 = arith.addf %239, %cst_1 fastmath<fast> : f32
                %241 = arith.mulf %240, %238 fastmath<fast> : f32
                %242 = arith.subf %241, %111 fastmath<fast> : f32
                %243 = arith.mulf %242, %156 fastmath<fast> : f32
                %244 = arith.subf %243, %237 fastmath<fast> : f32
                %245 = arith.mulf %44, %cst_3 fastmath<fast> : f32
                %246 = arith.subf %221, %104 fastmath<fast> : f32
                %247 = arith.mulf %246, %cst_2 fastmath<fast> : f32
                %248 = arith.addf %247, %cst_1 fastmath<fast> : f32
                %249 = arith.mulf %248, %246 fastmath<fast> : f32
                %250 = arith.subf %249, %111 fastmath<fast> : f32
                %251 = arith.mulf %250, %156 fastmath<fast> : f32
                %252 = arith.subf %251, %245 fastmath<fast> : f32
                scf.yield %115, %123, %129, %149, %155, %136, %142, %195, %228, %203, %236, %164, %172, %180, %187, %211, %219, %244, %252 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
              } else {
                scf.yield %26, %28, %27, %30, %29, %32, %31, %36, %35, %34, %33, %40, %39, %38, %37, %44, %43, %42, %41 : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
              }
              %51 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
              affine.store %50#0, %51[%arg7 + %arg6 * 128] : memref<?xf32>
              affine.store %50#1, %51[%arg7 + %arg6 * 128 + 2365568] : memref<?xf32>
              affine.store %50#2, %51[%arg7 + %arg6 * 128 + 4730752] : memref<?xf32>
              affine.store %50#3, %51[%arg7 + %arg6 * 128 + 7096321] : memref<?xf32>
              affine.store %50#4, %51[%arg7 + %arg6 * 128 + 9461759] : memref<?xf32>
              affine.store %50#5, %51[%arg7 + %arg6 * 128 + 11842560] : memref<?xf32>
              affine.store %50#6, %51[%arg7 + %arg6 * 128 + 14177280] : memref<?xf32>
              affine.store %50#7, %51[%arg7 + %arg6 * 128 + 16558209] : memref<?xf32>
              affine.store %50#8, %51[%arg7 + %arg6 * 128 + 18923647] : memref<?xf32>
              affine.store %50#9, %51[%arg7 + %arg6 * 128 + 21288833] : memref<?xf32>
              affine.store %50#10, %51[%arg7 + %arg6 * 128 + 23654271] : memref<?xf32>
              affine.store %50#11, %51[%arg7 + %arg6 * 128 + 26035328] : memref<?xf32>
              affine.store %50#12, %51[%arg7 + %arg6 * 128 + 28370048] : memref<?xf32>
              affine.store %50#13, %51[%arg7 + %arg6 * 128 + 30765952] : memref<?xf32>
              affine.store %50#14, %51[%arg7 + %arg6 * 128 + 33100672] : memref<?xf32>
              affine.store %50#15, %51[%arg7 + %arg6 * 128 + 35496961] : memref<?xf32>
              affine.store %50#16, %51[%arg7 + %arg6 * 128 + 37831681] : memref<?xf32>
              affine.store %50#17, %51[%arg7 + %arg6 * 128 + 40227839] : memref<?xf32>
              affine.store %50#18, %51[%arg7 + %arg6 * 128 + 42562559] : memref<?xf32>
            }
            "enzymexla.polygeist_yield"() : () -> ()
          }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_Z50__device_stub__performStreamCollide_kernel_wrapperPfS_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"]], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88"]>} : (index, index, index, index, index, index) -> index
          %18 = llvm.call @cudaGetLastError() {no_unwind, uniform_work_group_size} : () -> i32
          %19 = arith.cmpi eq, %18, %c0_i32 : i32
          %20 = arith.cmpi eq, %15, %7 : i32
          %21 = arith.extui %20 : i1 to i32
          %22 = arith.select %19, %21, %c2_i32 : i32
          %23 = arith.cmpi ne, %15, %7 : i32
          %24 = arith.andi %19, %23 : i1
          scf.yield %22, %18, %24 : i32, i32, i1
        } else {
          scf.yield %c3_i32, %1, %false : i32, i32, i1
        }
        scf.condition(%16#2) %15, %16#1, %13, %16#0 : i32, i32, i32, i32
      } do {
      ^bb0(%arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32):
        scf.yield %arg5 : i32
      }
      %9 = arith.index_castui %8#3 : i32 to index
      %10 = arith.cmpi ne, %9, %c1 : index
      %11 = arith.extui %10 : i1 to i32
      scf.if %10 {
        %12 = arith.cmpi eq, %9, %c2 : index
        scf.if %12 {
          %13 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
          %14 = affine.load %13[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
          %15 = llvm.call @cudaGetErrorString(%8#1) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
          %16 = llvm.call @fprintf(%14, %0, %c51_i32, %15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
          llvm.call @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
        } else {
          %13 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
          %14 = affine.load %13[0] {alignment = 8 : i64} : memref<?x!llvm.ptr>
          %15 = llvm.call @cudaGetErrorString(%8#2) {no_unwind, uniform_work_group_size} : (i32 {llvm.noundef}) -> !llvm.ptr
          %16 = llvm.call @fprintf(%14, %0, %c51_i32, %15) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {cold, no_unwind, uniform_work_group_size} : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
          llvm.call @exit(%c-1_i32) {cold, no_unwind, noreturn, uniform_work_group_size} : (i32 {llvm.noundef}) -> ()
        }
      }
      scf.yield %11 : i32
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
