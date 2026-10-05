# probe_vf_loop：3510 VF 内层循环 trip count 探针（独立 CMake 工程，不依赖仓库顶层 CMakeLists.txt）
#
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
# 全量复现 + 归档证据（推荐；含预期编译失败的靶子与 5 次独立进程运行）：
#   bash run_probes.sh [reps]        # reps 默认 5
#
# 两个可执行文件（变体由编译宏选择，见各自文件头注释）：
#   probe_vf_nest.asc         —— 嵌套循环 trip count 矩阵（outer × inner × body × rows）
#   probe_arange_gather.asc   —— ① Arange 支持类型取证（含 uint32_t 失败靶子）
#                                ② Gather(vgather2) 可寻址范围扫描
#
# target 命名：
#   nest_o<outer>i<inner>b<body>[_ct<rows>][_O2]   探针 nest 的一个变体
#   ag_<form>_sweep                                探针 ②（AG_FORM=0）
#   ag_arange_u32                                  probe ① 的失败靶子（EXCLUDE_FROM_ALL）

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ASC REQUIRED)

project(probe_vf_loop LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

# ---------------- 探针 nest：trip count 矩阵 ----------------
# 规格 "target=outer=inner=body=ct_rows"；ct_rows 只在 outer==2（编译期上界）时有意义。
set(NEST_TARGETS
    # A 组：运行期 rows（uint16_t），body 1 = m18 TrilSolveVF 同形态
    "nest_o0i0b1=0=0=1=0"     # ← finding 原文形态（内层上界 = 外层归纳变量）
    "nest_o0i1b1=0=1=1=0"     # 内层上界 i-1（带保护；不带保护的写法在 i=0 下溢 ⇒ 见 evidence/unguarded_inner1）
    "nest_o0i2b1=0=2=1=0"     # 内层上界 i+1
    "nest_o0i3b1=0=3=1=0"     # 内层上界 rows（**不**依赖外层归纳变量）—— 对照
    "nest_o0i4b1=0=4=1=0"     # 上界提出到内层循环外（u16）
    "nest_o0i5b1=0=5=1=0"     # 上界提出到内层循环外（u32 类型）
    "nest_o0i6b1=0=6=1=0"     # 上界提出到内层循环外（i32 类型）
    "nest_o0i7b1=0=7=1=0"     # `j+1 < i`（i-1 的不下溢写法）
    "nest_o0i7b0=0=7=0=0"     # 同上，body 0（只计数）
    # B 组：body 0 = 只做向量计数（隔离循环控制流）
    "nest_o0i0b0=0=0=0=0"
    "nest_o0i3b0=0=3=0=0"
    "nest_o0i4b0=0=4=0=0"
    # C 组：body 3 = 行 store→load 依赖（m18 行前代形态，检验内存可见性假设）
    "nest_o0i0b3=0=0=3=0"
    "nest_o0i3b3=0=3=3=0"
    # C2 组：body 4 = 逐轮序号标记落盘 —— **刻意不进默认 target 列表**：
    #   实测读数自相矛盾（rows=1 时 cnt=246 而 64 个标记槽位全非零；mk 的逐行重置被忽略，
    #   序号跨行连续），本探针未能解释，故不作为证据；原始日志留在
    #   evidence/negmark/negmark_o0i0b4_rep1.log.txt，代码保留（PROBE_BODY=4）供后续复核。
    #   要做该变体：把下面两行取消注释（rows 只测 0..4）。
    #   "nest_o0i0b4=0=0=4=0"
    #   "nest_o0i3b4=0=3=4=0"
    # D 组：内层循环 #pragma unroll
    "nest_o1i0b1=1=0=1=0"
    "nest_o1i3b1=1=3=1=0"
    # E 组：外层编译期常量（模板常量形态，donor InverseAIVVF<32> 同款）
    "nest_o2i0b1_ct1=2=0=1=1"
    "nest_o2i0b1_ct2=2=0=1=2"
    "nest_o2i0b1_ct3=2=0=1=3"
    "nest_o2i0b1_ct4=2=0=1=4"
    "nest_o2i0b1_ct64=2=0=1=64"
    "nest_o2i3b1_ct4=2=3=1=4"
    # F 组：外层归纳变量类型（u32 / i32 / u32 直接进地址算术）
    "nest_o3i0b1=3=0=1=0"
    "nest_o4i0b1=4=0=1=0"
    "nest_o5i0b1=5=0=1=0")

function(nest_add spec)
    string(REPLACE "=" ";" parts "${spec}")
    list(GET parts 0 NEST_TGT)
    list(GET parts 1 NEST_O)
    list(GET parts 2 NEST_I)
    list(GET parts 3 NEST_B)
    list(GET parts 4 NEST_CT)
    add_executable(${NEST_TGT} probe_vf_nest.asc)
    target_compile_definitions(${NEST_TGT} PRIVATE NPU_ARCH_DAV_3510
                                                     PROBE_OUTER=${NEST_O} PROBE_INNER=${NEST_I} PROBE_BODY=${NEST_B}
                                                     PROBE_CT_ROWS=${NEST_CT} PROBE_TAG=${NEST_TGT})
    target_compile_options(${NEST_TGT} PRIVATE ${ASC_ARCH_FLAGS})
endfunction()

foreach(SPEC IN LISTS NEST_TARGETS)
    nest_add("${SPEC}")
endforeach()

# -O2 对照组（探针是否与优化档位相关）：编译选项在 build-type flags 之后追加，故 -O2 生效
foreach(O2SPEC "nest_o0i0b1_O2=0=1" "nest_o0i3b1_O2=3=1")
    string(REPLACE "=" ";" p "${O2SPEC}")
    list(GET p 0 O2TGT_BASE)
    list(GET p 1 O2_INNER)
    list(GET p 2 O2_BODY)
    add_executable(${O2TGT_BASE} probe_vf_nest.asc)
    target_compile_definitions(${O2TGT_BASE} PRIVATE NPU_ARCH_DAV_3510 PROBE_OUTER=0 PROBE_INNER=${O2_INNER}
                                                     PROBE_BODY=${O2_BODY} PROBE_CT_ROWS=0 PROBE_TAG=${O2TGT_BASE})
    target_compile_options(${O2TGT_BASE} PRIVATE ${ASC_ARCH_FLAGS} -O2)
endforeach()

# ---- 诊断靶子：关掉编译器内部某个 pass，用于**归属**取证（M43 追加）----
# bisheng 驱动默认带这些内部开关（完整 cc1 命令行见 evidence/logs/bisheng_verbose_cc1_flags.txt）：
#   -mllvm -cce-aicore-hwloops=false  -mllvm -enable-vloop-lowering=true
#   -mllvm -cce-aicore-backend-loop-lower=true  -mllvm -cce-aicore-loop-unrotate=true
# 规格 "target=outer=inner=body=<额外 -mllvm 开关>"；现象若随开关消失/出现，即可定位到 pass 级。
set(DIAG_TARGETS
    "nest_o0i0b0_NOUNROT=0=0=0=-cce-aicore-loop-unrotate=false"
    "nest_o0i0b1_NOUNROT=0=0=1=-cce-aicore-loop-unrotate=false")

foreach(SPEC IN LISTS DIAG_TARGETS)
    string(REPLACE "=" ";" p "${SPEC}")
    list(GET p 0 D_TGT)
    list(GET p 1 D_O)
    list(GET p 2 D_I)
    list(GET p 3 D_B)
    list(GET p 4 D_FLAG)
    add_executable(${D_TGT} probe_vf_nest.asc)
    target_compile_definitions(${D_TGT} PRIVATE NPU_ARCH_DAV_3510 PROBE_OUTER=${D_O} PROBE_INNER=${D_I}
                                                 PROBE_BODY=${D_B} PROBE_CT_ROWS=0 PROBE_TAG=${D_TGT})
    target_compile_options(${D_TGT} PRIVATE ${ASC_ARCH_FLAGS} -mllvm ${D_FLAG})
endforeach()

# ---------------- 预期编译失败的归档靶子（EXCLUDE_FROM_ALL）----------------
# body 2：VF 循环体内的**标量算术**（预期 "Unsupported scalar instruction in AIV loop"）
add_executable(nest_o0i0b2_scal EXCLUDE_FROM_ALL probe_vf_nest.asc)
target_compile_definitions(nest_o0i0b2_scal PRIVATE NPU_ARCH_DAV_3510 PROBE_OUTER=0 PROBE_INNER=0 PROBE_BODY=2
                                                    PROBE_CT_ROWS=0 PROBE_TAG=nest_o0i0b2_scal)
target_compile_options(nest_o0i0b2_scal PRIVATE ${ASC_ARCH_FLAGS})

# body 5：VF 循环体内的**标量算术 + 标量 store**（同一句 fatal error；r1 评审附注要求把
#         "标量 store 也不行" 也做成自己的归档靶子，而不是只靠文字声称）
add_executable(nest_o0i0b2s_scalstore EXCLUDE_FROM_ALL probe_vf_nest.asc)
target_compile_definitions(nest_o0i0b2s_scalstore PRIVATE NPU_ARCH_DAV_3510 PROBE_OUTER=0 PROBE_INNER=0
                                                          PROBE_BODY=5 PROBE_CT_ROWS=0
                                                          PROBE_TAG=nest_o0i0b2s_scalstore)
target_compile_options(nest_o0i0b2s_scalstore PRIVATE ${ASC_ARCH_FLAGS})

# body 4：**正对照** —— 同样「每轮一次 store」，但 store 是向量 `StoreAlign`（地址含 j、值不含 j）
#         ⇒ **能编译**（只做编译，不跑；跑出来的读数异常另见 evidence/negmark/）。
#         用途：证明被拒的是「标量指令」而不是「循环里做 store」本身。
add_executable(nest_o0i0b4_marker_ctl EXCLUDE_FROM_ALL probe_vf_nest.asc)
target_compile_definitions(nest_o0i0b4_marker_ctl PRIVATE NPU_ARCH_DAV_3510 PROBE_OUTER=0 PROBE_INNER=0
                                                      PROBE_BODY=4 PROBE_CT_ROWS=0
                                                      PROBE_TAG=nest_o0i0b4_marker_ctl)
target_compile_options(nest_o0i0b4_marker_ctl PRIVATE ${ASC_ARCH_FLAGS})

# ---------------- 探针 arange / gather ----------------
add_executable(ag_gather_sweep probe_arange_gather.asc)
target_compile_definitions(ag_gather_sweep PRIVATE NPU_ARCH_DAV_3510 AG_FORM=0 AG_NEL=16384
                                                  AG_TAG=ag_gather_sweep)
target_compile_options(ag_gather_sweep PRIVATE ${ASC_ARCH_FLAGS})

# Arange<uint32_t> 失败靶子：预期 static_assert "current Arange data type is not supported on current device!"
add_executable(ag_arange_u32 EXCLUDE_FROM_ALL probe_arange_gather.asc)
target_compile_definitions(ag_arange_u32 PRIVATE NPU_ARCH_DAV_3510 AG_FORM=1 AG_NEL=16384 AG_TAG=ag_arange_u32)
target_compile_options(ag_arange_u32 PRIVATE ${ASC_ARCH_FLAGS})
