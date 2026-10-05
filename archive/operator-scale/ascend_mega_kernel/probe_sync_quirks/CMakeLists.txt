# probe_sync_quirks：3510 quirk 最小复现探针（独立 CMake 工程，不依赖仓库顶层 CMakeLists.txt）
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
# 运行 + 归档证据：
#   bash run_probes.sh                # 全量：构建每个 target（含预期失败的 B 变体）+ 跑 A + 落盘
#
# 目标：
#   probe_a_mrgsort4                 Kernel A（MrgSort/MrgSort4 变参矩阵），默认构建
#   probe_b_idx_<t>_<form>           Kernel B（VF 内运行时下标标量加载），见下方说明
#
# Kernel B 的 target 命名：probe_b_idx_<idx 类型>_<形式>
#   form = raw   向量循环内直接 w[idx]（疑似触发 "Unsupported Inst must be hoisted"）
#   form = hoist 把标量加载搬到 VF 外（怀疑是正确/规避写法）
# idx 类型：s16=int16_t s32=int32_t s64=int64_t u16=uint16_t u32=uint32_t u64=uint64_t
#
# 默认构建集合只含「已验证可编译」的 target（编译不过的变体用 EXCLUDE_FROM_ALL，
# 由 run_probes.sh 用 --target 显式构建并单独抓取 stderr 原文，见 evidence/）。

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_sync_quirks LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

# ---------------- Kernel A ----------------
add_executable(probe_a_mrgsort4 probe_a_mrgsort4.asc)
target_compile_definitions(probe_a_mrgsort4 PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_a_mrgsort4 PRIVATE ${ASC_ARCH_FLAGS})

# A2：MrgSort4 API vs MrgSort(MrgSort4Info) API 对照（3510 上 MrgSort4 落 deprecated no-op 重载，
# 会有 [[deprecated]] 编译告警——告警原文也是证据，构建日志由 run_probes.sh 全量归档）
add_executable(probe_a2_mrgsort4_api probe_a2_mrgsort4_api.asc)
target_compile_definitions(probe_a2_mrgsort4_api PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(probe_a2_mrgsort4_api PRIVATE ${ASC_ARCH_FLAGS})

# ---------------- Kernel B ----------------
# 每个变体一个可执行文件，用 "target=idx 类型=form" 三元组描述；
# form: 0=VF 内裸指针 wP[j]、1=标量加载搬到 VF 外(UB)、2=VF 内编译期界循环、3=编译期常量下标对照组、
#       4=VF 内 LocalTensor::GetValue、5=VF 内 GlobalTensor::GetValue（**预期编译失败**）、
#       6=标量累加器+UB 动态下标、7=GlobalTensor::GetValue 搬到 VF 外。
# 编译不过的 form 5 进 EXCLUDE_FROM_ALL：默认 `cmake --build build` 不受影响，
# run_probes.sh 用 --target 显式构建它并单独抓 stderr 原文（这就是要归档的错误证据）。

set(B_DEFAULT
    # ---- form 0：探针本体（VF 内运行时界循环 + 动态下标标量加载），6 种 idx 类型齐测 ----
    "probe_b_idx_s16_f0_raw=int16_t=0"
    "probe_b_idx_s32_f0_raw=int32_t=0"
    "probe_b_idx_s64_f0_raw=int64_t=0"
    "probe_b_idx_u16_f0_raw=uint16_t=0"
    "probe_b_idx_u32_f0_raw=uint32_t=0"
    "probe_b_idx_u64_f0_raw=uint64_t=0"
    # ---- form 1：标量加载 hoist 到 VF 外 ----
    "probe_b_idx_s16_f1_hoistub=int16_t=1"
    "probe_b_idx_s32_f1_hoistub=int32_t=1"
    "probe_b_idx_s64_f1_hoistub=int64_t=1"
    "probe_b_idx_u32_f1_hoistub=uint32_t=1"
    # ---- form 2：VF 内编译期界循环，下标仍为运行时量 ----
    "probe_b_idx_s32_f2_ctbound=int32_t=2"
    "probe_b_idx_s64_f2_ctbound=int64_t=2"
    # ---- form 3：对照组（编译期常量下标）----
    "probe_b_idx_s32_f3_constidx=int32_t=3"
    # ---- form 4：VF 内 LocalTensor::GetValue ----
    "probe_b_idx_s16_f4_ubget=int16_t=4"
    "probe_b_idx_s32_f4_ubget=int32_t=4"
    "probe_b_idx_s64_f4_ubget=int64_t=4"
    # ---- form 6：标量累加器 + UB 动态下标 ----
    "probe_b_idx_s32_f6_acc=int32_t=6"
    "probe_b_idx_s64_f6_acc=int64_t=6"
    # ---- form 7：GlobalTensor::GetValue 搬到 VF 外（正确写法）----
    "probe_b_idx_s32_f7_gmhoist=int32_t=7"
    "probe_b_idx_s64_f7_gmhoist=int64_t=7"
    # ---- form 9：VF 内 LoadAlign 非 32B 对齐的归档演示（编译通过、运行预期 aclError=507035）----
    "probe_b_idx_s32_f9_misalign_demo=int32_t=9")

set(B_EXTRA
    # ---- form 5：VF 内 GlobalTensor::GetValue（预期编译失败；idx 被强转 uint32 是 GetValue 的接口约束）----
    "probe_b_idx_s16_f5_gmget=int16_t=5"
    "probe_b_idx_s32_f5_gmget=int32_t=5"
    "probe_b_idx_s64_f5_gmget=int64_t=5"
    "probe_b_idx_u16_f5_gmget=uint16_t=5"
    "probe_b_idx_u32_f5_gmget=uint32_t=5"
    "probe_b_idx_u64_f5_gmget=uint64_t=5"
    # ---- form 8：VF 内 __gm__ 裸指针 + IdxT 原类型（无强转）（预期编译失败）----
    "probe_b_idx_s16_f8_gmraw=int16_t=8"
    "probe_b_idx_s32_f8_gmraw=int32_t=8"
    "probe_b_idx_s64_f8_gmraw=int64_t=8"
    "probe_b_idx_u16_f8_gmraw=uint16_t=8"
    "probe_b_idx_u32_f8_gmraw=uint32_t=8"
    "probe_b_idx_u64_f8_gmraw=uint64_t=8")

function(probe_add_b spec excluded)
    string(REPLACE "=" ";" parts "${spec}")
    list(GET parts 0 B_TGT)
    list(GET parts 1 B_TP)
    list(GET parts 2 B_FORM)
    if(excluded)
        add_executable(${B_TGT} EXCLUDE_FROM_ALL probe_b_vec_idx.asc)
    else()
        add_executable(${B_TGT} probe_b_vec_idx.asc)
    endif()
    target_compile_definitions(${B_TGT} PRIVATE NPU_ARCH_DAV_3510 PROBE_B_IDX_T=${B_TP} PROBE_B_FORM=${B_FORM}
                                              PROBE_B_TAG=${B_TGT})
    target_compile_options(${B_TGT} PRIVATE ${ASC_ARCH_FLAGS})
endfunction()

foreach(SPEC IN LISTS B_DEFAULT)
    probe_add_b("${SPEC}" FALSE)
endforeach()
foreach(SPEC IN LISTS B_EXTRA)
    probe_add_b("${SPEC}" TRUE)
endforeach()

# VF 循环归纳变量类型硬约束的归档靶子：form 0 + uint32_t 归纳变量（预期编译失败）
add_executable(probe_b_idx_s32_f0_u32loop EXCLUDE_FROM_ALL probe_b_vec_idx.asc)
target_compile_definitions(probe_b_idx_s32_f0_u32loop PRIVATE NPU_ARCH_DAV_3510 PROBE_B_IDX_T=int32_t
                                                          PROBE_B_FORM=0 PROBE_B_LOOPVAR=uint32_t
                                                          PROBE_B_TAG=probe_b_idx_s32_f0_u32loop)
target_compile_options(probe_b_idx_s32_f0_u32loop PRIVATE ${ASC_ARCH_FLAGS})
