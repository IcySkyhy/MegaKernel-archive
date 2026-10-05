# probe_vf_ldst：VF load/store 分布模式（LoadDist/StoreDist）语义最小探针（独立 CMake 工程）
# 构建（本目录下）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
# 运行 + 归档证据：
#   bash run_probes.sh 5        # 每个变体 5 次独立进程运行，逐次 dump + sha256
#
# 目标（变体矩阵全部编译期宏选择，同名 .asc 多次实例化）：
#   probe_a_first_elem  探针 A：StoreDist::DIST_FIRST_ELEMENT_B32（文档：忽略 mask，只写 src 第一个元素）
#   probe_a_norm_b32    探针 A 对照：同一掩码换 DIST_NORM_B32（掩码应当生效）
#   probe_b_*           探针 B：MTE2 写 UB 槽 → BufferID 交接 → V 侧读（BRC / 全宽 / GetValue）的可见性矩阵
#
# 三个互斥的探针 B 维度（宏，见 probe_b_ldst_handover.asc 顶部注释）：
#   PROBE_B_REL   历史 release 档位键；全项目 all-false 政策下恒 false（0/1/2 均 = CANN ASC_LOCK_BLOCK 默认、阻塞）
#   PROBE_B_READ  读法：0=DIST_BRC_B32 1=LoadAlign 全宽(DIST_NORM) 2=LocalTensor::GetValue 3=BRC+LocalMemBar
#   PROBE_B_GRAN  搬运粒度：0=整块256B 1=32B 2=4B(DataCopyPad)
#   PROBE_B_MID   中间 V 操作：0=无 1=有（对无关 32B scratch 做 store→load）
#   PROBE_B_LAYOUT 槽布局：0=所有 chunk 复用一个槽 1=每 chunk 独立槽
#   PROBE_B_PROD  生产者：0=MTE2 写 1=V 自己写（M36 的规避形式，阳性对照）
#   PROBE_B_OFF   读地址偏移（元素）：0=槽首 1=+4B

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(probe_vf_ldst LANGUAGES ASC CXX)

set(ASC_ARCH_FLAGS $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

# ---------------- 探针 A：StoreDist 掩码语义 ----------------
function(probe_add_a target dist alt)
    add_executable(${target} probe_a_storedist.asc)
    target_compile_definitions(${target} PRIVATE NPU_ARCH_DAV_3510 PROBE_A_DIST=${dist}
                                                 PROBE_A_ALT=${alt} PROBE_A_TAG=${target})
    target_compile_options(${target} PRIVATE ${ASC_ARCH_FLAGS})
endfunction()

# alt=0 连续掩码族（UpdateMask(0/1/8/64)）；alt=1 非连续/交替掩码族（M4 / M3 / MaskGenWithRegTensor 交替）
probe_add_a(probe_a_first_elem 0 0)
probe_add_a(probe_a_norm_b32 1 0)
probe_add_a(probe_a_first_elem_nc 0 1)
probe_add_a(probe_a_norm_b32_nc 1 1)

# ---------------- 探针 B：MTE2→V 交接可见性矩阵 ----------------
# 规格串：target=rel=read=gran=mid=layout=prod=off=vrel=backlog
set(B_VARIANTS
    # --- Stage 1 核心：{立即/drain/drain+barrier} × {BRC/全宽/GetValue}，整块 256B、复用一槽 ---
    "probe_b_rel0_brc=0=0=0=0=0=0=0=0=0"
    "probe_b_rel1_brc=1=0=0=0=0=0=0=0=0"
    "probe_b_rel2_brc=2=0=0=0=0=0=0=0=0"
    "probe_b_rel0_norm=0=1=0=0=0=0=0=0=0"
    "probe_b_rel1_norm=1=1=0=0=0=0=0=0=0"
    "probe_b_rel2_norm=2=1=0=0=0=0=0=0=0"
    "probe_b_rel0_getval=0=2=0=0=0=0=0=0=0"
    "probe_b_rel1_getval=1=2=0=0=0=0=0=0=0"
    "probe_b_rel2_getval=2=2=0=0=0=0=0=0=0"
    # --- Stage 2：BRC + LocalMemBar（负对照：membar 管不到 MTE2 数据通路）---
    "probe_b_rel1_brc_membar=1=3=0=0=0=0=0=0=0"
    # --- Stage 2：中间 V 操作（store→load 的无关 32B scratch）---
    "probe_b_rel1_brc_mid=1=0=0=1=0=0=0=0=0"
    # --- Stage 3：搬运粒度 32B / 4B ---
    "probe_b_rel1_brc_g32=1=0=1=0=0=0=0=0=0"
    "probe_b_rel1_brc_g4=1=0=2=0=0=0=0=0=0"
    # --- Stage 3：每 chunk 独立槽（无同槽复用）---
    "probe_b_rel1_brc_perchunk=1=0=0=0=1=0=0=0=0"
    # --- 阳性对照 1：V 自己写槽再 BRC（M36 的规避形式）---
    "probe_b_rel1_brc_vprod=1=0=0=0=0=1=0=0=0"
    # --- 读地址 +4B（4B 对齐、非 32B 对齐的槽首偏移）---
    "probe_b_rel1_brc_off4=1=0=0=0=0=0=1=0=0"
    # --- 非空洞性对照：生产者关闭（只握手不写）⇒ 必须全读到 POISON ---
    "probe_b_noprod=1=0=0=0=0=2=0=0=0"
    # --- 阳性对照 2：消费者释放改立即生效 ⇒ 生产者可覆盖同槽 ---
    "probe_b_vrelimm=1=0=0=0=0=0=0=1=0"
    "probe_b_rel0_vrelimm=0=0=0=0=0=0=0=1=0"
    # --- Stage 4 关键："我们少了一步 drain" 假说的可判别实验 —— MTE2 带 16KB 在飞 backlog ---
    "probe_b_backlog_rel1=1=0=0=0=0=0=0=0=1"
    "probe_b_backlog_rel0=0=0=0=0=0=0=0=0=1"
    "probe_b_backlog_rel2=2=0=0=0=0=0=0=0=1"
    # --- 判读器自检：生产者故意写上一 chunk 的值 ⇒ 分类器必须报 stale_prev ---
    "probe_b_prod3_selftest=1=0=0=0=0=3=0=0=0")

# ---------------- 探针 C：同核 GM 回环（MTE3: UB→GM 紧接 MTE2: GM→UB）--------------
function(probe_add_c target order shape)
    add_executable(${target} probe_c_gm_roundtrip.asc)
    target_compile_definitions(${target} PRIVATE NPU_ARCH_DAV_3510 PROBE_C_ORDER=${order}
                                                 PROBE_C_SHAPE=${shape} PROBE_C_TAG=${target})
    target_compile_options(${target} PRIVATE ${ASC_ARCH_FLAGS})
endfunction()

# shape=0 一次性 16KB 整块回写；shape=1 逐 chunk 16B 小回写 ×64（复刻 M36 的 W0 形态）
probe_add_c(probe_c_rt_unordered 0 0)
probe_add_c(probe_c_rt_event 1 0)
probe_add_c(probe_c_rt_bufid 2 0)
probe_add_c(probe_c_rt_twotok 3 0)             # 两个不同 BufferID（复刻 M36 事实），整块回写
probe_add_c(probe_c_rt_small_unordered 0 1)    # 逐 chunk 小回写 + 无排序
probe_add_c(probe_c_rt_small_twotok 3 1)       # 逐 chunk 小回写 + 两个不同 BufferID（完全复刻 M36）
probe_add_c(probe_c_rt_small_bufid 2 1)        # 逐 chunk 小回写 + 有排序（对照，应 PASS）

function(probe_add_b spec)
    string(REPLACE "=" ";" parts "${spec}")
    list(GET parts 0 B_TGT)
    list(GET parts 1 B_REL)
    list(GET parts 2 B_READ)
    list(GET parts 3 B_GRAN)
    list(GET parts 4 B_MID)
    list(GET parts 5 B_LAYOUT)
    list(GET parts 6 B_PROD)
    list(GET parts 7 B_OFF)
    list(GET parts 8 B_VREL)
    list(GET parts 9 B_BACKLOG)
    add_executable(${B_TGT} probe_b_ldst_handover.asc)
    target_compile_definitions(${B_TGT} PRIVATE NPU_ARCH_DAV_3510
                                                 PROBE_B_REL=${B_REL} PROBE_B_READ=${B_READ}
                                                 PROBE_B_GRAN=${B_GRAN} PROBE_B_MID=${B_MID}
                                                 PROBE_B_LAYOUT=${B_LAYOUT} PROBE_B_PROD=${B_PROD}
                                                 PROBE_B_OFF=${B_OFF} PROBE_B_VREL=${B_VREL}
                                                 PROBE_B_BACKLOG=${B_BACKLOG}
                                                 PROBE_B_TAG=${B_TGT})
    target_compile_options(${B_TGT} PRIVATE ${ASC_ARCH_FLAGS})
endfunction()

foreach(SPEC IN LISTS B_VARIANTS)
    probe_add_b("${SPEC}")
endforeach()
