# M28 GDN prefill epilog 转位/落位段：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m28_gdn_epilog/build -S m28_gdn_epilog -DCMAKE_BUILD_TYPE=Release
#   cmake --build m28_gdn_epilog/build -j4
# 运行（默认 m=4 与 m=4097 两档，可用参数换档）：
#   ./m28_gdn_epilog/build/m28_gdn_epilog [m ...]
#   M28_CASES=4,4097 M28_DUMP=1 ./m28_gdn_epilog/build/m28_gdn_epilog
#   M28_MUTANT=headstride ./m28_gdn_epilog/build/m28_gdn_epilog 4   # 负向对照（判据必须变红）

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m28_gdn_epilog LANGUAGES ASC CXX)

add_executable(m28_gdn_epilog
    m28_gdn_epilog.asc
)

target_compile_definitions(m28_gdn_epilog PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m28_gdn_epilog PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# ---- M149：epilog 链段（转位/落位 → S5 RMSNormGated → S6 out_proj → hcAttnOut）----
# 独立 target：自己的入口符号 + 自己的 main()；本 target 只验证「链段」，不接进 m15_layer_loop。
# 构建/运行：
#   cmake --build m28_gdn_epilog/build -j4 --target m28_epilog_chain
#   M28C_CASES=4,4097 M28C_DUMP=1 ./m28_gdn_epilog/build/m28_epilog_chain
#   M28C_MUT=nogamma ./m28_gdn_epilog/build/m28_epilog_chain 4   # 负向对照（判据必须变红）
add_executable(m28_epilog_chain
    m28_epilog_chain.asc
)

target_compile_definitions(m28_epilog_chain PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m28_epilog_chain PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# host 侧 C 参考用 sqrtf/expf
target_link_libraries(m28_epilog_chain PRIVATE m)
