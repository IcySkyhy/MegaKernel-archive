# M161 PLE 单元：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt，不碰 m15_layer_loop/CMakeLists.txt）
#
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m29_ple_unit/build -S m29_ple_unit -DCMAKE_BUILD_TYPE=Release
#   cmake --build m29_ple_unit/build -j4
# 数据：
#   /usr/local/python3.12.13/bin/python3 m29_ple_unit/gen_unit_data.py --mode synth --out m29_ple_unit/data_synth
#   /usr/local/python3.12.13/bin/python3 m29_ple_unit/gen_unit_data.py --mode real  --out m29_ple_unit/data_real
# 运行（在输出目录内，dump 写 cwd）：
#   M29_OUT=$(pwd) M29_DATA=<datadir> ./m29_ple_unit/build/m29_ple_unit   # step 数取自数据目录的 unit_meta.txt
# 判据：
#   /usr/local/python3.12.13/bin/python3 m29_ple_unit/check_ref.py <outdir> <datadir>

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m29_ple_unit LANGUAGES ASC CXX)

add_executable(m29_ple_unit
    m29_ple_unit.asc
)

target_include_directories(m29_ple_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(m29_ple_unit PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m29_ple_unit PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host 参考需与 kernel 内向量指令同一 IEEE 运算序列（禁止 FMA 融合）
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)

target_link_libraries(m29_ple_unit PRIVATE m)
