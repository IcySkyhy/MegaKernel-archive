# M8 MoE 专家槽位重排/反重排核：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m8_permute/build -S m8_permute -DCMAKE_BUILD_TYPE=Release
#   cmake --build m8_permute/build -j4
# 运行（默认消费 tools/golden/data/{m1,m33}，可用参数换数据根/用例名）：
#   ./m8_permute/build/m8_permute [tools/golden/data] [case ...]

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m8_permute LANGUAGES ASC CXX)

add_executable(m8_permute
    m8_permute.asc
)

target_compile_definitions(m8_permute PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m8_permute PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host C 参考需与 kernel 内独立 Muls/Add 向量指令同一 IEEE 运算序列
    # （禁止 FMA 融合），否则 unpermute 加权折叠无法逐位对齐
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)
