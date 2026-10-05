# M5 SwiGLU 融合 MXFP4 量化 epilogue：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m5_swiglu_quant/build -S m5_swiglu_quant -DCMAKE_BUILD_TYPE=Release
#   cmake --build m5_swiglu_quant/build -j4
# 运行：
#   ./m5_swiglu_quant/build/m5_swiglu_quant

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m5_swiglu_quant LANGUAGES ASC CXX)

add_executable(m5_swiglu_quant
    m5_swiglu_quant.asc
)

target_compile_definitions(m5_swiglu_quant PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m5_swiglu_quant PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# host 侧 CPU 参考（SwiGLU 的 expf）需要 libm
target_link_libraries(m5_swiglu_quant PRIVATE m)
