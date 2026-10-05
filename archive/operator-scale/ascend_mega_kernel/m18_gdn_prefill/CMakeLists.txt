# M18 GDN prefill chunk scan 核：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录，或在 m18_gdn_prefill 下用 -S .）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m18_gdn_prefill/build -S m18_gdn_prefill -DCMAKE_BUILD_TYPE=Release
#   cmake --build m18_gdn_prefill/build -j4
# 运行（在 build 目录内运行，dump 文件写 cwd）：
#   cd m18_gdn_prefill/build && ./m18_gdn_prefill
# 数值校验（numpy float64 参考，逐段相对容差）：
#   /usr/local/python3.12.13/bin/python3 ../check_ref.py

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m18_gdn_prefill LANGUAGES ASC CXX)

add_executable(m18_gdn_prefill
    m18_gdn_prefill.asc
)

target_compile_definitions(m18_gdn_prefill PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m18_gdn_prefill PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
