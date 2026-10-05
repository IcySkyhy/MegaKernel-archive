# M4 GDN decode 递推核：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m4_gdn_recurrent/build -S m4_gdn_recurrent -DCMAKE_BUILD_TYPE=Release
#   cmake --build m4_gdn_recurrent/build -j4
# 运行（在 build 目录内运行，dump 文件写 cwd）：
#   cd m4_gdn_recurrent/build && ./m4_gdn_recurrent
# 数值校验（numpy fp32 参考，逐 head 相对容差 1e-5）：
#   /usr/local/python3.12.13/bin/python3 ../check_ref.py

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m4_gdn_recurrent LANGUAGES ASC CXX)

add_executable(m4_gdn_recurrent
    m4_gdn_recurrent.asc
)

target_compile_definitions(m4_gdn_recurrent PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m4_gdn_recurrent PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
