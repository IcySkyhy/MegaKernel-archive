# M6 Add+RMSNorm 融合残差归一化：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m6_rmsnorm/build -S m6_rmsnorm -DCMAKE_BUILD_TYPE=Release
#   cmake --build m6_rmsnorm/build -j4
# 运行：
#   ./m6_rmsnorm/build/m6_rmsnorm

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m6_rmsnorm LANGUAGES ASC CXX)

add_executable(m6_rmsnorm
    m6_rmsnorm.asc
)

target_compile_definitions(m6_rmsnorm PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m6_rmsnorm PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# host 侧 C 参考用 sqrtf
target_link_libraries(m6_rmsnorm PRIVATE m)
