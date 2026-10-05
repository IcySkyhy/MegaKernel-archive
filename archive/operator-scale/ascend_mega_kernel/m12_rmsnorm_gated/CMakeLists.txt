# M12 GDN 输出 RMSNormGated：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m12_rmsnorm_gated/build -S m12_rmsnorm_gated -DCMAKE_BUILD_TYPE=Release
#   cmake --build m12_rmsnorm_gated/build -j4
# 运行：
#   ./m12_rmsnorm_gated/build/m12_rmsnorm_gated

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m12_rmsnorm_gated LANGUAGES ASC CXX)

add_executable(m12_rmsnorm_gated
    m12_rmsnorm_gated.asc
)

target_compile_definitions(m12_rmsnorm_gated PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m12_rmsnorm_gated PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# host 侧 C 参考用 sqrtf/expf
target_link_libraries(m12_rmsnorm_gated PRIVATE m)
