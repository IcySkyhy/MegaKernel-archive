# M36 hyper-connection 融合 kernel：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录或本目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m20_hyperconn/build -S m20_hyperconn -DCMAKE_BUILD_TYPE=Release
#   cmake --build m20_hyperconn/build -j4
# 运行（在空目录里跑，dump 会落在 cwd）：
#   ./m20_hyperconn/build/m20_hyperconn [int|real|zero ...]
#   M20_DUMP=1 ./m20_hyperconn/build/m20_hyperconn        # 落盘供 check_ref.py
#   M20_STAGE_LIMIT=<n> ...                              # 段序截断（bring-up 定位）

cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
find_package(ASC REQUIRED)
project(m20_hyperconn LANGUAGES ASC CXX)
add_executable(m20_hyperconn m20_hyperconn.asc)
target_compile_definitions(m20_hyperconn PRIVATE NPU_ARCH_DAV_3510)
target_compile_options(m20_hyperconn PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    # host C 参考需与 kernel 内同一 IEEE 运算序列（禁止 FMA 融合），
    # 否则 h' / S1 无法做位级判据
    $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
)
# host 侧 C 参考用 sqrtf/expf
target_link_libraries(m20_hyperconn PRIVATE m)
