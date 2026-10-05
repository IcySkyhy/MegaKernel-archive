# M10 attention decode FA core：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m10_attn_decode/build -S m10_attn_decode -DCMAKE_BUILD_TYPE=Release
#   cmake --build m10_attn_decode/build -j4
# 运行（在 build 目录内运行，dump 文件写 cwd；中间量 dump 与日志归档见 data/、logs/，见 README §5/§6）：
#   cd m10_attn_decode/build && ./m10_attn_decode
# 数值校验（独立 numpy 参考；判定口径 = 边界感知 ulp，见 README §6.6 / check_ref.py 头部）：
#   /usr/local/python3.12.13/bin/python3 ../check_ref.py            # 三档判定 + gp + pv
#   /usr/local/python3.12.13/bin/python3 ../check_partials.py       # seq=4096 的 P̃ 翻转定位（需 wsacc/wsm/wss dump）
# 探针（各自的 README §6.x 有说明）：
#   ./m10_probe <mode>  ./m10_l0probe  ./m10_mmadprobe  ./m10_expprobe  ./m10_dup_probe <mode 0..13>

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m10_attn_decode LANGUAGES ASC CXX)

add_executable(m10_attn_decode
    m10_attn_decode.asc
)

add_executable(m10_probe
    m10_probe.asc
)

add_executable(m10_l0probe
    m10_l0probe.asc
)

add_executable(m10_mmadprobe
    m10_mmadprobe.asc
)

add_executable(m10_expprobe
    m10_expprobe.asc
)

target_compile_definitions(m10_attn_decode PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_definitions(m10_probe PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_definitions(m10_l0probe PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_definitions(m10_mmadprobe PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_definitions(m10_expprobe PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m10_attn_decode PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_options(m10_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_options(m10_l0probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_options(m10_mmadprobe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_options(m10_expprobe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

# 探针 host 侧用 libm（exp）
target_link_libraries(m10_expprobe PRIVATE m)

# AICore exception 507015 的隔离探针（reviewer r2 提供，本仓库收录；用法：./m10_dup_probe <mode 0..13>）
add_executable(m10_dup_probe
    m10_dup_probe.asc
)

target_compile_definitions(m10_dup_probe PRIVATE
    NPU_ARCH_DAV_3510
)

target_compile_options(m10_dup_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
