# M16 L0 装载几何标定：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m16_load_geom/build -S m16_load_geom -DCMAKE_BUILD_TYPE=Release
#   cmake --build m16_load_geom/build -j4
# 运行（在 build 目录内运行，dump 文件写 cwd）：
#   cd m16_load_geom/build && ./m16_geom 2d      # 2D LoadData2DParamsV2 逐字段标定
#   cd m16_load_geom/build && ./m16_geom 3d      # 3D LoadData3DParamsV2 enTranspose 标定
#   cd m16_load_geom/build && ./m16_pv           # 真实 BMM2 形状（P·V）验证

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m16_load_geom LANGUAGES ASC CXX)

add_executable(m16_geom m16_geom.asc)
add_executable(m16_pv m16_pv.asc)

foreach(tgt m16_geom m16_pv)
    target_compile_definitions(${tgt} PRIVATE NPU_ARCH_DAV_3510)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    )
endforeach()

# 说明：校验脚本需要 numpy，用 /usr/local/python3.12.13/bin/python3（干净 shell 里的 python3 没有 numpy）
