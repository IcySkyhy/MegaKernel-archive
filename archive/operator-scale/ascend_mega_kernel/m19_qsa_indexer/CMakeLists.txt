# M19 QSA indexer 前端 kernel：独立 CMake 工程（不依赖仓库顶层 CMakeLists.txt）
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m19_qsa_indexer/build -S m19_qsa_indexer -DCMAKE_BUILD_TYPE=Release
#   cmake --build m19_qsa_indexer/build -j4
# 运行（仓库根目录：
#   ./m19_qsa_indexer/build/m19_qsa_indexer                 # 设备自检（写 m19_out/）
#   M19_OUT=./m19_out ./m19_qsa_indexer/build/m19_qsa_indexer
# 交叉验证（对 check_ref.py 的 double 参考）：
#   /usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/check_ref.py ./m19_out

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m19_qsa_indexer LANGUAGES ASC CXX)

add_executable(m19_qsa_indexer
    m19_qsa_indexer.asc
)

# 探针：AIV 的 128 维 GemmaRMSNorm + partial NeoX RoPE 四段分离判定
add_executable(probe_norm_rope
    probe_norm_rope.asc
)

# probe_aic_blockidx 已撤回（M156）：其 AIC 分支经 UB 暂存落盘，AIC 访问不了 UB ⇒ 摆位不成立。
# 源文件保留为撤回说明，不再构建；见 README §5.4 与 probe_aic_blockidx.asc 文件头。

# 探针（M53）：AIV 间计数交换屏障的可见性（4 个变体；主 kernel 选择段的 K 分裂根因判据）
add_executable(probe_aiv_barrier
    probe_aiv_barrier.asc
)

foreach(tgt m19_qsa_indexer probe_norm_rope probe_aiv_barrier)
    target_compile_definitions(${tgt} PRIVATE NPU_ARCH_DAV_3510)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
    target_link_libraries(${tgt} m)
endforeach()
