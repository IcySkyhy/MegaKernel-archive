# M119（Wave B5）hc prefill 行分块的独立验证路：自带工程（**不碰 m15_layer_loop/CMakeLists.txt**）
#
# 构建（仓库根目录）：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m27_hc_prefill/build -S m27_hc_prefill -DCMAKE_BUILD_TYPE=Release
#   cmake --build m27_hc_prefill/build -j4
# 运行（在空目录里跑；dump 落在 cwd）：
#   flock -w 900 /tmp/npu0.lock bash -c 'cd /tmp/m27_out && M27_DUMP=1 <build>/m27_hc_prefill'
# 判据（三态退出码）：
#   python3.12 m27_hc_prefill/check_ref.py /tmp/m27_out

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(m27_hc_prefill LANGUAGES ASC CXX)

# 同一个 TU 编两次，只有块高不同（MT=8 缺省 / MT=12 上界）：第二个 target 是"块循环不是只对 8
# 调出来的"的编译期+运行期见证（见 README §验证）。
function(m27_add tgt mt)
    add_executable(${tgt} m27_hc_prefill.asc)
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../m15_layer_loop)
    target_compile_definitions(${tgt} PRIVATE NPU_ARCH_DAV_3510 M15_HC_PF_MT=${mt})
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
        # host 侧无 C 参考（判据在 python 的 numpy float64），但保持与其它 mission 同口径
        $<$<COMPILE_LANGUAGE:ASC>:-ffp-contract=off>
    )
    target_link_libraries(${tgt} PRIVATE m)
endfunction()

m27_add(m27_hc_prefill 8)
m27_add(m27_hc_prefill_mt12 12)
