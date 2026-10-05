# M95 的 E=512/top-10 独立锤点（m95_e512）—— 独立 CMake 工程（不依赖 m15_layer_loop/CMakeLists.txt）
#
# 构建：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -B m15_layer_loop/moe_relift/m95_e512/build -S m15_layer_loop/moe_relift/m95_e512 \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build m15_layer_loop/moe_relift/m95_e512/build -j4
#
# 档位（默认 = 真实规模 E=512 / top-10；改 -DM95_E / -DM95_TOPK 可换档）：
#   **不改仓库里的资源表**：本工程在构建目录里按字面替换两个规模常量生成 m15_moe_resources.h
#   的副本，并把它排在 include 路径**最前**以遮蔽仓库那份 ⇒ 被编进去的 m15_moe_layer.h 是仓库
#   本体，只有资源表的 NUM_EXPERTS / TOPK_MAX 两个字面值换了档（差异可用 diff 复核）。

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(M95_E 512 CACHE STRING "锤点档的 NUM_EXPERTS")
set(M95_TOPK 10 CACHE STRING "锤点档的 TOPK_MAX")

find_package(ASC REQUIRED)
project(m95_e512 LANGUAGES ASC CXX)

# M15_SRC = 被测代码所在的目录（默认本目录的 ../..）。用 `-DM15_SRC=<副本目录>` 可指向一份
# 变异副本做**负向对照**（把 m15_moe_layer.h / m15_moe_resources.h 拷到 /tmp 再改，见 README）。
set(M15_SRC_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
if (NOT DEFINED M15_SRC)
  set(M15_SRC "${M15_SRC_DEFAULT}" CACHE PATH "被测代码目录（m15_layer_loop）")
endif()

# ---- 档位资源表：只按字面替换两个字面值（其余逐字节相同）----
file(READ "${M15_SRC}/m15_moe_resources.h" M95_RES_TXT)
set(M95_RES_OUT "${M95_RES_TXT}")
string(REPLACE "constexpr uint32_t NUM_EXPERTS = 4;" "constexpr uint32_t NUM_EXPERTS = ${M95_E};" M95_RES_OUT "${M95_RES_OUT}")
string(REPLACE "constexpr uint32_t TOPK_MAX = 4;" "constexpr uint32_t TOPK_MAX = ${M95_TOPK};" M95_RES_OUT "${M95_RES_OUT}")
if (M95_RES_OUT STREQUAL M95_RES_TXT)
  message(FATAL_ERROR "档位替换没生效（资源表的字面值形态变了）—— 先核 m15_moe_resources.h 的规模常量写法")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/m15_moe_resources.h" "${M95_RES_OUT}")

# ---- 被测代码：m15_moe_layer.h 的**逐字节副本** ----
# 必须复制（而不是只靠 include 路径）：`#include "m15_moe_layer.h"` 找到 M15_SRC 那份后，它内部的
# `#include "m15_moe_resources.h"` 会**先**在「包含它的文件所在目录」里找 ⇒ 落回未改档的仓库那份
# （实测：第一次构建就踩到了，harness 里 TOPK_MAX 仍是 4 ⇒ 运行期 topk 10 被拒）。把副本放在构建
# 目录后，两者同目录 ⇒ 档位表生效，且被编进去的 m15_moe_layer.h 与仓库**逐字节相同**（见 tier 文件
# 里的 sha256 对照）。
file(COPY "${M15_SRC}/m15_moe_layer.h" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(SHA256 "${M15_SRC}/m15_moe_layer.h" M95_LAYER_SHA_SRC)
file(SHA256 "${CMAKE_CURRENT_BINARY_DIR}/m15_moe_layer.h" M95_LAYER_SHA_CPY)
file(SHA256 "${M15_SRC}/m15_moe_resources.h" M95_RES_SHA_SRC)
file(SHA256 "${CMAKE_CURRENT_BINARY_DIR}/m15_moe_resources.h" M95_RES_SHA_CPY)
if (NOT M95_LAYER_SHA_SRC STREQUAL M95_LAYER_SHA_CPY)
  message(FATAL_ERROR "m15_moe_layer.h 的副本与仓库不一致 —— 被测代码必须逐字节相同")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/m95_tier.txt"
     "NUM_EXPERTS=${M95_E}\nTOPK_MAX=${M95_TOPK}\n"
     "m15_moe_layer.h     源 sha256=${M95_LAYER_SHA_SRC}\n"
     "m15_moe_layer.h   副本 sha256=${M95_LAYER_SHA_CPY}   ← 必须相等（逐字节相同，只换了档位表）\n"
     "m15_moe_resources.h 源 sha256=${M95_RES_SHA_SRC}\n"
     "m15_moe_resources.h 副本 sha256=${M95_RES_SHA_CPY}   ← 只有 NUM_EXPERTS/TOPK_MAX 两个字面值不同\n")

add_executable(m95_e512 m95_e512.asc)

# 构建目录（档位资源表 + 被测代码副本）必须排在 M15_SRC 之前
target_include_directories(m95_e512 PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${M15_SRC}"
)

target_compile_definitions(m95_e512 PRIVATE
    NPU_ARCH_DAV_3510
    M95_HEADER="m15_moe_layer.h"
)

target_compile_options(m95_e512 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_link_libraries(m95_e512 m)
