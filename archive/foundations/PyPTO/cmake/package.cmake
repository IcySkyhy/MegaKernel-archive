# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
set(ARCH_LINUX_PATH "${CMAKE_SYSTEM_PROCESSOR}-linux")
message(STATUS "ARCH_LINUX_PATH=${ARCH_LINUX_PATH}")

# pypto 仓专有脚本
#       1. 全量安装对应脚本目录下的所有文件;
#       2. 新增文件需同步修改 scripts/package/pypto/pypto.xml 中配置;
set(_PyPtoScriptPrefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/${PyPTO_PkgName}/scripts)
install(DIRECTORY ${_PyPtoScriptPrefix}/
        DESTINATION share/info/${PyPTO_PkgName}/script
        FILE_PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
        DIRECTORY_PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
        COMPONENT ${PyPTO_PkgName}
)

# cann-cmake 公共脚本
set(SCRIPTS_FILES
        ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
        ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
        ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
        ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
        ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
        ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
        ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
        ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
        ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
        ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
        ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
        ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
)
install(FILES ${SCRIPTS_FILES}
        DESTINATION share/info/${PyPTO_PkgName}/script
        COMPONENT ${PyPTO_PkgName}
)

# version 配置
#       由 add_cann_version_info_targets 命令产出
install(FILES ${CMAKE_BINARY_DIR}/version.${PyPTO_PkgName}.info
        DESTINATION share/info/${PyPTO_PkgName}
        RENAME version.info
        COMPONENT ${PyPTO_PkgName}
)

# whl
#       — 打包阶段 install 到 aarch64-linux/lib64 (与 GE 一致, dst_path == install_path)
#       - 安装阶段 --copy_all 直接拷到 cann/aarch64-linux/lib64
#       - pip install --target 展开到 python/site-packages 目录
if (NOT ENABLE_UNIFIED_BUILD)
    install(FILES ${WHL_FILE_PATH}
            DESTINATION ${ARCH_LINUX_PATH}/lib64
            COMPONENT ${PyPTO_PkgName})
endif ()
