#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# PyPTO 版本依赖检查脚本
#
# 检查 PyPTO 与 driver 版本是否匹配.
# 自行解析 version.info 的 required_package_driver_version 字段与 driver 的 Version= 字段,
# 使用 common_func.inc 的 version_vaild() 做版本范围比较.

CURPATH="$(dirname "$(readlink -f "$0")")"
DEP_INFO_FILE="/etc/ascend_install.info"

if [ -z "$1" ]; then
    version_info_file="$(dirname "${CURPATH}")/version.info"
    driver_install_path_param="$(grep -iw Driver_Install_Path_Param "${DEP_INFO_FILE}" 2>/dev/null | cut -d"=" -f2-)"
    driver_ver_file="${driver_install_path_param}/driver/version.info"
else
    version_info_file="$1"
    driver_ver_file="$2"
fi

# shellcheck disable=SC1091
. "${CURPATH}/common_func.inc"

if [ ! -f "${version_info_file}" ]; then
    comm_log "WARNING" "file ${version_info_file} not exists!"
    exit 0
fi

required_driver_ver="$(grep -i "^required_package_driver_version=" "${version_info_file}" | cut -d"=" -f2- | tr -d '"')"
if [ -z "${required_driver_ver}" ]; then
    comm_log "WARNING" "required_package_driver_version not found in ${version_info_file}, skip driver version check."
    exit 0
fi

# version_vaild() 的 else 分支是 version_ge (>=) 语义, 去掉 >= 前缀后直接传纯版本号即可.
# 支持 >= 和 > 前缀: >= 去掉后用 version_vaild (>=语义), > 去掉后用 version_gt.
clean_required_ver="$(echo "${required_driver_ver}" | sed 's/^>=//' | sed 's/^>//')"
if [ "${clean_required_ver}" != "${required_driver_ver}" ]; then
    # 去掉了前缀, 需要区分 >= 和 > 语义
    if echo "${required_driver_ver}" | grep -q "^>="; then
        ver_check_cmp="ge"
    else
        ver_check_cmp="gt"
    fi
    required_driver_ver="${clean_required_ver}"
else
    ver_check_cmp="ge"
fi

if [ -z "${driver_ver_file}" ]; then
    comm_log "WARNING" "Cannot find the install path of driver."
    exit 0
fi

if [ ! -f "${driver_ver_file}" ]; then
    comm_log "WARNING" "file ${driver_ver_file} not exists!"
    exit 0
fi

actual_driver_ver="$(grep -i "^Version=" "${driver_ver_file}" | cut -d"=" -f2-)"
if [ -z "${actual_driver_ver}" ]; then
    comm_log "WARNING" "Version field not found in ${driver_ver_file}"
    exit 0
fi

if [ "${ver_check_cmp}" = "ge" ]; then
    if version_vaild "${required_driver_ver}" "${actual_driver_ver}"; then
        comm_log "INFO" "Check version matched! driver version ${actual_driver_ver} satisfies >=${required_driver_ver}"
        exit 0
    fi
elif [ "${ver_check_cmp}" = "gt" ]; then
    if version_gt "${actual_driver_ver}" "${required_driver_ver}"; then
        comm_log "INFO" "Check version matched! driver version ${actual_driver_ver} satisfies >${required_driver_ver}"
        exit 0
    fi
fi
comm_log "WARNING" "Check version does not matched! driver version ${actual_driver_ver} does not satisfy ${required_driver_ver}"
exit 1
