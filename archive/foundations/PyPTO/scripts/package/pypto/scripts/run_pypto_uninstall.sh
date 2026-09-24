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
# PyPTO 卸载子脚本
#
# 本脚本是 _do_uninstall() 委托的同步子进程, 负责实际文件删除与包特有后处理. 出于以下原因以子进程方式执行实际操作:
#
#   1. 子进程隔离: install_common_parser.sh 内部失败时可能直接 exit,
#      子进程隔离保证父进程 (_do_uninstall) 仍可优雅处理错误 (chmod_end + 错误日志), 而非被 exit 直接终止整个 install.sh
#
#   2. 参数组装: install_common_parser.sh 需大量动态参数
#      (--copy_all --package --uninstall --username --usergroup --version --version-dir
#      --custom-options 等), 本脚本负责根据当前环境组装这些参数
#
#   3. 包特有后处理: install_common_parser.sh 只做通用文件删除,
#      PyPTO 的后处理 (pip uninstall whl) 由本脚本在删除完成后通过 custom 回调执行
#
#   4. 架构一致性: CANN 所有包 (runtime/toolkit/nnal 等) 均采用 run_xxx_uninstall.sh 中间层模式, 保持一致便于维护
#
# 变量获取策略 (两层优先级):
#   1. 已安装包文件 (ascend_install.info / version.info / scene.info) — 权威来源
#   2. 环境变量 (install.sh / uninstall.sh export 继承) — fallback
#
# 三层分工:
#   _do_uninstall              -> 卸载框架 (元数据/权限/版本/错误处理)
#   run_pypto_uninstall.sh     -> 参数组装 + 文件删除调度 + 包特有后处理
#   install_common_parser.sh   -> 通用文件删除引擎 (按 filelist.csv 删除文件, 设置权限)

########################################################################################################################
# 已安装包路径推算
########################################################################################################################

INSTALLED_SCRIPTS_PATH="$(dirname "$(readlink -f "$0")")"
readonly INSTALLED_ROOT_PATH="${INSTALLED_SCRIPTS_PATH}/.."
readonly ASCEND_INSTALL_INFO="${INSTALLED_ROOT_PATH}/ascend_install.info"
readonly INSTALLED_VERSION_FILE="${INSTALLED_ROOT_PATH}/version.info"
readonly INSTALLED_SCENE_FILE="${INSTALLED_ROOT_PATH}/scene.info"

########################################################################################################################
# 函数引入
########################################################################################################################
# shellcheck disable=SC1091
. "${INSTALLED_SCRIPTS_PATH}/common_func_v2.inc"
# shellcheck disable=SC1091
. "${INSTALLED_SCRIPTS_PATH}/common_func.inc"

comm_init_log
set_comm_log "PyPTO" "${COMM_LOGFILE}"

########################################################################################################################
# 变量获取: 已安装包文件 (权威来源)
########################################################################################################################

_read_installed_info() {
    if [ -f "${ASCEND_INSTALL_INFO}" ]; then
        local _val
        _val=$(grep -i "PyPTO_Install_Path_Param=" "${ASCEND_INSTALL_INFO}" | cut -d"=" -f2-)
        [ -n "${_val}" ] && INSTALLED_ARG_INSTALL_PATH="${_val}"
        _val=$(grep -i "PyPTO_Install_Type=" "${ASCEND_INSTALL_INFO}" | cut -d"=" -f2-)
        [ -n "${_val}" ] && INSTALLED_ARG_MODE="${_val}"
        _val=$(grep -i "PyPTO_Install_For_All=" "${ASCEND_INSTALL_INFO}" | cut -d"=" -f2-)
        [ -n "${_val}" ] && INSTALLED_ARG_INSTALL_FOR_ALL="${_val}"
        _val=$(grep -i "PyPTO_PyLocal=" "${ASCEND_INSTALL_INFO}" | cut -d"=" -f2-)
        [ -n "${_val}" ] && INSTALLED_ARG_PY_LOCAL="${_val}"
    fi
    if [ -f "${INSTALLED_VERSION_FILE}" ]; then
        local _val
        _val=$(grep "^version_dir=" "${INSTALLED_VERSION_FILE}" | cut -d"=" -f2-)
        [ -n "${_val}" ] && INSTALLED_PKG_VERSION_DIR_NAME="${_val}"
        _val=""
        get_version "_val" "${INSTALLED_VERSION_FILE}"
        [ -n "${_val}" ] && INSTALLED_PKG_VERSION="${_val}"
        local _multi=""
        is_multi_version_pkg "_multi" "${INSTALLED_VERSION_FILE}"
        [ -n "${_multi}" ] && INSTALLED_PKG_IS_MULTI_VERSION="${_multi}"
    fi
    if [ -f "${INSTALLED_SCENE_FILE}" ]; then
        local _val
        _val=$(grep -iw arch "${INSTALLED_SCENE_FILE}" | cut -d"=" -f2- | awk '{print tolower($0)}')
        [ -n "${_val}" ] && INSTALLED_PKG_ARCH_NAME="${_val}"
    fi
}

_read_installed_info

########################################################################################################################
# 变量获取: 三层优先级合并
#   1. 已安装包文件 (INSTALLED_xxx)  — 权威来源
#   2. 环境变量 (xxx)                — fallback (install.sh / uninstall.sh export 继承)
########################################################################################################################
# 参数默认值 (无参数传入时使用)
# 框架参数
ARG_QUIET="${ARG_QUIET:-n}"

# 用户名, 用户组
tmp_user_name=$(id -un)
tmp_user_group=$(id -gn)
readonly USER_NAME="${tmp_user_name}"
readonly USER_GROUP="${tmp_user_group}"
unset tmp_user_name
unset tmp_user_group
export USER_NAME
export USER_GROUP

ARG_MODE="${INSTALLED_ARG_MODE:-${ARG_MODE}}"
ARG_INSTALL_PATH="${INSTALLED_ARG_INSTALL_PATH:-${ARG_INSTALL_PATH}}"
ARG_INSTALL_FOR_ALL="${INSTALLED_ARG_INSTALL_FOR_ALL:-${ARG_INSTALL_FOR_ALL:-n}}"
ARG_PY_LOCAL="${INSTALLED_ARG_PY_LOCAL:-${ARG_PY_LOCAL:-n}}"
ARG_DOCKER_ROOT="${ARG_DOCKER_ROOT:-}"
PKG_VERSION_DIR_NAME="${INSTALLED_PKG_VERSION_DIR_NAME:-${PKG_VERSION_DIR_NAME:-cann}}"
PKG_INSTALL_PATH="${PKG_INSTALL_PATH:-${ARG_INSTALL_PATH}/${PKG_VERSION_DIR_NAME}}"
PKG_SHARE_INFO_INSTALL_PATH="${INSTALLED_SCRIPTS_PATH}/.."
PKG_IS_MULTI_VERSION="${INSTALLED_PKG_IS_MULTI_VERSION:-${PKG_IS_MULTI_VERSION:-false}}"
PKG_VERSION="${INSTALLED_PKG_VERSION:-${PKG_VERSION:-}}"
PKG_ARCH_NAME="${INSTALLED_PKG_ARCH_NAME:-${PKG_ARCH_NAME:-}}"
PKG_WHL_INSTALL_PATH="${PKG_WHL_INSTALL_PATH:-${PKG_INSTALL_PATH}/python/site-packages}"

export ARG_QUIET ARG_MODE ARG_INSTALL_PATH ARG_INSTALL_FOR_ALL ARG_PY_LOCAL ARG_DOCKER_ROOT
export PKG_VERSION_DIR_NAME PKG_INSTALL_PATH PKG_SHARE_INFO_INSTALL_PATH
export PKG_IS_MULTI_VERSION PKG_VERSION PKG_ARCH_NAME PKG_WHL_INSTALL_PATH

# 已安装包内文件
SOURCE_INSTALL_COMMON_PARSER="${PKG_SHARE_INFO_INSTALL_PATH}/script/install_common_parser.sh"
SOURCE_FILELIST="${PKG_SHARE_INFO_INSTALL_PATH}/script/filelist.csv"

########################################################################################################################
# 日志与进度
########################################################################################################################
_output_progress() {
    comm_log "INFO" "${LOG_PKG_NAME} uninstall upgradePercentage:${1}%"
}

########################################################################################################################
# 主流程
########################################################################################################################

comm_log "INFO" "step into run_pypto_uninstall.sh ......"
comm_log "INFO" "uninstall target dir ${PKG_INSTALL_PATH}, type ${ARG_MODE}."

if [ ! -d "${PKG_SHARE_INFO_INSTALL_PATH}" ]; then
    comm_log "ERROR" "ERR_NO:0x0001;ERR_DES:path ${PKG_SHARE_INFO_INSTALL_PATH} is not exist."
    exit 1
fi
if [ ! -d "${PKG_INSTALL_PATH}" ]; then
    comm_log "ERROR" "ERR_NO:0x0001;ERR_DES:path ${PKG_INSTALL_PATH} is not exist."
    exit 1
fi

_do_uninstall() {
    _output_progress 10

    chmod +w -R "${SOURCE_INSTALL_COMMON_PARSER}" 2> /dev/null

    local custom_options=""
    # shellcheck disable=SC2086
    sh "${SOURCE_INSTALL_COMMON_PARSER}" --package="pypto" --uninstall \
        --username="${USER_NAME}" --usergroup="${USER_GROUP}" \
        --version="${PKG_VERSION}" --version-dir="${PKG_VERSION_DIR_NAME}" --use-share-info \
        --docker-root="${ARG_DOCKER_ROOT}" ${custom_options} "${ARG_MODE}" "${ARG_INSTALL_PATH}" "${SOURCE_FILELIST}"
    local sh_ret=$?
    if [ "${sh_ret}" -ne 0 ]; then
        comm_log "ERROR" "ERR_NO:0x0090;ERR_DES:Failed to uninstall package."
        return 1
    fi

    return 0
}

if ! _do_uninstall; then
    exit 1
fi

_output_progress 100
exit 0
