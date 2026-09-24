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
# PyPTO 独立卸载入口脚本
#
# 安装后放在 cann/share/info/pypto/script/uninstall.sh, 供用户在已安装目录下直接 bash uninstall.sh 卸载.
# 本脚本从 ascend_install.info / version.info 获取安装信息, export 与 install.sh 同名的变量,
# 然后调用 run_pypto_uninstall.sh (run_pypto_uninstall.sh 会优先从已安装包文件读取变量).

########################################################################################################################
# 全局变量
########################################################################################################################

# 参数默认值
ARG_QUIET=n                     # 静默标记, 用于控制日志输出的静默模式

# 包安装路径
ARG_INSTALL_PATH=""             # 安装顶层目录, 从 ascend_install.info 读取
PKG_VERSION_DIR_NAME=""         # 包安装路径中 version 级名称
PKG_INSTALL_PATH=""             # 包安装路径 (包含多版本层级)
PKG_SHARE_INFO_INSTALL_PATH=""  # 包共享信息安装路径 (含多版本层级 + share/info/pypto)

# 已安装包路径
#   uninstall.sh 运行在已安装目录下, 而非临时包目录
tmp_cfd="$(dirname "$(readlink -f "$0")")"
readonly INSTALLED_SCRIPTS_PATH="$tmp_cfd"
unset tmp_cfd
tmp_dir="$(readlink -f "${INSTALLED_SCRIPTS_PATH}/..")"
readonly PKG_SHARE_INFO_INSTALL_PATH="${tmp_dir}"
unset tmp_dir
readonly INSTALLED_VERSION_FILE="${PKG_SHARE_INFO_INSTALL_PATH}/version.info"
readonly ASCEND_INSTALL_INFO_FILE="${PKG_SHARE_INFO_INSTALL_PATH}/ascend_install.info"
readonly PKG_INSTALL_INFO_FILE="${PKG_SHARE_INFO_INSTALL_PATH}/ascend_install.info"
readonly ASCEND_INSTALL_INFO_OLD_FILE="/etc/ascend_install.info"

UNINSTALL_SHELL="${INSTALLED_SCRIPTS_PATH}/run_pypto_uninstall.sh"

########################################################################################################################
# 函数引入
#   在实际运行时, 对应脚本会拷贝到对应位置, 但在源码目录结构中, 对应脚本可能不存在, 故屏蔽对应告警.
########################################################################################################################
# shellcheck disable=SC1091
. "${INSTALLED_SCRIPTS_PATH}/common_func_v2.inc"
# shellcheck disable=SC1091
. "${INSTALLED_SCRIPTS_PATH}/common_func.inc"
# shellcheck disable=SC1091
. "${INSTALLED_SCRIPTS_PATH}/pypto_func.sh"

########################################################################################################################
# 全局初始化
########################################################################################################################

# 日志初始化
pypto_comm_log_init

# 多版本包标记
tmp_pkg_mulit_version=""
is_multi_version_pkg "tmp_pkg_mulit_version" "$INSTALLED_VERSION_FILE"
PKG_IS_MULTI_VERSION="${tmp_pkg_mulit_version}"
unset tmp_pkg_mulit_version
export PKG_IS_MULTI_VERSION

# 包版本
tmp_pkg_verison=""
get_version "tmp_pkg_verison" "$INSTALLED_VERSION_FILE"
PKG_VERSION="${tmp_pkg_verison}"
unset tmp_pkg_verison
export PKG_VERSION

# 版本目录名
PKG_VERSION_DIR_NAME="cann"
if [ -f "${INSTALLED_VERSION_FILE}" ]; then
    tmp_version_dir="$(grep "^version_dir=" "${INSTALLED_VERSION_FILE}" | cut -d"=" -f2-)"
    if [ -n "${tmp_version_dir}" ]; then
        PKG_VERSION_DIR_NAME="${tmp_version_dir}"
    fi
    unset tmp_version_dir
fi

########################################################################################################################
# 辅助函数定义
########################################################################################################################

# 功能: 从 install_info 文件中读取键值
#   参数: key, file
_get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param=""
    if [ ! -f "${_file}" ]; then
        return 1
    fi
    _param=$(grep -i "${_key}=" "${_file}" | cut -d"=" -f2-)
    echo "${_param}"
}

_parse_args() {
    while true; do
        case "$1" in
        --quiet)
            ARG_QUIET=y
            shift
            ;;
        *)
            if [ -n "$1" ]; then
                comm_log "ERROR" "ERR_NO:0x0004;ERR_DES:Unrecognized parameters: $1. Only support '--quiet'."
                exit 1
            fi
            break
            ;;
        esac
    done
}

########################################################################################################################
# 业务处理流程
########################################################################################################################

# 解析入参
_parse_args "$@"

# 读取已安装包的安装信息
num=""
ARG_MODE=""
if [ -f "${ASCEND_INSTALL_INFO_FILE}" ]; then
    ARG_INSTALL_PATH=$(_get_install_param "PyPTO_Install_Path_Param" "${ASCEND_INSTALL_INFO_FILE}")
    ARG_MODE=$(_get_install_param "PyPTO_Install_Type" "${ASCEND_INSTALL_INFO_FILE}")
elif [ -f "${ASCEND_INSTALL_INFO_OLD_FILE}" ]; then
    num=$(grep -c -i pypto_install_path_param "${ASCEND_INSTALL_INFO_OLD_FILE}")
    if [ "${num}" != "0" ]; then
        ARG_INSTALL_PATH="$(grep -iw pypto_install_path_param "${ASCEND_INSTALL_INFO_OLD_FILE}" | cut -d"=" -f2-)"
        ARG_MODE="$(grep -iw pypto_install_type "${ASCEND_INSTALL_INFO_OLD_FILE}" | cut -d"=" -f2-)"
    fi
else
    comm_log "ERROR" "ERR_NO:0x0080;ERR_DES:please complete ${ASCEND_INSTALL_INFO_FILE} or ${ASCEND_INSTALL_INFO_OLD_FILE}"
    exit_log 1
fi

# 组合包安装路径
if [ "$PKG_IS_MULTI_VERSION" = "true" ]; then
    PKG_INSTALL_PATH="${ARG_INSTALL_PATH}/${PKG_VERSION_DIR_NAME}"
else
    PKG_INSTALL_PATH="${ARG_INSTALL_PATH}"
fi

# 导出参数 (与 install.sh 同名, 让 run_pypto_uninstall.sh 的 fallback 机制生效)
export ARG_QUIET
export ARG_INSTALL_PATH
export ARG_MODE
export PKG_VERSION_DIR_NAME
export PKG_INSTALL_PATH
export PKG_SHARE_INFO_INSTALL_PATH
export PKG_INSTALL_INFO_FILE

# 权限校验: 非 root 用户必须和安装目录 owner 一致
_user_auth

# 解锁权限: 卸载前需要解锁已安装目录的写权限 (chmod_end 把 share/info/pypto 设为 550, owner 无写权限)
_unchattr_files
chmod_start

start_log

comm_log "INFO" "uninstall ${ARG_INSTALL_PATH} ${ARG_MODE}"
if bash "${UNINSTALL_SHELL}"; then
    chmod_start
    rm -f "${ASCEND_INSTALL_INFO_FILE}" "${INSTALLED_VERSION_FILE}"
    if [ -f "${ASCEND_INSTALL_INFO_OLD_FILE}" ] && [ -w "${ASCEND_INSTALL_INFO_OLD_FILE}" ] && [ "${num}" != "0" ]; then
        _tmp_file=$(mktemp "${ASCEND_INSTALL_INFO_OLD_FILE}.tmpXXXXXX")
        sed '/pypto_install_path_param=/Id' "${ASCEND_INSTALL_INFO_OLD_FILE}" > "${_tmp_file}" 2>/dev/null
        sed '/pypto_install_type=/Id' "${_tmp_file}" > "${_tmp_file}.2" 2>/dev/null
        cat "${_tmp_file}.2" > "${ASCEND_INSTALL_INFO_OLD_FILE}"
        rm -f "${_tmp_file}" "${_tmp_file}.2"
    fi
    _remove_dir_recursive "${ARG_INSTALL_PATH}" "${PKG_SHARE_INFO_INSTALL_PATH}"
    comm_log "INFO" "PyPTO package uninstalled successfully! Uninstallation takes effect immediately."
else
    comm_log "ERROR" "Uninstall failed"
    exit_log 1
fi

exit_log 0
