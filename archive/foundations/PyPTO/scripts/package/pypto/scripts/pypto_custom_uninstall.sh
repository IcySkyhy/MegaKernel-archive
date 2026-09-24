#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# PyPTO 自定义卸载脚本 (install_common_parser.sh 回调)
#
# 本脚本由 install_common_parser.sh 的 package_custom_uninstall() 机制自动调用:
#   --package="pypto" -> 拼接为 pypto_custom_uninstall.sh -> 在 curpath 目录查找并执行
#
# 变量获取: 通过环境变量获取 (由 run_pypto_uninstall.sh export 继承, 或 install.sh export 继承).

########################################################################################################################
# 环境变量
########################################################################################################################
unset PYTHONPATH    # 避免 PYTHONPATH 环境变量干扰 whl 包的安装路径

########################################################################################################################
# 全局变量
########################################################################################################################
tmp_cfd="$(dirname "$(readlink -f "$0")")"
readonly TMP_PKG_SCRIPTS_PATH="$tmp_cfd"
unset tmp_cfd

########################################################################################################################
# 函数引入
#   在实际运行时, 对应脚本会拷贝到对应位置, 但在源码目录结构中, 对应脚本可能不存在, 故屏蔽对应告警.
########################################################################################################################
# shellcheck disable=SC1091
. "${TMP_PKG_SCRIPTS_PATH}/common_func_v2.inc"
# shellcheck disable=SC1091
. "${TMP_PKG_SCRIPTS_PATH}/common_func.inc"
# shellcheck disable=SC1091
. "${TMP_PKG_SCRIPTS_PATH}/pypto_func.sh"

########################################################################################################################
# 全局初始化
########################################################################################################################

# 日志初始化
comm_init_log
set_comm_log "PyPTO" "${COMM_LOGFILE}"

PKG_WHL_INSTALL_PATH="${PKG_WHL_INSTALL_PATH:-${PKG_INSTALL_PATH}/python/site-packages}"
PYPTO_MODULE_NAME="pypto"

if [ -n "${PKG_ARCH_NAME}" ]; then
    ARCH_NAME="${PKG_ARCH_NAME}"
else
    ARCH_NAME=$(pypto_get_pkg_arch_name)
fi

########################################################################################################################
# 业务函数 (仅 uninstall 专属)
########################################################################################################################

whl_uninstall_package() {
    local _module="$1"
    local _module_path="$2"
    local _module_dist_name="${_module//-/_}"

    if [ "${ARG_PY_LOCAL}" = "y" ]; then
        if [ -d "${_module_path}/${_module_dist_name}" ] || \
            ls "${_module_path}/${_module_dist_name}-"*.dist-info >/dev/null 2>&1; then
            export PYTHONPATH="${_module_path}"
            export PIP_BREAK_SYSTEM_PACKAGES=1
            if ! pypto_run_pip uninstall -y "${_module}" > /dev/null 2>&1; then
                comm_log "WARNING" "uninstall ${_module} failed."
                exit 1
            else
                comm_log "INFO" "${_module} uninstalled successfully!"
            fi
        else
            if ! pypto_run_pip show "${_module}" > /dev/null 2>&1; then
                comm_log "INFO" "${_module} is not installed, skip pip uninstall."
            else
                if ! pypto_run_pip uninstall -y "${_module}" > /dev/null 2>&1; then
                    comm_log "WARNING" "uninstall ${_module} failed."
                    exit 1
                else
                    comm_log "INFO" "${_module} uninstalled successfully!"
                fi
            fi
        fi
    else
        export PIP_BREAK_SYSTEM_PACKAGES=1
        if ! pypto_run_pip show "${_module}" > /dev/null 2>&1; then
            comm_log "INFO" "${_module} is not installed, skip pip uninstall."
        else
            if ! pypto_run_pip uninstall -y "${_module}" > /dev/null 2>&1; then
                comm_log "WARNING" "uninstall ${_module} failed."
                exit 1
            else
                comm_log "INFO" "${_module} uninstalled successfully!"
            fi
        fi
    fi
}

########################################################################################################################
# 主流程
########################################################################################################################

comm_log "INFO" "step into pypto_custom_uninstall.sh ......"

custom_uninstall() {
    if [ ! -d "${PKG_SHARE_INFO_INSTALL_PATH}" ]; then
        comm_log "ERROR" "ERR_NO:0x0001;ERR_DES:pypto directory ${PKG_SHARE_INFO_INSTALL_PATH} is not exist."
        exit 1
    fi

    chmod +w -R "${PKG_WHL_INSTALL_PATH}/pypto" 2> /dev/null
    chmod +w -R "${PKG_WHL_INSTALL_PATH}/pypto-*.dist-info" 2> /dev/null

    comm_log "INFO" "uninstall PyPTO tool begin..."
    whl_uninstall_package "${PYPTO_MODULE_NAME}" "${PKG_WHL_INSTALL_PATH}"

    rm -fr "${PKG_WHL_INSTALL_PATH}/pypto" 2> /dev/null
    rm -fr "${PKG_WHL_INSTALL_PATH}/pypto-*.dist-info" 2> /dev/null

    comm_log "INFO" "PyPTO tool uninstalled successfully!"

    local wheel_dir="${PKG_INSTALL_PATH}/${ARCH_NAME}-linux/lib64"
    if [ -d "${wheel_dir}" ]; then
        chmod u+w "${wheel_dir}" 2> /dev/null
        rm -f "${wheel_dir}/pypto-"*.whl 2> /dev/null
    fi

    _remove_dir_recursive "${PKG_INSTALL_PATH}" "${PKG_WHL_INSTALL_PATH}"

    return 0
}

if ! custom_uninstall; then
    exit 1
fi

exit 0
