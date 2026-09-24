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
# PyPTO 自定义安装脚本 (install_common_parser.sh 回调)
#
# 本脚本由 install_common_parser.sh 的 package_custom_install() 机制自动调用:
#   --package="pypto" -> 拼接为 pypto_custom_install.sh -> 在 临时包解压后的脚本 目录查找并执行
#
# 执行时机在 install_common_parser.sh 流程中位于文件拷贝之后, do_chmod_file_dir() 之前,
# 确保 pip install 生成的文件能被后续 do_chmod_file_dir() 正确设置权限.
#
# 关键路径层级 (由 install.sh 导出, 通过 custom_options 传入):
#   ARG_INSTALL_PATH           — 安装顶层目录 (如 /usr/local/Ascend)
#   PKG_INSTALL_PATH           — CANN 根目录, 含 version_dir (如 /usr/local/Ascend/cann)
#   PKG_SHARE_INFO_INSTALL_PATH — 包安装目录 (如 /usr/local/Ascend/cann/share/info/pypto)
#   PKG_WHL_INSTALL_PATH       — pip install --target 目标目录 (如 /usr/local/Ascend/cann/python/site-packages)
#   TMP_PKG_ROOT_PATH          — makeself 临时解压根目录, whl 文件从此目录取

########################################################################################################################
# 环境变量
########################################################################################################################
unset PYTHONPATH    # 避免 PYTHONPATH 环境变量干扰 whl 包的安装路径

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
pypto_comm_log_init

########################################################################################################################
# 辅助函数定义
########################################################################################################################

pypto_install_package() {
    local _package="$1"
    local _pythonlocalpath="$2"
    comm_log "INFO" "install python module package in ${_package}"
    comm_log "INFO" "python package installer: $(pypto_get_python_info)"
    if ! pypto_has_python_installer; then
        comm_log "ERROR" "install ${_package} failed, python3 -m pip or pip3 is not installed."
        exit 1
    fi
    if [ -f "$_package" ]; then
        export PIP_BREAK_SYSTEM_PACKAGES=1
        if [ "$ARG_PY_LOCAL" = "y" ]; then
            if ! pypto_run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" -t "${_pythonlocalpath}" 1> /dev/null; then
                comm_log "WARNING" "install ${_package} failed."
                exit 1
            fi
        else
            if [ "$(id -u)" -ne 0 ]; then
                if ! pypto_run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" --user 1> /dev/null; then
                    comm_log "WARNING" "install ${_package} failed."
                    exit 1
                fi
            else
                if ! pypto_run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package}" 1> /dev/null; then
                    comm_log "WARNING" "install ${_package} failed."
                    exit 1
                fi
            fi
        fi
        comm_log "INFO" "install ${_package} successfully!"
    else
        comm_log "ERROR" "ERR_NO:0x0080;ERR_DES:Install ${_package} failed, can not find the matched package for this platform."
        exit 1
    fi
}


custom_install() {
    if [ ! -d "${PKG_SHARE_INFO_INSTALL_PATH}" ]; then
        comm_log "ERROR" "ERR_NO:0x0001;ERR_DES:${LOG_PKG_NAME} directory ${PKG_SHARE_INFO_INSTALL_PATH} is not exist."
        exit 1
    fi

    comm_log "INFO" "install ${LOG_PKG_NAME} extension module begin..."

    local pypto_whl_path="${PKG_INSTALL_PATH}/${PKG_ARCH_NAME}-linux/lib64/pypto-0.2.1-cp37-abi3-manylinux2014_${PKG_ARCH_NAME}.whl"
    if [ ! -f "${pypto_whl_path}" ]; then
        comm_log "ERROR" "ERR_NO:0x0080;ERR_DES:can not find ${LOG_PKG_NAME} whl package in ${PKG_INSTALL_PATH}/${PKG_ARCH_NAME}-linux/lib64/, pypto_whl_path=${pypto_whl_path}"
        exit 1
    fi

    pypto_install_package "${pypto_whl_path}" "${PKG_WHL_INSTALL_PATH}"

    comm_log "INFO" "The ${LOG_PKG_NAME} extension module installed successfully!"

    # 预编译 pypto_impl.so (静默, 失败不影响安装, 留给运行时缓存回退)
    # 直接 import pypto 端到端触发在线编译, 用 exit code 判断结果
    # pylocal 模式下包安装在自定义路径, 需临时设置 PYTHONPATH 使子进程能找到 pypto
    if [ "${ARG_PY_LOCAL}" = "y" ]; then
        if PYTHONPATH="${PKG_WHL_INSTALL_PATH}" python3 -c "import pypto" 2>/dev/null; then
            comm_log "INFO" "pypto_impl.so pre-compiled successfully."
        else
            comm_log "INFO" "pypto_impl.so pre-compilation skipped, will compile on first import."
        fi
    else
        if python3 -c "import pypto" 2>/dev/null; then
            comm_log "INFO" "pypto_impl.so pre-compiled successfully."
        else
            comm_log "INFO" "pypto_impl.so pre-compilation skipped, will compile on first import."
        fi
    fi

    if [ "${ARG_PY_LOCAL}" = "y" ]; then
        comm_log "INFO" "please make sure PYTHONPATH include ${PKG_WHL_INSTALL_PATH}."
    else
        comm_log "INFO" "The package ${LOG_PKG_NAME} is already installed in python default path. It is recommended to install it using the '--pylocal' parameter, install the package ${LOG_PKG_NAME} in the ${PKG_WHL_INSTALL_PATH}."
    fi

    return 0
}

########################################################################################################################
# 业务处理流程
########################################################################################################################

# 执行安装
comm_log "INFO" "step into pypto_custom_install.sh ......"
if ! custom_install; then
    exit 1
fi

exit 0
