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
# run 包子命令入口, 本脚本必须命名为 install.sh
# help.info 中声明的子命令, 均以本文件作为处理入口.
#
# 用户执行 bash cann-pypto*.run 时, makeself 先将其解压到临时目录, 然后执行该入口脚本进行 run 包安装.

########################################################################################################################
# 全局变量
########################################################################################################################

# 用户名, 用户组
user_name=$(id -un)
user_group=$(id -gn)
readonly USER_NAME="${user_name}"
readonly USER_GROUP="${user_group}"
unset user_name
unset user_group
export USER_NAME    # 导出给子进程使用, 避免重复获取
export USER_GROUP   # 导出给子进程使用, 避免重复获取

# 参数默认值 (无参数传入时使用)
# 框架参数
ARG_RUN_FILE_NAME=""    # run 文件名, 如 cann-pypto_9.1.0_linux-aarch64.run
ARG_CWD=""              # 执行路径, 如 ${HOME}/packages/ 代表执行 bash *.run 时所在的目录
ARG_REMAIN_PARAMS=""    # 实际输入的参数
ARG_QUIET=n             # 快速标记, 用于控制日志输出的静默模式
ARG_CHECK=n             # 检查标记

# 自定义参数
ARG_MODE=""             # 模式类型 (run/full/devel/upgrade/uninstall)
ARG_INSTALL_PATH=""     # 安装路径, 代表用户指定的安装顶层目录
ARG_INSTALL_FOR_ALL=n   # 为所有用户安装标记
ARG_PY_LOCAL=n          # PyLocal标记
ARG_DOCKER_ROOT=""      # Docker root路径 (用于容器场景安装, 拼接在安装路径前)

# 包安装路径
PKG_VERSION_DIR_NAME=""         # 包安装路径中 version 级名称(用于区分多版本目录名)
PKG_INSTALL_PATH=""             # 包安装路径(包含多版本层级)
PKG_SHARE_INFO_INSTALL_PATH=""  # 包共享信息安装路径(含多版本层级 + share/info/pypto)
PKG_INSTALL_INFO_FILE=""        # 包安装信息文件, 记录安装类型/用户/路径等元数据, 供后续升级/卸载时读取

# 临时包内路径
#   makeself 会先将 run 包解压到临时目录下
tmp_cfd="$(dirname "$(readlink -f "$0")")"
readonly TMP_PKG_SCRIPTS_PATH="$tmp_cfd"
unset tmp_cfd
readonly TMP_PKG_ROOT_PATH="${TMP_PKG_SCRIPTS_PATH}/.."             # 临时包根目录
readonly TMP_PKG_VERSION_FILE="${TMP_PKG_ROOT_PATH}/version.info"   # 临时包版本信息文件
export TMP_PKG_SCRIPTS_PATH     # 导出给子进程使用, 避免重复获取
export TMP_PKG_ROOT_PATH        # 导出给子进程使用, 避免重复获取
export TMP_PKG_VERSION_FILE     # 导出给子进程使用, 避免重复获取

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

# 多版本包标记
#   支持多版本安装的包是指, 在统一安装路径下可以共存多个版本的包. 通过 version.info 中的 version_dir(如 cann) 作为中间目录层
#   来区分不同的版本. 如 /usr/local/Ascend/cann-9.0.0 /usr/local/Ascend/cann-9.1.0 代表了不同的版本, 但是其统一的安装路径
#   均为 /usr/local/Ascend 路径
tmp_pkg_mulit_version=""
is_multi_version_pkg "tmp_pkg_mulit_version" "$TMP_PKG_VERSION_FILE"
PKG_IS_MULTI_VERSION="${tmp_pkg_mulit_version}"
unset tmp_pkg_mulit_version
export PKG_IS_MULTI_VERSION  # 导出给子进程使用, 避免重复获取

# 包版本
tmp_pkg_verison=""
get_version "tmp_pkg_verison" "$TMP_PKG_VERSION_FILE"
PKG_VERSION="${tmp_pkg_verison}"
unset tmp_pkg_verison
export PKG_VERSION

########################################################################################################################
# 辅助函数定义
########################################################################################################################

# 功能: 设置模式类型
_set_mode() {
    local _check_var="$1"
    local _mode_val="$2"
    local _val
    eval "_val=\"\${${_check_var}}\""
    if [ ! -z "$_val" ]; then
        local msg="ERR_NO:0x0004;ERR_DES:" \
            "Only support one type: '--full', '--run', '--devel', '--upgrade', '--uninstall', operation failed!"
        comm_log "ERROR" "${msg}"
        exit 1
    fi
    eval "${_check_var}=\"True\""
    ARG_MODE="${_mode_val}"
}

_parse_args() {
    # 框架参数解析
    ARG_RUN_FILE_NAME="${1#--}"
    ARG_CWD="${2#--}"
    shift 2
    ARG_REMAIN_PARAMS="$*"

    # 自定义参数解析
    local mode_unique_flag=""
    while true
    do
        case "$1" in
        --help | -h)
            comm_log "INFO" "Please input this command for help: ./${ARG_RUN_FILE_NAME} --help"
            exit 0
            ;;
        --version)
            local version
            if ! get_package_version "version" "${TMP_PKG_VERSION_FILE}" || [ -z "${version}" ]; then
                echo "none"
            else
                echo "${version}"
            fi
            exit 0
            ;;
        --quiet)
            ARG_QUIET=y
            shift
            ;;
        --check)
            ARG_CHECK=y
            shift
            ;;
        --run | --full | --devel | --uninstall | --upgrade)
            local mode
            mode=$(echo "$1" | awk -F"--" '{print $2}')
            _set_mode "mode_unique_flag" "${mode}"
            shift
            ;;
        --install-path=*)
            local install_path
            install_path=$(echo "$1" | cut -d"=" -f2-)
            comm_judgmentpath "${install_path}" "${LOG_PKG_NAME}"
            # 转绝对路径, 并去除行尾的 '/'
            install_path=$(relative_path_to_absolute_path "${install_path}")
            install_path=$(echo "${install_path}" | sed "s/\/*$//g")
            if [ -z "${install_path}" ]; then
                install_path="/"
            fi
            ARG_INSTALL_PATH="${install_path}"
            shift
            ;;
        --install-for-all)
            ARG_INSTALL_FOR_ALL=y
            shift
            ;;
        --pylocal)
            ARG_PY_LOCAL=y
            shift
            ;;
        --docker-root=*)
            local docker_path
            docker_path=$(echo "$1" | cut -d"=" -f2-)
            docker_path=$(echo "${docker_path}" | sed "s/\/*$//g")
            if ! check_docker_path "${docker_path}"; then
                exit_log 1
            fi
            ARG_DOCKER_ROOT="${docker_path}"
            shift
            ;;
        -*)
            comm_log "ERROR" "ERR_NO:0x0004;ERR_DES: Unsupported parameters : $1"
            comm_log "INFO" "Please input this command for help: ./${ARG_RUN_FILE_NAME} --help"
            exit 0
            ;;
        *)
            break
            ;;
        esac
    done

    # 自定义参数组合合法性检查
    # 1. 当未传递任何需要处理的参数时, 直接退出, 由外层框架继续处理. 如只输入了 --extract 命令等情况
    if [ -z "${ARG_REMAIN_PARAMS}" ]; then
        exit 0
    fi
    # 2. 模式存在检测, 至少指定: full/run/devel/upgrade/uninstall/check 中任意一个
    #    注意: check 可单独执行, 也可随 full/run/devel/upgrade/uninstall 一起执行
    if [[ ( -z "${ARG_MODE}" || -z "${mode_unique_flag}" ) && "${ARG_CHECK}" = "n" ]]; then
        comm_log "ERROR" "ERR_NO:0x0004;ERR_DES:One of parameters '--full', '--run', '--devel', '--upgrade', '--uninstall' or '--check' must be used."
        exit 1
    fi
    if [ "$ARG_MODE" = "uninstall" ]; then
        if [ "$ARG_CHECK" = "y" ]; then
            comm_log "ERROR" "ERR_NO:0x0004;ERR_DES:Parameter conflict, '--check' is not supported with '--uninstall'."
            exit 1
        fi
    fi
    # 3. 参数 --install-path 检查
    if [ -n "${ARG_INSTALL_PATH}" ]; then
        local _allow_mode_list=("full" "run" "devel" "upgrade" "uninstall")
        if ! in_array "${ARG_MODE}" "${_allow_mode_list[@]}"; then
            comm_log "ERROR" "ERR_NO:0x0004;ERR_DES:Parameter '--install-path' is not supported to used by this way, please use with '--full', '--run', '--devel', '--upgrade', '--uninstall'."
            exit 1
        fi
        ppath="$(dirname "$ARG_INSTALL_PATH")"
        if [ -n "${ppath}" ] && [ ! -d "${ppath}" ]; then
            comm_log "ERROR" "ERR_NO:0x0004;ERR_DES:Parent path doesn't exist, please create ${ppath} first."
            exit 1
        fi
    fi
    # 4. 参数 --install-for-all 检查
    if [ "${ARG_INSTALL_FOR_ALL}" = "y" ]; then
        local _allow_mode_list=("full" "run" "devel" "upgrade")
        if ! in_array "${ARG_MODE}" "${_allow_mode_list[@]}"; then
            comm_log "ERROR" "ERR_NO:0x0004;ERR_DES:Parameter '--install-for-all' is not supported to used by this way, please use with '--full', '--run', '--devel', '--upgrade'."
            exit 1
        fi
    fi
    # 5. 参数 --pylocal 检查
    if [ "${ARG_PY_LOCAL}" = "y" ]; then
        local _allow_mode_list=("full" "run" "devel" "upgrade")
        if ! in_array "${ARG_MODE}" "${_allow_mode_list[@]}"; then
            comm_log "ERROR" "ERR_NO:0x0004;ERR_DES:Parameter '--pylocal' is not supported to used by this way, please use with '--full', '--run', '--devel', '--upgrade'."
            exit 1
        fi
    fi

    # 参数取值修正
    # 1. 参数 --install-path
    #   1) 默认安装路径填充, 区分 root 用户和普通用户;
    #   2) 最终安装路径处理: 从 version.info 读取 version_dir 作为版本目录名,
    #      若用户传入的路径末尾恰好等于 version_dir, 则剥离版本目录层;
    #      使用 is_version_dirpath 推断版本目录 (检查 share/info 或 ascend_package_db.info)
    if [ -z "${ARG_INSTALL_PATH}" ]; then
        if [ "$(id -u)" = "0" ]; then
            ARG_INSTALL_PATH="/usr/local/Ascend"
        else
            ARG_INSTALL_PATH=$(relative_path_to_absolute_path "${HOME}/Ascend")
        fi
    fi
    if is_version_dirpath "${ARG_INSTALL_PATH}"; then
        PKG_VERSION_DIR_NAME="$(basename "$ARG_INSTALL_PATH")"
        ARG_INSTALL_PATH="$(dirname "$ARG_INSTALL_PATH")"
    else
        PKG_VERSION_DIR_NAME="cann"
    fi
    if [ "$PKG_IS_MULTI_VERSION" = "true" ]; then
        PKG_INSTALL_PATH="${ARG_INSTALL_PATH}/${PKG_VERSION_DIR_NAME}"
        PKG_SHARE_INFO_INSTALL_PATH="${PKG_INSTALL_PATH}/share/info/pypto"
    else
        PKG_INSTALL_PATH="${ARG_INSTALL_PATH}"
        PKG_SHARE_INFO_INSTALL_PATH="${PKG_INSTALL_PATH}/pypto"
    fi
    # docker_root 拼接: 逻辑路径不变, 实际路径拼接 docker_root 前缀
    if [ -n "${ARG_DOCKER_ROOT}" ]; then
        PKG_INSTALL_PATH=$(concat_docker_install_path "${ARG_DOCKER_ROOT}" "${PKG_INSTALL_PATH}")
        PKG_SHARE_INFO_INSTALL_PATH=$(concat_docker_install_path "${ARG_DOCKER_ROOT}" "${PKG_SHARE_INFO_INSTALL_PATH}")
    fi
    PKG_INSTALL_INFO_FILE="${PKG_SHARE_INFO_INSTALL_PATH}/ascend_install.info"
    # 2. 参数 --install-for-all
    #   1) Linux 安全规范要求 root 安装的目录应当对所有用户可读可执行(755 权限), 而非仅 Owner/Group 可访问(750).
    #   2) CANN 体系中 run 包对应 install-for-all 参数的语义就是 "安装后的目录权限设置为 755, 而非 750".
    #      故此处检查如果是 root 用户安装软件包, 强制设置 ARG_INSTALL_FOR_ALL=y
    if [ "$(id -u)" = "0" ]; then
        ARG_INSTALL_FOR_ALL=y
    fi
}

_process() {
    local installed_version
    get_package_version "installed_version" "${PKG_SHARE_INFO_INSTALL_PATH}/version.info"

    if [ -n "${installed_version}" ] && [ "${installed_version}" != "none" ] || [ -f "${PKG_INSTALL_INFO_FILE}" ]; then
        # === 已安装场景 ===
        case "${ARG_MODE}" in
        uninstall)
            _unchattr_files
            _do_uninstall "y" "y"
            _save_user_files_to_log "${PKG_SHARE_INFO_INSTALL_PATH}"
            exit_log 0
            ;;
        upgrade)
            _unchattr_files
            _do_uninstall "n" "n"
            _save_user_files_to_log "${PKG_SHARE_INFO_INSTALL_PATH}"
            _do_install
            exit_log 0
            ;;
        full|run|devel)
            if [ "${ARG_QUIET}" = "n" ]; then
                _confirm_action "PyPTO package has been installed in ${PKG_SHARE_INFO_INSTALL_PATH}, version is ${installed_version}. Do you want to continue?"
            fi
            _unchattr_files
            _do_uninstall "n" "n"
            _do_install
            exit_log 0
            ;;
        esac
    else
        # === 未安装场景 ===
        case "${ARG_MODE}" in
        uninstall|upgrade)
            comm_log "ERROR" "PyPTO package is not installed in ${PKG_SHARE_INFO_INSTALL_PATH}, ${ARG_MODE} failed!"
            exit_log 1
            ;;
        full|run|devel)
            _pkg_install_path_create
            _do_install
            exit_log 0
            ;;
        esac
    fi
}

########################################################################################################################
# 业务处理流程
########################################################################################################################

# 解析入参, 导出参数
_parse_args "$@"
export ARG_CWD
export ARG_QUIET
export ARG_MODE
export ARG_INSTALL_PATH
export ARG_INSTALL_FOR_ALL
export ARG_PY_LOCAL
export ARG_DOCKER_ROOT
export PKG_VERSION_DIR_NAME
export PKG_INSTALL_PATH
export PKG_SHARE_INFO_INSTALL_PATH
export PKG_INSTALL_INFO_FILE

# 包架构
PKG_ARCH_NAME=$(pypto_get_pkg_arch_name) || { comm_log "ERROR" "Failed to get package arch name from scene.info."; exit_log 1; }
if [ -z "${PKG_ARCH_NAME}" ]; then
    comm_log "ERROR" "Package arch name is empty, scene.info may be corrupted."
    exit_log 1
fi
export PKG_ARCH_NAME

# whl 安装路径
PKG_WHL_INSTALL_PATH="${PKG_INSTALL_PATH}/python/site-packages"
export PKG_WHL_INSTALL_PATH

# 预检查
if [ "${ARG_CHECK}" = "y" ]; then
    comm_log "INFO" "Archive integrity check passed."
    _check strict
    # --check 模式: 版本检查通过即可结束 (无 ARG_MODE 时), 或继续后续安装流程
    if [ -z "${ARG_MODE}" ]; then
        exit_log 0
    fi
else
    # 无 mode (如 --extract 场景), 不走检查, 直接退出
    if [ -z "${ARG_MODE}" ]; then
        exit_log 0
    fi
    _check interactive
fi

# 流程处理
start_log
_process

# 退出
exit 0
