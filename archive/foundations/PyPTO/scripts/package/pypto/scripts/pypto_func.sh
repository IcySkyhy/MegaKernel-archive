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
# 常用函数

########################################################################################################################
# 日志管理
########################################################################################################################
pypto_comm_log_init() {
    comm_init_log
    set_comm_log "PyPTO" "${COMM_LOGFILE}"  # COMM_LOGFILE 在 common_func.inc 内定义
}

output_progress() {
    comm_log "INFO" "${LOG_PKG_NAME} install upgradePercentage:${1}%"
}

start_log() {
    local cur_date
    cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    comm_log "INFO" "Start time:${cur_date}"
    comm_log "INFO" "LogFile:${COMM_LOGFILE}"
    comm_log "INFO" "InputParams:${ARG_REMAIN_PARAMS}"
    comm_log "INFO" "OperationLogFile:${COMM_OPERATION_LOGFILE}"
}

exit_log() {
    local cur_date
    cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    comm_log "INFO" "End time:${cur_date}"
    exit "$1"
}

########################################################################################################################
# 权限管理
########################################################################################################################

check_chmod_length() {
    local mod_num="$1"
    local new_mod_num=""
    local mod_num_length="${#mod_num}"
    if [ "$mod_num_length" -eq 3 ]; then
        new_mod_num="$mod_num"
        echo "$new_mod_num"
    elif [ "$mod_num_length" -eq 4 ]; then
        new_mod_num="${mod_num:1:3}"
        echo "$new_mod_num"
    fi
}

# 功能: install-for-all 时将权限从 750 转为 755 (other 位加读执行)
#   输入: permission (如 750, 550, 640)
#   输出: 转换后的 permission (如 755, 555, 645)
#   逻辑: 取前 2 位 + 取第 2 位重复作为第 3 位
set_file_chmod() {
    local permission="${1}"
    local new_permission=""
    if [ "${ARG_INSTALL_FOR_ALL}" = "y" ]; then
        new_permission="${permission:0:2}${permission:1:1}"
        echo "$new_permission"
    else
        echo "$permission"
    fi
}

# 功能: 递归 chmod 目录或文件
chmod_recur() {
    local file_path="${1}"
    local permission="${2}"
    local type="${3}"
    permission=$(set_file_chmod "$permission")
    if [ "$type" = "dir" ]; then
        find "$file_path" -type d -exec chmod "$permission" {} \; 2> /dev/null
    elif [ "$type" = "file" ]; then
        find "$file_path" -type f -exec chmod "$permission" {} \; 2> /dev/null
    fi
}

# 功能: 单项 chmod
chmod_single_dir() {
    local file_path="${1}"
    local permission="${2}"
    local type="${3}"
    permission=$(set_file_chmod "$permission")
    if [ "$type" = "dir" ]; then
        chmod "$permission" "$file_path"
    elif [ "$type" = "file" ]; then
        chmod "$permission" "$file_path"
    fi
}

# 功能: 创建包安装路径
_pkg_install_path_create() {
    local _perm
    local _own

    if [ "$(id -u)" = "0" ]; then
        _perm="755"
        _own="root:root"
    else
        _perm="750"
        _own="${USER_NAME}:${USER_GROUP}"
    fi

    comm_create_dir "${PKG_SHARE_INFO_INSTALL_PATH}" "${_perm}" "${_own}" "${ARG_INSTALL_FOR_ALL}"
    if [ -n "${PKG_VERSION_DIR_NAME}" ]; then
        comm_create_dir "$(dirname "${PKG_SHARE_INFO_INSTALL_PATH}")" "${_perm}" "${_own}" "${ARG_INSTALL_FOR_ALL}"
    fi
}

# 功能: 检查包安装路径是否满足 install-for-all 的要求
_pkg_install_path_check_install_for_all() {
    local mod_num=""
    local other_mod_num=""
    if [ "$ARG_INSTALL_FOR_ALL" = "y" ] && [ -d "$PKG_SHARE_INFO_INSTALL_PATH" ]; then
        mod_num="$(stat -c %a "${PKG_SHARE_INFO_INSTALL_PATH}")"
        mod_num="$(check_chmod_length "$mod_num")"
        other_mod_num="${mod_num:2:1}"
        if [ "${other_mod_num}" -ne 5 ] && [ "${other_mod_num}" -ne 7 ]; then
            comm_log "ERROR" "${PKG_SHARE_INFO_INSTALL_PATH} permission is ${mod_num}, this permission does not support install_for_all param."
            exit_log 1
        fi
    fi
}

# 功能: 卸载时用户权限校验
#   逻辑: root 用户可以卸载任何包; 非 root 用户必须和安装目录 owner 一致才能卸载
_user_auth() {
    if [ ! -d "${PKG_SHARE_INFO_INSTALL_PATH}" ]; then
        return 0
    fi
    local dir_user_id
    dir_user_id=$(stat -c "%u" "${PKG_SHARE_INFO_INSTALL_PATH}")
    local run_user_id
    run_user_id=$(id -u)
    if [ "${run_user_id}" -ne 0 ]; then
        if [ "${run_user_id}" -ne "${dir_user_id}" ]; then
            comm_log "ERROR" "ERR_NO:0x0093;ERR_DES:Permission denied, current user is not supported to uninstall the PyPTO package"
            exit_log 1
        fi
    fi
}

# 功能: 安装前解锁权限 (递归设 750)
#   参数: 可选路径, 默认为 PKG_SHARE_INFO_INSTALL_PATH
chmod_start() {
    local tmpdir="${1:-$PKG_SHARE_INFO_INSTALL_PATH}"
    chmod_recur "$tmpdir" 750 dir 2> /dev/null
}

# 功能: 安装后精细化锁定权限
chmod_end() {
    chmod_recur "${PKG_SHARE_INFO_INSTALL_PATH}/script" 550 dir 2> /dev/null
    chmod_recur "${PKG_SHARE_INFO_INSTALL_PATH}/script" 550 file 2> /dev/null
    chmod_single_dir "${PKG_SHARE_INFO_INSTALL_PATH}/script/install.sh" 500 file 2> /dev/null
    chmod_single_dir "${PKG_SHARE_INFO_INSTALL_PATH}/ascend_install.info" 640 file 2> /dev/null
    chmod_single_dir "${PKG_SHARE_INFO_INSTALL_PATH}/version.info" 440 file 2> /dev/null
    chmod_single_dir "${PKG_SHARE_INFO_INSTALL_PATH}/scene.info" 640 file 2> /dev/null
    chmod_single_dir "${PKG_SHARE_INFO_INSTALL_PATH}" 550 dir 2> /dev/null
    if [ "$(id -u)" = "0" ]; then
        chown "root:root" "${PKG_SHARE_INFO_INSTALL_PATH}" 2> /dev/null
        chmod 755 "${PKG_SHARE_INFO_INSTALL_PATH}" 2> /dev/null
        chown -R "root:root" "${PKG_SHARE_INFO_INSTALL_PATH}/script" 2> /dev/null
    fi
}

########################################################################################################################
# 安装信息管理
########################################################################################################################

# 功能: 更新/追加 install_info 文件中的键值对
#   参数: key, value, file
_update_install_param() {
    local _key="$1"
    local _val="$2"
    local _file="$3"
    if [ ! -f "${_file}" ]; then
        return 1
    fi
    local _param
    _param=$(grep -i "${_key}=" "${_file}")
    if [ "${_param}" = "" ]; then
        echo "${_key}=${_val}" >> "${_file}"
    else
        sed -i "/^${_key}=/Ic ${_key}=${_val}" "${_file}" 2> /dev/null
    fi
}

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

# 功能: 创建 install_info 元数据文件并写入安装信息
_update_install_info() {
    chmod_start
    if [ ! -f "${PKG_INSTALL_INFO_FILE}" ]; then
        comm_create_file "${PKG_INSTALL_INFO_FILE}" 640 "${USER_NAME}:${USER_GROUP}" "${ARG_INSTALL_FOR_ALL}"
    fi
    _update_install_param "PyPTO_Install_Type" "${ARG_MODE}" "${PKG_INSTALL_INFO_FILE}"
    _update_install_param "PyPTO_UserName" "${USER_NAME}" "${PKG_INSTALL_INFO_FILE}"
    _update_install_param "PyPTO_UserGroup" "${USER_GROUP}" "${PKG_INSTALL_INFO_FILE}"
    _update_install_param "PyPTO_Install_Path_Param" "${ARG_INSTALL_PATH}" "${PKG_INSTALL_INFO_FILE}"
    _update_install_param "PyPTO_Install_For_All" "${ARG_INSTALL_FOR_ALL}" "${PKG_INSTALL_INFO_FILE}"
    _update_install_param "PyPTO_PyLocal" "${ARG_PY_LOCAL}" "${PKG_INSTALL_INFO_FILE}"
}

# 功能: 复制包内 version.info 到安装路径
_update_version_info() {
    if [ -f "${PKG_SHARE_INFO_INSTALL_PATH}/version.info" ]; then
        rm -f "${PKG_SHARE_INFO_INSTALL_PATH}/version.info"
        cp -f "${TMP_PKG_VERSION_FILE}" "${PKG_SHARE_INFO_INSTALL_PATH}"
        comm_log "INFO" "Upgrade base version successfully!"
    else
        cp -f "${TMP_PKG_VERSION_FILE}" "${PKG_SHARE_INFO_INSTALL_PATH}"
        comm_log "INFO" "Base version set successfully!"
    fi
    chmod_single_dir "${PKG_SHARE_INFO_INSTALL_PATH}/version.info" 440 file 2> /dev/null
}

########################################################################################################################
# Docker root 路径处理
########################################################################################################################

# 功能: 拼接 docker_root + install_path
#   逻辑: 去除 docker_path 行尾 '/', 空则置 '/', 返回 docker_path + install_path
concat_docker_install_path() {
    local docker_path="$1"
    local install_path="$2"
    docker_path=$(echo "${docker_path}" | sed "s/\/*$//g")
    if [ -z "${docker_path}" ]; then
        docker_path="/"
    fi
    echo "${docker_path}${install_path}"
}

# 功能: 校验 --docker-root 参数
#   逻辑: 必须是绝对路径 (以 '/' 开头), 且目录必须存在
check_docker_path() {
    local docker_path="$1"
    if [ "${docker_path:0:1}" != "/" ]; then
        comm_log "ERROR" "ERR_NO:0x0002;ERR_DES:Parameter --docker-root must be absolute path starting with /."
        return 1
    fi
    if [ ! -d "${docker_path}" ]; then
        comm_log "ERROR" "ERR_NO:0x0002;ERR_DES:The directory:${docker_path} not exist, please create this directory."
        return 1
    fi
    return 0
}

########################################################################################################################
# Driver / Firmware 存在性检查
########################################################################################################################

# 功能: 安装/升级前检查 driver 和 firmware 是否已安装
#   逻辑: 从 /etc/ascend_install.info 检查 Driver/Firmware 记录;
#         缺失时 WARNING + _confirm_action (非静默) 或 WARNING + continue (静默)
#   注意: docker 场景下不检查 firmware (driver 在宿主机, firmware 可能不在容器内)
_driver_firmware_check() {
    local install_info_old="/etc/ascend_install.info"
    local confirm=n
    if [ ! -f "${install_info_old}" ]; then
        comm_log "WARNING" "driver and firmware is not exists, please install first."
        confirm=y
    elif [ "$(grep -c -i "Driver" "${install_info_old}")" -eq 0 ]; then
        comm_log "WARNING" "driver is not exists, please install first."
        confirm=y
    elif [ "$(grep -c -i "Firmware" "${install_info_old}")" -eq 0 ]; then
        if [ "${ARG_DOCKER_ROOT}" != "" ]; then
            comm_log "INFO" "firmware is not exists in docker scene, skip check."
        else
            comm_log "WARNING" "firmware is not exists, please install first."
            confirm=y
        fi
    fi
    if [ "${confirm}" = "y" ]; then
        if [ "${ARG_QUIET}" = "n" ]; then
            _confirm_action "Driver/firmware is not installed, do you want to continue?"
        else
            comm_log "WARNING" "Driver/firmware is not installed, continue in quiet mode."
        fi
    fi
}

########################################################################################################################
# 路径安全检查
########################################################################################################################

# 功能: root 用户安装时递归检查父目录权限
#   逻辑: 递归检查安装路径的每一层父目录是否属于 root 且权限 >= 755
#   返回: 0=通过, 1=owner非root, 2=权限过低, 3=权限过高(非静默时)
_parent_dirs_permission_check() {
    local current_dir="$1"
    local parent_dir
    local short_install_dir
    local owner
    local mod_num

    parent_dir="$(dirname "${current_dir}")"
    short_install_dir="$(basename "${current_dir}")"

    if [ "${current_dir}" = "/" ]; then
        comm_log "INFO" "parent_dirs_permission_check succeeded"
        return 0
    fi

    if [ ! -d "${current_dir}" ]; then
        _parent_dirs_permission_check "${parent_dir}"
        return $?
    fi

    owner="$(stat -c %U "${current_dir}")"
    if [ "${owner}" != "root" ]; then
        comm_log "WARNING" "[${short_install_dir}] permission isn't right, it should belong to root."
        return 1
    fi
    comm_log "INFO" "[${short_install_dir}] belongs to root."

    mod_num="$(stat -c %a "${current_dir}")"
    mod_num="$(check_chmod_length "$mod_num")"
    if [ "${mod_num}" -lt 755 ]; then
        comm_log "WARNING" "[${short_install_dir}] permission is too small, it is recommended that the permission be 755 for the root user."
        return 2
    elif [ "${mod_num}" -eq 755 ]; then
        comm_log "INFO" "[${short_install_dir}] permission is ok."
    else
        comm_log "WARNING" "[${short_install_dir}] permission is too high, it is recommended that the permission be 755 for the root user."
        [ "${ARG_QUIET}" = "n" ] && return 3
    fi

    _parent_dirs_permission_check "${parent_dir}"
}

########################################################################################################################
# 预检查汇总
########################################################################################################################

# 功能: driver 版本兼容性检查
#   逻辑: 从 /etc/ascend_install.info 读 Driver_Install_Path_Param 定位 driver version.info,
#         调用 ver_check.sh 做版本范围比较.
#   参数: check_mode — "strict" (--check 模式, 失败直接阻断) 或 "interactive" (安装模式, 失败交互确认)
_ver_check() {
    local check_mode="${1:-interactive}"
    local dep_info_file="/etc/ascend_install.info"
    local driver_install_path
    driver_install_path="$(grep -iw Driver_Install_Path_Param "${dep_info_file}" 2>/dev/null | cut -d"=" -f2-)"
    if [ -z "${driver_install_path}" ]; then
        comm_log "WARNING" "Cannot find the install path of driver."
        return 0
    fi
    local driver_ver_file="${driver_install_path}/driver/version.info"
    local version_info_file="${TMP_PKG_VERSION_FILE}"
    local ret
    bash "${TMP_PKG_SCRIPTS_PATH}/ver_check.sh" "${version_info_file}" "${driver_ver_file}"
    ret=$?
    if [ "${ret}" -eq 1 ]; then
        if [ "${check_mode}" = "strict" ]; then
            return 1
        elif [ "${ARG_QUIET}" = "n" ]; then
            _confirm_action "Check version does not matched, do you want to continue?"
        else
            comm_log "WARNING" "Check version does not matched!"
        fi
    elif [ "${ret}" -eq 0 ]; then
        comm_log "INFO" "Check version matched!"
    fi
    return 0
}

# 功能: 安装前预检查汇总
#   流程 (strict 模式): 仅 driver 版本兼容性检查
#   流程 (interactive 模式): driver 版本检查 → parent_dirs 权限检查 → driver/firmware 存在性检查
#   参数: check_mode — "strict" (--check 模式) 或 "interactive" (安装模式)
_check() {
    local check_mode="${1:-interactive}"

    # 1. driver 版本兼容性检查
    if ! _ver_check "${check_mode}"; then
        exit_log 1
    fi

    # strict 模式 (--check): 仅做版本检查, 到此结束
    if [ "${check_mode}" = "strict" ]; then
        return 0
    fi

    # 2. root 用户安装时递归检查父目录权限
    #   docker_root 场景下跳过: 容器路径结构由宿主机 root 创建, 不适用本机权限检查
    if [ "$(id -u)" = "0" ] && [ -z "${ARG_DOCKER_ROOT}" ]; then
        local ret=0
        _parent_dirs_permission_check "${PKG_INSTALL_PATH}" || ret=$?
        if [ "${ARG_QUIET}" = "y" ] && [ "${ret}" -ne 0 ]; then
            comm_log "ERROR" "the given dir, or its parents, permission is invalid."
            exit 1
        fi
        if [ "${ret}" -ne 0 ]; then
            _confirm_action "You are going to put run-files on an insecure install-path, do you want to continue?"
        fi
    fi

    # 3. driver / firmware 存在性检查
    #   非 install-for-all 场景下, 安装前检查 driver/firmware 是否已安装
    if [ "${ARG_INSTALL_FOR_ALL}" = "n" ]; then
        if [ "${ARG_MODE}" = "run" ] || [ "${ARG_MODE}" = "full" ] || [ "${ARG_MODE}" = "devel" ]; then
            _driver_firmware_check
        fi
    fi
}

########################################################################################################################
# 安装/卸载/升级执行框架
########################################################################################################################

# 功能: 解锁 immutable 标记 (chattr -i), 防御性占位
_unchattr_files() {
    if [ -f "${PKG_INSTALL_INFO_FILE}" ]; then
        if [ -d "${PKG_SHARE_INFO_INSTALL_PATH}" ]; then
            if ! chattr -R -i "${PKG_SHARE_INFO_INSTALL_PATH}" >/dev/null 2>&1; then
                find "${PKG_SHARE_INFO_INSTALL_PATH}" -exec chattr -i {} + >/dev/null 2>&1
            fi
        fi
    fi
}

# 功能: 安装执行框架
#   流程: update_install_path -> update_install_info -> 实际文件拷贝(占位) -> update_version_info -> chmod_end
_do_install() {
    _pkg_install_path_check_install_for_all
    _update_install_info
    comm_log "INFO" "install ${ARG_INSTALL_PATH} ${ARG_MODE}"

    # 文件拷贝委托给子进程 run_pypto_install.sh
    #   三层分工: 本函数管框架, 子进程管拷贝+后处理, install_common_parser.sh 管通用引擎
    #   参数: 所依赖的参数均已通过 环境变量导出的方式向子进程传递
    if bash "${TMP_PKG_SCRIPTS_PATH}/run_pypto_install.sh"; then
        _update_version_info
        comm_log "INFO" "PyPTO package installed successfully! The new version takes effect immediately."
        chmod_end
    else
        chmod_end
        comm_log "ERROR" "PyPTO package install failed, please retry after uninstall!"
        exit_log 1
    fi
}

# 功能: 卸载执行框架
#   参数: remove_info_files ("y"/"n"), remove_version_info ("y"/"n")
#   流程: user_auth -> unchattr_files -> chmod_start -> 实际文件删除(占位) -> 条件删除 info/version -> 递归清理空目录
_do_uninstall() {
    local remove_info_files="$1"
    local remove_version_info="$2"
    _user_auth
    _unchattr_files
    chmod_start
    comm_log "INFO" "uninstall ${ARG_INSTALL_PATH} ${ARG_MODE}"
    if bash "${PKG_SHARE_INFO_INSTALL_PATH}/script/run_pypto_uninstall.sh"; then
        if [[ "$remove_info_files" = "y" ]]; then
            [[ -f "${PKG_INSTALL_INFO_FILE}" ]] && rm -f "${PKG_INSTALL_INFO_FILE}"
        fi
        if [[ "$remove_version_info" = "y" ]]; then
            [[ -f "${PKG_SHARE_INFO_INSTALL_PATH}/version.info" ]] && rm -f "${PKG_SHARE_INFO_INSTALL_PATH}/version.info"
        fi
        _remove_dir_recursive "${ARG_INSTALL_PATH}" "${PKG_SHARE_INFO_INSTALL_PATH}"
        comm_log "INFO" "PyPTO package uninstalled successfully! Uninstallation takes effect immediately."
    else
        comm_log "ERROR" "PyPTO package uninstall failed!"
        exit_log 1
    fi
}

########################################################################################################################
# 辅助工具
########################################################################################################################

# 功能: 获取目录权限
_get_dir_mod() {
    local path="$1"
    stat -c %a "$path"
}

# 功能: 递归删除空父目录 (从 dir_end 向上回溯到 dir_start)
_remove_dir_recursive() {
    local dir_start="$1"
    local dir_end="$2"

    if [[ "$dir_end" == "$dir_start" ]]; then
        return 0
    fi
    if [[ ! -e "$dir_end" ]]; then
        return 0
    fi

    # 替换 ls -A 判断空目录（规避 ls 解析告警)
    shopt -s dotglob nullglob
    local files=("$dir_end"/*)
    shopt -u dotglob nullglob
    if [[ ${#files[@]} -gt 0 ]]; then
        return 0
    fi

    local up_dir oldmod
    up_dir="$(dirname "$dir_end")"
    oldmod="$(_get_dir_mod "$up_dir")"
    chmod u+w "$up_dir"

    # 直接在 if 内执行 rm -rf，不再单独拿 $?
    if ! rm -rf "$dir_end"; then
        chmod "$oldmod" "$up_dir"
        return 1
    fi

    chmod "$oldmod" "$up_dir"
    _remove_dir_recursive "$dir_start" "$up_dir"
}

# 功能: 交互确认 (y/n)
_confirm_action() {
    local prompt_msg="$1"
    while true; do
        comm_log "INFO" "${prompt_msg} [y/n]"
        read -r yn
        case "$yn" in
            n|N) comm_log "INFO" "Operation cancelled."; exit_log 0 ;;
            y|Y) break ;;
            *)   comm_log "ERROR" "ERR_NO:0x0002;ERR_DES: Invalid input, please enter y or n." ;;
        esac
    done
}

# 功能: 保存残留用户文件列表到日志 (卸载后检查)
_save_user_files_to_log() {
    local target_dir="$1"
    if [ "$target_dir" = "${PKG_SHARE_INFO_INSTALL_PATH}" ] && [ -s "$target_dir" ]; then
        local filenum
        local dirnum
        local totalnum
        filenum=$(find "$target_dir" -type f 2>/dev/null | wc -l)
        dirnum=$(find "$target_dir" -type d 2>/dev/null | grep -c .)
        totalnum=$(( filenum + dirnum ))
        if [ "$totalnum" -eq 2 ]; then
            if [ -f "${PKG_INSTALL_INFO_FILE}" ] && [ -f "${PKG_SHARE_INFO_INSTALL_PATH}/version.info" ]; then
                return 0
            fi
        fi
        if [ "$totalnum" -eq 1 ]; then
            if [ -f "${PKG_INSTALL_INFO_FILE}" ] || [ -f "${PKG_SHARE_INFO_INSTALL_PATH}/version.info" ]; then
                return 0
            fi
        fi
        comm_log "INFO" "Some files generated by user are not cleared, if necessary, manually clear them, get details in ${COMM_LOGFILE}"
    fi
    if [[ -s "$target_dir" ]]; then
        shopt -s dotglob nullglob
        for file in "$target_dir"/*; do
            # 提取纯文件名
            local base="${file##*/}"
            if [[ -d "$file" && ! -L "$file" ]]; then
                if [[ "$base" != "." && "$base" != ".." ]]; then
                    echo "$file" >> "${COMM_LOGFILE}"
                    _save_user_files_to_log "$file"
                fi
            else
                echo "$file" >> "${COMM_LOGFILE}"
            fi
        done
        shopt -u dotglob nullglob
    fi
}

# 功能: 判断给定字符串是否在给定列表中
in_array() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

# 功能: 相对路径转绝对路径
relative_path_to_absolute_path() {
    local relative_path_="${1}"
    if [ "$relative_path_" = "" ]; then
        return
    fi
    local fstr="${relative_path_:0:1}"
    if [ "$fstr" = "~" ]; then
        relative_path_="${HOME}$(echo "${relative_path_}" | cut -d'~' -f 2-)"
    elif [ "$fstr" != "/" ]; then
        relative_path_="${relative_path_#./}"
        relative_path_="${ARG_CWD}/${relative_path_}"
    fi
    echo "$relative_path_"
}

########################################################################################################################
# Python / pip 环境
########################################################################################################################

pypto_has_python_installer() {
    if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
        return 0
    fi
    command -v pip3 >/dev/null 2>&1
}

pypto_run_pip() {
    if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip "$@"
    else
        pip3 "$@"
    fi
}

pypto_get_python_info() {
    local _pip_info
    local _python_pip_info
    local _python_info
    if command -v python3 >/dev/null 2>&1; then
        _python_pip_info=$(python3 -m pip --version 2>/dev/null)
    fi
    _pip_info=$(pip3 --version 2>/dev/null)
    if command -v python3 >/dev/null 2>&1; then
        _python_info=$(python3 -c 'import sys; print("%s %d.%d.%d" % (sys.executable, sys.version_info[0], sys.version_info[1], sys.version_info[2]))' 2>/dev/null)
    fi
    if [ -n "${_python_info}" ]; then
        echo "python3 ${_python_info}; python3 -m pip ${_python_pip_info}; pip3 ${_pip_info}"
    else
        echo "pip3 ${_pip_info}"
    fi
}

########################################################################################################################
# 架构名获取
########################################################################################################################

pypto_get_pkg_arch_name() {
    local scene_info="${PKG_SHARE_INFO_INSTALL_PATH}/scene.info"
    if [ ! -f "$scene_info" ]; then
        scene_info="${PKG_INSTALL_PATH}/share/info/pypto/scene.info"
    fi
    if [ ! -f "$scene_info" ]; then
        scene_info="${TMP_PKG_SCRIPTS_PATH}/../scene.info"
    fi
    if [ ! -f "$scene_info" ]; then
        comm_log "ERROR" "scene.info file cannot be found!"
        return 1
    fi
    local arch
    arch="$(grep -iw arch "$scene_info" | cut -d"=" -f2- | awk '{print tolower($0)}')"
    if [ -z "$arch" ]; then
        comm_log "ERROR" "var arch cannot be found in scene.info!"
        return 1
    fi
    echo "$arch"
}
