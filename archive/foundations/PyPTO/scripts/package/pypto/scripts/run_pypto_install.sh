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
# PyPTO 安装子脚本
#
# 本脚本是 _do_install() 委托的同步子进程, 负责实际文件拷贝与包特有后处理. 出于以下原因以子进程方式执行实际操作:
#
#   1. 子进程隔离: install_common_parser.sh 内部失败时可能直接 exit,
#      子进程隔离保证父进程 (_do_install) 仍可优雅处理错误 (chmod_end + 错误日志), 而非被 exit 直接终止整个 install.sh
#
#   2. 参数组装: install_common_parser.sh 需大量动态参数
#      (--copy_all --package --install --username --usergroup --version --version-dir
#      --chip --feature --custom-options 等), 本脚本负责根据当前环境组装这些参数
#
#   3. 包特有后处理: install_common_parser.sh 只做通用文件拷贝,
#      PyPTO 的后处理 (pip install whl 等) 由本脚本在拷贝完成后执行
#
#   4. 架构一致性: CANN 所有包 (runtime/toolkit/nnal 等) 均采用 run_xxx_install.sh 中间层模式, 保持一致便于维护
#
# 三层分工:
#   _do_install              -> 安装框架 (元数据/权限/版本/错误处理)
#   run_pypto_install.sh     -> 参数组装 + 文件拷贝调度 + 包特有后处理
#   install_common_parser.sh -> 通用文件拷贝引擎 (按 filelist.csv 拷贝文件, 设置权限)

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

_do_install() {
    output_progress 10

    # 执行安装: 调用 install_common_parser.sh 拷贝文件
    #   --copy_all: 要求 dst_path == install_path (与 pypto.xml 的 copy_all=true 一致)
    #   --package="pypto": 包名
    #   --install: 安装模式
    #   --set-cann-uninstall: 允许后续卸载
    #   --use-share-info: 使用 share/info 目录结构
    #   --version / --version-dir: 版本信息
    #   --docker-root: docker 场景路径前缀
    #
    #   install_for_all_param: 由 ARG_INSTALL_FOR_ALL 组装的 CLI flag (y->"--install_for_all", n->空)
    #   末尾三个位置参数: 安装类型, 安装目录, filelist.csv 路径
    local install_for_all_param=""
    if [ "${ARG_INSTALL_FOR_ALL}" = "y" ]; then
        install_for_all_param="--install_for_all"
    fi
    local custom_options=""
    # shellcheck disable=SC2086
    sh "${TMP_PKG_SCRIPTS_PATH}/install_common_parser.sh" --copy_all --package="pypto" --install \
        --username="${USER_NAME}" --usergroup="${USER_GROUP}" --set-cann-uninstall \
        --version="${PKG_VERSION}" --version-dir="${PKG_VERSION_DIR_NAME}" --use-share-info \
        ${install_for_all_param} --docker-root="${ARG_DOCKER_ROOT}" ${custom_options} \
        "${ARG_MODE}" "${ARG_INSTALL_PATH}" "${TMP_PKG_SCRIPTS_PATH}/filelist.csv"
    local sh_ret=$?
    if [ "${sh_ret}" -ne 0 ]; then
        comm_log "ERROR" "ERR_NO:0x0085;ERR_DES:Failed to install package."
        return 1
    fi

    return 0
}

########################################################################################################################
# 主流程
########################################################################################################################
comm_log "INFO" "step into run_pypto_install.sh ......"
comm_log "INFO" "install target dir ${PKG_INSTALL_PATH}, type ${ARG_MODE}."

if ! _do_install; then
    exit 1
fi

output_progress 100
exit 0
