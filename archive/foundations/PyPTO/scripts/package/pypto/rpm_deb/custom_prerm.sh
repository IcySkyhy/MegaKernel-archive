#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------
# RPM/DEB pre-uninstall hook. The common CANN packager appends this file to prerm.

unset PYTHONPATH
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip()
{
    if python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip "$@"
    elif command -v pip3 >/dev/null 2>&1; then
        pip3 "$@"
    else
        return 127
    fi
}

if run_pip uninstall -y pypto; then
    echo "PyPTO Python package uninstalled successfully."
else
    echo "PyPTO Python package is not installed or could not be uninstalled; continuing."
fi

# Only remove directories when empty; never remove files belonging to another package.
rmdir "${INSTALL_PATH}/python/site-packages" 2>/dev/null || true
rmdir "${INSTALL_PATH}/python" 2>/dev/null || true

# Remove metadata created by custom_postinst before the common CANN cleanup.
rm -f "${INSTALL_PATH}/cann_uninstall.sh"
rm -f "${INSTALL_PATH}/share/info/pypto/ascend_install.info"
rmdir "${INSTALL_PATH}/aarch64-linux/include/version" 2>/dev/null || true
if [ -f "${DB_FILE}" ]; then
    sed -i '/^PyptoSo|/d; /^pypto|/d' "${DB_FILE}"
fi

unset PIP_BREAK_SYSTEM_PACKAGES
