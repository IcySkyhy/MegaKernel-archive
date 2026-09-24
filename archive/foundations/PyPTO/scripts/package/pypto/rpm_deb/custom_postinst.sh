#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------
# RPM/DEB post-install hook. The common CANN packager appends this file to postinst.

set -e
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

wheel_dir="${INSTALL_PATH}/${PKG_ARCH_NAME}-linux/lib64"
wheel_file=$(find "${wheel_dir}" -maxdepth 1 -type f -name 'pypto-*.whl' 2>/dev/null | head -n 1 || true)
if [ -z "${wheel_file}" ]; then
    echo "PyPTO wheel not found under ${wheel_dir}, skip Python package installation."
elif ! run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${wheel_file}"; then
    echo "Failed to install PyPTO wheel: ${wheel_file}" >&2
    unset PIP_BREAK_SYSTEM_PACKAGES
    exit 1
else
    echo "PyPTO wheel installed successfully."
    chmod 444 "${wheel_file}"
fi

# Keep the DEB installation layout aligned with the CANN run installer.
chmod 755 "${INSTALL_PATH}"
mkdir -p "${INSTALL_PATH}/aarch64-linux/include/version"
chmod 555 "${INSTALL_PATH}/aarch64-linux/include/version"
chmod 500 "${INSTALL_PATH}/share/info/pypto/script/install.sh"

uninstall_wrapper="${INSTALL_PATH}/cann_uninstall.sh"
install_common_parser="${INSTALL_PATH}/share/info/pypto/script/install_common_parser.sh"
if [ -x "${install_common_parser}" ]; then
    sh "${install_common_parser}" --add-cann-uninstall \
        --install-path="${INSTALL_PATH}" \
        --username="$(id -un)" --usergroup="$(id -gn)" --install_for_all \
        "share/info/pypto/script"
    chmod 500 "${uninstall_wrapper}"
fi

install_info="${INSTALL_PATH}/share/info/pypto/ascend_install.info"
cat > "${install_info}" <<EOF
PyPTO_Install_Type=full
PyPTO_UserName=$(id -un)
PyPTO_UserGroup=$(id -gn)
PyPTO_Install_Path_Param=$(dirname "${INSTALL_PATH}")
PyPTO_Install_For_All=y
PyPTO_PyLocal=n
EOF
chmod 644 "${install_info}"

for db_entry in 'PyptoSo|pypto' 'pypto|pypto'; do
    if ! grep -qxF "${db_entry}" "${DB_FILE}"; then
        printf '%s\n' "${db_entry}" >> "${DB_FILE}"
    fi
done

unset PIP_BREAK_SYSTEM_PACKAGES
exit 0
