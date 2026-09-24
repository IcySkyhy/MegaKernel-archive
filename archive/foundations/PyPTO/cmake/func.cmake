# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 源码列表调试
#[[
Parameters:
  options:
      DETAIL                        : [Optional] 是否输出详细信息
  one_value_keywords:
      NAME                          : [Required] 指定目标名称/标识名称
  multi_value_keywords:
      LIST                          : [Required] 源码列表
]]
function(PTO_Fwk_Debug_Source_List)
    cmake_parse_arguments(
            ARG
            "DETAIL"
            "NAME"
            "LIST"
            ""
            ${ARGN}
    )
    list(LENGTH ARG_LIST _Len)
    message(STATUS "${ARG_NAME}: Length: ${_Len}")
    if (ARG_DETAIL)
        foreach (_f ${ARG_LIST})
            message(STATUS "${_f}")
        endforeach ()
    endif ()
endfunction()

# 分析二进制的符号信息
#[[
Parameters:
  options:
      DF                            : [Optional] 输出已定义的关系
      IGNORE_UDF_SELF               : [Optional] 忽略自身的未定义符号(UDF: Undefined)
      IGNORE_UDF_PASSED             : [Optional] 忽略传递的未定义符号(UDF: Undefined)
  one_value_keywords:
      TARGET                        : [Required] 指定目标名称
]]
function(PTO_Fwk_AnalysisTargetSymbols)
    cmake_parse_arguments(
            ARG
            "DF;IGNORE_UDF_SELF;IGNORE_UDF_PASSED"
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    if (ENABLE_COMPILE_DEPENDENCY_CHECK
            AND (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
            AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"))
        set(_file $<TARGET_FILE:${ARG_TARGET}>)
        get_filename_component(_PyScript "${PTO_FWK_SRC_ROOT}/cmake/scripts/analysis_binary_symbol.py" REALPATH)
        set(_Args "-f=${_file}")
        if (ARG_DF)
            list(APPEND _Args "--print_defined_relations")
        endif ()
        if (ARG_IGNORE_UDF_SELF)
            list(APPEND _Args "--ignore_undefined_symbols_self")
        endif ()
        if (ARG_IGNORE_UDF_PASSED)
            list(APPEND _Args "--ignore_undefined_symbols_pass")
        endif ()
        set(_LdLibPathExt)
        if (ENABLE_COMPILE_DEPENDENCY_CHECK AND CMAKE_SKIP_RPATH)
            set(_LdLibPathExt ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        endif ()
        add_custom_command(
                TARGET ${ARG_TARGET} POST_BUILD
                COMMAND cmake -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${_LdLibPathExt} ${Python3_EXECUTABLE} ${_PyScript} ARGS ${_Args}
                COMMENT "Analysis symbol of ${ARG_TARGET}"
        )
    endif ()
endfunction()

# 分析二进制的头文件依赖信息
#[[
Parameters:
  one_value_keywords:
      TARGET                        : [Required] 指定目标名称
]]
function(PTO_Fwk_AnalysisTargetHeaderFiles)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    if (ENABLE_COMPILE_DEPENDENCY_CHECK
            AND (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
            AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"))
        set(_TargetFile $<TARGET_FILE:${ARG_TARGET}>)
        get_target_property(_TargetBinaryDir ${ARG_TARGET} BINARY_DIR)
        get_filename_component(_TargetBinaryDir "${_TargetBinaryDir}" REALPATH)
        set(_TargetObjects $<TARGET_OBJECTS:${ARG_TARGET}>)
        get_filename_component(_PyScript "${PTO_FWK_SRC_ROOT}/cmake/scripts/analysis_binary_header_files.py" REALPATH)
        get_filename_component(_JsonCfg "${PTO_FWK_SRC_ROOT}/cmake/scripts/analysis_binary_header_files.json" REALPATH)
        set(_Args
                "-s=${PTO_FWK_SRC_ROOT}"
                "-b=${PTO_FWK_BIN_ROOT}"
                "-t=${_TargetFile}"
                "--target_binary_dir=${_TargetBinaryDir}"
                "-o='${_TargetObjects}'"
                "-j=${_JsonCfg}"
        )
        # 获取 gcc 默认头文件搜索路径
        execute_process(
                COMMAND ${CMAKE_C_COMPILER} --print-sysroot
                RESULT_VARIABLE _RST
                OUTPUT_VARIABLE _SUFFIX
                ERROR_QUIET
        )
        list(APPEND _Args "-f=${SYS_ROOT}/usr/include")
        list(APPEND _Args "-f=${SYS_ROOT}/usr/lib")
        # CANN
        if (BUILD_WITH_CANN)
            list(APPEND _Args "-f=${ASCEND_CANN_PACKAGE_PATH}/include")
            list(APPEND _Args "-f=${ASCEND_CANN_PACKAGE_PATH}/pkg_inc")
        endif ()
        # OpenSource
        get_target_property(json_inc json INTERFACE_INCLUDE_DIRECTORIES)
        set(OpenSourceInc ${json_inc})
        foreach (_Inc ${OpenSourceInc})
            list(APPEND _Args "-f=${_Inc}")
        endforeach ()
        list(REMOVE_DUPLICATES _Args)

        add_custom_command(
                TARGET ${ARG_TARGET} POST_BUILD
                COMMAND ${Python3_EXECUTABLE} ${_PyScript} ARGS ${_Args}
                COMMENT "Analysis Header-File of ${ARG_TARGET}"
        )
    endif ()
endfunction()

# 分析 Python3 环境信息, 主要是 pip 包信息
#[[
Parameters:
  one_value_keywords:
      OUTPUT_FILE            : [Require] 输出 cmake 文件路径
]]
function(PTO_Fwk_AnalysisPython3Environ)
    cmake_parse_arguments(
            ARG
            ""
            "OUTPUT_FILE"
            ""
            ""
            ${ARGN}
    )
    get_filename_component(_PyScript "${PTO_FWK_SRC_ROOT}/cmake/scripts/analysis_python3_environ.py" REALPATH)
    set(_Args "-o=${ARG_OUTPUT_FILE}")
    execute_process(
            COMMAND ${Python3_EXECUTABLE} ${_PyScript} ${_Args}
            RESULT_VARIABLE _Rst
    )
    if (NOT ${_Rst} EQUAL 0)
        message(FATAL_ERROR "Analysis python3 environ failed.")
    endif ()
    if (NOT EXISTS ${ARG_OUTPUT_FILE} OR IS_DIRECTORY ${ARG_OUTPUT_FILE})
        message(FATAL_ERROR "Analysis python3 environ failed, ${ARG_OUTPUT_FILE} not exist.")
    endif ()
endfunction()

function(PTO_Fwk_InstallBinaries)
    cmake_parse_arguments(
            ARG
            ""
            "EXPORT;WHL_NAME;INSTALL_BINDIR;INSTALL_LIBDIR"
            "TARGETS"
            ""
            ${ARGN}
    )
    # 设置 二进制文件 导出属性
    foreach (Target ${ARG_TARGETS})
        set_target_properties(${Target}
                PROPERTIES
                    IMPORTED_LOCATION   "$<TARGET_FILE:${Target}>"
                    OUTPUT_NAME         "${Target}"
        )
    endforeach ()
    # 安装路径设置
    if (ARG_INSTALL_BINDIR)
        set(_InstallBinDir ${ARG_INSTALL_BINDIR})
    elseif (ENABLE_FEATURE_PYTHON_FRONT_END)
        set(_InstallBinDir "${ARG_WHL_NAME}/${CMAKE_INSTALL_BINDIR}")
    else ()
        set(_InstallBinDir ${CMAKE_INSTALL_BINDIR})
    endif ()
    if (ARG_INSTALL_LIBDIR)
        set(_InstallLibDir ${ARG_INSTALL_LIBDIR})
    elseif (ENABLE_FEATURE_PYTHON_FRONT_END)
        set(_InstallLibDir "${ARG_WHL_NAME}/lib")
    else ()
        set(_InstallLibDir ${CMAKE_INSTALL_LIBDIR})
    endif ()
    # 安装 二进制文件
    install(TARGETS ${ARG_TARGETS}
            EXPORT ${ARG_EXPORT}
            RUNTIME DESTINATION ${_InstallBinDir} COMPONENT pypto
            LIBRARY DESTINATION ${_InstallLibDir} COMPONENT pypto
            ARCHIVE DESTINATION ${_InstallLibDir} COMPONENT pypto
    )
endfunction()

function(PTO_Fwk_InstallCMakeConfig)
    cmake_parse_arguments(
            ARG
            ""
            "EXPORT;WHL_NAME;CMAKE_PARENT_DIR"
            "TARGETS"
            ""
            ${ARGN}
    )

    # 安装导出目标
    install(EXPORT ${ARG_EXPORT}
            FILE ${ARG_EXPORT}.cmake
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION ${ARG_CMAKE_PARENT_DIR}/cmake/${PROJECT_NAME}
    )

    # 创建别名目标, 并创建聚合目标
    add_library(${PROJECT_NAME} INTERFACE)
    foreach (Target ${ARG_TARGETS})
        add_library(${PROJECT_NAME}::${Target} ALIAS ${Target})
        target_link_libraries(${PROJECT_NAME} INTERFACE ${PROJECT_NAME}::${Target})
    endforeach ()

    # 安装聚合目标
    install(TARGETS ${PROJECT_NAME} EXPORT ${ARG_EXPORT})

    # 生成 ConfigVersion 配置文件
    write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMajorVersion
    )

    # 配置 Config 配置文件
    set(WHL_NAME    ${ARG_WHL_NAME})
    set(TARGETS     ${ARG_TARGETS})
    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
            INSTALL_DESTINATION ${ARG_CMAKE_PARENT_DIR}/cmake/${PROJECT_NAME}
            PATH_VARS
                PROJECT_NAME
                PROJECT_VERSION
                PROJECT_VERSION_MAJOR
                PROJECT_VERSION_MINOR
                PROJECT_VERSION_PATCH
                WHL_NAME
                TARGETS
            NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )

    # 安装配置文件
    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
            DESTINATION ${ARG_CMAKE_PARENT_DIR}/cmake/${PROJECT_NAME}
    )
endfunction()

function(PTO_Fwk_CleanEmptyDir)
    cmake_parse_arguments(
            ARG
            ""
            "DIR"
            ""
            ""
            ${ARGN}
    )
    if (EXISTS ${ARG_DIR} AND IS_DIRECTORY ${ARG_DIR})
        file(GLOB _DirItemLst LIST_DIRECTORIES true RELATIVE "${ARG_DIR}" "${ARG_DIR}/*")
        list(LENGTH _DirItemLst _DirItemNum)
        if (_DirItemNum EQUAL 0)
            file(REMOVE_RECURSE FORCE ${ARG_DIR})
            message(STATUS "Remove empty dir: ${ARG_DIR}")
        endif()
    endif ()
endfunction()

# 生成 gcov 配置文件 (JSON 格式)
# 用于 Python 前端覆盖率生成场景，将 CMake Configure 阶段的参数持久化供外部使用
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 指定所依赖的目标(POST_BUILD)
  multi_value_keywords:
      FILTER_DIRECTORIES : [Optional] 覆盖率结果过滤目录
]]
function(PTO_Fwk_GenerateGCovConfigJson)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "FILTER_DIRECTORIES"
            ""
            ${ARGN}
    )
    if (ENABLE_FEATURE_PYTHON_FRONT_END AND ENABLE_GCOV)
        # 获取 gcc 默认头文件搜索路径
        execute_process(
                COMMAND ${CMAKE_C_COMPILER} --print-sysroot
                RESULT_VARIABLE _SYSROOT_RST
                OUTPUT_VARIABLE _SYSROOT_SUFFIX
                ERROR_QUIET
        )
        if (_SYSROOT_RST)
            get_filename_component(SYS_ROOT "/usr/include" REALPATH)
        else ()
            get_filename_component(SYS_ROOT "${_SYSROOT_SUFFIX}/usr/include" REALPATH)
        endif ()

        set(Filter_Dirs
                ${PTO_FWK_SRC_ROOT}/framework/tests
                $<TARGET_PROPERTY:json,INTERFACE_INCLUDE_DIRECTORIES>
                ${SYS_ROOT}
                ${Python3_INCLUDE_DIRS}
                ${pybind11_INCLUDE_DIR}
                ${ARG_FILTER_DIRECTORIES}
        )
        if (ENABLE_TORCH_VERIFIER)
            list(APPEND Filter_Dirs ${PY3_MOD_TORCH_ROOT_PATH}/include)
        endif ()
        if (BUILD_WITH_CANN)
            list(APPEND Filter_Dirs ${ASCEND_CANN_PACKAGE_PATH}/include)
            list(APPEND Filter_Dirs ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc)
        endif ()

        # 参数组织
        set(_Args "-d=${PTO_FWK_BIN_ROOT}")
        foreach (_dir ${Filter_Dirs})
            list(APPEND _Args "-f=${_dir}")
        endforeach ()
        list(REMOVE_DUPLICATES _Args)

        # 脚本调用
        get_filename_component(GenCoverageCfgPy ${PTO_FWK_SRC_ROOT}/cmake/scripts/gen_coverage_config.py REALPATH)
        add_custom_command(
                TARGET ${ARG_TARGET} POST_BUILD
                COMMAND ${Python3_EXECUTABLE} ${GenCoverageCfgPy} ARGS ${_Args}
                VERBATIM
                COMMENT "Generate coverage config for ${ARG_TARGET}"
        )
    endif()
endfunction()
