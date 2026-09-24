# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(PTO_Fwk_STestCaseLibraries             "" CACHE INTERNAL "" FORCE)     # STest 各模块 用例实现二进制
set(PTO_Fwk_STestCaseLdLibrariesExt        "" CACHE INTERNAL "" FORCE)     # STest 各模块 额外 Load 二进制
set(PTO_Fwk_STestCaseGoldenScriptPathList  "" CACHE INTERNAL "" FORCE)     # STest 各模块 Golden 脚本路径配置

# 切换完成前, 增加原有目录
set(PTO_Fwk_STestCaseGoldenScriptPathList ${PTO_Fwk_STestCaseGoldenScriptPathList} ${PTO_FWK_SRC_ROOT}/framework/tests/cmake/scripts/golden/op CACHE INTERNAL "" FORCE)

# 用于添加 STest 测试用例二进制库
#[[
Parameters:
  one_value_keywords:
      TARGET                : [Required] 具体测试用例二进制库名称
  multi_value_keywords:
      SOURCES               : [Required] 编译源码
      LD_LIBRARIES_EXT      : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      GOLDEN_SCRIPT_DIR     : [Optional] Golden 脚本所在路径, 便于 Golden 处理公共逻辑查找和载入对应脚本
Attention:
    1.  一般 LD_LIBRARIES_EXT 内配置的二进制, 在正常 source CANN 包环境变量后, LD_LIBRARY_PATH 内也应包含其所在路径;
]]
function(PTO_Fwk_STest_AddLib)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "SOURCES;LD_LIBRARIES_EXT;GOLDEN_SCRIPT_DIR"
            ""
            ${ARGN}
    )
    add_Library(${ARG_TARGET} STATIC)
    target_sources(${ARG_TARGET} PRIVATE ${ARG_SOURCES})
    target_link_libraries(${ARG_TARGET}
            PRIVATE
                ${PTO_Fwk_STestNamePrefix}_utils
                GTest::gtest
    )
    set(PTO_Fwk_STestCaseLibraries            ${PTO_Fwk_STestCaseLibraries}            ${ARG_TARGET}            CACHE INTERNAL "" FORCE)
    set(PTO_Fwk_STestCaseLdLibrariesExt       ${PTO_Fwk_STestCaseLdLibrariesExt}       ${ARG_LD_LIBRARIES_EXT}  CACHE INTERNAL "" FORCE)
    set(PTO_Fwk_STestCaseGoldenScriptPathList ${PTO_Fwk_STestCaseGoldenScriptPathList} ${ARG_GOLDEN_SCRIPT_DIR} CACHE INTERNAL "" FORCE)
endfunction()

function(PTO_Fwk_STest_GetGTestFilterList GTEST_FILTER_LIST)
    get_filename_component(_ClsFile "${PTO_FWK_SRC_ROOT}/framework/tests/st/configs" REALPATH)
    PTO_Fwk_GTest_GetGTestFilterStr(GTestFilterStr
            CLASSIFY        ${_ClsFile}
            TESTS_TYPE      stest
            TESTS_GROUP     ${ENABLE_STEST_GROUP}
            CHANGED_FILE    ${ENABLE_TESTS_EXECUTE_CHANGED_FILE}
    )
    string(REPLACE ":" ";" GTestFilterList "${GTestFilterStr}")
    list(LENGTH GTestFilterList YamlGTestFilterListLen)
    list(REMOVE_DUPLICATES GTestFilterList)
    set(${GTEST_FILTER_LIST} ${GTestFilterList} PARENT_SCOPE)
    list(LENGTH GTestFilterList RstGTestFilterListLen)
    message(STATUS "GetSTestFilterList: Yaml(${YamlGTestFilterListLen}), Total(${RstGTestFilterListLen})")
endfunction()

# STest 生成 Golden 数据
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 指定所依赖的目标(POST_BUILD)
  multi_value_keywords:
      GTEST_FILTER_LIST  : [Required] GTestFilter 配置, Filter 间以 ';' 分割
]]
function(PTO_Fwk_STest_RunExe_GenerateGolden)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "GTEST_FILTER_LIST"
            ""
            ${ARGN}
    )
    if (ENABLE_TESTS_EXECUTE)
        set(_Args)
        list(APPEND _Args "-o=${ENABLE_STEST_GOLDEN_PATH}")

        string(REPLACE ";" ":" GTestFilterStr "${ARG_GTEST_FILTER_LIST}")
        list(APPEND _Args "-c=${GTestFilterStr}")
        list(REMOVE_DUPLICATES PTO_Fwk_STestCaseGoldenScriptPathList)
        foreach (_Path ${PTO_Fwk_STestCaseGoldenScriptPathList})
            list(APPEND _Args "--path=${_Path}")
        endforeach ()

        if (ENABLE_STEST_GOLDEN_PATH_CLEAN)
            list(APPEND _Args "--clean")
        endif ()

        get_filename_component(GoldenCtrlPy    "${PTO_FWK_SRC_ROOT}/cmake/scripts/golden_ctrl.py" REALPATH)
        get_filename_component(GoldenCtrlPyCwd "${PTO_FWK_SRC_ROOT}/cmake/scripts" REALPATH)
        list(LENGTH ARG_GTEST_FILTER_LIST GTestFilterListLen)
        add_custom_command(
                TARGET ${ARG_TARGET} POST_BUILD
                COMMAND ${Python3_EXECUTABLE} ${GoldenCtrlPy} ARGS ${_Args}
                COMMENT "Generator Golden(${GTestFilterListLen}) for ${ARG_TARGET}"
                WORKING_DIRECTORY ${GoldenCtrlPyCwd}
        )
    endif ()
endfunction()

# STest 执行可执行程序
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行目标, 用例会在该目标编译完成后(POST_BUILD)启动执行
  multi_value_keywords:
      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      ENV_LINES_EXT      : [Optional] 需要额外配置的环境变量, 按照 "K=V" 格式组织
      GTEST_FILTER_LIST  : [Optional] GTestFilter 配置, Filter 间以 ';' 分割
Attention:
    1. 可以多次调用本函数以添加多个'执行任务'; 单次调用本函数时, 可以通过在 GTEST_FILTER_LIST 中配置多个过滤条件('gtest_filter') 以实现执行多用例;
]]
function(PTO_Fwk_STest_RunExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;ENV_LINES_EXT;GTEST_FILTER_LIST"
            ""
            ${ARGN}
    )
    if (ENABLE_TESTS_EXECUTE)
        # 命令行参数处理
        PTO_Fwk_GTest_RunExe_GetPreExecSetup(PyCmdSetup PyEnvLines BashCmdSetup
                TARGET              ${ARG_TARGET}
                ENV_LINES_EXT       ${ARG_ENV_LINES_EXT}
                LD_LIBRARIES_EXT    ${ARG_LD_LIBRARIES_EXT}
        )
        # 执行流程
        list(LENGTH ARG_GTEST_FILTER_LIST GtestFilterListLen)
        string(REPLACE ";" ":" GtestFilterStr "${ARG_GTEST_FILTER_LIST}")
        list(LENGTH PTO_Fwk_StestExecuteDeviceIdList DeviceIdListLen)
        string(REPLACE ";" ", " DeviceIdStr "${PTO_Fwk_StestExecuteDeviceIdList}")
        message(STATUS "Run GTest(${ARG_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN}), Device(${DeviceIdListLen})=[${DeviceIdStr}], GTestFilter(${GtestFilterListLen})=${GtestFilterStr}")
        set(Comment "Run GTest(${ARG_TARGET}), XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN})")

        if (ARG_GTEST_FILTER_LIST)
            if (ENABLE_TESTS_EXECUTE_PARALLEL OR (DeviceIdListLen GREATER 1))
                # 仅在使能并行执行全局开关, 且需要做 filter 时才进行执行加速
                set(_File $<TARGET_FILE:${ARG_TARGET}>)
                set(_Args "-t=${_File}" "--cases=${GtestFilterStr}" "--halt_on_error")
                foreach (DevId ${PTO_Fwk_StestExecuteDeviceIdList})
                    list(APPEND _Args "--device=${DevId}")
                endforeach ()
                if (PyEnvLines)
                    list(APPEND _Args "--env" "${PyEnvLines}")
                endif ()
                get_filename_component(ParallelPy    "${PTO_FWK_SRC_ROOT}/cmake/scripts/stest_accelerate.py" REALPATH)
                get_filename_component(ParallelPyCwd "${PTO_FWK_SRC_ROOT}/cmake/scripts" REALPATH)
                add_custom_command(
                        TARGET ${ARG_TARGET} POST_BUILD
                        COMMAND ${PyCmdSetup} ${Python3_EXECUTABLE} ${ParallelPy} ARGS ${_Args}
                        COMMENT "${Comment} With Parallel Execute Accelerate"
                        WORKING_DIRECTORY ${ParallelPyCwd}
                )
            else ()
                set(GtestFilterListIdx 1)
                foreach (Filter ${ARG_GTEST_FILTER_LIST})
                    add_custom_command(
                            TARGET ${ARG_TARGET} POST_BUILD
                            COMMAND ${BashCmdSetup} ./${ARG_TARGET} ARGS '--gtest_filter=${Filter}'
                            COMMENT "${Comment} [${GtestFilterListIdx}/${GtestFilterListLen}] With --gtest_filter=${Filter}"
                            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                    )
                    math(EXPR GtestFilterListIdx "${GtestFilterListIdx} + 1")
                endforeach ()
            endif ()
        else ()
            message(STATUS "No cases need to run.")
        endif ()
    endif ()
endfunction()

# STest 用于添加并执行 可执行文件
#[[
Parameters:
  one_value_keywords:
      TARGET                        : [Required] 用于指定具体 GTest 可执行目标
]]
function(PTO_Fwk_STest_AddExe_RunExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            ""
            ""
            ${ARGN}
    )
    #
    # 编译
    #
    set(_Sources ${CMAKE_CURRENT_BINARY_DIR}/${PTO_Fwk_STestNamePrefix}_main_stub.cpp)
    execute_process(COMMAND touch ${_Sources})
    list(REMOVE_DUPLICATES PTO_Fwk_STestCaseLibraries)
    set(PTO_Fwk_Libraries
            tile_fwk_utils
            tile_fwk_adapter
            tile_fwk_cann_host_runtime
            tile_fwk_platform
            tile_fwk_interface
            tile_fwk_codegen
            tile_fwk_compiler
            tile_fwk_runtime
            tile_fwk_simulation
            tile_fwk_operator
    )
    PTO_Fwk_GTest_AddExe(
            TARGET                      ${ARG_TARGET}
            SOURCES                     ${_Sources}
            PRIVATE_LINK_LIBRARIES      ${PTO_Fwk_STestNamePrefix}_utils ${PTO_Fwk_Libraries} ${PTO_Fwk_STestCaseLibraries}
    )
    if (ENABLE_TORCH_VERIFIER)
        add_dependencies(${ARG_TARGET} tile_fwk_calculator)
    endif()

    #
    # 执行
    #
    set(EnvLinesExt
            "TILE_FWK_STEST_GOLDEN_PATH=${ENABLE_STEST_GOLDEN_PATH}"
    )
    PTO_Fwk_STest_GetGTestFilterList(GTestFilterList)
    if (NOT "${ENABLE_STEST}" STREQUAL "ON")
        set(GTestFilterList ${ENABLE_STEST})
        string(REPLACE ":" ";" GTestFilterList "${GTestFilterList}")
    endif ()
    list(REMOVE_DUPLICATES GTestFilterList)
    list(REMOVE_DUPLICATES PTO_Fwk_STestCaseLdLibrariesExt)

    if (NOT "$ENV{GTEST_START}" STREQUAL "")
        list(FIND GTestFilterList $ENV{GTEST_START} idx)
        if (NOT ${idx} EQUAL -1)
            list(SUBLIST GTestFilterList ${idx} -1 GTestFilterList)
        endif ()
    endif ()

    if ("${GTestFilterList}x" STREQUAL "x")
        message(STATUS "No Case to Execute")
    else ()
        # 精度用例
        PTO_Fwk_STest_RunExe_GenerateGolden(TARGET ${ARG_TARGET} GTEST_FILTER_LIST ${GTestFilterList})
        PTO_Fwk_STest_RunExe(
                TARGET              ${ARG_TARGET}
                ENV_LINES_EXT       ${EnvLinesExt}
                LD_LIBRARIES_EXT    ${PTO_Fwk_STestCaseLdLibrariesExt}
                GTEST_FILTER_LIST   ${GTestFilterList}
        )
    endif ()
endfunction()

# Distributed 获取 GTestFilterList
#[[
Parameters:
  multi_value_keywords:
      GTEST_FILTER_CONFIG   : [Required] GtestFilterConfig, 以 CaseName RankSize 顺序存储
      GTEST_FILTER_LIST     : [Required] GtestFilterList, 仅包含 CaseName
]]
function(PTO_Fwk_STest_Distributed_GetGTestFilterList GTEST_FILTER_LIST)
    cmake_parse_arguments(
            ARG
            ""
            ""
            "GTEST_FILTER_CONFIG"
            ""
            ${ARGN}
    )
    # Config 到 List 的转换
    set(Idx 0)
    set(FilterList)
    foreach (CFG ${ARG_GTEST_FILTER_CONFIG})
        math(EXPR Idx "${Idx} + 1")
        math(EXPR Remainder "${Idx} % 2")  # 计算索引除以2的余数
        if (Remainder EQUAL 1)
            # 支持由 ENABLE_STEST_DISTRIBUTED 传入指定的 Filter
            if ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "ON")
                list(APPEND FilterList ${CFG})
            else ()
                string(REPLACE ":" ";" SpecifyGtestFilterList ${ENABLE_STEST_DISTRIBUTED})
                list(FIND SpecifyGtestFilterList ${ENABLE_STEST_DISTRIBUTED} _CfgIdx)
                if (NOT "${_SepIdx}" STREQUAL "-1")
                    list(APPEND FilterList ${ENABLE_STEST_DISTRIBUTED})
                endif ()
            endif ()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES FilterList)
    set(${GTEST_FILTER_LIST} ${FilterList} PARENT_SCOPE)
endfunction()

# Distributed 查询 RankSize
#[[
Parameters:
  multi_value_keywords:
      GTEST_FILTER_CONFIG   : [Required] GtestFilterConfig, 以 CaseName RankSize 顺序存储
      RANK_SIZE             : [Required] RankSize
]]
function(PTO_Fwk_STest_Distributed_GetRankSize RANK_SIZE)
    cmake_parse_arguments(
            ARG
            ""
            "GTEST_FILTER"
            "GTEST_FILTER_CONFIG"
            ""
            ${ARGN}
    )
    set(Idx 0)
    set(RandSize)
    foreach (CFG ${ARG_GTEST_FILTER_CONFIG})
        if ("${CFG}" STREQUAL "${ARG_GTEST_FILTER}")
            math(EXPR RankSizeIdx "${Idx} + 1")
            list(GET ARG_GTEST_FILTER_CONFIG ${RankSizeIdx} RandSize)
            break()
        endif ()
        math(EXPR Idx "${Idx} + 1")
    endforeach()
    if (NOT RandSize)
        message(FATAL_ERROR "Can't get RandSize, GTestFilter(${ARG_GTEST_FILTER})")
    endif ()
    set(${RANK_SIZE} ${RandSize} PARENT_SCOPE)
endfunction()

# 用于执行 Distributed 相关的 ST 用例
#[[
Parameters:
  one_value_keywords:
      TARGET             : [Required] 用于指定具体 GTest 可执行文件, 用例会在该目标编译完成后启动执行
  multi_value_keywords:
      LD_LIBRARIES_EXT   : [Optional] 需要在执行时将所在路径配置到环境变量 LD_LIBRARY_PATH 中的 Libraries
      ENV_LINES_EXT      : [Optional] 需要额外配置的环境变量, 按照 "K=V" 格式组织
      GTEST_FILTER_LIST  : [Optional] GTestFilter 配置, Filter 间以 ';' 分割
      GOLDEN_SCRIPT_DIR  : [Optional] Golden 脚本所在路径, 便于 Golden 处理公共逻辑查找和载入对应脚本
]]
function(PTO_Fwk_STest_Distributed_RunExe)
    cmake_parse_arguments(
            ARG
            ""
            "TARGET"
            "LD_LIBRARIES_EXT;ENV_LINES_EXT;GTEST_FILTER_CONFIG;GOLDEN_SCRIPT_DIR"
            ""
            ${ARGN}
    )
    set(PTO_Fwk_STestCaseGoldenScriptPathList ${PTO_Fwk_STestCaseGoldenScriptPathList} ${ARG_GOLDEN_SCRIPT_DIR} CACHE INTERNAL "" FORCE)
    if (ENABLE_TESTS_EXECUTE)
        # Config 到 List 转换, 并处理由 ENABLE_STEST_DISTRIBUTED 传入指定的 Filter 的情况
        PTO_Fwk_STest_Distributed_GetGTestFilterList(GTestFilterList
                GTEST_FILTER_CONFIG ${ARG_GTEST_FILTER_CONFIG}
        )
        # Distributed 用例当前需配置 RankSize, 故此处做判空处理:
        # 1. 当指定执行的某个用例不在 Distributed 范围, 此处 GTestFilterList 为空, 不触发执行;
        # 2. 当指定执行的某个用例属于 Distributed 范围, 此处 GTestFilterList 非空, 会触发执行;
        #    此时 tile_fwk_stest 由于无法判断指定用例是否在执行范围, 默认会触发执行, 但由于对应用例不在 tile_fwk_stest 承载,
        #    执行会报错; 所以此种场景需要 Distributed 指定具体 target, 如:
        #    python3 build_ci.py -t=tile_fwk_stest_distributed -stest_distributed=xxx
        set(MaxRankSize 0)
        foreach (Filter ${GTestFilterList})
            PTO_Fwk_STest_Distributed_GetRankSize(RankSize
                    GTEST_FILTER        ${Filter}
                    GTEST_FILTER_CONFIG ${ARG_GTEST_FILTER_CONFIG}
            )
            # 比较并更新最大RankSize
            if (${RankSize} GREATER ${MaxRankSize})
                set(MaxRankSize ${RankSize})
            endif ()
        endforeach ()
        if (GTestFilterList)
            # 命令行参数处理
            PTO_Fwk_GTest_RunExe_GetPreExecSetup(PyCmdSetup PyEnvLines BashCmdSetup
                    TARGET              ${ARG_TARGET}
                    ENV_LINES_EXT       ${ARG_ENV_LINES_EXT}
                    LD_LIBRARIES_EXT    ${ARG_LD_LIBRARIES_EXT}
            )
            # Golden 生成
            PTO_Fwk_STest_RunExe_GenerateGolden(TARGET ${ARG_TARGET} GTEST_FILTER_LIST ${GTestFilterList})
            # 执行流程
            math(EXPR MaxRankSizeTimes2 "${MaxRankSize} * 2")
            string(REPLACE ";" ":" GtestFilterStr "${GTestFilterList}")
            list(LENGTH PTO_Fwk_StestExecuteDeviceIdList DeviceIdListLen)
            set(Comment "Run GTest(${ARG_TARGET}) XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN})")
            if(ENABLE_TESTS_EXECUTE_PARALLEL OR (DeviceIdListLen GREATER_EQUAL MaxRankSizeTimes2))
                set(_File $<TARGET_FILE:${ARG_TARGET}>)
                set(_Args "-t=${_File}" "--cases=${GtestFilterStr}" "--halt_on_error")
                foreach (DevId ${PTO_Fwk_StestExecuteDeviceIdList})
                    list(APPEND _Args "--device=${DevId}")
                endforeach ()
                if (ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT)
                    list(APPEND _Args "--timeout=${ENABLE_TESTS_EXECUTE_PARALLEL_TIMEOUT}")
                endif ()
                if (PyEnvLines)
                    list(APPEND _Args "--env" "${PyEnvLines}")
                endif ()
                list(APPEND _Args "--rank_size" "${MaxRankSize}")
                get_filename_component(ParallelPy    "${PTO_FWK_SRC_ROOT}/cmake/scripts/distributed_stest_accelerate.py" REALPATH)
                get_filename_component(ParallelPyCwd "${PTO_FWK_SRC_ROOT}/cmake/scripts" REALPATH)
                add_custom_command(
                        TARGET ${ARG_TARGET} POST_BUILD
                        COMMAND ${PyCmdSetup} ${Python3_EXECUTABLE} ${ParallelPy} ARGS ${_Args}
                        COMMENT "${Comment} With Parallel Execute Accelerate"
                        WORKING_DIRECTORY ${ParallelPyCwd}
                )
            else ()
                set(GtestFilterListIdx 1)
                message(STATUS "Run GTest(${ARG_TARGET}) XSAN(ASAN:${ENABLE_ASAN} UBSAN:${ENABLE_UBSAN}) GTestFilter(${GtestFilterListLen})=${GtestFilterStr}")
                foreach (Filter ${GTestFilterList})
                    PTO_Fwk_STest_Distributed_GetRankSize(RankSize
                            GTEST_FILTER        ${Filter}
                            GTEST_FILTER_CONFIG ${ARG_GTEST_FILTER_CONFIG}
                    )
                    add_custom_command(
                            TARGET ${ARG_TARGET} POST_BUILD
                            COMMAND ${BashCmdSetup} mpirun -n ${RankSize} ./${ARG_TARGET} ARGS '--gtest_filter=${Filter}'
                            COMMENT "${Comment} [${GtestFilterListIdx}/${GtestFilterListLen}] With --gtest_filter=${Filter}"
                            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                    )
                    math(EXPR GtestFilterListIdx "${GtestFilterListIdx} + 1")
                endforeach ()
            endif ()
        endif ()
    endif ()
endfunction()
