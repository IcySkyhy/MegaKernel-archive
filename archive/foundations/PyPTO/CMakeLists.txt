# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.3)

# 必须在 project() 命令之前执行
option(ENABLE_FEATURE_PACKING_WHL_INTO_RUN "Pack whl into run package" OFF)
option(ENABLE_UNIFIED_BUILD "Enable CANN unified full-source build (CMake-driven whl + run)" OFF)
set(PACKAGE_TYPE "run" CACHE STRING "Package type: run, rpm, deb, or all")
include(cmake/prepare.cmake)
include(cmake/fetch_cann_cmake.cmake)

project(PTO_Framework
        LANGUAGES C CXX
        VERSION 1.0.0
)
init_cann_project(PREPEND_MODULE_PATH)
add_cann_target_options()

if (ENABLE_FEATURE_PACKING_WHL_INTO_RUN)
    # run 包 名称
    #    统一用于 COMPONENT / share/info 路径 / scripts/package 路径
    #    最终 run 包名 "cann-pypto-9.1.0-Linux.run" 由 XML product_name=cann + func_name=pypto 组合, 不由此变量决定
    set(PyPTO_PkgName "pypto")
    # run 包 版本
    include(version.cmake)
    add_cann_version_info_targets()
    # run 包 打包流程
    include(cmake/package.cmake)
    set_cann_cpack_config(${PyPTO_PkgName} OUTPUT "${RUN_OUTPUT_DIR}" PACKAGE_TYPE "${PACKAGE_TYPE}")
    return()
endif()


########################################################################################################################
# 基础配置
########################################################################################################################

# 路径配置
get_filename_component(PTO_FWK_SRC_ROOT         "${CMAKE_CURRENT_SOURCE_DIR}"   REALPATH)
get_filename_component(PTO_FWK_BIN_ROOT         "${CMAKE_CURRENT_BINARY_DIR}"   REALPATH)
get_filename_component(PTO_FWK_BIN_OUTPUT_ROOT  "${PTO_FWK_BIN_ROOT}/output"    REALPATH)  # 二进制 BUILD 阶段输出路径

message(STATUS "PTO_FWK_SRC_ROOT=${PTO_FWK_SRC_ROOT}")
message(STATUS "PTO_FWK_BIN_ROOT=${PTO_FWK_BIN_ROOT}")

# 公开编译开关(build)
option(BUILD_WITH_CANN "Build with CANN." ON)  # Need to install CANN package and set corresponding environment variables.
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_GCOV "Enable GNU Coverage Instrumentation Tool" OFF)
option(ENABLE_COMPILE_DEPENDENCY_CHECK "Enable compile dependency relation check" OFF)

# 公开编译开关(Feature 特性级开关)
option(ENABLE_FEATURE_PYTHON_FRONT_END "Enable Python Front-end" ON)
option(ENABLE_FEATURE_PYBIND11_IMPL_COMPILE_ONLINE "Enable Pybind11 impl compile online" OFF)

# 公开编译开关(tests)
option(ENABLE_TESTS_EXECUTE "Enable auto execute tests executable" OFF)
option(ENABLE_TESTS_EXECUTE_PARALLEL "Auto execute tests executable parallel" OFF)
option(ENABLE_TESTS_EXECUTE_CHANGED_FILE "Specify tests changed file" OFF)
option(ENABLE_UTEST "Enable UTest" OFF)
option(ENABLE_UTEST_MODULE "Enable UTest Module compile and execute" OFF)
option(ENABLE_STEST_GOLDEN_PATH "Specific STest Golden Path" OFF)
option(ENABLE_STEST_GOLDEN_PATH_CLEAN "Clean STest Golden" OFF)
option(ENABLE_STEST_EXECUTE_DEVICE_ID "Auto execute tests executable with Device Id" OFF)
option(ENABLE_STEST_DUMP_JSON "Enable STest Dump Json" OFF)
option(ENABLE_STEST_INTERPRETER_CONFIG "Enable STest Interpreter Config" OFF)
option(ENABLE_STEST_BINARY_CACHE "Enable STest Binary Cache" OFF)
option(ENABLE_STEST "Enable STest" OFF)
option(ENABLE_STEST_GROUP "Enable STest Group" OFF)
option(ENABLE_STEST_DISTRIBUTED "Enable STest(Distributed)" OFF)

# 缺省编译配置, 编译函数
find_package(Threads REQUIRED)
include(ExternalProject)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/func.cmake)
include(cmake/config.cmake)
include(cmake/intf.cmake)
find_cann_package(acl_rt REQUIRED)
find_cann_package(ascend_dump REQUIRED)
find_cann_package(ascend_hal REQUIRED)
find_cann_package(hcomm REQUIRED)
find_cann_package(runtime REQUIRED)
find_cann_package(securec REQUIRED)
add_cann_third_party(json)
if (ENABLE_TESTS)
    add_cann_third_party(gtest)
endif()


########################################################################################################################
# 编译流程
########################################################################################################################

add_subdirectory(framework)
add_subdirectory(python)


########################################################################################################################
# 安装流程
########################################################################################################################

if (ENABLE_FEATURE_PYTHON_FRONT_END)
    # whl 包名配置
    set(PyPTO_WhlName "pypto")

    # 安装 二进制文件
    set(InstallExport ${PROJECT_NAME}Targets)
    set(InstallTargets
            tile_fwk_utils
            tile_fwk_adapter
            tile_fwk_cann_host_runtime
            tile_fwk_platform
            tile_fwk_interface
            tile_fwk_codegen
            tile_fwk_compiler
            tile_fwk_runtime
            tile_fwk_bundle
            tile_fwk_simulation
            tile_fwk_simulation_pv
    )
    # 判断 c_sec 存在, 不存在时属异常场景, 应在构建阶段发现
    install(FILES $<TARGET_FILE:c_sec>
            DESTINATION "${PyPTO_WhlName}/lib"
            PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
            COMPONENT pypto
    )
    PTO_Fwk_InstallBinaries(
            EXPORT      ${InstallExport}
            WHL_NAME    ${PyPTO_WhlName}
            TARGETS     ${InstallTargets}
    )
    install(TARGETS pypto_impl LIBRARY DESTINATION "${PyPTO_WhlName}" COMPONENT pypto)

    # 安装 配置文件
    install(FILES
                framework/src/interface/configs/tile_fwk_config.json
                framework/src/interface/configs/tile_fwk_types.json
                framework/src/interface/configs/tile_fwk_config_schema.json
                framework/src/platform/parser/platforminfo.ini
                framework/src/platform/parser/simulation_platform/platform_config/A2A3.ini
                framework/src/platform/parser/simulation_platform/platform_config/950DT_957x.ini
                framework/src/platform/parser/simulation_platform/platform_config/950PR_957x.ini
                framework/src/platform/parser/simulation_platform/platform_config/950DT_9581.ini
                framework/src/platform/parser/simulation_platform/platform_config/950DT_9581x.ini
                framework/src/platform/parser/simulation_platform/platform_config/950DT_958x.ini
                framework/src/platform/parser/simulation_platform/platform_config/950PR_958x.ini
                framework/src/platform/parser/simulation_platform/platform_config/Kirin9030.ini
                framework/src/platform/parser/simulation_platform/platform_config/KirinX90.ini
            DESTINATION ${PyPTO_WhlName}/lib/configs
            COMPONENT pypto
    )

    # 安装 脚本文件
    install(FILES
                tools/profiling/draw_swim_lane.py
                framework/src/cost_model/simulation/scripts/print_swim_lane.py
                framework/src/cost_model/simulation/scripts/draw_comm_swim_lane_png.py
                framework/src/cost_model/simulation/scripts/draw_pipe_swim_lane.py
                tools/profiling/function_json_convert.py
                tools/profiling/parse_pipe_time_trace.py
                tools/scripts/machine_perf_trace.py
                tools/scripts/extract_pass_log.py
                tools/scripts/debug_aicore_error.py
            DESTINATION ${PyPTO_WhlName}/lib/scripts
            COMPONENT pypto
    )

    # 安装 头文件(运行时)
    # 1. tileop 目录下为 tileop 实现, 整目录安装打包;
    # 2. tilefwk 目录下为 machine 所需, 整目录安装打包;
    # 上述目录均为运行时编译 aicpu/aicore 所需二进制依赖
    install(DIRECTORY framework/src/interface/tileop/
            DESTINATION ${PyPTO_WhlName}/lib/include/tileop
            COMPONENT pypto
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )
    install(DIRECTORY framework/src/interface/machine/device/tilefwk/
            DESTINATION ${PyPTO_WhlName}/lib/include/tilefwk
            COMPONENT pypto
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )

    # 3. kernel bundle 对外 ABI 头文件
    #    libtile_fwk_bundle.so 的调用方(脱离编译前端, 仅消费 .pyptokb)需要
    install(DIRECTORY framework/include/tile_fwk_bundle/
            DESTINATION ${PyPTO_WhlName}/lib/include/tile_fwk_bundle
            COMPONENT pypto
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )

    # 安装 在线编译二进制必需文件

    # 1. cmake 配置文件
    #    在线编译 pypto_impl 场景 / 在线编译 calculator 场景 均需要
    install(FILES cmake/intf.cmake
            DESTINATION ${PyPTO_WhlName}/lib/framework/cmake
            COMPONENT pypto
    )
    install(FILES ${CANN_CMAKE_DIR}/intf_pub/intf_pub_linux.cmake
            DESTINATION ${PyPTO_WhlName}/lib/framework/cmake
            RENAME intf_pub.cmake
            COMPONENT pypto
    )

    # 2. 头文件
    # 2.1 hwSecureC 头文件 在线编译 calculator 场景 / 在线编译 pypto_impl 场景 均需要;
    # 2.2 json 头文件 在线编译 pypto_impl 场景需要;
    # 2.3 framework/include/ 路径
    #     1) 在线编译 pypto_impl 场景, 安装全量头文件;
    #     2) 在线编译 calculator 场景, 仅安装必需的头文件;
    # 2.4 framework/src/ 路径
    #     1) 仅在线编译 pypto_impl 场景需要
    set(_cSecIncludeDir $<TARGET_PROPERTY:c_sec_headers,INTERFACE_INCLUDE_DIRECTORIES>)
    install(FILES
                ${_cSecIncludeDir}/securec.h
                ${_cSecIncludeDir}/securectype.h
            DESTINATION ${PyPTO_WhlName}/lib/framework/3rd/include
            COMPONENT pypto
    )
    if (ENABLE_FEATURE_PYBIND11_IMPL_COMPILE_ONLINE)
        get_target_property(_JsonIncludeDir json INTERFACE_INCLUDE_DIRECTORIES)
        install(DIRECTORY ${_JsonIncludeDir}/nlohmann/
                DESTINATION ${PyPTO_WhlName}/lib/framework/3rd/include/nlohmann
                COMPONENT pypto
                PATTERN "*.h"
                PATTERN "*.hpp"
                PATTERN ".svn" EXCLUDE
                PATTERN ".git" EXCLUDE
        )
        install(DIRECTORY framework/include/
                DESTINATION ${PyPTO_WhlName}/lib/framework/include
                COMPONENT pypto
                PATTERN "*.h"
                PATTERN "*.hpp"
                PATTERN ".svn" EXCLUDE
                PATTERN ".git" EXCLUDE
        )
    else ()
        # 头文件(在线编译, framework/include/ 路径)
        #
        install(FILES
                    framework/include/tilefwk/data_type.h
                    framework/include/tilefwk/error.h
                    framework/include/tilefwk/error_code.h
                    framework/include/tilefwk/lazy.h
                    framework/include/tilefwk/element.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/include/tilefwk
                COMPONENT pypto
        )
    endif ()
    if (ENABLE_FEATURE_PYBIND11_IMPL_COMPILE_ONLINE)
        install(FILES
                    framework/src/adapter/api/acl_api.h
                    framework/src/adapter/api/acl_define.h
                    framework/src/adapter/api/msprof_define.h
                    framework/src/adapter/api/runtime_api.h
                    framework/src/adapter/api/runtime_capture_context.h
                    framework/src/adapter/api/runtime_define.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/adapter/api
                COMPONENT pypto
        )
        install(DIRECTORY framework/src/cost_model/simulation/
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/cost_model/simulation
                COMPONENT pypto
                PATTERN "*.h"
                PATTERN "*.hpp"
                PATTERN ".svn" EXCLUDE
                PATTERN ".git" EXCLUDE
        )
        install(FILES
                    framework/src/interface/cache/function_cache.h
                    framework/src/interface/cache/hash.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/cache
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/compiler_monitor/monitor_manager.h
                    framework/src/interface/compiler_monitor/monitor_stage_scope.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/compiler_monitor
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/configs/config_manager_ng.h
                    framework/src/interface/configs/config_manager.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/configs
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/function/function.h
                    framework/src/interface/function/rebuildable_attribute.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/function
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/inner/element.h
                    framework/src/interface/inner/function.h
                    framework/src/interface/inner/hash_buffer.h
                    framework/src/interface/inner/pre_def.h
                    framework/src/interface/inner/tilefwk.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/inner
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/interpreter/calc.h
                    framework/src/interface/interpreter/interpreter_log.h
                    framework/src/interface/interpreter/raw_tensor_data.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/interpreter
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/interpreter/calculator/calc_api.h
                    framework/src/interface/interpreter/calculator/dtype_utils.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/interpreter/calculator
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/machine/device/tilefwk/aicpu_common.h
                    framework/src/interface/machine/device/tilefwk/aicpu_perf.h
                    framework/src/interface/machine/device/tilefwk/aikernel_data.h
                    framework/src/interface/machine/device/tilefwk/aikernel_define.h
                    framework/src/interface/machine/device/tilefwk/aikernel_device_task.h
                    framework/src/interface/machine/device/tilefwk/aikernel_drco_leaf_task_ready_queue.h
                    framework/src/interface/machine/device/tilefwk/aikernel_root_function.h
                    framework/src/interface/machine/device/tilefwk/aikernel_runtime_data_ring_buffer.h
                    framework/src/interface/machine/device/tilefwk/aikernel_runtime.h
                    framework/src/interface/machine/device/tilefwk/aikernel_tensor.h
                    framework/src/interface/machine/device/tilefwk/core_func_data.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/machine/device/tilefwk
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/machine/host/perf_analysis.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/machine/host
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/operation/attr_holder.h
                    framework/src/interface/operation/attribute.h
                    framework/src/interface/operation/opcode.h
                    framework/src/interface/operation/operation_common.h
                    framework/src/interface/operation/operation.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/operation
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/program/program.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/program
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/schema/schema_base.h
                    framework/src/interface/schema/schema_def_attr.h
                    framework/src/interface/schema/schema_def_common.h
                    framework/src/interface/schema/schema_def_trace.h
                    framework/src/interface/schema/schema_parser.h
                    framework/src/interface/schema/schema.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/schema
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/tensor/float.h
                    framework/src/interface/tensor/hypercube_overlap_checker.h
                    framework/src/interface/tensor/irbuilder.h
                    framework/src/interface/tensor/logical_tensor.h
                    framework/src/interface/tensor/raw_tensor.h
                    framework/src/interface/tensor/runtime_slot.h
                    framework/src/interface/tensor/storage.h
                    framework/src/interface/tensor/symbol_handler.h
                    framework/src/interface/tensor/symbolic_scalar.h
                    framework/src/interface/tensor/tensor_offset.h
                    framework/src/interface/tensor/tensor_slot.h
                    framework/src/interface/tensor/tensormap.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/tensor
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/tileop/distributed/comm_context.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/tileop/distributed
                COMPONENT pypto
        )
        install(FILES
                    framework/src/interface/utils/common.h
                    framework/src/interface/utils/error.h
                    framework/src/interface/utils/op_info_manager.h
                    framework/src/interface/utils/string_utils.h
                    framework/src/interface/utils/enum_flags.h
                    framework/src/interface/utils/entry_registrar.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/interface/utils
                COMPONENT pypto
        )

        install(FILES
                    framework/src/machine/device/distributed/common.h
                    framework/src/machine/device/distributed/shmem_wait_until.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/device/distributed
                COMPONENT pypto
        )
        install(FILES
                    framework/src/machine/device/dynamic/aicore_constants.h
                    framework/src/machine/device/dynamic/aicpu_instrumentation.h
                    framework/src/machine/device/dynamic/costmodel_utils.h
                    framework/src/machine/device/dynamic/device_common.h
                    framework/src/machine/device/dynamic/device_perf.h
                    framework/src/machine/device/dynamic/device_trace.h
                    framework/src/machine/device/dynamic/device_utils.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/device/dynamic
                COMPONENT pypto
        )
        install(FILES
                    framework/src/machine/runtime/context/stream_context.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/runtime/context
                COMPONENT pypto
        )
        install(FILES
                    framework/src/machine/runtime/distributed/distributed_context.h
                    framework/src/machine/runtime/distributed/hccl_context.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/runtime/distributed
                COMPONENT pypto
        )
        install(FILES
                    framework/src/machine/runtime/launcher/aicore_model_launcher.h
                    framework/src/machine/runtime/launcher/cell_match_dynamic.h
                    framework/src/machine/runtime/launcher/device_launcher_binding.h
                    framework/src/machine/runtime/launcher/device_launcher_types.h
                    framework/src/machine/runtime/launcher/device_launcher.h
                    framework/src/machine/runtime/launcher/emulation_launcher.h
                    framework/src/machine/runtime/launcher/eslmodel_launcher.h
                    framework/src/machine/runtime/launcher/launcher_router.h
                    framework/src/machine/runtime/launcher/ctrl_flow_cache_manager.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/runtime/launcher
                COMPONENT pypto
        )
        install(FILES
                    framework/src/machine/runtime/memory_utils/device_memory_utils.h
                    framework/src/machine/runtime/memory_utils/emulation_memory_utils.h
                    framework/src/machine/runtime/memory_utils/eslmodel_memory_utils.h
                    framework/src/machine/runtime/memory_utils/memory_pool.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/runtime/memory_utils
                COMPONENT pypto
        )
        install(FILES
                    framework/src/machine/runtime/runner/device_perf.h
                    framework/src/machine/runtime/runner/load_aicpu_op.h
                    framework/src/machine/runtime/runner/host_prof.h
                    framework/src/machine/runtime/runner/runtime_agent.h
                    framework/src/machine/runtime/runner/runtime_utils.h
                    framework/src/machine/runtime/runner/kernel_binary.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/runtime/runner
                COMPONENT pypto
        )
        install(FILES
                    framework/src/machine/utils/device_log.h
                    framework/src/machine/utils/device_switch.h
                    framework/src/machine/utils/machine_ws_intf.h
                    framework/src/machine/utils/queues.h
                    framework/src/machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/utils
                COMPONENT pypto
        )
        install(DIRECTORY framework/src/machine/utils/dynamic/
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/machine/utils/dynamic
                COMPONENT pypto
                PATTERN "*.h"
                PATTERN "*.in"
                PATTERN ".svn" EXCLUDE
                PATTERN ".git" EXCLUDE
        )
        install(FILES
                    framework/src/passes/pass_utils/subfunc_utils.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/passes/pass_utils
                COMPONENT pypto
        )
        install(FILES
                    framework/src/utils/file_utils.h
                DESTINATION ${PyPTO_WhlName}/lib/framework/src/utils
                COMPONENT pypto
        )
    endif ()

    # 3. 源文件
    # 3.1 在线编译 calculator 二进制必需源文件, CMakeLists.txt 文件;
    # 3.2 在线编译 pypto_impl 二进制必需源文件, CMakeLists.txt 文件;
    install(DIRECTORY framework/src/interface/interpreter/calculator/
            DESTINATION ${PyPTO_WhlName}/lib/framework/src/calculator
            COMPONENT pypto
            PATTERN "*.h"
            PATTERN "*.cpp"
            PATTERN "CMakeLists.txt"
            PATTERN ".svn" EXCLUDE
            PATTERN ".git" EXCLUDE
    )

    if (ENABLE_FEATURE_PYBIND11_IMPL_COMPILE_ONLINE)
        install(DIRECTORY python/src/
                DESTINATION ${PyPTO_WhlName}/lib/pypto_impl/src
                COMPONENT pypto
                PATTERN "*.h"
                PATTERN "*.cpp"
                PATTERN "CMakeLists.txt"
                PATTERN ".svn" EXCLUDE
                PATTERN ".git" EXCLUDE
        )
    endif ()
endif ()

if (BUILD_WITH_CANN)
    add_cann_device_project(pypto)
endif ()


########################################################################################################################
# 全量源码统一构建(E)
#   — 以 CMake 为唯一入口, 一次流程产出: 二进制 → whl → run
#   — ENABLE_UNIFIED_BUILD=OFF(默认) 时完全不影响现有 build_ci.py 流程
########################################################################################################################

if (ENABLE_UNIFIED_BUILD AND ENABLE_FEATURE_PYTHON_FRONT_END)
    set(PyPTO_PkgName "pypto")
    include(version.cmake)
    add_cann_version_info_targets()

    include(cmake/package.cmake)
    set_cann_cpack_config(pypto PRE_PKG ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pre_package.cmake PACKAGE_TYPE "${PACKAGE_TYPE}")
endif ()
