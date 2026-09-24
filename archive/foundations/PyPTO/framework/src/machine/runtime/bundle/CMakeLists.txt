# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# libtile_fwk_bundle.so -- the standalone .pyptokb consumer ABI (PyptoWorkspace / PyptoLaunch [+FromMemory]).
#
# Only the LOAD/QUERY/LAUNCH half of the bundle code lives in THIS directory. The PACK half lives one level down
# in pack/ and stays in tile_fwk_runtime, because its hooks are called from the runtime's own emulation/launch
# paths; splitting on that boundary is what keeps the two targets acyclic.
#
# The glob is deliberately non-recursive so pack/ is excluded by directory rather than by a file list: the
# runtime's own CMakeLists.txt removes ${CMAKE_CURRENT_SOURCE_DIR}/bundle/*.cpp from its recursive glob with the
# same pattern, so the two targets stay in sync as files are added on either side.
file(GLOB _TileFwkBundle_Srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Export only the four C entry points; everything else (registry/loader/dev-cache internals) stays local so the
# library presents exactly the documented ABI surface. Shared by both flavours below.
set(_TileFwkBundle_VerScript ${CMAKE_CURRENT_SOURCE_DIR}/tile_fwk_bundle.version)

add_library(tile_fwk_bundle SHARED)
target_sources(tile_fwk_bundle PRIVATE ${_TileFwkBundle_Srcs})
target_compile_definitions(tile_fwk_bundle
        PUBLIC
            $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>>
)
target_link_libraries(tile_fwk_bundle
        PRIVATE
            -Wl,--no-as-needed
            tile_fwk_runtime
            tile_fwk_interface
            tile_fwk_codegen
            tile_fwk_compiler
            tile_fwk_adapter
            tile_fwk_platform
            tile_fwk_cann_host_runtime
            tile_fwk_utils
            -Wl,--as-needed
            $<BUILD_INTERFACE:json>
            tile_fwk_intf_pub
            dl
            c_sec
)
target_link_options(tile_fwk_bundle PRIVATE "-Wl,--version-script=${_TileFwkBundle_VerScript}")
set_target_properties(tile_fwk_bundle PROPERTIES LINK_DEPENDS ${_TileFwkBundle_VerScript})

PTO_Fwk_AnalysisTargetHeaderFiles(TARGET tile_fwk_bundle)
PTO_Fwk_AnalysisTargetSymbols(TARGET tile_fwk_bundle)

# ---------------------------------------------------------------------------------------------------------------
# Optional: libtile_fwk_bundle_standalone.so -- same four symbols, but self-contained.
#
# WHY: the plain tile_fwk_bundle above still needs libtile_fwk_runtime.so (and transitively interface/adapter/
# platform/utils) present on the target machine. This flavour statically absorbs that whole closure so a consumer
# that only runs .pyptokb files can ship ONE .so. CANN itself is never absorbed -- it is dlopen'd by name at
# runtime (adapter/manager/plugin_handler.cpp), so it never appears in DT_NEEDED either way.
#
# OFF by default ON PURPOSE: it recompiles ~195 upstream sources into its own objects (they cannot be shared,
# because sharing them would mean editing the upstream CMakeLists outside this directory), which roughly doubles
# a full build. Turn it on only when producing a delivery artifact:
#     cmake -DENABLE_TILE_FWK_BUNDLE_STANDALONE=ON ...
#
# CONSTRAINT: this library must NOT be loaded into a process that also loads the regular pypto stack. Both would
# carry their own copy of every Meyers singleton on the closure (Platform::Instance, DeviceRunner::Get,
# ConfigManager::Instance, OpInfoManager, plus the bundle's own registry/dev-cache), so state set through one
# would be invisible to the other -- silent divergence, not a crash. Pure .pyptokb consumers satisfy this
# naturally; anything that also drives the Python frontend must use the plain tile_fwk_bundle instead.
# ---------------------------------------------------------------------------------------------------------------
option(ENABLE_TILE_FWK_BUNDLE_STANDALONE "Build the self-contained libtile_fwk_bundle_standalone.so" OFF)

if (ENABLE_TILE_FWK_BUNDLE_STANDALONE)
    set(_FwkRoot ${PTO_FWK_SRC_ROOT}/framework/src)

    # The upstream source globs are directory-scoped variables, so they are not visible here; they are re-globbed
    # with the SAME patterns as the upstream CMakeLists. Keeping the patterns (rather than a file list) means new
    # upstream files are picked up automatically. The upstream per-target compile definitions / include dirs are
    # NOT visible either and are restated below -- if upstream changes those, this target must be updated to match.
    file(GLOB _SA_Interface_Srcs
            ${_FwkRoot}/interface/compiler_monitor/*.cpp
            ${_FwkRoot}/interface/core/*.cpp
            ${_FwkRoot}/interface/ir/*.cpp
            ${_FwkRoot}/interface/ir/op/*.cpp
            ${_FwkRoot}/interface/ir/op/*/*.cpp
            ${_FwkRoot}/interface/ir/reporter/*.cpp
            ${_FwkRoot}/interface/ir/transforms/*.cpp
            ${_FwkRoot}/interface/ir/transforms/utils/*.cpp
            ${_FwkRoot}/interface/ir/verifier/*.cpp
            ${_FwkRoot}/interface/schema/*.cpp
            ${_FwkRoot}/interface/configs/*.cpp
            ${_FwkRoot}/interface/tensor/*.cpp
            ${_FwkRoot}/interface/program/*.cpp
            ${_FwkRoot}/interface/function/*.cpp
            ${_FwkRoot}/interface/interpreter/*.cpp
            ${_FwkRoot}/interface/utils/*.cpp
            ${_FwkRoot}/interface/plugin/*.cpp
            ${_FwkRoot}/interface/operation/*.cpp
            ${_FwkRoot}/interface/operation/vector/*.cpp
            ${_FwkRoot}/interface/operation/distributed/*.cpp
            ${_FwkRoot}/interface/cache/*.cpp
            ${_FwkRoot}/interface/machine/*.cpp
            ${_FwkRoot}/interface/machine/host/*.cpp
            ${_FwkRoot}/interface/registry/*.cpp
    )
    file(GLOB_RECURSE _SA_PyptoPro_Srcs
            ${_FwkRoot}/interface/pypto_pro/backend/*.cpp
            ${_FwkRoot}/interface/pypto_pro/codegen/*.cpp
    )
    # framework/src/passes lives in its own directory but injects itself into tile_fwk_interface
    # (passes/CMakeLists.txt: target_sources(tile_fwk_interface ...)), so it is part of the interface library
    # even though it is outside interface/. Missing it shows up as undefined CommonUtils::* at dlopen time.
    file(GLOB_RECURSE _SA_Passes_Srcs ${_FwkRoot}/passes/*.cpp)
    file(GLOB_RECURSE _SA_Utils_Srcs
            ${_FwkRoot}/utils/host_log/*.cpp
            ${_FwkRoot}/utils/error_manager/*.cpp
            ${_FwkRoot}/utils/file_utils.cpp
    )
    file(GLOB _SA_Platform_Srcs
            ${_FwkRoot}/platform/*.cpp
            ${_FwkRoot}/platform/parser/*.cpp
            ${_FwkRoot}/platform/parser/simulation_platform/*.cpp
    )
    file(GLOB_RECURSE _SA_Adapter_Srcs
            ${_FwkRoot}/adapter/api/*.cpp
            ${_FwkRoot}/adapter/manager/*.cpp
            ${_FwkRoot}/adapter/stubs/*.cpp
    )
    file(GLOB _SA_CannHostRt_Srcs ${_FwkRoot}/cann_host_runtime/*.cpp)

    # Runtime half: everything tile_fwk_runtime builds, plus the bundle load-side sources, plus the two
    # dev_encode TUs that only libtile_fwk_compiler exports (DevControlFlowCache::Init and
    # DevAscendProgram::GetCtrlFlowCacheSlottedOutcastBlockCount). Pulling those two files in directly -- rather
    # than moving them out of the compiler glob upstream -- is what keeps this change inside bundle/.
    file(GLOB_RECURSE _SA_Runtime_Srcs ${_FwkRoot}/machine/runtime/*.cpp)
    list(REMOVE_ITEM _SA_Runtime_Srcs ${_TileFwkBundle_Srcs})
    file(GLOB_RECURSE _SA_Device_Srcs ${_FwkRoot}/machine/device/*.cpp)
    file(GLOB_RECURSE _SA_Simulation_Srcs ${_FwkRoot}/machine/simulation/*.cpp)
    file(GLOB _SA_DevUtils_Srcs
            ${_FwkRoot}/machine/utils/dynamic/dev_encode_function.cpp
            ${_FwkRoot}/machine/utils/dynamic/dev_encode_function_dupped_data.cpp
            ${_FwkRoot}/machine/utils/dynamic/dev_encode_program.cpp
            ${_FwkRoot}/machine/utils/dynamic/dev_cell_match_dump.cpp
            ${_FwkRoot}/machine/utils/dynamic/dev_encode_program_ctrlflow_cache.cpp
            ${_FwkRoot}/machine/utils/dynamic/dev_workspace.cpp
            ${_FwkRoot}/machine/utils/dynamic/dev_encode.cpp
            ${_FwkRoot}/machine/utils/dynamic/dev_encode_workspace.cpp
    )

    add_library(tile_fwk_bundle_standalone SHARED)
    target_sources(tile_fwk_bundle_standalone PRIVATE
            ${_TileFwkBundle_Srcs}
            ${_SA_Interface_Srcs} ${_SA_PyptoPro_Srcs} ${_SA_Passes_Srcs}
            ${_SA_Utils_Srcs} ${_SA_Platform_Srcs} ${_SA_Adapter_Srcs} ${_SA_CannHostRt_Srcs}
            ${_SA_Runtime_Srcs} ${_SA_Device_Srcs} ${_SA_Simulation_Srcs} ${_SA_DevUtils_Srcs}
    )
    set_target_properties(tile_fwk_bundle_standalone PROPERTIES OUTPUT_NAME tile_fwk_bundle_standalone)

    # Include dirs / definitions restated from the upstream targets (see note above the globs).
    target_include_directories(tile_fwk_bundle_standalone
            PRIVATE
                ${_FwkRoot}/interface
                ${_FwkRoot}/utils
                ${_FwkRoot}/adapter
                ${_FwkRoot}/cost_model
                ${PTO_FWK_SRC_ROOT}/framework/include/pypto_pro
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
    )
    target_compile_definitions(tile_fwk_bundle_standalone
            PRIVATE
                $<$<BOOL:${ENABLE_TESTS}>:ENABLE_TESTS>
                $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_STEST_INTERPRETER_CONFIG}>:ENABLE_STEST_INTERPRETER_CONFIG>>
                $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_WITH_CANN}>,$<NOT:$<BOOL:${ENABLE_UTEST}>>>:BUILD_WITH_CANN>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN_MOBILE}>:BUILD_WITH_CANN_MOBILE>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:PROCESSOR_SUBPATH="${CMAKE_SYSTEM_PROCESSOR}-linux">>
    )
    target_link_directories(tile_fwk_bundle_standalone
            PRIVATE
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/lib64>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:/usr/local/Ascend/driver/lib64/driver>>
                $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib>>
    )
    # No pypto libraries here on purpose -- only the header-only / INTERFACE deps and system libs.
    # c_sec is deliberately absent: it is handled just below so the shared flavour stays out of DT_NEEDED.
    target_link_libraries(tile_fwk_bundle_standalone
            PRIVATE
                $<BUILD_INTERFACE:json>
                tile_fwk_intf_pub
                dl rt
                Threads::Threads
    )
    # c_sec normally resolves to a shared libc_sec.so, which would be the one pypto-side runtime dependency left
    # in DT_NEEDED. Absorb the static archive instead when one is available, so the delivered .so needs nothing
    # from the pypto/CANN lib directory at load time. Falls back to the shared flavour when absent.
    #
    # `c_sec` is an IMPORTED SHARED target, so linking it puts libc_sec.so back into DT_NEEDED. Take the include
    # dir only -- via the header-only INTERFACE target `c_sec_headers` declared by cann-cmake's Findsecurec.cmake
    # -- and link the archive explicitly. (Guarded: the target name belongs to cann-cmake, not to us.)
    find_library(_TileFwkBundle_CSecStatic
            NAMES libc_sec.a
            PATHS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
                  ${ASCEND_CANN_PACKAGE_PATH}/lib64
                  ${PYPTO_THIRD_PARTY_PATH}/${CMAKE_BUILD_TYPE}/lib
            NO_DEFAULT_PATH
    )
    if (_TileFwkBundle_CSecStatic)
        if (TARGET c_sec_headers)
            target_link_libraries(tile_fwk_bundle_standalone PRIVATE c_sec_headers)
        else ()
            # Older cann-cmake without the header-only target: fall back to the include dir off `c_sec` itself.
            get_target_property(_CSecIncDir c_sec INTERFACE_INCLUDE_DIRECTORIES)
            target_include_directories(tile_fwk_bundle_standalone PRIVATE ${_CSecIncDir})
        endif ()
        target_link_libraries(tile_fwk_bundle_standalone PRIVATE ${_TileFwkBundle_CSecStatic})
        # Do not re-export the absorbed archive's symbols from this .so; only the four ABI entries are public.
        target_link_options(tile_fwk_bundle_standalone PRIVATE "-Wl,--exclude-libs,ALL")
        message(STATUS "tile_fwk_bundle_standalone: absorbing static c_sec ${_TileFwkBundle_CSecStatic}")
    else ()
        target_link_libraries(tile_fwk_bundle_standalone PRIVATE c_sec)
        message(STATUS "tile_fwk_bundle_standalone: no static c_sec found; libc_sec.so stays a runtime dependency")
    endif ()
    # Section-level GC: the closure touches well under 1% of the interface library's exported symbols, so most of
    # what gets compiled in is dead weight without this.
    target_compile_options(tile_fwk_bundle_standalone PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(tile_fwk_bundle_standalone PRIVATE
            "-Wl,--gc-sections"
            "-Wl,--version-script=${_TileFwkBundle_VerScript}"
            # Fail at LINK time on any unresolved symbol. Without this a source-glob gap (e.g. a directory that
            # injects itself into an upstream target from elsewhere) links cleanly and only surfaces as an
            # "undefined symbol" OSError when the consumer dlopen's the delivered .so.
            "-Wl,--no-undefined"
    )
    set_target_properties(tile_fwk_bundle_standalone PROPERTIES LINK_DEPENDS ${_TileFwkBundle_VerScript})

    # The aicpu backend .so (tile_fwk_server) is built by the separate `cann_device` ExternalProject
    # (cmake/device/CMakeLists.txt), so it is NOT a target in this host tree -- an unguarded
    # add_dependencies() here fails configure outright with "dependency target does not exist".
    # Depend on it only when it really is a host target; the device build is otherwise driven by its own entry.
    if (BUILD_WITH_CANN AND TARGET tile_fwk_server)
        add_dependencies(tile_fwk_bundle_standalone tile_fwk_server)
    endif ()
endif ()
