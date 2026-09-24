################################################################################
#
# Helper functions for compiling .cpp to bitcode via the bisheng (cce) compiler.
# Defined here so that all sibling subdirectories under lib/Template/ can use
# them without re-declaring.
#
# Ported from bishengir/lib/Template/CMakeLists.txt (compile_single_cpp_to_bc +
# compile_cpp_to_bc), with shmem include paths removed.
#
################################################################################

# Resolve BISHENG_TOOL / INCLUDE_PATH if bishengir's top-level CMakeLists.txt
# did not run in this project (i.e. when this file is processed by triton_dist).
if(NOT BISHENG_TOOL)
    if(DEFINED ENV{BISHENG_TOOL})
        set(BISHENG_TOOL "$ENV{BISHENG_TOOL}")
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(BISHENG_TOOL "$ENV{ASCEND_HOME_PATH}/bin/ccec")
    else()
        message(FATAL_ERROR "BISHENG_TOOL not set and ASCEND_HOME_PATH not in env. "
                            "Set ASCEND_HOME_PATH (e.g. /usr/local/Ascend/ascend-toolkit/latest) "
                            "or pass -DBISHENG_TOOL=/path/to/ccec.")
    endif()
endif()
if(NOT EXISTS "${BISHENG_TOOL}")
    message(FATAL_ERROR "BISHENG_TOOL not found: ${BISHENG_TOOL}")
endif()

if(NOT DEFINED INCLUDE_PATH)
    set(INCLUDE_PATH "${PROJECT_SOURCE_DIR}/3rdparty/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/lib/Template/include")
endif()

# Lower-level: compile one .cpp to one .bc (ported verbatim from bishengir).
function(compile_single_cpp_to_bc cpp_with_path core_type platform_type)
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})

    set(bisheng_args "-O2")
    list(APPEND bisheng_args -x cce)
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args --cce-auto-sync=off)
    list(APPEND bisheng_args --cce-aicore-only)
    list(APPEND bisheng_args --cce-generic-addrspace=off)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if ((${core_type} STREQUAL "mix_aic") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args -cce-enable-mix)
        list(APPEND bisheng_args -mllvm -enable-mix=true)
    endif ()

    if ((${core_type} STREQUAL "aic") OR (${core_type} STREQUAL "mix_aic"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-cube)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-cube)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aic")
            string(REGEX REPLACE ".cpp" ".aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aic${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif ((${core_type} STREQUAL "aiv") OR (${core_type} STREQUAL "mix_aiv"))
        if (${platform_type} STREQUAL "Ascend910D")
            list(APPEND bisheng_args --cce-aicore-arch=dav-c310-vec)
            set(bc_arch_suffix ".c310")
        else ()
            list(APPEND bisheng_args --cce-aicore-arch=dav-c220-vec)
            set(bc_arch_suffix ".c220")
        endif ()
        list(APPEND bisheng_args --cce-enable-print)
        if (${core_type} STREQUAL "aiv")
            string(REGEX REPLACE ".cpp" ".aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        else ()
            string(REGEX REPLACE ".cpp" ".mix.aiv${bc_arch_suffix}.bc" bc_name ${cpp_name})
        endif ()
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
    else ()
        message(FATAL_ERROR "Core type must be aic/mix_aic/aiv/mix_aiv or host.")
    endif ()

    if (${core_type} STREQUAL "host")
        message(STATUS "Running on host platform.")
    else ()
        if (NOT ${platform_type} STREQUAL "Ascend910B" AND
            NOT ${platform_type} STREQUAL "Ascend910D")
            message(FATAL_ERROR
                    "Unsupported platform! Currently support Ascend910B and Ascend910D.")
        endif ()
    endif ()

    list(APPEND bisheng_args -std=c++17)
    if ("${platform_type}" STREQUAL "Ascend910D")
        list(APPEND bisheng_args -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif ()
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/)
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl)
    list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
        list(APPEND bisheng_args -I $ENV{ASCEND_HOME_PATH}/x86_64-linux/include/)
    endif ()
    list(APPEND bisheng_args -o ${bc_name})

    if(CMAKE_SYSROOT)
        list(APPEND bisheng_args --sysroot=${CMAKE_SYSROOT})
    endif()

    string(REGEX REPLACE ";" " " compile_command "${BISHENG_TOOL} ${bisheng_args}")
    message("-- command of compile cpp to bc is ${compile_command}")

    execute_process(
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${BISHENG_TOOL} ${bisheng_args}
    )

    if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
        message(FATAL_ERROR "Compile ${cpp_with_path} failed, please check "
                "${CMAKE_CURRENT_BINARY_DIR}/${bc_name} exist or not!")
    endif ()
endfunction()

# Higher-level wrapper: auto-glob *.cpp in this dir and compile each (ported
# verbatim from bishengir). Usage: compile_cpp_to_bc(<core_type> <platform_type>
# [cpp_file ...]); when no cpp files are listed, globs *.cpp in this dir.
function(compile_cpp_to_bc)
    set (core_type " ")
    set (platform_type " ")

    set(core_type ${ARGV0})
    if(NOT ${ARGV0} STREQUAL "host")
        set(platform_type ${ARGV1})
    endif ()

    if (ARGC GREATER 2)
        list(SUBLIST ARGV 2 -1 cpp_list)
        set(explicit_cpp_list)
        foreach (cpp_source ${cpp_list})
            if (IS_ABSOLUTE "${cpp_source}")
                list(APPEND explicit_cpp_list "${cpp_source}")
            else ()
                list(APPEND explicit_cpp_list
                     "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_source}")
            endif ()
        endforeach ()
        set(cpp_list ${explicit_cpp_list})
    else ()
        file(GLOB cpp_list "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    endif ()
    foreach (cpp_with_path ${cpp_list})
        if (ENABLE_CPU_TRACE_INTRINSIC AND
            (NOT ${core_type} STREQUAL "mix_aic") AND
            (NOT ${core_type} STREQUAL "mix_aiv"))
            message(FATAL_ERROR "compile_single_cpp_to_bc_debug not available in triton_dist copy.")
        else ()
            compile_single_cpp_to_bc(${cpp_with_path} ${core_type} ${platform_type})
        endif ()
    endforeach ()
endfunction()

add_subdirectory(Megakernel)
