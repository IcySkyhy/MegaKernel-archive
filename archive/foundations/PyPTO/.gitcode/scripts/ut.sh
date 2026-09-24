#!/bin/bash
set -euo pipefail

export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version
rm -rf /home/jenkins/opensource/json

python3 -m pip install build
python3 -m pip install --upgrade packaging==24.2
python3 -m pip install --upgrade pytest-xdist pytest-forked
sudo apt install -y ninja-build
apt install -y libclang-rt-15-dev

if [ "${GIT_TARGET_BRANCH}" = "br_0.1.1_20260313_beta" ]; then
    wget https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/br_0313/package/cann-pto-isa_9.0.0_linux-x86_64.run
elif [ "${GIT_TARGET_BRANCH}" = "0.2.0" ]; then
    wget https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/package/pto/9.0.0/20260330/x86_64/cann-pto-isa_9.0.0_linux-x86_64.run
elif [ "${GIT_TARGET_BRANCH}" = "9.1.0" ]; then
    wget https://opencann-obs.obs.cn-north-4.myhuaweicloud.com/pto-isa/9.1.0/cann-pto-isa_linux-x86_64.run
else
    wget https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-x86_64.run
fi
chmod +x *.run
echo "y" | su - jenkins -c "cd ${WORKSPACE} && bash *.run --full --quiet --install-path=/home/jenkins/Ascend"
source /home/jenkins/Ascend/cann/bin/setenv.bash

set +e
check_ret() {
    if [ $? -ne 0 ]; then
        echo "$1"
        exit 1
    fi
}

case "${GE_ST_RT2}" in
    Py3_ninja_simulation)
        python3 build_ci.py --clean --plat_name=manylinux2014 --timeout=360 --no_isolation --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --compile_dependency_check --verbose
        check_ret "PyPTO(Py3-Simulation) build whl failed"
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            bash build_out/cann-pypto_*.run --full -q --pylocal --install-path=./build_out
            check_ret "PyPTO(Py3-Simulation) install run package failed"
            export PYTHONPATH=./build_out/cann/python/site-packages:$PYTHONPATH
        else
            chmod +x build_out/*.whl
            pip install build_out/*.whl
        fi
        python3 python/tests/ut/simulator/costmodel_cpu_swimlane.py
        check_ret "PyPTO(Py3-Simulation) build and run cann UTest failed"
        rm -rf /home/jenkins/Ascend/cann
        unset ASCEND_HOME_PATH
        rm -rf output
        python3 python/tests/ut/simulator/costmodel_cpu_swimlane.py
        check_ret "PyPTO(Py3-Simulation) build with cann and run with uncann UTest failed"
        rm -rf build_out output
        python3 build_ci.py --clean --plat_name=manylinux2014 --timeout=300 --no_isolation --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --compile_dependency_check --verbose
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            bash build_out/cann-pypto_*.run --full -q --pylocal --install-path=./build_out
            check_ret "PyPTO(Py3-Simulation) install run package failed"
            export PYTHONPATH=./build_out/cann/python/site-packages:$PYTHONPATH
        else
            chmod +x build_out/*.whl
            pip install build_out/*.whl
        fi
        python3 python/tests/ut/simulator/costmodel_cpu_swimlane.py
        check_ret "PyPTO(Py3-Simulation) build with uncann and run with uncann UTest failed"
        ;;
    Py3_ninja)
        python3 build_ci.py --clean --generator=Ninja '--utest=python/tests/ut --ignore=python/tests/ut/kirin' --py_abi=37 --case_execute_timeout=90 --changed_files=${WORKSPACE}/pr_filelist.txt --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --verbose --editable --no_isolation --gcov --utest_module=ds_v32:interface:ir:kirin:operation:operator:pypto_pro:simulator
        check_ret "PyPTO(Py3) UTest failed"
        rm -rf python/pypto/pypto_impl*.so
        python3 -c "import pypto"
        check_ret "import pypto failed"
        ;;
    Cpp_make_clang)
        python3 build_ci.py --clean --frontend=cpp --build_type=Debug --utest --case_execute_timeout=90 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --clang --asan --job_num=32 --target=tile_fwk_utest
        check_ret "Run PyPTO(Cpp-Clang) UTest failed"
        ;;
    make_clang1)
        python3 build_ci.py --clean --frontend=cpp --build_type=Debug --utest --case_execute_timeout=90 --utest_module=machine:simulation:passes --clang --asan --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=32 --target=tile_fwk_utest
        check_ret "Run PyPTO(Cpp-Clang-1x) UTest failed"
        ;;
    make_clang2)
        python3 build_ci.py --clean --frontend=cpp --build_type=Debug --utest --case_execute_timeout=90 --utest_module=interface --clang --asan --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=32 --target=tile_fwk_utest
        check_ret "Run PyPTO(Cpp-Clang-2x) UTest failed"
        ;;
    make_clang3)
        python3 build_ci.py --clean --frontend=cpp --build_type=Debug --utest --case_execute_timeout=90 --utest_module=codegen:operator --clang --asan --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=32 --target=tile_fwk_utest
        check_ret "Run PyPTO(Cpp-Clang-3x) UTest failed"
        ;;
    make_gnu_1)
        python3 build_ci.py --clean --frontend=cpp --timeout=420 --build_type=Release --utest --case_execute_timeout=90 --utest_module=machine:simulation:passes --gcov --changed_files=${WORKSPACE}/pr_filelist.txt --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --target=tile_fwk_utest
        check_ret "PyPTO(cpp) UTest failed"
        ;;
    make_gnu_2)
        python3 build_ci.py --clean --frontend=cpp --timeout=420 --build_type=Release --utest --case_execute_timeout=90 --utest_module=interface --gcov --changed_files=${WORKSPACE}/pr_filelist.txt --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --target=tile_fwk_utest
        check_ret "PyPTO(cpp) UTest failed"
        ;;
    make_gnu_3)
        python3 build_ci.py --clean --frontend=cpp --timeout=420 --build_type=Release --utest --case_execute_timeout=90 --utest_module=codegen:operator --gcov --changed_files=${WORKSPACE}/pr_filelist.txt --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16 --target=tile_fwk_utest
        check_ret "PyPTO(cpp) UTest failed"
        ;;
    kirinx90)
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0
        export LD_LIBRARY_PATH=/home/jenkins/Ascend/ascend-toolkit/latest/x86_64-linux/simulator/KirinX90/lib:$LD_LIBRARY_PATH
        python3 build_ci.py --utest=python/tests/ut/kirin/kirinx90  --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16
        check_ret "kirinx90 UTest failed"
        ;;
    kirin9030)
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0
        export LD_LIBRARY_PATH=/home/jenkins/Ascend/ascend-toolkit/latest/x86_64-linux/simulator/Kirin9030/lib:$LD_LIBRARY_PATH
        python3 build_ci.py --utest=python/tests/ut/kirin/kirin9030 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} --job_num=16
        check_ret "kirin9030 UTest failed"
        ;;
    *)
        echo "Unknown ut_type: ${GE_ST_RT2}"
        exit 1
        ;;
esac

coverage_info=$(find ${WORKSPACE} -name "coverage_filtered.info" 2>/dev/null | head -n1)
if [ -n "${coverage_info}" ]; then
    lcov --list ${coverage_info} 2>/dev/null || true
    if [[ "${GE_ST_RT2}" =~ make_gnu.* ]] || [[ "${GE_ST_RT2}" = "Py3_ninja" ]]; then
        mv "${coverage_info}" "${WORKSPACE}/coverage_${GE_ST_RT2}.info"
        echo "ut_process=coverage" >> "${ATOMGIT_OUTPUT}"
    fi
else
    echo "no coverage_filtered.info found, skip coverage packaging"
fi
/usr/local/ccache/bin/ccache -s
exit 0
