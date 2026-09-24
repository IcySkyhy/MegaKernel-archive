# CANN包不兼容

## 问题现象描述

算子上板执行时出现如下报错字样：

```text
ErrorTracking callback in, task_id = 0, stream_id = 3.
[ERROR] Exception Type: exception invalid error
taskid: 0, streamid: 3, tid: 6495, deviceid: 0, retcode: 507018
kernelName = (null)
ErrorTracking callback in, task_id = 1, stream_id = 3.
[ERROR] Exception Type: exception invalid error
taskid: 1, streamid: 3, tid: 6495, deviceid: 0, retcode: 507018
kernelName = (null)
```

且device日志中出现类似如下接口为空报错：

```text
~/ascend/log/debug/device-0/device-6495_20251222194004973.log
[ERROR] CCECPU(5670,aicpu_scheduler):2025-12-22-19:40:01.899.541 [ae_kernel_lib_aicpu_kfc.cpp:105][CallKernelApi][tid:5680][AICPU_PROCESSER] Get KFC DynTileFwkKernelServerInit api success, but func is nullptr: (null)
[ERROR] CCECPU(5670,aicpu_scheduler):2025-12-22-19:40:01.902.745 [ae_kernel_lib_aicpu_kfc.cpp:105][CallKernelApi][tid:5681][AICPU_PROCESSER] Get KFC DynTileFwkKernelServer api success, but func is nullptr: (null)
```

## 原因分析

PyPTO驱动包支持25.2.0以上版本，CANN包支持8.5.0以上版本。

## 解决措施

可以查看驱动包安装目录下的version信息，如：

```text
/usr/local/Ascend/driver/version.info
    Version=25.3.rc1
    ascendhal_version=7.35.23
    aicpu_version=1.0
    tdt_version=1.0
    log_version=1.0
    prof_version=2.0
    dvppkernels_version=1.1
    tsfw_version=1.0
    Innerversion=V100R001C23SPC002B212
    compatible_version=[V100R001C19],[V100R001C20],[V100R001C21],[V100R001C22],[V100R001C23]
    compatible_version_fw=[7.0.0,8.9.9]
    package_version=25.3.rc1
```

同样可以查看CANN包安装目录下opp包内的version信息，如：

```text
/usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    Version=8.5.0.2.220
    version_dir=8.5.0
    timestamp=20251117_000024591
    required_package_amct_acl_version="8.5"
```

按上述方法查看驱动包和CANN包是否满足版本要求，如不满足请升级对应版本。
