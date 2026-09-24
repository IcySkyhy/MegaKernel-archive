# Tile Framework 分布式通信算子（Distributed OP）开发者自测试工程设计

## 路径设计

```text
.
├── ops
│   ├── include
│   ├── src
│   ├── script
│   └── CMakeLists.txt
├── framework
│   ├── include
│   ├── src
│   └── CMakeLists.txt
├── CMakeLists.txt
```


## ops目录说明
提供OP的测试用例集合，注意所有代码仅依赖前端（function/operation/etc.），不依赖运行时环境（runtime）
该目录用于UT，和仿真环境ST测试

### include目录说明
提供对外头文件，若需要添加内部专用头文件，新建inner子目录放置

### script目录说明
脚本用于生成用例参数和Golden数据，仅提供DistributedTestGolden类作为外部接口

### src目录说明
测试用例代码

## framework目录说明
提供测试框架代码实现，用于运行时初始化通信资源、拉起多卡进程等
该目录用于硬件单板环境ST测试

### include
提供对外头文件

### src
测试框架代码
