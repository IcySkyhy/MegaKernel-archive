### 使用方式

1. **编译项目**
   在 `Triton-distributed-ascend/` 根目录下执行编译脚本：
   ```bash
   TRITON_BUILD_WITH_CLANG_LLD=ON TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF TRITON_PLUGIN_DIRS=${REPO_PATH}/3rdparty/triton-ascend/third_party/ascend pip install -e ./python --verbose --no-build-isolation
   ```

2. **运行示例程序**
   进入示例目录并执行运行脚本：
   ```bash
   pytest python/triton_dist/test/ascend/ -m dist
   ```
