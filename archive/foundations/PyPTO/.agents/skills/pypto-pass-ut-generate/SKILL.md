---
name: pypto-pass-ut-generate
description: 根据 Pass 业务描述，生成单元测试用例（UT）。当用户输入业务情况时，能根据业务，生成对应 Pass 的 UT 用例。
---

# Pass 业务单元测试生成

## 概述

本技能用于分析如何根据用户描述生成对应的 Pass 侧单元测试用例（UT），结合调用 `pypto-pass-module-analyzer` 技能分析 Pass 业务，帮助设计相关单元测试用例（UT）。

## 目录结构

```
pypto-pass-ut-generate/
├── SKILL.md                  # 主技能文档
├── scripts/                   # 脚本工具目录
│   ├── common_utils.py       # 公共工具函数
│   ├── pr_utils.py           # PR 处理工具（完整功能）
│   ├── get_ut_status.py    # 简化版 UT 状态获取
│   └── ut_coverage.py        # 覆盖率分析工具
└── references/
    ├── check_list.md         # 检查清单
    ├── usage.md              # 使用指南
    ├── trouble_shooting.md   # 故障排查
    └── common_errors.md      # 常见错误记录
```

## 触发机制

当用户输入包含以下关键字或相关内容时，自动触发此技能：

| 触发词 | 说明 |
|--------|------|
| "设计 Pass 模块 XXX 的 UT 用例" | 设计指定 Pass 模块的功能的 UT 用例 |
| "设计 Pass 模块 XXX 的 XXX 功能" | 设计指定 Pass 模块的指定功能的 UT 用例 |
| "设计 XXX 功能的相关 Pass 模块的 UT 用例" | 设计与 XXX 功能相关 Pass 的 UT 用例 |
| "为PR XXXX补充UT" | 为指定 PR 补充 UT 测试用例（在线） |
| "分析覆盖率报告" | 分析 UT 覆盖率报告，补充未覆盖代码（离线） |
| "分析本地diff文件" | 分析用户指定的本地 diff 文件（离线） |
| "分析本地覆盖率文件" | 分析用户指定的本地覆盖率 .tar.gz 文件（离线） |
| "分析本地文件" | 同时分析本地 diff 和覆盖率文件（离线） |

**触发示例**：
- "设计 Pass 模块 AutoCast 的 UT 用例"
- "设计 Pass 模块 AutoCast 的对于不支持 BF16 OP 插入 Cast 的功能"
- "设计删除冗余 Op 功能的相关 Pass 的 UT 用例"
- "为PR 1894补充UT"
- "分析覆盖率报告，补充未覆盖代码"
- "分析本地 /path/to/ut_cov.tar.gz 文件"
- "分析本地 diff 文件 /path/to/pr.diff"

## 使用场景

本技能支持以下 5 种使用场景：

### 场景一：为指定 Pass 生成完整 UT（从零开始）

**适用情况**：需要为某个 Pass 设计完整的 UT 测试套件

**执行步骤**：
1. 分析 Pass 业务逻辑（参考 pypto-pass-module-analyzer/SKILL.md）
2. 在 `pypto/framework/tests/ut/passes/src/test_xxx.cpp` 创建测试文件
3. 按照 UT 生成流程一或流程二编写测试用例
4. 执行 `python3 build_ci.py -c -u=TestPassName.* -j=24 -f=cpp` 验证
5. 运行 `python3 build_ci.py -c -u=TestPassName.* --gcov -j=24 -f=cpp` 检查覆盖率

### 场景二：为 PR 补充 UT

**适用情况**：PR 已有部分 UT，需要补充未覆盖的代码

**执行步骤**：
1. 使用 `scripts/get_ut_status.py` 快速获取 UT 状态
2. 使用 `scripts/pr_utils.py` 处理 PR，获取 diff 和覆盖率报告
3. 分析未覆盖的代码行
4. 针对未覆盖行设计 UT 用例
5. 验证编译和覆盖率

**示例**：
```bash
# 快速获取 UT 状态
python3 scripts/get_ut_status.py 2017

# 完整处理 PR
python3 scripts/pr_utils.py 2017
```

### 场景三：离线分析本地 diff 文件

**适用情况**：用户指定本地 diff 文件，需要离线分析

**执行步骤**：
1. 使用 `scripts/ut_coverage.py` 解析用户指定的本地 diff 文件
2. 提取变更的 Pass 文件和代码行
3. 分析变更代码的业务逻辑
4. 生成 UT 设计建议
5. 手动实现测试用例

**示例**：
```bash
# 解析本地 diff 文件
python3 scripts/ut_coverage.py --diff /path/to/pr.diff

# 简短写法（文件在当前目录）
python3 scripts/ut_coverage.py --diff pr.diff
```

### 场景四：离线分析本地覆盖率报告

**适用情况**：用户指定本地覆盖率报告（.tar.gz 格式），需要分析未覆盖代码

**执行步骤**：
1. 解析用户指定的本地覆盖率报告文件（支持 .tar.gz 格式）
2. 提取未覆盖的代码行
3. 针对未覆盖行分析业务逻辑
4. 设计测试用例覆盖这些行

**示例**：
```bash
# 解析本地覆盖率报告
python3 scripts/ut_coverage.py --report /path/to/ut_cov.tar.gz

# 简短写法（文件在当前目录）
python3 scripts/ut_coverage.py --report ut_cov.tar.gz
```

### 场景五：离线综合分析

**适用情况**：用户同时指定本地 diff 文件和覆盖率报告，需要关联分析

**执行步骤**：
1. 解析用户指定的本地 diff 文件，获取变更的 Pass 文件
2. 解析用户指定的本地覆盖率报告，获取未覆盖行
3. 关联 diff 和覆盖率，识别需要补充 UT 的代码
4. 生成 UT 设计建议

**示例**：
```bash
# 综合分析（同时指定 diff 和覆盖率文件）
python3 scripts/ut_coverage.py --diff /path/to/pr.diff --report /path/to/ut_cov.tar.gz

# 输出 JSON 格式建议
python3 scripts/ut_coverage.py --diff pr.diff --report ut_cov.tar.gz --json
```

## UT 生成流程一

### 步骤 1：分析业务

    根据用户描述的业务情况，分析相关业务：
        （1）当描述为具体 Pass 的具体业务时，根据 pypto-pass-module-analyzer，分析业务场景，根据当前业务设计相应的 UT 用例，例如：设计 ProcessAtomic 这个 Pass 消除 RedunceAcc 功能的 UT；
        （2）当描述为模糊 Pass 业务时，根据 pypto-pass-module-analyzer，进行相关 Pass 业务总结，挑选出符合业务的 Pass，分析业务场景，设计对应的 UT 用例，例如：请针对 Pass 中对于视图类 Op，view、assemble 处理的 Pass 设计对应的 UT，验证功能实现；
        （3）当描述为设计 Pass 的 UT 时，根据 pypto-pass-module-analyzer，进行相关 Pass 业务总结，分析业务场景，设计该 Pass 业务的 UT 用例，例如：针对 ProcessAtomic 这个 Pass 设计相关 UT 用例;
         (4) 当描述为设计 PR xxxx 的 UT 或者提供 diff 文件、ut 覆盖率报告时，解析变更变更代码或者未覆盖到的代码行，进行相关 Pass 业务总结，分析业务场景，设计该 Pass 业务的 UT 用例，例如：针对 PR01 设计相关 UT 用例或者针对 01.diff 文件生成 UT 用例或者针对 ut-report.tar.gz 生成 UT 用例;

### 步骤 2：环境配置

    在 pypto/framework/tests/ut/passes/src/test_xxx.cpp 寻找对应的测试文件，观察当前文件中，是否初始化环境，若已完成初始化，则跳过该步骤。其中 xxx 一般为 Pass 名称，
    当未找到对应的测试文件时，可以查找 test_xxx.cpp 中创建类的名字是否与所要求设计 Pass 名字是否一致。
    初始化环境详细步骤：
        （1）创建新的类，命名 xxx.cpp，文件名为 test_xxx.cpp，其中 xxx 为输入的 pass 名称
        （2）声明该类继承于 gtest 框架
        （3）该类中，编写相关函数：
                1.所有测试用例全局初始化函数--static void SetUpTestCase() {}，若未明确指定内容，则为空实现；
                2.所有测试用例全局清理函数--static void TearDownTestCase() {}，  若未明确指定内容，则为空实现；
                3.每个测试用例执行前的初始化测试环境函数--void SetUp() override {}，若未指定内容，则默认生成以下代码体；
```cpp
                    void SetUp() override {
                        Program::GetInstance().Reset();
                        config::Reset();
                        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH); // 其中 pypto/framework/src/interface/configs/config_manager_ng.h中的COMPILE_STAGE策略，通过pass所在文件夹目录，得到所处的编译策略
                        config::SetHostConfig(KEY_STRATEGY, "XXXTestStrategy"); // 其中xxx为pass名字，表示host侧KEY_STRATEGY
                        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false); // 其中 pypto/framework/src/interface/configs/config_manager.h中的Platform KEYs表示平台策略，需要根据用户传入的进行修改，若未传入，则采用该默认方式
                        TileShape::Current().SetVecTile({64, 64}); // 设置vector的tile块大小
                        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64}); // 设置cube的tile块大小
                    }
```
                4.每个测试用例执行后的清理测试环境函数--void TearDown() override {}，若为明确指定内容，则为空实现。

### 步骤 3：搭建测试用例框架

    搭建测试用例框架 TEST_F(XXXX, XXX){ }，其中XXXX为上述新建的测试类，XXX为该测试用例名字，可根据根据业务内容生成。
    另外，在TEST_F(XXXX, XXX){ }上方位置处可以添加该测试用例注释，描述经过该pass前后的变化，例如：
 ```cpp
            /*
        TESTRemoveDummyExpand
        inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                            ->sqrt->outCast2{8,16}
                                            ->reciprocal->outCast3{8,16}
        inCast{8,16}->exp->outCast1
                    ->sqrt->outCast2
                    ->reciprocal->outCast3
        */
        TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1) {
            ...
        }
```

### 步骤 4：构建function

    构建整张计算图function，利用智能指针进行创建，并在创建后判断是否为空。若用户未明确指定参数，则belongTo为新建Program实例，funcMagicName和funcRawName均为TestXXX，XXX为pass名字，parentFunc为空指针。

    详细的信息如下：
        Function类常用构造函数：
```cpp
        Function(const Program &belongTo, const std::string &funcMagicName, const std::string &funcRawName,
            Function *parentFunc);
```
        Program常用获取实例：
```cpp
        Program &Program::GetInstance() {
            static Program sProgram;
            return sProgram;
        }
```

        详细代码可以参考：pypto/framework/src/interface/program/program.cpp和pypto/framework/src/interface/function/function.h

### 步骤 5：创建Tensor

    构建计算图中的Tensor，利用智能指针，根据业务需求，创建所需要的Tensor。

    LogicalTensor类常用构造函数：
        LogicalTensor(Function &function, DataType t, Shape tshape, TileOpFormat tformat = TileOpFormat::TILEOP_ND, std::string tname = "",
        NodeType tnodetype = NodeType::LOCAL);

    详细LogicalTensor类信息，请参考：pypto/framework/src/interface/tensor/logical_tensor.h

### 步骤 6：创建Operation及绑定function输入输出

    构建计算图中的Operation，利用智能指针，根据业务需求，创建所需要的Operation。

    常用的创建Operation及绑定function函数：
```cpp
            Operation &Program::AddOperation(const Opcode opCode,
        const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
        const std::vector<std::shared_ptr<LogicalTensor>> &oOperand) {
        // Add the operation to the current function
        if (currentFunctionMagicName_ == PROGRAM_ENTRY_FUNCTION_NAME) {
            FE_LOGE("Error: No active function to add operation.");
            ASSERT(false) << "No active function to add operation.";
        }
        return currentFunctionPtr_->AddOperation(opCode, iOperand, oOperand);
    }
```

    详细创建Operation及绑定function函数信息，请参考：pypto/framework/src/interface/program/program.cpp

    Opcode信息，请参考：pypto/framework/src/interface/operation/opcode.h和pypto/framework/src/interface/operation/opcode.cpp

### 步骤 7：对业务功能进行校验

    根据业务功能，校验pass运行后的处理结果是否符合预期。

    例如校验function经过pass后，assemble数量是否符合预期，可以通过遍历function，查找assemble的数量进行比对，具体代码如下：
```cpp
        uint32_t assemble_num = kNumZero;
        for (auto &op : currFunctionPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                ++assemble_num;
            }
        }
        EXPECT_EQ(assemble_num, kNumZero);
```

### 步骤 8：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)

    当生成UT用例后，执行执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确，xxx为该测试用例类的名字，对于错误进行改正。
    当执行超时时，优先执行Python3 build_ci.py -u=xxx.* -j=24。
    若此时还存在执行超时问题，则执行Python3 build_ci.py -u=xxx.xx -j=24，xx为测试用例
    例如: python3 build_ci.py -c -u=TestRemoveRedundantOpPass.* -j=24 -f=cpp
          python3 build_ci.py -u=TestRemoveRedundantOpPass.* -j=24 -f=cpp
          python3 build_ci.py -c -u=TestRemoveRedundantOpPass.TestIntermediateOutcast -j=24 -f=cpp

### 步骤 9：统计UT覆盖率

    在当前环境中，通过GCov来统计代码UT覆盖率，主要包括行覆盖率和方法覆盖率。
    使用方法：
        python3 build_ci.py -c -u=xxx.* --gcov -j=24 -f=cpp
    xxx为该测试用例类的名字,xx为测试用例
    执行结束时,会在build/路径下，生成cov_result 目录，打开index.html观察对应pass的UT覆盖率情况，若覆盖率<=80%，则针对未覆盖的业务，重复以上步骤，进行该Pass UT补充。

## UT生成流程二

### 步骤 1：分析业务

    同UT生成流程一中步骤1：分析业务

### 步骤 2：环境配置

    同UT生成流程一中步骤2：环境配置

### 步骤 3：搭建测试用例框架

    同UT生成流程一中步骤3：搭建测试用例框架

### 步骤 4：构建function

    利用ComputationalGraphBuilder类来构建function，通过调用AddTensor()和AddTensors()来实现function中Tensor的构建，调用AddOp()和AddOps()来实现function中Op的构建。
    通过调用SetInCast()和SetOutCast()来实现对function的输入输出构建。

    ComputationalGraphBuilder类信息，请参考：pypto/framework/tests/ut/passes/src/computational_graph_builder.h
    Opcode信息，请参考：pypto/framework/src/interface/operation/opcode.h和pypto/framework/src/interface/operation/opcode.cpp
    Operation信息，请参考: pypto/framework/src/interface/operation/operation.cpp
    详细LogicalTensor类信息，请参考: pypto/framework/src/interface/tensor/logical_tensor.h
    Tensor创建过程中DataType信息，请参考：pypto/framework/include/tilefwk/data_type.h

### 步骤 5：对业务功能进行校验

    同UT生成流程一中步骤7：对业务功能进行校验

### 步骤 6：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)

    同UT生成流程一中步骤8：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)

### 步骤 7：统计UT覆盖率

    同UT生成流程一中步骤9：统计UT覆盖率

## 注意事项

    要符合常见的CPP代码编程规范，例如常见的编程错误：未使用的变量定义、未修改的引用加入const、魔鬼数字等；
    生成的用例请真实执行，并对生成的用例进行验证修改；
    对于UT覆盖率，请打印出来当前所设计UT的覆盖率。

## 工具使用

### pr_utils.py - PR 处理工具（在线）

功能：
- 通过 PR 编号获取代码变更内容
- 解析 PR 评论中的 UT-REPORT，获取覆盖率和未覆盖行
- 将 diff 应用到本地仓库
- 检查编译状态
- 生成 UT 设计建议

使用示例：
```bash
# 在线处理 PR
python3 scripts/pr_utils.py 1894
```

### ut_coverage.py - 覆盖率分析工具

功能：
- 解析本地 diff 文件
- 解析本地覆盖率报告（支持 .tar.gz 格式）
- 关联 Diff 和覆盖率
- 生成 UT 设计建议
- 提取未覆盖行

使用示例：
```bash
# 解析本地 diff 文件
python3 scripts/ut_coverage.py --diff /path/to/pr.diff

# 解析本地覆盖率报告（.tar.gz 格式）
python3 scripts/ut_coverage.py --report /path/to/ut_cov.tar.gz

# 综合分析（同时指定 diff 和覆盖率文件）
python3 scripts/ut_coverage.py --diff /path/to/pr.diff --report /path/to/ut_cov.tar.gz

# 输出 JSON 格式建议
python3 scripts/ut_coverage.py --diff pr.diff --report ut_cov.tar.gz --json
```

详见 [references/usage.md](references/usage.md)

## 故障排查

详见 [references/trouble_shooting.md](references/trouble_shooting.md)

## 参考资料

| 类别 | 文件路径 |
|------|----------|
| 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` |
| 生成流程二示例 | `pypto/framework/tests/ut/passes/src/test_cube_process.cpp` |
| ComputationalGraphBuilder | `pypto/framework/tests/ut/passes/src/computational_graph_builder.h` |
| function 信息 | `pypto/framework/src/interface/program/program.cpp和pypto/framework/src/interface/function/function.h`|
| Opcode 定义 | `pypto/framework/src/interface/operation/opcode.h` |
| op 属性定义 | `framework/src/interface/operation/attribute.h` |
| DataType 定义 | `pypto/framework/include/tilefwk/data_type.h` |
| LogicalTensor 定义 | `pypto/framework/src/interface/tensor/logical_tensor.h` |
| Operation 信息 | `pypto/framework/src/interface/operation/operation.cpp`  |
| COMPILE_STAGE 策略 | `pypto/framework/src/interface/configs/config_manager_ng.h` |

---

## 编译阶段参考值

| 阶段 | 配置值 |
|------|--------|
| TENSOR GRAPH 执行 | `CS_TENSOR_GRAPH` |
| TILE GRAPH 执行 | `CS_TILE_GRAPH` |
| BLOCK GRAPH 执行 | `CS_EXECUTE_GRAPH` |

根据 Pass 所在文件夹目录选择对应的编译策略。

## 检查清单

详见 [references/check_list.md](references/check_list.md)
