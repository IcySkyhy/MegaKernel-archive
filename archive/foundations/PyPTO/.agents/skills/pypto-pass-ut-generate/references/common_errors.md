# Pass UT 开发常见错误记录

本文档记录在开发 Pass 单元测试（UT）过程中遇到的常见错误和解决方案，作为故障排查的参考手册。

---

## 快速查找索引

### 按错误现象查找

| 错误现象 | 参考错误 | 章节 |
|---------|---------|------|
| 编译失败 | 错误 1, 2, 3, 4, 38, 39, 40 | [语法错误类](#语法错误类), [编译相关类](#编译相关类) |
| 运行时崩溃 | 错误 14, 15, 16, 25, 32, 34 | [内存管理类](#内存管理类), [遍历相关类](#遍历相关类) |
| 断言失败 | 错误 6, 35, 36, 37 | [逻辑错误类](#逻辑错误类), [验证相关类](#验证相关类) |
| 测试通过但覆盖率低 | 错误 9, 10, 11 | [测试覆盖类](#测试覆盖类) |
| Pass 执行失败 | 错误 5, 12, 13, 29, 30, 31 | [环境配置类](#环境配置类), [Pass执行类](#pass-执行类) |
| 图结构错误 | 错误 8, 26, 27, 28 | [Tensor连接类](#tensor-连接类) |
| 数据类型错误 | 错误 7, 17, 18, 19 | [数据结构类](#数据结构类) |
| Opcode 相关错误 | 错误 20, 21, 22 | [Opcode相关类](#opcode-相关类) |
| Attribute 相关错误 | 错误 23, 24, 25 | [Attribute相关类](#attribute-相关类) |

### 按错误类别查找

| 错误类别 | 错误编号 |
|---------|---------|
| 语法错误类 | 1, 2, 3, 4 |
| 逻辑错误类 | 5, 6, 7, 8 |
| 测试覆盖类 | 9, 10, 11 |
| 环境配置类 | 12, 13 |
| 内存管理类 | 14, 15, 16 |
| 数据结构类 | 17, 18, 19 |
| Opcode 相关类 | 20, 21, 22 |
| Attribute 相关类 | 23, 24, 25 |
| Tensor 连接类 | 26, 27, 28 |
| Pass 执行类 | 29, 30, 31 |
| 遍历相关类 | 32, 33, 34 |
| 验证相关类 | 35, 36, 37 |
| 编译相关类 | 38, 39, 40 |
| 调试相关类 | 41, 42, 43 |

---

## 故障排查流程

### 编译错误排查流程

```
编译失败
    ↓
检查错误信息
    ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ 语法错误         │ 头文件缺失       │ 命名空间错误     │
└────────┬─────────┴────────┬─────────┴────────┬─────────┘
         ↓                  ↓                  ↓
   [错误 1-4]          [错误 40]           [错误 39]
         ↓                  ↓                  ↓
   检查括号匹配        添加缺失头文件     修正命名空间
   检查中文字符        检查头文件路径     添加 using 声明
   检查变量定义
```

### 运行时错误排查流程

```
运行时崩溃
    ↓
检查崩溃堆栈
    ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ 空指针访问       │ 内存泄漏         │ 容器遍历错误     │
└────────┬─────────┴────────┬─────────┴────────┬─────────┘
         ↓                  ↓                  ↓
   [错误 14-16]        [错误 15,16]        [错误 32-34]
         ↓                  ↓                  ↓
   检查智能指针使用    使用智能指针        先复制容器再修改
   检查指针初始化      检查资源释放        使用安全迭代器
   添加空指针检查
```

### 断言失败排查流程

```
断言失败
    ↓
检查断言信息
    ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ 期望值错误       │ 验证不完整       │ 连接关系错误     │
└────────┬─────────┴────────┬─────────┴────────┬─────────┘
         ↓                  ↓                  ↓
   [错误 6,35]         [错误 35,36]        [错误 8,26-28]
         ↓                  ↓                  ↓
   检查期望值设置      添加完整验证        验证消费者/生产者
   打印实际值          验证操作输入输出    验证图结构
   调整期望值          验证数据类型
```

---

## 语法错误类

### 错误 1：Operations() 调用语法错误

**错误代码：**
```cpp
function->Operations().().size()
```

**正确代码：**
```cpp
function->Operations().size()
```

**原因：** `Operations()` 返回的是引用或对象，不需要再调用 `()`

**解决方案：** 检查所有 `Operations()` 的调用，确保只调用一次 `()`

**相关章节：** [SKILL.md - 步骤 6](../SKILL.md#步骤-6创建operation及绑定function输入输出)

---

### 错误 2：中文字符混入代码

**错误代码：**
```cpp
op.GetOOperands()[0]->Datatype与其他()
```

**正确代码：**
```cpp
op.GetOOperands()[0]->Datatype()
```

**原因：** 输入法切换问题导致中文字符混入

**解决方案：** 使用英文输入法，仔细检查代码中的函数名

**相关章节：** [SKILL.md - 注意事项](../SKILL.md#注意事项)

---

### 错误 3：重复定义变量

**错误代码：**
```cpp
const int opNum1 = 1;
// ... some code ...
const int opNum1 = 1;  // 重复定义
```

**正确代码：**
```cpp
const int opNum1 = 1;
// ... some code ...
// 不要重复定义，直接使用已定义的变量
```

**原因：** 在不同测试用例中使用了相同的变量名，但忘记检查作用域

**解决方案：** 每个测试用例使用不同的变量名，或确保变量作用域正确

---

### 错误 4：类定义中使用中文冒号

**错误代码：**
```cpp
class AutoCastExtendedTest ： public testing::Test {
public:
    static void SetUpTestCase() {}
```

**正确代码：**
```cpp
class AutoCastExtendedTest : public testing::Test {
public:
    static void SetUpTestCase() {}
```

**原因：** 输入法切换导致冒号变成中文冒号

**解决方案：** 使用英文输入法编写代码

---

## 逻辑错误类

### 错误 5：错误的编译阶段配置

**错误代码：**
```cpp
config::SetHostOption(COMPILE_STAGE, CS::EXECUTE_GRAPH);
```

**正确代码（对于 TensorGraph 阶段的 Pass）：**
```cpp
config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
```

**原因：** 未根据 Pass 所在的文件夹目录选择正确的编译阶段

**解决方案：** 根据 Pass 所在目录选择编译策略
- `tensor_graph_pass`: `CS_TENSOR_GRAPH`
- `tile_graph_pass`: `CS_TILE_GRAPH`
- `block_graph_pass`: `CS_EXECUTE_GRAPH`

**相关章节：** [SKILL.md - 步骤 2](../SKILL.md#步骤-2环境配置)

---

### 错误 6：错误的期望操作数

**错误代码：** 期望的操作数与实际不符

**原因：** 未仔细分析 Pass 的业务逻辑，对插入或删除的操作数预估错误

**解决方案：**
1. 先运行 Pass，打印实际的操作数
2. 根据实际结果调整期望值
3. 确保理解每个功能点会插入/删除多少操作

**相关章节：** [SKILL.md - 步骤 7](../SKILL.md#步骤-7对业务功能进行校验)

---

### 错误 7：未验证数据类型转换

**错误代码：** 只验证操作数，不验证数据类型

**正确代码：** 验证 Cast 操作后的数据类型是否正确

**原因：** AutoCast Pass 的核心功能是数据类型转换，必须验证转换结果

**解决方案：** 在遍历操作时，检查每个操作输入输出的数据类型

**相关章节：** [SKILL.md - 步骤 7](../SKILL.md#步骤-7对业务功能进行校验)

---

### 错误 8：未验证 Tensor 连接关系

**错误代码：** 只验证操作数，不验证 Tensor 的连接关系

**正确代码：** 验证操作删除后，Tensor 的消费者和生产者关系是否正确

**原因：** 删除操作后需要确保图结构正确

**解决方案：** 验证关键 Tensor 的消费者和生产者数量

**相关章节：** [SKILL.md - 步骤 7](../SKILL.md#步骤-7对业务功能进行校验)

---

## 测试覆盖类

### 错误 9：测试用例覆盖不全

**错误：** 只测试了部分场景

**原因：** 未全面分析 Pass 的所有功能点

**解决方案：**
1. 列出 Pass 的所有功能点（如 InsertBF16Cast、RemoveRedundantCastChain 等）
2. 为每个功能点设计至少一个测试用例
3. 考虑边界情况（空图、无 Cast、冗余链等）

**相关章节：** [SKILL.md - 步骤 9](../SKILL.md#步骤-9统计ut覆盖率)

---

### 错误 10：未测试架构差异

**错误：** 未测试不同 NPU 架构下的行为差异

**原因：** AutoCast Pass 在 DAV_3510 和其他架构上有不同行为

**解决方案：** 针对不同架构设计专门的测试用例

---

### 错误 11：未测试边界情况

**错误：** 只测试正常场景，未测试边界情况

**原因：** 边界情况容易暴露 Pass 的 bug

**解决方案：** 测试以下边界情况
- 空图
- 单个操作
- 多个相同操作
- 操作链
- 多个消费者

---

## 环境配置类

### 错误 12：未重置 Program 和 Config

**错误代码：** 测试用例间未重置环境

**正确方法：** 在 `SetUp()` 中重置 Program 和 Config

**原因：** 测试用例间可能相互影响

**解决方案：** 确保每个测试用例开始前环境是干净的

**相关章节：** [SKILL.md - 步骤 2](../SKILL.md#步骤-2环境配置)

---

### 错误 13：NPU 架构未恢复

**错误代码：** 修改 NPU 架构后未恢复

**正确代码：** 测试结束后恢复默认架构

**原因：** 影响后续测试用例

**解决方案：** 在测试用例结束前恢复 `NPUArch::DAV_UNKNOWN`

---

## 内存管理类

### 错误 14：智能指针使用不当

**错误代码：**
```cpp
auto tensor = new LogicalTensor(...);  // 未使用智能指针
```

**正确代码：**
```cpp
auto tensor = std::make_shared<LogicalTensor>(...);  // 使用智能指针
```

**原因：** 未使用智能指针可能导致内存泄漏

**解决方案：** 使用 `std::make_shared` 创建对象

**相关章节：** [SKILL.md - 步骤 4](../SKILL.md#步骤-4构建function)

---

### 错误 15：Operation 引用获取错误

**错误代码：**
```cpp
Operation *op = function.AddOperation(...);  // 返回的是引用
```

**正确代码：**
```cpp
Operation &op = function.AddOperation(...);  // 使用引用
```

**原因：** `AddOperation` 返回的是引用，不是指针

**解决方案：** 使用引用接收 `AddOperation` 的返回值

**相关章节：** [SKILL.md - 步骤 6](../SKILL.md#步骤-6创建operation及绑定function输入输出)

---

### 错误 16：未使用智能指针管理资源

**错误代码：**
```cpp
LogicalTensor *tensor = new LogicalTensor(...);
// 忘记 delete tensor
```

**正确代码：**
```cpp
auto tensor = std::make_shared<LogicalTensor>(...);
// 自动管理内存
```

**原因：** 手动管理内存容易导致内存泄漏

**解决方案：** 优先使用智能指针管理所有动态分配的资源

---

## 数据结构类

### 错误 17：Shape 比较错误

**错误代码：**
```cpp
if (tensor1->shape == tensor2->shape) { ... }  // shape 是成员变量
```

**正确代码：**
```cpp
if (tensor1->GetShape() == tensor2->GetShape()) { ... }  // 使用 getter 方法
```

**原因：** shape 是成员变量，应该使用 `GetShape()` 方法

**解决方案：** 使用 `GetShape()` 方法获取 shape

**相关章节：** [SKILL.md - 步骤 5](../SKILL.md#步骤-5创建tensor)

---

### 错误 18：DataType 比较错误

**错误代码：**
```cpp
if (tensor->Datatype() == DataType::DT_FP32) { ... }  // 方法名错误
```

**正确代码：**
```cpp
if (tensor->GetDatatype() == DataType::DT_FP32) { ... }  // 使用 GetDatatype()
```

**原因：** DataType 的 getter 方法名是 `GetDatatype()`

**解决方案：** 使用 `GetDatatype()` 方法

**相关章节：** [SKILL.md - 步骤 5](../SKILL.md#步骤-5创建tensor)

---

### 错误 19：直接访问成员变量

**错误代码：**
```cpp
auto shape = tensor->shape;
auto consumers = tensor->consumers;
```

**正确代码：**
```cpp
auto shape = tensor->GetShape();
auto consumers = tensor->GetConsumers();
```

**原因：** 应该使用 getter 方法访问成员变量

**解决方案：** 优先使用 getter 方法访问类的成员变量

---

## Opcode 相关类

### 错误 20：Opcode 枚举使用错误

**错误代码：**
```cpp
if (op.GetOpcode() == OP_CAST) { ... }  // 缺少命名空间
```

**正确代码：**
```cpp
if (op.GetOpcode() == Opcode::OP_CAST) { ... }  // 使用 Opcode 命名空间
```

**原因：** Opcode 是枚举类，需要使用 `Opcode::` 前缀

**解决方案：** 使用 `Opcode::` 前缀访问枚举值

**相关章节：** [SKILL.md - 步骤 6](../SKILL.md#步骤-6创建operation及绑定function输入输出)

---

### 错误 21：Opcode 集合查找错误

**错误代码：**
```cpp
std::set<Opcode> opSet = {Opcode::OP_ADD, Opcode::OP_MUL};
if (opSet.find(op.GetOpcode()) != opSet.end()) { ... }  // 正确
```

**原因：** Opcode 是枚举类，可以直接在集合中查找

**解决方案：** 确保使用正确的集合查找方法

---

### 错误 22：使用错误的 Opcode 值

**错误代码：**
```cpp
if (op.GetOpcode() == 123) { ... }  // 使用魔法数字
```

**正确代码：**
```cpp
if (op.GetOpcode() == Opcode::OP_ADD) { ... }  // 使用枚举值
```

**原因：** 使用魔法数字降低代码可读性，容易出错

**解决方案：** 始终使用 `Opcode::` 枚举值

---

## Attribute 相关类

### 错误 23：Attribute 获取错误

**错误代码：**
```cpp
auto attr = op.GetOpAttribute();  // 返回的是 shared_ptr
```

**正确代码：**
```cpp
auto attr = op.GetOpAttribute().get();  // 获取原始指针
```

**原因：** `GetOpAttribute()` 返回的是 `shared_ptr`，需要使用 `get()` 获取原始指针

**解决方案：** 使用 `get()` 方法获取原始指针或直接使用 `shared_ptr`

**相关章节：** [SKILL.md - UT生成流程二 - 步骤 4](../SKILL.md#步骤-4构建function-1)

---

### 错误 24：Attribute 类型转换错误

**错误代码：**
```cpp
auto attr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute());  // 错误
```

**正确代码：**
```cpp
auto attr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());  // 正确
```

**原因：** 需要先获取原始指针再进行类型转换

**解决方案：** 先使用 `get()` 获取原始指针，再进行 `dynamic_cast`

**相关章节：** [SKILL.md - UT生成流程二 - 步骤 4](../SKILL.md#步骤-4构建function-1)

---

### 错误 25：未检查 Attribute 是否为空

**错误代码：**
```cpp
auto attr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
attr->GetViewOpType();  // attr 可能为 nullptr
```

**正确代码：**
```cpp
auto attr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());
ASSERT_NE(attr, nullptr);
attr->GetViewOpType();
```

**原因：** 类型转换可能失败，返回 `nullptr`

**解决方案：** 使用 `ASSERT_NE` 或 `EXPECT_NE` 检查转换结果

---

## Tensor 连接类

### 错误 26：消费者/生产者获取错误

**错误代码：**
```cpp
auto consumers = tensor->consumers;  // consumers 是成员变量
```

**正确代码：**
```cpp
auto consumers = tensor->GetConsumers();  // 使用 getter 方法
```

**原因：** 应该使用 `GetConsumers()` 方法

**解决方案：** 使用 `GetConsumers()` 和 `GetProducers()` 方法

---

### 错误 27：Tensor 连接关系验证错误

**错误代码：**
```cpp
EXPECT_EQ(tensor->GetConsumers().size(), 1);  // 未验证具体消费者
```

**正确代码：**
```cpp
EXPECT_EQ(tensor->GetConsumers().size(), 1);
EXPECT_EQ(tensor->GetConsumers()[0]->GetOpcode(), Opcode::OP_ADD);  // 验证具体消费者
```

**原因：** 需要验证连接关系的正确性

**解决方案：** 验证消费者/生产者的具体操作类型

---

### 错误 28：未验证 Tensor 的数据类型

**错误代码：**
```cpp
auto tensor = function->GetInputTensor(0);
EXPECT_NE(tensor, nullptr);  // 只验证存在性
```

**正确代码：**
```cpp
auto tensor = function->GetInputTensor(0);
EXPECT_NE(tensor, nullptr);
EXPECT_EQ(tensor->GetDatatype(), DataType::DT_FP32);  // 验证数据类型
```

**原因：** 需要确保 Tensor 的数据类型正确

**解决方案：** 验证 Tensor 的数据类型、Shape 等属性

---

## Pass 执行类

### 错误 29：未调用 PreCheck/PostCheck

**错误代码：**
```cpp
pass.RunOnFunction(function);  // 只调用 RunOnFunction
```

**正确代码：**
```cpp
EXPECT_EQ(pass.PreCheck(function), SUCCESS);  // 调用 PreCheck
EXPECT_EQ(pass.RunOnFunction(function), SUCCESS);  // 调用 RunOnFunction
EXPECT_EQ_EQ(pass.PostCheck(function), SUCCESS);  // 调用 PostCheck
```

**原因：** 需要验证 Pass 的前置和后置检查

**解决方案：** 调用 `PreCheck` 和 `PostCheck` 并验证返回值

---

### 错误 30：Pass 返回值未验证

**错误代码：**
```cpp
pass.RunOnFunction(function);  // 未验证返回值
```

**正确代码：**
```cpp
EXPECT_EQ(pass.RunOnFunction(function), SUCCESS);  // 验证返回值
```

**原因：** 需要验证 Pass 执行是否成功

**解决方案：** 使用 `EXPECT_EQ` 验证 Pass 的返回值

---

### 错误 31：未验证 Pass 的副作用

**错误代码：**
```cpp
pass.RunOnFunction(function);
EXPECT_EQ(function->Operations().size(), 3);  // 只验证操作数量
```

**正确代码：**
```cpp
pass.RunOnFunction(function);
EXPECT_EQ(function->Operations().size(), 3);
// 验证具体的操作类型和连接关系
uint32_t cast_num = 0;
for (auto &op : function->Operations()) {
    if (op.GetOpcode() == Opcode::OP_CAST) {
        ++cast_num;
    }
}
EXPECT_EQ(cast_num, 1);
```

**原因：** 需要验证 Pass 执行后的具体变化

**解决方案：** 验证操作类型、数量、连接关系等

---

## 遍历相关类

### 错误 32：遍历时修改容器

**错误代码：**
```cpp
for (auto &op : function->Operations()) {
    if (shouldRemove) {
        function.RemoveOperation(op);  // 遍历时修改容器
    }
}
```

**正确代码：**
```cpp
auto opList = function->Operations().DuplicatedOpList();  // 先复制
for (auto op : opList) {
    if (shouldRemove) {
        function.RemoveOperation(op);  // 安全删除
    }
}
```

**原因：** 遍历时修改容器会导致未定义行为

**解决方案：** 先复制操作列表，再进行修改

---

### 错误 33：遍历顺序依赖

**错误代码：** 假设遍历顺序固定

**原因：** 容器的遍历顺序可能不固定

**解决方案：** 不要依赖遍历顺序，使用操作 ID 或 Magic ID 进行验证

---

### 错误 34：遍历时使用错误的迭代器

**错误代码：**
```cpp
for (auto it = function->Operations().begin(); it != function->Operations().end(); ++it) {
    if (shouldRemove) {
        function.RemoveOperation(*it);  // 迭代器失效
    }
}
```

**正确代码：**
```cpp
auto opList = function->Operations().DuplicatedOpList();
for (auto op : opList) {
    if (shouldRemove) {
        function.RemoveOperation(op);
    }
}
```

**原因：** 删除操作会导致迭代器失效

**解决方案：** 使用 `DuplicatedOpList()` 复制操作列表

---

## 验证相关类

### 错误 35：只验证数量不验证内容

**错误代码：**
```cpp
EXPECT_EQ(function->Operations().size(), 3);  // 只验证数量
```

**正确代码：**
```cpp
EXPECT_EQ(function->Operations().size(), 3);
uint32_t cast_num = 0;
for (auto &op : function->Operations()) {
    if (op.GetOpcode() == Opcode::OP_CAST) {
        ++cast_num;
    }
}
EXPECT_EQ(cast_num, 2);  // 验证具体操作数量
```

**原因：** 需要验证操作的具体类型

**解决方案：** 遍历操作，统计各类型操作的数量

---

### 错误 36：未验证操作输入输出

**错误代码：** 只验证操作存在，不验证输入输出

**正确代码：** 验证操作的输入输出 Tensor 是否正确

**原因：** 需要确保操作的输入输出连接正确

**解决方案：** 验证操作的 `IOperands` 和 `OOperands`

---

### 错误 37：未验证 Tensor 的连接关系

**错误代码：**
```cpp
EXPECT_NE(tensor, nullptr);  // 只验证 Tensor 存在
```

**正确代码：**
```cpp
EXPECT_NE(tensor, nullptr);
EXPECT_EQ(tensor->GetConsumers().size(), 1);  // 验证消费者数量
EXPECT_EQ(tensor->GetConsumers()[0]->GetOpcode(), Opcode::OP_ADD);  // 验证消费者类型
```

**原因：** 需要验证 Tensor 的连接关系

**解决方案：** 验证 Tensor 的消费者和生产者

---

## 编译相关类

### 错误 38：头文件包含错误

**错误代码：** 缺少必要的头文件

**原因：** 未包含必要的头文件导致编译错误

**解决方案：** 参考现有测试文件，确保包含所有必要的头文件

---

### 错误 39：命名空间使用错误

**错误代码：**
```cpp
using namespace std;
using namespace npu;
// ... some code ...
Operation op;  // 未指定命名空间
```

**正确代码：**
```cpp
using namespace std;
using namespace npu::tile_fwk;
// ... some code ...
Operation op;  // 在正确的命名空间中
```

**原因：** 命名空间使用不当导致编译错误

**解决方案：** 确保在正确的命名空间中编写代码

---

### 错误 40：未包含必要的测试头文件

**错误代码：**
```cpp
#include "gtest/gtest.h"  // 可能缺失
```

**udi 正确代码：**
```cpp
#include "gtest/gtest.h"
#include "framework/tests/ut/passes/src/computational_graph_builder.h"
// 其他必要的头文件
```

**原因：** 缺少必要的头文件导致编译错误

**解决方案：** 参现有测试文件，包含所有必要的头文件

---

## 调试相关类

### 错误 41：缺少调试信息

**错误代码：** 测试失败时没有足够的调试信息

**正确代码：** 使用 `EXPECT_EQ` 并提供有意义的消息

**原因：** 缺少调试信息难以定位问题

**解决方案：** 使用 `EXPECT_EQ`、`EXPECT_NE` 等宏并提供有意义的消息

---

### 错误 42：未打印中间结果

**错误代码：** 不打印中间结果进行调试

**正确代码：** 打印关键中间结果

**原因：** 难以定位问题所在

**解决方案：** 在关键位置打印中间结果

---

### 错误 43：未使用断言宏

**错误代码：**
```cpp
if (tensor == nullptr) {
    return;  // 直接返回
}
```

**正确代码：**
```cpp
ASSERT_NE(tensor, nullptr);  // 使用断言宏
```

**原因：** 未使用断言宏难以发现错误

**解决方案：** 使用 `ASSERT_*` 和 `EXPECT_*` 宏进行验证

---

## 最佳实践

### 避免错误的最佳实践

1. **仔细检查代码语法**：特别是函数调用和括号匹配
2. **使用不同的变量名**：避免重复定义变量
3. **根据 Pass 所在目录选择正确的编译阶段**
4. **验证数据类型转换的正确性**：检查每个操作的输入输出数据类型
5. **使用英文输入法编写代码**：避免中文字符混入
6. **在 SetUp() 中重置环境**：确保每个测试用例开始前环境是干净的
7. **恢复 NPU 架构**：测试用例结束前恢复默认架构
8. **使用智能指针**：避免内存泄漏
9. **使用引用接收 AddOperation 的返回值**
10. **先复制操作列表再修改**：避免遍历时修改容器

### 调试技巧

1. **打印中间结果**：在关键位置打印中间结果
2. **分步验证**：逐步验证每个功能点
3. **使用断言**：使用 `EXPECT_EQ`、`EXPECT_NE` 等宏
4. **检查图结构**：验证操作和 Tensor 的连接关系
5. **使用 gcov**：统计代码覆盖率，确保测试充分

### 测试设计原则

1. **覆盖所有功能点**：为每个功能点设计测试用例
2. **测试边界情况**：空图、单个操作、操作链等
3. **验证图结构**：不仅验证操作数，还要验证连接关系
4. **测试架构差异**：针对不同架构设计专门的测试用例
5. **逐步验证**：先验证简单场景，再验证复杂场景

### 代码质量检查清单

- [ ] 无未使用的变量
- [ ] 无重复定义的变量
- [ ] 使用智能指针管理动态资源
- [ ] 使用 getter 方法访问成员变量
- [ ] 使用 `Opcode::` 前缀访问枚举值
- [ ] 遍历时先复制容器再修改
- [ ] 验证 Pass 返回值
- [ ] 验证操作的输入输出
- [ ] 验证 Tensor 的连接关系
- [ ] 使用 `ASSERT_*` 和 `EXPECT_*` 宏
