# probe_host_dma —— M86：AIV kernel 经 `DataCopy`/MTE2 直接读 host 内存的探针 + ngram mmap 方案

> **一句话**：在 **CANN 9.1.0 / Ascend950PR（dav-3510，28 AIC + 56 AIV）** 上，
> `aclrtMallocHost` + `aclrtHostRegisterV2(ACL_HOST_REG_MAPPED)` + `aclrtHostGetDevicePointer`
> 得到的 device 指针**确实能被 AIV 的 `DataCopy`（MTE2）直接读**，逐字节正确（0.9–10 MB 全对，
> 多次独立进程）；聚合带宽约 **20–22.5 GB/s**（对照 HBM ~1.1 TB/s，空 kernel 基线 ~2.5 µs）。
> **但**：**file-backed mmap 不能注册**（`rc=507899`），只有匿名/堆/`aclrtMallocHost` 能注册
> ⇒ ngram 方案必须是「文件 mmap 做按需页 + 搬运到**一次性注册、反复复用**的 staging 窗口」。

本目录所有结论**只有两种来源**，全文分栏不混排：

* **[实测]** —— 本目录探针在本机 NPU 上跑出来的日志（`evidence/logs/`，逐条可查）；
* **[文档]** —— CANN 头文件 / 官方文档 / 仓内知识库的原文片段（`文件:行 + 原文`）。

---

## 1. 交付物与目录

| 文件 | 作用 |
|---|---|
| `probe_host_dma.asc` | 主探针（单文件：AIV kernel + host ACL 代码），mode = `caps`/`read`/`hbm`/`empty`/`filemmap`/`unreg`/`align`/`page`/`regbench` |
| `CMakeLists.txt` | 独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`，仿 `probe_v_align`/`m0`） |
| `run_probes.sh` | 一键复现：构建 → 逐 mode **独立进程**运行 → 归档 `evidence/`（末尾自跑源码指纹自检） |
| `mmap_ngram_plan.py` | ngram 的 mmap 方案（`--real` 打印真实几何方案；`--demo` 用小几何**真落盘→滑窗→逐字节校验**；`--real` 的注册代价**从 `evidence/logs/regbench*.log` 解析**，不硬编码） |
| `check_locale.sh` | 判据命令的 locale 一致性 + 扫描正对照 |
| `check_fingerprints.sh` | **源码指纹守卫**：核对 `commands.txt` 记的 sha256 是否等于当前工作树（把「这份 log 由哪份源码产出」变成一条可跑命令） |
| `audit_numbers.py` | **逐数字冒烟测试（启发式，见 §2；不是可复现性证明）**：扫 README 里每个数字，要求它能在 `evidence/` 里找到、或紧贴逃逸标记、或同行有语义词标注 |
| `evidence/logs/` | 归档原始日志 + `commands.txt`（含 `HEAD_at_run`、**源码 sha256 指纹**、CANN 版本、跑前/跑后 `npu-smi` 并发背景） |
| `evidence/summary.txt` | 从各 log 抽取的关键行汇总表（含逐 mode 中位数 + **regbench 中位/区间与抽取正对照**） |

```bash
cd probe_host_dma
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
bash run_probes.sh 5                 # 全量 + 证据归档（约 5 分钟；共享卡上更久）
./build/probe_host_dma caps          # 单 mode 手工复现
./build/probe_host_dma read --launches 20
PY=/usr/local/python3.12.13/bin/python3   # 本机 /usr/bin/python3 无 numpy 需求，但统一用它
$PY mmap_ngram_plan.py --real
$PY mmap_ngram_plan.py --demo --dir /tmp/m86_mmap_demo
```

---

## 2. 方法（reviewer 应先看这段）

* **一个 mode = 一次独立进程**：设备异常（`aclError`）会污染上下文；负向对照必须与正例隔离，
  否则分不清「谁报的错」。`run_probes.sh` 逐 mode 起独立进程并单独抓 log。
* **正向判据 = 逐字节**（docs/17 §1.1 的 **T1**，整数域/位域，**无容差**）：
  host 侧生成 `ByteModel(i) = (i*131 + (i>>7)*17 + 0x5A) & 0xFF`，kernel 读回后用 `aclrtMemcpy` D2H，
  与模型逐字节比对；**不写「已全部/无残留/0 命中」**，只报 `mismatches=<数>` 与首个坏字节偏移。
* **带宽口径**：host 墙钟（`steady_clock`），**K 次 launch + 一次 `aclrtSynchronizeStream`** 求均值，
  再与 `empty`（同形状空 kernel）相减区分「空跑基线 / 实际搬运」。每个正例跑 **5 次独立进程**。
* **设备侧另有 `SYS_CNT`**（`AscendC::GetSystemCycle()`）：每个 block 把 `[t0,t1]` 经 UB→MTE3 DMA 落 GM
  （标量直接写 GM 可见性无保证，沿用 m0 的教训）。
* **地址不是编译期常量**：`src`/偏移都来自 host 运行时实参。
* **一道守卫 + 一个冒烟测试**（本 mission 的纪律①在本 mission 内复发过两次，故做成可跑命令）：
  * `check_fingerprints.sh` —— **源码指纹守卫**：`commands.txt` 记的 sha256 是否等于当前工作树
    （判"这份 log 由哪份源码产出"，与 HEAD 时序解耦）；`run_probes.sh` 末尾自跑。
  * `audit_numbers.py` —— **逐数字"冒烟测试"（启发式；不是可复现性证明）**：扫 README 里每个数字，
    要求它能在 `evidence/` 的文本里找到、或**紧贴**逃逸标记（`[外部]` 别人给的读数 / `[git史]` 只在历史 rev 里）、
    或同行有保留语义词（推算/示意/常量/错误码/几何/定义/边界…）。
    **定位（按 R3/R4 复审要求降级，请把它当成"便宜的冒烟测试"读）**：
    - **通过有多便宜：`LABELED` 是作者自证** —— 只要**行内出现任一保留语义词，该行所有数字就被归入这个桶**
      （现网里 token `8162` 就是靠行内"注册**边界**"三个字被豁免的）；脚本**无法验证**作者的这个声称。
    - **`IN-EVIDENCE` 不等于可复现**：它只表示"该字符串出现在 `evidence/` 下某文件里"，
      把数塞进任意命名普通的文件即可命中。
    - **逃逸标记**同理：`[外部]`/`[git史]` 可以紧贴任意数写。
      紧贴判定跳过的不看字符只有 `` ` `` `*` `_` `~` 与空白（即 `DECOR_RE`；**比早期 docstring 写的更严**——
      `，, : ： ）)` 并不豁免，R4 复审实测"标记在前"与"只隔一个逗号"两种写法**都是 FLAG=1**）。
    - 故 **rc=0 只说明"没出现最粗的形态"，不构成任何保证**。
    **带负向对照**（`evidence/audit_negative_controls.txt`，6 个用例，每个都给了 rc 与 FLAG 数；
    注入的假数**不在本 README 里复述**，只在归档文件中）：
    NC0 平铺假数 / NC1 假数+同行"中位·区间" / NC2 假数+同行 `[外部]`（注：**这里复述的写法本身也是证据**——
    它们已被收紧后的守卫判为 FLAG，见归档） / NC4 数字紧贴单位（形如 `<数>ms`） / NC5 数字紧贴单位（形如 `<数>GB/s`）
    五个**必须报警**（已修，现均 FLAG 1, rc=1）；
    NC3（把数塞进普通名 evidence 文件）**仍可绕过**，**如实登记为已知口子**，不假装已闭合。

---

## 3. 结论

### 3.1 能力面：官方 API 与原文证据（[文档]）

**链路**（四步，全部实测 `rc=0`）：`aclrtMallocHost` → `aclrtHostRegisterV2(MAPPED)` →
`aclrtHostGetDevicePointer` → 把该 device 指针当 `GM_ADDR` 喂给 kernel。

| 位置 | 原文片段 |
|---|---|
| `$CANN/include/acl/acl_rt.h:74-79` | `// Host register flags for aclrtHostRegisterV2` / `#define ACL_HOST_REG_MAPPED 0x2UL // Map host memory to device address` / `ACL_HOST_REG_IOMEMORY 0x4UL` / `ACL_HOST_REG_READONLY 0x8UL` / `ACL_HOST_REG_PINNED 0x10000000UL // Pin memory to prevent swapping` |
| `acl_rt.h:1984-2012` | `aclError aclrtHostRegister(void *ptr, uint64_t size, aclrtHostRegisterType type, void **devPtr);` / `aclError aclrtHostRegisterV2(void *ptr, uint64_t size, uint32_t flag);` / `aclError aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag);`（注释：*return device pointer of mapped host memory registered by aclrtHostRegister or aclrtHostRegisterV2*） |
| `acl_rt.h:2036` | `aclError aclrtHostMemMapCapabilities(uint32_t deviceId, aclrtHacType hacType, aclrtHostMemMapCapability *capabilities);` |
| `$CANN/include/driver/ascend_hal_define.h:877-884` | `enum drvRegisterTpye { HOST_MEM_MAP_DEV = 0, ... HOST_MEM_MAP_DEV_PCIE_TH, /* HOST_MEM map to device, accessed by pcie_through */ ... }` —— **人类说的「pcie through」在驱动枚举里就叫这个名字** |
| `$CANN/include/driver/ascend_hal_base.h:2424-2432`（`@attention` 块；`:2433` 是 `@param src_ptr`） | `halHostRegister(void *src_ptr, UINT64 size, UINT32 flag, UINT32 devid, void **dst_ptr);` 的约束：`:2426` `2. HOST_MEM_MAP_DMA don't support read-only memory.`；`:2429-2430` `5. ... register os malloc va to dma is not support in Linux versions below 5.19.`；`:2431` `6. HOST_SVM_MAP_DEV don't support in virt machine.`；`:2432` `7. Not support vmm va, use may result in unexpected behavior.`；`:2433` `srcPtr must be page aligned` |
| `cannbot-skills/.../references/api_support_table.md:36-37,55-57` | `cudaMallocHost() → aclrtMallocHost()`；`cudaHostAlloc() → aclrtMallocHostAndRegister() / aclrtMallocHost()+aclrtHostRegisterV2()`，*mapped flag 会显式转换为 mapped+pinned*；`cudaHostRegister() → aclrtHostRegisterV2()`，*mapped 走 mapped+pinned*；`cudaHostGetDevicePointer() → aclrtHostGetDevicePointer()` |
| `ops-transformer/mc2/common/torch_extension/csrc/elastic_buffer.cpp:641-658` | **仓内真实用法**：`aclrtMallocHost(...)` → `aclrtHostRegisterV2(guard.hostPtr, numCpuBytes, ACL_HOST_REG_MAPPED)` → `aclrtHostGetDevicePointer(hostPtr, &devPtr, 0)` → `CommMem mem; mem.type = COMM_MEM_TYPE_DEVICE; mem.addr = devPtr;` 交给 HCCL |
| `cannbot-knowledge/.../techniques/cann_shmem.md:26,45-56` | 「**混合编址**：支持 NPU HBM 与 CPU DDR 混合编址，把 Host 内存映射成可直接当 Device tensor 使用的内存」；SHMEM 侧 API `aclshmem_malloc(bytes, MemType.HOST_SIDE)`（外部仓 `cann/shmem` 提供 kernel 实现） |
| `docs/14-hyperconnection-ple-indexer-spec.md:46-49` | 「ngram 表 **128 个分片 / 95.37 GiB**，占整模型 56.2%，**HBM（128 GiB）和本容器 host 内存（cgroup 32 GB）都装不下**。vLLM 的默认答案是 **pinned host memory + UVA 稀疏行查找 + 提前一层的异步 prefetch**。每 token 只查 **16 行 × 320 B = 5 KiB**」 |

**能力探测（[实测]，`evidence/logs/caps.log`）**：`aclrtHostMemMapCapabilities` 逐 hac 类型查：

```
[CAPS] aclrtHostMemMapCapabilities(hac=STARS   ) rc=0 capability=1(SUPPORTED)
[CAPS] aclrtHostMemMapCapabilities(hac=AIV     ) rc=0 capability=1(SUPPORTED)
[CAPS] aclrtHostMemMapCapabilities(hac=AIC/AICPU/PCIEDMA/RDMA/SDMA/DVPP/UDMA/CCU) rc=0 capability=0(NOT_SUPPORTED)
[CAPS] aclrtMallocHost(1048576) rc=0 ptr=0x100000280000
[ATTR] mallocHost    ptr=0x100000280000     location.type=0(HOST)
[CAPS] aclrtHostRegisterV2(MAPPED=0x2) rc=0
[CAPS] aclrtHostGetDevicePointer rc=0 hostPtr=0x100000280000 devPtr=0x40000000000
[ATTR] mappedDev     ptr=0x40000000000      rc=507899 (aclrtPointerGetAttributes failed)
[ATTR] plainDev      ptr=0x120000017000     location.type=1(DEVICE)
[ATTR] rawMalloc     ptr=0x55e996e5b440     location.type=2(UNREGISTERED)
```

要点：**AIV/STARS 报 SUPPORTED**；`aclrtPointerGetAttributes` **对映射后的 device 指针返回 507899**
（对 host 指针、普通 device 指针、未注册指针都 rc=0）⇒ 该 API 在这个映射地址上不可用，是一个已知边界。

### 3.2 正向：AIV kernel 用 `DataCopy`(MTE2) 直接读 host 内存 —— **可行**（[实测]）

kernel 形状：每个 AIV block 从源地址读 `repeats` 个 `tileBytes` 的 tile 到 UB（**背靠背、无中间等待**），
读循环结束用一次 `MTE2_MTE3` 事件排空，再 `MTE3` 写回真 device 内存；host 侧逐字节比对。

`evidence/logs/read_run{1..5}.log`（**5 次独立进程**，`--tile 8192 --repeats 8`，region = 56×8×8192 = 3.67 MB）：

| 进程 | perLaunch (ns) | 带宽 | 设备 cycSpan | 逐字节 |
|---|---|---|---|---|
| read_run1 | 164772.9 | 22.27 GB/s | 1279 | `mismatches=0` PASS |
| read_run2 | 165314.7 | 22.20 GB/s | 1211 | `mismatches=0` PASS |
| read_run3 | 177437.0 | 20.68 GB/s | 1118 | `mismatches=0` PASS |
| read_run4 | 165036.8 | 22.24 GB/s | 1156 | `mismatches=0` PASS |
| read_run5 | 163236.6 | 22.48 GB/s | 1204 | `mismatches=0` PASS |

⇒ **host-mapped 内存被 AIV 的 MTE2 直接读，逐字节正确**。
**本批 5 次独立进程紧密成簇**（perLaunch 163236.6–177437.0 ns，max/min = 1.09）
⇒ **20.68–22.48 GB/s**。5 次全部 `mismatches=0`、设备 `cycSpan` 1118–1279。
注意 `commands.txt` 的整个 run 期间 `m15_layer_loop` 都在（PID 202440→207199）——见 §3.3 并发说明。
**（对照：本会话更早的批次里同型 read 出现过被同卡并发进程饿住的样本——墙钟 ×15 到 2.49 ms，
而设备 `cycSpan` 与逐字节结果不变。"host 读 ~20–22.5 GB/s"这个量级在干净/受扰批次里都成立。）**

### 3.3 带宽与基线（[实测]，区分「空跑」与「实际搬运」）

| mode | 说明 | perLaunch（本批 5 进程，min / 中位 / max） | 有效带宽 | 逐字节 |
|---|---|---|---|---|
| `empty` | 同形状空 kernel（只落周期标记） | 2136.8 / **2488.2** / 3727.6 ns | — | — |
| `hbm` | 3.67 MB 从 **HBM** 读（对照） | 3379.1 / **3401.1** / 3659.5 ns | 中位 **1079 GB/s** | `mismatches=0` PASS |
| `read` | 3.67 MB 从 **host-mapped** 读 | 163236.6 / **165036.8** / 177437.0 ns | **20.68–22.48 GB/s** | `mismatches=0` PASS |

> **并发背景与读法（务必连同数字一起读）**：本批 `commands.txt` 的**跑前快照是
> `m15_layer_loop_`(PID 202440, 6867 MB)**、**跑后快照是 `m15_layer_loop`(PID 207199, 6797 MB)**
> ——即**整个批次期间卡上都另有进程**（PID 变了说明它中途重启过）。**这不是一批「离线」样本**。
> 尽管如此，同批 read 的 5 个样本仍紧密（perLaunch 163236.6–177437.0 ns，max/min=1.09），
> 所以 read 的数量级可用；**但不能再称之为「干净批次」**。
> **关键辨识**：被拉长的样本，其**设备 `cycSpan` 与逐字节结果都不变** ⇒ 纯粹是 host/发射被饿住。
> 因此**带宽类结论取"无干扰聚簇"，不取中位**（中位在"过半样本被饿住"时会误导）。
> **对照**：复审（`reviewer-m86`）在 `m15_layer_loop` 活跃时独立复现 21.27 **`[外部]`** GB/s（该读数不在本仓 evidence 内）
> ⇒ **"host 读 ~20–22.5 GB/s"这个量级在干净与受扰两种批次下都稳健**。

尺寸扫描（本批，每点独立进程，`read`；档位 region = 917504 / 1835008 / 3670016 / 6422528 / 7340032 / 10092544 B）
全在 **20–23 GB/s** 量级
（`tile=8192` 系列；`tile=2048/4096 r=16` 同量级）。**带宽在 0.9–10 MB 上平稳**，
且**减掉 ~2.5 µs 空跑基线后**仍与总量线性 ⇒ 是真实搬运，不是发射开销（同批 `empty` 中位 2488 ns）。

### 3.4 负向对照（[实测]）

| 形态 | mode | 结果 | 证据 |
|---|---|---|---|
| **未注册的 host 指针**当 GM 地址 | `unreg` | **`aclError=507035`**；`mismatches=3655680/3670016`，首坏偏移 0 | `evidence/logs/unreg_unregistered_ptr.log` |
| **只注册一部分**（region 减去最后 1 个 tile），后段读未注册 host VA | `page` | **`aclError=507035`**；`mismatches≈8160–8162`（最末 32B 落盘与否有抖动），**首坏偏移 = 3661824 = 注册边界** | `evidence/logs/page_partial_registration.log` |
| 映射 device 指针 + 非 32B 偏移 | `align --off 1` | **不是故障**：`aclError=0`，与 `Model(i+1)` 逐字节一致（`mismatches=0`） | `evidence/logs/align_off1.log` |
| 同上 off=32 | `align --off 32` | `aclError=0`，与 `Model(i+32)` 逐字节一致 | `evidence/logs/align_off32.log` |
| HBM + 非对齐偏移（对照） | `hbm --off 1` | 同样 `aclError=0`、逐字节平移一致 | `evidence/logs/hbm_off1_contrast.log` |

**两条可判定的结论**：
1. **越界/未注册访问会被硬件拦下**：`unreg` 与「注册区外」都报 **507035**，且 `page` 的首个坏字节
   **精确落在注册边界**（3661824）——这是一个可复现的「边界即故障点」判据。
2. **`DataCopy` 对 `uint8_t` 是字节粒度的**（off=1 都逐字节对），**不是 32B 对齐要求**；且 HBM 路径同样
   如此 ⇒ 「非对齐」不是 host 路径特有的约束。（注意：这与 `probe_v_align` 里 **UB 侧 `LoadAlign`
   必须 32B** 是**不同指令**，不矛盾。）

### 3.5 注册来源的能力边界（ngram 方案的决定性输入，[实测]）

`filemmap` mode 把「内存来源」作为变量，其余照旧（3.67 MB，`tile=8192 r=8`）：

| 内存来源 | `aclrtHostRegisterV2(MAPPED)` | device 读 | 证据 |
|---|---|---|---|
| `mmap(FILE, MAP_SHARED, R|W)` | **rc=507899 拒绝** | — | `filemmap_file.log` |
| `mmap(FILE, MAP_SHARED, R)` | **rc=507899 拒绝** | — | `filemmap_filerd.log` |
| `mmap(FILE, MAP_PRIVATE, R)` | **rc=507899 拒绝** | — | `filemmap_filepriv.log` |
| `mmap(ANON, MAP_PRIVATE|MAP_ANONYMOUS)` | **rc=0** | **rc=0 且 `mismatches=0`**（本批 3 进程 ⇒ 22.04–22.35 GB/s） | `filemmap_anon_run{1..3}.log` |
| `posix_memalign(4096, …)`（堆） | **rc=0** | **22.23–22.40 GB/s，`mismatches=0`** ×3 进程 | `filemmap_malloc_run{1..3}.log` |
| `aclrtMallocHost` | **rc=0** | **20.68–22.48 GB/s，`mismatches=0`** | `read_run*` |

⇒ **file-backed mmap 一律不能注册；匿名/堆/`aclrtMallocHost` 都能。**
（与驱动注释一致：`ascend_hal_base.h:2426` 「HOST_MEM_MAP_DMA don't support read-only memory」、
`:2432` 「Not support vmm va」——文件映射不是它可以 pin 的普通匿名页。）
**注册是否成功的判据是 `rc`，与耗时无关**：`file`/`filerd`/`filepriv` 三种都是 rc=507899（三批归档里稳定复现，
复审也独立复现），所以「能注册」这个结论不受并发/带宽噪声影响。

**注册本身不额外增加常驻**：`filemmap` 里 RSS 在 **touch（填充）后**就到达 ≈ region 大小
（本批 anon：before-mmap 112732 kB → after-touch 115804 kB；malloc：113264 → 115312 kB；region = 3670016 B），
**register 之后再测 delta = 0 kB**（anon/malloc 两种都如此）。
⇒ **推断（未隔离测）**：「页需先常驻，设备才能读到想要的数据」——因为本探针**所有注册测试都是
先 fill/touch 再 register**，**没有**「未 touch 直接注册」的对照，所以这句是推断而非实测结论。
**「注册是否阻止后续回收（真 pin）」本次也未直接测**（见 §6 开放问题 2「未 touch 对照」与 3「真 pin」）。

**注册总量必须远小于 32 GiB cgroup**：窗口池的常驻成本 ≈ 池大小。

### 3.6 注册/注销代价（[实测]，**方差主导 · 不足以用来定池大小**）

`regbench`（匿名 mmap，一次性 `register → getDevPtr → unregister`）。

> **单位注**：下表与 §4.3 里的 **4 / 64 / 256** 指的是 **MiB**（`regbench` 用 `sz >> 20` 算大小，却把标签印成
> "MB"）。本 README 引用日志时沿用日志原样印出的 "MB" 字样，**含义等同 MiB**；与十进制 MB 无关。

**两组读数（都是 tip 上 5 次独立进程；两组都逐次记录了 `npu-smi`）**：

| region | **B 组＝开跑前等到空闲窗口**（本批） | **A 组＝开跑前已有并发**（`c126bb6`） | 中位之比 |
|---|---|---|---|
| 4 MB | 中位 **48.161 ms**，[6.780, 72.969] | 中位 58.235 ms，[53.445, 83.617] | 0.83× |
| 64 MB | 中位 **576.875 ms**，[309.046, 836.124] | 中位 840.184 ms，[638.266, 857.864] | 0.69× |
| 256 MB | 中位 **6730.705 ms**，[1144.784, 10211.197] | 中位 23929.387 ms，[2196.840, 41020.044] | **0.28×** |

| `getDevPtr` / `unregister` 中位（B 组） | 4 MB | 64 MB | 256 MB |
|---|---|---|---|
| getDevPtr | 0.013 ms | 0.013 ms | 0.013 ms |
| unregister | 0.541 ms | 2.129 ms | 8.210 ms |

**并发窗口记录（这是两组的分野，逐条可查）**：
* **B 组** = 本批。`run_probes.sh` 先等「卡上无别的进程」再跑 `regbench`：
  `evidence/regbench_idle_window.txt` 记 `got_idle_window=1（等了 79s）`；
  **regbench 前快照 = "No running processes"**；**regbench 后快照 = `m15_layer_loop`(PID 207199, 6797 MB)**
  ⇒ 窗口是"开跑空闲、跑的过程中被并发挤进来"，**不是完全隔离**。
* **A 组** = `c126bb6` 那批。该 rev 的 `evidence/logs/commands.txt` **跑前快照就写
  `m15_layer_loop_`(PID 193909 **`[git史]`**, 6797 MB)**、跑后为空 ⇒ 开跑时卡上就有别的进程。
  （该 PID 出自 `c126bb6` 这次 commit 的 `evidence/logs/commands.txt`——当前 tip 的 evidence 里没有它，
  要看请 `git show c126bb6:probe_host_dma/evidence/logs/commands.txt`。）

**这组对照回答了「散布是不是并发造成的」**：
1. **是，且影响很大**：256 MB 的**中位从 23929 → 6731 ms（快 3.6×）**，4/64 MB 也各快 1.2×/1.5×；
2. **散布也收窄但没消失**：256 MB 区间宽度 **18.7× → 8.9×**（A 组 `41020/2196`，B 组 `10211/1145`）；
3. **但两组都仍非隔离测量**（B 组跑的中途就被挤进来了），所以**依然不能给出"固有代价"的数**。

⇒ **可下的结论只有定性的**：
1. 注册**确实有实打实的代价**（256 MB 量级在**秒**，并发下可达**几十秒**），`unregister` 便宜得多（~ms 级）；
2. 代价**受并发强烈影响**（上面对照），**单点、甚至单批中位都不足以外推**；
3. 故**设计规则**：窗口**一次注册、反复复用**，绝不按次注册；**池大小必须在目标机（空闲窗口）上实测确定**。

**跨批次全部归档样本**（含 A 组与更早两批）：为让这些**只在历史 rev 里**的数在**当前 tip** 上也能直接 grep，
已把对应 log **逐字节复制**到 `evidence/hist/`（来源 rev + sha256 见 `evidence/hist/README.md`，
也可用 `git show <rev>:probe_host_dma/evidence/logs/...` 复核）：

| region | A 组 | B 组 | 更早两批（单样本） | 全部归档跨度 |
|---|---|---|---|---|
| 4 MB | [53.445, 83.617] ms | [6.780, 72.969] ms | 7.617（`5bf1b9b`）/ 50.391（`103540f`） | **6.78 – 83.62 ms（≈12×）** |
| 64 MB | [638.266, 857.864] ms | [309.046, 836.124] ms | 680.581 / 1009.236 | 309.0 – 1009.2 ms（≈3.3×） |
| 256 MB | [2196.840, 41020.044] ms | [1144.784, 10211.197] ms | **1312.202** / 6827.784 | **[1144.784, 41020.044] ms（≈36×）** |

> **口径说明与自纠**：
> ① 本节最早给过一组**未归档的错值**（是写 `run_probes.sh` **之前**的手工单跑，**从未落进任何 log**）
> ——已删除并改为归档读数。**原始错值不在此复述**（复述就等于把它们再写进仓库一次）；
> 出处用**可跑的追踪命令**（不依赖 `.tower/` 下的 review 文件——那不在 git 里；**此处不复述旧值本身**）：
> 对被移除的每个旧值 `<V>`，`git log --oneline -S "<V>" feat/m86-ngram-mmap-and-host-memory-device-re -- probe_host_dma/README.md`
> 可看到它在哪几次提交里被引入/移除；`<V>` 本身用 `git log -p -- probe_host_dma/README.md` 检索得到
> （R4 复审已用这两条命令验证过：读者能追到"这里曾经错过"，追踪路径真实存在）。
> ② 修掉了 `run_probes.sh` 汇总里一处**正则误咬**：`.*register rc=` 贪婪匹配到 `unregister rc=`，
> 抽出的是**注销**耗时。现在汇总里带**正对照**：并排打印原始行 + `register_extracted=` + `unregister_field=`。
> ③ 原写的「16× 大小 → 29× 时间」**算错了**；既然方差达 36×，**任何「线性/超线性」拟合都不作结论**。
> ④ `README` §5 曾残留一组**在任何 log 里都不存在的旧中位数**——§5 已改为**只给指针、不重复数字**
> （从根上让它不可能过期），并加了 `audit_numbers.py` 冒烟测试（**定位见 §2，是启发式、不是保证**）。
### 3.7 异步语义（[实测]）

* host 侧：launch 之后立刻 `aclrtStreamQuery` → `status=1(NOT_READY)`（异步发射）；
* 核内：MTE2 读循环 `repeats` 个 `DataCopy` **背靠背、无 per-tile 等待**，只在进入 MTE3 写回前排空一次
  ⇒ 读本身是流水化的；`PipeBarrier<PIPE_MTE2>` 未使用（同 pipe 保序靠事件，沿用 m0 结论）。

```
[ASYNC] aclrtStreamQuery immediately after launch: rc=0 status=1(NOT_READY)
[ASYNC] aclrtSynchronizeStream rc=0 (device-side MTE2 read loop is back-to-back, no per-tile wait)
```

---

## 4. ngram（128 分片 / 95.37 GiB）的 mmap 方案

见 `mmap_ngram_plan.py`（`--real` 打印方案，`--demo` 真跑通全链路）。**复核过塔给的数字**：

```
128 × 2,500,012 行 = 320,001,536 行；行宽 = 160 × 2 B = 320 B
总大小 = 102,400,491,520 B = 102.40 GB = 95.37 GiB
单分片数据段 = 800,003,840 B = 762.9 MiB；每 token 查 16 行 = 5120 B
cgroup 32 GiB ⇒ 表是它的 2.98 倍（装不进）；磁盘 402 GB ⇒ 占 25%（放得下）
```

### 4.1 行 id → 文件偏移（跨 128 分片）

```text
shard    = id // 2,500,012                 # 0..127
local    = id %  2,500,012                 # 分片内行号
file_off = data_off[shard] + local * 320   # data_off = 8 + 对齐后的 safetensors JSON 头长
```

`data_off` 每分片一次（读 8 B 头长 → 跳过 JSON）；safetensors 的分片是独立文件，
所以「全局行 id」必须先做 `//`、`%` 跨片换算。`mmap_ngram_plan.py --demo` 用一个真实写的
多分片 safetensors 集合把这条链路**逐字节验证过**（320 抽样行，`不匹配行数: 0`）。

### 4.2 为什么不能「直接注册 mmap 文件让设备读」

[实测] 见 §3.5：file-backed mmap 一律 `rc=507899`。所以：

```text
文件分片 --mmap(MAP_SHARED, PROT_READ, 按需页)--> host page cache
                       |
                       +--(拷贝需要的行/块)--> 注册过的匿名 staging 窗口 --MTE2--> AIV
```

**设备读的是 staging 窗口，不是文件映射本身。**

### 4.3 窗口大小与驻留策略（建议）

* **窗口池起点：4 × 256 MiB = 1 GiB**，`aclrtHostRegisterV2(MAPPED)` **只在启动时注册一次**，之后反复复用。
  依据：§3.6 注册代价**大且极不稳定**（256 MB 单次跨归档样本 **1144.784 ms ↔ 41020.044 ms**，≈36×；且**受并发强烈影响**：开跑空闲的 B 组中位 6730.705 ms vs 开跑有并发的 A 组 23929.387 ms）
  ⇒ **池大小必须在目标机上实测后再定**，本文不给可照搬的秒数。
  §3.5 注册不额外增常驻，池的常驻成本 ≈ 池大小（1 GiB 只占 cgroup 3.1%）。
  ⇒ 若启动预算紧，方向是**减总量**（如 8×64 MiB=512 MiB），用更高的 miss 率换启动时间；
  但**单价能否随窗口变小而下降，本文的 regbench 数据回答不了**（方差盖过了尺寸效应）。
* **搬运粒度按「行块」**：ngram 的访问是稀疏行查找（每 token 16 行 × 320 B），
  所以 staging 以「文件内对齐的块」（如 256 MiB）搬，行落在块内偏移；命中率由请求局部性决定。
* **一个 token 的搬运量极小**（5 KiB），真正的开销是**块填充**（miss 时一次 256 MiB，约 11.4 ms @22 GB/s，
  若命中 page cache 更快）⇒ 策略上要**提前一层异步 prefetch**（`docs/14:48` 记录 vLLM 默认就这么做）。
* **不要**注册整表（注册即 pin，102 GB > 32 GiB cgroup，且 256 MB 单次就要 4.76 s）。

### 4.4 与设备侧的接口（已实测）

* staging 窗口 = `mmap(ANON)` 或 `posix_memalign` 或 `aclrtMallocHost` 皆可（§3.5 三者都 rc=0）；
* 取到 `aclrtHostGetDevicePointer` 的 device 指针后直接当 `GM_ADDR`；同一次注册内可反复 launch；
* **越界会 507035**（§3.4）⇒ 每个窗口的注册长度必须 ≥ kernel 可访问的最大偏移；
  staging 池要做边界检查，不要依赖「多读一点没关系」。

---

## 5. 命令 → 输出/rc 表（都在本目录实跑过）

| 命令 | 关键输出 | rc |
|---|---|---|
| `./build/probe_host_dma caps` | `AIV ... SUPPORTED`；`HostRegisterV2 rc=0`；`devPtr=0x40000000000` | 0 |
| `./build/probe_host_dma read --launches 20` | `[VERIFY] PASS`，`mismatches=0`（带宽**逐批不同**，见 `evidence/summary.txt` 的 read 行与 §3.2/§3.3） | 0 |
| `./build/probe_host_dma hbm --launches 20` | `[VERIFY] PASS`（带宽见 `evidence/summary.txt` 的 hbm 行，逐批不同） | 0 |
| `./build/probe_host_dma empty --launches 20` | 空跑基线（逐批不同，见 `evidence/summary.txt` 的 empty 行与 §3.3） | 0 |
| `./build/probe_host_dma unreg` | `aclError_after_last=507035` | 0（程序自身；设备报错记录在 log） |
| `./build/probe_host_dma page` | `507035`，`firstBadOff=3661824` | 0 |
| `./build/probe_host_dma align --off 1` | `aclError=0`，`VERIFY-SHIFT mismatches=0` | 0 |
| `./build/probe_host_dma filemmap --mapmode file` | `HostRegisterV2 rc=507899`，`overall=REGISTER-REJECTED` | 3 |
| `./build/probe_host_dma filemmap --mapmode anon` | `rc=0`，`[VERIFY] PASS` | 0 |
| `./build/probe_host_dma regbench` | 5 次独立进程；**具体读数不在此重复**，见 `evidence/logs/regbench_run{1..5}.log` 与 `evidence/summary.txt` 尾部的 regbench 汇总（与 §3.6 同源） | 0 |
| `$PY mmap_ngram_plan.py --real` | 真实几何方案（见 §4） | 0 |
| `$PY mmap_ngram_plan.py --demo` | `不匹配行数: 0`，`PASS` | 0 |
| `bash check_fingerprints.sh` | `PASS: commands.txt 记的源码指纹与当前工作树逐字节一致` | 0 |
| `$PY audit_numbers.py README.md evidence` | `PASS（无 FLAG）`（**启发式冒烟测试**，rc=0 不构成可复现性保证；定位与已知口子见 §2）；读数见 `evidence/numbers_audit.txt` | 0 |

**判据命令的 locale 一致性**（正则字符类只用 ASCII）：`evidence/logs/summary.txt` 的抽取只用
`[0-9.-]`/字面量，`LC_ALL=C` 与 `LC_ALL=C.UTF-8` 下读数一致（复核见 `evidence/locale_check.txt`）。

---

## 6. 仍存疑 / 开放问题（不猜，如实列出）

1. **设备侧 `SYS_CNT` 跨度的「16 tile 悬崖」未解释**：[实测] 当每 block 的 `DataCopy` 数 ≥ 16 时，
   `[t0,t1]` 跨度 ≈ host 墙钟（本批 `read_sweep_r16`：cycSpan 321390 vs perLaunch 324946.9 ns
   ⇒ 尺度上一致，约 1 cycle/ns 量级）；
   ≤ 14 时跨度只有 ~1.2k cycles（`read_sweep_r8`/`read_tile8192_r14`），**远小于墙钟**。
   两种 tile 尺寸（2048/4096/8192）下阈值都落在 **count=16** 而非字节数。机制未定，
   **不作为任何结论的依据**；带宽结论只取 host 墙钟口径。
2. **「未 touch 直接注册」的对照未做**：本探针所有注册测试（`read`/`filemmap`/`align`/`page`）
   **都是先 fill/touch 再 register**，所以「页必须先常驻，设备才读到想要数据」只是**推断**，
   没有被隔离验证（见 §3.5 的推断标注）。要证需要「mmap 后不 touch 直接注册并让设备读」的对照。
3. **「注册是否真 pin（阻止后续回收）」未直接测**：只测到「register 相对 touch 不额外增 RSS」
   （delta=0 kB）。要判定真 pin，需要注册后 `madvise(MADV_PAGEOUT)` 再测 RSS/设备读是否正确——
   本次未做（有设备故障风险）。ngram 方案按「注册窗口 = 常驻内存」保守记账。
4. **`ACL_HOST_REG_PINNED` / `ACL_HOST_REG_IOMEMORY` / `ACL_HOST_REG_READONLY` 三个 flag 未逐测**
   （只测了 `ACL_HOST_REG_MAPPED=0x2`）。`READONLY` 是否能让 file-backed mmap 通过？未测。
5. **`aclrtHostMemMapCapabilities` 里 `AIV` 报 SUPPORTED，但只验证了「AIV 能读」**；
   **写** host（AIV→host 回写）未测。ngram 只读，故未展开。
6. **`aclrtPointerGetAttributes` 对映射 device 指针返回 507899** 的含义未查证（是「不支持该地址」
   还是别的）；未影响本探针（只用它做打印，失败即跳过）。
7. **未做 page-cache 冷启动实验**：demo 里文件与页都热（`drop_caches` 需要额外权限，未动）。
   「miss 时块填充 11.4 ms」是按 22 GB/s 的**推导**，不是实测。
8. **并发背景逐批不同，必须逐批读 `commands.txt`**：本设备是共享卡，`m15_layer_loop`（M84/M85）
   经常在跑。`c126bb6` 那批：**跑前快照见到 `m15_layer_loop_`(PID 193909 **`[git史]`**)，跑后为空**；
   更早的批次也见过「两次快照都空、但同批样本被拉长」（并发出现在两次快照之间）。
   故所有结论都用**多次独立进程 + 区间/聚簇**口径，未依赖单点；regbench 另有 `WAIT_IDLE`
   空闲窗口机制与 `evidence/regbench_idle_window.txt` 逐次快照记录。
9. **`regbench` 的注册耗时受并发强烈影响，且仍有残余散布未解释**：
   B 组（开跑前等到空闲窗口）5 次区间 4 MB [6.780, 72.969] ms、64 MB [309.046, 836.124] ms、
   256 MB [1144.784, 10211.197] ms；连同 A 组与更早两批，**256 MB 跨归档跨度 1144.784 ms ↔ 41020.044 ms（≈36×）**。
   §3.6 的 A/B 对照**已坐实并发是主因之一**（256 MB 中位 23929.387 ms → 6730.705 ms），
   但**两组都非隔离测量**（B 组跑到中途 `m15_layer_loop` 就进来了），
   所以「完全无并发时的固有代价」仍**未测到**，残余散布（B 组内仍有 8.9×）来源不明。
   ⇒ **§3.6/§4.3 因此不给可照搬的秒数**。

---

## 7. 证据索引（`evidence/`）

| 路径 | 内容 |
|---|---|
| `logs/commands.txt` | 复现命令、`HEAD_at_run`、**源码 sha256 指纹**、CANN 版本、**跑前/跑后 npu-smi（并发背景）** |
| `logs/caps.log` | 能力面普查（hac 能力表、四类指针属性、注册 rc） |
| `logs/empty_run{1..5}.log` | 空跑基线（独立进程 ×5） |
| `logs/hbm_run{1..5}.log` | HBM 对照（独立进程 ×5，逐字节 + 带宽） |
| `logs/read_run{1..5}.log` | **正向 host-mapped 读**（独立进程 ×5，逐字节 + 带宽） |
| `logs/read_sweep_r{2,4,8,16,22}.log`、`read_tile2048_r16`、`read_tile4096_r16`、`read_tile8192_r14` | 尺寸扫描（含 16-tile 悬崖的两侧） |
| `logs/filemmap_{file,filerd,filepriv}.log` | **file-backed mmap 全被拒**（rc=507899） |
| `logs/filemmap_{anon,malloc}_run{1..3}.log` | 匿名/堆注册成功且逐字节读对（独立进程 ×3） |
| `logs/unreg_unregistered_ptr.log`、`logs/page_partial_registration.log` | 负向对照（507035；边界即首坏字节） |
| `logs/align_off1.log`、`align_off32.log`、`hbm_off1_contrast.log` | 非对齐读 = 字节粒度（正对照） |
| `logs/regbench_run{1..5}.log` | 注册/注销代价（4/64/256 MB）**×5 独立进程**；中位+区间见 `summary.txt` 尾部 |
| `hist/` | **历史 rev 的 regbench 原始 log 逐字节副本**（`c126bb6`=A 组 5 份、`5bf1b9b`/`103540f` 各 1 份）+ 来源与 sha256（`hist/README.md`） |
| `regbench_idle_window.txt` | 跑 regbench 前后的 `npu-smi` 快照 + 是否等到空闲窗口（§3.6 A/B 两组的分野依据） |
| `recon_quotes.txt` | **能力面勘查原文**（CANN 头文件片段，grep/sed 实跑输出） |
| `locale_check.txt` | 判据命令的 locale 一致性 + 扫描正对照 |
| `fingerprint_check.txt` | 源码指纹自检输出（commands.txt 记录 vs 当前工作树） |
| `numbers_audit.txt` | **`audit_numbers.py` 的真实运行读数**（每个数归类为 IN-EVIDENCE / 紧贴逃逸标记 / 作者自证 / 结构引用 / FLAG） |
| `audit_negative_controls.txt` | **冒烟测试的 4 个负向对照**（平铺假数 / 假中位数 / 带 `[外部]` 的假数 / 塞进普通名 evidence 文件），每个给 rc 与 FLAG 数 |
| `mmap_plan_real.txt` / `mmap_plan_demo.txt` | ngram mmap 方案（真实几何 / 小几何全链路 demo） |
| `summary.txt` | 关键行汇总表 |
| `logs/build.log`、`build_configure.log` | 构建输出 |
