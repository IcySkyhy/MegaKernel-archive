# M16：L0 装载几何标定（bf16，L1(Nz) → L0B / L0A，2D `LoadData2DParamsV2` + 3D `enTranspose`）

M24（attention decode BMM2 的 `P·V`）卡在"L0B 装载几何"上：3D `enTranspose` 的源偏移静默不生效、
2D `ifTranspose` 多组挂死，且最小探针"参数有效果但读数不可复现"。本工程把这件事**当作标定问题**
查到底，产出：

1. **一份逐字段的几何映射表**（本 README §3）：`LoadData2DParamsV2` 的 8 个字段各自控制哪个轴、
   单位是什么、是否生效；并且是**逐点验证过的模型**（不是外推）。
2. **一套可信的探针方法学**（§2）：为什么"1 conf/launch"还不够，必须再加"每次 launch 前显式清零
   目的 buffer" —— 这是 M24"读数不可复现"的真正机制。
3. **真实 BMM2 形状的验证表**（§4）：M=16、K=256、N=128/256 的 `P·V` 用标定结果预测再实测，
   在 M24 的真实 P/V dump 上**逐位**等于 numpy/fp32 参考。
4. **对现有仓库代码的警示**（§5）：哪些地方在"照抄别的形状的参数"。
5. 明确的【已标定】/【仍存疑】边界（§6）。

> 结论先行：**BMM2 的 V^T 不需要 3D 转置装载**。V 以 `[k=S2][n=HD]` 行主序存放时，
> `Nd2Nz`（`dstNzC0Stride=K`）进 L1 后，用 **2D `LoadData2DParamsV2` +
> `ifTranspose=true`、`mStep=K/16`、`kStep=N/16`、`srcStride=K/16`、`dstStride=N/16`** 即可让
> `B_mmad[k][n] = V[k][n]`（§4 实测逐位一致）。反过来，**3D `LoadData3DParamsV2` 到 L0B 在 29 组 conf 里
> 一次写入都没观察到**（§3.6；注意是「未观察到」而非「证明无此通路」），而文档也明说该通路
> 「自动转置、`enTranspose` 无效」—— M24 选的正是这条走不通的路线（限度见 §3.6 末）。

---

## 1. 交付物与工程结构

| 文件 | 作用 |
|---|---|
| `reproduce.sh` | **一键复现**：构建 → 2D 探针 ×5（确定性证据）+ 3D 探针 → 6 种 BMM2 运行（合成/map/dump/kmap/baddst/real）→ 独立 numpy 复核 → 隔离运行判定实验 → 校验和核对（约 6 分钟，串行；勿并发占用 NPU） |
| `m16_geom.asc` | 标定探针（AIC-only，单 AIC 核）：`./m16_geom 2d [lo hi]` / `./m16_geom 3d [lo hi]`，逐 conf 输出源坐标映射表 + FNV 校验和，并 dump 原始 L0C 到 `m16_geom_{2d,3d}_raw.bin`（+ sidecar `.tsv` 记录每个 conf 的字段） |
| `m16_pv.asc` | 真实形状验证：`./m16_pv`（合成）/ `dump` / `map`（直接读回 V 的映射）/ `baddst`（反例）/ `kmap`（K 轴探针）/ `real <p.bin> <v.bin>`（用真实 dump）；配方常量由 kernel 与"预测配方"打印**共用**，不会分叉 |
| `tools/analyze_geom.py` | 把 raw dump 解成"哪个源分形落在哪个 L0B 槽位"，逐 conf 打印 |
| `tools/check_isolated.py` | P2-2 判定实验：对"非确定集合"里的 conf 做单 conf 隔离运行，验证 in-range 槽位可复现 |
| `tools/check_determinism.py` | 从 `geom2d_run{1..5}.log` 生成 `geom2d_determinism.txt`（由 reproduce.sh 调用） |
| `tools/make_sha256.py` | 生成 `evidence/dump_sha256.txt`（只覆盖数据产物，不含 log） |
| `tools/refresh_evidence.sh` | **一键刷新归档**：reproduce.sh → 复制产物/日志进 evidence/ → 重算校验和 → 校验 |
| `check_ref.py` | **独立 numpy 复核**：`check_ref.py pv` 复算 `P·V`；`check_ref.py geom2d` 用 §3 的模型**逐点**预测全部 conf 并比对 |
| `evidence/` | `reproduce_run.log`（一键复现的完整输出）+ 逐模式的 `pv_{synth,map,kmap,real,baddst}.log` + `geom2d_run1..5.log`（确定性）+ `geom2d_isolated_run.txt`（P2-2 判定实验）+ `check_ref_*.log` + raw dump/sidecar + **`dump_sha256.txt`（校验和）** + M24 的 P/V dump 切片（**含提取命令，见 §4.1**） |

> 本机 `python3`（`/usr/bin/python3`）**没有 numpy**；所有 Python 步骤请用
> **`/usr/local/python3.12.13/bin/python3`**（numpy 2.5.1）。下面命令里的 `python3` 一律指它。

构建/运行（`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后）：

```bash
bash m16_load_geom/reproduce.sh              # ← 一步到位（构建 + 全部运行 + 独立复核）
# 若要连 evidence/ 归档一起刷新（跑 + 复制 + 重算校验和 + 校验），用：
bash m16_load_geom/tools/refresh_evidence.sh
```

逐步手动跑（与 reproduce.sh 等价）：

```bash
PY=/usr/local/python3.12.13/bin/python3     # 干净 shell 里的 python3 没有 numpy，必须用这个
$PY -c "import numpy; print(numpy.__version__)"   # 期望 2.5.1
cmake -B m16_load_geom/build -S m16_load_geom -DCMAKE_BUILD_TYPE=Release
cmake --build m16_load_geom/build -j8
cd m16_load_geom/build
./m16_geom 2d > geom2d_run1.log 2>&1          # 49 个 conf，逐 conf 一次 launch + launch 前清零 L0B
cp m16_geom_2d_raw.bin m16_geom_2d_raw_run1.bin # 第 1 次的 dump（命名与内容对应同一次运行）
for r in 2 3 4 5; do ./m16_geom 2d > geom2d_run$r.log 2>&1; done   # 确定性证据
$PY ../tools/check_determinism.py
./m16_geom 3d                     # 29 个 conf
./m16_pv && ./m16_pv map && ./m16_pv kmap && ./m16_pv baddst
./m16_pv dump                     # 合成数据落盘（m16_pv_{p,v,c_device,c_ref}.bin）
./m16_pv real ../evidence/m24_s256_P_unit0_par0.bin ../evidence/m24_s256_V_n2_0.bin
$PY ../check_ref.py pv
$PY ../check_ref.py geom2d m16_geom_2d_raw_run1.bin m16_geom_2d_raw_run1.bin.tsv
$PY ../tools/analyze_geom.py m16_geom_2d_raw_run1.bin m16_geom_2d_raw_run1.bin.tsv
$PY ../tools/check_isolated.py    # P2-2 判定实验（约 3 分钟：11 个 conf × 2 次隔离 launch）
```

> `./m16_geom 2d` 会把结果写 `m16_geom_2d_raw.bin` + `.tsv`；`check_isolated.py` 会用这两个文件跑
> 单 conf 隔离运行并覆盖它们，所以**先跑 `check_ref.py geom2d` 再跑 `check_isolated.py`**，
> 或跑完 `check_isolated` 后重跑一次 `./m16_geom 2d`。

---

## 2. 探针方法学：读数为什么不可复现，以及怎么修

### 2.1 读回装置

- 源：GM 里一张 `[32 行][32 列]` bf16，元素用**可逆位型编码** `bits = 0x4000 + (r*32+c)`
  （指数域 ∈[0x80,0x88)，全为正规数、两两不同，bf16 精确）。解码只做整数运算：`k = fp32 高 16 位 - 0x4000`。
  **M24 的 `i*1000+j` 不是 bf16 精确值**（1000 与 1001 在 bf16 上同值）——这是它读数不可解释的原因之一。
- L1：`Nd2Nz(nValue=32, dValue=32, srcDValue=32, dstNzC0Stride=32)` ⇒ 源 Nz 布局：
  元素 `(r,c)` 在 L1 的元素下标 `(c/16)*512 + r*16 + (c%16)`；**分形 `(r1,c1)` 在"512B 单元"下标 `r1 + 2*c1`**。
- 读回：L0A = 16×16 单位阵（m0 已验证的单分形装载），`mmad m=16, n=64, k=16`
  ⇒ `C[m][n] = Σ_k A[m][k]·B[k][n] = B_mmad[k=m][n]` ⇒ **L0C 就是 mmad 眼里的 L0B 内容**，
  host 反查每个元素的源 `(r,c)`。
- 全部跨 pipe 次序用 m0 已验证的 `SetFlag/WaitFlag<HardEvent>`（`PipeBarrier` 只用于同 pipe）。

### 2.2 三处方法学缺陷（全部实测确认）

| # | 缺陷 | 证据 | 修复 |
|---|---|---|---|
| 1 | 多个 conf 共用一次 launch，conf 之间不重置 L0B ⇒ 不写 L0B 的 conf 会读到前一个 conf 的内容 | M24 `m10_l0probe` §6 已记录 | **每次 launch 只跑 1 个 conf**（host 循环 + 逐个 sync） |
| 2 | **"1 conf/launch"仍然不够**：L0B 内容会跨 launch 残留，一个静默不生效/NOP 的 conf 照样读到上一次 launch 的 L0B | **已归档的确凿证据（同一进程内）**：conf 48（`zeroFill=0` + `mStep=kStep=0` 的 NOP 装载）**逐位复现同一进程里前一个 conf（47）的输出**（`evidence/geom2d_run1.log`，reviewer 也独立复现了 conf 48 ≡ 47/46）。**跨进程残留属推断**（标定早期调试时见过同一单 conf 三次独立运行给出三个不同结果，但那次 pre-fix 日志未归档；见 §6【仍存疑】） | **每次 launch 在跑 conf 之前显式清零 L0B**（用 m0 已验证的单分形 `ifTranspose=1` 装载，从全零 L1 逐分形写 `l0B[256*i]`，i<8）；conf 00 = 「清零后跑 NOP 装载」自检 conf（`zeroFill=1` + `mStep=kStep=0`，期望且实测全 0） |
| 3 | 编码值不是 bf16 精确 ⇒ 解码会串 | `0x4000 + idx` 位型编码（可逆） | 见 §2.1 |

> 本文里的 **conf 编号 = 探针 conf 表的数组下标 = 标签 `2d-xx` 里的 `xx`**（`2e54a89` 已把三处错位的标签改齐；改名前的日志里 index 46/47/48 的标签曾是 `2d-47`/`2d-48`/`2d-46`）。

### 2.3 清零之后的确定性（5 次独立进程运行）

`evidence/geom2d_determinism.txt`：

- 全部**良构** conf（参数落在源范围内的多分形 conf + A/B/C 三段方法学自检）在 5 次运行中**逐位一致**（逐 conf FNV 相同）；
- 5 次运行中出现不一致的 conf 集合 = `{24,27,28,29,36,38,41,42,43}`（archive 的 `geom2d_determinism.txt` 给出逐对
  比对与分组；reviewer 自己的 5 次里还多出 `30`/`44` —— 这类 conf 的越界内容本身不稳定，
  具体集合会随设备残留浮动）。
  **准确的说法是「非确定集合 ⊂ 窗口内越界读集合」，不是「恰好等于」**：越界读但越界数据落在回读窗口（n<64）之外、
  或落在稳定为零的 L1 区域（如 conf 22 的 `kStep=4`、conf 31/45 的 `srcStride=-2`），实测是**确定**的；
  反之 `srcStride=3`、`mStart=1`、`kStart=1`、`mStart=16`、`kStart=16` 这类越界数据落进窗口的 conf 就不确定。
- 这些 conf 读出的是越界 L1 的残留内容 ⇒ **任何越界/错参数组合的读数都不可解释，不用于推断**。
- §3.4 里 mStartPosition/kStartPosition 的两行判据：**单 conf 隔离运行**（每个 conf 2 次）后
  这些 conf 的 **in-range 槽位逐位一致**、不一致只出现在越界槽位
  （`evidence/geom2d_isolated_run.txt`，`tools/check_isolated.py` 可复跑）。
  因此这两条结论只引用 in-range 槽位，整 conf 的 FNV 不作为判据。
  **样本量小的限度**：每个 conf 只跑了 2 次隔离运行（见 §6）。

⇒ **通用规则**：探针/调试代码在读数之前必须清零目的 buffer；并且只有"参数保证落在源范围内"的
conf 才能拿来推断语义。M24 的"参数有效果但读数不可复现"，机制就在这里。

---

## 3. 几何映射表（核心交付）

### 3.1 L0B 在 mmad 眼里的地址（已标定）

由 `2d-09..2d-12`（单分形装载 + 目的偏移 0/256/512/768 元素）逐点确认：

```
mmad 的 (k, n)  ↔  L0B 元素下标 = 16·n + k        （k<16、n<64 的窗口内）
                 ↔  L0B 行（512B 分形槽） = n/16，行内偏移 = (n%16)·16 + k
```

- **L0B 一块 512B 分形槽里，连续的 16 个元素（32B）是 mmad 的 k**；按 n 每 16 个换一次行。
- 分形行号的一般形式（Zn 排布）：`行 = (k/16)·(N/16) + (n/16)`，即 **k 分形为大方向、n 分形为小方向**。
- `dstOff` 的元素单位：+256 元素 = +1 行 = mmad 的 n 分形 +1（+1024 元素落到窗口外，读回全 0）。

### 3.2 单个 16×16 分形内的变换（已标定）

源分形基准 `(r0, c0)`（`r0,c0 ∈ {0,16}`）落在 L0B 第 `R` 行后，mmad 看到的：

| `ifTranspose` | L0B 分形内容 | mmad 看到 | 判据 conf |
|---|---|---|---|
| `false`（T0） | 源分形按行主序**原样**（分形内不转置） | `B[k][n] = src(r0 + n%16, c0 + k)` | `2d-01`(T0)、`2d-18` |
| `true`（T1） | 分形内做 16×16 **转置** | `B[k][n] = src(r0 + k, c0 + n%16)` | `2d-01`(T1)、`2d-32` |

⇒ **要 `B_mmad[k][n] = V[k][n]`（BMM2 的 V 以 `[k,n]` 行主序存放）必须用 `ifTranspose=true`**；
用 `false` 拿到的是转置（这就是"多 fractal 转置装载"需求的本质——它不是"加一个转置选项"，
而是"选 T0 还是 T1"）。

### 3.3 多分形搬运的迭代与落位（已标定，且逐点验证）

```
for k1 in [0, kStep):                  # 拷贝序：k1 外层
  for m1 in [0, mStep):                #          m1 内层（后写覆盖先写）
      源分形单元 u_src = (mStartPosition + m1) + (kStartPosition + k1) * srcStride
      目的分形行 u_dst = (m1 * dstStride + k1)      # ifTranspose = true
      u_dst          = (m1 + k1 * dstStride)        # ifTranspose = false
      （srcOff/dstOff 以"512B 单元 = 256 元素"为单位整体平移 u_src / u_dst）
      分形 (r1,c1) = (u_src % 2, u_src / 2)         # 本探针 R_pad=32 的源；
                                                    # 一般地 r1 = u_src % (R_pad/16)，c1 = u_src / (R_pad/16)
```

⇒ 与官方文档的配方完全一致（`examples/.../load_data_2dv2_l12l0` §5.4.2、§5.3）：
`srcStride` = 源**列方向**相邻分形的 512B 间隔（= `R_pad/16`），`dstStride` = 目的地 **k 方向**分形行距
（= `N/16`），`mStep`/`kStep` = 源行/列方向的搬运分形个数。

### 3.4 逐字段标定表

| 字段 | 官方文档说法 | **实测语义（bf16，L1(Nz)→L0B，2D V2）** | 单位 | 生效判据（传两个值看结果是否变） |
|---|---|---|---|---|
| `mStep` | 「源矩阵 M 轴方向搬运长度，单位 16 元素」 | 拷贝循环 `m1 ∈ [0,mStep)`：**源行方向的分形个数** | 分形（=16 行） | `1` vs `2`：conf 19 vs 18 输出不同（少搬 2 个分形）；`4` 与 `2` 在本 2×2 分形源上等价（源已搬完） |
| `kStep` | 「源矩阵 K 轴方向搬运长度，单位 32 字节」 | 拷贝循环 `k1 ∈ [0,kStep)`：**源列方向的分形个数** | 分形（=16 元素） | `1` vs `2`：conf 21 vs 18 输出不同（k1=1 的列分形没搬） |
| `srcStride` | 正文「源矩阵 K 方向前/后分形起始地址间隔」；**样例 README 却写"row 方向"** | **源列方向**相邻分形在 512B 单位的间隔；源行方向间隔隐式 = 1（连续） | 512B | `2`→`1`：conf 18 vs 23 不同（重复读同一分形）；`2`→`3`：conf 18 vs 24 不同（越界）⇒ **正文对、样例 README 错** |
| `dstStride` | 正文「目标矩阵 K 方向前/后分形起始地址间隔」 | **目的分形行号 = (k/16)·dstStride + (n/16)** ⇒ `dstStride` 必须 = L0B 的 **n 分形数 = N/16** | 512B | `2`→`1`：conf 18 vs 25 不同；`2`→`4`：conf 18 vs 26 不同 |
| `mStartPosition` | 「源 M 轴起始位置，**单位 16 个元素**」 | **源分形起始下标（行方向）**，直接加在 `m1` 上；只作用在源、不改变目的槽位。对 bf16，沿 M 轴 16 个元素 = 1 个 M 分形行 = 该单位与"分形下标"**等价** | 16 个 M 元素（= 1 个 M 分形行） | `0`→`1`：conf 18 vs 27，**仅比较窗口内 in-range 槽位**（整 conf FNV 非确定，见 §2.3）内容整体平移 1 个分形，与模型逐点一致（隔离运行证据 `evidence/geom2d_isolated_run.txt`）；`16` → 源读越界，窗口内无 in-range 槽位 |
| `kStartPosition` | 「源 K 轴起始位置，**单位 32 字节**」 | **源分形起始下标（列方向）**，加在 `k1` 上（地址上表现为整体平移 `kStartPosition × srcStride` 个分形）。对 bf16，沿 K 轴 32 字节 = 16 个 K 元素 = 1 个 K 分形列 ⇒ 与"分形下标"等价 | 32 字节（= 1 个 K 分形列） | `0`→`1`：conf 18 vs 28，**仅 in-range 槽位**平移 2 个分形（= srcStride），与模型逐点一致；`16` → 越界 |
| `ifTranspose` | 「对每个分形矩阵进行转置」 | **分形内 16×16 转置**（§3.2）；分形间的重排由"m/k 轴互换 + `dstStride` 语义"表达 | bool | `false` vs `true`：conf 18 vs 32 输出不同（`B[k][n]` 换成 `B[k][n] = src(r0+k,c0+n)`） |
| `sid` | 预留，配 0 | 配 0（未单独标定） | — | — |
| `mStep=0` 或 `kStep=0` | NOP | **NOP**（清零后读回全 0） | — | conf 00（清零 + `mStep=kStep=0`）读回全 0 |
| `srcStride<0` | 公式里写 `\|srcStride\|`（取绝对值） | **不取绝对值**，按有符号用 ⇒ 负值把源读到 buffer 之前（越界）；实测等价于"k1=1 的列分形写到垃圾" | — | conf 31（T0）/ 45（T1）与 kStep=1 的 conf 21/35 输出同值 |
| （非字段）`srcOff` / `dstOff` | API 无此参数，用 `LocalTensor` 切片表达（M24 实证：装载的目的偏移**只能**靠切片） | 元素偏移（bf16 = 2B/元素）；`srcOff` 加源 L1 地址、`dstOff` 加 L0B 地址 | 元素 | conf 3/4/5（srcOff 256/512/768 → 分别取到源分形 1/2/3）、conf 10/11/12（dstOff 256/512/768 → 落在 L0B 行 1/2/3） |

### 3.5 与官方文档的对照（1 处真分歧 + 2 条单位澄清 + 1 条真差异（细节））

**只有 1 处真分歧 + 2 条单位澄清 + 1 条真差异（细节）**（表内 4 行逐条对应；先前的第 3 条单位换算（`mStep`/`kStep`）见下方注与 §3.4，不在表内、也不另计条目。`2e54a89` 修正：先前把 3 条单位澄清写成了"不一致"，其中一条还与 §3.3 自相矛盾，已改；**M73 按表内 4 行的标签校正计数口径**：第 4 行是 `|srcStride|` 的**真差异（细节）**，不计入单位澄清）。

| # | 项 | 结论 |
|---|---|---|
| 1 | **`srcStride`/`dstStride` 的轴向 — 唯一真分歧（措辞）** | API 正文（`LoadData_2D_V2.md:80-81`）写「**K 方向**前一个分形起始地址与后一个分形起始地址的间隔」；官方样例 README（`load_data_2dv2_l12l0/README.md:353`）把两者都写成「**row 方向**」。实测：**`srcStride` 是源列（K）方向的 512B 分形间隔**（合正文）；**`dstStride` 的实测语义是「目的分形行号 = (k/16)·dstStride + (n/16)」**，即它等于目的 L0B 的 n 分形数 —— 样例算例里 `dstStride = nAlignL0/16` 与该式数值相同，所以两种措辞都"能算对"，一直没暴露。 |
| 2 | **`mStartPosition` 的单位（澄清，非分歧）** | 文档「M 轴方向，单位 16 个元素」与实测的"源 M 分形行下标"**是同一件事**：对 bf16，沿 M 轴 16 个元素正好 = 1 个 M 分形行（`LoadData_2D_V2.md:46` 说 b16 分形为 16×16）。样例 README `:354` 更直接写「本次搬运在 L1 源矩阵中的起始**小分形位置**」。**先前的"不是元素级平移"写法已删除**。 |
| 3 | **`startAddr` 公式（澄清，非分歧）** | 文档 `LoadData_2D_V2.md:128` 的 `startAddr = srcAddr + (kStartPosition×\|srcStride\| + mStartPosition)×512B` 在 `m1=k1=0` 时与 §3.3 的模型 `u_src = (mStartPosition+m1) + (kStartPosition+k1)×srcStride`（u 的单位就是 512B）**逐项一致**。**先前的"在 bf16 上不成立"是错的，已删除**（唯一细节差异见第 4 条）。 |
| 4 | **`\|srcStride\|` 取了绝对值 — 真差异（细节）** | 文档公式写 `\|srcStride\|`；实测按**有符号**使用（负值把源读到 buffer 之前 ⇒ 非法组合，见 §3.4 的 `srcStride<0` 行）。 |

> 单位换算（记住这一句就够）：**对 bf16，文档的 per-axis 单位 `M 轴 16 元素` / `K 轴 32 字节`
> 分别等于"1 个 M 分形行"与"1 个 K 分形列"**，与实测的"源分形下标"完全等价；
> `mStep`/`kStep` 同理（样例 README `:353` 用的措辞就是"小分形**个数**"）。
> 真正要小心的不是"文档单位不对"，而是**分形下标 ≠ 线性字节偏移**：K 轴那个起始位置在地址上
> 要乘 `srcStride` 才变成分形偏移（文档公式正是这么写的）。

### 3.6 3D `LoadData3DParamsV2` → L0B：**29 组 conf 均未观察到写入**（限度见本节末）

文档（`LoadData_3D.md`）明确写着：

- 「非转置场景下，`L1 Buffer->L0B Buffer` 通路**不支持**。`L1 Buffer->L0B Buffer` 通路下**会自动进行
  转置**，不需要配置 `enTranspose`，此时 `enTranspose` 参数**无效**。」
- `enTranspose` 有效条件（950PR）：目的为 L0A(A2) 且源为 b8/b16/b32。
- 950PR 上「必须使用辅助配置接口 `SetLoadDataRepeat` 配置 `dstStride`」。

实测（`2d` 之外的 `3d` 模式，29 组 conf）：**没有任何一组写入过数据**（读回全 0），包括
- 按 2D 矩阵形态解释的 `l1H/l1W/channelSize`；
- 按 NC1HWC0 解释（`H=C/16, W=R, C0=16`）的组合；
- `enTranspose` 0/1、`kExtension`/`mExtension`/`channelSize`/`l1H`/`l1W`/`mStartPt`/`kStartPt`/
  `strideW`/`filterW`/`srcOff`/`dstOff` 各自的扰动。

⇒ 结论：**3D 装载不是 BMM2 V^T 的可行路线**（文档说它到 L0B 自动转置且 `enTranspose` 无效，
实测在 29 组 conf 的回读窗口内一次写入都没观察到）。M24 用「3D `enTranspose` 转置装载 V」命中了两重坑
（`enTranspose` 对 L0B 是静默 no-op + 该通路本就自动转置）。
**3D→L0A（文档承认 `enTranspose` 有效的唯一通路）本 mission 未标定**，见 §6。

**这条负结论的限度（`2e54a89` 补写）**：
- 观察范围只有**回读窗口**（L0B 行 0–3，即 mmad `n<64`；清零覆盖到行 0–7）—— 窗口之外是否被写过**不可判定**；
- 29 组里**没有任何一组是「已知能写」的正对照**，所以严格说只能得出「**未观察到写入**」，
  不能排除「某组本可以写、只是参数还不对」。文档的两条硬约束（`LoadData_3D.md:168,244`）
  是支持「这条路不该走」的独立依据，但**不能把实测升级成「证明无此通路」**。

---

## 4. 验证表：真实 BMM2 形状上"先预测再实测"

`m16_pv`：A = P `[M=16, K=256]`，B = V `[K=256, N]`，`C = P·V`；K 一次 mmad（k=256，A 整块驻留 L0A）；
N 按 128 分块：每块 B 的 `[K=256, nTile=128]` 进 L0B，占 `(K/16)×(nTile/16) = 16×8 = 128 行 × 512B = 64KB`
（正好占满 L0B）⇒ **N=128 作为分块、N=256 作为整块都被覆盖**。

**预测的配方**（由 §3 推出，运行前就写在程序里打印出来）：

```
A(P)：GM[16,K] 行主序 → Nd2Nz(nValue=16, dValue=K, srcDValue=K, dstNzC0Stride=16)
      → LoadData2DParamsV2{mStep=1, kStep=K/16, srcStride=1, dstStride=1, ifTranspose=false}
        （与 m11 LoadA / M24 BMM1 的 Q 侧同款）
B(V)：GM[K,N] 行主序 → Nd2Nz(nValue=K, dValue=N, srcDValue=N, dstNzC0Stride=K)
      → LoadData2DParamsV2{mStep=K/16, kStep=N/16, srcStride=K/16, dstStride=N/16, ifTranspose=true}
```

| 用例 | 形状 | 结果 | 日志 |
|---|---|---|---|
| `map`：P = 16×256 单位阵、V 用可逆位型编码 ⇒ 直接读回 `B_mmad[k][n]` 的映射 | M=16,K=256,N=256 | **解码命中 4096/4096** ⇒ `B_mmad[k][n] = V[k][n]` 逐点成立 | `evidence/pv_map.log` |
| `P·V` 合成数据（[-8,8] 小整数 ⇒ 精确整数域） | M=16,K=256,N=256 | device vs host fp32 参考（RNE→bf16）：**0/4096 逐位不等** | `evidence/pv_synth.log` |
| `P·V` **真实 dump**：M24 `m10_case_s256` 的 P 中转 `gp[0][0][16][256]` + V `v[n2=0][256][256]` | M=16,K=256,N=256 | **0/4096 逐位不等**（`check_ref.py pv` 用 numpy 独立复算同样 0/4096） | `evidence/pv_real.log` |
| `kmap`：P 只选中单个 k（`P[m][k]=1 iff k==kOff`）、V 的每行填成 `1+k` ⇒ `C[m][n]` 应等于 `1+kOff` | M=16,K=256,N=256 | 实测 `C=1+kOff`（`kOff` 取 0/1/7/8/15/16/17/31/32/63/64/127/128/192/255 全部命中）⇒ mmad 的 k 轴与 V 的行一一对应 | `evidence/pv_kmap.log` |
| 反例 `baddst`：B 装载的 `dstStride` 照抄 **BMM1 的参数元组**（= `K/16` = 16），而不是 L0B 的 n 分形数 8 | 同上 | **device 报错 507015**（L0B 行号越界） | `evidence/pv_baddst.log` |

### 4.1 跨分支输入（M24 切片）的提取命令与校验和

`evidence/m24_s256_P_unit0_par0.bin` / `m24_s256_V_n2_0.bin` 是本 mission 唯一的跨分支输入，
提取自 wt-24（分支 `feat/attention-decode-fa-core-continuation`）的 `m10_attn_decode/data/`。
**提取命令**（`cd m16_load_geom` 后执行；`WT24=/workspace/ascend_mega_kernel/.tower/worktrees/wt-24`）：

```bash
# P：gp.bin 的前 16*256*2 = 8192 字节 = gp[unit=0][parity=0][16][256]
head -c 8192 "$WT24/m10_attn_decode/data/m10_case_s256_gp.bin" > evidence/m24_s256_P_unit0_par0.bin
# V：v.bin 的前 256*256*2 = 131072 字节 = v[n2=0][seqPad=256][256]
head -c 131072 "$WT24/m10_attn_decode/data/m10_case_s256_v.bin" > evidence/m24_s256_V_n2_0.bin
# 逐字节确认切片与源文件的关系（应输出 0 差异）
cmp -n 8192   evidence/m24_s256_P_unit0_par0.bin "$WT24/m10_attn_decode/data/m10_case_s256_gp.bin"
cmp -n 131072 evidence/m24_s256_V_n2_0.bin      "$WT24/m10_attn_decode/data/m10_case_s256_v.bin"
```

布局依据见 wt-24 的 `m10_attn_decode.asc`（`gPBytes = 28*3*16*256*2`，注释 `P 的 GM 中转：[unit][parity][16][256]`）
与 README §1（K/V `[N2][seqPad][256]`）。**全部 dump 的 sha256 见 `evidence/dump_sha256.txt`**
校验和**只覆盖数据产物**（两个 M24 切片 + 2D/3D raw dump + sidecar + `m16_pv` 的 5 个 .bin 产物，共 11 项）：
在工程根目录用 `sha256sum -c evidence/dump_sha256.txt` 复核。**`*.log` 不纳入**——它们含 §2.3 记录的
9 个非确定 conf 的越界槽位内容、且 `reproduce_run.log` 每次运行都会变，纳入会让校验和本身不稳定
（头部注释里也写了这一点）；日志里的稳定数字（`0/4096`、`23552/23552`、`4096/4096` 等）逐条列在 §2.3/§4。
**归档刷新**：`tools/refresh_evidence.sh` 一条命令完成「跑 reproduce → 复制产物与日志进 evidence/ →
重算 `dump_sha256.txt` → 校验」，避免再次出现"树是绿的、归档是旧运行日志"（`cdbfa20`）。

### 4.2 关于"BMM1 的参数只在其形状下恰好自洽"

- M24 的 BMM1 B 侧元组是 `(mStep=S2T/16, kStep=128/16, srcStride=S2T/16, dstStride=S2T/16)`。
  在 **它自己的形状**（`S2T = 256 = N`，`N/16 = K/16 = 16`）下，`dstStride = S2T/16` **恰好等于**
  `N/16` ⇒ 按本标定，**M24 的 BMM1 参数在它自己的形状上是对的**（这也是为什么它的 BMM1 数值一直很好）。
- 但**它不是通用常量**：同一元组换到别的形状就错。`m16_pv baddst` 直接给出 device 报错；
  在 2-fractal 探针源上（conf 46）它"看起来对"—— 因为该源的 M 分形步长恰为 1 单元，
  使得 `m1` 走到的恰好是正确的分形；**这正是"在小形状上看起来对的参数不能外推"的活样本**。
- 同理 conf 47：BMM2 的 A 侧元组 `(1, 16, 1, 1)` 放到只有 2 个列分形的源上也"看起来对"（同上原因）。

> 教训：**装载几何必须在本形状下用"解码出源坐标"的方式标定**，不能靠"小用例上跑出一个像样的数"。

---

## 5. 对现有仓库代码的警示

1. **谁在"照抄别的形状的参数"**
   > 引用口径（`2e54a89`）：`m10_attn_decode.asc` **不在 M27 的 base（`d5cac73`）里**，
   > 行号天然跨分支、无版本锚。下面一律用「分支 `feat/attention-decode-fa-core-continuation` +
   > 函数名 / 代码片段」引用，不用行号。
   - 该分支的 `LoadL0B_Transpose()`（3D `LoadData3DParamsV2` + `enTranspose=1`，装载 V^T）——
     按 §3.6 该路径在 29 组 conf 里**一次写入都没观察到**（限度见 §3.6 末），且文档说 L0B 下 `enTranspose` 无效 ⇒ **这条路线应废弃**。
     改用 §4 的 2D T1 配方（V 本来就是 `[k,n]` 形态，连预转置都不需要）。
     **注**：该分支当前 tip 的 BMM2 B(V) 调用点已经换成 `LoadL0_2D<bfloat16_t>(l0B, l1KV1, kOff,
     S2T/16, 128/16, S2T/16, 128/16, /*ifTranspose=*/true)`（注释写明"按 M27 几何标定表"），
     即 §4 的配方；`LoadL0B_Transpose()` 作为遗留函数仍在文件里。
   - 同一分支 BMM1 的 B 侧 `LoadL0_2D(l0B, l1KV0, kOff, S2T/16, 128/16, S2T/16, S2T/16)`：
     在 S2T=256 下正确（`dstStride = S2T/16 = 16 = N/16`），但换 tile 大小（`seqPad` 变、
     或 N≠S2T）时必须重算 `srcStride`（= `R_pad/16`）与 `dstStride`（= `N/16`）。
2. **`dstStride` 不是"抄一个常数"**：它必须等于目的 L0B 的 **n 分形数 `N/16`**；
   `srcStride` 必须等于源的 **列方向分形间隔 = 源行数(带 pad)/16**。两者都随形状变。
   （附带说明：m1/m3/m11 的装载参数看起来与这两条规则一致，但**那是描述性引用、本 mission 没有验证过它们**
   —— 它们的数值应由各自 README 的实证负责。）关键是**改 BASE_M/BASE_N/BASE_K 就要重算这两个字段**。
3. **`mStartPosition`/`kStartPosition` 的单位与文档一致**（bf16：16 个 M 元素 = 1 个 M 分形行；
   32 字节 = 1 个 K 分形列），**但要记住它们是"源分形下标"而不是可以直接加到字节地址上的线性偏移**：
   K 轴那一个在地址上要乘 `srcStride`（文档公式就是 `kStartPosition×|srcStride|`），
   而实测 `srcStride` 不取绝对值（§3.4）。
4. **`srcOff`/`dstOff` 只能靠 `LocalTensor` 切片表达**（M24 已实证；本探针同样只用切片），
   且单位是元素（bf16 = 2B/元素）。
5. **调试/探针代码读完 L0 之前必须清零目的 buffer**，否则会读到上一次 launch 的残留（§2.2）。
6. **V 的 L1 布局**：要让 2D T1 配方成立，V 必须是 `Nd2Nz` 出来的 Nz（行 = k、列 = n，`dstNzC0Stride = K`）。
   M24 尝试过的「dim-major 预转置 V」正好破坏了这一点（那是 EXP2 被排除的原因之一）。

---

## 6. 结论：【已标定】/【仍存疑】

### 【已标定】（有实测证据 + 逐点模型验证）

- L0B 在 mmad 眼里的地址：元素下标 `16n+k`；块行 = `n/16`；分形行号 = `(k/16)(N/16)+(n/16)`。
- 分形内变换：T0 = 原样、T1 = 16×16 转置；`B[k][n]` 与源 `(r0,c0)` 的关系（§3.2）。
- 多分形搬运的迭代与落位公式（§3.3）——`check_ref.py geom2d` 在 42 个可判定 conf 的 **23552 个槽位**上
  **逐点命中、0 不符**。
- 8 个字段的语义/单位/生效性（§3.4），含 `mStep=kStep=0` 为 NOP、`srcStride` 按有符号使用。
- BMM2 在 M=16、K=256、N=256 上的配方，且 `P·V` 与 numpy/fp32 参考**逐位一致**（合成 + 真实 dump）。
- 3D `LoadData3DParamsV2` → L0B **在 29 组 conf（含 NC1HWC0 解释、全字段扰动）的回读窗口内未观察到写入**；
  因此**不建议**把它当作 BMM2 V^T 的路线 —— 该结论的依据是**文档的两条硬约束**（该通路自动转置、
  L0B 下 `enTranspose` 无效），实测只提供"未观察到写入"这一支持性证据。

### 【仍存疑】

- **3D → L0A 的 `enTranspose`**（文档承认唯一有效的 3D 转置通路）**未标定**：要标定它需要把未知矩阵
  放到 L0A、把单位阵放到 L0B（本探针的读回装置是反的）。本 mission 的时间盒内未做。
- **越界/错参数 conf 的读数内容非确定**（5 次运行分成 4 组，见 §2.3 与 `evidence/geom2d_determinism.txt`）
  ——因此这类读数**不可用于推断**；本 README 的所有结论只建立在"参数落在源范围内"的 conf 上。
- **§3.4 的 mStart/kStart 两行只做了 2 次/conf 的隔离运行**（样本量小，"in-range 可复现"这个判定
  的置信度有限）；`tools/check_isolated.py` 可加大重复次数（目前硬编码 2 次）。
- **`b8`/`b32` 与 MX（`fp4`）/`LoadData2DMxParams` 路径未标定**（本 mission 只覆盖 bf16）。
  b8/b32 的方形分形合并规则（`fractalNum`）与 `LoadData_2D_V2.md` 的额外倍数约束未验证。
- **`SetLoadDataRepeatWithStride` 对 2D V2 是否需要**：文档只对 3D 路径要求它；本探针 2D 未调用也能
  工作，但"不调用时 `dstStride` 是否走默认值"未做对照实验。
- **N 分形为大方向还是小方向**：`分形行号 = (k/16)(N/16)+(n/16)` 中「k 分形在外」这一点，
  在 k 只有 1 个分形（本探针 mmad k=16）时**观测不到**；由「与 §3.3 的落位公式自洽 + 官方样例配方」
  推出，未用 k≥32 的 mmad 直接验证。
- **3D 负结论缺正对照**：29 组 conf 里没有任何「已知能写」的配置，且只观察了回读窗口（L0B 行 0–3）
  ⇒ 只能得出「未观察到写入」，不能区分「参数还不对」与「通路不可用」（§3.6 已写明限度）。
- **「预测配方」打印与执行参数的一致性**：现在靠「两者共用同一组 `A_xxx` / `B_xxx` 常量」这一
  **结构性约定**保证（`de68fe3` 曾因两者分叉而打印出错误元组，见 `2e54a89`）；**没有自动化检查**
  （如编译期断言、或把实际参数一并 dump）来防止以后再次分叉。
- **L0B 残留是否跨进程**：同一进程内的残留已确凿（conf 48 ≡ conf 47）；「跨进程也残留」只有
  标定早期调试时的观察（同一单 conf 三次独立运行给出三个不同结果），**那次 pre-fix 日志未归档**
  ⇒ 本文只把它当推断（§2.2 已标注）。
