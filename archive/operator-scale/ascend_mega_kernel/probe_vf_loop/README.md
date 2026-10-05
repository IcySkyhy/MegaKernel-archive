# probe_vf_loop —— 3510 VF 内层循环 trip count 探针（M43）

> 触发：M34（GDN prefill chunk scan）报的 high 级 finding
> `.tower/comms/findings/20260926-agent-gdnprefill-bug-3510-vf-quirk.md`。
> 本工程的唯一目标：把「单 kernel 观测」变成**可归档的最小 repro + 已确证/仍存疑的边界**。
> **本目录只做取证，不改任何别的目录**（不动 `m18_gdn_prefill/**`、不动 `docs/**`）。

## 0. 一句话结论

`__simd_vf__`（寄存器 VF）里写
`for (uint16_t i = 0; i < rows; ++i) { for (uint16_t j = 0; j < i; ++j) {…} }`
（`rows` 为运行期 `uint16_t`），实测**内层循环体只执行 `i − 1` 次**（`rows∈{0,1}` 退化档为 0），
即每行丢掉最后一项 `j = i − 1` —— finding 的核心主张**原样复现**
（27 个 nest 变体 × 5 次独立进程；**判据列跨 rep 逐字符一致**，`rows≥2` 才出现、`rows∈{0,1}` 退化档正确）。

- **与循环体内容无关**：内层体只做两个向量寄存器累加、一次访存都没有（body 0）同样复现；
  带行 store→load 依赖（body 3）也复现 ⇒ 属**循环控制流**层面，**不是**内存可见性/数据通路问题。
- **与优化档位无关**（`-O2` 同 `-O3`）；`#pragma unroll`、把上界「提到循环外先算」都**无效**。
- 内层上界换成**不依赖外层归纳变量**的运行期值（`j < rows`）或**编译期常量** ⇒ 全部档位正确
  （这两族是唯一在全部档位都正确的）。
- ⚠️ **但条件与现象"未对齐"**（§3.2 ②，如实记录）：`j < i+1`（同样依赖外层归纳变量）**正确**；
  外层归纳变量换成 `uint32_t/int32_t` 后 `j < i` 也**正确**；`j < i−1`（带保护）则变成另一种错
  （`cnt = 期望 + 255/+256`）。⇒ 既不能说"恒少一次"，也不能说"上界依赖外层 IV 就出错"。
- **"确定性"只主张判据列**（`cnt`/`tri`/`verdict` 等）：27/27 target 跨 5 rep 逐字符一致。
  **例外**：`nest_o0i1b1`（`j<i−1` 病态变体）的**数据面列**（`acc_maxdiff`/`hash`）跨 rep 不同 ——
  它读越界 UB，**不是判据**；位置与原因见 §3.2 ⑧、核验见 `evidence/logs/summarize_verify.txt`。
- **归因未定**：没做 ISA 级取证，**不宣称**这是"硬件 bug"还是编译器 codegen 缺陷（§3.2 ①）。

## 1. 复现（本目录下执行）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
bash run_probes.sh 5          # 全量：探针 nest 25+2 变体 + gather 边界扫描；每变体 5 次独立进程
```

> 本机那张卡由 10+ 个 worker 共用，长驻 run 会被别人的清场动作打断（M43 实际被打断两次）。
> 因此本工程归档的证据是**同一脚本、同一源码修订**下分块跑出来的：
> `ONLY='<正则>' bash run_probes.sh 5` 只做匹配的 target，且只清这些 target 的旧日志、汇总文件按追加写。
> 全量跑（`ONLY` 为空）会先 `--target clean` 再重建，保证每个 target 的 build 日志都含完整编译命令行。

最小 repro 的单条手工命令（不经 CMake，直接调编译器）：

```bash
/usr/local/Ascend/cann-9.1.0/bin/bisheng -DNPU_ARCH_DAV_3510 -DPROBE_OUTER=0 -DPROBE_INNER=0 \
    -DPROBE_BODY=0 -DPROBE_CT_ROWS=0 -DPROBE_TAG=nest_o0i0b0 -std=c++17 --npu-arch=dav-3510 -O3 -DNDEBUG \
    -c --asc-aicore-lang probe_vf_nest.asc -o /tmp/o.o
```

### 1.1 采集环境与并发背景（读数可信度的前提）

| 项 | 值 |
|---|---|
| 日期 / 机时 | 2026-09-26，13:0x–14:xx（bash `date -Is` 见 `evidence/commands.txt`） |
| 设备 | 单卡 Ascend950PR，**10+ worker 共用**（tower 记录 inflight 6–8；当时同卡上另有 `wt-45/probe_vf_ldst`（agent-vecdist）与 `wt-57/probe_v_align`（agent-valign）在持续跑真机探针，见 M43 归档的 `ps` 输出与两封 tower 消息） |
| 编译器 | bisheng 15.0.5（CANN 9.1.0），驱动内部 cc1 开关全文见 `evidence/logs/bisheng_verbose_cc1_flags.txt` |
| 被打断的 run | 与 wt-45 同时跑时，我的长驻 `run_probes.sh` 被**别人过宽的 `pkill` 模式连带杀死两次**（tower 已在全队广播"清场只允许限定自己 worktree/PID"的规则）⇒ 本工程证据最终是**同一脚本、同一源码修订**下 `ONLY='<正则>'` **分块**跑出来的（每块跑完即 commit） |
| 对读数的影响 | **判据列无影响**：27 个变体的 `cnt`/`expect`/`tri`/`lane_mismatch`/`verdict` 这些**判据列**在 5 次独立进程之间逐字符一致（`summarize.py verify` 逐 target 核过，0 条差异；见 `evidence/logs/summarize_verify.txt`）。**唯一例外是数据面列**：病态变体 `nest_o0i1b1`（`j<i−1` 带保护版）的 `acc_maxdiff`/`hash` 跨 rep 不同 —— 它那几行读的是**越界 UB**（残留内容随上次 launch 变），**不是判据**，详见 §3.2 ⑧。并发负载只影响**耗时**（单档位从 ~4s 拖到 ~100s） |

## 2. 最小 repro 与读数设计（`probe_vf_nest.asc`）



### 2.1 内核形态（与 m18 `TrilSolveVF` 同构）

```cpp
template <typename RowsT, uint16_t CTK>            // CTK=0 ⇒ 外层上界取运行期实参
__simd_vf__ inline void RunNestVF(__ubuf__ float* aP, __ubuf__ float* xP, ... , RowsT rows)
{
    for (RowsT i = 0; i < outerBound; ++i) {       // outerBound = rows（或编译期常量 CTK）
        Duplicate(cnt, 0.0f); Duplicate(tri, 0.0f); Duplicate(acc, 0.0f); Duplicate(accA, 0.0f);
        for (uint16_t j = 0; j < i16; ++j) {       // ← 被测形态：内层上界 = 外层归纳变量
            Adds(cnt, cnt, 1.0f, allF);            // 向量计数器：内层体跑了几次
            Add(tri, tri, cnt, allF);              // Σ_{k≤t} k（自洽校验）
            LoadAlign<float, LoadDist::DIST_BRC_B32>(av, aP + i16*ASTRIDE + j);   // m18 同形态
            LoadAlign(xj, xP + j*VL);
            MulAddDst(acc,  av, xj,  allF);        // Σ a[i][j]·x[j][lane]
            MulAddDst(accA, av, one, allF);        // Σ a[i][j]（可反解实际执行的 j 集合）
        }
        StoreAlign(cntP + i16*VL, cnt, allF);  /* tri/acc/accA 同理 */ ;
    }
}
```

### 2.2 为什么计数只能用**向量寄存器**（实测硬约束，顺带取证）

VF 循环体里出现**任何标量指令**都编译不过：

```
$ bisheng ... -DPROBE_BODY=2 ...            # 内层体里写 `cntS = cntS + 1;`（标量计数）
fatal error: error in backend: probe_vf_nest.asc:277:8: Unsupported scalar instruction in AIV loop.
```

同样地，**标量 store 到 UB**（`mk[i*RW+j] = j` 这种逐轮写标记）也报同一句错
（靶子 `nest_o0i0b2s_scalstore` = 标量算术 + 标量 store，`evidence/logs/build_nest_o0i0b2s_scalstore.log`）；
**正对照** `nest_o0i0b4_marker_ctl`（每轮一次 **向量** `StoreAlign`）**能编过** ⇒ 被拒的是标量指令，
不是"循环里做 store"（该正对照只做编译归档，不跑）。
⇒ 这是 finding 里 `cnt += 1` 写法无法照抄的原因，也是本探针用向量计数器的原因。
两个失败靶子的完整 stderr 已归档：`build_nest_o0i0b2_scal.log` / `build_nest_o0i0b2s_scalstore.log`。

另外两条**编译器层面**的实测事实（都归档了原文）：

| 事实 | 证据 |
|---|---|
| `j + 1 < i` 这种「把减一放进比较里」的内层上界写法，**让 bisheng 直接段错误**（`error: unable to execute command: Segmentation fault (core dumped)`）⇒ 该写法在本版本不可用，也无法纳入矩阵 | `evidence/crash_inner7/compile_inner7_crash.log` |
| 不带保护的 `j < i - 1` 在 `i==0` 时 uint16 下溢 ⇒ 内层界变 65535 ⇒ `xP + j*VL` 地址远超 UB ⇒ `507035`（不是被测现象，是 C 语义 + 越界） | `evidence/unguarded_inner1/inner1_unguarded_rep1.log.txt`；矩阵里 `nest_o0i1b1` 用的是**带保护**的形态 |
| 驱动的内部 cc1 开关（含 `-mllvm -cce-aicore-hwloops=false`、`-mllvm -enable-vloop-lowering=true`、`-mllvm -cce-aicore-loop-unrotate=true`、`-mllvm -cce-aicore-backend-loop-lower=true`）完整清单 | `evidence/logs/bisheng_verbose_cc1_flags.txt`（`bisheng -v` 原文） |

### 2.3 读数与判据（整数精确，无浮点容差）

| 读数 | 落盘位置 | 含义 |
|---|---|---|
| `cnt[i]` | out[0 .. 64·rows) | 行 i 内层循环体执行次数（= trip count） |
| `tri[i]` | out[64·64 ..] | Σ_{k≤t} k：与 `cnt` 自洽（应 = t(t+1)/2） |
| `accA[i]` | out[3·64·64 ..] | Σ_{执行的 j} a[i][j]，结合 `cnt` 精确反解 Σj ⇒ 判定执行的 j 是不是**前缀**（丢的是不是最后一项） |
| `acc[i][lane]` | out[2·64·64 ..] | Σ_{执行的 j} a[i][j]·x[j][lane]，与 host 参考逐位比 |

判据整数精确：`a[i][j] = i*4+j+1 ≤ 256`、`x[j][lane] = j*8+lane+1 ≤ 576` ⇒ 单项乘积 ≤ 147456、
64 项和 ≤ 9.4e6 < 2²⁴ ⇒ fp32 序列和**精确**（与累加顺序 / FMA 融合无关）。
所以 `acc_maxdiff=0.0` 是逐位相等，`0.0` 之外的值就是"参考里多/少了项"。

### 2.4 单变体的完整读数样例（`nest_o0i0b0`，body 0 = **只做向量计数、零访存**）

```
[probe vf nest] tag=nest_o0i0b0 outer=0 inner=0 body=0 ct_rows=0
  rows=4
    i      : 0 1 2 3
    cnt    : 0 0 1 2          ← 实测：内层体执行 0,0,1,2 次
    expect : 0 1 2 3          ← 应该 0,1,2,3
    tri    : 0 0 1 3          （= t(t+1)/2，与 cnt 自洽）
    lane_mismatch=0  tri_odd=0  少一次的行数=3/4  hash=0x...
    rows=4 verdict=FAIL
  rows=64
    cnt    : 0 0 1 2 3 ... 62  （第 i 项 = i−1）
```

对照（`nest_o0i3b0`，同 body，只把内层上界换成**不依赖外层归纳变量**的 `rows`）：

```
  rows=64
    cnt    : 64 64 64 ... 64   expect : 64 ...   verdict=PASS
```

两次运行的**仪器完全相同**，只有内层上界表达式不同 ⇒ 差异可干净归到「上界是否由外层归纳变量导出」。

## 3. 【已确证】/【仍存疑】

### 3.1 已确证（每条都有 5 次独立进程运行 + sha256 归档；见 §4 表格与 §8 证据）

> **评审引用口径**：下文 `评审 pN-x` 指评审文件
> `.tower/comms/reviews/review-feat-vf-nested-loop-trip-count-probe-reviewer-m43-r1.md`（该文件头部 `round: 1`、`reviewed_commit: 72af722`）
> Findings 里的第 N 条；对应整改 commit = `72c76b0`（README/脚本/工具）与 `8313d6b`（§7 措辞）。
> **数字口径**：本 README 里出现的归档计数（27 target / 135 rep 日志 / 47 gather 档位 / 35 条编译结论）
> 都与 `tools/summarize.py verify` 的同一次运行输出对齐（见 `evidence/logs/summarize_verify.txt` 首节）。

| # | 结论 | 证据（target） |
|---|---|---|
| C1 | **`for (uint16_t j = 0; j < i; ++j)`（外层 u16 运行期上界）⇒ 内层体实际执行 `max(i−1, 0)` 次**：rows=4 时 `cnt = [0,0,1,2]`（期望 `[0,1,2,3]`），rows=64 时 `cnt[i] = i−1`；即每行丢最后一项 `j = i−1` | `nest_o0i0b0`（零访存，最干净）· `nest_o0i0b1`（m18 同形态）· `nest_o0i0b3`（+行 store→load 依赖）· `nest_o0i0b1_O2` · `nest_o0i0b0_NOUNROT` · `nest_o0i0b1_NOUNROT` |
| C2 | 内层上界改成**不依赖外层归纳变量**的运行期值（`j < rows`）⇒ **全部正确**（rows ∈ {0,1,2,3,4,64}，cnt[i] = rows） | `nest_o0i3b0` · `o0i3b1` · `o0i3b3` · `o1i3b1` · `o2i3b1_ct4` 全 RUN PASS |
| C3 | 现象与**循环体内容**无关：内层体只有两个向量寄存器累加（零访存，body 0）也复现；加行 store→load 依赖（body 3）也复现 ⇒ 属**循环控制流**层面，不是内存可见性 / LocalMemBar / 数据通路 | C1 的 body 0 / body 1 / body 3 三组读数相同 |
| C4 | **判据列确定性**：每个变体 5 次独立进程的**判据列**（`cnt` / `expect` / `tri` / `MISMATCH` / `lane_mismatch` / `tri_odd` / `rows=N verdict=`）逐字符一致 —— 27 target × 5 rep 全核过，**0 条差异**。**边界必须一起读**：`acc_maxdiff` / `hash=0x…` 属**数据面列**，在病态变体 `nest_o0i1b1` 上跨 rep **不同**（3 个日志 sha256）⇒ 本报告的"确定"**只对判据列成立**，数据面列的非确定性见 §3.2 ⑧ | `summarize.py verify`（`evidence/logs/summarize_verify.txt`：判据列 0 差异 + 数据面列点名 o0i1b1 rep3/rep4）；`sha256.txt` 的 `judge=` / `JUDGE-SAME\|DIFF` 列；`run_matrix.txt` 每 target 第三行「数据面列跨 rep …」 |
| C5 | **与优化档位无关**：`-O2` 与默认 `-O3` 结果相同（`j<i` 复现、`j<rows` 正确，cnt 读数逐项相同） | `nest_o0i0b1_O2` FAIL · `nest_o0i3b1_O2` PASS |
| C6 | 内层加 `#pragma unroll` **不规避**（编译器同时报 `loop not unrolled` 警告，见 `evidence/logs/build_nest_o1i0b1.log`） | `nest_o1i0b1` FAIL · `nest_o1i3b1` PASS |
| C7 | 把上界「提出到内层循环外先算成一个变量」（`hi = (uint16_t)i;` 再 `j < hi`）**不规避**；把上界变量类型换成 u32 / i32 **也不规避**（读数与 `j<i` 逐项相同） | `nest_o0i4b0` · `nest_o0i4b1` · `nest_o0i5b1`（u32 hi）· `nest_o0i6b1`（i32 hi）全 FAIL |
| C8 | 外层上界改成**编译期常量**不是可靠规避：`ct=1/2` 恰好全对（外层被完全展开，内层上界随之常量），`ct=3/4/64` **仍复现**（`cnt=[0,0,1,2]`） | `nest_o2i0b1_ct1/ct2` PASS vs `ct3/ct4/ct64` FAIL |
| C9 | **外层归纳变量换成 uint32_t / int32_t 时不复现**（`cnt=[0,1,2,3]`，含"归纳变量直接进地址算术"的形态） | `nest_o3i0b1` · `nest_o4i0b1` · `nest_o5i0b1` 全 PASS —— 但只有事实、无机制，见 §3.2 ② |
| C10 | 内层上界写 `j < i + 1`（同样依赖外层归纳变量）**不复现**（`cnt=[1,2,3,4]` = 期望） ⇒ **不能**把现象简单归到"上界是否依赖外层归纳变量" | `nest_o0i2b1` PASS |
| C11 | 内层上界写 `j < i − 1`（带保护避开 uint16 下溢）**同样错，但错法不同**：rows=4 时 `cnt=[0,256,257,258]`（= 期望 + 255/+256） | `nest_o0i1b1` FAIL（`jset_odd=3`，读数与 `j<i` 完全不同） |
| C12 | **带保护**形式之外，不带保护的 `j<i−1` 在 `i==0` 时 uint16 下溢 ⇒ 界变 65535 ⇒ 地址远超 UB ⇒ `507035`（不可测，只能作为 C 语义层面的 foot-gun 记录） | `evidence/unguarded_inner1/inner1_unguarded_rep1.log.txt` |
| C13 | 内层上界写 `j + 1 < i`（"i−1"的不下溢写法）会**让编译器段错误**（`error: unable to execute command: Segmentation fault (core dumped)`）⇒ 该写法在本版本不可用，无法纳入矩阵 | `evidence/crash_inner7/compile_inner7_crash.log` |
| C14 | **关掉 `-mllvm -cce-aicore-loop-unrotate=false` 这个 loop 变换 pass，现象依旧** ⇒ 至少不是该 pass 单独造成 | `nest_o0i0b0_NOUNROT` / `nest_o0i0b1_NOUNROT` FAIL（读数与默认组逐项相同） |
| C15 | VF（`__VEC_SCOPE__`/`__simd_vf__`）循环体内不允许**任何标量指令**：**标量整数算术**（`cntS = cntS + 1;`）与**标量 store**（`mk[i*RW+j] = j;`）都编译失败 `fatal error: error in backend: Unsupported scalar instruction in AIV loop`。**正对照**：同样是"每轮一次 store"、但用**向量** `StoreAlign`（地址含 j、值不含 j）**能编过** ⇒ 被拒的是标量指令本身，不是"循环里做 store"（评审 p2-附注要求把"标量 store 也不行"自证，已补靶子） | ① `nest_o0i0b2_scal`（标量算术，`build_nest_o0i0b2_scal.log`）② `nest_o0i0b2s_scalstore`（标量算术 + 标量 store，`build_nest_o0i0b2s_scalstore.log`）③ 正对照 `nest_o0i0b4_marker_ctl` **COMPILE-OK**（`build_nest_o0i0b4_marker_ctl.log`） |
| C16 | `Reg::Arange` 的 `static_assert` 支持列表 = `int8/int16/int32/float/half/int64` ⇒ **无符号类型全不支持**；绕法 `Arange<int32_t>` + `reinterpret_cast<RegTensor<uint32_t>&>` 交给 `Gather` **可行** | `ag_arange_u32` 归档原文 + `ag_gather_sweep` 正常路径 |
| C17 | `Reg::Gather`（vgather2，UB 源）按**元素**索引，可寻址范围 = 本核 UB 物理窗口（256KB），**没有 8191 之类的窗口限制**；越出 UB ⇒ `aclError=507035` | §5② 全档扫描（47 档位） |

### 3.2 仍存疑（禁止当事实引用）

| # | 存疑点 | 现状 |
|---|---|---|
| ① | **归属**：硬件 bug、还是编译器（bisheng/ccec）codegen 缺陷？ | **未取证到 ISA 级**：设备代码在 `.aicore_binary` fatbin 里，device side 不支持 `-S`，`msobjdump` 只解析 ELF 结构不做反汇编 ⇒ 本工程**不宣称**归属。两条旁证（只是旁证）：(a) 驱动内部带 `-mllvm -cce-aicore-hwloops=false`（`evidence/logs/bisheng_verbose_cc1_flags.txt`）⇒ 该路径上没有"硬件 loop"指令；(b) 关掉 `loop-unrotate` pass 现象依旧（C14） |
| ② | **条件与现象未对齐**：没有一个单一条件能解释全部格子 | 观测到的格子：`j<i`（外层 u16，运行期/ct≥3）**少一次**；`j<i` 但外层 IV 是 u32/i32 **正确**；`j<i+1`（同样依赖外层 IV、外层 u16）**正确**；`j<i−1`（带保护）**变成 +256**；`j<rows` 正确；`ct≤2` 正确。⇒ 既不能说"恒少一次"（被 `i+1` 否掉），也不能说"上界依赖外层 IV 就出错"（被 `i+1` 否掉）。**如实写成：未对齐，但观测到以上 X**。其中"外层 IV 类型"这条最反直觉（u16 错、u32/i32 对），机制不明，**不作为规避手段** |
| ③ | 「内层体执行 `上界−1` 次」与「循环跑了 `上界` 次、但最后一轮的向量算术被丢弃」两种解释，现有读数**不可区分** | 两者对数值结果的含义相同（都丢最后一项），故不影响 C1 的实用结论；但要写成"硬件语义"还不够 |
| ④ | 编译期外层上界的行为细则：`ct=2` 对、`ct=3` 错 | 与"编译器是否把外层完全展开"有关（未对齐）。⇒ 表述应为"编译期常量**不是可靠**规避"，而不是"编译期常量可规避" |
| ⑤ | body 4（逐轮序号标记落盘，试图做"仪器无关读数"）读数**自相矛盾**：rows=1 时 `cnt=246` 而 64 个标记槽位全非零、`mk` 的逐行重置被忽略 | 未解释；**刻意不进默认 target 列表**（CMakeLists 注释掉），原始日志留 `evidence/negmark/negmark_o0i0b4_rep1.log.txt`。因此本报告只以「向量计数器 + 数据累加器」两种读数互证 |
| ⑥ | `j<i−1`（带保护）为什么是 `+255/+256` 而不是"少一次"或"少两次" | 只有读数（C11），无解释。256 = 2⁸ 可疑（像某个 8 bit 计数字段被当成 9 bit 用），但**未取证** |
| ⑦ | 未覆盖的形态 | 内层上界为 `rows−i` / `i/2` / 非单调表达式；外层步长 ≠ 1；三层及以上嵌套；内层归纳变量非 uint16_t；`#pragma unroll N`（带因子）；其他 CANN/bisheng 版本。`j+1<i` 因编译器段错误（C13）无法覆盖 |
| ⑧ | **`nest_o0i1b1`（`j<i−1` 带保护）的「数据面列」非确定** | 该变体 5 次独立进程有 **3 个不同的全日志 sha256**：rep1/rep2/rep5 = `3e682ab9411a81be…`、rep4 = `85a1d8d0a815b63b…`、rep3 = `dfa251abbd3c26c7…`。**差异只在 `acc_maxdiff …` 与 `hash=0x…` 两行**（rows≥2 档），因为该变体 `cnt=256+` 后 `acc` 读的是**越界 UB**，残留内容随上一次 launch 而变；而**判据列 5 次逐字符相同** —— 归档里该 target 的 5 行 `judge=` 列都是 **`7556d6cb43de6f31`**（`grep ' nest_o0i1b1  rep' evidence/logs/sha256.txt | awk '{print $5}' \| sort -u` 当场可复算；判据列定义 = `tools/summarize.py` 的 `judge_lines()`，含 `cnt/expect/tri/MISMATCH/JSET-ODD/lane_mismatch/verdict` 并剔除 `hash=0x…`）。⇒ 本报告的"确定"**只主张判据列**；数据面列在该病态变体上不可比。**这条是评审 p2-1 的点名项**：先前 README 写成"全部变体逐字节一致"是错的，现已按列收窄并显式披露（`run_matrix.txt` 每 target 第三行 + `summarize_verify.txt` 数据面节） |

## 4. 变体矩阵与读数

矩阵维度：外层（运行期 u16 / 运行期 u16+内层 unroll / 编译期常量 / 运行期 u32 / 运行期 i32 / 运行期 u32 直接进地址算术）
× 内层上界（`i` / `i-1`（带保护）/ `i+1` / `rows` / 提到循环外 u16 / 提到循环外 u32 / 提到循环外 i32）
× 内层体（0=只计数、1=m18 同形态、3=+行 store→load 依赖）
× rows ∈ {0,1,2,3,4,64}（运行期传入；编译期态取 `ct` 值）。每个变体 **5 次独立进程运行**。
另有 `-O2` 组与两个"关编译器内部 pass"的诊断组（`_O2` / `_NOUNROT`）。

**矩阵（原文见 `evidence/logs/run_matrix.txt`；本表由 `tools/summarize.py nest` 从 rep1 日志生成）**

| target | OUTER | INNER | BODY | rows 档位 | 结果 | 失败档位 | 关键读数（rows=4：cnt vs expect） |
|---|---|---|---|---|---|---|---|
| nest_o0i0b0 | 0 rt16 | 0 `j<i` | 0 只计数 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]` / expect `[0 1 2 3]` |
| nest_o0i0b1 | 0 rt16 | 0 `j<i` | 1 m18 同形 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]` / expect `[0 1 2 3]` |
| nest_o0i0b3 | 0 rt16 | 0 `j<i` | 3 +store→load | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]` / expect `[0 1 2 3]` |
| nest_o0i0b1_O2 | 0 rt16 `-O2` | 0 `j<i` | 1 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]`（与 `-O3` 逐项相同） |
| nest_o0i0b0_NOUNROT | 0 rt16 `loop-unrotate=false` | 0 `j<i` | 0 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]`（与默认逐项相同） |
| nest_o0i0b1_NOUNROT | 0 rt16 `loop-unrotate=false` | 0 `j<i` | 1 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]` |
| nest_o1i0b1 | 1 rt16 + `#pragma unroll` | 0 `j<i` | 1 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]`（unroll 无效） |
| nest_o0i1b1 | 0 rt16 | 1 `j<i−1`（带保护） | 1 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 256 257 258]` / expect `[0 0 1 2]`（错法不同！） |
| nest_o0i2b1 | 0 rt16 | 2 `j<i+1` | 1 | 6 | RUN PASS | — | `cnt=[1 2 3 4]` = expect（**反例**：不"少一次"） |
| nest_o0i3b0 | 0 rt16 | 3 `j<rows` | 0 只计数 | 6 | RUN PASS | — | `cnt=[4 4 4 4]` = expect |
| nest_o0i3b1 | 0 rt16 | 3 `j<rows` | 1 | 6 | RUN PASS | — | `cnt=[4 4 4 4]` |
| nest_o0i3b3 | 0 rt16 | 3 `j<rows` | 3 | 6 | RUN PASS | — | `cnt=[4 4 4 4]` |
| nest_o0i3b1_O2 | 0 rt16 `-O2` | 3 `j<rows` | 1 | 6 | RUN PASS | — | `cnt=[4 4 4 4]` |
| nest_o1i3b1 | 1 rt16 + unroll | 3 `j<rows` | 1 | 6 | RUN PASS | — | `cnt=[4 4 4 4]` |
| nest_o0i4b0 | 0 rt16 | 4 `hi=u16(i)` 提出 | 0 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]`（提出到循环外无效） |
| nest_o0i4b1 | 0 rt16 | 4 同上 | 1 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]` |
| nest_o0i5b1 | 0 rt16 | 5 `hi=u32(i)` 提出 | 1 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]`（换 u32 上界变量无效） |
| nest_o0i6b1 | 0 rt16 | 6 `hi=i32(i)` 提出 | 1 | 6 | **RUN FAIL** | 2,3,4,64 | `cnt=[0 0 1 2]` |
| nest_o2i0b1_ct1 | 2 编译期 rows=1 | 0 `j<i` | 1 | 1 | RUN PASS | — | （rows=1 ⇒ 内层 0 次，退化档） |
| nest_o2i0b1_ct2 | 2 编译期 rows=2 | 0 `j<i` | 1 | 1 | RUN PASS | — | `cnt=[0 1]` = expect（外层被完全展开） |
| nest_o2i0b1_ct3 | 2 编译期 rows=3 | 0 `j<i` | 1 | 1 | **RUN FAIL** | 3 | `cnt=[0 0 1]` / expect `[0 1 2]` |
| nest_o2i0b1_ct4 | 2 编译期 rows=4 | 0 `j<i` | 1 | 1 | **RUN FAIL** | 4 | `cnt=[0 0 1 2]` |
| nest_o2i0b1_ct64 | 2 编译期 rows=64 | 0 `j<i` | 1 | 1 | **RUN FAIL** | 64 | rows=64 `cnt=[0 0 1 2 … 62]`（第 i 项 = i−1） |
| nest_o2i3b1_ct4 | 2 编译期 rows=4 | 3 `j<rows` | 1 | 1 | RUN PASS | — | `cnt=[4 4 4 4]` |
| nest_o3i0b1 | 3 运行期 **u32** 归纳变量 | 0 `j<i` | 1 | 6 | RUN PASS | — | `cnt=[0 1 2 3]` = expect（**反例**） |
| nest_o4i0b1 | 4 运行期 **i32** 归纳变量 | 0 `j<i` | 1 | 6 | RUN PASS | — | `cnt=[0 1 2 3]` |
| nest_o5i0b1 | 5 运行期 **u32** IV 直接进地址算术 | 0 `j<i` | 1 | 6 | RUN PASS | — | `cnt=[0 1 2 3]` |
| nest_o0i0b2_scal | 0 rt16 | 0 `j<i` | 2 纯标量**算术** | — | **COMPILE-FAIL** | — | `Unsupported scalar instruction in AIV loop`（C15） |
| nest_o0i0b2s_scalstore | 0 rt16 | 0 `j<i` | 5 标量算术 + **标量 store** | — | **COMPILE-FAIL** | — | 同一句 fatal error（C15；评审 p2-附注要求自证） |
| nest_o0i0b4_marker_ctl | 0 rt16 | 0 `j<i` | 4 **向量** store（每轮一次，地址含 j） | — | **COMPILE-OK**（只做编译） | — | 正对照：被拒的是**标量指令**，不是"循环里做 store"（C15） |

读数说明：`rows 档位 6` = {0,1,2,3,4,64}；`rows 档位 1` = 编译期态只有一个 rows 值。
`失败档位` 列是 verdict=FAIL 的 rows 值。所有变体的 rows=0/1 都 PASS（退化档：内层 0 次），
`rows≥2` 才出现 `cnt[i] = i−1`。

**矩阵读出来的三件事**（也是 §3.2 ② 的依据）：
1. `j<i`（外层运行期 u16 或"编译期但未完全展开"）**稳定少一次**：body 无关、`-O2`/`-O3` 无关、
   `#pragma unroll` 无效、提出到循环外无效、上界变量换 u32/i32 无效。
2. **两个反例**否掉了"恒少一次"与"上界依赖外层 IV 就出错"两种简化说法：
   `j<i+1`（同样依赖外层 IV）**正确**；外层 IV 用 u32/i32 时 `j<i` **正确**。
3. 唯一在所有档位正确的两族：**上界不依赖外层归纳变量**（`j<rows`）与**上界是编译期常量**（m18 做法）。

### 4.1 每个变体的 sha256 与确定性（**按列读，别整句读**）

`evidence/logs/sha256.txt` 逐 (target, rep) 记录：`<全日志 sha256>` + `exit=` + **`judge=<判据列 sha16>`** +
`JUDGE-FIRST/SAME/DIFF`。`run_matrix.txt` 每个 target 三行：`RUN PASS/FAIL` / `rows 档位 N，FAIL M，判据列跨 rep …` /
`数据面列跨 rep …`。核验入口：`bash` 跑完会调 `tools/summarize.py verify`（三态退出码）并把结果写进
`evidence/logs/summarize_verify.txt`；负向对照写进 `summarize_selftest.txt`。

| 列 | 跨 5 次独立进程 | 说明 |
|---|---|---|
| **判据列**（`cnt`/`expect`/`tri`/`MISMATCH`/`lane_mismatch`/`tri_odd`/`verdict`） | **27/27 target 逐字符一致**（0 条差异） | 本报告的结论只依赖这一列 |
| 数据面列（`acc_maxdiff`/`hash`） | 26/27 一致；**`nest_o0i1b1` 不一致**（3 个日志 sha256） | 该变体那几行读**越界 UB**，非判据，见 §3.2 ⑧ |

⇒ **正确表述**：判据列确定；`nest_o0i1b1` 的**数据面列**非确定（位置与原因如上）。
先前版本写成"全部变体 5 次输出逐字节一致"是**错的**（评审 p2-1），已按列收窄并显式披露。

## 5. 两条次要项（finding 里提到，只给事实）

### ① `Reg::Arange` 不支持 uint32_t —— 已确证（含绕法可行性）

头文件原文（CANN 9.1.0，dav_3510）：

```cpp
// .../reg_compute/dav_3510/kernel_reg_compute_vec_arange_impl.h:47
static_assert((SupportType<ActualT, int8_t, int16_t, int32_t, float, half, int64_t>()),
              "current Arange data type is not supported on current device!");
```

`Arange<uint32_t>` 的编译错误原文（`evidence/logs/build_ag_arange_u32.log`）：

```
.../kernel_reg_compute_vec_arange_impl.h:47:5: error: static assertion failed due to requirement
  'SupportType<unsigned int, signed char, short, int, float, half, long>()':
  current Arange data type is not supported on current device!
.../kernel_reg_compute_vec_arange_impl.h:53:9: error: no matching function for call to 'vci'
```

⇒ 列表里**没有**任何无符号类型（uint8/16/32/64 全不在列）。
**绕法可行**：`Arange<int32_t>` 后用 `reinterpret_cast<RegTensor<uint32_t>&>` 交给 `Gather`
（m18 / arch35 同款写法）——本工程 `ag_gather_sweep` 就是这条路径，实测逐 lane 正确。

### ② `Reg::Gather(dst, __ubuf__ base, indexReg, mask)`（vgather2）的可寻址范围 —— 已确证

探针：UB 偏移 0 放 fp32 源数组 `NEL=16384`（64KB，值 `1000+i`），索引 `idx[lane] = start + lane*stride`
（元素），host 逐 lane 精确比对；每个 (start, stride) 档位**一个独立进程**（越界后 context 不可再用）。
全档读数见 `evidence/logs/run_matrix.txt`（`s=... x=...` 行）与 `evidence/dumps/ag_*.bin`。

| 观测 | 实测 |
|---|---|
| 源数组范围内（idx ≤ 16383） | **全部逐 lane 正确**（47 个档位里所有"数组内 lane"全部 0 不符），含 `start=8191/8192/8193`（任意非对齐）、`stride ∈ {1,2,8,64,128}` |
| 越出源数组但仍在 UB 窗口内（16384 ≤ idx ≤ 65535） | **不报错**，读到该处 UB 的**当前内容**（可能是 0、也可能是残留数据 —— 档位各异，如 start=16321 读到 17320、start=32000 读到 0.021）⇒ 静默、不回绕、不异常 |
| `idx*4 ≥ 256KB`（fp32 且 base 在 UB 偏移 0 ⇒ `max_idx = start + 63·stride ≥ 65536`） | **`aclError=507035`**（通用 aicore exception 码：只说明该组合异常，不指向具体指令） |
| 阈值精度 | stride=1 下 `start=65472`（max_idx=65535）正常，`start=65473`（max_idx=65536）即异常 —— 恰好是 UB 物理容量 256KB |

完整 47 档位表（由 `tools/summarize.py ag` 从运行日志生成；`数组内 lane 不符` 是判定列，`越界 lane` 只报事实）：

| stride | start | max_idx = start+63·stride | 数组内 lane 不符 | 越界 lane（未异常） | 越界读数是否 0 | 结果 |
|---|---|---|---|---|---|---|
| 1 | 0 | 63 | 0 | 0 | - | PASS |
| 1 | 8191 | 8254 | 0 | 0 | - | PASS |
| 1 | 8192 | 8255 | 0 | 0 | - | PASS |
| 1 | 8193 | 8256 | 0 | 0 | - | PASS |
| 1 | 16000 | 16063 | 0 | 0 | - | PASS |
| 1 | 16320 | 16383 | 0 | 0 | - | PASS |
| 1 | 16321 | 16384 | 0 | 1 | 0/1 | PASS |
| 1 | 16383 | 16446 | 0 | 63 | 0/63 | PASS |
| 1 | 16384 | 16447 | 0 | 64 | 0/64 | PASS |
| 1 | 20000 | 20063 | 0 | 64 | 0/64 | PASS |
| 1 | 32000 | 32063 | 0 | 64 | 0/64 | PASS |
| 1 | 60000 | 60063 | 0 | 64 | 64/64 | PASS |
| 1 | 65400 | 65463 | 0 | 64 | 64/64 | PASS |
| 1 | 65472 | **65535** | 0 | 64 | 64/64 | PASS（**边界内最后一档**） |
| 1 | 65473 | 65536 | - | - | - | **运行失败 aclError=507035** |
| 1 | 65500 / 65535 / 65536 / 65537 / 70000 / 131072 / 1048576 | ≥65536 | - | - | - | 全部 **运行失败 aclError=507035** |
| 128 | 0 | 8064 | 0 | 0 | - | PASS |
| 128 | 1 | 8065 | 0 | 0 | - | PASS（非对齐 start） |
| 128 | 64 | 8128 | 0 | 0 | - | PASS |
| 128 | 8064 | 16128 | 0 | 0 | - | PASS |
| 128 | 8191 / 8192 / 8193 | 16255 / 16256 / 16257 | 0 | 0 | - | 全部 PASS |
| 128 | 16000 | 24064 | 0 | 61 | 60/61 | PASS |
| 128 | 16321 | 24385 | 0 | 63 | 16/63 | PASS |
| 128 | 20000 | 28064 | 0 | 64 | 38/64 | PASS |
| 128 | 1048576 | ≥65536 | - | - | - | **运行失败 aclError=507035** |
| 2 / 8 / 64 | 0 / 16000 / 16321 | ≤20353 | 0 | 0–63 | 视档位 | 全部 PASS |

⇒ 与 finding 里「实测可寻址到 8191、未见窗口限制」一致，并给出完整边界：**可寻址范围 = base 起、落
在本核 UB 窗口内的元素**（我们 base 在 UB 内 ⇒ 窗口即整个 UB；fp32/base=0 时 idx ≤ 65535）。
越界既**不**回绕成 0 也**不**静默丢写，而是先读到 UB 现有内容、再越出 UB 时异常。
（对 m18 的用法：64×128 矩阵最大索引 8191 ≪ 65536，天然安全。）

## 6. 规避方式、适用与不适用边界

| 做法 | 实测（对应 target） | 评价 |
|---|---|---|
| **内层上界改成编译期常量**（m18 现行做法：A 严格下三角 ⇒ 多算项恒 0） | 正确（`o2i0b1_ct1/ct2` 一族；m18 实跑全 PASS） | ✅ **推荐**。代价：行前代 BT²/2 → BT²，m18 实测本核 +12% 向量指令 |
| **内层上界与外层归纳变量解耦**（换成任意**不依赖 i** 的运行期值，如 `rows`） | 正确（`o0i3b0/b1/b3`、`o1i3b1`、`o2i3b1_ct4`、`o0i3b1_O2`） | ✅ **推荐**（"固定上界 + 掩码/稀疏性让多算项为 0"的推广） |
| 把上界**提到内层循环外先算成一个变量**（`hi = (uint16_t)i` 再 `j < hi`） | **仍复现**（`o0i4b0`、`o0i4b1`） | ❌ 无效 —— docs/05 里**不要**写成规避方式 |
| 同上但上界变量类型换 u32 / i32 | **仍复现**（`o0i5b1`、`o0i6b1`） | ❌ 无效 |
| 内层加 `#pragma unroll` | **仍复现**（`o1i0b1`，编译器还报 loop not unrolled） | ❌ 无效 |
| 把**外层**上界改成编译期/模板常量 | `ct=1/2` 对，`ct=3/4/64` **仍复现**（`o2i0b1_ct*`） | ⚠️ **不是可靠规避**（取决于编译器是否把外层完全展开） |
| 外层归纳变量用 u32/i32 | 不复现（`o3i0b1`、`o4i0b1`、`o5i0b1`） | ⚠️ 只有行为事实、无机制解释，**不推荐**当规避手段（可能随编译器版本变化） |
| 内层上界写 `j < i + 1` | 不复现（`o0i2b1`） | ⚠️ 同样是"未对齐"的观测（见 §3.2 ②），**不可**当规避手段 |

**能安全说的边界**：`__simd_vf__`（VF 内）、外层为 **uint16_t 运行期循环**、内层上界写成 `j < i`
（或由 `i` 直接导出的上界变量）⇒ 内层体少执行一次（C1）。**唯一在所有档位都正确的两族**是：
上界不依赖外层归纳变量 / 上界是编译期常量。其余"有效"做法（u32 外层、`i+1`）都只有行为事实、
没有可复用的条件，不构成规则。未覆盖的形态见 §3.2 ⑦。

## 7. 给 docs/05 §6 的**建议文本**（本工程未改 docs，交 tower 定稿）

> §6 的现行结构：6.1 规格/API 约束 · 6.2 实测硬件行为 · 6.3 待复核 · 外加一张
> 「3510 VF / store / 循环类硬约束」表（a–g 行，带【已确证】/【仍存疑】标签，且明确
> "507035/507015 是通用异常码，不指向具体指令"）。下面的建议按这三处的口径写。

### 7.1 建议 6.3 新增一条（编号 #24）

**#24 VF 内层循环上界写成外层归纳变量时内层少执行一次（丢最后一项）**
- 现象（M43 探针，wt-43 `probe_vf_loop/`，27 变体 × 5 次独立进程；**判据列**跨 rep 逐字符一致，
  数据面列在病态变体 `j<i−1` 上非确定、不影响本条）：
  `__simd_vf__` 内 `for (uint16_t i = 0; i < rows; ++i) { for (uint16_t j = 0; j < i; ++j) {…} }`
  内层体实际执行 `max(i−1, 0)` 次（`rows ∈ {0,1,2,3,4,64}`；rows=0/1 退化档 0 次，均 PASS）。
  读数用向量计数器直接读 trip count：`cnt[i] = i−1`（对照 `j < rows` 时 `cnt[i] = rows` 全对）。
  内层体**只有寄存器累加、零访存**时同样复现 ⇒ 循环控制流层面，不是数据通路/可见性问题。
- 规避（这两族在全部档位都正确）：① 内层上界改**编译期常量**（m18 现行做法，A 严格下三角使多算项
  为 0；代价 BT²/2→BT²，m18 实测本核 +12% 向量指令）；② 内层上界换成任何**不依赖 i** 的运行期值。
  **实测无效**：把上界提到内层循环外先算（u16/u32/i32 上界变量都不行）、内层 `#pragma unroll`、
  把**外层**上界改成编译期常量（ct=1/2 恰好对、ct=3/4/64 仍复现 ⇒ 取决于编译器是否完全展开外层）。
- ⚠️ **条件未对齐（如实记录，不要写成规则）**：`j < i+1`（同样依赖外层归纳变量）**不复现**；
  外层归纳变量换 u32/i32 时 `j < i` **不复现**；`j < i−1`（带保护）变成另一种错（`cnt = 期望+255/+256`）；
  不带保护的 `j < i−1` 因 uint16 下溢直接越界异常（507035）；`j+1 < i` 让编译器**段错误**。
  ⇒ 建议措辞为"**观测到**：`j<i`（u16 外层）少一次；**未能把现象与单一条件对齐**"。
- 取证：`probe_vf_loop/`（`bash run_probes.sh 5` 一键复现；含逐 target 完整编译日志/编译选项、
  每个变体 5 次独立进程的完整读数与 sha256、编译不过靶子的 stderr 原文、以及两个反例的读数）。
- **归属未定**：本工程未做 ISA 级取证（设备代码在 `.aicore_binary` fatbin 内，device side 不支持
  `-S`，`msobjdump` 不是反汇编器）⇒ **不宣称**"硬件 bug"或"编译器 codegen 缺陷"。
  本条**留在 6.3**（按"未对齐"措辞）；**可用写码规则另立一条升 6.2**（见 §7.2），并从这里回指。

### 7.2 建议 6.2 新增一条**可用写码规则**（tower 裁定：升 6.2，保留回 6.3 的指针）

> 依据：失败**静默且后果严重**（少算一项、无报错），且"上界与外层归纳变量解耦"这一族在本工程全部档位
> 都正确；但**现象的条件未对齐**（§3.2 ②，两个反例），所以规则只写"怎么安全写"，不写"为什么"。

**规则（6.2 实测硬件行为 / 写码必须遵守）**
- **VF（`__simd_vf__`/`__VEC_SCOPE__`）内层循环的上界，不要写成外层归纳变量本身（`j < i`）或由它直接
  导出的上界变量。** 实测该形态内层体少执行一次（丢 `j=i−1`，静默）；**写成不依赖外层归纳变量的值
  （如 `j < rows`）或编译期常量**（m18 现行做法：A 严格下三角使多算项为 0）在全部档位都正确。
- **不要**用这些"看起来能规避"的写法（实测**无效或不可靠**）：把上界提到内层循环外先算（u16/u32/i32
  上界变量都不行）、内层 `#pragma unroll`、把**外层**上界改成编译期常量（取决于编译器是否完全展开外层）。
- **未对齐指针**：本规则是"安全写法"，不是机制结论；为什么 `j<i` 错而 `j<i+1`/u32 外层对，**未对齐**，
  详见 §6.3 #24（禁止当因果规则引用）。

### 7.3 建议在「VF / store / 循环类硬约束」表新增/修订四行

| # | 约束 | 标签 | 复现条件 / 依据 |
|---|---|---|---|
| **h（新增·硬规则）** | **VF（`__VEC_SCOPE__` / `__simd_vf__`）循环体内不得出现任何标量指令**：标量整数算术（`cntS = cntS + 1;`）与标量 store（`mk[i*RW+j] = j;` 逐轮写标记）都**编译期**报 `fatal error: error in backend: Unsupported scalar instruction in AIV loop`。**正对照**：同样"每轮一次 store"，但用**向量** `StoreAlign`（地址含 j、值不含 j）**能编过** ⇒ 被拒的是标量指令本身，不是"循环里做 store"。⇒ VF 内计数/标记只能用向量寄存器，或"循环外落盘" | 【已确证】（tower 裁定：**升为硬规则**，不放在说明里 —— 确定性编译期硬错误） | M43 `probe_vf_loop`：靶子 `nest_o0i0b2_scal`（标量算术，`evidence/logs/build_nest_o0i0b2_scal.log`）+ `nest_o0i0b2s_scalstore`（标量算术+标量 store，`build_nest_o0i0b2s_scalstore.log`）+ 正对照 `nest_o0i0b4_marker_ctl`（**COMPILE-OK**，`build_nest_o0i0b4_marker_ctl.log`） |
| i（新增） | **`Reg::Gather(dst, __ubuf__ base, idxReg, mask)`（vgather2）按元素索引，可寻址范围 = 本核 UB 窗口**：idx 越出源数组但仍在 UB 内 = 静默读到该处 UB 内容（不回绕、不异常）；`idx × sizeof(T) ≥ 256KB` → `507035`（通用异常码，只说明该组合异常）。实测 base 在 UB 偏移 0、fp32：`max_idx=65535` 正常、`=65536` 异常 | 【已确证】 | M43 `probe_vf_loop`（`ag_gather_sweep` 全档 47 档位；`8191/8192/8193`、任意非对齐 start、`stride ∈ {1,2,8,64,128}` 的源数组内 lane 全对） |
| **j（新增，替代原"并入 e"的做法）** | **`Reg::Arange` 不支持任何无符号类型**：`static_assert((SupportType<ActualT, int8_t,int16_t,int32_t,float,half,int64_t>()))` ⇒ `Arange<uint32_t>` 编译失败（`current Arange data type is not supported on current device!`）。索引寄存器绕法 = `Arange<int32_t>` + `reinterpret_cast<RegTensor<uint32_t>&>` 交给 `Gather`（arch35/m18 同款，实测逐 lane 正确） | 【已确证】 | M43 `probe_vf_loop`（靶子 `ag_arange_u32` 原文 + `ag_gather_sweep` 正常路径）。**注**：行 e「Arange 只填 64 lane」是 **VL 语义**（另一件事），与本条无符号类型不支持**拆成两行**，不要并条 |
| d（修订） | 行 d 现说"AIV 循环内不支持 scalar float 算术"⇒ **收窄/纠正**：实际是**任何标量指令都不行**（整数算术同样不行，且**标量 store 也不行**）：行 h 已独立列出，行 d 里这句删掉或指向 h | 【已确证·修订】 | 同上（h 的三个靶子）；原行 d 依据（M14 编译期报错 + M23 复现）与 h 不冲突 |


## 8. 证据索引

```
probe_vf_loop/
├── CMakeLists.txt                 # 独立 ASC 工程（find_package(ASC) + --npu-arch=dav-3510）
├── probe_vf_nest.asc              # 探针本体（嵌套循环 trip count 矩阵，宏选变体）
├── probe_arange_gather.asc        # 次要项 ①②（Arange 支持类型 / Gather 可寻址范围）
├── run_probes.sh                  # 一键复现 + 归档（含预期编译失败靶子/正对照的单独构建）
├── tools/summarize.py             # 矩阵表回放 + **跨 rep 核验（三态退出码）** + 汇总重建 + selftest
└── evidence/
    ├── commands.txt               # 环境/编译器版本 + 复现命令
    ├── negmark/                   # body 4 标记矩阵的异常原始日志（§3.2 ⑤）
    ├── unguarded_inner1/          # 不带保护的 `j<i−1` 下溢 ⇒ 507035 的原始日志（C12）
    ├── crash_inner7/              # `j+1<i` 让编译器段错误的原始 stderr 全文（C13）
    ├── dumps/                     # gather 扫描的 64 lane 输出 .bin
    └── logs/
        ├── build_<target>.log     # 逐 target 完整构建输出（VERBOSE=1：含完整编译命令行）
        ├── bisheng_verbose_cc1_flags.txt  # `bisheng -v` 原文：驱动内部全部 -mllvm 开关
        ├── compile_matrix.txt     # 编译通过/失败矩阵
        ├── compile_options.txt    # 逐 target 的完整编译选项摘要
        ├── run_<target>_rep<k>.log# 第 k 次独立进程运行的完整读数（**原始证据**）
        ├── run_<target>_s<start>_x<stride>.log  # gather 逐档位读数（原始证据）
        ├── run_matrix.txt         # 运行矩阵（**由 summarize.py reindex 从原始日志重建**；
        │                          #   每 target 三行：RUN PASS/FAIL、判据列跨 rep、数据面列跨 rep）
        ├── sha256.txt             # 逐 (target, rep) 摘要（同上重建）：日志 sha256 + exit + **判据列 sha16** + JUDGE-SAME/DIFF
        ├── summarize_verify.txt   # 跨 rep 核验 + 判据列独立复算（三态退出码 0/1/2 + 覆盖范围自述）
        └── summarize_selftest.txt # 负向对照：/tmp 副本里改字节 ⇒ 核验会报差异（不改工作区）
```

> **重建说明**：`run_matrix.txt` / `sha256.txt` 是**派生索引**，由 `tools/summarize.py reindex` 从
> `run_*.log`（原始证据）重建 —— 同一份原始日志必得同一份索引，且重建后**判据列结论与逐 rep 签名
> 都在文件里**（评审 p2-1/p2-2 的整改：先前这两个文件由脚本内第二份格式化实现写出，丢了 target 标签，
> 也没把"判据列 vs 数据面列"分开陈述）。原始 `run_*.log` **未做任何修改**。

## 9. 与 finding / m18 的对齐

| finding 的说法 | 本工程的实测 |
|---|---|
| 内层上界依赖外层归纳变量 ⇒ 内层少执行一次（丢 j=i−1，系数恒 ≈0） | **一致**：`cnt[i] = i−1`；`acc` 与 host 参考 `ref(cnt)` 逐位相等、与 `ref(expect)` 差 320/576/1200…（§2.4）；`jset_odd=0` 说明执行的 j 就是前缀 `0..i−2`（丢的正是最后一项） |
| 同 kernel 里内层上界不依赖外层变量 ⇒ 逐位级一致 | **一致**：`j < rows` 全档 PASS |
| 怀疑与「rows 是运行期参数、不能展开」有关 | **部分一致**：编译期 `ct=1/2` 全对、`ct≥3` 仍复现 ⇒ "外层被完全展开"确实能规避，但**不是"编译期 vs 运行期"这么简单**（§3.2 ④）；且 `j<i+1` 也是运行期却正确（§3.2 ②） |
| donor 用模板常量 N=32 未被发现 | 同上；且 donor 判据是 bf16 三方 cross_check，容差较松，与"小 N 完全展开"叠加更不易暴露 |
| 内层上界改编译期常量后 u/w 误差 8.7e-8/1.1e-8 | **未复测 m18 数值**（本工程不碰 m18 范围）；本工程只证明"内层上界与 i 解耦 / 改常量"这一族规避在循环控制流层面确实成立（全部档位 PASS） |
| 次要项① Arange 不支持 uint32_t；绕法可行 | **一致**（C16），并给出 `static_assert` 原文与绕法的逐 lane 验证 |
| 次要项② Gather 可寻址到 8191、未见窗口限制 | **一致且更强**：窗口 = 整个 UB（fp32/base=0 时 idx ≤ 65535），越界先静默读 UB、再越出 UB 报 507035（C17 + §5② 全表） |
