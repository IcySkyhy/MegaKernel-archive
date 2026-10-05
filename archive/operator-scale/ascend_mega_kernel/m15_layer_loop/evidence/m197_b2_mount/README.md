# M197：attention prefill B2 前端挂进层路径 —— 设备证据

## 本目录

| 文件 | 内容 |
|---|---|
| `m197_layer_path_run.log` | `M15_PREFILL_WIRE=1 M15_PREFILL_KIND=3 M15_PREFILL_M=1 ./m15_layer_loop weights_manifest.txt prefill` 的完整 stdout（层路径 B2 见证 + 整臂响亮失败） |
| `m197_m24_fp64_run.log` | M116 独立验证路 `./m24_attn_prefill`（未改）的完整 stdout：B2 段体的 fp64 判据读数 |
| `attn_fa_flag_unbalanced.md` | §4f.2 的 flag 用量判据重推（硬件 4bit 计数器 = 未配平累计；B3 未配平峰值 = 2） |

## 命令（离线可复现；设备纪律见 `.tower/comms` 的 flock 口径）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B /tmp/m197_build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/m197_build -j8
flock -w 300 /tmp/npu0.lock bash -c 'npu-smi info | sed -n "1,16p";
  M15_PREFILL_WIRE=1 M15_PREFILL_KIND=3 M15_PREFILL_M=1 \
    timeout 200 /tmp/m197_build/m15_layer_loop m15_layer_loop/weights_manifest.txt prefill'

cmake -B /tmp/m24_build -S m24_attn_prefill -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/m24_build -j8
flock -w 300 /tmp/npu0.lock bash -c 'npu-smi info | sed -n "1,16p"; timeout 500 /tmp/m24_build/m24_attn_prefill'
```

## 读数

### 1) B2 在真实层路径里被调用（正档，wired=1）

逐字（`m197_layer_path_run.log`）：

```
[m15]   Pf.attn：M197 B2 层路径（wired=1, mUse=64）⇒ y0 活行 64/64（[64,128) 仍毒 0）、out 活行 64（尾毒 0）、
        in 活字节 149247、kv 131072、comp 4096、ring 1120、pack 16352、pooled 4096、flag 32
```

- `y0`/`out` 的 **行数契约**成立：只 `[0,64)`（本 chunk 一个整 m-tile）被写，`[64,128)` **仍是 0xCD 毒值**；
- `in`（cache 输入面）、主 KV（k=0，4 页 × 32768 B = 131072 活字节）、compressed（4096）、raw ring（1120）、
  packed seed 抄写（16352）、pooled（4096）、门控 lane（32）全部从毒值变活。

### 2) 挂载关掉则不写（负控，wired=0）

逐字：

```
[m15]   Pf.attn：M197 B2 负控（wired=0）⇒ y0 活字节 0、out 0、in 0、kv 0、ring 0
        （应全 0 = 挂载关掉不写，仍是毒；pack 是输入 seed 面，不判）
```

⇒ 「plan 变活」这条判据**非空洞**：挂载关掉时同一判据读数为 0。（`pack` 是本档 H2D 预置的输入 seed 面，
不管挂载开关都"活"，故不作判据。）

### 3) 整条 attention 臂响亮失败（不变量 ②）

逐字（同一次运行）：

```
[m15][FAIL] M15_PREFILL_WIRE=1 且 M15_PREFILL_KIND=3 含 attention：B2 前端已挂载（部分产出），
           但 B3（稠密 causal core）未接 ⇒ 整条 attention 臂不完整（响亮失败，rc≠0）
...
[m15] ===== FAILURES PRESENT（checks=153, guards=57, fails=1）=====
```

⇒ 进程 rc=1；`fails=1` 恰是这条"B3 缺席"的硬失败（B2 的 9 条 wiring 判据全 PASS）。

### 4) B2 段体的 fp64 数值（M116 独立验证路；未改）

逐字（`m197_m24_fp64_run.log`）：

```
[M24]   big4097_step：checks=327 fails=3 guards=65(guardFails=0)
[M24]   失败判据（前若干个）： Pf.in.kv.c4 Ac.mkv.c4 Ac.mkv.final      （= 已接受的 1 ulp K 槽 tie）
[M24]   m1ctx4097：checks=23 fails=0 guards=1(guardFails=0)
[M24]   rag65_step：checks=93 fails=0 ...
[M24]   m128：checks=31 fails=0 ...   m64：checks=23 fails=0 ...   m2：checks=23 fails=0 ...
[M24]   big4097_seq_refused：launch 已中止（= 期望的响亮失败）⇒ nChunks>1 的 Trap 硬拦仍在
```

⇒ 与 M176 survey 引用的 M116 读数逐项一致（327 / 23 / 31），且 12 个变异档全红。

## 与「B2 在层路径」的关系

层路径（`M15L_PrefillPhaseA<KIND_ATTN>`）调用的是**同一个** `M15PF::AttnPrefillPhaseA` 段体
（即 M116 fp64 判据所测的对象）；M197 的层路径见证额外覆盖**挂载层**：逐 chunk 循环、行偏移、
KV/cache 的 `k` 层基址偏移、以及"行数契约只写本 chunk 行"。
