# M5 证据归档（M32：MX 量化非有限 parity 修复）

| 文件 | 内容 |
|---|---|
| `run_after.log` | 修复后全量运行：**104 条判据全 PASS**（基线 25 组 × 2 判据 + 非有限 27 组 × 2 判据），rc=0 |
| `run_prefix_nofix.log` | **修复前**（删掉官方 `:413` 那句 `Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);` 后重编）：86 PASS / 18 FAIL —— (a) SwiGLU 判据全 PASS；(b) 量化判据 inf=1/2 全 FAIL（device 出 `0x77` 等 ±6 饱和码）、inf=3（NaN）本来 PASS |
| `check_ref_inf.log` | `check_ref.py`（numpy 独立见证）对 `m=17 seed=0 inf ∈ {0..3}` 的 dump：silu ULP + 量化逐字节 4 项判据全 PASS |
| `ocp_independent_witness.log` | **独立见证**：`M5_DUMP=1` 的全部 52 个用例（量化段输入 = device 自己的 SwiGLU 输出）与 M26 分支 `6f68f8e` 的 `tools/golden/moe_block_ref.quantize_ocp`（官方 ops-nn 序列的逐句 numpy 转写，另一个 agent 的独立实现）**逐字节一致**：52/52 PASS |
| `dump_sha256.txt` | `inf_dumps/` 下 12 个 dump 的 sha256（按统一验收口径入库） |
| `inf_dumps/m1_s0_i{1,2,3}/` | 非有限用例的小尺寸 dump（`x` / `swiglu_ref` / `qx_ref` / `scale_ref`，C 参考链输出），供离线复算 |

## 复跑

```bash
cd <repo> && source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m5_swiglu_quant/build -S m5_swiglu_quant -DCMAKE_BUILD_TYPE=Release && cmake --build m5_swiglu_quant/build -j2
./m5_swiglu_quant/build/m5_swiglu_quant                   # 期望 ALL PASS，104 条判据

# 修复前对照（临时副本，只删那一句 Select）
cp -r m5_swiglu_quant /tmp/m5_prefix
sed -i '/nanRegTensor, cmpResult/d' /tmp/m5_prefix/m5_swiglu_quant.asc
cmake -B /tmp/m5_prefix/build -S /tmp/m5_prefix -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/m5_prefix/build -j2
/tmp/m5_prefix/build/m5_swiglu_quant                      # 期望 86 PASS / 18 FAIL（(b) inf=1/2 全 FAIL）

# numpy 独立见证（dump 第 4 个参数 = infKind）
mkdir -p /tmp/m5_dump && cd /tmp/m5_dump
<build>/m5_swiglu_quant dump 17 0 1
<build>/m5_swiglu_quant dump 17 0 3
M5_DUMP=1 <build>/m5_swiglu_quant                         # 落盘全部用例（含 inf=1..3）的 device 输出
/usr/local/python3.12.13/bin/python3 <repo>/m5_swiglu_quant/check_ref.py 17 0 1
```

## 判据口径

`infKind = 0` 为既有基线（25 组）；`infKind ∈ {1,2,3}` 为任务 ①②③（整组 amax=+Inf / 组内部分 ±Inf / NaN 组），在 **gate|up** 上注入，经 SwiGLU 后再量化。判据：

* (a) SwiGLU：device bf16 vs host `expf` 五元组参考，bf16 网格 ≤1 ULP，且「有限/非有限归类」逐元素一致（同类非有限值即等：NaN 之间不比符号/payload，±Inf 各自同档）；
* (b) 量化：对 **device 自己的 bf16 输出**施以 host C 参考（官方语义，含 `:413`），qx 与 scale **逐字节一致**；另打印「注入组数 / device `scale=0xFF` 组数 / 注入组 nibble 非零数」作见证（修复后注入组 nibble 全 0）。

`check_ref.py` 的 info 行（numpy 全链 vs device qx）在 inf=3 只有 4080/5440 字节一致，原因见 README「已知限制」：设备 `Cast<float→bf16>(NaN)` 给正值 `0x7FFF`（丢 NaN 符号），x86 host 保留符号，故 NaN 组「零的符号位」两条链不同 —— 该行是**信息项**，不是判据。
