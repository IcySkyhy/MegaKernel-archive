# M2 证据归档（M32：MX 量化非有限 parity 修复）

| 文件 | 内容 |
|---|---|
| `run_after.log` | 修复后全量运行：**104 组判据全 PASS**（50 组基线 + 54 组 ±Inf/NaN 非有限），rc=0 |
| `run_prefix_nofix.log` | **修复前**（删掉官方 `:413` 那句 `Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);` 后重编）：判据行 68 PASS / 36 FAIL，inf=1/2 全 FAIL（device 出 `0x77` 等 ±6 饱和码），inf=3（NaN）本来 PASS |
| `check_ref_inf.log` | `check_ref.py`（numpy 独立见证）对 `K ∈ {640,2560} × m ∈ {1,17} × seed=0 × inf ∈ {0..3}` 的 dump 逐字节比对：16/16 全 True |
| `check_ref_inf_prefix_numpy.log` | 同一 dump 上**修复前的 check_ref.py**（HEAD 版）与修复后版本的对比：修复前 inf=1 算出 `0x77`、inf=3 算出 `0x7f`（`np.digitize(NaN)`=7 与算术式 bf16 解码所致），修复后 byte-exact |
| `ocp_independent_witness.log` | **独立见证**：`M2_DUMP=1` 的全部 104 个 device dump 与 M26 分支 `6f68f8e` 的 `tools/golden/moe_block_ref.quantize_ocp`（官方 ops-nn 序列的逐句 numpy 转写，另一个 agent 的独立实现）**逐字节一致**：K=640 52/52、K=2560 52/52 |
| `dump_sha256.txt` | `inf_dumps/` 下 9 个 dump 的 sha256（按统一验收口径入库） |
| `inf_dumps/K640_m1_s0_i{1,2,3}/` | 非有限用例的小尺寸 dump（`x.bin` / `qx.bin` / `scale.bin`，C 参考输出），供离线复算 |

## 复跑

```bash
cd <repo> && source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m2_mxfp4_quant/build -S m2_mxfp4_quant -DCMAKE_BUILD_TYPE=Release && cmake --build m2_mxfp4_quant/build -j2
./m2_mxfp4_quant/build/m2_mxfp4_quant                     # 期望 ALL PASS，104 组

# 修复前对照（临时副本，只删那一句 Select）
cp -r m2_mxfp4_quant /tmp/m2_prefix
sed -i '/nanRegTensor, cmpResult/d' /tmp/m2_prefix/m2_mxfp4_quant.asc
cmake -B /tmp/m2_prefix/build -S /tmp/m2_prefix -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/m2_prefix/build -j2
/tmp/m2_prefix/build/m2_mxfp4_quant                       # 期望 68 PASS / 36 FAIL（inf=1/2 全 FAIL）

# numpy 独立见证（dump 第 5 个参数 = infKind：1 整组 +Inf / 2 部分 ±Inf / 3 NaN 组）
mkdir -p /tmp/m2_dump && cd /tmp/m2_dump
<build>/m2_mxfp4_quant dump 640 17 0 1
/usr/local/python3.12.13/bin/python3 <repo>/m2_mxfp4_quant/check_ref.py 640 17   # 期望两行 True
```

判据口径：`infKind = 0` 为既有基线（`quant_ref.py` 算法，50 组）；`infKind ∈ {1,2,3}` 为任务 ①②③，参考侧走官方语义（组 `maxexp == 0x7F80 → scale = 0xFF、halfScale = 0x7F81 → Mul(±Inf, NaN) → Cast → 0.0`）。device 与参考**逐字节**比对 `qx` 与 `scale`；非有限用例另打印「注入组数 / device `scale=0xFF` 组数 / 注入组 nibble 非零数」作见证。
