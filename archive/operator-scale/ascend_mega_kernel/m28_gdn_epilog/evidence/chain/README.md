# m28_epilog_chain 证据（M149）

采集时刻：2026-10-04。来源：`bash m28_gdn_epilog/reproduce_chain.sh /tmp/m28c_run`
（正向档 `M28C_CASES=4,4097`；负向档 `M28C_MUTS=nogamma,zhead0` × `M28C_MUT_CASES=4,4097`）。
全程单进程、**每档各自进一次** `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（快照落盘）、
`timeout 280` 放在锁内；本轮 6 次进锁均取得读数（各档 log 首行均有 `[lock] acquired`，见
`pos_m*.log` / `mut_*_m*.log`）。

## 文件

| 文件 | 内容 |
| ---- | ---- |
| `sources_sha256.txt` | 本链段源文件（`m28_epilog_chain.asc` / `check_chain_ref.py` / `reproduce_chain.sh` / `CMakeLists.txt` / `README.md`）采集时刻的 sha256（在 `m28_gdn_epilog/` 内 `sha256sum -c evidence/chain/sources_sha256.txt`） |
| `cmake.log` / `build.log` | 独立工程配置与构建（rc=0；两个 target 均 built） |
| `pos_m{4,4097}.log` | 正向链段设备档（设备 stdout；含 `[lock] acquired`） |
| `mut_{nogamma,zhead0}_m{4,4097}.log` | 负向对照设备档（S5 mutant） |
| `check_chain_pos.log` | 链端到端判据（参考 = `m21_layer_ref/ref/gdn.py`） |
| `check_chain_mut_{nogamma,zhead0}.log` | 负向链判据（S5 必须变红，且须传播到 `hcAttnOut`） |
| `check_m12_m{4,4097}.log` | S5 段判据（`m12_rmsnorm_gated/check_ref.py`） |
| `check_m11_m{4,4097}.log` | S6 段判据（`m11_bf16_gemm/check_ref.py`，精确整数域、逐位） |
| `check_m12_mut_{nogamma,zhead0}_m{4,4097}.log` | mutant 档的 S5 判据（预期 FAIL） |
| `npu_smi/*.txt` | 各档进锁后的 `npu-smi info` 快照（6 档 Health 均 `OK`、无设备进程） |

## 1. 逐字命令

```bash
cd <repo>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash m28_gdn_epilog/reproduce_chain.sh /tmp/m28c_run
```

等价的分步命令（`reproduce_chain.sh` 内部即此）：

```bash
cmake -B m28_gdn_epilog/build -S m28_gdn_epilog -DCMAKE_BUILD_TYPE=Release
cmake --build m28_gdn_epilog/build -j4            # 两个 target

# 正向档（每档一次 flock）
flock -w 300 /tmp/npu0.lock bash -c "cd /tmp/m28c_run/pos && npu-smi info > npu_smi_m4.txt 2>&1 && \
  M28C_DUMP=1 M28C_OUT=/tmp/m28c_run/pos timeout 280 m28_gdn_epilog/build/m28_epilog_chain 4"
# m=4097 同上

# 负向档（S5 mutant），例：
flock -w 300 /tmp/npu0.lock bash -c "cd /tmp/m28c_run/mut_nogamma && npu-smi info > npu_smi_m4.txt 2>&1 && \
  M28C_DUMP=1 M28C_OUT=/tmp/m28c_run/mut_nogamma M28C_MUT=nogamma timeout 280 m28_gdn_epilog/build/m28_epilog_chain 4"

# 判据
/workspace/venvs/baseline/bin/python3 m28_gdn_epilog/check_chain_ref.py /tmp/m28c_run/pos --mode pos
/usr/local/python3.12.13/bin/python3 m12_rmsnorm_gated/check_ref.py 4 0        # 在 /tmp/m28c_run/pos 内
/usr/local/python3.12.13/bin/python3 m11_bf16_gemm/check_ref.py 6144 2560 4   # 在 /tmp/m28c_run/pos/m4_s6exact 内
```

## 2. 正向读数（逐字）

设备档（`pos_m*.log`）：

```
[M28C] chain_m4    : PASS (o 0 bad, z 0 bad, S5 y match bad=0 worstRel=0.00412 first=0, m=4    mutant=0, blk=28)
[M28C] chain_m4097 : PASS (o 0 bad, z 0 bad, S5 y match bad=0 worstRel=0.00781 first=0, m=4097 mutant=0, blk=28)
```

链端到端判据（`check_chain_pos.log`；参考 = `m21_layer_ref/ref/gdn.py`）：

```
m=4    : o(T1)=True z(T1)=True | S5 bf16 rel<=0.01: True (max 0.00559) | S6 T3-isolated over: 0/10240      | chain T3 over-frac 0      -> PASS
m=4097 : o(T1)=True z(T1)=True | S5 bf16 rel<=0.01: True (max 0.00781) | S6 T3-isolated over: 0/10488320 | chain T3 over-frac 3.81e-07 -> PASS
===== chain check mode=pos: PASS =====
```

分段判据：

```
m12 check_ref m=4    : NPU-dev vs numpy: out rows all within 0.01 rel: True (max rel 0.00559)
m12 check_ref m=4097 : NPU-dev vs numpy: out rows all within 0.01 rel: True (max rel 0.00781)
m11 check_ref m=4    : numpy fp32 ref == device C: True
m11 check_ref m=4097 : numpy fp32 ref == device C: True
```

## 3. 负向对照读数（逐字）

S5 段两 mutant（都不越界、确定性；只扰动 S5→`hcAttnOut` 一路，段①转位不受影响）：

| mutant | 语义 | m | 段① T1 | S5 判据 | 链端到端 T3 over-frac | 读数出处 |
| ------ | ---- | - | ------- | ------- | --------------------- | -------- |
| `nogamma` | gamma 预转表写成全 1.0（权重未生效） | 4 | o/z 位级一致 | FAIL，max rel **105**，24192/24576 超界 | 1.0 | `mut_nogamma_m4.log`、`check_chain_mut_nogamma.log`、`check_m12_mut_nogamma_m4.log` |
| `nogamma` | 同上 | 4097 | 位级一致 | FAIL，max rel **105**，5900739/25171968 超界 | 1.0 | `mut_nogamma_m4097.log` 等 |
| `zhead0` | 所有 head 都用 head 0 的 z 做 sigmoid 门 | 4 | 位级一致 | FAIL，max rel **8.39e+06**，5308/24576 超界 | 0.998 | `mut_zhead0_m4.log`、`check_chain_mut_zhead0.log`、`check_m12_mut_zhead0_m4.log` |
| `zhead0` | 同上 | 4097 | 位级一致 | FAIL，max rel **8.39e+06**，5372005/25171968 超界 | 0.997 | `mut_zhead0_m4097.log` 等 |

**签名**：两 mutant 下段①（转位 o / z 落位）仍与参考**逐位相等**（README §2 的 T1 判据不动），
而 S5 出口与链出口的 `hcAttnOut` 都变红 ⇒ 正/负向判据确实盯着 S5→S6 这一段，不是空洞通过。
`zhead0` 的 `chain T3 over-frac` 为 0.997/0.998（非 1.0），正是「h=0 列仍对」的指纹——
与「只扰动非 0 head 的 z」一致。

## 4. 容差分档（与 `docs/22-prefill-prolog-epilog-wiring.md:333-344` 对齐）

| 段 | 档 | 判据 | 依据 |
| -- | -- | ---- | ---- |
| 段① 转位 o / z 落位 | **T1（容差 0）** | device 与 C 参考、numpy 参考逐位（fp32 视 uint32、bf16 视 uint16） | 纯数据搬移 |
| 段② S5 RMSNormGated | bf16 网格 rel ≤ 1e-2 | `m12_rmsnorm_gated/check_ref.py` 既有口径 | 出口 bf16，网格量化 ~4e-3 相对 |
| 段③ S6 out_proj | **T3** `|got-ref| ≤ ε·Σ|terms| + 0.5·ulp_bf16(ref)`，ε=5e-5 | `docs/22:341` / `m18 check_ref` 口径 | 含 fp32 累加链 |
| 链端到端 | T1 + S5 + S6 三段合取 | `check_chain_ref.py` | 参考 = `m21_layer_ref/ref/gdn.py` |

说明：mission 逐字给出的另一支口径 `1e-5·|exp|+1e-6` 是**逐元素相对式**；S5/S6 的出口是 bf16，
该式对正确核也会「超界」（bf16 量化本身 ~4e-3 相对），脚本把它作为**信息量计数**报告
（`check_chain_pos.log` 的 `(info) strict 1e-5|exp|+1e-6 count`），不作为 PASS 条件。S6 采用
`docs/22:341` 的 `ε·Σ|terms| + 0.5·ulp` 形式（ε=5e-5，与 m18/m11 同一口径）。

## 5. 口径说明（本段未做 / 未主张）

- **未接进 `m15_layer_loop`** 的 epilog 挂载点；不改 B1 的 `wsO` 契约（`m15_layer_kernel.h:263`）。
- `z` 仍为**合成输入**（in_proj 生产者未接，`docs/22` §2.3 G1-c / §3.3 G2-b）；`wsO` 亦为合成输入。
- 未做多 stage 流水 / 带宽调优（链段以正确性为先）。
- **不主张端到端**：残差 S7、HC 边界 #2 的 `bo` 落位不在本段；H2 挂载点未接。
- 设备已实测 m 集合 = {4, 4097}；更大 m 未逐个实测。
