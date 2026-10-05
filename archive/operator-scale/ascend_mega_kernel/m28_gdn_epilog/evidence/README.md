# m28_gdn_epilog 证据（M148）

采集时刻：2026-10-04。来源：`bash m28_gdn_epilog/reproduce.sh /tmp/m28_out`
（正向档 `M28_CASES=1,4,4097`，负向档 `M28_MUT_CASES=4,4097`），全程单进程、每档各自
进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`、`timeout 180` 放在锁内。

## 文件

| 文件 | 内容 |
| ---- | ---- |
| `sources_sha256.txt` | 采集时刻 5 个源文件的 sha256 |
| `cmake.log` / `build.log` | 独立工程配置与构建（rc=0） |
| `reproduce_output.log` | `reproduce.sh` 全程 stdout |
| `m28_m{1,4,4097}.log` | 正向设备档（设备 stdout；每档含 `[lock] acquired` 行） |
| `mut_m28_m{4,4097}.log` | 负向对照设备档（`M28_MUTANT=headstride`） |
| `check_pos.log` | numpy 参考正向判据（位级） |
| `check_mut.log` | 负向判据（o 必须判红、z 仍须位级相等） |
| `*npu_smi*.txt` | 每档进锁后的 `npu-smi info` 快照 |

## 读数（逐字）

正向设备档（`m28_m*.log`）：

```
[M28] m28_m1    : PASS (o vs C-ref 0 bad, z vs C-ref 0 bad, m=1    mutant=0, blk=28)
[M28] m28_m4    : PASS (o vs C-ref 0 bad, z vs C-ref 0 bad, m=4    mutant=0, blk=28)
[M28] m28_m4097 : PASS (o vs C-ref 0 bad, z vs C-ref 0 bad, m=4097 mutant=0, blk=28)
```

正向判据（`check_pos.log`，numpy 参考，位级）：

```
m=1    : o bit-exact=True z bit-exact=True -> PASS
m=4    : o bit-exact=True z bit-exact=True -> PASS
m=4097 : o bit-exact=True z bit-exact=True -> PASS
===== mode=pos: PASS =====
```

负向对照（`mut_m28_m*.log` + `check_mut.log`，`M28_MUTANT=headstride`）：

```
[M28] mut_m28_m4    : FAIL (o vs C-ref 5 bad, z vs C-ref 0 bad, m=4    mutant=1)
[M28] mut_m28_m4097 : FAIL (o vs C-ref 5 bad, z vs C-ref 0 bad, m=4097 mutant=1)
mut m=4    : o bit-exact=False (want False) z bit-exact=True (want True) -> PASS (o first mismatch [0][128], total 24064)
mut m=4097 : o bit-exact=False (want False) z bit-exact=True (want True) -> PASS (o first mismatch [0][128], total 24647552)
===== mode=mut: PASS =====
```

说明（判据非空洞）：headstride 错值下 **o 路**与正确参考不同（m=4 有 24064 / 24576 个元素
不同；m=4097 有 24647552 / 25171968 个元素不同，恰好只剩 h=0 列因两式在该列重合而相等），
**z 路**未被扰动、仍逐位等于参考。该对照证明正向判据确实盯着 o 转位，不是空洞通过。

## 复现命令

```bash
cd <repo>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m28_gdn_epilog/build -S m28_gdn_epilog -DCMAKE_BUILD_TYPE=Release
cmake --build m28_gdn_epilog/build -j4
M28_CASES=1,4,4097 M28_MUT_CASES=4,4097 bash m28_gdn_epilog/reproduce.sh /tmp/m28_out
```

`npu-smi` 快照里 4 档的 Health 均为 `OK`；进锁时 NPU 0 无其它设备进程（各档
`Process id` 表为空）。
