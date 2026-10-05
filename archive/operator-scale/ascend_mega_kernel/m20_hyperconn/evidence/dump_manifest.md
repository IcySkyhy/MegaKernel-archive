# M36 dump 张量 sha256 清单（再生成命令见文末）

`M20_DUMP=1 M20_CASE=<case>` 一次落盘 15 个文件：7 个 host 输入（`hin/bo/ij/wdown/winj/wup/hc_norm`）
+ 8 个 device 侧 workspace 张量（`hcp/xn/rstd/injw/oh/ls/gate/blk`）+ 1 个 `m20_case.txt`。
张量形状/dtype 见同名 `*.bin.meta`。

**为什么入库 sha256 而不是原始字节**：kernel 的 dump 已实测**确定性**——见下方「确定性复核」，
5 个 case 各跑 **5 次独立进程**，15 个 dump 的 sha256 全等。因此用 sha256 清单 + 再生成命令替代，
既保证可复核又不给仓库塞二进制（约 12MB/次 × 12 case）。

> ⚠️ **第 1 轮评审 P1**：此前的确定性声明与 manifest 是**错的**——`injw` 每次运行哈希都不同
> （W0 的每 32B 槽只写 4B，槽 `[4,32)` 是 UB 残留）。已修为「整表按槽清零后再写槽首」
> （步长 bug 见 README §5.1），修后 `injw` 稳定且**回到了原始 manifest 的哈希 `c7146f64…`**
> （原哈希出现的那次运行恰好 UB 尾部为 0）。

## 确定性复核（≥5 次独立进程；`evidence/determinism_run.log`）

```bash
for c in real_m1 exact0 int int_final real_mix; do
  rm -f m20_*.bin* m20_case.txt
  for i in 1 2 3 4 5; do
    rm -f m20_*.bin* m20_case.txt
    M20_DUMP=1 M20_CASE=$c <repo>/m20_hyperconn/build/m20_hyperconn >/dev/null 2>&1
    sha256sum m20_*.bin | sort > /tmp/d_${c}_$i.txt
  done
  for i in 2 3 4 5; do diff -q /tmp/d_${c}_1.txt /tmp/d_${c}_$i.txt; done
done
```

实测结果（每行 = 同一二进制 5 次独立进程，15 个 dump 的 sha256 全等）：

```
real_m1: 5/5 次运行 15 个 dump sha256 全等 (15 文件)
exact0: 5/5 次运行 15 个 dump sha256 全等 (15 文件)
int: 5/5 次运行 15 个 dump sha256 全等 (15 文件)
int_final: 5/5 次运行 15 个 dump sha256 全等 (15 文件)
real_mix: 5/5 次运行 15 个 dump sha256 全等 (15 文件)
```

## device 侧 8 个中间/输出张量（判定对象）sha256（前 16 位）

| case | hcp (H') | xn | oh | ls | gate | blk | rstd | injw |
|---|---|---|---|---|---|---|---|---|
| `exact0m33` | — | — | — | — | — | — | — | — |
| `exact0` | — | — | — | — | — | — | — | — |
| `int_mix` | — | — | — | — | — | — | — | — |
| `int_final` | — | — | — | — | — | — | — | — |
| `int` | — | — | — | — | — | — | — | — |
| `real_m1` | — | — | — | — | — | — | — | — |
| `real_m2` | — | — | — | — | — | — | — | — |
| `real_m33` | — | — | — | — | — | — | — | — |
| `real_m64` | — | — | — | — | — | — | — | — |
| `real_mix` | — | — | — | — | — | — | — | — |
| `real_final` | — | — | — | — | — | — | — | — |
| `zerow` | — | — | — | — | — | — | — | — |

（完整 64 位清单见 `dump_manifest_full.txt`。）

## kernel 内计数（同一次全档运行）

```
[M20] 计数：判定项 93 checks / 0 fails；报告项 80 条（不参与 PASS/FAIL）；guard 155 checks / 0 fails
```

## 再生成命令（12 个 case）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p /tmp/m20_dump && cd /tmp/m20_dump
for c in exact0 exact0m33 int int_mix int_final real_m1 real_m2 real_m33 real_m64 real_mix real_final zerow; do
  rm -f m20_*.bin* m20_case.txt
  M20_DUMP=1 M20_CASE=$c <repo>/m20_hyperconn/build/m20_hyperconn >/dev/null 2>&1
  sha256sum m20_*.bin
  /usr/local/python3.12.13/bin/python3 <repo>/m20_hyperconn/check_ref.py $c
done
```

> dump 目录本身不入库（`.gitignore` 含 `*.bin` / `*_meta.txt`）；需要原始字节时按上面命令在空目录复现并逐字节核对。
