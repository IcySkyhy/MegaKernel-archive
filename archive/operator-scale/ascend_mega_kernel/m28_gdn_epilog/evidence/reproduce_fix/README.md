# M148 复审修复证据：`reproduce.sh` 输出目录未创建（p2-1，r1 后）

复审 r1 发现：`m28_gdn_epilog/reproduce.sh` 在 `OUT` 不存在时直接往 `$OUT/cmake.log`
写入 ⇒ README 的一命令复现（`... reproduce.sh /tmp/m28_out`）在新机器/新目录上第一步即失败。

## 修复

`reproduce.sh` 在取到 `OUT` 之后、第一次写日志之前加：

```bash
mkdir -p "$OUT" || { echo "[FAIL] 无法创建输出目录 $OUT"; exit 1; }
```

（同族参照 `m27_hc_prefill/reproduce.sh:33` 的 `mkdir -p "$OUT"`。）
**只改 shell 脚本；`m28_gdn_epilog.asc` 内核与已提交设备读数/哈希一字未动。**

## 实测两种场景

| 场景 | 前置 | 命令 | 结果 |
| ---- | ---- | ---- | ---- |
| A：目标目录不存在（含父目录，嵌套） | `test -e /tmp/m28_fix_scenA/nested` = **NO** | `M28_CASES=4,4097 M28_MUT_CASES=4,4097 bash m28_gdn_epilog/reproduce.sh /tmp/m28_fix_scenA/nested` | **rc=0**（见 `scenario_new_dir.log`） |
| B：目标目录已存在（预先 `mkdir -p` 的空目录） | `test -d /tmp/m28_fix_scenB` = **YES** | `M28_CASES=4,4097 M28_MUT_CASES=4,4097 bash m28_gdn_epilog/reproduce.sh /tmp/m28_fix_scenB` | **rc=0**（见 `scenario_existing_dir.log`） |

两场景的判据读数（与已提交档一致）：正向 m=4/4097 设备 vs 内建 C 参考 `0 bad`、
numpy 判据 `o/z bit-exact=True`；负向 `M28_MUTANT=headstride` 两档 o 判红
（m=4: 24064 不同；m=4097: 24647552 不同）、z 仍逐位相等。设备档各自进一次
`flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`、`timeout 180` 在锁内；本轮锁未出现等不到。

说明：两个 `scenario_*.log` 里的构建绝对路径已把工作树前缀重写为 `<repo>`（纯路径脱敏；
rc / PASS / FAIL / 计数等读数逐字未动），以免新增文件里出现协议目录名。既提交的
`../cmake.log`、`../reproduce_output.log` 按「不得动既提交读数/哈希」原样保留。

## 修复前失败形态（对照，`prefix_failure_demo.log`）

```text
$ echo x > "$OUT/cmake.log"          # OUT=/tmp/m28_fix_precheck_absent（不存在）
bash: line 1: /tmp/m28_fix_precheck_absent/cmake.log: No such file or directory
pre-fix rc=1
```

即复审复现的 `REPRO_RC=1` 的机制。

## 新 `reproduce.sh` 哈希

见 `reproduce.sh.sha256`（本轮修复后的脚本哈希）。`../sources_sha256.txt` 是**采集时刻**
（修复前）的源文件哈希记录，按复审「不得动既提交读数/哈希」的要求**未回改**；两处并列即
「修复前 / 修复后」两个时点的 `reproduce.sh` 哈希。

## 关于复审的第二条（并发进程描述）

我在 review-request 里写过「设备跑时 NPU0 上有别家 `m11_bf16_gemm` 进程」——那是我在**开跑
前**手动跑 `npu-smi` 看到的一次观测；本目录 `scenario_*.log` 对应的 **进入锁后的快照**里
没有该进程。也就是说「设备跑时」这个时点我**未记录到并发设备进程**，那条描述不准确，
已在本轮 review-request 里更正。已提交的 `../README.md`（§npu-smi 快照说明）本就写的是
「进锁时 NPU 0 无其它设备进程（各档 Process id 表为空）」，与快照一致，无需改动。
