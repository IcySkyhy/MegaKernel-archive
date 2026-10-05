# `ASCEND_SLOG_PRINT_TO_STDOUT=1` 重跑（人类指令，2026-09-27）—— **未取得读数（设备槽没抢到）**

## 执行的命令（逐字，照人类要求：进锁、单进程、timeout、上界）
```
$ cd m23_gdn_prefill/build
$ timeout 250 flock -w 900 /tmp/npu0.lock bash -c '
    npu-smi info 2>&1 | sed -n "3p"
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    echo "env: ASCEND_SLOG_PRINT_TO_STDOUT=$ASCEND_SLOG_PRINT_TO_STDOUT"
    timeout 45 stdbuf -oL ./m23_gdn_prefill_j12 > /tmp/j12_slog.log 2>&1; echo "rc=$?"
    wc -c /tmp/j12_slog.log'
```

## 结果（**不是**"没复现"，是**没跑**）
* 外层在**工具 250s 上限**被 kill ⇒ **锁没等到**（连续第 3 次：本轮及前两轮的 `j12_onlyst` / 补集三档同样如此）。
* 随后 `/tmp/j12_slog.log` **不存在**、`wc -c` 无输出；拷贝到本目录的归档文件**大小 0**。
* 关键词行数（`aicore error` / `mte error info` / `507015` / `exception` / …）**全部为空** —— 因为**根本没产生输出**。
⇒ **不得**读成"加了 SLOG 后签名变了或不再复现"；**该实验一次都还没执行**。

## 复现命令（下一轮直接跑；**每档一次独立调用**，不要与外层长等待叠加）
```
timeout 900 flock -w 900 /tmp/npu0.lock bash -c 'npu-smi info | sed -n 3p; \
  export ASCEND_SLOG_PRINT_TO_STDOUT=1; \
  timeout 60 stdbuf -oL ./m23_gdn_prefill_j12 > /tmp/j12_slog.log 2>&1; echo rc=$?'
# 归档（3MB 上界）：head -c 3000000 /tmp/j12_slog.log > evidence/logs/j12_slog_r1.log
```
建议把外层 `timeout` 设到 **≥ 900s**（等锁可能很久），或**先只探测锁是否空闲**（`flock -n`）再决定是否进锁。

## 二值问题的回答（人类要的那条）
**答：无法回答** —— "加了 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 之后 stdout 里有没有原先没有的更详细信息"
这个问题**本次没有执行、没有捕获到任何输出**，因此**既不能说"有"也不能说"没有"**。
（`evidence/logs/` 下现存的设备读数里，所有日志都是**未设该环境变量**时产生的。）
