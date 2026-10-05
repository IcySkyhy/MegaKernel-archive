# 设备档未返回记录（r1）

设备槽纪律：`flock -w 900 /tmp/npu0.lock <命令>`（`-w` 在锁文件之前），进锁后 `npu-smi info` 复查。

## C1（2026-09-27，本 mission）
```
$ cd m23_gdn_prefill/build && timeout 280 flock -w 900 /tmp/npu0.lock bash -c \
    'npu-smi info 2>&1 | head -6; echo "=== RUN ==="; ./m23_gdn_prefill 2>&1 | tail -20'
=> 进程在 280s 上限被 kill（exit 124）；无任何输出可见。
   限度：输出经管道 + 缺 stdbuf ⇒ 缓冲未落盘，**不能**据此判断是否已在第 1 档（m=1）打印过。
```

## C2（同日，加 `stdbuf -oL` 与 120s 内层上限）
```
$ timeout 200 flock -w 600 /tmp/npu0.lock bash -c \
    'npu-smi info 2>&1 | sed -n "1,4p;9,12p"; timeout 120 stdbuf -oL ./m23_gdn_prefill > /tmp/m23_run.log 2>&1; \
     echo "run rc=$?"; cat /tmp/m23_run.log'
=> 整条流水线在 200s 工具上限被 kill，`tail` 未吐任何内容 ⇒ 仍未取得读数。
```

## 观测与判断
* 两次都**未返回**（不是"返回 FAIL"）。⇒ 现状是**未验证**，不能写成"不通过"或"通过"。
* 首嫌疑排序（**未取证，不得当结论**）：
  1. **核间 mode 2 flag 的配对**：本段每个 job 用「2 个 AIV set 同一 id → AIC wait 一次」+
     「AIC set 一次 → 2 个 AIV wait」。m15 的 decode 段是同形态（可用），但本段的 set 点在
     `PIPE_V`（J1 前放行）与 `PIPE_MTE3`（上传后）两处，且**每 chunk 3 个 job 连续 6 次握手**；
  2. **BufferID 死锁**：AIV 的 `GP_BUF_UP` 令牌在「V 写者 / MTE2 写者 / MTE3 读者」三方轮转，
     若某条路径漏配 acquire/release 即成死锁（docs/05 §6.1 规则 ⓔ 的已知坑）；
  3. 段体逻辑错（例如 J3 的 `L1J3` 偏移、`DoMmad` 的 L1 stride）。
* **下一步定位入口**（未做）：把三段 job 收成一段（只留 J1）先跑通、逐段加回；
  并给 AIC/AIV 各加一条 `printf` 计数落 GM 的旁路（走 DMA，不用标量直写）。
