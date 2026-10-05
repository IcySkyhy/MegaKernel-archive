# 收窄读数 r1：**不是同步死锁，是 AIV 设备异常**（塔裁 ④ 的收窄第一步）

## 命令与读数

```
$ timeout 240 flock -w 900 /tmp/npu0.lock bash -c 'npu-smi info | sed -n 3p; \
    timeout 55 stdbuf -oL ./m23_gdn_prefill_j1 > ../evidence/logs/j1_only_r1.log 2>&1; echo rc=$?'
[J1-only = 只跑 J1 的编译期收窄档，M15GP_J1_ONLY=1，见 m23_gdn_prefill/CMakeLists.txt]
[npu-smi 复查：NPU 0 可用，单进程]
J1-only rc=0
[M23] mix(1,2) launch: nblk(AIC)=28 → AIV 线程数=56；slot=753664 B；scratch=21102592 B
[M23] m1       H=48 T=1    blk=28 : 0.059 ms
[M23] p4097    H=48 T=4097 blk=28 : 1.930 ms
[M23] ===== ALL RUNS OK =====
```
⇒ **只跑 J1 的档在真实 shape（m=4097 与 m=1）下返回 OK**，且这一档已经包含：
每 head **195 次** × 48 head 的 `GO1/DONE1` 核间握手（mode 2）、`LoadState`/`StoreState`（h0 转置路径）、
四个 MTE2 组、`TransposeVF`、`CumSumExpVF`、`GammaStrictVF`、两个 `TrilSolveVF`、KT' 上传。

## 全档（J1+J2+J3）的读数：**设备异常，不是挂死**

同一次锁内随后跑全档（`m23_gdn_prefill`），设备返回的不是超时而是**异常报告**（逐字，摘录）：

```
The error from device(chipId:0, dieId:0), serial number is 1199, there is an aivec error exception,
core id is 57, error code = 0, dump info: pc start: 0x120041000d00, current: 0x120041001850,
mte error info: 0x850a500000020043, vec error info: 0x410062180031126e, ...
An error occurred in the kernel task, retCode=0x26, [aicore exception].
fault kernel_name=_Z22m23_gdn_prefill_kernelPhS_S_S_S_S_S_S_S_jjjf
```

**限度（如实标注）**：上面那段异常文本来自同一次锁内**全档的控制台输出**（该次运行的 stdout 未另存为文件，
`j1_only_r1.log` 只含 J1-only 档的输出）；它是**逐字**摘录，但没有对应的归档文件。
下一次跑全档时会把它落到 `evidence/logs/full_r1.log`。

## 判读（**这是本轮最有价值的一条**）
1. **不是同步死锁**：J1-only 档把**同一套核间握手协议**在满量级（65 chunk × 2 head × 28 AICore、
   每 head 195 次握手）跑通并返回 ⇒ mode 2 配对、`PIPE_V`/`PIPE_FIX` set、GM 中介、
   核内 BufferID 的**公共脚手架**都被这一档"洗清"。这与 `cc_probe_r1.md` 独立排除三个核间假设**互相印证**。
2. **异常在 AIV（aivec）+ MTE**：`mte error info` 与 `vec error info` 都非零。⇒ 故障点在
   **J2/J3 专属代码路径**（J1-only 未覆盖的部分）：`SubTransposeVF` / `MulRows1VF` / `ConstScale1VF` /
   `AddRows2VF` / `ScaleAddRowsBrc2VF` 这 5 个 J2/J3 新原语，WT/ABD/SDT 的 MTE2 装载参数，
   以及 ST/W/DT/ABM 的上传握手。**未取证是哪一处**（不写绝对断言）。
3. **这改变了解 block 的优先级**：塔提示的"核内 BufferID 成对 acquire"（M111 同族坑）可以被降级 ——
   同一套令牌纪律在 J1-only 档下满量级跑通；首要嫌疑转为 **J2/J3 的新 VF 原语或装载参数**。
4. 附带说明：本 mission 早先两次"不返回"的记录（`run_hang_r1.md`）与本轮的异常报告同源
   —— 设备异常走 `rtStreamSynchronize` 时表现为长等待，故当时只看到"未返回"。

## 下一步（未做）
按塔裁 ④ 继续收窄：**J1+J2-only → J1+J2+J3-only**，每档留读数；若某档仍出 aivec，用
"逐原语注释掉"的方式定位到具体原语。定位入口 = 本文件 + `j1_only_r1.log`。
