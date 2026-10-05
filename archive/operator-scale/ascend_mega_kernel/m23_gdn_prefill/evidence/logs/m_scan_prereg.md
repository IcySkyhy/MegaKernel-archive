# 预登记：M 扫描找"最小复现的 M"（判读规则**先写后跑**）

## 开关（一般化；**默认行为逐字不变**）
`M15GP_ONLY_M=<int>`：只跑**一个** case，`H=48`、`T=该 M`、`salt=0x2304`；日志里打印见证行
`[M23] M15GP_ONLY_M=<m> ⇒ 单 case 运行（H=48, T=<m>）`。
不设它时：若设了 `M15GP_ONLY_P4097=1` 仍走旧行为（case 表第 2 项），否则跑两个 case（**与改动前一致**）。

## 扫描策略：先倍增、再二分
倍增序列：`1,2,4,8,16,32,64,65,128,256,512,1024,2048,4096,4097`
（含 `64` 与 `65`，因为 `chunk=64`、`4097 = 64×64+1`；若临界在 64/65 之间 ⇒ 指向**跨 chunk**状态/令牌复用）。
再在"最后一个 OK"与"第一个 FAIL"之间**二分**到最小 FAIL。

## 判据（先定）
* **FAIL** = `rc != 0` **且** stdout 含 `507015` **且** `mte error info` 计数 > 0；
* **OK** = `rc == 0` 且 stdout 含 `ALL RUNS OK`（无 `507015`）；
* 两者都不是 ⇒ 记 **OTHER** 并如实列出（不塞进上面两类）。

## 命令模板（逐字；**每条 M 一次独立进锁**，`-w ≤300`、`timeout`、进锁前 `npu-smi` 复查）
```
cd m23_gdn_prefill/build
for m in <M列表>; do
  M15GP_ONLY_M=$m timeout 30 flock -w 300 /tmp/npu0.lock stdbuf -oL ./m23_gdn_prefill \
      > ../evidence/logs/mscan_m$m.log 2>&1; rc=$?
  echo "M=$m rc=$rc  mte=$(grep -o 'mte error info: 0x[0-9a-f]*' ../evidence/logs/mscan_m$m.log | head -1) \
        ECC=$(grep -c 'multi-bit ECC' ../evidence/logs/mscan_m$m.log) \
        trap=$(grep -c 'timeout or trap' ../evidence/logs/mscan_m$m.log)"
done
```
## 必答问题（读出后逐条回答）
最小的 FAIL M / 最大的 OK M；临界是否落在 chunk 边界；失败时 `mte error info` 是否与 4097 档逐位相同；
`M=1` 与最小 FAIL M 之间是否有**非单调**（如实列，不抹平）；最小 FAIL 那条**跑两遍**确认确定性。

## 限度
* harness 定尺：`Tp = m + 64`（每 head 尾部补齐）、`tp = align8(m)`、`salt` 固定 ⇒ 任何 `m ≥ 1` 都可表达，**无下限/对齐限制**；
* 每次 launch **两个 case 之一**或**单 case**；不改被测段体、不动 L1 布局/资源记账。
