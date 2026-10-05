# 收窄 r3：**J2 的 MTE2 Nd2Nz 跨 chunk 重复 ⇒ 触发 507015**（塔裁 ② 的 chunk 计数实验）

## 实验
在 J1+J2 档上加编译期开关 `M15GP_J2_NZ_SKIP_FROM=1`：**`Job2` 里的三条 MTE2 Nd2Nz（q 重装 / ST / W）
从第 1 个 chunk 编号起整块跳过**（第 0 个 chunk 照跑）。变量只有"是否重复执行这段 MTE2"。

```
$ flock -w 900 /tmp/npu0.lock bash -c 'npu-smi 复查; timeout 45 stdbuf -oL ./m23_gdn_prefill_j12nz ...'
J12NZ rc=0
[M23] m1    H=48 T=1    blk=28 : 0.067 ms  (dump ok)
[M23] p4097 H=48 T=4097 blk=28 : 2.360 ms  (dump ok)
```

## 判读
| 档 | m=1 | m=4097 |
|---|---|---|
| 只 J1 | OK | OK |
| J1+J2（Job2 的 Nd2Nz 每 chunk 都跑） | OK | **507015 aicore error** |
| J1+J2（**Job2 的 Nd2Nz 从第 1 chunk 起跳过**） | OK | **OK**（2.360 ms） |

⇒ 故障由 **`Job2` 里"每 chunk 重复的 MTE2→L1 写入"**触发（第 0 个 chunk 跑不出错，从第 1 个 chunk 起出错
——因为该档第一个 chunk 仍执行了这段 Nd2Nz）。**与"L1 令牌的跨次复用"这一形态吻合**（塔裁 ② 的猜测方向），
且 **J1 自己的 Nd2Nz（每 chunk 也重复、写同一批 L1 区）在 65 chunk 下没问题** ⇒ 不是"MTE2 重复写 L1"这个笼统形态，
而是 **J2 这一处**（三条里至少一条）与 L1 的生命周期/载体有关。

## 下一步（同法，机械可执行，未做）
把跳过改成**按条**（编译期枚举 which ∈ {q 重装, ST, W}），一次只跳一条，三档各留读数：
* 若**只跳 ST** 就恢复 OK ⇒ 嫌疑落在 **[128,128] 的那次 L1 写入**（`L1St(h)`，65536B，dstNzC0Stride=128）
  —— 与"L1 的 bank/窗口语义"高度相关，塔裁 ② 排在其后的 L1 窗口探针就要**优先做 ST 那档**；
* 若**只跳 W** 或**只跳 q** 恢复 OK ⇒ 嫌疑在 `L1W(h)` / `L1Kq(h)+32KB` 这两处；
* 若三条各跳一条都仍复现 ⇒ 说明是"**这一处 MTE2 与紧邻的 MTE1/FIXP 之间的令牌次序**"，而不是具体地址。
