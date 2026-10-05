# 新强制规则自查：「矩阵乘法 ⇒ cube；其余全部 VF；scalar 只做控制流」

**对象**：`m15_layer_loop/m15_gdn_prefill.h` 的**设备段**（host 侧 `m23_gdn_prefill.asc` 的
`Hash3`/`HashU11`/`GenQk`/累加器只用于生成输入与判据，不在规则射程内）。

## 逐项自查（命令 + 读数）
| 检查 | 命令 | 读数 | 结论 |
|---|---|---|---|
| 有没有用 scalar 把**数据**写进 UB | `grep -c 'SetValue\|GetValue' m15_gdn_prefill.h` | **0** | 合规（无标量直写 UB 的计算落盘） |
| `scale` 是否被 scalar 算进数据 | `grep -n 'scale_' …` 全部 4 处：`768` 取参、`954` `VecScaleConstVF(scSE, scEG, scale_)`、`1038` `ConstScale1VF(aUb, …, scale_, …)`、`1110` 成员初值 | 两处都是**VF 的 `Muls` 标量操作数**（乘法在向量单元里执行）；另两处是取参与初值 | 合规（**数据乘法在 VF 内**，scalar 只传操作数） |
| 其余标量是否都是控制流 | `cv` / `t0` / `hk` / `kqOff` / `ch` / `gLen` / `(cv-1)`（地址）| 全是**下标/边界/地址/条件** | 合规（规则界线的合法侧） |
| matmul 是否都走 cube | 本段 6 个收缩全在 `GdnPrefillAic` 的 `Job1/2/3` 里走 `Mmad`；AIV 侧无收缩 | — | 合规 |

## 结论（**无需改动**）
本 mission 的设备段**没有"顺手用标量算一下"的地方** ⇒ 按新规则**未改动任何计算路径**。
如实说明：**不是"我改完了"，而是"自查未发现违规"**（两条 grep 的原始读数已列在上表）。

## 限度
* 只查了本 mission 的 3 个文件（`m15_gdn_prefill.h` / `m15_gdn_resources.h` / `m15_gdn_prefill_host.h`）
  与 `m23_gdn_prefill/` 的验证工程；**其余段体不在本 mission scope**。
* 判据是"数据 vs 控制流"的**人工分类**（`scale_` 属数据、但它的乘法落在 VF 的 `Muls` 操作数位置），
  未做工具化扫描（塔已另派通读扫描出清单）。
