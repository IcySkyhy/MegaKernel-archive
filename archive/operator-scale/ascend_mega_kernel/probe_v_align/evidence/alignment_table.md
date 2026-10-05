| API（本探针 op） | API 形态 / 底层指令 | 落盘窗口 | **实测最小 UB 偏移对齐** | 判定方式（P2-3 要求逐行可辨） | 触发条件（实测） | 错误码 | 依据日志（evidence/logs/） |
|---|---|---|---|---|---|---|---|
| `ls` | LoadAlign→StoreAlign（vlds/vsts） | 256B（1 个 VL） | **源 32B / 目的 32B** | **完整扫描 {0,1,2,4,8,16,24,32}** | 源: 偏移 16B ⇒ FAULT(507035); 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_ls_a*_b*_n*.log`（逐点，共 16 条） |
| `vld` | LoadAlign(reg,addr,AddrReg)（vld） | 256B | **源 32B** | **完整扫描 {0,1,2,4,8,16,24,32}** | 源: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_vld_a*_b*_n*.log`（逐点，共 8 条） |
| `gather` | Gather（vgather2，UB base + 索引寄存器） | 256B | **源 4B** | **完整扫描 {0,1,2,4,8,16,24,32}** | 源: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_gather_a*_b*_n*.log`（逐点，共 8 条） |
| `ldbrc` | LoadAlign<…,DIST_BRC_B32>（单元素广播 load） | 256B | **源 4B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_ldbrc_a*_b*_n*.log`（逐点，共 5 条） |
| `stfirst` | StoreAlign<…,DIST_FIRST_ELEMENT_B32>（单元素 store） | 4B | **目的 4B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_stfirst_a*_b*_n*.log`（逐点，共 5 条） |
| `stpack` | Cast<bf16,float>+StoreAlign<…,DIST_PACK_B32>（pack b16） | 128B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_stpack_a*_b*_n*.log`（逐点，共 2 条） |
| `stpack4` | Cast<fp4>+StoreAlign<…,DIST_PACK4_B32> | 64B（128 个 fp4 nibble 打包；P2-4 由实测订正，原写 32B） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_stpack4_a*_b*_n*.log`（逐点，共 2 条） |
| `vldas` | LoadUnAlignPre+LoadUnAlign（vldas/vldus） | 256B | **源 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_vldas_a*_b*_n*.log`（逐点，共 6 条） |
| `vstus` | StoreUnAlign(+Post)（vstus/vstas） | 256B | **目的 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_vstus_a*_b*_n*.log`（逐点，共 6 条） |
| `load1` | Reg::Load（vldas+vldus） | 256B | **源 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_load1_a*_b*_n*.log`（逐点，共 6 条） |
| `store1` | Reg::Store（vstus+vstas） | 256B | **目的 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_store1_a*_b*_n*.log`（逐点，共 6 条） |
| `gatherb` | GatherB（vgatherb，位索引） | 未取到（见备注） | **不作结论**（偏移 0 即故障，非对齐问题；见备注） | 降序推断 {0,16,8,4,2,1}（首个故障停） | 偏移 0 ⇒ FAULT(507035) | 507035 | `run_gatherb_a0_b0_n64.log` |
| `scatter` | Scatter（vscatter） | 256B | **目的 4B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_scatter_a*_b*_n*.log`（逐点，共 5 条） |
| `pld` | LoadAlign(MaskReg,…)/StoreAlign(…,MaskReg)（plds/psts） | 32B | **源 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_pld_a*_b*_n*.log`（逐点，共 2 条） |
| `dupub` | 经典 Duplicate(UB 目的)（507015 历史现场） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_dupub_a*_b*_n*.log`（逐点，共 2 条） |
| `brcb` | 经典 Brcb(UB 目的) | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_brcb_a*_b*_n*.log`（逐点，共 2 条） |
| `dupreg` | Reg::Duplicate（寄存器目的）+ StoreAlign | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_dupreg_a*_b*_n*.log`（逐点，共 2 条） |
| `add` | Add（vadd） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_add_a*_b*_n*.log`（逐点，共 2 条） |
| `mul` | Mul（vmul） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_mul_a*_b*_n*.log`（逐点，共 2 条） |
| `exp` | Exp（vexp） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_exp_a*_b*_n*.log`（逐点，共 2 条） |
| `redsum` | Reduce<SUM>（vcadd） | 4B（lane0） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_redsum_a*_b*_n*.log`（逐点，共 2 条） |
| `redmax` | Reduce<MAX>（vcmax） | 4B（lane0） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_redmax_a*_b*_n*.log`（逐点，共 2 条） |
| `cmpsel` | Compares+Select（vcmps/vsel） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_cmpsel_a*_b*_n*.log`（逐点，共 2 条） |
| `arange` | Arange<float>（vci） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_arange_a*_b*_n*.log`（逐点，共 2 条） |
| `f32bf16norm` | Cast<bf16,float> + NORM StoreAlign | 256B（值在偶数 16-bit lane） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_f32bf16norm_a*_b*_n*.log`（逐点，共 2 条） |
| `f32bf16nb16` | 同上但落盘用 bf16 ALL 掩码 | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_f32bf16nb16_a*_b*_n*.log`（逐点，共 2 条） |
| `f32bf16pack` | Cast<bf16,float> + DIST_PACK_B32 | 128B（紧凑 64 bf16） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_f32bf16pack_a*_b*_n*.log`（逐点，共 2 条） |
| `bf16f32norm` | LoadAlign<bf16,NORM>+Cast<float> | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_bf16f32norm_a*_b*_n*.log`（逐点，共 2 条） |
| `bf16f32unpk` | LoadAlign<bf16,UNPACK_B16>+Cast<float> | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_bf16f32unpk_a*_b*_n*.log`（逐点，共 2 条） |
| `lmbarin` | LocalMemBar 在 __VEC_SCOPE__ 内（控制组） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_lmbarin_a*_b*_n*.log`（逐点，共 2 条） |
| `lmbarout` | LocalMemBar 在 __VEC_SCOPE__ 外（靶子） | 256B | **不作结论**（偏移 0 即故障，非对齐问题；见备注） | 降序推断 {0,16,8,4,2,1}（首个故障停） | 偏移 0 ⇒ FAULT(507035) | 507035 | `run_lmbarout_a0_b0_n64.log` |
| `widthf32` | Duplicate(7.0f)+StoreAlign，dtype=f32 | 256B = 64 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widthf32_a0_b0_n*.log` |
| `widthbf16` | 同上，dtype=bf16 | 256B = 128 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widthbf16_a0_b0_n*.log` |
| `widths16` | 同上，dtype=int16 | 256B = 128 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widths16_a0_b0_n*.log` |
| `widths8` | 同上，dtype=int8 | 256B = 256 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widths8_a0_b0_n*.log` |
| `aranges32` | Arange<int32> | 256B = 64 lane | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_aranges32_a0_b0_n*.log` |
| `aranges16` | Arange<int16> | 256B = 128 lane | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_aranges16_a0_b0_n*.log` |
| `aranges8` | Arange<int8> | 256B = 256 lane（实测 lane i == i 全 256 个） | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_aranges8_a0_b0_n*.log` |
| `vnoop` | 控制组：V 段不做任何事 | - | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_vnoop_a0_b0_n*.log` |

### 负向对照定点与探测到的无效路径（`nc:` 分组；**不作对齐结论**，只作'对照是否活着'的证据）

| op | 观测 | 结论 | 日志 |
|---|---|---|---|
| `vldasnopre` | a=1 b=0 ⇒ WRONG(0) | 缺 init/post ⇒ **无故障但与逐字节模型不符**（负向对照生效）——有状态协议的 init/post 不是可选的 | `run_vldasnopre_a*_b*_n*.log` |
| `vstusnopost` | a=0 b=1 ⇒ WRONG(0) | 缺 init/post ⇒ **无故障但与逐字节模型不符**（负向对照生效）——有状态协议的 init/post 不是可选的 | `run_vstusnopost_a*_b*_n*.log` |
| `psts` | a=0 b=0 ⇒ OK(0); a=0 b=16 ⇒ FAULT(507035) | 谓词落盘（psts）**限 32B 对齐**（0 通过 / 16 故障），落盘 32B | `run_psts_a*_b*_n*.log` |

### 长度维度（`lsn`：地址固定 32B 对齐，掩码长度 n 变化）

| n（元素） | 偏移 b | 实测写出窗口 | 窗口外被写字节 | 结果 | 日志 |
|---|---|---|---|---|---|
| 1 | 0 | window=4 | outside=0 | OK | `run_lsn_a0_b0_n1.log` |
| 3 | 0 | window=12 | outside=0 | OK | `run_lsn_a0_b0_n3.log` |
| 7 | 0 | window=28 | outside=0 | OK | `run_lsn_a0_b0_n7.log` |
| 8 | 0 | window=32 | outside=0 | OK | `run_lsn_a0_b0_n8.log` |
| 63 | 0 | window=252 | outside=0 | OK | `run_lsn_a0_b0_n63.log` |
| 64 | 0 | window=256 | outside=0 | OK | `run_lsn_a0_b0_n64.log` |
