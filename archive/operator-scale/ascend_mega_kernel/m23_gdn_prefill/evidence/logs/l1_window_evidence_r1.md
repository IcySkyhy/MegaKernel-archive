# L1 的 A1/B1 窗口语义取证（塔裁 ②：先取证再动手，**取证前不对外广播**）

**问题**：L1 是不是**两个各 256KB 的有语义约束窗口（A1/B1）**？混用声明是否合法？跨窗声明会不会触发
`l1 error`？（= 本段 `Job3` 把 `TPosition::A1` 的张量声明在偏移 360448/376832 是否非法。）

## 拿到的权威依据（逐字）

**来源**：`/workspace/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/L1_L0A_B_memory_structure_intro.md`
（本机 CANN 9.1.0 随附的 AscendC API 文档；`/workspace/asc-devkit` 的 HEAD = `648a6018`）

> L1 Buffer的总容量为512K字节，由16个Bank组成。每个Bank的容量为32K字节，由1024行组成，每行的宽度为32字节。
> **同一Bank最多同时允许一读或一写**。
> 这16个Bank被进一步组织为8个Bank Group。每个Bank Group包含2个Bank（Bank0和Bank1）。
> **同属于一个Bank Group的两个Bank支持一个读、另一个写，但不支持同时读或同时写**。
>
> ```cpp
> L1_ADDR[18:0] = {BANK[0:0], BANK_DEPTH[9:0], BG[2:0], BANK_WIDTH[4:0]}
> // BANK：地址所在 Bank 的编号，占 1 位，取值 [0,1]
> // BANK_DEPTH：Bank 中的行数，占 10 位，取值 [0,1023]
> // BG：Bank Group 编号，占 3 位，取值 [0,7]
> // BANK_WIDTH：Bank 一行上的偏移，占 5 位，取值 [0,31]
> ```

**它说明的是**：L1 是**一根扁平的 512KB 空间**，按 `(BG[2:0], BANK[0:0], BANK_DEPTH[9:0], BANK_WIDTH[4:0])`
组成 19 位地址；**文档没有把 A1 / B1 描述成两个地址窗**。同一 Bank 一读一写、同 BankGroup 内"一读一写可，
同读/同写不可"是**带宽/冲突**约束，不是"声明越窗即非法"。

## 没拿到的（**如实列为未确定**）
* **CANN 9.1.0 / bisheng 头文件里没有找到 A1/B1 的基址常量**（已查：`x86_64-linux/include/` 全目录下
  `A1_OFFSET`/`B1_OFFSET`/`POSITION_A1`/`TSCM_A1`/`A1_BASE`/`B1_BASE` **0 命中**；
  `bisheng_compiler/lib/clang/15.0.5/include` 下按 `TPosition` 文件名/枚举 **未定位到定义处**）。
  ⇒ 我不能据此断言"A1/B1 各 256KB 且有硬约束"。
* 本仓的用法只是**约定**：`m11_bf16_gemm.asc:108-116`、`m15_gdn_layer.h:1427-1430`、`m15_hc_layer.h:251-254`
  都是"A 区放 [0,256KB)、B 区放 [256KB,512KB)"。**"m11 这么放"是间接线索，不是约束依据**（按塔裁 ② 的口径）。

## 结论（按塔裁 ② 的写法要求）
* **未确定**：现有依据**不足以**判定"L1 的 A1/B1 各 256KB、跨窗声明会触发 `l1 error`"。
  本段 `Job3` 的 A1 声明（偏移 360448 / 376832）**尚未被证明非法**，但也**未被证明合法**。
* 因此**先不改 L1 布局**（塔裁 ②：取证前不动手、不广播）。

## 需要什么实验（最小、可复现、带正/负对照）
一个 AIC 探针（自有 target，照 `probe_*` 形态）：
1. **正对照**：`LocalTensor<float>(TPosition::A1, off=0, 8192)` → Nd2Nz 写 → LoadData → Mmad → Fixpipe，
   逐元素对 host 参考（应正确）；
2. **靶子**：同代码，只把 `off` 改成 `> 256KB`（如 360448）——**若报 `l1 error` 或数值错 ⇒ 窗口语义成立**；
   **若完全正确 ⇒ 扁平 512KB 成立，本段 Job3 的声明无需改**；
3. 再补一档：A1 与 B1 声明**交错落在同一物理区间**（如都在 256KB 附近），看是否出现与本次
   `mte error info: 0x13d1…0202ce` 同签名的错。
判据：与本次故障的 `mte error info` / `l1 error info` **逐位比对**（同签名 = 同一类形态）。
本 mission **未做该实验**（不在剩余预算内），如实标注。
