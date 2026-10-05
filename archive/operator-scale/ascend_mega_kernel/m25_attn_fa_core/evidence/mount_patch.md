# M117 / Wave-B3 挂载点补丁文本（可直接贴进 `m15_layer_kernel.h`）

> **本文件不在 M117 的 scope 内**（scope = `m15_layer_loop/m15_attn_fa_core.h`、
> `m15_layer_loop/m15_attn_fa_core_host.h`、`m25_attn_fa_core/**`）⇒ 只给**文本**，不改那两个文件。
> 挂载点形态来自 M110（Wave A）在 `m15_layer_kernel.h` §3d 写下的契约。

## 1. 挂载点位置

`m15_layer_loop/m15_layer_kernel.h` 的 `M15L_PrefillPhaseA<KIND>`：

```cpp
    if (A.pfWired != 0u) {
        // ---- 段体挂载点（B1 / B2+B3）----
        // 段体落地后，这里换成：`M15PF::GdnPrefillPhaseA(A, ...)` / `M15PF::AttnPrefillPhaseA(A, ...)`
        // ...
        return;                        // ← **就是这一支**
    }
```

## 2. hunk（`KIND_ATTN` 的稠密 causal attention core）

```cpp
    if (A.pfWired != 0u) {
        // ---- M117 / Wave-B3：稠密 causal prefill attention core ----
        // 段体签名只吃「指针 + 标量」（Wave B 共同契约第 1 条）；**不吃 LayerArgs**。
        if constexpr (KIND == KIND_ATTN) {
            M15FAC::FaCoreBody</*Causal=*/true>(
                A.apOut,                       // ① Q 平面 = M88 prolog 的 out 平面（行距 AP_OUT_N）
                M15AP::AP_OUT_N,               // Q 平面行距（元素）= 13952
                A.attnKv,                      // ② 主 KV 平面（**本层**层基址；M15KV_KV_* 编址，只读）
                A.attnOut,                     // 输出平面 bf16 [m][NH*HD = 6144]
                A.attnPScratch,                // P 中转 scratch（nBlk * 32 KB，32B 对齐）
                A.attnWitScratch,              // 跨核记账见证区（(28+2*nBlk)*32*2*4 B；默认惰性）
                /*m       =*/A.m,
                /*posBase =*/A.pfPosBase,      // Q 行 0 的位置（chunked prefill 的起点）
                /*ctx     =*/A.pfCtx,          // KV 长度（可见 token 数）
                /*lane    =*/M15L::AttnSlot(A.layer),   // attention 层序号 k ∈ [0,12)
                /*nBlk    =*/AscendC::GetBlockNum());   // = AIC 数（mix(1,2) 下 AIV 侧同值）
        }
        return;
    }
```

## 3. `LayerArgs` 要加的字段（Wave A / C 加；本 mission 的 scope 不含该文件）

```cpp
    // ---- M117 / Wave-B3：稠密 causal attention core 的输入输出（**只在 KIND_ATTN 档填**）----
    // **5 个 `__gm__` 指针 + 6 个标量**（`FaCoreBody` 是 11 参，`witScratch` 必填）
    __gm__ uint8_t* attnQ;          // Q 平面（= M88 prolog 的 apOut），行距 = M15AP::AP_OUT_N
    __gm__ uint8_t* attnKv;         // 主 KV 平面（本层）基址；`M15KV_KV_BYTE_OFF_CONTIG` 编址
    __gm__ uint8_t* attnOut;        // 输出：bf16 [m][NH*HD]，行距 NH*HD = 6144
    __gm__ uint8_t* attnPScratch;   // P 中转：nBlk * 2 * FAC_P * FAC_SIN * 2 B（= nBlk * 32 KB）
    __gm__ uint8_t* attnWitScratch; // 跨核记账见证区（只被 M25FA_WIT 用；默认惰性、不读不写）：
                                    //   (28 + 2*nBlk) * 32(id) * 2(set/wait) * 4 B；nBlk=28 时 21,504 B
                                    //   ⚠ 本档测试 host 的 kWitBytes 按 56 槽分配（nBlk ≤ 14 够用）——
                                    //     探针构建的已知不足，见 evidence/exclusions.md §8
    uint32_t        qStride;        // = M15AP::AP_OUT_N（Q 平面行距，元素）
    uint32_t        pfPosBase;      // Q 行 0 的位置（decode 档 = ctx-1）
    uint32_t        pfCtx;          // KV 长度（可见 token 数）
    // `pfWired` 已由 M110 提供（默认 0 ⇒ decode 零回归）
```

## 4. 宿主侧需要备好的东西

| 平面 | 大小 | 约束 |
|---|---|---|
| Q 平面（`apOut`） | `ceil(m/64)*64 × 13952 × 2 B` | **必须按 64 行向上圆整分配**（尾 tile 整块读入） |
| 主 KV 平面（`attnKv`） | `BlocksFor(ctx) × 32768 B`（+ 层 stride） | 只读；只能用 `M15KV_KV_*` 算地址 |
| 输出平面（`attnOut`） | `m × 6144 × 2 B` | — |
| P 中转（`attnPScratch`） | `nBlk × 32 KB` | 32B 对齐；设备自写自读，host 不需初始化 |

## 5. 融合期必须同步的两张表

1. **flagId 打平表**：本段在 mode 4 用 6 个 AIV 侧 id（0,1,2,3,5,6）+ AIC 侧 `+16` 通道
   ⇒ 物理点 12 个。并入 `m15_layer_resources.h` §4 的打平表时，须核对与 M101 的
   `CC_MM0/MM1/CC_P0/P1/P2/CC_BAR/CC_AIVDONE/CC_RDY/CC_ALLDONE` 是否撞号
   （本段与 M101 在 mode 4 都用低编号）。
2. **每 id 的 set 次数**：本段每 id 的 set 次数 = 该核处理到的 tile 数（随 `m` 与负载均衡变化），
   **Wave C 必须按档断言 ≤ 15**（`m15_layer_resources.h` 的相邻性断言族）。

## 6. 默认关闭

按 §M103-2.7 第 3 条，本段的开关沿用 M110 的 `A.pfWired`（`O.pfWired = (H_EnvU32("M15_PF_WIRE",0u)!=0u)?1u:0u;`
一类，缺省 0）⇒ **融合 = 翻开关 + 填清单**，`runs=all` 的 decode 路径在翻开关前后都可复跑。
