#!/usr/bin/env python3
"""MoE 段 router 的静态判据：M91-#3（权重流式）+ **M95-#4（top-k 归并树）**。

本文件由 M91 建立，判据分两栏；M95 追加了 #4 的三条（A10/A11/A12）并**反转 A10 的语义**
（M91 时它钉「#4 未落地」这一事实；现在钉「已落地」——复审判据「合法落地会报 FAIL」的预告
至此兑现），同时保留 #3 的全部判据与两档布局算术。

改的是什么：
  · **#3（M91）**：`m15_moe_layer.h` 的 S2 `RouterStage` 从「`[NUM_EXPERTS+1][HIDDEN]` 算子全量
    预转 fp32 常驻 UB」改成 **m17_moe_real 的流式形态**（x 行块常驻 + 权重行 bf16 ping/pong +
    `RT_EGRP` 个专家一组的 fp32 预转窗 + 共享门权重单独 1 行常驻）。形态抄自 m17 的 `RouterStage`，
    代码改抄自 m15 自己的 `PrecastWeights` / `GemvRow`（算术一字未改）。
  · **#4（M95）**：top-k 从**单块 `Sort32`（正确性上界 E ≤ 32）**改成 **16 块 `Sort32` + 4 级
    二路 `MrgSort` 归并树 → Extract 前 64 对 → 前 TOPK 对 renorm**（形状抄 m17 的
    `MergeTree`/`Merge2`），并把**对数行**行距 64 → `RT_ROWL`（= 512 lane）一起重排；top-k
    **结果**的 IDS/WS staging 行距固定 `RT_WROW`（= 64 lane，只放 ≤ 64 个结果，与 E 无关）。
    E=512 的数值锚点不在本文件，在 `m95_e512/`（独立锤点，对拍 `tools/golden`）。

判据分两栏：
  A. **形态**：流式函数到位、旧形态（`PrecastWeights` / `RT_NROWS`）不留、分组循环与写盘掩码
     的形状正确（**E=4 下不会被“一组 8 个专家”写成越界 lane** —— 这正是 m17 那份原样搬到
     E=4 会坏的地方，见 m91_README.md）；#4 的归并树形态（块数实参、MergeTree/Merge2、无经典
     Extract / MrgSort4）、行距重排、索引模板宽度与 Extract 重复数。
  B. **布局算术**：把 `m15_moe_resources.h` / `m15_moe_layer.h` / `m15_gdn_resources.h` 的
     `constexpr uint32_t` 就地求值，在当前档（E=4）与 E=512 假设档各算一遍，核对：
       · **每一个 `UB_RT_*` 都与 NUM_EXPERTS 无关**（两档读数相同）——
         #3 买的是「权重窗与 E 解耦」，#4 买的是「行宽覆盖 E ≤ 512 且**行宽读数也不随档漂移**」
         （固定 16 块 ⇒ `RT_ROWL`/`RT_SORTLN` 是常量）；
       · `UB_RT_END <= UB_RC_END`（GDN 相位峰值）⇒ 融合 UB 峰值不被抬高（两档都查）；
       · `RT_EGRP` 个窗口槽的读点各 1 处、ping/pong 的 Acquire/Release 配对；
       · IG 区（S3 复用 router x 块区）在两档都放得进 `RT_RB` 行的窗；
       · **行宽覆盖**：两档都必须 `RT_ROWL == RT_SORTLN == RT_SORT_NBLK × 32` 且 `≥ NUM_EXPERTS`
         （否则 top-k 候选集被截断 = 结果错，这正是 #4 要拆的墙）。

用法：
    python3 check_stream.py                 # 当前档 + E=512 假设档
    python3 check_stream.py --header /tmp/副本.h    # 变异副本（负向对照）
rc：全部 PASS ⇒ 0；任一 FAIL ⇒ 1。
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
M15 = HERE.parent
sys.path.insert(0, str(HERE))

from check_prep import Report, collect, evaluate, strip_comments   # noqa: E402

STREAM_FNS = ["LoadWRow", "PrecastWRow", "PrecastSgateW", "PadLogitsRow", "SgateRows", "GemvGroupRow"]


def _defs(code: str) -> dict:
    return {fn: len(re.findall(r"__aicore__ inline void " + fn + r"\(", code)) for fn in STREAM_FNS}


def shape_checks(rep: Report, header: str, res_text: str, res_big: str):
    # 判据一律在**剥注释后的代码视图**上判（`check_prep.strip_comments`）。两个理由：
    #   ① 生成物的头部注释会**合法地提到**旧形态（讲第 8/9 类替换时点名 `PrecastWeights` /
    #      `RT_NROWS`）⇒ 按原文判会把说明误判成残留；
    #   ② 反过来，按原文判会让**注释把坏代码「打回绿」**——把实现弄坏、再用注释补上被匹配的文本，
    #      正向计数就恢复了（r1 复审实测出这个洞，r2 修：本文件所有 A 组判据都改判代码视图）。
    # `strip_comments` 本身也修了顺序（先 `//` 后 `/*...*/`）：否则生成物里行注释中的 `reg_compute/**`
    # 会把「代码视图」从那一行一路吞掉（追加一条无害块注释就能复现，见 m95_negctl.sh 的 nc10_benign）。
    # 常备对照：`moe_relift/m95_negctl.sh`（**13 条 = 12 条破坏 + 1 条健全性正对照**；脚本结尾自报这个数，
    #   复现：`grep -c '^run ' m95_negctl.sh`、`bash m95_negctl.sh | tail -2`）。
    #   注释绕过有两个方向、都必须红：正向 nc9a/nc9b/nc9c（注释补回被匹配文本）、
    #   反向 nc11_survivor/nc11b_survivor（块注释的 `*/` 与 `//` 同行 ⇒ 两趟扫描器会让注释文本存活）；
    #   nc10_benign 是健全性正对照（无害块注释 ⇒ 读数不许变）。
    code = strip_comments(header)
    rcode = strip_comments(res_text)
    d = _defs(code)
    rep.check("A1 流式权重路径的 6 个函数各定义 1 处",
              all(v == 1 for v in d.values()) and len(d) == 6, f"定义处数 = {d}")
    gone = [s for s in ("PrecastWeights", "RT_NROWS", "UB_RT_WF + i * HIDDEN * 4") if s in code]
    rep.check("A2 旧形态（PrecastWeights / RT_NROWS / 索引槽位的全量窗）已不适用的目标串（只判代码、剥注释）",
              not gone, f"仍出现的串 = {gone if gone else '（三者均未出现在代码里）'}")

    # 分组循环：步长 RT_EGRP、上界 NUM_EXPERTS；写盘掩码 ng = min(RT_EGRP, NUM_EXPERTS - e0)
    grp = re.search(r"for \(uint32_t e0 = 0; e0 < NUM_EXPERTS; e0 \+= RT_EGRP\) \{", code)
    rep.check("A3 分组循环 = `for e0 < NUM_EXPERTS step RT_EGRP`（E < RT_EGRP 时只跑一组）",
              bool(grp), f"分组循环 = {'命中' if grp else '未找到'}")
    mask = re.search(
        r"uint32_t ng = \(NUM_EXPERTS - e0 < RT_EGRP\) \? \(NUM_EXPERTS - e0\) : RT_EGRP;", code)
    rep.check("A4 写盘掩码 ng = min(RT_EGRP, NUM_EXPERTS - e0)（不写越界 lane）",
              bool(mask), f"掩码表达式 = {'命中' if mask else '未找到'}")
    # 槽内循环跑满 RT_EGRP 个槽（0..RT_EGRP-1），每个槽的尾部预取各 1 处
    slot_loop = re.search(
        r"for \(uint32_t j = 0; j < RT_EGRP; \+\+j\) \{\s*\n\s*PrecastWRow\(j & 1, j\);\s*\n\s*"
        r"LoadWRow\(e0 \+ j \+ 2, j & 1\);", code)
    rep.check("A5 槽循环 = 0..RT_EGRP-1，逐槽「预转 + 预取下一槽」",
              bool(slot_loop), f"槽循环体 = {'命中' if slot_loop else '未找到'}")

    # 窗口读点：RT_EGRP 个（0..RT_EGRP-1）各 1 处
    reads = [len(re.findall(r"UB_RT_WF \+ " + str(k) + r" \* HIDDEN \* 4\)", code))
             for k in range(8)]
    rep.check("A6 权重流式窗的 8 个槽各读 1 处（RT_EGRP = 8）",
              reads == [1] * 8, f"槽 0..7 的读点数 = {reads}")

    # ping/pong token 的 Acquire/Release 配对（MTE2 与 V 各一对，按槽选择）
    acq = len(re.findall(r"BufAcquire<PIPE_MTE2>\(bufSel \? BUF_AIV_WST1 : BUF_AIV_WST0\)", code))
    rel = len(re.findall(r"BufRelease<PIPE_MTE2>\(bufSel \? BUF_AIV_WST1 : BUF_AIV_WST0\)", code))
    vacq = len(re.findall(r"BufAcquire<PIPE_V>\(bufSel \? BUF_AIV_WST1 : BUF_AIV_WST0\)", code))
    vrel = len(re.findall(r"BufRelease<PIPE_V>\(bufSel \? BUF_AIV_WST1 : BUF_AIV_WST0\)", code))
    rep.check("A7 ping/pong token 的 Acquire/Release 配对（MTE2 写与 V 读各 1 对）",
              (acq, rel, vacq, vrel) == (1, 1, 1, 1), f"MTE2 {acq}/{rel}；V {vacq}/{vrel}")
    # 越界槽用 `e % NUM_EXPERTS` 的真实行填充（保持配对 + 不读越权重区）
    mod = code.count("(e < NUM_EXPERTS) ? e : (e % NUM_EXPERTS)")
    rep.check("A8 越界槽用真实行填充（Acquire/Release 必须配对、不读越权重区）",
              mod == 1, f"`(e < NUM_EXPERTS) ? e : (e % NUM_EXPERTS)` = {mod} 处")
    # 两档资源表的 §4 读数：UB_RT_* 与 E 无关
    rep.check("A9 资源表的 MoE AIV token 编号落在已登记的 15..23 内（新增 22/23）",
              "BUF_AIV_WST0 = 22" in rcode and "BUF_AIV_WST1 = 23" in rcode,
              f"WST0/WST1 = {22 if 'BUF_AIV_WST0 = 22' in rcode else '?'}/"
              f"{23 if 'BUF_AIV_WST1 = 23' in rcode else '?'}")
    # ---- M95-#4：top-k 从单块 Sort32 改成 16 块 Sort32 + 4 级二路 MrgSort 归并树 ----
    # A10 的语义在 M95 落地后**反转**（M91 时它钉的是「#4 未落地」这一事实；现在钉「已落地」）。
    n_sort = len(re.findall(r"\bSort32\(", code))          # 只判代码：注释里会**合法地**引用形态
    n_mrg = len(re.findall(r"\bMrgSort\(", code))
    n_sort_nblk = len(re.findall(r"Sort32\(pairT, valT, idxT, RT_SORT_NBLK\);", code))
    n_mt = len(re.findall(r"__aicore__ inline void MergeTree\(\)", code))
    n_m2 = len(re.findall(r"__aicore__ inline void Merge2\(", code))
    n_mt_call = len(re.findall(r"(?m)^\s*MergeTree\(\);", code))
    n_m2_tot = len(re.findall(r"\bMerge2\(", code))       # 1 处定义 + 4 处调用点（各在 8/4/2/1 级循环里）
    n_classic_extract = len(re.findall(r"(?<![A-Za-z_])Extract\s*\(", code))
    n_mrg4 = len(re.findall(r"\bMrgSort4\s*\(", code))
    rep.check("A10 top-k 已落地 #4 归并树（Sort32 块数 = RT_SORT_NBLK 且**被调用**、MergeTree 定义+调用、"
              "Merge2 = 1 定义 + 4 调用点、无经典 Extract、无 MrgSort4）",
              n_sort == 1 and n_mrg == 1 and n_sort_nblk == 1 and n_mt == 1 and n_m2 == 1
              and n_mt_call == 1 and n_m2_tot == 5
              and n_classic_extract == 0 and n_mrg4 == 0,
              f"Sort32 = {n_sort}（块数实参 RT_SORT_NBLK = {n_sort_nblk}）；MrgSort = {n_mrg}；"
              f"MergeTree 定义/调用 = {n_mt}/{n_mt_call}；Merge2 定义 = {n_m2}、`Merge2(` 合计 = {n_m2_tot}"
              f"（应为 5 = 1 定义 + 4 调用点，即 8+4+2+1 级）；经典 Extract = {n_classic_extract}；"
              f"MrgSort4 = {n_mrg4}")
    # A13：4 级结构本身（8 / 4 / 2 / 1）—— 只删某一级或改级数会在此 FAIL
    lvl = [len(re.findall(p, code)) for p in (
        r"for \(uint32_t g = 0; g < 8; \+\+g\) \{",
        r"for \(uint32_t g = 0; g < 4; \+\+g\) \{",
        r"for \(uint32_t g = 0; g < 2; \+\+g\) \{",
        r"Merge2\(bT\[0\], aT\[0\], aT\[128\], 32\);",
    )]
    rep.check("A13 归并树 4 级结构齐备（8 / 4 / 2 / 1 各 1 处）", lvl == [1, 1, 1, 1],
              f"各级命中数（8/4/2/1）= {lvl}")
    # A11：行距重排 —— 对数行一律 RT_ROWL、top-k 结果 staging（IDS/WS）一律 RT_WROW
    n_log64 = len(re.findall(r"UB_RT_LOG \+ r \* 64 \* 4", code))
    n_logrowl = len(re.findall(r"UB_RT_LOG \+ r \* RT_ROWL \* 4", code))
    n_stage64 = len(re.findall(r"UB_RT_(?:IDS|WS) \+ r \* 64 \* 4", code))
    n_stagewrow = len(re.findall(r"UB_RT_(?:IDS|WS) \+ r \* RT_WROW \* 4", code))
    rep.check("A11 行距重排：对数行 4 处 RT_ROWL（0 处写死 64）、top-k staging 4 处 RT_WROW（0 处写死 64）",
              n_log64 == 0 and n_logrowl == 4 and n_stage64 == 0 and n_stagewrow == 4,
              f"UB_RT_LOG + r * 64 * 4 = {n_log64}；RT_ROWL = {n_logrowl}（PadLogitsRow/GemvGroupRow/"
              f"SoftmaxTopkRow/CopyOutBlock）；IDS/WS + r * 64 * 4 = {n_stage64}；"
              f"RT_WROW = {n_stagewrow}（IDS/WS 各 2 处）")
    # A12：索引模板铺满行宽（否则 Sort32 的第 2..16 块索引错）+ Extract 重复数接上 RT_EXTRACT_REP
    n_tmpl = len(re.findall(r"for \(uint16_t c = 0; c < static_cast<uint16_t>\(RT_NCHUNK_W\); \+\+c\) \{\s*\n\s*"
                            r"RegTensor<int32_t> rg;", code))
    n_ext = len(re.findall(r"for \(uint32_t rep = 0; rep < RT_EXTRACT_REP; \+\+rep\)", code))
    rep.check("A12 索引模板按 RT_NCHUNK_W 铺满行宽（1 处）且 Extract 用 RT_EXTRACT_REP 重复（1 处）",
              n_tmpl == 1 and n_ext == 1, f"模板 chunk 循环 = {n_tmpl}；Extract 重复循环 = {n_ext}")
    return rep


def layout_checks(rep: Report, exprs: dict, tag: str, e: int, topk_max: int):
    rep.scales += 1
    return evaluate(exprs, {"NUM_EXPERTS": e, "TOPK_MAX": topk_max,
                            "TOTAL_MAX": evaluate(exprs, {})["M_MAX"] * topk_max})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default=str(M15 / "m15_moe_layer.h"))
    ap.add_argument("--res", default=str(M15 / "m15_moe_resources.h"))
    ap.add_argument("--E", type=int, default=512, help="假设档的 NUM_EXPERTS（默认 512）")
    ap.add_argument("--topk", type=int, default=10, help="假设档的 TOPK_MAX（默认 10）")
    args = ap.parse_args()

    header = pathlib.Path(args.header).read_text()
    res_text = pathlib.Path(args.res).read_text()
    # 符号优先级：**M15M 的两个文件放最后**（同名常量在多个资源表里都有定义时，后收的覆盖前者；
    #   `UB_VEC` 在 m15_gdn_resources.h 里是另一个值 —— 先收它、后收 MoE 两件，才取到 M15M 的那份）
    exprs = collect([str(M15 / "m15_gdn_resources.h"), args.res, args.header])
    rep = Report()
    shape_checks(rep, header, res_text, res_text)

    e_now = evaluate(exprs, {})["NUM_EXPERTS"]
    tk_now = evaluate(exprs, {})["TOPK_MAX"]
    print(f"[check] ---- 布局算术：当前档 NUM_EXPERTS={e_now} / TOPK_MAX={tk_now} ----")
    env_now = layout_checks(rep, exprs, f"E={e_now}", e_now, tk_now)
    env_big = layout_checks(rep, exprs, f"E={args.E}", args.E, args.topk)

    ub_names = sorted(n for n in exprs if n.startswith("UB_RT_"))
    fixed = [n for n in ub_names if env_now[n] == env_big[n]]
    moved = [n for n in ub_names if env_now[n] != env_big[n]]
    rep.check("B1 每一个 UB_RT_* 都与 NUM_EXPERTS 无关（两档读数相同）",
              not moved, f"UB_RT_* 共 {len(ub_names)} 项；随 E 变的 = {moved if moved else '（无一项变化）'}")
    rep.check("B2 UB_RT_END <= GDN 相位峰值 UB_RC_END（不抬高融合 UB 峰值）",
              env_now["UB_RT_END"] <= env_now["UB_RC_END"],
              f"UB_RT_END = {env_now['UB_RT_END']} B ≤ UB_RC_END = {env_now['UB_RC_END']} B"
              f"（余量 {env_now['UB_RC_END'] - env_now['UB_RT_END']} B；"
              f"历史读数：M91 改前 193600 / M91 改后 192544 / M95 改后 222752）")
    rep.check("B3 UB_RT_END <= UB_BYTES_TOTAL（248KB）",
              env_now["UB_RT_END"] <= env_now["UB_BYTES_TOTAL"],
              f"UB_RT_END = {env_now['UB_RT_END']} B ≤ {env_now['UB_BYTES_TOTAL']} B")
    # 权重流式窗的常驻字节数（诊断读数）
    win = env_now["RT_EGRP"] * env_now["HIDDEN"] * 4
    rep.check("B4 权重流式窗的常驻字节数（诊断读数）", True,
              f"RT_EGRP = {env_now['RT_EGRP']} 行 × HIDDEN × 4 B = {win} B"
              f"（改前：全量 (E+1) 行 = {e_now + 1} × {env_now['HIDDEN']} × 4 = "
              f"{(e_now + 1) * env_now['HIDDEN'] * 4} B；E={args.E} 假设档仍为 {win} B）",
              judged=False, site="B 权重窗常驻字节")
    # IG 区（S3 复用 router x 块区）两档都要放得进
    for tag, env in ((f"E={e_now}", env_now), (f"E={args.E}", env_big)):
        need = env["UB_IG_SCAL_END"] - env["UB_RT_XB"]
        winb = env["RT_RB"] * env["HIDDEN"] * 2
        rep.check(f"[{tag}] S3 的 IG 区放得进 router x 块窗",
                  need <= winb, f"IG 需 {need} B ≤ 窗 {winb} B（RT_RB = {env['RT_RB']} 行）")
        # B5（M95-#4）：行宽/排序宽度必须覆盖该档的 NUM_EXPERTS，且形状自洽
        #   —— 这条是 #4 的**覆盖性**判据：把 RT_ROWL 退回 64（或 RT_SORT_NBLK 退成 < ceil(E/32)）
        #   就会在此 FAIL（最小绕过对照见 m95_README.md 的负向对照 nc_row）。
        rep.check(f"[{tag}] top-k 覆盖性：RT_ROWL == RT_SORTLN == RT_SORT_NBLK × 32 ≥ NUM_EXPERTS",
                  env["RT_ROWL"] == env["RT_SORTLN"] == env["RT_SORT_NBLK"] * 32
                  and env["RT_ROWL"] >= env["NUM_EXPERTS"],
                  f"RT_ROWL = {env['RT_ROWL']}，RT_SORTLN = {env['RT_SORTLN']}，"
                  f"RT_SORT_NBLK = {env['RT_SORT_NBLK']}，NUM_EXPERTS = {env['NUM_EXPERTS']}")
        # B6（M95-#4）：该档的 UB_RT_END 也必须 ≤ GDN 相位峰值（否则融合峰值被抬高）
        rep.check(f"[{tag}] UB_RT_END <= UB_RC_END（融合 UB 峰值不被抬高）",
                  env["UB_RT_END"] <= env["UB_RC_END"],
                  f"UB_RT_END = {env['UB_RT_END']} B ≤ UB_RC_END = {env['UB_RC_END']} B"
                  f"（余量 {env['UB_RC_END'] - env['UB_RT_END']} B）")
    return rep.done()


if __name__ == "__main__":
    sys.exit(main())
