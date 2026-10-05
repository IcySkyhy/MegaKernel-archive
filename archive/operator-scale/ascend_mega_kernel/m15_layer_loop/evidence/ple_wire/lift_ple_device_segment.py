#!/usr/bin/env python3
"""M100：从 `m15_ple.asc` **机械抽取** PLE 的 device 段，生成 `m15_layer_loop/m15_ple_wire.h`。

为什么需要这个文件（而不是直接 `#include "m15_ple.asc"`）：
  · `m15_ple.asc` 是 M85 的**独立 kernel TU**（自带 `main()`、匿名 namespace 的 `Ctx`、
    以及 `<fcntl.h>/<sys/mman.h>/<sys/stat.h>/<unistd.h>` 等 host 侧头）；
  · 把它 include 进层循环 TU 后，`runs=all` 出现 36 条 `M.moews.*` FAIL（4 字节，落在 MoE 段
    `WS_OFFSETS+32` 的**专家偏移表尾部槽位**）；而**只抽取 device 段**（本脚本的产物）时
    `runs=all` 干净。⇒ 抽取形态是**实跑选出来的**，不是偏好（见 WITNESS.md §6）。

"两份逐字一致"的见证（塔的硬要求 #1）：
  · 生成式：本脚本从 `m15_ple.asc` 抽出 `namespace M85P { … }`（含命名空间声明与结尾注释）逐字写入；
  · `--check`：重新抽取并与磁盘上的 `m15_ple_wire.h` 逐字节比 —— 不一致就 rc=1。
  ⇒ `m15_ple.asc` 一改，`--check` 立刻红（不存在"改了 A 没改副本"的静默漂移）。

用法：
  python3 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py           # 生成
  python3 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check   # 校验（rc=1 表示漂移）
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_M15 = os.path.abspath(os.path.join(HERE, "..", ".."))          # m15_layer_loop/
SRC = os.path.join(REPO_M15, "m15_ple.asc")
DST = os.path.join(REPO_M15, "m15_ple_wire.h")

HEAD = '''// ============================================================
// m15_ple_wire.h —— **机械生成物，勿手改**（M100）
// ============================================================
// 由 `m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py` 从 `m15_ple.asc` 逐字抽出
// `namespace M85P { … }`（M85 的 PLE device 段：①`IdsOneToken` / ②`PleGather` / ③`PleGemv` /
// ④`PleGateItem` / ⑤`PleConvItem` 与它们的常量、UB 布局、flag id）。
//
// **为什么是抽取而不是 `#include "m15_ple.asc"`**（实测，见 evidence/ple_wire/WITNESS.md §6）：
// 直接把 `m15_ple.asc` 借进层循环 TU（它自带 `main()`/匿名 namespace `Ctx`/system 头）会让
// `runs=all` 出现 36 条 `M.moews.*` FAIL（MoE 段 `WS_OFFSETS+32` 的 4 字节）；只抽 device 段
// 则 `runs=all` 干净。两份一致性由 `--check` 复跑保证。
//
// **包含前提**：本文件**不自带任何 include**（不引入 system 头、不引入 AscendC 头），
// 依赖 include 它的 TU 已经包含 `m15_hc_layer.h`（提供 `M15H::NormDonor` / `M15H::BarrierAiv` /
// `M15H::Block1` / cast traits / `M15H::CHUNK`）与 AscendC 头。层循环 TU 里的包含顺序是
// `m15_gdn_layer.h` → `m15_hc_layer.h` → `m15_ple_wire.h` → … 见该 TU 的 include 块。
'''

BEGIN = "namespace M85P {"
END = "}  // namespace M85P"


def extract(src_path):
    text = open(src_path).read()
    i = text.index(BEGIN)
    j = text.index(END)
    body = text[i:j] + END + "\n"
    if "M15_PLE_DEVICE_ONLY" in body:                      # 防呆：不该出现
        raise SystemExit("[FAIL] 抽取片段里出现了意外的标记")
    # 抽出来的片段不许含任何预处理指令（含 include / define），否则"device only"不成立
    for ln, line in enumerate(body.splitlines(), 1):
        s = line.lstrip()
        if s.startswith("#include") or s.startswith("#define") or s.startswith("#undef"):
            raise SystemExit(f"[FAIL] 抽取片段的第 {ln} 行含预处理指令：{s[:60]}")
    return HEAD + "\n" + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验磁盘上的 m15_ple_wire.h 与重新抽取一致")
    args = ap.parse_args()
    want = extract(SRC)
    if args.check:
        try:
            have = open(DST).read()
        except OSError as e:
            raise SystemExit(f"[FAIL] 读不到 {DST}: {e}")
        if have != want:
            raise SystemExit(f"[FAIL] {DST} 与 `m15_ple.asc` 抽出的片段不一致（漂移）—— 重新生成")
        print(f"[ok] {DST} == 从 {SRC} 重新抽取的结果（{len(want)} 字节，逐字节）")
        return
    with open(DST, "w") as f:
        f.write(want)
    print(f"[ok] 写出 {DST}（{len(want)} 字节，抽自 {SRC} 的 namespace M85P）")


if __name__ == "__main__":
    main()
