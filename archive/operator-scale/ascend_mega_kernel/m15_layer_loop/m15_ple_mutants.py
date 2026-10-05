#!/usr/bin/env python3.12
"""m15_ple_mutants.py —— **变异矩阵驱动**：证明「每一条上游分歧都有一条判据能咬住它」

`docs/17 §4` 要求「把该算件的规则/方向故意反过来跑一次，凡被声明为有咬合力的判定项必须至少有一条 FAIL」。
本脚本把它做成一键可复现的矩阵：对 `m15_ple.asc` 的每一个变异位跑一次，要求**指定的那条判据 FAIL**，
并要求**基线（mask=0）全 PASS**。

变异位表（每一位 = 一条我们相对上游实现的分歧被故意写错；对照表见 `ple/DIVERGENCES.md`）：

| bit | 变异（故意写错的方式） | 对应的分歧 | 必须 FAIL 的判据 |
|---|---|---|---|
| 0 | ① 取模方向反转（floor→ceil） | D5（`docs/14 §6.2` 未明说 floor-mod 方向） | `B1.ids.A` |
| 1 | ① ctx 列退回「与 c 无关」 | **D3**（n-gram 上下文列，c=1） | `B1.ids.A` |
| 2 | ④ 跳过 dot 和的 bf16 物化 | **D4 / P2**（dot 和的取整点） | `B4.gated` |
| 3 | ⑤ tap 归属反转（kt=KCONV-1-k） | D2 家族（⑤ 的 tap 序） | `B5.out` |
| 4 | ⑤ 残差加分组反转 | D2 家族（⑤ 的残差顺序） | `B5.out` |
| 5 | ④ value 换成 key 切片（破坏 4 流共享） | D6（value 4 流共享） | `B4.gated` |
| 6 | ② head 落点顺序反转 | D7（gather 到 head 槽的映射） | `B2.emb` |
| 7 | ③ key/value 输出块对调（**M124 起 = 读权重时 N 侧 160 行整块平移 ±HID**，见下） | **D1**（kv 融合顺序 key 在前） | `B3.kv` |
| 8 | ⑤ 状态移位方向反转 | D8（状态移位方向） | `B5.state` |
| 9 | ⑤ null 槽位整行跳过 | **N1**（NULL_STATE_ID 仍写 out） | `B5.null.out` |
| 10 | ① 漏掉取模与偏移 | ① 值域判据的咬合力 | `B1.ids.A.range` |
| 11 | ⑤ null 行也写状态 | N1 家族（null 行不写状态） | `B5.null.state` |
| 12 | ⑤ 活跃行不写回状态 | D8 家族（状态必须演化） | `B5.state_evolve` |
| 13 | ⑤ 跳过 conv_output 的 bf16 物化 | **D2 本身**（⑤ 的 conv_output 取整点） | `B5.out` |
| 15 | ② 的 id 被推离词表值域（`id += VOCAB_ROWS`） | ② 自身输入的值域守卫（**M111 新增**：设备侧计数） | `B_dev.fail` |

用法（仓库根目录）：
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_mutants.py
  env: M15_PLE_BIN=<kernel 路径>（默认 m15_layer_loop/build/m15_ple）
输出：表格 + `RESULT|` 行；整段经 `tee` 由调用方归档（见 ple/logs/mutants.log）
退出码：0 = 全部期望满足；1 = 有期望未被满足（**或**基线不是全 PASS）；2 = 环境/输入缺失
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent          # m15_layer_loop/
ROOT = HERE.parent                                       # 仓库根
PY = sys.executable

MUTANTS = [
    (0, "① 取模方向反转（floor→ceil）", "D5", "B1.ids.A"),
    (1, "① ctx 列退回与 c 无关", "D3", "B1.ids.A"),
    (2, "④ 跳过 dot 和的 bf16 物化", "D4/P2", "B4.gated"),
    (3, "⑤ tap 归属反转", "⑤ 的 tap 序", "B5.out"),
    (4, "⑤ 残差加分组反转", "⑤ 的残差分组", "B5.out"),
    (5, "④ value 换成 key 切片", "D6", "B4.gated"),
    (6, "② head 落点顺序反转", "D7", "B2.emb"),
    # M124：③ 改成 cube mmad 之后，这一位的**形态跟着算法走** —— 不再是"逐列读对侧权重行"，
    #   而是"整个 N-tile（160 行）的权重读取基址平移 ±HID 行"；key 半区 nt<64、value 半区 nt≥64
    #   （HYPER/BASE_N = 64，一个 tile 不跨边界）。语义仍是 D1 的顺序反 ⇒ `B3.kv` 必须 FAIL。
    (7, "③ key/value 输出块对调（cube 档：N-tile 读取基址 ±HID）", "D1", "B3.kv"),
    (8, "⑤ 状态移位方向反转", "D8", "B5.state"),
    (9, "⑤ null 槽位整行跳过（漏写 out）", "N1", "B5.null.out"),
    (10, "① 漏掉取模与偏移（id = mixed）", "① 值域", "B1.ids.A.range"),
    (11, "⑤ null 行也写状态", "N1 家族", "B5.null.state"),
    (12, "⑤ 活跃行不写回状态", "D8 家族", "B5.state_evolve"),
    (13, "⑤ 跳过 conv_output 的 bf16 物化", "**D2**（那个取整点本身）", "B5.out"),
    # M111 新增：② 自己的输入值域守卫（`PleGather` 真的数「词表外的 id」，落 `G.fail[bid]`）。
    #   这一位**不破坏任何上游分歧**，它是「把被测对象弄坏 ⇒ 新判据必须变红」的见证：
    #   `id += VOCAB_ROWS` ⇒ 该 item 不进表读（不发越界 DataCopy）而只计数 ⇒ `B_dev.fail` 必须 FAIL。
    (15, "② 的 id 被推离词表值域（值域守卫）", "② 自身输入值域（结构）", "B_dev.fail"),
]

# ---- M92：host-mapped 真实表路径的变异（`ple/REAL_TABLE.md` §5）----
# 与上面同一张矩阵的**另一个数据面**（真实分片 + 注册窗口）。每个变异必须满足：
#   ① 指定的 host 判据 FAIL；② **设备侧**计数器也要报出来（否则"设备侧判据"是装饰）。
HM_MUTANTS = [
    (14, "窗口行号按方向反转（row = winRows-1-(id-winBase)）", "id→偏移换算的方向",
     "H2.emb", "Hd.row_fail"),
    (6, "窗口模式下 head 落点顺序反转", "D7（gather 到 head 槽的映射）", "H2.emb", None),
]


def run_checker(outdir):
    """跑判据脚本，返回 {判据名: PASS|FAIL}（按机器可读的 RESULT| 行解析）。"""
    r = subprocess.run([PY, str(HERE / "m15_ple_check.py"), str(outdir), "--brief"],
                       capture_output=True, text=True, cwd=str(ROOT))
    res = {}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT|"):
            f = line.split("|")
            res[f[1]] = f[5]
    return res, r.stdout, r.returncode


def run_kernel(outdir, mut, binary, data, hm=False):
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["M15_PLE_OUT"] = str(outdir)
    env["M15_PLE_DATA"] = str(data)
    env["M15_PLE_MUT"] = str(mut)
    if hm:
        env["M15_PLE_HM"] = "1"
    r = subprocess.run([str(binary)], capture_output=True, text=True, cwd=str(ROOT), env=env)
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.environ.get("M15_PLE_BIN", str(HERE / "build" / "m15_ple")))
    ap.add_argument("--data", default=str(HERE / "ple" / "data"))
    ap.add_argument("--outroot", default=str(HERE / "ple"))
    ap.add_argument("--hm-data", default=str(HERE / "ple" / "data_hm"),
                    help="M92 host-mapped 真实表路径的输入目录（real_table_probe.py --emit 生成）")
    args = ap.parse_args()
    binary = pathlib.Path(args.bin)
    data = pathlib.Path(args.data)
    outroot = pathlib.Path(args.outroot)
    for p, what in ((binary, "kernel 二进制"), (data / "meta.json", "输入数据")):
        if not p.exists():
            print("[mut] 缺 %s: %s（先 build / 先跑 gen_ple_data.py）" % (what, p))
            return 2

    print("[mut] 基线（mask=0）")
    base_dir = outroot / "out"
    rc, _ = run_kernel(base_dir, 0, binary, data)
    if rc != 0:
        print("[mut] 基线 kernel rc=%d（非 0）" % rc)
        return 2
    base, _, _ = run_checker(base_dir)
    n_fail_base = sum(1 for v in base.values() if v == "FAIL")
    print("[mut] 基线判据：%d 条，FAIL %d 条" % (len(base), n_fail_base))
    ok_all = (n_fail_base == 0)

    print("\n[mut] 变异矩阵（每行：位 / 分歧 / 期望 FAIL 的判据 / 实测 FAIL 集合 / 结论）")
    rows = []
    for bit, desc, div, expect in MUTANTS:
        d = outroot / ("out_mut_%d" % bit)
        rc, _ = run_kernel(d, 1 << bit, binary, data)
        # kernel rc 非 0 **不算**失败：有的变异就是要让设备侧计数器（dev_range_fail）报出来
        res, _, _ = run_checker(d)
        if not res:
            print("[mut] bit%-2d 判据脚本没有产出任何 RESULT 行（kernel rc=%d）" % (bit, rc))
            rows.append((bit, desc, div, expect, "(no-result)", "NG"))
            ok_all = False
            continue
        # bit0 是唯一的"只跑 ①"变异（body 不跑），期望集合只有 B1.ids.A
        failed = sorted([k for k, v in res.items() if v == "FAIL"])
        good = (expect in failed) and (base.get(expect) == "PASS")
        rows.append((bit, desc, div, expect, ",".join(failed) if failed else "(none)", "OK" if good else "NG"))
        if not good:
            ok_all = False
    for bit, desc, div, expect, obs, verdict in rows:
        print("RESULT|mut|%d|%s|%s|%s|%s|%s" % (bit, div, expect, obs, verdict, desc))

    # ---- 覆盖核查：**每一条**基线判据都必须出现在某个变异体的 FAIL 集合里 ----
    # 这是"有咬合力 N 条"这句话的唯一合法来源（不是数非取反行）
    covered = set()
    for _, _, _, _, obs, _ in rows:
        for name in obs.split(","):
            name = name.strip()
            if name in base:
                covered.add(name)
    uncovered = sorted([k for k, v in base.items() if k not in covered])
    cov_ok = (not uncovered)
    print("\nRESULT|coverage|%d|%d|%s|%s" % (len(covered), len(base),
                                             ",".join(uncovered) if uncovered else "(none)",
                                             "OK" if cov_ok else "NG"))
    print("[mut] 咬合力**被变异演示**过的判据 = %d / %d；未被演示 = %s"
          % (len(covered), len(base), uncovered if uncovered else "（无）"))
    ok_all = ok_all and cov_ok

    # ============ M92：host-mapped 真实表数据面 ============
    print("\n[mut] ---- M92 host-mapped 真实表（数据 = ple/data_hm，加 M15_PLE_HM=1）----")
    hmdata = pathlib.Path(args.hm_data)
    ok_hm = True
    if not (hmdata / "hm_meta.txt").exists():
        print("[mut] 缺 M92 输入 %s（先跑 real_table_probe.py --emit）⇒ 本段 SKIP，不计入结论"
              % (hmdata / "hm_meta.txt"))
    else:
        hm_base_dir = outroot / "out_hm"
        rc, _ = run_kernel(hm_base_dir, 0, binary, hmdata, hm=True)
        if rc != 0:
            print("[mut] M92 基线 kernel rc=%d（非 0）" % rc)
            return 2
        hmb, _, _ = run_checker(hm_base_dir)
        nf = sum(1 for v in hmb.values() if v == "FAIL")
        print("[mut] M92 基线判据：%d 条，FAIL %d 条" % (len(hmb), nf))
        ok_hm = (nf == 0)
        hm_rows = []
        for bit, desc, div, expect, expect_dev in HM_MUTANTS:
            d = outroot / ("out_hm_mut_%d" % bit)
            rc, _ = run_kernel(d, 1 << bit, binary, hmdata, hm=True)
            res, _, _ = run_checker(d)
            failed = sorted([k for k, v in res.items() if v == "FAIL"])
            good = (expect in failed) and (hmb.get(expect) == "PASS")
            if expect_dev is not None:
                # 设备侧判据也必须报出来：**这是"设备侧判据不是装饰"的证明**
                good = good and (expect_dev in failed)
            hm_rows.append((bit, desc, div, expect, expect_dev, ",".join(failed) if failed else "(none)",
                            "OK" if good else "NG"))
            if not good:
                ok_hm = False
        for bit, desc, div, expect, expect_dev, obs, verdict in hm_rows:
            print("RESULT|mut_hm|%d|%s|%s|%s|%s|%s" % (bit, div, expect, obs, verdict, desc))
        covered_hm = set()
        for _, _, _, _, _, obs, _ in hm_rows:
            for name in obs.split(","):
                name = name.strip()
                if name in hmb:
                    covered_hm.add(name)
        unc_hm = sorted([k for k, v in hmb.items() if k not in covered_hm])
        print("RESULT|coverage_hm|%d|%d|%s|%s" % (len(covered_hm), len(hmb),
                                                  ",".join(unc_hm) if unc_hm else "(none)",
                                                  "OK" if not unc_hm else "NG"))
        print("[mut] M92 咬合力被演示 = %d / %d；未被演示 = %s"
              % (len(covered_hm), len(hmb), unc_hm if unc_hm else "（无）"))
        if unc_hm:
            print("[mut] 注：%s 未被任何 M92 变异咬到属**预期** —— 它们是设备侧结构性守卫"
                  "（Hd.cores / Hd.nonvac）与窗口范围判定（Hd.miss，bit14 不动它）"
                  % ",".join(unc_hm))
        # 覆盖不全**不单独判负**：守卫类判据本就不该被「错行」类变异咬动，强制全覆盖会诱导
        # 为凑覆盖而写假的变异。本段的硬要求是：每个 HM_MUTANTS 项指定的 host 判据 +
        # （如有）设备侧判据都必须 FAIL —— 见上面的 OK/NG。
    ok_all = ok_all and ok_hm
    ok_all = ok_all and ok_hm
    print("\n[mut] ===== %s =====" % ("全部期望满足，且每条基线判据都被变异演示过" if ok_all else "有期望未被满足或有判据未被演示"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
