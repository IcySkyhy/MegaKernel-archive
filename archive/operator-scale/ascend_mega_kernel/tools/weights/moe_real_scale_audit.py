#!/usr/bin/env python3
"""M93 第一步：MoE 段的**真规模**取证（只读 checkpoint 头，不读 payload）。

回答人类的两句原话：
  * 「权重就是要常驻」 —— MoE 段全 48 层常驻 HBM 要多少字节、占多少 HBM；
  * 「一层的内存应该不多吧」 —— **单层 MoE 权重的真实数字**是多少（本文件的 `per-layer` 表）。

本文件**只读 safetensors 头**（`ShardReader.info()` → `data_begin`/`data_end`/`shape`/`dtype`），
一个 payload 字节都不读 ⇒ 不触碰 32GB cgroup 上限（见任务纪律 ⑪）。逐张量字节数用**头部
自己给的 data_offsets 之差**算，不靠 shape 推导；再用 shape×dtype 做交叉核对（两者不符即 FAIL）。

口径（配置文件是唯一权威，不写死常量）：
  num_experts / num_experts_per_tok / moe_intermediate_size / hidden_size / num_hidden_layers
  全部从 `config.json` 的 `text_config` 读；本文件里的期望值只用于**核对**，不用于计算。

判据（每条都打印 PASS/FAIL 与读数）：
  C1 每层 12 个 MoE 张量都在，且 48 层逐层字节数完全相同（层间不一致 ⇒ FAIL）
  C2 头部的 `data_end-data_begin` == prod(shape) × dtype 字节（逐张量）
  C3 单专家字节数 == shape[1:] 之积（专家在**最外维** ⇒ 单专家连续段 = 次维起之积）
  C4 `weight` 的 K 打包 == K/2（2 nibble/byte）、`weight_scale` 的 K 尺度 == K/GROUP（group 32）
  C5 config 的 num_experts / moe_intermediate_size / hidden_size 与头部 shape 一致
  C6 全 512 专家段 = 单专家字节 × 512 = 该张量头部全量字节（逐张量）
  C7 HBM 占比 = 总数 / --hbm-mib 的 MiB 口径（默认真机读数 131072）

负向对照（有判别力，必须 FAIL）：
  `--negative-experts N` 把「专家数」当 N（默认 4，即当前缩形档）来算。
  C3/C6 会因为 `单专家字节 × N != 头部全量` 而 FAIL ⇒ 证明这两条判据真的咬得住
  「把 512 当 4」/「stride 算错」这一形态。

用法（仓库根目录）：
    PY=/usr/local/python3.12.13/bin/python3.12
    $PY tools/weights/moe_real_scale_audit.py --out tools/weights/evidence/moe_real_scale.txt
    $PY tools/weights/moe_real_scale_audit.py --negative-experts 4   # 必须 FAIL（rc=1）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from safetensors_reader import ShardReader  # noqa: E402

MODEL_PREFIX = "model.language_model"

# 一层的 MoE 张量：role → (mlp 段的后缀, 是否是「按专家最外维」的张量)
# `routed=True` 表示第一维是专家维（512），可整段或按专家切片；
# `routed=False` 表示非专家维（router 的 [E, HIDDEN] 是特例：它按专家切，但语义是"每专家一行"）。
MOE_TENSORS = [
    ("router_w", "gate.weight", True),
    ("sgate_w", "shared_expert_gate.weight", False),
    ("experts_gate_up", "experts.gate_up_proj", True),
    ("experts_gate_up_scale", "experts.gate_up_proj.weight_scale", True),
    ("experts_down", "experts.down_proj", True),
    ("experts_down_scale", "experts.down_proj.weight_scale", True),
    ("shared_gate", "shared_expert.gate_proj.weight", False),
    ("shared_gate_scale", "shared_expert.gate_proj.weight_scale", False),
    ("shared_up", "shared_expert.up_proj.weight", False),
    ("shared_up_scale", "shared_expert.up_proj.weight_scale", False),
    ("shared_down", "shared_expert.down_proj.weight", False),
    ("shared_down_scale", "shared_expert.down_proj.weight_scale", False),
]

DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "I32": 4, "U32": 4, "F64": 8}

MIB = 1024 * 1024


def prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


class Report:
    """收集判据行；`FAIL` 一旦出现 rc 就非 0。"""

    def __init__(self, neg_experts=None):
        self.lines = []
        self.fails = 0
        self.checks = 0
        self.neg_experts = neg_experts

    def p(self, s=""):
        self.lines.append(s)
        print(s)

    def check(self, ok, tag, detail):
        self.checks += 1
        if not ok:
            self.fails += 1
        self.p(f"[chk] {tag:<34s} {'PASS' if ok else 'FAIL'}  {detail}")
        return ok


def load_config(model_dir: Path) -> dict:
    """text_config 与顶层（quantization_config 在顶层）合并成一份，顶层优先。"""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    merged = dict(cfg.get("text_config", {}))
    for k, v in cfg.items():
        merged.setdefault(k, v)
    return merged


def audit(model_dir: Path, hbm_mib: int, neg_experts=None) -> Report:
    R = Report(neg_experts)
    cfg = load_config(model_dir)
    n_layers = int(cfg["num_hidden_layers"])
    n_experts = int(cfg["num_experts"])
    topk = int(cfg["num_experts_per_tok"])
    inter = int(cfg["moe_intermediate_size"])
    hidden = int(cfg["hidden_size"])
    sh_inter = int(cfg["shared_expert_intermediate_size"])
    group = int(cfg["quantization_config"]["group_size"])

    R.p("# M93 · MoE 段真规模取证（只读 safetensors 头；无 payload 读）")
    R.p(f"# model_dir={model_dir}")
    R.p(f"# config: num_hidden_layers={n_layers} num_experts={n_experts} "
        f"num_experts_per_tok={topk} moe_intermediate_size={inter} hidden_size={hidden} "
        f"shared_expert_intermediate_size={sh_inter} group_size={group}")
    R.p(f"# quant_method={cfg['quantization_config']['quant_method']} "
        f"format={cfg['quantization_config']['format']} "
        f"packing={cfg['quantization_config']['packing']}")
    R.p(f"# HBM 口径 = {hbm_mib} MiB（真机 npu-smi 读数）")
    if neg_experts is not None:
        R.p(f"# !! 负向对照：把专家数当 {neg_experts}（真值 {n_experts}）——C3/C6 应 FAIL")
    R.p()

    rdr = ShardReader(model_dir)
    incomplete = {k: v for k, v in rdr.incomplete.items() if v}
    R.check(not incomplete, "C0 分片完整性", f"incomplete shards = {len(incomplete)}")

    # ---- 逐层逐张量：只用头部 ----
    per_layer = {}          # L -> {role: (file, offset, hdr_bytes, per_expert_bytes, n_axis, shape, dt)}
    c2_bad = []             # C2 反例（按匹配片段收集，不按行）
    c5_bad = []             # C5 专家维反例
    for L in range(n_layers):
        row = {}
        for role, suffix, routed in MOE_TENSORS:
            name = f"{MODEL_PREFIX}.layers.{L}.mlp.{suffix}"
            if name not in set(rdr.names()):
                raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
            info = rdr.info(name)
            shape = list(info.shape)
            dt = info.st_dtype
            if dt not in DT_BYTES:
                raise SystemExit(f"[FAIL] 未知 dtype {dt} for {name}")
            hdr_bytes = int(info.data_end) - int(info.data_begin)
            calc_bytes = prod(shape) * DT_BYTES[dt]
            if hdr_bytes != calc_bytes:      # C2：头部 offset 差 == shape×dtype
                c2_bad.append(f"{suffix}@{L} hdr={hdr_bytes} shape×dt={calc_bytes}")
            n_axis = shape[0] if len(shape) > 0 else 1
            if routed and n_axis != n_experts:
                c5_bad.append(f"{suffix}@{L} shape[0]={n_axis}")
            per_expert = (prod(shape[1:]) if len(shape) > 1 else 1) * DT_BYTES[dt]
            row[role] = (info.file.name, int(info.data_begin), hdr_bytes, per_expert, n_axis,
                         shape, dt)
        per_layer[L] = row

    R.check(not c2_bad, "C2 头部 offset 差 == shape×dtype",
            f"{n_layers*len(MOE_TENSORS)} 个 (层,张量) 对；反例 {len(c2_bad)}"
            + (f" ⇒ {c2_bad[:4]}" if c2_bad else ""))
    R.check(not c5_bad, "C5 专家维 shape[0] == num_experts",
            f"反例 {len(c5_bad)}" + (f" ⇒ {c5_bad[:4]}" if c5_bad else ""))

    # ---- C1：48 层逐层字节数完全相同 ----
    ref = {r: per_layer[0][r][2] for r, _, _ in MOE_TENSORS}
    same = all({r: per_layer[L][r][2] for r, _, _ in MOE_TENSORS} == ref for L in range(n_layers))
    R.check(same, "C1 48 层逐层字节数相同",
            f"{'全部相同' if same else '存在层间差异'}；层 0 各张量 bytes={ref}")

    # ---- C3/C6：单专家字节 × 512（或被改错的 N）== 头部全量 ----
    R.p()
    R.p("## 逐张量：单专家字节 / 全 512 专家段（专家在**最外维**，专家维 = shape[0]）")
    R.p(f"{'role':<24s} {'dtype':>5s} {'shape':>22s} {'1 专家 B':>12s} {'N 专家 B':>14s} "
        f"{'头部全量 B':>14s}")
    for role, suffix, routed in MOE_TENSORS:
        _, _, hdr_bytes, per_expert, n_axis, shape, dt = per_layer[0][role]
        N = neg_experts if neg_experts is not None else n_experts
        if routed:
            full = per_expert * N
            R.p(f"{role:<24s} {dt:>5s} {str(shape):>22s} {per_expert:>12d} {full:>14d} "
                f"{hdr_bytes:>14d}")
            R.check(full == hdr_bytes, f"C3/C6 {role}",
                    f"1 专家 {per_expert} × {N} = {full} vs 头部全量 {hdr_bytes}")
        else:
            R.p(f"{role:<24s} {dt:>5s} {str(shape):>22s} {'—':>12s} {'—':>14s} "
                f"{hdr_bytes:>14d}   （无专家维，整张量）")

    # ---- C4：K 方向打包 / 尺度步长 ----
    R.p()
    gu = per_layer[0]["experts_gate_up"][5]
    gus = per_layer[0]["experts_gate_up_scale"][5]
    dn = per_layer[0]["experts_down"][5]
    dns = per_layer[0]["experts_down_scale"][5]
    R.check(gu[2] * 2 == hidden, "C4 gate_up 打包 K/2",
            f"gate_up K 打包 {gu[2]} ×2 = {gu[2]*2} == hidden {hidden}")
    R.check(gus[2] * group == hidden, "C4 gate_up scale K/group",
            f"gate_up scale {gus[2]} ×{group} = {gus[2]*group} == hidden {hidden}")
    R.check(dn[2] * 2 == inter, "C4 down 打包 K/2",
            f"down K 打包 {dn[2]} ×2 = {dn[2]*2} == inter {inter}")
    R.check(dns[2] * group == inter, "C4 down scale K/group",
            f"down scale {dns[2]} ×{group} = {dns[2]*group} == inter {inter}")
    R.check(gu[1] == 2 * inter, "C4 gate_up N=2·inter",
            f"gate_up N {gu[1]} == 2×inter {2*inter}")
    R.check(dn[1] == hidden, "C4 down N=hidden", f"down N {dn[1]} == hidden {hidden}")

    # ---- 汇总 ----
    routed_roles = [r for r, _, routed in MOE_TENSORS if routed and r != "router_w"]
    other_roles = [r for r, _, _ in MOE_TENSORS if r not in routed_roles]
    per_layer_routed = sum(per_layer[0][r][2] for r in routed_roles)
    per_layer_router = per_layer[0]["router_w"][2]
    per_layer_shared = sum(per_layer[0][r][2] for r in other_roles if r != "router_w")
    per_layer_moe = per_layer_routed + per_layer_router + per_layer_shared

    R.p()
    R.p("## 单层 MoE 权重（回答「一层的内存应该不多吧」）")
    R.p(f"  routed 专家（512 × 4 张量）        = {per_layer_routed:>13d} B "
        f"= {per_layer_routed/MIB:9.3f} MiB")
    R.p(f"  router gate（512×2560 bf16）       = {per_layer_router:>13d} B "
        f"= {per_layer_router/MIB:9.3f} MiB")
    R.p(f"  shared expert（gate/up/down + scale）= {per_layer_shared:>13d} B "
        f"= {per_layer_shared/MIB:9.3f} MiB")
    R.p(f"  ---- 单层 MoE 合计                 = {per_layer_moe:>13d} B "
        f"= {per_layer_moe/MIB:9.3f} MiB = {per_layer_moe/1024**3:.4f} GiB")
    R.p(f"  （单层占 HBM {hbm_mib} MiB 的 {100.0*per_layer_moe/(hbm_mib*MIB):.3f}%）")

    total_moe = per_layer_moe * n_layers
    R.p()
    R.p(f"## 全 {n_layers} 层 MoE 常驻预算")
    R.p(f"  routed 专家（512 专家 × 48 层）    = {per_layer_routed*n_layers:>13d} B "
        f"= {per_layer_routed*n_layers/MIB:11.2f} MiB = {per_layer_routed*n_layers/1024**3:8.3f} GiB")
    R.p(f"  router + shared 专家                = "
        f"{(per_layer_router+per_layer_shared)*n_layers:>13d} B "
        f"= {(per_layer_router+per_layer_shared)*n_layers/MIB:11.2f} MiB")
    R.p(f"  ---- MoE 段全量                    = {total_moe:>13d} B "
        f"= {total_moe/MIB:11.2f} MiB = {total_moe/1024**3:8.3f} GiB")
    R.p(f"  占 HBM {hbm_mib} MiB 的 {100.0*total_moe/(hbm_mib*MIB):.4f}%"
        f"（routed 部分 {100.0*per_layer_routed*n_layers/(hbm_mib*MIB):.4f}%）")

    # ---- 全模型 HBM 预算（回答「权重常驻可行否」；ngram 表在 host，不计入） ----
    # 人类约束：「权重就是要常驻，只有 ngram 权重在 host」⇒ 进 HBM 的是**非 ngram** 那部分。
    total_all = 0
    ngram_bytes = 0
    moe_routed_mtp = 0
    for nm in rdr.names():
        b = rdr.info(nm)
        nb = int(b.data_end) - int(b.data_begin)
        total_all += nb
        if "ngram" in nm.lower():
            ngram_bytes += nb
        if nm.startswith("mtp.") and ".mlp.experts." in nm:
            moe_routed_mtp += nb
    non_ngram = total_all - ngram_bytes
    hbm_bytes = hbm_mib * MIB
    R.p()
    R.p("## 全模型 HBM 预算（ngram 表在 host，按人类约束不进 HBM）")
    R.p(f"  checkpoint 全部张量          = {total_all:>13d} B = {total_all/MIB:10.2f} MiB "
        f"= {total_all/1024**3:7.2f} GiB")
    R.p(f"  ngram 表（host 侧）          = {ngram_bytes:>13d} B = {ngram_bytes/MIB:10.2f} MiB "
        f"= {ngram_bytes/1024**3:7.2f} GiB（占整模型 {100.0*ngram_bytes/total_all:.2f}%）")
    R.p(f"  ---- 非 ngram（需进 HBM）    = {non_ngram:>13d} B = {non_ngram/MIB:10.2f} MiB "
        f"= {non_ngram/1024**3:7.2f} GiB = HBM 的 {100.0*non_ngram/hbm_bytes:.2f}%")
    R.p(f"     其中 MoE routed 48 层      = {per_layer_routed*n_layers:>13d} B "
        f"= {per_layer_routed*n_layers/1024**3:7.2f} GiB = HBM 的 "
        f"{100.0*per_layer_routed*n_layers/hbm_bytes:.2f}%")
    R.p(f"     其中 MoE routed（mtp 段）  = {moe_routed_mtp:>13d} B = "
        f"{moe_routed_mtp/1024**3:7.2f} GiB")
    R.p(f"     其余（GDN/attention/hc/embedding/lm_head/…）= "
        f"{(non_ngram - per_layer_routed*n_layers - moe_routed_mtp)/1024**3:7.2f} GiB")
    R.p(f"  ---- 留给 KV / 激活 / 运行时 = {(hbm_bytes - non_ngram):>13d} B = "
        f"{(hbm_bytes - non_ngram)/1024**3:7.2f} GiB（HBM 的 "
        f"{100.0*(hbm_bytes - non_ngram)/hbm_bytes:.2f}%）")
    R.check(non_ngram <= hbm_bytes, "C8 非 ngram 权重可全部常驻 HBM",
            f"非 ngram {non_ngram/1024**3:.2f} GiB ≤ HBM {hbm_bytes/1024**3:.2f} GiB "
            f"⇒ 常驻可行，余 {(hbm_bytes-non_ngram)/1024**3:.2f} GiB 给 KV/激活")

    # 交叉见证：E=4 缩形档的倍数关系
    for role in ("experts_gate_up", "experts_gate_up_scale", "experts_down", "experts_down_scale"):
        _, _, hdr, per_expert, _, _, _ = per_layer[0][role]
        mult = hdr // per_expert
        R.check(mult == n_experts, f"C6 {role} 全量/单专家",
                f"{hdr} / {per_expert} = {mult} == num_experts {n_experts}")

    R.p()
    R.p(f"[summary] checks={R.checks} fails={R.fails}")
    rdr.close()
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path,
                    default=Path("/workspace/Qwen3.8-Flash-Next-MXFP4"))
    ap.add_argument("--hbm-mib", type=int, default=131072,
                    help="HBM 容量（MiB）；真机 npu-smi 读数 131072")
    ap.add_argument("--negative-experts", type=int, default=None,
                    help="负向对照：把专家数当 N 来算（默认 4 = 当前缩形档）；C3/C6 必 FAIL")
    ap.add_argument("--out", type=Path, default=None, help="把报告也写到该文件")
    args = ap.parse_args()

    R = audit(args.model_dir, args.hbm_mib, args.negative_experts)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(R.lines) + "\n")
        print(f"[ok] 报告写入 {args.out}")
    sys.exit(1 if R.fails else 0)


if __name__ == "__main__":
    main()
