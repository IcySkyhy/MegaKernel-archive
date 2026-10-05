#!/usr/bin/env python3
"""M29：MoE block 真实权重 manifest 生成器（从真实 checkpoint 的 safetensors 头取 offset/bytes）。

产出 `m17_moe_real/m17_weight_manifest.txt`（纯 key=value 行，便于 C++ 直接解析），
让 host 侧（C++，无 Python 依赖）用 pread 从分片里**字节原样**取出某一层 MoE block 的
全部权重，不做任何离线转换：

    model_dir=/workspace/Qwen3.8-Flash-Next-MXFP4
    layer=0
    tensor role=router_w file=model-00003-of-00131.safetensors offset=... bytes=... dtype=BF16 rows=512 cols=2560
    ...
    tensor role=gamma1 ... rows=10240 cols=1 row0=0 nrows=2560

行字段含义：
    offset  分片文件内 payload 起始的绝对字节偏移（safetensors data_offsets[0] + payload_base）
    bytes   该张量（或 row 切片）的字节数
    row0/nrows  可选：只取前者的第 row0 行起的 nrows 行（行连续存储时 = 一段连续字节）

用法（仓库根目录）：
    /usr/local/python3.12.13/bin/python3.12 m17_moe_real/gen_weight_manifest.py \
        --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4 --layer 0
    ... --check        # 只校验已生成的 manifest 与 checkpoint 现况一致（不写盘）

只用 numpy + 标准库（复用 tools/weights/safetensors_reader.py，不依赖 torch）。

退出码（三态；`--check` 模式下由脚本返回，其余错误由 SystemExit 带码）：
    0 = 生成成功 / `--check` 比过且一致（OK 文案带实际比较的张量记录条数）
    1 = `--check` 比过但不一致（需重新生成）
    2 = 没得比/输入缺失（manifest 不存在，或 checkpoint 缺张量/分片提前 EOF）
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "weights"))

from safetensors_reader import ShardReader  # noqa: E402

MODEL_PREFIX = "model.language_model"

HIDDEN = 2560
INTER = 640
GU_N = 2 * INTER          # 1280
GROUP = 32

# (role, checkpoint 张量后缀, 期望 shape, dtype)
# shape 为 None 表示只做字节数自洽检查（不写死形状）
TENSORS = [
    # ---- routed 专家（512 个专家的全部量化权重）----
    ("router_w",       "mlp.gate.weight",                        (512, HIDDEN),          "BF16"),
    ("sgate_w",        "mlp.shared_expert_gate.weight",          (1, HIDDEN),            "BF16"),
    ("gu_packed",      "mlp.experts.gate_up_proj",               (512, GU_N, HIDDEN // 2), "U8"),
    ("gu_scale",       "mlp.experts.gate_up_proj.weight_scale",  (512, GU_N, HIDDEN // GROUP), "U8"),
    ("dn_packed",      "mlp.experts.down_proj",                  (512, HIDDEN, INTER // 2), "U8"),
    ("dn_scale",       "mlp.experts.down_proj.weight_scale",     (512, HIDDEN, INTER // GROUP), "U8"),
    # ---- 共享专家（checkpoint 是独立 gate/up；kernel 侧 host 组板成 gate|up 行拼接）----
    ("shd_gate",       "mlp.shared_expert.gate_proj.weight",             (INTER, HIDDEN // 2), "U8"),
    ("shd_gate_scale", "mlp.shared_expert.gate_proj.weight_scale",       (INTER, HIDDEN // GROUP), "U8"),
    ("shd_up",         "mlp.shared_expert.up_proj.weight",               (INTER, HIDDEN // 2), "U8"),
    ("shd_up_scale",   "mlp.shared_expert.up_proj.weight_scale",         (INTER, HIDDEN // GROUP), "U8"),
    ("shd_down",       "mlp.shared_expert.down_proj.weight",             (HIDDEN, INTER // 2), "U8"),
    ("shd_down_scale", "mlp.shared_expert.down_proj.weight_scale",       (HIDDEN, INTER // GROUP), "U8"),
    # ---- 层输入（真实 embedding 行）与两处 RMSNorm gamma ----
    ("x_embed",        "embed_tokens.weight",                    None,                   "BF16"),
    ("hc_norm",        "mlp_hyper_connection.hc_norm.weight",    None,                   "BF16"),
]

# hc_norm 的行切片：MoE block 的两处 norm 各取 2560 行（见 README「真实输入与 gamma 的来源」）
GAMMA_ROW_SLICES = [("gamma1", 0), ("gamma2", HIDDEN)]


def _sha256_of(path, off: int, nbytes: int) -> str:
    """对分片文件的 [off, off+nbytes) 区间独立求 sha256（分块读，不占内存）。

    这是「host pread 的字节 == 源切片字节」的第一条独立腿：Python 直接从分片文件
    按 data_offsets 读并哈希，完全不经过 C++ 的 manifest 解析/pread 代码路径。
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = nbytes
        while left > 0:
            chunk = f.read(min(left, 1 << 22))
            if not chunk:
                raise SystemExit(f"[FAIL] {path} 在 offset={off} 处提前 EOF")
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def build(model_dir: str, layer: int, out_path: str, check_only: bool = False) -> None:
    rdr = ShardReader(model_dir)
    names = set(rdr.names())
    lines = [
        "# M17 MoE block 真实权重 manifest —— 由 gen_weight_manifest.py 生成，勿手改",
        f"# 张量一律字节原样（不做离线转换）；MXFP4 = packed u8(e2m1 lohi) + e8m0 scale，group {GROUP} 沿 K",
        f"model_dir={model_dir}",
        f"layer={layer}",
        f"hidden={HIDDEN}",
        f"inter={INTER}",
        f"group={GROUP}",
    ]
    n_tensor = 0
    for role, suffix, shape, dtype in TENSORS:
        name = f"{MODEL_PREFIX}.layers.{layer}.{suffix}" if not suffix.startswith("embed_") \
            else f"{MODEL_PREFIX}.{suffix}"
        if name not in names:
            raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
        info = rdr.info(name)
        if shape is not None and tuple(info.shape) != shape:
            raise SystemExit(f"[FAIL] {name} shape {tuple(info.shape)} != 期望 {shape}")
        if shape is not None and info.st_dtype != dtype:
            raise SystemExit(f"[FAIL] {name} dtype {info.st_dtype} != 期望 {dtype}")
        rows = info.shape[0] if len(info.shape) > 0 else 1
        cols = 1
        for d in info.shape[1:]:
            cols *= d
        nbytes = int(info.data_end) - int(info.data_begin)
        # sha256：**Python 侧独立**对分片里该字节区间求哈希（不经过 C++ reader），
        # host 侧 pread 后再求一次，两者必须相等 → 「逐字节与源切片一致」的第一条腿
        sha = "-" if role == "x_embed" else _sha256_of(info.file, int(info.data_begin), nbytes)
        lines.append(
            f"tensor role={role} file={os.path.basename(str(info.file))} "
            f"offset={int(info.data_begin)} bytes={nbytes} "
            f"dtype={info.st_dtype} rows={rows} cols={cols} sha256={sha}"
        )
        n_tensor += 1
        if role == "hc_norm":
            row_bytes = cols * 2   # BF16
            for grow_role, row0 in GAMMA_ROW_SLICES:
                goff = int(info.data_begin) + row0 * row_bytes
                gbytes = HIDDEN * row_bytes
                lines.append(
                    f"tensor role={grow_role} file={os.path.basename(str(info.file))} "
                    f"offset={goff} bytes={gbytes} "
                    f"dtype={info.st_dtype} rows={HIDDEN} cols={cols} row0={row0} nrows={HIDDEN} "
                    f"sha256={_sha256_of(info.file, goff, gbytes)}"
                )
                n_tensor += 1
    rdr.close()

    text = "\n".join(lines) + "\n"
    if check_only:
        if not os.path.exists(out_path):
            print(f"[check] RESULT: SKIPPED（{out_path} 不存在，无从对账） —— 退出码 2；未比较任何张量记录")
            raise SystemExit(2)
        with open(out_path) as f:
            old = f.read()
        n_cmp = sum(1 for l in text.splitlines() if l.startswith("tensor "))
        if old != text:
            print(f"[check] RESULT: DIFF（{out_path} 与 checkpoint 现况不一致，"
                  f"已比较 {n_cmp} 条张量记录） —— 退出码 1")
            raise SystemExit(1)
        print(f"[check] RESULT: OK（已比较 {n_cmp} 条张量记录的 offset/bytes/shape/dtype/sha256；"
              f"{out_path} 与 checkpoint 现况一致） —— 退出码 0")
        return
    with open(out_path, "w") as f:
        f.write(text)
    packed = 512 * GU_N * (HIDDEN // 2) + 512 * HIDDEN * (INTER // 2)
    print(f"[ok] 写出 {out_path}（layer={layer}，{n_tensor} 条张量记录）")
    print(f"     512 专家的 packed 权重量 = {packed / 2**30:.3f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/Qwen3.8-Flash-Next-MXFP4")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m17_weight_manifest.txt"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    build(args.model_dir, args.layer, args.out, args.check)


if __name__ == "__main__":
    main()
