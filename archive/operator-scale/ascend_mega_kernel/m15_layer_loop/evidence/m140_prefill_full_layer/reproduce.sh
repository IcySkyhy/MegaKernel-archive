#!/usr/bin/env bash
# M140 整层 GDN prefill（H1/A/H2/B 四相位）的**证据复算脚本**。
#
#   bash m15_layer_loop/evidence/m140_prefill_full_layer/reproduce.sh            # 只做离线复算
#   M140_DEVICE=1 bash m15_layer_loop/evidence/m140_prefill_full_layer/reproduce.sh  # 重跑设备档 + 复算
#
# ## 洁净检出上能跑什么（M140 复审 r1 P2-2 的口径）
#   入库的（11 个 m=1 档目录的激活面 + meta/sha256 + m=1 p15 的 h0/ht + 一份 router pad）：
#     · hc 四相位判据（`check_full_layer.py`）：权重面不入 git，脚本/判据改从
#       `m15_layer_loop/weights_manifest.txt` + checkpoint 取（与 host 装载同源）；
#       需要 checkpoint 在 `<(manifest 的 model_dir)>` 上可达，否则该档记 SKIP。
#     · 相位 A 判据（m23）：只用 `dumps_m1_p15`（该档的 h0/ht/q/k/v/g/β/o 入库）。
#     · 相位 B 判据（m26）：只用 `dumps_m1_p15`（wpad 入库一份，脚本拷进各 tag 子目录）。
#   **跑不了**（脚本会打印 SKIP 原因，不再抛 traceback）：`dumps_m4097_p15`（整档 *.bin 不入 git）。
#   要复算 m=4097 档：按下面的设备命令重跑该档，或 `M140_DEVICE=1` 全量重跑。
#
# ## 设备纪律（塔）
#   每档**各自进锁**、进锁先 `npu-smi` 并把快照写进该档日志、`timeout` 在锁内、一次进锁一条短命令、
#   `flock -w ≤300`。等锁没拿到 ⇒ 该档记「未取得读数」并跳过（不静默）。
#
# ## 相位阶梯（每档一个掩码，位 = H1:1 / A:2 / H2:4 / B:8）
#   p1=1（只 H1） p3=3（H1+A） p7=7（H1+A+H2） p15=15（全四相位）
#   p8=8：**隔离档** —— 只开相位 B，输入取 `A.xLayer`（host 填的确定性平面），用来把"MoE 自身在
#          本 kernel 里对不对"与"hc 段先跑过的影响"分开（M140 复审 r1 的相位 B 归因）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-/usr/local/python3.12.13/bin/python3}"
BIN="$REPO/m15_layer_loop/build/m15_layer_loop"
MAN="$REPO/m15_layer_loop/weights_manifest.txt"
LOCK=/tmp/npu0.lock

# 一档设备运行：$1 = 相位掩码, $2 = m, $3 = 档名, $4 = timeout(s), $5 = 额外 env（可空）
run_one() {
  local mask="$1" m="$2" name="$3" to="$4" extra="${5:-}"
  local dir="$HERE/dumps_$name"
  local log="$HERE/logs/run_${name}.log"
  mkdir -p "$dir" "$HERE/logs"
  flock -w 300 "$LOCK" bash -c "
    { echo '=== lock acquired '\$(date -Is)' mask=$mask m=$m extra=[$extra] ==='
      echo '--- npu-smi (lock entry) ---'
      npu-smi info
      timeout $to env $extra M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 \
        M15_PREFILL_PHASES=$mask M15_PREFILL_M=$m M15_PREFILL_DUMPDIR=$dir \
        '$BIN' '$MAN' prefill
      echo \"exit=\$?\"
    } > '$log' 2>&1
  " || { echo "[m140] 未取得读数：锁未拿到（$name）"; return 1; }
  echo "[m140] 设备档完成：$log"
}

if [ "${M140_DEVICE:-0}" = "1" ]; then
  run_one 1   1    m1_p1     300
  run_one 3   1    m1_p3     300
  run_one 7   1    m1_p7     300
  run_one 15  1    m1_p15    300
  run_one 8   1    m1_p8     300
  run_one 12  1    m1_p12    300
  # M140 复审 r2：pad 置零的**单变量 A/B**控制侧（同一二进制，只多一个 M15_PREFILL_NO_PAD=1）
  run_one 15  1    m1_p15_nopad 300 "M15_PREFILL_NO_PAD=1"
  run_one 12  1    m1_p12_nopad 300 "M15_PREFILL_NO_PAD=1"
  # M140 复审 r2：pad **内容**控制侧（同样的 memset 尺寸，只把填充字节从 0x00 换成 0x3C；0x3C3C 按
  # bf16 解码 = 2^(120-127) × (1 + 60/128) = 0.0114746，**不是** 1.0 —— 复审 r3 指出、已复算更正）
  run_one 15  1    m1_p15_pad3c 300 "M15_PREFILL_PAD_BYTE=60"
  run_one 12  1    m1_p12_pad3c 300 "M15_PREFILL_PAD_BYTE=60"
  # M140 r3：repeat-run **方差基线**（同一二进制、同一输入、同一档重复跑；用来把 pad 的差值与
  # clean 自身的 run-to-run 抖动分开）
  run_one 12  1    m1_p12_rep2 300
  run_one 12  1    m1_p12_rep3 300
  run_one 15  1    m1_p15_rep2 300
  run_one 15  1    m1_p15_rep3 300
  run_one 15  4097 m4097_p15 500
fi

# ---- M140 复审 r2：pad 置零的单变量 A/B 差异签名（同一二进制，只差 M15_PREFILL_NO_PAD）----
echo "== MT-pad 单变量 A/B（有 pad vs 无 pad；同一二进制，只差 M15_PREFILL_NO_PAD）=="
ab_pair() {
  local a="$HERE/dumps_$1/m26_Pf.gdn" b="$HERE/dumps_$2/m26_Pf.gdn" label="$1 vs $2"
  if [ ! -f "$a/m26_x.bin" ] || [ ! -f "$b/m26_x.bin" ]; then
    echo "-- $label：缺 m26 dump ⇒ 跳过（需 M140_DEVICE=1 先生成）--"; return
  fi
  echo "-- $label --"
  "$PY" - "$a" "$b" <<'PYEOF'
import os, sys
import numpy as np
A, B = sys.argv[1], sys.argv[2]
def rd(p, d):
    q = os.path.join(p, d)
    return np.fromfile(q, dtype="<u1") if os.path.exists(q) else None
for f, dt in [("m26_x.bin", "<u2"), ("m26_logits.bin", "<f4"), ("m26_ids.bin", "<i4"), ("m26_w.bin", "<f4")]:
    a, b = rd(A, f), rd(B, f)
    if a is None or b is None:
        print("  %-16s 缺文件（A=%s B=%s）" % (f, a is not None, b is not None)); continue
    same = np.array_equal(a, b)
    if dt == "<f4":
        af, bf = a.view(np.float32), b.view(np.float32)
        print("  %-16s 逐字节相同=%s  maxAbs(A)=%.4e maxAbs(B)=%.4e max|A-B|=%.4e"
              % (f, same, np.max(np.abs(af)), np.max(np.abs(bf)), np.max(np.abs(af - bf))))
    else:
        print("  %-16s 逐字节相同=%s  n=%d" % (f, same, a.size))
PYEOF
}
ab_pair m1_p15 m1_p15_nopad
ab_pair m1_p12 m1_p12_nopad
echo "-- pad **内容**控制（同样的 memset 尺寸：0x00 vs 0x3C）--"
ab_pair m1_p15 m1_p15_pad3c
ab_pair m1_p12 m1_p12_pad3c

echo "-- repeat-run 方差基线（同档重复跑的 pairwise logits 差；同一二进制）--"
rep_group() {
  local label="$1"; shift
  local dirs="$@"
  "$PY" - "$HERE" "$label" $dirs <<'PYEOF'
import os, sys
import numpy as np
HERE, label = sys.argv[1], sys.argv[2]
names = sys.argv[3:]
base = os.path.join(HERE, "dumps_" + names[0], "m26_Pf.gdn")
ref = np.fromfile(os.path.join(base, "m26_logits.bin"), dtype="<f4") if os.path.exists(
    os.path.join(base, "m26_logits.bin")) else None
if ref is None:
    print("-- %s：缺基准 dump ⇒ 跳过 --" % label); raise SystemExit(0)
print("-- %s（基准 %s，maxAbs=%.4e）--" % (label, names[0], float(np.max(np.abs(ref)))))
for n in names[1:]:
    p = os.path.join(HERE, "dumps_" + n, "m26_Pf.gdn", "m26_logits.bin")
    if not os.path.exists(p):
        print("   %-18s 缺 dump" % n); continue
    a = np.fromfile(p, dtype="<f4")
    print("   %-18s 逐字节相同=%-5s  maxAbs=%.4e  max|diff|=%.4e"
          % (n, np.array_equal(ref.view(np.uint8), a.view(np.uint8)),
             float(np.max(np.abs(a))), float(np.max(np.abs(a - ref)))))
PYEOF
}
rep_group "p12 组" m1_p12 m1_p12_rep2 m1_p12_rep3 m1_p12_nopad m1_p12_pad3c
rep_group "p15 组" m1_p15 m1_p15_rep2 m1_p15_rep3 m1_p15_nopad m1_p15_pad3c

# ---- 把入库的那一份 router pad 拷进各 tag 子目录（m26 判据按目录读它）----
WPAD_SRC="$HERE/dumps_m1_p15/m26_Pf.gdn/m26_wpad.bin"
if [ -f "$WPAD_SRC" ]; then
  for d in "$HERE"/dumps_m1_*/m26_*/ "$HERE"/dumps_m4097_p15/m26_*/; do
    [ -d "$d" ] || continue
    [ -f "$d/m26_wpad.bin" ] || cp "$WPAD_SRC" "$d/m26_wpad.bin" 2>/dev/null || true
  done
fi

echo "== hc 四相位（H1/H2）对拍：m20 的独立 numpy float64 参考（权重面缺则从 manifest+checkpoint 取）=="
for spec in "m1_p1 1" "m1_p3 3" "m1_p7 7" "m1_p15 15" "m1_p8 8"; do
  set -- $spec; name="$1"; ph="$2"
  d="$HERE/dumps_$name"
  [ -d "$d" ] || { echo "-- $name 无 dump，跳过 --"; continue; }
  for tg in "Pf.gdn 0" "Pf.gdn_mut2 1"; do
    set -- $tg; tag="$1"; wantfail="$2"
    echo "-- $name tag $tag phases=$ph $([ "$wantfail" = 1 ] && echo '（期望变红）')--"
    "$PY" "$HERE/check_full_layer.py" "$d" "$tag" 1 "$ph"; echo "rc=$?"
  done
done
if [ -d "$HERE/dumps_m4097_p15" ] && [ -f "$HERE/dumps_m4097_p15/m140_Pf.gdn_h1_hcp.bin" ]; then
  d="$HERE/dumps_m4097_p15"
  echo "-- m4097_p15 tag Pf.gdn phases=15 --"; "$PY" "$HERE/check_full_layer.py" "$d" Pf.gdn 4097 15; echo "rc=$?"
  echo "-- m4097_p15 tag Pf.gdn_mut2 phases=15（期望变红）--"
  "$PY" "$HERE/check_full_layer.py" "$d" Pf.gdn_mut2 4097 15; echo "rc=$?"
else
  echo "-- m4097_p15：活动面 *.bin 不入 git ⇒ 离线跳过（要复算请重跑设备档，见脚本头）--"
fi

echo "== 相位 A（GDN 扫描段）对拍：m23 的 numpy float64 逐句参考（只 m1_p15 档的输入入库）=="
d="$HERE/dumps_m1_p15"
if [ -f "$d/m23_Pf.gdn_h0.bin" ] && [ -f "$d/m23_Pf.gdn_q.bin" ] && [ -f "$d/m23_Pf.gdn_meta.txt" ]; then
  echo "-- m1_p15 clean --"; "$PY" "$REPO/m23_gdn_prefill/check_ref.py" --dir "$d"; echo "rc=$?"
  echo "-- m1_p15 --mutant 1（期望变红）--"
  "$PY" "$REPO/m23_gdn_prefill/check_ref.py" --dir "$d" --mutant 1; echo "rc=$?"
else
  echo "-- m1_p15：缺 m23 输入（h0/q/…）⇒ 离线跳过 --"
fi
if [ -d "$HERE/dumps_m4097_p15" ] && [ -f "$HERE/dumps_m4097_p15/m23_Pf.gdn_q.bin" ]; then
  d="$HERE/dumps_m4097_p15"
  echo "-- m4097_p15 clean --"; "$PY" "$REPO/m23_gdn_prefill/check_ref.py" --dir "$d"; echo "rc=$?"
  echo "-- m4097_p15 --mutant 1（期望变红）--"
  "$PY" "$REPO/m23_gdn_prefill/check_ref.py" --dir "$d" --mutant 1; echo "rc=$?"
else
  echo "-- m4097_p15：缺 m23 输入 ⇒ 离线跳过 --"
fi

echo "== 相位 B（MoE，E=512）对拍：m26 的 numpy 参考（最后一块 tile；按 tag 分子目录）=="
for name in m1_p15 m1_p15_rep2 m1_p15_rep3 m1_p15_nopad m1_p15_pad3c m1_p8 m1_p12 m1_p12_rep2 m1_p12_rep3 m1_p12_nopad m1_p12_pad3c m4097_p15; do
  d="$HERE/dumps_$name"
  [ -d "$d" ] || continue
  for tag in Pf.gdn Pf.gdn_mut1 Pf.gdn_mut2; do
    sub="$d/m26_$tag"
    if [ ! -f "$sub/m26_meta.txt" ] || [ ! -f "$sub/m26_wpad.bin" ] || [ ! -f "$sub/m26_x.bin" ]; then
      echo "-- $name tag $tag：缺 m26 输入（meta/wpad/x）⇒ 离线跳过 --"; continue
    fi
    echo "-- $name tag $tag --"; "$PY" "$REPO/m26_moe_prefill/check_ref.py" "$sub"; echo "rc=$?"
  done
done

echo "== 相位 B 的纯函数性见证（clean vs mut1：比较面 h2_blk 逐字节相同）=="
# 注：h2_blk 是 H2 的输出 —— 它 = 相位 B 的输入面**仅**在跑 H2 的档成立；p8 档相位 B 的实际
# 输入是 pfXDev/m26_x，不是 h2_blk。本段只比 h2_blk 与 moe_y。
for name in m1_p15 m1_p15_rep2 m1_p15_rep3 m1_p15_nopad m1_p15_pad3c m1_p8 m1_p12 m1_p12_rep2 m1_p12_rep3 m1_p12_nopad m1_p12_pad3c m4097_p15; do
  a="$HERE/dumps_$name/m140_Pf.gdn_moe_y.bin"
  b="$HERE/dumps_$name/m140_Pf.gdn_mut1_moe_y.bin"
  x1="$HERE/dumps_$name/m140_Pf.gdn_h2_blk.bin"
  x2="$HERE/dumps_$name/m140_Pf.gdn_mut1_h2_blk.bin"
  if [ -f "$a" ] && [ -f "$b" ]; then
    if [ -f "$x1" ] && [ -f "$x2" ]; then
      cmp -s "$x1" "$x2" && xi=SAME || xi=DIFF
    else
      xi=NA
    fi
    cmp -s "$a" "$b" && yi=SAME || yi=DIFF
    echo "-- $name：h2_blk（H2 出；**仅** H2 开的档才是相位 B 的输入）clean-vs-mut1 = $xi；产出(moe_y) clean-vs-mut1 = $yi --"
  else
    echo "-- $name：缺 moe_y ⇒ 跳过 --"
  fi
done

echo "== dump 的 sha256（入库清单；大 bin 不入 git，见 .gitignore）=="
for d in "$HERE"/dumps_*; do
  [ -d "$d" ] || continue
  ( cd "$d" && { find . -name '*.bin' -o -name '*.txt'; } | sort | xargs -r sha256sum > sha256sums.txt )
  echo "-- $(basename "$d") sha256sums.txt 生成 --"
done
