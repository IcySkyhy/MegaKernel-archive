#!/usr/bin/env python3
"""真实规模（E=512 / topK=10）MoE 段的验收与对拍骨架 —— M90（M77 §5 / §6-mission E 的落地）。

本文件**只**做参考侧与判据侧的事，不碰任何 device 源码，也**不写**任何别人 scope 里的文件。
它回答的问题：

  * 在 E=512 / topK=10 下，MoE 段的哪些输出可以 **T1 逐位判**、哪些只能 **T3 给界**；
  * 哪一条判据能把「路由配错」抓出来；
  * 负向对照的读数是什么；
  * 「非均匀分布 / 可变 active_num（含空专家、单槽专家）」的接口契约在真实规模上是否成立。

三条腿（M77 §5.1）：

1. **`tools/golden/data/real` 真实拓扑档**（m=8 / E=512 / topK=10）——**本 harness 的主腿**。
   该档只入库 `manifest.json`（逐张量 `size_bytes` + `sha256` + 偏差预算），`*.bin` 按需生成
   （`gen_dataset.py --groups real`，~1.25 GiB / ~2 分钟），**不进仓**。`prepare` 子命令把
   「生成 → sha256 对拍 → 不留巨物进仓」这条流程走一遍并打印读数。
2. **`m22_router512/evidence/mode2/nu_*`**（4 个受控非均匀档的**设备 dump**，只读引用）——
   真实 512 专家 + top-10 下的**接口契约**证据：把「由 `topk_ids` 到 counts / slot_base /
   perm_*」这条契约独立复算一遍，并要求非均匀画像（空专家 / 单槽专家）真的出现。
   这四个档正是 M40 自陈未交付的「活跃专家数可变 + 每专家 token 数不均」接口契约测试。
3. **m17 的 numpy/double 参考链**（`m17_moe_real/check_ref.py`，只读 import）——
   量化器与 T3 界的**来源**：`quant_hw`（kernel 的 `MxQuant` 镜像）、`dequant`、
   `tol_bf16` / `tol_l1` / `eps_cube` / `EPS_FAST`。**不新拍 ε**，一律沿用 m17/M60 已归档的推导。

## 参考的输入从哪来（`docs/17` §1.3 三分法；M90 r2 补）

`docs/17-verification-standard.md` §1.3（`grep -n '#### 1.3 参考的输入从哪来' docs/17-verification-standard.md`
命中）要求把参考实现吃的**每个输入**归入三类之一。逐条如下（`prov` 列就是下面这三个标记，
脚本运行时每条判据都打印它）：

| prov | 类别 | 本 harness 里的具体输入 | 合法性 |
|---|---|---|---|
| **①** | 声明输入 | `x.bin` / `x_res.bin` / `router_weight.bin` / `shared_expert_gate_weight.bin` / `experts.*` / `shared_expert.*`（真实档，manifest 有逐张量 `sha256`）；`--model-dir` 的真实 checkpoint | 合法（输入隔离 N1） |
| **②** | 上游输出（**设备**产物） | `m22_router512/evidence/mode2/nu_*/topk_ids.bin`（**设备 dump**） | 合法，**因为**它被点名判据覆盖（见下） |
| **③** | 被判量自身的产物 / 与其共享同一推导链 | **参考链本身只消费 ①**：M90 r2 把它改成从 ① 出发自己往前走（自己的 `x_sorted` gather → 自己的 `quant_hw` 字节 → 自己的 `GU`/`H`/`H_qx`/`Y` → 自己的 `routed`/`shared`/`moe`）。**例外 = 两条判据**：`res2` 与 `y_final` —— 它们的参考输入是被判量侧自己的 `moe_output`（③ 类），其正确性由 `T3 moe` 判据覆盖（满足 §1.3 ③ 的「不得作为唯一判据」） | 标为**非独立性判据**、单列计数；**条数以运行时打印行为准**（`§1.3 ③ 金标链产物（非独立性判据）：N 条`），并与脚本内声明常量 `CHAIN_CRIT_TAGS` + 那条 `§1.3 ③ 漂移自检` 对齐 —— 本 docstring **不复写数字** |

> **口径**：本 docstring 只说「哪一类输入喂给哪一些判据」；**数字一律以运行时打印的那几行为准**
> （`§1.3 ①/②/③ …：N 条` 与 `§1.3 ③ 判据清单` 那一行），文档不复写会漂的数字。
> 主腿 ③ 的**判据 id 集合**由下面这一行机器可读标记承载，`run` 会**真的读它**并与运行时集合、
> `CHAIN_CRIT_TAGS` 三方比对（不一致 ⇒ `§1.3 ③ 漂移自检` FAIL ⇒ `run` 退 1）：
> `<!-- M90-CHAIN-IDS[mainleg]: res2,y_final -->`

* **② 的点名覆盖**：`nu_*` 那 4 个档的参考输入是**设备产生的 `topk_ids`**。覆盖它的判据 =
  **`m22_router512/check_ref.py` 的 `J1`**（脚本里的 `J1` 是 `topk_ids`，见该脚本 docstring 的
  「判定项」段与它实跑打印的 `J1 PASS topk_ids 逐元素相等`；注意 `m22_router512/README.md` §6.2
  那张表的编号不同 —— 表里 `J1` 是 `router_logits`，**引用时必须写清是脚本的 `J1`**）。
  本 harness 在同一趟命令里用 `--w onehot` **只读重跑**了那个 checker 并要求 `rc=0`。
* **③ 的登记**：`inv_slot` 这条在真实档里**没有被判量侧的对应张量**（真实档没有 `inv_slot.bin`），
  所以它既不进 ① 也不进 ② —— 它是**参考自己的自洽契约**，按 §2.1 不计入判定项，已移入
  guard 侧（`T4 结构性：索引/槽位可区分`）。
* 参考侧另有**规则**来源问题（§1.2，见下），与上面的**输入**三分是两件事。

## 规则的来源与「同源转录」caveat（`docs/17` §1.2）

量化字节判据的规则托在 `M17.quant_hw` 与 `tools/golden/moe_block_ref.py::quantize_ocp` 上。
两者都是对**同一份官方头文件**的转写 ⇒ **同源转录（correlated transcription）**，
`docs/17` §1.2 明说它「**仍咬不住共同误读**」。引用时必须带上该节的 caveat：
m17 的规则在 **M60 之前无外部 pin**，M60 补的是**同源转录**（`m17_moe_real/check_ref.py` 的
`W1`–`W6` 见证），**不是独立第二来源**。因此本 harness 只把它算作
「**同规则的两份转写交叉见证**」，**不**声称「两套独立实现」。

同理，主腿是 **golden vs 从 ① 出发的复算**，不是设备对拍：它是**参考链的一致性/结构性检查**，
设备证据面只有 `nu_*` 那 4 个档（②）。

## 判据分档（`docs/17-verification-standard.md` §1.1 的 `| T1 …` / `| T3 …` 两行，逐字口径）

* **T1 整数域/位域**（逐位/逐字节，无例外）：`topk_ids`、`perm_src_token`、`perm_expert`、
  `expert_token_counts`、`expert_offsets`/`slot_base`（前缀和）、
  `x_sorted`（gather）、`A_qx`/`A_scale`/`H_qx`/`H_scale`（routed + 共享）、`res1`、`res2`。
* **T3 长 fp32 链 / 含超越函数近似 / mmad 累加**（`|out − ref| ≤ nulp·ulp(out) + ε·Σ|terms|`，
  逐元素检查；`≤1ulp 比例` / `maxRel` / 位级一致率**降为报告项**）：`router_logits`（含 `Exp`，
  §1.1 触发条件 ①）、`GU` / `Y`（cube 累加，触发条件 ③）、`H` / `shared` / `y_final`（超越函数）、
  `routed`（长 fp32 折叠）、`moe`（combine）。
* **T3 的网格项（`ulp(out)` 那半）取值口径 —— 必须写明是哪一半**（`docs/17` §1.1 该段末条要求
  「写明哪个量 + `e` 是哪套约定」）：
  * `docs/17` §1.1 的 `BfUlp(v) = 2^(e-7)`（`e` = binade 指数）= **一个格点间距**；
  * **参考为实值** ⇒ 取 `0.5·BfUlp`（最大 RNE 舍入误差）；
  * **参考本身已落在 bf16 格点** ⇒ 取 `1.0·BfUlp`（相邻格点对的最大合法差，§1.1 的 M36 实例）。
  * 本 harness 对**device 落盘就是 bf16 的那几段**（`GU`/`H`/`Y`/`routed`/`shared`/`moe`/`y_final`
    及共享专家三条）**先按 device 的序列 `RndBf16` 一次**，于是它们是第二种情形，网格项取
    `1.0·BfUlp`。**为什么不用「实值参考 + 0.5·BfUlp」这一支**：它会在「参考恰好落在两个格点
    中点」的元素上被顶到**占用 1.000**（`docs/17` §1.1 明说这类边界元素本应由 T2′ 吸收），
    而格点对格点的口径按舍入理论有 `|Δ| ≤ 1.0·BfUlp` 的保证。
  * `router_logits` 是 **fp32** 输出（无 bf16 网格）⇒ 网格项 0，只有 `ε_gemv·Σ|a·w|`。
  * m17 的 `spacing_bf16` 给的是 `2^(e-8)` = `0.5·BfUlp`（其 docstring 称「网格步长」，是命名
    与单位差）⇒ 本文件里 `M17.tol_bf16(..., nulp=2.0, ...)` 就是 `1.0·BfUlp`，`nulp=1.0` 就是
    `0.5·BfUlp`。已就这条命名差报塔（`m17_moe_real/**` 不在本 mission 的写权限内，**未改**）。
* **两处必须写明的对账**（不默默改）：
  1. M77 §5.2 把 `topk_weights` 列进 T1（写「bf16 网格」）。它在**值域**上经 `Exp` + 一次除法
     （§1.1 触发条件 ①）⇒ 按 `docs/17` §1.1 属 **T3**；它在**位域**上的对应物才是 T1。
     参考侧因此两样都做：`topk_weights` 走 m17 的 `tol_bf16(ref, 1.0, EPS_FAST·|ref|)`
     （nulp = 1 = 设备把权重落成 bf16 网格），与 `m22_router512/check_ref.py` 的 `J2` 同口径；
     位级一致率进**报告项**。`w_tk_packed` 的位域契约单列一条 T1（两条独立 bf16-RNE 实现互证）。
  2. 真实档**没有** `w_tk_packed` / `inv_slot` / `slot_base` 张量（那是 device 侧对象）。
     本 harness 把它们的**契约**（由参考自己的 `topk_ids` 出发的独立计数排序 / 前缀和 / 行号映射）
     判掉，设备上的**张量级**逐位判据标记为 PENDING（见 `pending` 子命令）。

## FTZ 口径（`docs/17` §7.1 已接受偏差，**规则来源 = 设备观测**）

router path 的 fp32 次正规 FTZ 建模（`tools/golden/moe_block_ref.py::ftz_f32` / `router_topk`
的三个 materialize 点位 FTZ#1/#2/#3）**必须继承**，否则真档上会假红。本 harness 的做法：

* 路由参考一律走 `MB.router_topk`（**已建模 FTZ**）——它就是 §7.1 登记的那份实现；
* 另起一条 **FTZ 关闭**的参考（`router_no_ftz`）做**负向对照**：报出「本档真实数据上有几行
  top-10 集合因此不同」，并用一个**构造的深尾 logits** 用例证明该对照的机理（设备无关）。
* 参考侧口径登记（**已报塔，没有去改 m17**）：`m17_moe_real/check_ref.py` 的路由**没有**建模
  FTZ（它的 `e2e_chain` 与分段判据直接 `np.exp` 后 `argsort`）。本 harness 因此**不**把 m17
  的路由当路由参考，只用它的量化器与 T3 界。

## 负向对照怎么注入（M90 r2 改）

`NC1`–`NC3` 是**产物侧注入**：把**被判量侧**（真实档 golden，充当设备 dump）的对应张量改错，
然后**用同一套判据重跑一遍**（`run_criteria` 复入），把**真的变红的判据逐条列出**。
这样「⇒ 某某判据必红」这句话描述的就是实现的实际接线，不是断言。
`NC4`（FTZ 关闭）与 `NC5`（量化方向反转）是**规则级**反事实，不吃被判量字节，天然是设备无关的。

## 「待 B/C 落地后才能跑」的部分（显式）

融合 kernel 的 MoE 段在 512/10 上的 **device dump 尚不存在**（device 段升级要同步改
`m15_moe_resources.h` 与 `m15_layer_loop.asc`，那两个文件归并行中的 M88）。下列判据**只把
参考侧/判据侧就位**，标记为 PENDING（`pending` 子命令打印）：

* 融合 ws 的整段 sha256 身份 pin（等价 m17 的 `W0`）；
* `A_qx`/`H_qx` 的 **device 字节** vs 参考字节（本 harness 目前比的是 qaware golden 字节
  vs 参考复算字节，属**参考链一致性**见证，不是 device 证据）；
* `GU`/`H`/`Y`/`routed`/`shared`/`moe`/`y_final` 的 **device 值** vs 参考值；
* `res1`/`res2`/`w_tk_packed`/`inv_slot` 的 device 逐位（本 harness 目前验的是该档 golden 自身
  满足这些契约）。

## 用法（每条都在仓库里可复跑；`$PY` = `/usr/local/python3.12.13/bin/python3`）

    $PY tools/golden/moe_real_accept.py prepare        # 生成真实档 + sha256 对拍 + 巨物入仓检查
    $PY tools/golden/moe_real_accept.py run            # 主验收（T1/T3 分档 + nu_* 契约 + 负向对照）
    $PY tools/golden/moe_real_accept.py run --tier m1  # 同一条链在缩形档（E=4/topK=2）上交叉检查
    $PY tools/golden/moe_real_accept.py run --no-m22   # 跳过 nu_* 设备档（m22 evidence 不可读时）
    $PY tools/golden/moe_real_accept.py pending        # 打印待 B/C 落地的判据清单

退出码（三态，与 m17/m22 的约定一致）：
    0 = 判过且通过；1 = 判过且有差异（含负向对照没咬住）；2 = 没得比/输入缺失（打 SKIPPED）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # tools/golden
ROOT = HERE.parent.parent                          # repo 根
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "weights"))

import moe_block_ref as MB  # noqa: E402


def _load_py(name: str, path: Path):
    """按路径 import 一个同名模块（m13/m17 都有 `check_ref.py`，不能靠 sys.path 撞名）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# m17 的参考链：量化器 + T3 界的来源（只读 import，不改 m17_moe_real/**）。
M17 = _load_py("m90_m17_check_ref", ROOT / "m17_moe_real" / "check_ref.py")

HIDDEN, INTER, GROUP = 2560, 640, 32
GUN = 2 * INTER
E_ALL, TOPK, MM = 512, 10, 64
DATA = HERE / "data"
PY = sys.executable
REPORT: list[tuple[str, str]] = []
_RPT_PREFIX = [""]        # 当前报告项前缀（由 cmd_run 按档设置，避免多档时报告项混在一起）


def tier_prefix() -> str:
    return _RPT_PREFIX[0]


# `docs/17` §1.3 的输入溯源三分法标记（每条判据都带一个）
P_DECL = "①"     # 声明输入（host 按 seed 生成 / host dump 的输入张量）
P_UP = "②"       # 上游**设备**产物（仅在点名的判据覆盖它时合法）
P_CHAIN = "③"    # 被判量自身的产物 / 与其共享同一推导链（非独立性判据）
P_CONT = "契约"  # 无被判量侧对应物的自洽契约（按 §2.1 不计入判定项）

# 三处 §1.3 表格（本 docstring / `tools/golden/README.md` / `m15_layer_loop/check_moe_ref.py`
# 的 docstring）是**同一份内容的三次转写**，会漂。守卫分两层，两层都会 FAIL：
#   ① 代码层：运行时被标 `P_CHAIN` 的判据 id（`CHAIN_SEEN`）↔ 声明常量 `CHAIN_CRIT_TAGS`；
#   ② 文档层：`cmd_run` 会**真的去读那三个文件**，解析里面的机器可读标记（HTML 注释形式，
#      每份文件**恰好一处**；缺失 / 名字写错 / 重复 / 作用域写错都判 FAIL 并让 `run` 退 1）。
#   ⇒ 「③ 的**判据 id 集合**」在代码↔常量↔三处文档之间是**自动**同步的；
#     标记周围的**散文**（句子怎么描述）仍是人写的，不在这条守卫的范围内（如实写在 README 的已知边界里）。
CHAIN_CRIT_TAGS = ("res2", "y_final")
CHAIN_SEEN: list[str] = []      # 运行时被标 ③ 的判据 id（由带 crit= 的注册点登记）
# 标记的 ASCII 正则（字符类只用 ASCII）：HTML 注释形式的
# `<!-- M90-CHAIN-IDS[<scope>]: <id>,<id>,... -->`；`-` 表示空集。
# **只认 HTML 注释形式** ⇒ 正文里散字提及标记不会被误当成标记。
CHAIN_MARK_RE = re.compile(r"<!--\s*M90-CHAIN-IDS\[([a-z]+)\]:\s*([A-Za-z0-9_,-]+)\s*-->")
CHAIN_DOCS = (("tools/golden/moe_real_accept.py", "mainleg"),
              ("tools/golden/README.md", "mainleg"),
              ("m15_layer_loop/check_moe_ref.py", "devpath"))


def doc_chain_ids() -> dict[str, list[str] | None]:
    """读 `CHAIN_DOCS` 列出的三处文档，取 HTML 注释形式的标记 `M90-CHAIN-IDS[<scope>]: <ids>`
    里的 id 集合（下面写作「标记」；此处不写注释记号的字面量，免得函数自己的文档被当成标记）。

    严格：每份文件必须**恰好一处**该作用域的标记（0 处 = 被删、>1 处 = 重复、出现**别的作用域**
    = 串台）⇒ 一律返回 `None`，调用点判 FAIL。`-` 表示空集。
    只读文本、不 import 被读文件，所以对 `m15_layer_loop/check_moe_ref.py`（别的 mission 的文件）
    也是纯只读。
    """
    out: dict[str, list[str] | None] = {}
    for rel, scope in CHAIN_DOCS:
        p = ROOT / rel
        try:
            txt = p.read_text(encoding="utf-8")
        except OSError:
            out[rel] = None
            continue
        hits = CHAIN_MARK_RE.findall(txt)
        if len(hits) != 1 or hits[0][0] != scope:
            out[rel] = None      # 缺失 / 重复 / 作用域不符 ⇒ 判 FAIL
        else:
            out[rel] = sorted({x for x in hits[0][1].split(",") if x and x != "-"})
    return out


# ---------------------------------------------------------------------------
# 判据登记（判定项 / guard / 报告项分栏；docs/17 §2）
# ---------------------------------------------------------------------------
class Verdicts:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool, str]] = []   # (tier, prov, tag, ok, info)

    def add(self, tier: str, tag: str, ok: bool, info: str = "", prov: str = P_DECL,
            crit: str | None = None) -> None:
        self.rows.append((tier, prov, tag, bool(ok), info))
        if prov == P_CHAIN and crit:
            CHAIN_SEEN.append(crit)

    def judged(self) -> list[tuple[str, str, str, bool, str]]:
        """判定项 = T1/T3/NC（不含 GUARD）。"""
        return [r for r in self.rows if r[0] != "GUARD"]

    def nbad(self) -> int:
        return sum(1 for _, _, _, ok, _ in self.judged() if not ok)

    def by(self, key: int) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for r in self.judged():
            k = r[key]
            n, b = out.get(k, (0, 0))
            out[k] = (n + 1, b + (0 if r[3] else 1))
        return out


def t1_exact(V: Verdicts, tag: str, got: np.ndarray, want: np.ndarray,
             what: str = "元素", prov: str = P_DECL, crit: str | None = None) -> None:
    """T1：逐位/逐字节（`docs/17` §1.1 第一行，无例外）。"""
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        V.add("T1", tag, False, f"shape {got.shape} vs {want.shape}", prov, crit)
        return
    nbad = int((got != want).sum())
    V.add("T1", tag, nbad == 0, f"{nbad}/{got.size} {what}不符; n={got.size}", prov, crit)


def t3_bound(V: Verdicts, tag: str, got: np.ndarray, ref: np.ndarray, budget,
             noise_note: str, prov: str = P_DECL, crit: str | None = None) -> None:
    """T3：`|out − ref| ≤ budget`，逐元素；budget 由调用点按推导式给出（**不是**实测最大值）。"""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        V.add("T3", tag, False, f"shape {got.shape} vs {ref.shape}", prov, crit)
        return
    b = np.broadcast_to(np.asarray(budget, dtype=np.float64), got.shape).astype(np.float64)
    floor = 1e-6 * float(np.abs(ref).max()) if ref.size else 0.0
    b = np.maximum(b, max(floor, 1e-30))
    ad = np.abs(got - ref)
    nonfin = ~np.isfinite(got) | ~np.isfinite(ref)
    nbad = int(nonfin.sum() + np.sum(~nonfin & (ad > b)))
    with np.errstate(invalid="ignore"):
        ratio = float(np.nanmax(np.where(nonfin, np.inf, ad / b)))
    q = np.percentile(ad, (50, 90, 99)) if ad.size else np.zeros(3)
    V.add("T3", tag, nbad == 0,
          f"n={got.size} 越界={nbad}（非有限 {int(nonfin.sum())}）| |Δ| p50={q[0]:.3e} p90={q[1]:.3e} "
          f"p99={q[2]:.3e} max={ad.max() if ad.size else 0.0:.3e} | 最差占预算={ratio:.3f} | {noise_note}",
          prov, crit)


def spacing_bf16(x: np.ndarray) -> np.ndarray:
    """**m17 口径**的网格量 `2^(floor(log2|x|)-8)`（m17 的 `spacing_bf16`）。

    ⚠ 单位说明（引用时必须写明，`docs/17` §1.1 该段末条明确要求）：按 `docs/17` §1.1 的定义
    **`BfUlp(v) = 2^(e-7)`**（`e` = binade 指数，`2^e ≤ |v| < 2^(e+1)`）= **一个格点间距**；
    本函数给的 `2^(e-8)` = **0.5·BfUlp** = **最大 RNE 舍入误差**（半个格点）。
    m17 的 docstring 把它写成「网格步长」，实际是后者 —— 好在 m17 的调用点一律 `nulp=1.0`，
    那正好等于 `docs/17` §1.1 对「参考为实值」档的默认项 `0.5·BfUlp`，所以 m17 的分段判据是合规的。
    **参考本身已在 bf16 格点上**时 `docs/17` §1.1 要求取 `1.0·BfUlp` ⇒ 本文件用 `bf_ulp`（下一个函数）。
    已就这条命名/单位差报塔（不在本 mission 的写权限内，**未改** m17）。
    """
    return M17.spacing_bf16(np.asarray(x, dtype=np.float64))


def bf_ulp(x: np.ndarray) -> np.ndarray:
    """`docs/17` §1.1 的格点间距口径：`BfUlp(v) = 2^(e-7)`，`e` = binade 指数。

    逐字依据：`docs/17-verification-standard.md` 的「**`BfUlp(v)` 的真值 = `2^(e-7)`**」条
    （`grep -n 'BfUlp' docs/17-verification-standard.md` 命中）。 = 2 × `spacing_bf16`。
    """
    a = np.abs(np.asarray(x, dtype=np.float64))
    return np.where(a > 0, 2.0 ** (np.floor(np.log2(np.where(a > 0, a, 1.0))) - 7.0), 0.0)


def bf16_rne_bits(x: np.ndarray) -> np.ndarray:
    """**独立**的 bf16 RNE 落格（用 fp32 位域做 round-half-to-even）——与
    `MB.f32_to_bf16_bits` 互为两条实现，用来把「w_tk_packed 的低 16 位 = bf16(权重)」这条契约
    钉在**两条独立代码路径**上，而不是同一条算两遍。"""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> np.uint64(16)) & np.uint64(1)
    b = (b + np.uint64(0x7FFF) + lsb) >> np.uint64(16)
    return (b & np.uint64(0xFFFF)).astype(np.uint16)


# ---------------------------------------------------------------------------
# 真实档的装载 / 校验
# ---------------------------------------------------------------------------
def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def verify_tier(tier_dir: Path, verbose: bool = True) -> tuple[int, int, int]:
    """按入库 manifest 的 `size_bytes` + `sha256` 逐张量核对盘上的 `*.bin`。

    返回 (核对张量数, 缺失数, 不符数)。缺任何一张都直接报缺失（不静默跳过）。
    """
    checked = missing = bad = 0
    for rel in ("manifest.json", "qaware/floor/manifest.json", "qaware/ceil/manifest.json"):
        mp = tier_dir / rel
        if not mp.exists():
            if verbose:
                print(f"[verify] 缺 manifest {mp.relative_to(tier_dir)}")
            missing += 1
            continue
        man = json.loads(mp.read_text())
        base = mp.parent
        for d in man["files"]:
            fp = base / d["tensor"]
            if not fp.exists():
                missing += 1
                if verbose:
                    print(f"[verify] 缺 {fp.relative_to(tier_dir)}")
                continue
            if fp.stat().st_size != int(d["size_bytes"]):
                bad += 1
                if verbose:
                    print(f"[verify] 大小不符 {d['tensor']}: {fp.stat().st_size} != {d['size_bytes']}")
                continue
            got = sha256_file(fp)
            checked += 1
            if got != d["sha256"]:
                bad += 1
                if verbose:
                    print(f"[verify] sha256 不符 {d['tensor']}: {got} != {d['sha256']}")
    return checked, missing, bad


def load_tier(tier: str = "real", rule: str = "floor"):
    td = DATA / tier
    man = json.loads((td / "manifest.json").read_text())
    tens = {d["tensor"]: MB.read_bin(td / d["tensor"], d) for d in man["files"]}
    qd = td / "qaware" / rule
    qman = json.loads((qd / "manifest.json").read_text())
    # 键去掉 `.bin` 后缀（qaware 的 tensor 字段带后缀，顶层同名字段也带）
    q = {d["tensor"][:-4] if d["tensor"].endswith(".bin") else d["tensor"]:
         MB.read_bin(qd / d["tensor"], d) for d in qman["files"]}
    return man, tens, qman, q


# ---------------------------------------------------------------------------
# 负向对照的工具（全部在内存里做变异；不落盘、不碰 device 源码）
# ---------------------------------------------------------------------------
def router_no_ftz(x: np.ndarray, w: np.ndarray, top_k: int):
    """**不建模 FTZ** 的路由：与 `MB.router_topk` 同语义，只去掉三处 `ftz_f32`。

    这是 `docs/17` §7.1 那条已接受偏差的**反事实**：拿它当参考就会与设备分叉。
    """
    logits = x @ w.T
    logits = logits - logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    order = np.argsort(-scores, axis=1, kind="stable")
    ids = order[:, :top_k].astype(np.int32)
    rows = np.arange(x.shape[0])[:, None]
    wts = scores[rows, ids].astype(np.float32)
    wts = wts / wts.sum(axis=1, keepdims=True)
    return logits.astype(np.float32), ids, wts.astype(np.float32)


def slot_contract(ids: np.ndarray, num_experts: int) -> dict:
    """由 `topk_ids` **独立**复算槽位契约（计数排序 / 前缀和 / compact 与 padded 两种行号）。

    两种 `inv_slot` 口径都算出来，好让调用点用**读数**（不是假设）区分 m15 的紧凑口径与
    m17 的 padded 口径。
    """
    ids = np.asarray(ids, dtype=np.int64)
    m, k = ids.shape
    flat = ids.reshape(-1)
    src = np.tile(np.arange(m, dtype=np.int64)[:, None], (1, k)).reshape(-1)
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=num_experts).astype(np.int64)
    offsets = np.zeros(num_experts + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = int(counts.sum())
    pos = np.empty(total, dtype=np.int64)
    pos[order] = np.arange(total, dtype=np.int64)
    inv_compact = pos.reshape(m, k)
    inv_padded = np.zeros((m, k), dtype=np.int64)
    for t in range(m):
        for kk in range(k):
            e = int(ids[t, kk])
            inv_padded[t, kk] = e * MM + (int(inv_compact[t, kk]) - int(offsets[e]))
    return {"perm_src_token": src[order].astype(np.int32),
            "perm_expert": flat[order].astype(np.int32),
            "expert_token_counts": counts.astype(np.int32),
            "expert_offsets": offsets.astype(np.int32),
            "inv_slot_compact": inv_compact.astype(np.int32),
            "inv_slot_padded": inv_padded.astype(np.int32),
            "total": total}


# ---------------------------------------------------------------------------
# 主验收：一条**从声明输入出发**的参考链，对金标逐段比对
# ---------------------------------------------------------------------------
def run_criteria(man: dict, T: dict, Q: dict, V: Verdicts, quiet: bool = False) -> None:
    """把真实档的**全部判据**跑一遍。`T`/`Q` 可被调用方替换（负向对照就是这么注入的：
    换一份被判量侧的张量，再调用本函数，看**哪些判据真的变红**）。

    参考链的输入**只有 ①**：`x.bin` / `x_res.bin` / 权重（含 checkpoint 布局的 MXFP4 字节）；
    中间量（`x_sorted` gather、A/H 量化字节、`GU`/`H`/`Y`、`routed`/`shared`/`moe`/`y_final`）
    全部由参考自己算出来 —— **不取金标链的任何中间产物**（`docs/17` §1.3 的 ③ 在本腿为 0 条）。
    """
    E = int(man["num_experts"])
    K = int(man["top_k"])
    m = int(man["m"])
    eps_norm = float(man["layer_chain"]["norm_eps"])
    if not quiet:
        print(f"[accept] m={m} E={E} topK={K} rule=floor (seed={man['seed']}, "
              f"hidden={man['hidden']}, moe_inter={man['moe_intermediate']})")

    x = np.asarray(T["x.bin"], dtype=np.float32)                        # [m, HIDDEN] bf16 网格 ①
    rw = np.asarray(T["router_weight.bin"], dtype=np.float32)           # [E, HIDDEN]          ①
    x_res = np.asarray(T["x_res.bin"], dtype=np.float32)                                   # ①
    gamma2 = np.asarray(T["gamma2.bin"], dtype=np.float32)                                 # ①
    sgw = np.asarray(T["shared_expert_gate_weight.bin"], dtype=np.float32)                 # ①

    ids_g = np.asarray(T["topk_ids.bin"], dtype=np.int64)               # 被判量侧
    counts_g = np.asarray(T["expert_token_counts.bin"], dtype=np.int64)  # 被判量侧
    x = x[:ids_g.shape[0]]
    x_res = x_res[:ids_g.shape[0]]

    # ---- ulp 口径并排打印（docs/17 §1.1 该段末条要求「写明哪个量 + e 是哪套约定」）----
    s_probe = float(np.abs(np.asarray(T["router_logits.bin"], dtype=np.float32)).max())
    if not quiet:
        REPORT.append(("ulp 口径并排（报告项；docs/17 §1.1 要求写明量 + e 约定）",
                       f"以 |v|≈{s_probe:.4g} 为例：BfUlp = 2^(e-7) = "
                       f"{float(bf_ulp(np.array([s_probe]))[0]):.3e}（格点间距，docs/17 §1.1）；"
                       f"m17 spacing_bf16 = 2^(e-8) = {float(spacing_bf16(np.array([s_probe]))[0]):.3e}"
                       f"（= 0.5·BfUlp = 最大 RNE 舍入误差，m17 的 docstring 称「网格步长」）"))

    # ---------------- GUARD：非空洞性（docs/17 §4；0 vs 0 的空过比没有判据更糟）--------
    nz_x = int((x != 0).sum())
    span_x = float(x.max() - x.min())
    V.add("GUARD", "① 输入 x 非全零且非常数（非空洞前提）", nz_x > 0 and span_x > 1e-6,
          f"非零 {nz_x}/{x.size}；span={span_x:.3e}", P_DECL)
    n_active = int((counts_g > 0).sum())
    n_empty = int((counts_g == 0).sum())
    n_single = int((counts_g == 1).sum())
    V.add("GUARD", "① 被判量侧路由非退化（活跃专家数可变 / 含空专家与单槽专家）",
          n_active > 0 and n_active < E and n_empty > 0 and n_single > 0
          and int(counts_g.sum()) == m * K,
          f"活跃 {n_active}/{E}、空 {n_empty}、单槽 {n_single}、最大槽 {int(counts_g.max())}、"
          f"Σt_e={int(counts_g.sum())}（= m·topK = {m * K}）", P_DECL)
    V.add("GUARD", "① 层输入 x_res 非全零（res1 判据的非空洞前提）",
          int((x_res != 0).sum()) > 0, f"非零 {int((x_res != 0).sum())}/{x_res.size}", P_DECL)
    if not quiet:
        REPORT.append((f"{tier_prefix()}被参考链消费的金标链中间产物条数（报告项；§1.3 的 ③ 计数）",
                       "参考链只吃 ①（x/x_res/权重）；③ = 0 条"))

    # ---------------- S2 路由：参考 = MB.router_topk（docs/17 §7.1 的 FTZ 实现）--------
    logits_g = np.asarray(T["router_logits.bin"], dtype=np.float32)[:m]
    wts_g = np.asarray(T["topk_weights.bin"], dtype=np.float32)[:m]
    _lg, ids, wts = MB.router_topk(x, rw, K)
    t1_exact(V, "T1 topk_ids（FTZ 由 MB.router_topk 建模）", ids, ids_g, "id")
    x64, w64 = x.astype(np.float64), rw.astype(np.float64)
    lg_ref = x64 @ w64.T
    lg_ref = lg_ref - lg_ref.max(axis=1, keepdims=True)
    l1 = np.abs(x64) @ np.abs(w64).T
    t3_bound(V, "T3 router_logits（含 Exp/max-shift）", logits_g, lg_ref,
             M17.tol_l1(lg_ref, l1, M17.EPS_GEMV), "ε_gemv·Σ|a·w|（m17 EPS_GEMV）")
    e_ref = np.exp(lg_ref)
    order = np.argsort(-e_ref, axis=1, kind="stable")
    ew = np.take_along_axis(e_ref, order[:, :K], axis=1)
    w_ref = ew / ew.sum(axis=1, keepdims=True)
    t3_bound(V, "T3 topk_weights（含 Exp + 除法 ⇒ §1.1 触发①）",
             wts_g, w_ref, M17.tol_bf16(w_ref, 1.0, M17.EPS_FAST * np.abs(w_ref)),
             "0.5·BfUlp + EPS_FAST·|ref|（m17 S2 口径）")
    # T1 位域契约：w_tk_packed 的低 16 位 = bf16 RNE(权重) —— 两条独立 RNE 实现互证（吃参考自己的 wts）
    t1_exact(V, "T1 bf16 RNE 落格两条独立实现互证（w_tk_packed 位域契约）",
             MB.f32_to_bf16_bits(wts), bf16_rne_bits(wts), "bit")
    d_w = np.abs(MB.f32_to_bf16_bits(wts_g).astype(np.int64) -
                 MB.f32_to_bf16_bits(wts).astype(np.int64))
    if not quiet:
        REPORT.append((f"{tier_prefix()}topk_weights 位级一致率（报告项；docs/17 §1.1 降档量）",
                       f"{int((d_w == 0).sum())}/{d_w.size} 位级一致，最大位距 {int(d_w.max())}"))

    # ---------------- S3 索引链：参考自己从 ① 路由结果推（不取金标 ids）----------------
    c = slot_contract(ids, E)
    t1_exact(V, "T1 expert_token_counts", c["expert_token_counts"], counts_g, "count")
    t1_exact(V, "T1 perm_src_token", c["perm_src_token"],
             np.asarray(T["perm_src_token.bin"], dtype=np.int32), "slot")
    t1_exact(V, "T1 perm_expert", c["perm_expert"],
             np.asarray(T["perm_expert.bin"], dtype=np.int32), "slot")
    off = c["expert_offsets"]
    V.add("T1", "T1 expert_offsets/slot_base 前缀和自洽 vs 被判量 counts",
          bool(np.array_equal(off[1:] - off[:-1], counts_g) and off[E] == int(counts_g.sum())
               and off[0] == 0),
          f"offsets[0]={int(off[0])}、offsets[E]={int(off[E])}、Σcounts={int(counts_g.sum())}、"
          f"逐段差与 counts 不符 {int((off[1:] - off[:-1] != counts_g).sum())}/{E}")
    V.add("GUARD", "① 被判量 topk_ids 值域合法且行内互异",
          bool(np.all((ids_g >= 0) & (ids_g < E))
               and all(len(set(row.tolist())) == K for row in ids_g)),
          f"值域合法、行内互异 = 全部 {m} 行", P_DECL)
    # T4 结构性：索引/槽位可区分（compact 行号是 [0,Σt_e) 的双射）。**参考自己的自洽契约**，
    # 真实档没有被判量侧的 `inv_slot` 张量 ⇒ 按 docs/17 §2.1 不计入判定项（列在 guard 侧）。
    V.add("GUARD", "契约 inv_slot 是 [0,Σt_e) 上的双射（参考自洽；真实档无 inv_slot 张量，不计判定）",
          bool(np.array_equal(np.sort(c["inv_slot_compact"].reshape(-1)),
                              np.arange(c["total"]))),
          f"compact 双射 {c['total']} 个；padded 口径 = e·M_MAX+局部行号；两口径在 E={E} 下"
          f"{'逐元素相同（巧合档）' if np.array_equal(c['inv_slot_compact'], c['inv_slot_padded']) else '不同'}"
          f"（m15 紧凑寻址用 compact，m17 用 padded）", P_CONT)

    # ---------------- S4 permute：参考自己 gather（①）----------------
    x_bits = MB.f32_to_bf16_bits(x)
    xs_bits_mine = np.ascontiguousarray(x_bits[c["perm_src_token"]])
    xs_g = np.asarray(T["x_sorted.bin"], dtype=np.float32)
    t1_exact(V, "T1 x_sorted == gather(x, perm_src)",
             MB.f32_to_bf16_bits(xs_g), xs_bits_mine, "bf16")
    xs_mine = MB.bf16_bits_to_f32(xs_bits_mine).astype(np.float32)     # 参考自己的 x_sorted（①）
    xs32 = np.ascontiguousarray(xs_mine)

    # ---------------- S5 参考自己的 A 量化字节，对金标逐字节（T1）----------------
    total = c["total"]
    aq_g = np.asarray(Q["a_qx"], dtype=np.uint8)
    asc_g = np.asarray(Q["a_scale"], dtype=np.uint8)
    pq, sq = M17.quant_hw(xs32)                       # 参考自己的 A 侧字节（compact 行序）
    t1_exact(V, "T1 A_qx/A_scale 逐字节 == golden(floor 序列)",
             np.concatenate([pq.ravel(), sq.ravel()]),
             np.concatenate([aq_g[:, :HIDDEN // 2].ravel(), asc_g[:, :HIDDEN // GROUP].ravel()]),
             "byte")
    pq_o, sq_o = MB.quantize_ocp(xs32)
    t1_exact(V, "T1 quant_hw == quantize_ocp on A 行（同规则两份转写交叉见证）",
             np.concatenate([pq.ravel(), sq.ravel()]),
             np.concatenate([pq_o.ravel(), sq_o.ravel()]), "byte")
    aq_mine, asc_mine = pq, sq

    # ---------------- S6/S7/S8 参考自己的整条 w4a4 链（T3 界；中间量全自算）--------
    gu_g = np.asarray(Q["gu"], dtype=np.float32)
    h_g = np.asarray(Q["h_swiglu"], dtype=np.float32)
    ys_g = np.asarray(Q["y_sorted"], dtype=np.float32)
    hq_g = np.asarray(Q["h_qx"], dtype=np.uint8)
    hs_g = np.asarray(Q["h_scale"], dtype=np.uint8)
    e_pack = np.asarray(T["experts.gate_up_proj.bin"], dtype=np.uint8)
    e_scan = np.asarray(T["experts.gate_up_proj.weight_scale.bin"], dtype=np.uint8)
    d_pack = np.asarray(T["experts.down_proj.bin"], dtype=np.uint8)
    d_scan = np.asarray(T["experts.down_proj.weight_scale.bin"], dtype=np.uint8)
    s_pack = {k: np.asarray(T[f"shared_expert.{k}_proj.bin"], dtype=np.uint8)
              for k in ("gate", "up", "down")}
    s_scan = {k: np.asarray(T[f"shared_expert.{k}_proj.weight_scale.bin"], dtype=np.uint8)
              for k in ("gate", "up", "down")}
    hq_mine = np.zeros((total, INTER // 2), dtype=np.uint8)
    hs_mine = np.zeros((total, INTER // GROUP), dtype=np.uint8)
    y_mine = np.zeros((total, HIDDEN), dtype=np.float64)
    bad_gu = bad_h = bad_y = 0
    n_gu = n_h = n_y = 0
    ratio = {"GU": 0.0, "H": 0.0, "Y": 0.0}
    for e in range(E):
        t = int(c["expert_token_counts"][e])
        if t == 0:
            continue
        sl = slice(int(off[e]), int(off[e]) + t)
        adq = M17.dequant(aq_mine[sl], asc_mine[sl], t, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
        wgu = M17.dequant(e_pack[e], e_scan[e], GUN, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
        gu_ref = M17.bf16_round(adq @ wgu.T)
        b = M17.tol_bf16(gu_ref, 2.0, M17.eps_cube(HIDDEN) * (np.abs(adq) @ np.abs(wgu).T))
        d = np.abs(gu_g[sl].astype(np.float64) - gu_ref)
        bad_gu += int((d > b).sum())
        n_gu += t * GUN
        ratio["GU"] = max(ratio["GU"], float((d / b).max()))
        gd = gu_ref[:, :INTER]
        ud = gu_ref[:, INTER:]
        h_ref = M17.bf16_round((gd / (1.0 + np.exp(-gd))) * ud)
        bh = M17.tol_bf16(h_ref, 2.0, M17.EPS_FAST * np.abs(h_ref))
        dh = np.abs(h_g[sl].astype(np.float64) - h_ref)
        bad_h += int((dh > bh).sum())
        n_h += t * INTER
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio["H"] = max(ratio["H"], float((dh / bh).max()))
        pqh, sqh = M17.quant_hw(np.ascontiguousarray(h_ref.astype(np.float32)))
        hq_mine[sl] = pqh
        hs_mine[sl, :INTER // GROUP] = sqh
        hd = M17.dequant(pqh, sqh, t, INTER, INTER // 2, INTER // GROUP)
        wd = M17.dequant(d_pack[e], d_scan[e], HIDDEN, INTER, INTER // 2, INTER // GROUP)
        y_ref = M17.bf16_round(hd @ wd.T)
        y_mine[sl] = y_ref
        by = M17.tol_bf16(y_ref, 2.0, M17.eps_cube(INTER) * (np.abs(hd) @ np.abs(wd).T))
        dy = np.abs(ys_g[sl].astype(np.float64) - y_ref)
        bad_y += int((dy > by).sum())
        n_y += t * HIDDEN
        ratio["Y"] = max(ratio["Y"], float((dy / by).max()))
    V.add("T3", "T3 GU 逐元素界（cube 累加；格点对格点 1.0·BfUlp + ε_cube(2560)·Σ|a·w|）",
          bad_gu == 0, f"n={n_gu} 越界={bad_gu}；最差占预算={ratio['GU']:.3f}", P_DECL)
    V.add("T3", "T3 H(swiglu) 逐元素界（超越函数；格点对格点 + fast-math 噪声）",
          bad_h == 0, f"n={n_h} 越界={bad_h}；最差占预算={ratio['H']:.3f}", P_DECL)
    V.add("T3", "T3 Y 逐元素界（cube 累加；格点对格点 1.0·BfUlp + ε_cube(640)·Σ|a·w|）",
          bad_y == 0, f"n={n_y} 越界={bad_y}；最差占预算={ratio['Y']:.3f}", P_DECL)
    t1_exact(V, "T1 H_qx/H_scale 逐字节 == golden(floor 序列)（参考自算的 H 上的量化字节）",
             np.concatenate([hq_mine.ravel(), hs_mine.ravel()]),
             np.concatenate([hq_g[:, :INTER // 2].ravel(), hs_g[:, :INTER // GROUP].ravel()]),
             "byte")

    # 共享专家：参考自己从 x（①）走完
    aqs_g = np.asarray(Q["a_qx_shd"], dtype=np.uint8)
    ass_g = np.asarray(Q["a_scale_shd"], dtype=np.uint8)
    hqs_g = np.asarray(Q["h_qx_shd"], dtype=np.uint8)
    hss_g = np.asarray(Q["h_scale_shd"], dtype=np.uint8)
    gu_s_g = np.asarray(Q["gu_shd"], dtype=np.float32)
    hs_shd_g = np.asarray(Q["h_swiglu_shd"], dtype=np.float32)
    y_s_g = np.asarray(Q["y_shd"], dtype=np.float32)
    pqs, sqs = M17.quant_hw(np.ascontiguousarray(x.astype(np.float32)))
    t1_exact(V, "T1 A_qx_shd/A_scale_shd 逐字节（共享专家的 A 侧取 x）",
             np.concatenate([pqs.ravel(), sqs.ravel()]),
             np.concatenate([aqs_g[:, :HIDDEN // 2].ravel(), ass_g[:, :HIDDEN // GROUP].ravel()]),
             "byte")
    adq_s = M17.dequant(pqs, sqs, m, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
    wg = M17.dequant(s_pack["gate"], s_scan["gate"], INTER, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
    wu = M17.dequant(s_pack["up"], s_scan["up"], INTER, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
    wgu_s = np.concatenate([wg, wu], axis=0)
    gu_s_ref = M17.bf16_round(adq_s @ wgu_s.T)
    t3_bound(V, "T3 GU_shd（参考自算；真实共享专家）", gu_s_g, gu_s_ref,
             M17.tol_bf16(gu_s_ref, 2.0, M17.eps_cube(HIDDEN) * (np.abs(adq_s) @ np.abs(wgu_s).T)),
             "1.0·BfUlp + ε_cube(2560)·Σ|a·w|")
    h_s_ref = M17.bf16_round((gu_s_ref[:, :INTER] / (1.0 + np.exp(-gu_s_ref[:, :INTER])))
                             * gu_s_ref[:, INTER:])
    t3_bound(V, "T3 H_shd（参考自算；超越函数）", hs_shd_g, h_s_ref,
             M17.tol_bf16(h_s_ref, 2.0, M17.EPS_FAST * np.abs(h_s_ref)), "1.0·BfUlp + EPS_FAST·|ref|")
    pqhs, sqhs = M17.quant_hw(np.ascontiguousarray(h_s_ref.astype(np.float32)))
    t1_exact(V, "T1 H_qx_shd/H_scale_shd 逐字节（参考自算的 H_shd 上）",
             np.concatenate([pqhs.ravel(), sqhs.ravel()]),
             np.concatenate([hqs_g[:, :INTER // 2].ravel(), hss_g[:, :INTER // GROUP].ravel()]),
             "byte")
    hd_s = M17.dequant(pqhs, sqhs, m, INTER, INTER // 2, INTER // GROUP)
    wd_s = M17.dequant(s_pack["down"], s_scan["down"], HIDDEN, INTER, INTER // 2, INTER // GROUP)
    y_s_ref = M17.bf16_round(hd_s @ wd_s.T)
    t3_bound(V, "T3 Y_shd（参考自算；cube 累加）", y_s_g, y_s_ref,
             M17.tol_bf16(y_s_ref, 2.0, M17.eps_cube(INTER) * (np.abs(hd_s) @ np.abs(wd_s).T)),
             "1.0·BfUlp + ε_cube(640)·Σ|a·w|")

    # ---------------- S9 combine：参考自己算（T3）----------------
    routed_g = np.asarray(Q["routed_output"], dtype=np.float32)[:m]
    shared_g = np.asarray(Q["shared_output"], dtype=np.float32)[:m]
    moe_g = np.asarray(Q["moe_output"], dtype=np.float32)[:m]
    wb = MB.bf16_bits_to_f32(MB.f32_to_bf16_bits(wts).astype(np.uint16)).astype(np.float64)
    routed_ref = np.zeros((m, HIDDEN), dtype=np.float64)
    l1r = np.zeros((m, HIDDEN), dtype=np.float64)
    for t in range(m):
        for k in range(K):
            yy = y_mine[int(c["inv_slot_compact"][t, k])]
            routed_ref[t] += wb[t, k] * yy
            l1r[t] += abs(float(wb[t, k])) * np.abs(yy)
    routed_ref = M17.bf16_round(routed_ref)
    br = M17.tol_bf16(routed_ref, 2.0, M17.EPS_FOLD * l1r)
    t3_bound(V, "T3 routed（加权折叠；格点对格点 1.0·BfUlp + ε_fold·Σ|w·Y|）",
             routed_g, routed_ref, br, "1.0·BfUlp + EPS_FOLD·Σ|w·Y|")
    ur = np.abs(routed_g.astype(np.float64) - routed_ref) / br
    it = np.unravel_index(int(np.argmax(ur)), ur.shape)
    if not quiet:
        REPORT.append((f"{tier_prefix()}routed 预算占用最差元素（报告项）",
                       f"idx={it} ref={routed_ref[it]:.6e} dev={float(routed_g[it]):.6e} "
                       f"|Δ|={abs(float(routed_g[it]) - routed_ref[it]):.3e} budget={br[it]:.3e} "
                       f"占用={ur[it]:.3f}；Σ|w·Y|={l1r[it]:.3e}"))
    sgl_mine = (x64 @ sgw.reshape(-1).astype(np.float64))[:, None]
    gv = 1.0 / (1.0 + np.exp(-sgl_mine))
    gated_raw = gv * y_s_ref
    shared_ref = M17.bf16_round(gated_raw)
    t3_bound(V, "T3 shared（参考自算的 sigmoid 门控；格点对格点 1.0·BfUlp + fast-math 噪声）",
             shared_g, shared_ref,
             M17.tol_bf16(shared_ref, 2.0, M17.EPS_FAST * np.abs(shared_ref)),
             "1.0·BfUlp + EPS_FAST·|ref|")
    # 设备 combine（m13/m17 段序）：moe = bf16(routed + **未取整的** gated)；而 qaware golden 用的是
    # **取整后的** shared（`qaware_ref` 的 `moe_output = f32_to_bf16(shared_output + routed_output)`）。
    # 两套 combine 约定的差被显式建模为一项 `1.0·BfUlp(max(|shared|,|moe|))`（推导量，不是拟合）。
    moe_ref = M17.bf16_round(routed_ref + gated_raw)
    extra = bf_ulp(np.maximum(np.abs(shared_ref), np.abs(moe_ref)))
    t3_bound(V, "T3 moe（combine；含 golden 取整 shared 的约定差项）", moe_g, moe_ref,
             M17.tol_bf16(moe_ref, 2.0, 2.0 * 2.0 ** -24 * (np.abs(routed_ref) + np.abs(shared_ref))
                          + M17.EPS_FAST * np.abs(shared_ref)) + extra,
             "2·2^-24·Σ|加数| + EPS_FAST·|gated| + 1·BfUlp(max|shared|,|moe|)")

    # ---------------- S10 层链：参考自己算（T1 res1/res2 位级 + T3 y_final）----------------
    res1_g = np.asarray(Q["res1"], dtype=np.float32)[:m]
    res2_g = np.asarray(Q["res2"], dtype=np.float32)[:m]
    yfin_g = np.asarray(Q["y_final"], dtype=np.float32)[:m]
    res1_mine = x_res.astype(np.float32)
    t1_exact(V, "T1 res1 == fp32(x_res) 位级", res1_g.view(np.uint32),
             res1_mine.view(np.uint32), "bit")
    # `res2` / `y_final` 的**参考输入**是被判量侧自己的 `moe_output`（③，§1.3）；它的正确性由
    # 上面那条 `T3 moe` 判据覆盖（§1.3 ③ 的「不得作为唯一判据」在这里满足）。用参考自己的
    # `moe_ref` 去比会**双重计入** combine 约定差（golden 用取整后的 shared、设备用未取整的
    # gated），实测 6157/20480 位不符 —— 那是约定差、不是缺陷。
    t1_exact(V, "T1 res2 == fp32(res1 + moe_output) 位级契约（③：输入取被判量侧 moe，由 T3 moe 覆盖）",
             res2_g.view(np.uint32),
             np.float32(res1_g + moe_g.astype(np.float32)).view(np.uint32), "bit", P_CHAIN, "res2")
    yfin_ref = MB.rmsnorm_ref(moe_g, gamma2.reshape(-1), residual=res1_g,
                              eps=np.float32(eps_norm))[1]
    t3_bound(V, "T3 y_final（RMSNorm#2；③：输入取被判量侧 moe，由 T3 moe 覆盖）", yfin_g, yfin_ref,
             M17.tol_bf16(yfin_ref, 2.0, M17.EPS_FAST * np.abs(yfin_ref)),
             "1.0·BfUlp + EPS_FAST·|ref|", P_CHAIN, "y_final")


# ---------------------------------------------------------------------------
# 负向对照（产物侧注入 → 用同一套判据重跑 → 列出**真的**变红的判据）
# ---------------------------------------------------------------------------
def _rebuild_route(T: dict, ids_new: np.ndarray) -> dict:
    """按一份改过的 `topk_ids` 重建**被判量侧**的整组路由派生张量（模拟设备路由配错）。

    注入的是**产物侧**（判据实际读的那几个字节），不是参考侧。
    """
    T2 = dict(T)
    E = int(T["expert_token_counts.bin"].shape[0])
    m, K = ids_new.shape
    perm = MB.moe_permute(ids_new.astype(np.int32), E)
    T2["topk_ids.bin"] = ids_new.astype(np.int32)
    T2["topk_weights.bin"] = np.asarray(T["topk_weights.bin"], dtype=np.float32)[:m]
    T2["expert_token_counts.bin"] = perm["expert_token_counts"]
    T2["perm_src_token.bin"] = perm["perm_src_token"]
    T2["perm_expert.bin"] = perm["perm_expert"]
    x = np.asarray(T["x.bin"], dtype=np.float32)
    T2["x_sorted.bin"] = MB.bf16_bits_to_f32(
        MB.f32_to_bf16_bits(x)[perm["perm_src_token"]]).astype(np.float32)
    return T2


def run_negative_controls(man: dict, T: dict, Q: dict, V: Verdicts) -> None:
    x = np.asarray(T["x.bin"], dtype=np.float32)[:int(man["m"])]
    rw = np.asarray(T["router_weight.bin"], dtype=np.float32)
    E = int(man["num_experts"])
    K = int(man["top_k"])
    ids_g = np.asarray(T["topk_ids.bin"], dtype=np.int64)

    # ---- NC1 路由配错：被判量侧的 top-k 由「前 K」改成「前 K-1 + 重复第 K-1 名」----
    _l9, ids9, _w9 = MB.router_topk(x, rw, K - 1)
    ids1 = np.concatenate([ids9, ids9[:, -1:]], axis=1).astype(np.int64)
    T1m = _rebuild_route(T, ids1)
    V1 = Verdicts()
    run_criteria(man, T1m, Q, V1, quiet=True)
    red1 = [tag for (_, _, tag, ok, _) in V1.judged() if not ok]
    # 期望 = 对**专家标号 / 每专家槽数**敏感的 T1 判据。
    # `perm_src_token` / `x_sorted` 只在「槽位→token 的分配」变化时才红；m=1 时每个槽位都只能是
    # token 0 ⇒ 无论怎么改路由它们都不变（这是正确读数，不是漏判，故不计入期望集）。
    need1 = ["topk_ids", "expert_token_counts", "perm_expert", "expert_offsets/slot_base"]
    hit1 = [n for n in need1 if any(n in t for t in red1)]
    extra1 = [n for n in ("perm_src_token", "x_sorted") if any(n in t for t in red1)]
    note1 = ("另 `perm_src_token`/`x_sorted` 也红了（该档 m>1 ⇒ 槽位→token 的分配真的变了）"
             if extra1 else
             "另 `perm_src_token`/`x_sorted` 未红 —— 该档 m=1，单 token 下每个槽位只能映射到 "
             "token 0 ⇒ 任何路由注入都改不了这两个量（正确读数，不是漏判；m>1 才可能变红）")
    V.add("NC", f"自检-NC1 路由配错（被判量侧 top-{K}→top-{K - 1}+重复末位）"
          "必须让 T1 索引类判据变红", len(hit1) == len(need1),
          f"产物侧注入后**判据真的变红** {len(red1)} 条；期望的 {len(need1)} 类全中 = {hit1}；"
          f"{note1}；红名单：{red1}")

    # ---- NC2 专家序错位 1：被判量侧的 ids 按「router 权重行 roll 1」产生 ----
    _lr, ids_r, _wr = MB.router_topk(x, np.roll(rw, 1, axis=0), K)
    T2m = _rebuild_route(T, np.asarray(ids_r, dtype=np.int64))
    V2 = Verdicts()
    run_criteria(man, T2m, Q, V2, quiet=True)
    red2 = [tag for (_, _, tag, ok, _) in V2.judged() if not ok]
    nrow = int((np.asarray(ids_r, dtype=np.int64) != ids_g).any(axis=1).sum())
    off2 = int((ids_r.astype(np.int64) != ids_g).sum())
    # 期望 = 对**专家标号**敏感的判据。`perm_src_token` / `x_sorted` **不**必红：整体 ±1 的标号
    # 平移保持「专家之间的相对次序」，计数排序按专家分组后再按 token 稳定排 ⇒ 每个槽位落到哪个
    # token 不变 ⇒ 这两个量逐位不变。这是读数、不是漏判（原 r1 的「必红」声称过宽，已更正）。
    need2 = ["topk_ids", "expert_token_counts", "perm_expert"]
    hit2 = [n for n in need2 if any(n in t for t in red2)]
    V.add("NC", "自检-NC2 专家序错位 1（被判量侧 ids = router 权重行 roll 1 的结果）"
          "必须让 T1 抓住", len(hit2) == len(need2),
          f"注入后 ids 有 {nrow}/{int(man['m'])} 行不同（churn {off2}/{ids_g.size} 个 id）；"
          f"**判据真的变红** {len(red2)} 条 = {red2}；期望的 {len(need2)} 类全中 = {hit2}；"
          f"`perm_src_token`/`x_sorted` 不红是**正确读数**：整体标号平移不改变专家间的相对次序，"
          f"计数排序的每槽 token 不变 ⇒ 这两个量逐位不变（它们对「槽位分给了哪个 token」敏感，"
          f"对「专家叫什么名字」不敏感）")
    REPORT.append((f"{tier_prefix()}NC2 的判据分工（报告项）",
                   "对专家**标号**敏感：topk_ids / expert_token_counts / perm_expert；"
                   "对**槽位- token 分配**敏感：perm_src_token / x_sorted / inv_slot"))

    # ---- NC3 槽位计数 off-by-one（被判量侧 counts 改错一个专家）----
    T3m = dict(T)
    c3 = np.asarray(T["expert_token_counts.bin"], dtype=np.int64).copy()
    e0 = int(np.argmax(c3))
    c3[e0] += 1
    T3m["expert_token_counts.bin"] = c3
    V3 = Verdicts()
    run_criteria(man, T3m, Q, V3, quiet=True)
    red3 = [tag for (_, _, tag, ok, _) in V3.judged() if not ok]
    need3 = ["expert_token_counts", "expert_offsets/slot_base"]
    hit3 = [n for n in need3 if any(n in t for t in red3)]
    V.add("NC", "自检-NC3 槽位计数 off-by-one（被判量侧 counts 第 %d 个专家 +1）必须让 T1 契约变红"
          % e0, len(hit3) == len(need3),
          f"注入后 counts[{e0}] {int(np.asarray(T['expert_token_counts.bin'])[e0])}→{int(c3[e0])}；"
          f"**判据真的变红** {len(red3)} 条 = {red3}")

    # ---- NC4 FTZ 关闭（规则级反事实；设备无关）----
    _lo, ids_o, _wo = router_no_ftz(x, rw, K)
    ftz_rows = int((ids_o != ids_g).any(axis=1).sum())
    lmin = float((x.astype(np.float64) @ rw.astype(np.float64).T -
                  (x.astype(np.float64) @ rw.astype(np.float64).T).max(axis=1, keepdims=True)).min())
    REPORT.append(("FTZ 关闭后 top-10 集合变化的行数（报告项；本档真实数据读数）",
                   f"{ftz_rows}/{int(man['m'])} 行；本档 shifted logit 下界 = {lmin:.3f}"
                   f"（落进次正规带需 < log(2^-126) = -87.34 ⇒ 本档{'触及' if lmin < -87.34 else '未触及'}该带）"))
    ea, eb = E - 2, E - 1
    deep_ident = np.eye(E, dtype=np.float32)
    deep = np.full((2, E), -100.0, dtype=np.float32)
    deep[:, 0] = 0.0
    deep[0, ea], deep[0, eb] = -90.0, -95.0
    deep[1, ea], deep[1, eb] = -95.0, -90.0
    _ld, ids_d_ftz, _wd1 = MB.router_topk(deep, deep_ident, 2)
    _ld2, ids_d_no, _wd2 = router_no_ftz(deep, deep_ident, 2)
    same = bool(np.array_equal(ids_d_ftz, ids_d_no))
    V.add("NC", "自检-NC4 FTZ 关闭必须在次正规带与设备口径分叉（构造深尾 logits，规则级、设备无关）",
          not same,
          f"构造行 shifted logits：专家 0 = 0（exp = 1，正规）、专家 {ea}/{eb} = -90/-95"
          f"（exp ≈ 8.2e-40 / 5.5e-42）、其余 = -100（exp ≈ 3.7e-44），后三者均 < 2^-126=1.18e-38"
          f" ⇒ 建模 FTZ 得 ids={ids_d_ftz.tolist()}（次正规全 flush ⇒ 与 bulk 并列 ⇒ 取小 id），"
          f"不建模得 ids={ids_d_no.tolist()}"
          f" ⇒ {'不同（对照咬住）' if not same else '相同（对照未咬住）'}"
          f"；本档真实数据上的同类反事实差异 {ftz_rows}/{int(man['m'])} 行")

    # ---- NC5 量化方向反转（规则级；借 m17 的 W4n 形态）----
    xs32 = np.ascontiguousarray(np.asarray(T["x_sorted.bin"], dtype=np.float32))
    aq_g = np.asarray(Q["a_qx"], dtype=np.uint8)
    asc_g = np.asarray(Q["a_scale"], dtype=np.uint8)
    pk_r, sc_r = M17.quant_rule_reversed(xs32)
    nrev = int((pk_r != aq_g[:, :HIDDEN // 2]).sum()) + int((sc_r != asc_g[:, :HIDDEN // GROUP]).sum())
    tot_rev = pk_r.size + sc_r.size
    V.add("NC", "自检-NC5 量化方向反转必须与 A 侧 golden 字节分叉（规则级）", nrev > 0,
          f"方向反规则与 golden 失配 {nrev}/{tot_rev} 字节 = {100.0 * nrev / max(tot_rev, 1):.1f}%"
          f" ⇒ 只比 scale 字节的判据对此无感；T1 的 nibble+scale 全比才有")


# ---------------------------------------------------------------------------
# nu_* 接口契约腿（m22 的受控非均匀档，只读；参考输入 = ② 设备 topk_ids）
# ---------------------------------------------------------------------------
NU_COVER = ("覆盖该 ② 输入的判据 = `m22_router512/check_ref.py` 的 `J1`（脚本里的 J1 是 "
            "`topk_ids`；README §6.2 表的编号不同，表里 J1 是 `router_logits`）")


def run_nu_contract(with_m22: bool, V: Verdicts) -> int:
    base = ROOT / "m22_router512" / "evidence" / "mode2"
    dirs = sorted(p for p in base.glob("nu_m*") if p.is_dir())
    if not dirs:
        print(f"[nu] SKIPPED（{base} 下没有 nu_m* 档）")
        return 0
    for d in dirs:
        name = d.name
        ids = np.fromfile(d / "topk_ids.bin", dtype=np.int32).reshape(-1, TOPK).astype(np.int64)
        m = ids.shape[0]
        counts = np.fromfile(d / "expert_counts.bin", dtype=np.int32)
        sb = np.fromfile(d / "expert_slot_base.bin", dtype=np.int32)
        psrc = np.fromfile(d / "perm_src_token.bin", dtype=np.int32)
        pexp = np.fromfile(d / "perm_expert.bin", dtype=np.int32)
        c = slot_contract(ids, E_ALL)
        t1_exact(V, f"nu {name} expert_counts == 独立计数排序（{NU_COVER}）",
                 c["expert_token_counts"], counts, "count", P_UP)
        t1_exact(V, f"nu {name} expert_slot_base == 独立前缀和（{NU_COVER}）",
                 c["expert_offsets"][:E_ALL], sb, "base", P_UP)
        t1_exact(V, f"nu {name} perm_src_token == 独立稳定排序（{NU_COVER}）",
                 c["perm_src_token"], psrc, "slot", P_UP)
        t1_exact(V, f"nu {name} perm_expert == 独立稳定排序（{NU_COVER}）",
                 c["perm_expert"], pexp, "slot", P_UP)
        V.add("GUARD", f"nu {name} 非均匀画像：可变活跃数 + 空专家 + 单槽专家",
              int((counts > 0).sum()) > 0 and int((counts == 0).sum()) > 0
              and int((counts == 1).sum()) > 0 and int(counts.sum()) == m * TOPK,
              f"m={m} 活跃 {int((counts > 0).sum())}/{E_ALL}、空 {int((counts == 0).sum())}、"
              f"单槽 {int((counts == 1).sum())}、最大槽 {int(counts.max())}、Σt_e={int(counts.sum())}",
              P_UP)
        if with_m22:
            p = subprocess.run([PY, str(ROOT / "m22_router512" / "check_ref.py"), str(d),
                                "--w", "onehot"], capture_output=True, text=True)
            tail = [ln.strip() for ln in p.stdout.splitlines()
                    if "RESULT" in ln or "分布画像" in ln]
            V.add("T1", f"nu {name} m22 device 档对拍 rc=0（只读重跑 m22 自己的 checker；"
                        f"它的 J1 是那个 ② 输入的覆盖判据）",
                  p.returncode == 0, f"rc={p.returncode}; " + " | ".join(tail[-2:]), P_UP)
    return len(dirs)


# ---------------------------------------------------------------------------
# prepare：生成 → 对拍 → 不留巨物进仓
# ---------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    td = DATA / "real"
    man_p = td / "manifest.json"
    if not man_p.exists():
        print(f"[prepare] 缺入库的 {man_p.relative_to(ROOT)}（真实档 manifest 必须在仓里）")
        return 2
    before = sha256_file(man_p)
    print(f"[prepare] 生成前：{len(list(td.rglob('*.bin')))} 个 .bin 在盘上；"
          f"manifest.json sha256={before[:16]}…")
    gen = subprocess.run([PY, str(HERE / "gen_dataset.py"), "--groups", "real"],
                         capture_output=True, text=True, cwd=str(ROOT))
    for ln in gen.stdout.splitlines():
        print(f"[prepare] gen: {ln}")
    if gen.returncode != 0:
        print(f"[prepare] gen_dataset rc={gen.returncode}\n{gen.stderr[-2000:]}")
        return 1
    after = sha256_file(man_p)
    print(f"[prepare] 生成后：manifest.json sha256={after[:16]}… "
          f"（{'逐字节一致 ⇒ 生成是确定性的' if after == before else '与生成前不一致 ⇒ 生成不确定'}）")
    checked, missing, bad = verify_tier(td)
    print(f"[prepare] sha256 对拍：核对 {checked} 个张量、缺 {missing}、不符 {bad}")
    tot = sum(f.stat().st_size for f in td.rglob("*.bin"))
    print(f"[prepare] 盘上 .bin 合计 {tot / 2**20:.1f} MiB（{tot} B）")
    st = subprocess.run(["git", "status", "--porcelain", "--", str(td)], cwd=str(ROOT),
                        capture_output=True, text=True)
    lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
    print(f"[prepare] git status --porcelain {td.relative_to(ROOT)}：{len(lines)} 行")
    for ln in lines[:20]:
        print(f"[prepare]   {ln}")
    bins = sorted(td.rglob("*.bin"))
    not_ignored = [str(f.relative_to(ROOT)) for f in bins
                   if subprocess.run(["git", "check-ignore", "-q", str(f)],
                                     cwd=str(ROOT)).returncode != 0]
    print(f"[prepare] 本轮枚举到的 {len(bins)} 个 .bin："
          f"{'均被 git check-ignore 命中' if not not_ignored else f'其中未被忽略 {not_ignored[:5]}'}")
    ok = (missing == 0 and bad == 0 and not not_ignored and after == before)
    print(f"[prepare] RESULT: {'OK' if ok else 'FAIL'}（确定性生成 + sha256 全对 + 这批 .bin 未入仓）")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    tiers = [t for t in args.tier.split(",") if t]
    for t in tiers:
        td = DATA / t
        if not (td / "manifest.json").exists():
            print(f"[accept] SKIPPED（{td.relative_to(ROOT)}/manifest.json 不存在 —— 先跑 `prepare`）")
            return 2
        checked, missing, bad = verify_tier(td, verbose=False)
        if missing or bad:
            print(f"[accept] SKIPPED（{td.relative_to(ROOT)}: 核对 {checked} 张量、缺 {missing}、"
                  f"sha256 不符 {bad}；按 manifest 的 sha256 先重生成）")
            return 2
    V = Verdicts()
    REPORT.clear()
    for t in tiers:
        man, T, _qman, Q = load_tier(t, "floor")
        _RPT_PREFIX[0] = f"{t} " if len(tiers) > 1 else ""
        run_criteria(man, T, Q, V)
        run_negative_controls(man, T, Q, V)
        # §1.3 ③ 的**漂移自检**（两层都要过）：
        #   代码层：运行时被标 P_CHAIN 的判据 id ↔ 声明常量 `CHAIN_CRIT_TAGS`；
        #   文档层：**真的去读**那三处文件里的 `M90-CHAIN-IDS[...]` 标记行 ↔ 运行时集合。
        # 任一层不一致 ⇒ 本判据 FAIL ⇒ `run` 退 1。
        chain = sorted(set(CHAIN_SEEN))
        want = sorted(CHAIN_CRIT_TAGS)
        docs = doc_chain_ids()
        doc_bad = {k: v for k, v in docs.items()
                   if (v is None and dict(CHAIN_DOCS)[k] != "devpath") or
                   (v is not None and sorted(set(v)) != (want if dict(CHAIN_DOCS)[k] == "mainleg" else []))}
        dev_ok = all(docs[k] == [] for k in docs if dict(CHAIN_DOCS)[k] == "devpath")
        V.add("T1", "§1.3 ③ 漂移自检：运行时 ③ == CHAIN_CRIT_TAGS == 三处文档的 M90-CHAIN-IDS 标记",
              chain == want and not doc_bad and dev_ok,
              f"运行时 ③ = {chain}（{len(chain)} 条）；常量 = {want}（{len(want)} 条）；"
              f"三处文档标记 = {docs}（mainleg 期望 == 常量；devpath 期望 == 空）"
              f" → {'一致' if (chain == want and not doc_bad and dev_ok) else '不一致：代码/常量/文档三者已漂'}"
              f"；不符处 = {doc_bad if doc_bad else '无'}",
              P_DECL)
    _RPT_PREFIX[0] = ""
    n_legs = run_nu_contract(args.m22, V) if args.m22 else 0
    print("\n[accept] ===== guard（非空洞性 / 结构性 / 覆盖画像；单列，不计入判定项）=====")
    for tier, prov, tag, ok, info in V.rows:
        if tier == "GUARD":
            print(f"[accept]   {'ok  ' if ok else 'FAIL'} {tag}  ({info})")
    print("\n[accept] ===== 判定项（T1 逐位 / T3 推导界 / NC 负向对照；每条带 §1.3 输入溯源标记）=====")
    for tier, prov, tag, ok, info in V.rows:
        if tier != "GUARD":
            print(f"[accept]   {tier:4s} {prov:2s} {'PASS' if ok else 'FAIL'} {tag}  ({info})")
    # §1.3 清单打印（三处文档里的 ③ 只引用这一行 + 上一行的自检，不复写会漂的数字）
    print(f"\n[accept]   [§1.3 ③ 判据清单] {sorted(set(CHAIN_SEEN))}（{len(set(CHAIN_SEEN))} 条；"
          f"与 CHAIN_CRIT_TAGS 的一致性见上面那条「§1.3 ③ 漂移自检」）")
    print("\n[accept] ===== 报告项（不参与 PASS/FAIL；docs/17 §1.1 的降档量）=====")
    for tag, info in REPORT:
        print(f"[accept]   {tag}: {info}")
    bt = V.by(0)
    bp = V.by(1)
    print("\n[accept] ===== 计数 =====")
    for tier in ("T1", "T3", "NC"):
        n, b = bt.get(tier, (0, 0))
        print(f"[accept]   {tier}: {n} 条，{n - b} PASS / {b} FAIL")
    n, b = bp.get(P_DECL, (0, 0))
    print(f"[accept]   §1.3 ① 声明输入（独立）：{n} 条，{n - b} PASS / {b} FAIL")
    n, b = bp.get(P_UP, (0, 0))
    print(f"[accept]   §1.3 ② 上游**设备**产物（点名覆盖后合法）：{n} 条，{n - b} PASS / {b} FAIL")
    n, b = bp.get(P_CHAIN, (0, 0))
    print(f"[accept]   §1.3 ③ 金标链产物（非独立性判据）：{n} 条，{n - b} PASS / {b} FAIL")
    nguard = sum(1 for r in V.rows if r[0] == "GUARD")
    nbad_g = sum(1 for r in V.rows if r[0] == "GUARD" and not r[3])
    print(f"[accept]   GUARD: {nguard} 条，{nguard - nbad_g} ok / {nbad_g} FAIL（不计入判定项）")
    print(f"[accept]   nu 设备档：{n_legs} 个（每个 4 条契约 T1 + 1 条画像 guard）")
    indep = sum(v[0] for k, v in bp.items() if k in (P_DECL, P_UP))
    njudged = len(V.judged())
    nbad = V.nbad()
    if njudged == 0:
        print("[accept] RESULT: SKIPPED (0 条判定项被比较) —— 退出码 2")
        return 2
    if nbad == 0:
        print(f"[accept] RESULT: OK ({njudged} 条判定项全部比较通过；其中输入独立（①+②）"
              f"{indep} 条、③ {njudged - indep} 条；guard {nguard} 条单列)")
        return 0
    print(f"[accept] RESULT: FAIL ({nbad}/{njudged} 条判定项越界) —— 退出码 1")
    return 1


def cmd_pending(args) -> int:
    """打印「待 device 段（B/C）落地后才能跑」的判据清单 —— 本 mission 只把参考侧就位。"""
    print("""[pending] device 段（512/10 融合 MoE 段）落地后才能跑的判据（本 mission 只就位参考侧）：
  1. 融合 ws 整段 sha256 身份 pin（等价 m17 的 W0）；
  2. device 的 A_qx/A_scale/H_qx/H_scale 字节 vs 参考复算字节（现为 qaware golden 字节 vs 参考字节，
     属参考链一致性见证，不是 device 证据）；
  3. device 的 GU/H/Y/routed/shared/moe/y_final vs 本 harness 的参考值（T3 推导界已就位）；
  4. device 的 res1/res2/w_tk_packed/inv_slot 逐位（现为 golden 自身满足这些契约）；
  5. `m15_layer_loop/check_moe_ref.py` 的 device dump 路径（无 dump 时 SKIPPED）。
本命令本身不产生合格证。""")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare", help="生成真实档 + sha256 对拍 + 巨物入仓检查")
    p1.set_defaults(func=cmd_prepare)
    p2 = sub.add_parser("run", help="主验收（T1/T3 分档 + nu_* 契约 + 负向对照）")
    p2.add_argument("--tier", default="real", help="逗号分隔的档名（默认 real）")
    p2.add_argument("--no-m22", dest="m22", action="store_false", default=True,
                    help="跳过 nu_* 设备档（m22 evidence 不可读时）")
    p2.set_defaults(func=cmd_run)
    p3 = sub.add_parser("pending", help="打印待 B/C 落地的判据清单")
    p3.set_defaults(func=cmd_pending)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
