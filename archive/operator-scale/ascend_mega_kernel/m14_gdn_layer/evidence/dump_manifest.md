# M14 dump 归档（sha256 + 字节数）

> 生成日期 2026-09-26（M62 刷新）；平台 Ascend950PR / CANN 9.1.0 / 28 AIC + 56 AIV。
> dump 原始字节 ~150MB/次**未入库**：kernel 已实测确定性（同一二进制两次运行 **38 个文件**逐字节一致，
> 见下「确定性」一节），故用 sha256 清单 + 再生成命令替代。
>
> **例外（M48 定位，见 §6）**：5 个 `*_ws.bin`（整段 workspace dump）**不可逐字节复现**——
> 其中 **97.15% 的字节内核从不写**、另有 2,688 B 是 g/β 32 B 槽尾被 32 B 槽 store 带出的 UB 残留
> ⇒ 它们的 sha256 **不是内核输出的性质**。§3 里它们只留历史值，**不要**当作可复现判据
> （可复现的 **38 个文件** + 两份判据日志才是契约）。

## 1. 构建与再生成命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m14_gdn_layer/build -S m14_gdn_layer -DCMAKE_BUILD_TYPE=Release
cmake --build m14_gdn_layer/build -j4
BIN=$PWD/m14_gdn_layer/build/m14_gdn_layer

# (a) 验收运行（4 个 case：chain / slice / chain2 / hostchain；退出码 0）→ evidence/accept_run.log
$BIN | tee m14_gdn_layer/evidence/accept_run.log

# (b) 段序截断矩阵 → evidence/accept_run_stagelimit.log（15 个组合；每块以 ########## 分隔、以 rc=N 收尾）
for L in 1 2 3 4 5 6 7; do for C in chain hostchain; do
  echo "########## M14_STAGE_LIMIT=$L (case $C) ##########"
  M14_STAGE_LIMIT=$L $BIN $C 2>&1; echo "rc=$?"
done; done
echo "########## M14_STAGE_LIMIT=7 (case slice) ##########"
M14_STAGE_LIMIT=7 $BIN slice 2>&1; echo "rc=$?"

# (c) 落盘 dump（在空目录里跑，dump 落在 cwd）
mkdir -p /tmp/m14_dump && cd /tmp/m14_dump && M14_DUMP=1 $BIN | tee dump_run.log

# (d) numpy 独立交叉校验 → evidence/check_ref_run.log
/usr/local/python3.12.13/bin/python3 <repo>/m14_gdn_layer/check_ref.py . | tee check_ref_run.log

# (e) 逐文件核对 sha256（清单见 §3）
sha256sum *.bin *.txt | sort -k2

# (f) 「0 判定项」负向对照：kernel 与 numpy 两侧都必须**不发合格证**
M14_STAGE_LIMIT=2 $BIN slice        # 期望末行 NO-CRITERIA（判定项 0：无判定力，不发合格证），rc=2
mkdir -p /tmp/m14_zero && cd /tmp/m14_zero
M14_DUMP=1 M14_STAGE_LIMIT=2 $BIN slice
/usr/local/python3.12.13/bin/python3 <repo>/m14_gdn_layer/check_ref.py .
                                    # 期望末行 RESULT: SKIPPED（判定项 0…不发合格证），rc=2
```

## 2. 二进制与源码指纹

**M62（2026-09-26）刷新后的现值**：

```
binary sha256                78eca29572cc221385cce1d4e3b2a1e0dddc0e12465a48c0366270cf434d9901
m14_gdn_layer.asc            d820264894ce9b4842507551c7181597fa98b53cfb82dfdf42f4fc2e7773cc3a
m14_resources.h              3fe5b4897b2edde085588f32f8df16f180ab7dcd7f2e6f586d41d9c379507815   (未改)
check_ref.py                 f7c47c02ab8bbca0145945281e4e9fb6f637d4ccc20426f9522544372c6e590c
CMakeLists.txt               301d44f7baf0d9656dc863ebe60f7dead820a6ed01546485848cd8dd0ac84e1d   (未改)
README.md                    9a6d7953e405745ae50b6cd42a6406df2eca2d9aad15d1402f4c9eb65154872c
evidence/ws_never_written_check.py  38bb03cd38c822bcd8dcac88cf0472159dc4256c4f5fa5d970052fa8cb61eab0
```

> **换 hash 记录（2026-09-26）**——左列 = 本分支第一个 commit `acf3f69` 的对应文件值（复核命令就是
> 本节末核对块里的 4 条 `git show acf3f69:<path> | sha256sum`），右列 = 上表现值（= 工作区实际文件）：
>
> | 文件 | `acf3f69` | 现值（= 上表） |
> |---|---|---|
> | `binary`（重建 `acf3f69` 得到） | `d2019fe36d50…` | `78eca29572cc…` |
> | `m14_gdn_layer.asc` | `2b1ccbec314f…` | `d820264894ce9…` |
> | `check_ref.py` | `ace69f388944…` | `f7c47c02ab8b…` |
> | `README.md` | `839dd2976fe2…` | `9a6d7953e405…` |
> | `evidence/ws_never_written_check.py` | `7686a7ffc1b3…` | `38bb03cd38c8…` |
>
> **binary 变化的来源是判据标签字面量**（`"slice: device 消费的…"` → 按 mode 生成
> `"hostchain"/"slice"` 前缀）与**收尾三态**（判定项计数 + `NO-CRITERIA`/`rc=2`），属**预期**；
> `m14_resources.h`/`CMakeLists.txt` 仍未改。**device 输出未受影响**：这些改动之后的 38 个可复现
> dump 文件 sha256 与 §3 清单**逐一相同**（脚本比对，`mismatch vs manifest: []`）——改动只碰 host 侧
> 判据标签/计数，device 语义与参数生成逐字未变。
>
> **两处必须一致**：本注的「现值」列与上表是**同一批数**；`79e085b` 版的本注曾把
> `check_ref.py` / `README.md` 的现值写成改动过程中的**中间态**（可用
> `git show 79e085b:m14_gdn_layer/evidence/dump_manifest.md | grep -n '5648055f\|6c6b715d'` 复核到那两行）。
> 现值已改为**实际文件 sha**，并在下面附了一条一次性的当场核对命令。
>
> **M62 复核用**：M48 的旧 `binary sha256` `42b9126c…` 已失效——M62 有**代码**改动（device 的
> `sliceMode=2` 路径 + host 侧新用例），与 M48 的"纯注释 ⇒ 二进制不变"不同，**这是预期的**。
> `m14_resources.h` 与 `CMakeLists.txt` **未改**（hash 与 M48 值相同）。
> （M62 过程中还有一次自己观测到的同类读数：只改 `.asc` 注释时 binary 与我前一次构建相同——但那次
> 的前一状态**未入库**、读者无法独立复核，故只作**自述**、不作证据；M48 那条有 commit 对照，强度不同。）

**本表在 M48（2026-09-26）刷新过两次**：`m14_gdn_layer.asc` 与 `README.md` 因**注释/口径订正**
而换 hash（README 与脚本另各因就地文字订正再换过一次）。**那几处是 M48 期间未入库的中间态，
属自述值**（当时按"记录测到的值"归档；无法用 `git show` 复核），三点如实记录：

1. **`binary sha256` 未变**（与刷新前逐字节同一二进制）——M48 对 `.asc` 的 36 行改动全是注释，
   编译器输出完全相同；这比 dump 对拍更强的证据表明"纯注释 = 零行为影响"。
2. `m14_resources.h` / `check_ref.py` / `CMakeLists.txt` **未改**（hash 与刷新前相同）。
3. 新增 `evidence/ws_never_written_check.py`：§6 的定位脚本（仅分析 dump，不参与 kernel 构建）。

**§2 的当场核对（改完任何源码/本节文字后都要跑一遍）**——本节是**指纹记账**，两处（上表与换 hash 记录）
必须逐值相等，且都必须等于实际文件：

```bash
# —— 下面两条都必须在本**分支的 checkout** 根目录里跑（wt-NN，或 `git worktree add` 出来的临时树）。
#    在 main 的 checkout 里跑会读到另一份 `m14_gdn_layer/`，得到「6 处里 4 处不符」的假象（命令没错、环境错了）。
# 工作区现值（应与上表逐行相同）
sha256sum m14_gdn_layer/m14_gdn_layer.asc m14_gdn_layer/m14_resources.h m14_gdn_layer/check_ref.py \
          m14_gdn_layer/CMakeLists.txt m14_gdn_layer/README.md \
          m14_gdn_layer/evidence/ws_never_written_check.py
# 历史值（应与换 hash 记录的左列相同；binary 需按该 commit 的源码重建后比）
git show acf3f69:m14_gdn_layer/m14_gdn_layer.asc | sha256sum
git show acf3f69:m14_gdn_layer/check_ref.py | sha256sum
git show acf3f69:m14_gdn_layer/README.md | sha256sum
git show acf3f69:m14_gdn_layer/evidence/ws_never_written_check.py | sha256sum
```

> **为什么把这条写进正文**：本节曾经犯过的错正是两个「现值」被写成了改动过程中的中间态，而那两个
> 值**在整个模块里只出现在那一行注里**（`git show 79e085b:m14_gdn_layer/evidence/dump_manifest.md`
> 可复核）——即**本 mission 立题要消灭的「自述数字与事实不一致」**。因此这里给出可执行的自查命令，
> 并要求「改完本节后逐值 `sha256sum` 对齐」。
>
> **本节的一次性核对结果（2026-09-26，在本分支 checkout 里当场跑）**：
> * 上表 **6 行 + `binary` 行**：全部等于工作区文件的 `sha256sum` / 新建二进制的 `sha256sum`（7/7 OK）；
> * 换 hash 记录的**左列 4 行**：全部等于上面核对块里那 4 条 `git show acf3f69:<path> | sha256sum` 的输出（4/4 OK）；
> * `42b9126c…`（M48 的 binary）：用 `git archive f0286f6 m14_gdn_layer` 到 /tmp 重新 cmake 构建，
>   得到**逐位相同**的 `42b9126c198b4e0ff523a10c531ca8025266a825fe94e43f6e8fc0153f04e562`（可复现）；
> * 本节不再出现任何「只在本注里出现、别处都查不到」的 sha（`79e085b` 版那两个中间态已删）；
> * §3 的 4 个 M48 期 `*_ws.bin` 历史值与 `m14_params.txt` 的旧值 `69d1c894…`，均可在
>   `git show f0286f6:m14_gdn_layer/evidence/dump_manifest.md` 中找到（属历史归档值，非可复现判据）；
> * **缩写计数口径**（两个口径都对，别按另一个口径当矛盾）：本文件里 `<hex>…` 缩写按**出现次数**是
>   **21 处**、按**去重值**是 **17 个**（`42b9126c…` 与 `426223d7…` 各出现 3 次，含本句自身的引用）。
>   复核：`grep -oE '[0-9a-f]{8,64}…' m14_gdn_layer/evidence/dump_manifest.md | wc -l` = 21、
>   同命令加 `| sort -u` = 17。两种口径下**全部是真前缀、0 UNK**。

## 3. dump 文件 sha256（M62 现值：**38 个可复现文件** = 17 参数 + `m14_params.txt` + 5×(manifest/cs/ssm) + 5 meta）

> **5 个 `*_ws.bin` 不进本表**（`chain`/`slice`/`chain2_s0`/`chain2_s1`/`hostchain`）：它们
> **不可逐字节复现**，sha256 不是内核输出的性质。原因与定位见 **§6**：这些文件是整段
> 5,258,240 B workspace 的 D2H dump，其中 97.15% 的字节内核从不写、2,688 B 是 g/β 32 B 槽尾的
> UB 残留 ⇒ 其内容随设备/UB 历史漂移。M48 实测：6 次运行两两比对，**差异 100% 落在这 2,688 B
> 槽尾上，0 字节落在内核语义写区**。复核 `*_ws.bin` 请改用 `evidence/ws_never_written_check.py`
> （比 sha256 更严格地对齐"内核真正写过的字节"；三态退出码 0/1/2）。
>
> **历史归档值（M48，仅作历史记录，勿当判据）**：`chain_ws.bin` `426223d7…`、
> `slice_ws.bin` `a6b607c1…`、`chain2_s0_ws.bin` `426223d7…`、`chain2_s1_ws.bin` `ab4eb7f1…`。


```
04aaf0f106eb689cd17a41f56701cd8a1198b35d1b67132aa108c54ca9b40a97  chain.txt                      1036 bytes
b55e8b6bc15e9a6506b5bf8e843a86c297a46676754087ea53852826015c0548  chain2_s0.txt                  1049 bytes
c59acfa103ac9ab53a99ab50210a905499711da870016fbcb4a909f46b9cb00d  chain2_s0_cs_out.bin           61440 bytes
85a41f9dbec9d85c74f1a757385215a2038591406b5ed0f8c8a83d58c3b22d3d  chain2_s0_meta.txt             67 bytes
e6552da7e41075ab4b7dab7b6302212c16d9d4ad5d00924a32056f969681b941  chain2_s0_ssm_out.bin          3145728 bytes
8d5f4bd4e7a1b03ab09193f54cbc456ef9d8c9ebb3d13929b9c920fc236f1b0f  chain2_s1.txt                  1049 bytes
77b9184148b5822d0bdda9a1c048cc9564b8157d1c572f571437c629a990d7ba  chain2_s1_cs_out.bin           61440 bytes
0145652a9439fda3ad1f8a10d4b5a8e6335351ebab0813b162af6eb86a992e5b  chain2_s1_meta.txt             67 bytes
427fadadc372df36a51d6efb7b26f3812dd3ea14ad4ae5689a34256c9b5634ed  chain2_s1_ssm_out.bin          3145728 bytes
c59acfa103ac9ab53a99ab50210a905499711da870016fbcb4a909f46b9cb00d  chain_cs_out.bin               61440 bytes
31584e3625734b76bddf30868038a236c17a88d1bbb3af04f37587a74f55a01a  chain_meta.txt                 66 bytes
e6552da7e41075ab4b7dab7b6302212c16d9d4ad5d00924a32056f969681b941  chain_ssm_out.bin              3145728 bytes
07abe14f68e1a910fd7f9559e5e6a48ce87f526d496cd7b2ac7ea669c6321068  hostchain.txt                  1067 bytes
6e8a036c3dbe2b83aed8d8ed6bf3cf3a002e6f46758410a6e3027e19d59e825d  hostchain_cs_out.bin           61440 bytes
42e5b9901d20050cc19f6626025cfcbcaa9aa265e2f5caa57a2cde8fe689c060  hostchain_meta.txt             70 bytes
a7df5ce9f9e8792ed47ccb97a3b8a111081fc1bcbdd3170abea76b8c201518c3  hostchain_ssm_out.bin          3145728 bytes
4b231bc36e67e64638c468afc30643d4682a19b110f75be40c73e8243996cba8  m14_param_alog.bin             256 bytes
2ff00f6f1575904c4f12ded0534d72ef7fe459226d8bc396bd77e4536966c0e9  m14_param_conv_bias.bin        20480 bytes
16d365426f869b1a5d5c0fe8e4a637f317a2257ce1929c38b1bcf728532fd148  m14_param_conv_w.bin           81920 bytes
21d7e8a04c5085e1ef0ea071f9eb95b8663e6cda6fab395f2f8b136260b7e78d  m14_param_cs_init.bin          61440 bytes
a9ac9fa0562ddf43dadc64c26b47f8aa7e106381a7c41ebe1a1cdc9ef1d2dd94  m14_param_dtbias.bin           256 bytes
eb3c5b213d884ef6bcbf8692ce3fe3144077ecd4bdfac7c68739e355ef166af6  m14_param_gamma1.bin           5120 bytes
d1952986426d571cbe051275f0333e89455b32fe05ed2ed462a4bab2da00bde5  m14_param_gamma2.bin           5120 bytes
b7182b7d0b2c6dffc18f32801d3193921c83feb1c9bd7a0c0335212b39af8ed6  m14_param_gamma_g.bin          256 bytes
8fd146c4380da15fba52614c9f5b6731772434a2d923c88e6a130a8f3c052249  m14_param_res_s0.bin           327680 bytes
a889dcd70acaa7a07833f045a777efb44b9ff9f87e1919ac5f7f4c7533d72408  m14_param_res_s1.bin           327680 bytes
30da004a9f4a75043d48daea67462d92211245bf42fc421d34d6334eeb53732f  m14_param_ssm_init.bin         3145728 bytes
076821c3094aa7ea68e6f3c4b7284c0501ab797a980bd6a544847f96bb925f7a  m14_param_w_in.bin             84377600 bytes
ada3e9fa46e82e990a425b781f6c56cec4d735090ef98144f5f6462306c72604  m14_param_w_out.bin            31457280 bytes
b02b22d1e4ca9e29ec1b3c8e42dc48b0ef53125f1ddfed2925f8258ce7e23944  m14_param_x_s0.bin             327680 bytes
0c7a71fefea7e7356dfc27e36a246fe47673e1719585c758754472c9ade23ad1  m14_param_x_s1.bin             327680 bytes
f74560b08d08483d2db09c2c916da83847695bfe140f8241ce4a9c78f38853db  m14_param_xqkvzba.bin          2109440 bytes
a5b07349052bd399c0c7f6a4f598e8327d563ba1d53794aef2ec3cdb615844d4  m14_param_xqkvzba_h.bin        2109440 bytes   ← M62 新增
644e8fd0b93c432c2bb9904a6532824015ee28753d8a4104bdf05c100ada2896  m14_params.txt                 1162 bytes   ← M62 改（+1 行 manifest）
8bfdb3274cde83a031b163358e26990d401e763e26bf282a82f6d5c79c34b787  slice.txt                      1049 bytes
1a119f462e90cbd060b0c9c7e5df314fa54dd07a19906a7aedc8a831d752d7c9  slice_cs_out.bin               61440 bytes
6fd62887580b06b67a443b034c46622896299b327c25382c4899a7adac80b2d1  slice_meta.txt                 66 bytes
bcbf203d2a26f2b28825131c432fc1b740dd7bcd379816c5a87e4a703eae768d  slice_ssm_out.bin              3145728 bytes
```

### 3.1 改前 / 改后 sha 对照（M62：证明「不静默重写已发布的 dump」）

用 M62 之前 §3 的归档值逐行核对本次复跑的 38 个文件（脚本对比，非人工）：

```
old entries: 37 | same: 32 | diff: 1 | missing(non-ws): []
DIFF m14_params.txt   old 69d1c894bf5a… new 644e8fd0b93c…      ← 只多了一行 manifest（x_qkvzba_chain_host）
new-only: hostchain.txt, hostchain_cs_out.bin, hostchain_meta.txt, hostchain_ssm_out.bin,
          m14_param_xqkvzba_h.bin                              ← 新增用例/新增 host 声明输入
```

⇒ **32 个既有可复现文件逐字节未变**；唯一变化的是文本 manifest（+1 行）与 5 个新增文件；
4 个旧 `*_ws.bin` 不参与（不可复现，见 §6）。**没有任何已发布的 `.bin` 被重写**。

**换 hash 后再次核对**：把 `79e085b` 之后落盘的 38 个文件逐个对 §3 清单比 sha256：

```
files: 38 | mismatch vs manifest: []
```

⇒ 那 5 条修复（README/`check_ref.py` 文件头/两条 printf 标签/docstring/计数三态）**没有改变任何
device 产物**——与「只碰 host 侧判据的文字与计数」一致。

## 4. 确定性

```
# 两次独立运行（d1 = 落盘于 /tmp/m14_new_run，d2 = 落盘于 /tmp/m14_ev_dump）的 38 个文件逐字节比对：
identical: 38/38 files, sha256 全部一致（differing: []；两 dir 文件集合也相同）
```

> **M48 订正（口径）**：上面这条结论对 **33/37** 个文件（16 参数、`*_cs_out.bin`、`*_ssm_out.bin`、
> `*_meta.txt`、各 case 的判据 `.txt`）在任何运行下都成立；对 4 个 `*_ws.bin` 只在
> "**前序设备/UB 历史完全相同**"时成立。M48 实测反例：两次相邻的普通运行（同二进制、同机器、
> 仅 cwd 不同）`chain_ws.bin` 的 sha256 也不同，差异 28 B、**全部落在 g/β 32 B 槽尾**（见 §6）。
> ⇒ d1/d2 当年一致是"同一历史下的一致"，不是内核输出的可复现性。

## 5. 结果汇总

```
kernel 内建判据 (evidence/accept_run.log): 138 条判定项 PASS，0 条 FAIL；
  末尾 banner 自述：===== ALL PASS（判定项 138）=====   [rc=0]
  - 口径：`grep -c PASS` 给 139，多出的 1 条是末尾 banner（也含 PASS）；真实判定项 = 138
    （M48 时代的 114 同含 banner ⇒ 当时真实判定项 113，本表顺手订正）
  - 另「报告项」(state 位级统计 + host 全链的 res2) 11 条、「诊断项」(纯参考链传播) 33 条 + 3 条段头，均不参与判定
段序截断矩阵 (evidence/accept_run_stagelimit.log): 15 个组件（M14_STAGE_LIMIT=1..7 × {chain, hostchain} + =7 × slice）
  - 13 个组件**有判定项**（合计 254 条）：全 PASS、0 FAIL、rc=0；mode 2 的 1..7 全档无挂死
  - 2 个组件**判定项为 0**（`hostchain` @ stage_limit=1,2：①S3 起全不产出、mode 2 无 S1/S2 判据）
    ⇒ 打 NO-CRITERIA（判定项 0：无判定力，不发合格证）且 **rc=2**，**不进 PASS 计数**
    （旧口径把 15 条 banner 也算成「15/15 ALL PASS、269 PASS 行」⇒ 真实判定项 254；订正见 README §5.3）
numpy 交叉校验 (evidence/check_ref_run.log): RESULT: OK（判定项 139，报告项 25，未覆盖 8），退出码 0
  - 两份参考是**同源转录**（同一作者的两次转写），不是两个独立来源 —— 见 README §5.6 / check_ref.py 文件头
  - 判定项为 0 时打 `RESULT: SKIPPED（判定项 0…不发合格证）` 并 **rc=2**（与「比过且通过」区分）
```

## 6. `*_ws.bin` 不可复现的定位（M48，2026-09-26）

**结论**：`*_ws.bin` 的 sha256 **不是内核输出的性质**，归档值无法事后复现；**差异只落在内核
语义写过的 147,008 B 之外**——实测落在 g/β 32 B 槽尾（2,688 B，32 B 单块 store 把 UB 残留一并
写出），其余 99.7% 中本次 6 次运行 **0 漂移**。定位脚本：`evidence/ws_never_written_check.py`
（仅分析 dump，不参与 kernel 构建；退出码 0=比过且通过 / 1=比过有差异 / 2=没得比）。

### 6.1 `ws.bin` 是什么

`H_DumpStep`（`m14_gdn_layer.asc` 的 `static void H_DumpStep(...)`）把 host 侧
`std::vector<uint8_t> ws(WS_BYTES)`（= 整段 **5,258,240 B** workspace，即 `m14_resources.h` §5 的
13 张量）**一次 D2H 全量落盘**。而 m=1 的四个 case 只用到其中一小部分：13 张量按 `M_MAX=64` 行
上界分配，实际只写第 0 行（`hostchain` 档另外由 host 预置 `WS_RES1`，那 2560×4 B 是 host 声明值）。

### 6.2 逐字节分类（脚本输出，可复跑）

| 类 | 字节 | 占比 | 说明 |
|---|---|---|---|
| `written`（内核语义写） | **147,008** | 2.80% | `M_MAX` 尺寸张量的**第 0 行** + 按实际尺寸分配的 `q/k/v/o` 全部 + g/β 每槽的 `[h][0]` |
| `slotpad`（g/β 32 B 槽尾） | **2,688** | 0.05% | 每 head 28 B × 48 head × 2 张量（g、β）。S3 用 32 B 单块写回（`DataCopy(gGm_[h*8], gL, cpSlot{1,1,0,0})`），把 UB 槽里 `[h][1..7]` 的残留一并写出 |
| `unused`（内核从不写） | **5,108,544** | 97.15% | `M_MAX` 张量第 1..63 行、槽外对齐填充等 |

张量偏移/尺寸（脚本打印，与 `m14_resources.h` 常量逐项一致，可据此定位任意差异字节）：
`XNORM@0`、`RES1@327680`、`QKVZBA@983040`（以上 row0-only）、`Q@3092480`、`K@3100672`、
`V@3108864`、`G@3133440`、`BETA@3134976`、`O@3136512`（以上全写）、`OPIN@3161088`、
`OPOUT@3947520`、`YFINAL@4275200`、`RES2@4602880`（row0-only）。

### 6.3 实测证据（6 次运行 × 4 case）

运行集合 = 4 次普通运行（2 次同目录背靠背、2 次不同目录）+ 1 次**扰动历史**运行（其前先跑
一遍 m14 验收（116 MB 权重 H2D）与 m7 m=64，再落盘）+ 1 次 **HEAD 原样单独构建**的落盘
（把 HEAD 的 4 个源文件复制到临时目录、用同一 `CMakeLists.txt` 编译）——即 A/B 对照。

| case | 运行间有差异的字节 | 落在 `written` | 落在 `slotpad` | 落在 `unused` |
|---|---|---|---|---|
| `chain` | 2,674 | **0** | 2,674 | **0** |
| `slice` | 2,575 | **0** | 2,575 | **0** |
| `chain2_s0` | 2,626 | **0** | 2,626 | **0** |
| `chain2_s1` | 2,626 | **0** | 2,626 | **0** |

M62 复跑补测（2 目录 × 5 case，新增的 `hostchain` 档在内）：`hostchain` 也给出
`written 0 | slotpad 716 | never-written 0` ⇒ 新档满足同一判据（它的 `WS_XNORM` 从不写、
`WS_RES1` 由 host 预置，都属确定性字节）。

两两比对（`chain` 的 15 对组合）同样 **0 字节**落在 `written`/`unused` 区。⇒

1. **内核真正写过的 147,008 B，在所有运行、所有构建之间逐字节一致**（与 §5 的判据全 PASS 一致）；
2. 本次 6 次运行中，全部漂移来自 `slotpad`（UB 残留随前序块内容而变）；**`unused`（97.15%）
   在这 6 次运行里 0 字节漂移**——它在同一设备/驱动下是不动点，**跨设备/驱动版本是否有漂移
   未测**，故不作已实测的漂移来源（原始归档 dump 未入库，事后也无法逐字节回验）；
3. 因此 §3 里那 4 行 ws 记录**只作历史归档值**；复核 `*_ws.bin` 请改跑（该脚本的退出码三态：
   `0` = 比过且通过 / `1` = 比过有差异 / `2` = 没得比）：

```bash
# 真机：按 §1(c) 落盘 2 次以上（建议不同目录），然后
/usr/local/python3.12.13/bin/python3 m14_gdn_layer/evidence/ws_never_written_check.py DIR1 DIR2 ...
# 期望末行：RESULT: OK (5/5 cases compared; all variation confined to the g/beta slot pad)，退出码 0
#   （只落了 M62 之前那 4 个 case 的旧目录时会是 4/4 或更少 —— 末行始终带**实际比较计数**）
```

**脚本的负向对照（M48 实测归档，两个读数都留档）**——审校脚本必须能分辨"比过通过"与
"根本没比"（tower 规则，2026-09-26）：

```
########## NEGATIVE CONTROL（空目录 + 不存在的目录；M62 复跑，5 case）##########
chain      : SKIPPED (need >=2 dumps, got 0)
slice      : SKIPPED (need >=2 dumps, got 0)
chain2_s0  : SKIPPED (need >=2 dumps, got 0)
chain2_s1  : SKIPPED (need >=2 dumps, got 0)
hostchain  : SKIPPED (need >=2 dumps, got 0)
RESULT: SKIPPED (no case had >=2 *_ws.bin dumps — nothing was compared)
EXIT=2                                  # 没得比 ⇒ 非零退出，不发合格证

########## POSITIVE（M48 时代，4 case；5 个 dump 目录：4 普通 + 1 扰动历史）##########
chain       dumps=5 varying   2573 B -> written 0 | slotpad 2573 | never-written 0
slice       dumps=5 varying   2550 B -> written 0 | slotpad 2550 | never-written 0
chain2_s0   dumps=5 varying   2062 B -> written 0 | slotpad 2062 | never-written 0
chain2_s1   dumps=5 varying   2062 B -> written 0 | slotpad 2062 | never-written 0
RESULT: OK (4/4 cases compared; all variation confined to the g/beta slot pad)
EXIT=0                                  # 比过且通过，且印出实际比较计数

########## POSITIVE（M62 复跑，5 case；2 个 dump 目录 /tmp/m14_r2_dump2 + /tmp/m14_final_dump）##########
cases checked: 5 (chain, slice, chain2_s0, chain2_s1, hostchain)      # ← 名单/计数由 CASES 推出（不再写死数字）
chain       dumps=2 varying    571 B -> written 0 | slotpad 571 | never-written 0
slice       dumps=2 varying    571 B -> written 0 | slotpad 571 | never-written 0
chain2_s0   dumps=2 varying    571 B -> written 0 | slotpad 571 | never-written 0
chain2_s1   dumps=2 varying    571 B -> written 0 | slotpad 571 | never-written 0
hostchain   dumps=2 varying    571 B -> written 0 | slotpad 571 | never-written 0
RESULT: OK (5/5 cases compared; all variation confined to the g/beta slot pad)
EXIT=0
```

（差异字节数随目录组合变化，属预期：它衡量的正是"不同设备/UB 历史"漂移了多少槽尾字节。）

> **与 M48 改动的关系**：M48 对 m14 的改动**全是注释**（§2 的 `binary sha256` 与刷新前**逐字节相同**），
> 且上表含"A/B 对照"列 ⇒ 本现象**不是** M48 引入的；M48 只是把它定位清楚并留下明账
> （原始归档 dump ~150MB/次未入库，故无法逐字节回验归档值本身）。

**某次运行的 ws sha256（示例，勿当判据——每次运行都会变）**：

```
e652fb16eb8677420ca0df15c25f36c4864db8b4b5eaf1ca96bf08f48e39306d  chain_ws.bin
064f9ef327623b8639774e9053ba7e8f298862861e074e3e0a6577dae513dc5c  slice_ws.bin
09135c93d1517c633e512d688a9f8e2bf9d57d39c9f259aa74452cfbf8dd4411  chain2_s0_ws.bin
2ef3e896cbe053f4c610f65eadeca6003746be022fc803e43748a6543cf1f49d  chain2_s1_ws.bin
```
