# PTO MegaMoE 阶段伪码

<style>
.pto-api { color: #ff6a00; }
h2, h3 {
  color: #8B0000;
  font-weight: bold;
}
</style>

本文按当前 A2/A3 `dispatch_mega_combine` 生产主路径描述七个阶段的数据流、核心分工和同步关系。
伪码省略边界检查、UB 地址计算和具体 event ID；分组参数及 shape 配置见
[overview_v1.md](./overview_v1.md)。

组内阶段同步使用 producer/consumer slot 表达：数组下标是逻辑 core ID，`epoch` 是 slot 中保存的
expert 进度值。对 expert `e`，producer 发布 `2 * e + 1`，coordinator 汇总后向 consumer 发布
`2 * e + 2`。

```text
FrontReorder:
  x[M, K] + expertId[M, topK]
    -> source rank remoteWindow.offsetA[expert-major rows, K + 32]
    -> expandedRowIdx / tokenPerExpert / preSumBeforeRank / cumsumMM / expertTokenNums

Dispatch:
  destination rank pulls its local-expert rows from every source rank offsetA
    -> gmA[expert-major rows, K] + perTokenScale1[rows]

GMM1:
  gmA[int8] @ weight1[int8] + scale1
    -> gmC[rows, N] half

SwiGLU:
  gmC * perTokenScale1
    -> silu(up) * gate
    -> dynamic quant
    -> gmPermutedToken[rows, N/2] int8 + perTokenScale2[rows]

GMM2:
  gmPermutedToken[int8] @ weight2[int8] + scale2
    -> gmm2Output[rows, K] half

Combine:
  gmm2Output * perTokenScale2
    -> source rank remoteWindow.offsetD[expanded row, K]
    -> per-source-rank expert progress / DataReady

Unpermute:
  offsetD + probs + expandedRowIdx
    -> phase 1: 32 AIV process tokens whose routes are ready
    -> phase 2: 48 AIV process the remaining tokens
    -> out[M, K]
```

## 1. FrontReorder：源 rank 内 route 排序、量化、发布 count

关键流程：

- AIV-only 阶段；将 `M * topK` 条 route 按 global expert 排序，生成 expert-major row 布局。
- host 根据 UB 工作集选择 `FullLoad`、`OneCore` 或 `MultiCore`：
  - `FullLoad`：排序、反排、count 和量化工作集留在 UB，活跃 AIV 各自持有完整排序结果并分担量化。
  - `OneCore`：单 AIV 在 UB 内完成排序，随后全部 AIV 进入共享后处理。
  - `MultiCore`：多 AIV 生成有序 run，再通过多轮 4 路归并得到最终顺序。
- 排序结果生成 `expandedRowIdx[srcRoute] = dstRow`；同一 token 的 topK route 复用一次动态量化结果，
  scatter 到 `offsetA[dstRow]`。
- `localTokenPerExpert` 统计本 rank 发往每个 global expert 的 route 数。
- count row 加 marker 后写到各 peer；peer 使用 `TWAIT` 等待并恢复真实 count，随后生成
  `preSumBeforeRank`、`cumsumMM` 和 `expertTokenNums`。

关键 PTO 接口：

- 排序和抽取：`TSORT32`、`TMRGSORT`、`TGATHER`
- Tile 绑定和搬运：`TASSIGN`、`TLOAD`、`TSTORE`
- 动态量化：`TCVT`、`TABS`、`TROWMAX`、`TMAX`、`TDIV`
- 跨 rank count ready：`TWAIT`
- AIV 阶段边界：`SYNCALL<AIVOnly>`

伪码：

<pre><code>
if frontCase == FullLoad:
  each active AIV:
    expertUb = <span class="pto-api">PtoLoadVector</span>(expertId[0 : M * topK])
    srcRouteUb = 0 .. M * topK - 1
    packedUb = <span class="pto-api">TSORT32</span>(expertUb, srcRouteUb)
    packedUb = <span class="pto-api">TMRGSORT</span>(packedUb)
    sortedExpert, sortedSrcRoute = <span class="pto-api">TGATHER</span>(packedUb)
    expandedRowIdxUb = inverse_sort(sortedSrcRoute)       # srcRoute -> expert-major dstRow

  owner AIV:
    localTokenPerExpert = count_runs(sortedExpert)
  AIV0:
    <span class="pto-api">PtoStoreVector</span>(expandedRowIdx, expandedRowIdxUb)

  each active AIV for assigned source-token rows:
    qInt8, scale = dynamic_quant_once(x[token, 0:K])
    for topkSlot in 0 .. topK - 1:
      dstRow = expandedRowIdxUb[token * topK + topkSlot]
      <span class="pto-api">PtoStoreVector</span>(offsetA[dstRow], packed(qInt8, scale))

else:
  if frontCase == OneCore:
    AIV0:
      packedRuns = <span class="pto-api">TSORT32</span>(expertId, srcRoute)
      packedRuns = <span class="pto-api">TMRGSORT</span>(packedRuns)
  else:                                                   # MultiCore
    each sort AIV builds sorted runs with <span class="pto-api">TSORT32</span> + <span class="pto-api">TMRGSORT</span>
    merge sorted runs in multiple 4-way <span class="pto-api">TMRGSORT</span> rounds

  AIV0:
    sortedExpert, sortedSrcRoute = <span class="pto-api">TGATHER</span>(packedRuns)
    store sortedExpert / sortedSrcRoute to front workspace

  <span class="pto-api">SYNCALL&lt;AIVOnly&gt;</span>()
  all AIVs:
    localTokenPerExpert = count(sortedExpert)
    expandedRowIdx[srcRoute] = dstRow
    qInt8, scale = dynamic_quant_once(x[token, 0:K])
    scatter packed(qInt8, scale) -> offsetA[expandedRowIdx[route]]

<span class="pto-api">SYNCALL&lt;AIVOnly&gt;</span>()

for dstRank assigned to this AIV:
  countRow = <span class="pto-api">PtoLoadVector</span>(localTokenPerExpert)
  countRow += countMarker                              # zero count can also represent arrival
  <span class="pto-api">PtoStoreVector</span>(peer[dstRank].tokenPerExpert[myRank], countRow)

for srcRank assigned to this AIV:
  <span class="pto-api">TWAIT</span>(tokenPerExpert[srcRank].markers != 0)
  countRow = tokenPerExpert[srcRank] - countMarker
  preSumBeforeRank[srcRank] = prefix_before_my_local_experts(countRow)

<span class="pto-api">SYNCALL&lt;AIVOnly&gt;</span>()
AIV0:
  cumsumMM = inclusive_prefix_across_source_ranks(tokenPerExpert)
  expertTokenNums = cumsumMM[lastSourceRank]
</code></pre>

## 2. Dispatch：目的 rank 拉取 source rank 的 packed A

关键流程：

- 物理核 `0..15` 的 AIV0 组成 Dispatch 组；其中前 `rankSize` 个 AIV 分别负责一个 source rank。
- 按 local expert 顺序处理；每个 worker 根据 `preSumBeforeRank` 从 peer `offsetA` 找到读起点，根据
  `cumsumMM` 找到本地 `gmA` 写起点。
- 每次最多拉取 2 行 packed row，两个 96 KiB UB buffer 做 ping-pong；payload 和 per-token scale
  分别写入 `gmA` 和 `perTokenScale1`。
- 每个 expert 的所有 source-rank worker 完成后，由 coordinator 发布该 expert 的 GMM1 ready。
  GMM1 前部放行 24 个 AIC，后续放行 16 个 AIC。

关键 PTO 接口：

- 远端 packed row 读取：`TLOAD`
- payload / scale 拆包写回：`TSTORE`
- Tile 绑定：`TASSIGN`
- 组间通知：GM arrival/ready epoch + MTE 批量汇总/发布

伪码：

<pre><code>
for expert in 0 .. expertPerRank - 1:
  if dispatchLocalId &lt; rankSize:
    srcRank = dispatchLocalId
    rows = tokenPerExpert[srcRank, globalExpert(myRank, expert)]
    srcRowBase = preSumBeforeRank[srcRank, expert]
    dstRowBase = groupBase + cumsum_before_source(srcRank, expert)

    for rowOffset in 0 .. rows step 2:
      bufferId = next_pingpong_buffer()
      packedTile = <span class="pto-api">TLOAD</span>(peer[srcRank].offsetA[srcRowBase + rowOffset], maxRows=2)
      payloadTile = packedTile[:, 0:K]
      scaleTile = packedTile[:, K]
      <span class="pto-api">TSTORE</span>(gmA[dstRowBase + rowOffset], payloadTile)
      <span class="pto-api">TSTORE</span>(perTokenScale1[dstRowBase + rowOffset], scaleTile)

    publish arrival[dispatchLocalId] = 2 * expert + 1

  coordinator:
    wait min(arrival[0 : rankSize]) &gt;= 2 * expert + 1
    gmm1Consumers = (expert &lt; fullAicGmm1ExpertCount) ? 24 : 16
    publish ready[0 : gmm1Consumers] = 2 * expert + 2

  groupBase += cumsumMM[lastSourceRank, expert]
</code></pre>

## 3. GMM1：按 expert 分组做第一个 int8 GEMM

关键流程：

- 按 local expert 顺序计算。前 `fullAicGmm1ExpertCount` 个 expert 使用全部 24 个 AIC，之后缩为
  物理核 `0..15` 的 16 个 AIC。
- 每个 expert 等待自己的 Dispatch ready slot；按 `128 x 256` output tile 在参与 AIC 间轮转分配。
- 小 M 且默认 tile 数不足以覆盖参与 AIC 时，沿 N 维按 32 列粒度重新均衡，使更多 AIC 参与。
- GMM1/GMM2 复用同一套 GMM pipeline：L1 A/B、L0A/L0B 双缓冲，L0C 单缓冲，并使用
  N 方向 9 列蛇形 swizzle。
- 每个 expert 的参与 AIC 全部完成后，发布该 expert 的 SwiGLU ready；GMM1 全部完成后发布 done，
  供 GMM2 决定扩组时机。

关键 PTO 接口：

- Tile 绑定和搬运：`TASSIGN`、`TLOAD`、`TEXTRACT`
- Cube 计算：`TMATMUL`、`TMATMUL_ACC`
- FixPipe scale 和输出：`TMOV`、`TSTORE_FP`
- 阶段同步：GM ready/arrival epoch

伪码：

<pre><code>
groupBase = 0
startCore = 0

for expert in 0 .. expertPerRank - 1:
  activeAic = (expert &lt; fullAicGmm1ExpertCount) ? 24 : 16
  if physicalAicId &gt;= activeAic:
    leave GMM1 and enter the primary GMM2 group

  wait dispatchReady[physicalAicId] &gt;= 2 * expert + 2
  currentM = clip(cumsumMM[lastSourceRank, expert], groupBase, maxOutputSize)
  tiles = build_output_tiles(currentM, N, tileM=128, tileN=256)
  tiles = balance_small_m_along_n_if_needed(tiles, activeAic)

  for tile assigned to this AIC with rotating startCore:
    blockM, blockN = swizzle_9_columns_snake_m(tile)
    for kTile in 0 .. K step 512:
      <span class="pto-api">TLOAD</span>(A_l1[pingpong], gmA[groupBase + blockM, kTile])
      <span class="pto-api">TLOAD</span>(B_l1[pingpong], weight1[expert, kTile, blockN])
      for l0k in 0 .. 512 step 128:
        <span class="pto-api">TEXTRACT</span>(A_l0, A_l1, l0k)
        <span class="pto-api">TEXTRACT</span>(B_l0, B_l1, l0k)
        firstK ? <span class="pto-api">TMATMUL</span>(acc, A_l0, B_l0)
               : <span class="pto-api">TMATMUL_ACC</span>(acc, A_l0, B_l0)
    <span class="pto-api">TMOV</span>(fixpipeScale, scale1[expert, blockN])
    <span class="pto-api">TSTORE_FP</span>(gmC[groupBase + blockM, blockN], acc, fixpipeScale)

  synchronize participating AICs
  publish SwiGLU ready for expert
  groupBase += currentM
  startCore = rotate_start_core(startCore, tileCount, activeAic)

GMM1 coordinator publishes gmm1Done
</code></pre>

## 4. SwiGLU：GMM1 输出反量化、激活、再量化

关键流程：

- 物理核 `0..15` 的 AIV1 组成 SwiGLU 组；M=16 使用 8 个 AIV，其余目标 shape 使用 16 个。
- 按 expert 逐个等待 GMM1 ready，并将当前 expert 的 row 均分给活跃 AIV。
- 每个 worker 使用双 UB stage 做 full-row load/compute/store 流水。
- `gmC` 转 fp32 后乘 `perTokenScale1`，计算 `silu(up) * gate`；再按 row 动态量化为 int8。
- `perTokenScale2` 在 UB 中按最多 128 行聚合后写回。
- 当前 expert 的全部活跃 AIV 完成后，coordinator 向 8 个 GMM2 AIC 发布 ready。

关键 PTO 接口：

- 读写和类型转换：`TLOAD`、`TSTORE`、`TCVT`
- 反量化和激活：`TMULS`、`TEXP`、`TADDS`、`TDIV`、`TMUL`
- 动态量化归约：`TABS`、`TROWMAX`、`TMAX`
- scale 批量写回：`PtoStoreVector`
- 阶段同步：GM arrival/ready epoch

伪码：

<pre><code>
groupBase = 0

for expert in 0 .. expertPerRank - 1:
  coordinator waits until all GMM1 producers finish expert
  wait swigluReady[swigluLocalId] &gt;= 2 * expert + 2

  currentM = clip(cumsumMM[lastSourceRank, expert], groupBase, maxOutputSize)
  localRowStart, localRows = split_rows(currentM, swigluActiveAiv)
  prefetch first gmC row

  for row in assigned rows:
    bufferId = row % 2
    if hasNextRow:
      prefetch next gmC row into the other UB stage

    cFp32 = <span class="pto-api">TCVT</span>(gmC[row], CAST_NONE)
    dequant = <span class="pto-api">TMULS</span>(cFp32, perTokenScale1[row])
    expNegUp = <span class="pto-api">TEXP</span>(-dequant[0 : N/2])
    silu = <span class="pto-api">TDIV</span>(dequant[0 : N/2],
                                  <span class="pto-api">TADDS</span>(expNegUp, 1.0))
    y = <span class="pto-api">TMUL</span>(silu, dequant[N/2 : N])

    maxAbs = reduce_max(<span class="pto-api">TABS</span>(y))
    scale2 = max(maxAbs, eps) / 127
    qInt8 = <span class="pto-api">TCVT</span>(y / scale2, CAST_RINT)
    <span class="pto-api">TSTORE</span>(gmPermutedToken[row], qInt8)
    append scale2 to current 128-row scale buffer

  flush perTokenScale2 scale buffers
  publish arrival[swigluLocalId] = 2 * expert + 1
  coordinator waits all active SwiGLU arrivals
  coordinator publishes gmm2Ready[0 : 8] = 2 * expert + 2
  groupBase += currentM
</code></pre>

## 5. GMM2：动态扩组的第二个 int8 GEMM

关键流程：

- 物理核 `16..23` 的 8 个 AIC 从 expert 0 开始 GMM2；完成 GMM1 的 16 个 AIC 等待加入时机。
- 原 GMM2 组从配置的 `gmm2JoinCheckStartExpert` 开始，在每个 expert 边界检查 GMM1 done。
  coordinator 将统一决策写入 join slot，确保 producer 和 consumer 对同一个 join expert 达成一致。
- join 前由 8 个 AIC 计算；join expert 及之后由全部 24 个 AIC 计算。
- 每个 expert 等待 SwiGLU ready，随后复用 GMM1 的 tile、swizzle 和多级双缓冲 pipeline。
- expert 完成后，实际参与的 8 或 24 个 AIC 分别发布 arrival，供 Combine coordinator 汇总。

关键 PTO 接口：

- Tile 绑定和搬运：`TASSIGN`、`TLOAD`、`TEXTRACT`
- Cube 计算：`TMATMUL`、`TMATMUL_ACC`
- FixPipe scale 和输出：`TMOV`、`TSTORE_FP`
- 动态扩组和阶段同步：GM join/ready/arrival epoch

伪码：

<pre><code>
primaryGroup = physical AIC 16 .. 23
helperGroup = physical AIC 0 .. 15
joined = false

helperGroup:
  joinExpert = wait for join decision
  rebuild groupBase and rotating startCore at joinExpert

for expert from (helper ? joinExpert : 0) to expertPerRank - 1:
  wait gmm2Ready[logicalAicId % 8] &gt;= 2 * expert + 2

  if primary and not joined and expert &gt;= gmm2JoinCheckStartExpert:
    coordinator checks gmm1Done and publishes a unified decision for this expert
    if decision joins at this expert:
      joined = true
      activeAic = 24

  currentM = clip(cumsumMM[lastSourceRank, expert], groupBase, maxOutputSize)
  for tile assigned to this AIC with rotating startCore:
    blockM, blockN = swizzle_9_columns_snake_m(tile)
    for kTile in 0 .. N/2 step 512:
      <span class="pto-api">TLOAD</span>(A_l1[pingpong], gmPermutedToken[groupBase + blockM, kTile])
      <span class="pto-api">TLOAD</span>(B_l1[pingpong], weight2[expert, kTile, blockN])
      for l0k in 0 .. 512 step 128:
        <span class="pto-api">TEXTRACT</span>(A_l0, A_l1, l0k)
        <span class="pto-api">TEXTRACT</span>(B_l0, B_l1, l0k)
        firstK ? <span class="pto-api">TMATMUL</span>(acc, A_l0, B_l0)
               : <span class="pto-api">TMATMUL_ACC</span>(acc, A_l0, B_l0)
    <span class="pto-api">TMOV</span>(fixpipeScale, scale2[expert, blockN])
    <span class="pto-api">TSTORE_FP</span>(gmm2Output[groupBase + blockM, blockN], acc, fixpipeScale)

  synchronize participating AICs
  each active AIC publishes gmm2Arrival[logicalAicId] = 2 * expert + 1
  groupBase += currentM
</code></pre>

## 6. Combine：8 个 AIV 按 source rank 写回并发布进度

关键流程：

- 物理核 `16..23` 的 AIV0 组成 8-AIV Combine 组；任务按 source rank 分配，EP8 时每个 AIV
  负责一个 source rank，EP16 时每个 AIV 轮转负责两个。
- Combine 先等待 `combineStartAfterGmm2Expert` 指定的 GMM2 expert ready，再从 expert 0 开始追赶；
  配置值 0 表示等待 expert 0，而不是跳过等待。
- 每行 `gmm2Output` 转 fp32，乘 `perTokenScale2` 反量化，再转换为输出类型并写入 source rank
  的 `offsetD`。
- 达到 `unpermutePhase1ReadyExpertCount` 时，向各 source rank 发布已完成的 expert 数；全部 expert
  完成后发布最终 expert progress 和 DataReady。
- 本卡 8 条 Combine lane 全部完成后清理 count window，并放行后 16 个 Unpermute worker。

关键 PTO 接口：

- GMM2 结果读取和远端写回：`TLOAD`、`TSTORE`
- 类型转换和反量化：`TCVT`、`TMULS`
- 跨 rank 进度通知：`TNOTIFY`
- 组间同步：GM GMM2 arrival/Combine ready epoch

伪码：

<pre><code>
initialReadyExpert = combineStartAfterGmm2Expert
coordinator waits until all GMM2 producers finish initialReadyExpert
all Combine lanes wait combineReady &gt;= 2 * initialReadyExpert + 2

groupBase = 0
for expert in 0 .. expertPerRank - 1:
  if expert &gt; initialReadyExpert:
    coordinator waits until all 8-or-24 GMM2 producers finish expert
  wait combineReady[combineLocalId] &gt;= 2 * expert + 2
  currentM = cumsumMM[lastSourceRank, expert]

  for srcRank assigned to this Combine AIV:
    rows = tokenPerExpert[srcRank, globalExpert(myRank, expert)]
    srcRowBase = groupBase + cumsum_before_source(srcRank, expert)
    dstRowBase = preSumBeforeRank[srcRank, expert]

    for row in 0 .. rows - 1:
      c = <span class="pto-api">TLOAD</span>(gmm2Output[srcRowBase + row, 0:K])
      fp32 = <span class="pto-api">TCVT</span>(c, CAST_NONE)
      fp32 = <span class="pto-api">TMULS</span>(fp32, perTokenScale2[srcRowBase + row])
      d = <span class="pto-api">TCVT</span>(fp32, CAST_RINT)
      <span class="pto-api">TSTORE</span>(peer[srcRank].offsetD[dstRowBase + row, 0:K], d)

  if expert + 1 == unpermutePhase1ReadyExpertCount:
    drain remote stores
    for srcRank assigned to this AIV:
      <span class="pto-api">TNOTIFY</span>(peer[srcRank].ExpertProgress,
                                readyExpertCount=expert + 1, op=Set)

  groupBase += currentM

drain remote stores
for srcRank assigned to this AIV:
  <span class="pto-api">TNOTIFY</span>(peer[srcRank].ExpertProgress, readyExpertCount=expertPerRank, op=Set)
  <span class="pto-api">TNOTIFY</span>(peer[srcRank].DataReady, currentLaunchEpoch, op=Set)

publish localCombineDone[combineLocalId]
Combine coordinator:
  wait all 8 Combine lanes done
  clear tokenPerExpert for next launch
  publish UnpermuteStart[32 : 48]
</code></pre>

## 7. Unpermute：32+16 AIV 两阶段恢复原 token 顺序

关键流程：

- Unpermute 不单独占核：Dispatch/SwiGLU 所在物理核的 32 个 AIV 先加入；Combine 所在物理核的
  16 个 AIV 在本卡 Combine 完成后加入。
- 第一阶段 coordinator 等待 Dispatch/SwiGLU 释放，以及所有 producer rank 的 expert progress 达到
  `unpermutePhase1ReadyExpertCount`，随后向前 32 个 worker 发布 start。
- 第一阶段按 32 worker 切 token；仅当一个 token 的所有有效 topK route 都已由对应 producer rank
  写回时才处理该 token。
- 前 32 个 worker 完成第一阶段后，coordinator 等待所有 producer rank 全部完成，并等待后 16 个
  worker 可用，然后向全部 48 个 worker 发布第二阶段 ready。
- 第二阶段按 48 worker 重新切 token，只处理第一阶段未满足 ready 条件的 token。
- 每个 token 沿 K 维分块，读取 topK 对应的 `offsetD` row，乘路由权重后用 fp32 累加，最后转换并写回。

关键 PTO 接口：

- metadata 预取：`PtoLoadVector`
- offsetD 读取和输出写回：`TLOAD`、`TSTORE`
- 类型转换、权重缩放和累加：`TCVT`、`TMULS`、`TADD`
- 组内进度读取/发布：MTE 批量 `TLOAD`/`TSTORE` + GM epoch

伪码：

<pre><code>
phase1Coordinator:
  wait all Dispatch AIVs and SwiGLU AIVs released
  wait every producerRank.ExpertProgress &gt;= unpermutePhase1ReadyExpertCount
  snapshot phase1ReadyExpertCount[producerRank]
  publish UnpermuteStart[0 : 32]

first 32 workers:
  tokenRange = split_tokens(M, workerCount=32, workerId)
  for token in tokenRange:
    phase1Ready = true
    for topkSlot in 0 .. topK - 1:
      globalExpert = expertId[token, topkSlot]
      producerRank, localExpert = split_global_expert(globalExpert)
      phase1Ready &amp;= localExpert &lt; phase1ReadyExpertCount[producerRank]
    if phase1Ready:
      process_token(token)
  publish phase1Done[workerId]

phase1Coordinator:
  wait phase1Done[0 : 32]
  wait every producerRank.ExpertProgress &gt;= expertPerRank
  wait UnpermuteStart[32]                         # 后 16 个 AIV 已完成本卡 Combine
  publish phase2Ready[0 : 48]

all 48 workers:
  wait phase2Ready[workerId]
  tokenRange = split_tokens(M, workerCount=48, workerId)
  for token in tokenRange:
    if token was not ready in phase 1:
      process_token(token)

process_token(token):
  expandedRows = <span class="pto-api">PtoLoadVector</span>(expandedRowIdx[token, 0:topK])
  routeProbs = <span class="pto-api">PtoLoadVector</span>(probs[token, 0:topK])
  for col in 0 .. K step unpermuteTileCols:
    acc = 0.0f
    for topkSlot in 0 .. topK - 1:
      row = expandedRows[topkSlot]
      if row is valid:
        d = <span class="pto-api">TLOAD</span>(offsetD[row, col : col + unpermuteTileCols])
        fp32 = <span class="pto-api">TCVT</span>(d, CAST_NONE)
        weighted = <span class="pto-api">TMULS</span>(fp32, routeProbs[topkSlot])
        acc = <span class="pto-api">TADD</span>(acc, weighted)
    outTile = <span class="pto-api">TCVT</span>(acc, CAST_RINT)
    <span class="pto-api">TSTORE</span>(out[token, col : col + unpermuteTileCols], outTile)
</code></pre>
