/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

using namespace pto;

#define VEC_CORES 2

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

// ---------------------------------------------------------------------------------------
// Vector side, phase 1: build tileC = tileA + tileB and push it V2C.
//
// FLAG CONTRACT: on return, (PIPE_V, PIPE_MTE2, EVENT_ID1) and (PIPE_MTE3, PIPE_V,
// EVENT_ID1) are left ARMED for runVecPopC2V, which consumes them. The two halves are one
// flag ledger split across two functions; keep them balanced as a pair.
// ---------------------------------------------------------------------------------------
template <typename T, int TOTAL_M, int K, int N, TileSplitAxis SplitAxis, typename PipeT>
AICORE inline void runVecPushV2C(PipeT& pipe, __gm__ T* srcA, __gm__ T* srcB, uint32_t subBlockIdx)
{
    constexpr uint32_t V2C_ROWS = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? (TOTAL_M / VEC_CORES) : TOTAL_M;
    constexpr uint32_t V2C_COLS = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (K / VEC_CORES) : K;

    using VecTileK = Tile<TileType::Vec, T, V2C_ROWS, V2C_COLS, BLayout::RowMajor, V2C_ROWS, V2C_COLS>;
    using GlobalAB = GlobalTensor<
        T, pto::Shape<1, 1, 1, V2C_ROWS, V2C_COLS>, pto::Stride<TOTAL_M * K, TOTAL_M * K, V2C_ROWS * V2C_COLS, K, 1>>;

    VecTileK tileA, tileB, tileC;
    TASSIGN(tileA, 0x0);
    TASSIGN(tileB, 0x4000);
    TASSIGN(tileC, 0x8000);

    size_t abOffset;
    if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
        abOffset = static_cast<size_t>(subBlockIdx * V2C_ROWS) * K;
    } else {
        abOffset = static_cast<size_t>(subBlockIdx) * V2C_COLS;
    }

    GlobalAB globalA(srcA + abOffset);
    GlobalAB globalB(srcB + abOffset);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    TLOAD(tileA, globalA);
    TLOAD(tileB, globalB);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    TADD(tileC, tileA, tileB);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TPUSH<PipeT, VecTileK, SplitAxis>(pipe, tileC);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

// ---------------------------------------------------------------------------------------
// Vector side, phase 2: pop the C2V tile, subtract tileF, store. Consumes the two flags
// runVecPushV2C left armed (see its FLAG CONTRACT) and drains the ledger before returning.
// ---------------------------------------------------------------------------------------
template <typename T, int TOTAL_M, int K, int N, TileSplitAxis SplitAxis, typename PipeT>
AICORE inline void runVecPopC2V(PipeT& pipe, __gm__ T* out, __gm__ T* srcF, uint32_t subBlockIdx)
{
    constexpr uint32_t C2V_ROWS = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? (TOTAL_M / VEC_CORES) : TOTAL_M;
    constexpr uint32_t C2V_COLS = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? (N / VEC_CORES) : N;

    using VecTileN = Tile<TileType::Vec, T, C2V_ROWS, C2V_COLS, BLayout::RowMajor, C2V_ROWS, C2V_COLS>;
    using GlobalFOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, C2V_ROWS, C2V_COLS>, pto::Stride<TOTAL_M * N, TOTAL_M * N, C2V_ROWS * C2V_COLS, N, 1>>;

    VecTileN vecTileHalf, tileF, tileG;
    TASSIGN(tileF, 0x10000);
    TASSIGN(tileG, 0x18000);

    size_t fOutOffset;
    if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
        fOutOffset = static_cast<size_t>(subBlockIdx * C2V_ROWS) * N;
    } else {
        fOutOffset = static_cast<size_t>(subBlockIdx) * C2V_COLS;
    }
    GlobalFOut globalF(srcF + fOutOffset);
    GlobalFOut globalOut(out + fOutOffset);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    TPOP<PipeT, VecTileN, SplitAxis>(pipe, vecTileHalf);
    TLOAD(tileF, globalF);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    TSUB(tileG, vecTileHalf, tileF);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(globalOut, tileG);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    pipe_barrier(PIPE_ALL);
}

// ---------------------------------------------------------------------------------------
// Cube side: move both operands into L0A/L0B and matmul. Called twice with different left
// operands (from GM, then from the V2C pop), which is why it is factored out.
//
// FLAG CONTRACT: the caller must have (PIPE_FIX, PIPE_M, EVENT_ID1) and
// (PIPE_M, PIPE_MTE1, EVENT_ID1) armed. This consumes both and re-arms
// (PIPE_M, PIPE_MTE1, EVENT_ID1) for the next call or the caller's trailing drain.
// Forgetting that re-arm is what makes the core spin forever on the final wait_flag.
// ---------------------------------------------------------------------------------------
template <typename LeftT, typename RightT, typename AccT, typename SrcT, typename MatDT>
AICORE inline void movAndMatmul(LeftT& leftTile, RightT& rightTile, AccT& accTile, SrcT& srcTile, MatDT& matTileD)
{
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    TMOV(leftTile, srcTile);
    TMOV(rightTile, matTileD);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

    TMATMUL(accTile, leftTile, rightTile);

    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
}

// ---------------------------------------------------------------------------------------
// Cube side.
//
// ── The ONLY difference from tpushpop_dir_both ───────────────────────────────
// There, the cube pops the V2C tile and feeds it straight into the matmul whose result it
// then pushes back C2V. That data dependency means the C2V push cannot start until the V2C
// tile has been consumed, so the two directions never hold a tile at the same time and the
// shared-slot aliasing stays hidden.
//
// Here the cube's matmul takes BOTH operands from GM, so the C2V push does not depend on
// the V2C pop, and the push is issued FIRST. Cube and vector then each have one tile in
// flight in the opposite direction simultaneously. Both are tileIndex 0, so both address
// slot 0 of GM_SLOT_BUFFER -- the two payloads land on top of each other and each consumer
// can read a mix of the two.
//
// Nothing here is exotic: the two directions have fully independent flow control (C2V uses
// FlagID/FlagID+1, V2C uses FlagID+2/FlagID+3), so neither producer ever waits on the
// other, and each is independently granted SlotNum credits against a buffer that only has
// SlotNum slots in total.
//
// BOTH directions are checked, on purpose. An earlier version of this case checked only the
// vector's output and PASSED on stock pto-isa: the two payloads are different sizes --
// C2V is a FULL slot (TOTAL_M x N x 4 = 64 KiB = SLOT_SIZE), V2C is HALF a slot
// (TOTAL_M x K x 4 = 32 KiB) -- so whichever push lands second overwrites only part of the
// other tile, and the surviving portion can be enough for one consumer to verify. The
// second matmul below feeds `popTile` into (popTile x D) and stores it, golden
// (A + B) @ D, so the cube's read is observable too. On an affected build BOTH comparisons
// fail. This is the same reason the upstream tpushpop_dir_both case passes: the defect is
// invisible unless you both (a) let the directions overlap and (b) check both directions.
// ---------------------------------------------------------------------------------------
// NOTE: these type aliases must stay INSIDE the function. Lifting them to a namespace-scope
// traits struct does not compile: C0_SIZE_BYTE is a non-dependent name there and is not
// visible at that scope in the mix-kernel TU, whereas inside the function template it is.
template <typename T, int TOTAL_M, int K, int N, TileSplitAxis SplitAxis, typename PipeT>
AICORE inline void runCubeHalf(PipeT& pipe, __gm__ T* srcA, __gm__ T* srcD, __gm__ T* outCube)
{
    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(TOTAL_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(N, blockAlign);

    using PopTileV2C =
        Tile<TileType::Mat, T, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, TOTAL_M, K, SLayout::RowMajor, 512>;
    using TileMatD = Tile<TileType::Mat, T, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using GlobalD = GlobalTensor<T, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalA =
        GlobalTensor<T, pto::Shape<1, 1, 1, TOTAL_M, K>, pto::Stride<TOTAL_M * K, TOTAL_M * K, TOTAL_M * K, K, 1>>;
    using GlobalOutCube =
        GlobalTensor<T, pto::Shape<1, 1, 1, TOTAL_M, N>, pto::Stride<TOTAL_M * N, TOTAL_M * N, TOTAL_M * N, N, 1>>;

    PopTileV2C popTile, matTileA; // matTileA: left operand from GM, not from the V2C pop
    TileMatD matTileD;
    TASSIGN(matTileA, 0x20000);
    TASSIGN(matTileD, 0x40000);

    TileLeft<T, ALIGNED_M, ALIGNED_K, TOTAL_M, K> leftTile;
    TileRight<T, ALIGNED_K, ALIGNED_N, K, N> rightTile;
    TileAcc<T, TOTAL_M, N, TOTAL_M, N> accTile;
    TASSIGN(leftTile, 0x0);
    TASSIGN(rightTile, 0x0);
    TASSIGN(accTile, 0x0);

    GlobalA globalACube(srcA);
    GlobalD globalD(srcD);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    TLOAD(matTileA, globalACube);
    TLOAD(matTileD, globalD);

    movAndMatmul(leftTile, rightTile, accTile, matTileA, matTileD);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    // C2V push FIRST -- independent of anything the vector has sent.
    TPUSH<PipeT, TileAcc<T, TOTAL_M, N, TOTAL_M, N>, SplitAxis>(pipe, accTile);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

    // V2C pop SECOND -- by now both rings hold a live tile in slot 0.
    TPOP<PipeT, PopTileV2C, SplitAxis>(pipe, popTile);

    // Make the popped V2C tile OBSERVABLE. Without this, nothing downstream ever reads the
    // cube's V2C tile, so half the evidence is thrown away. This does NOT re-serialise the
    // directions: the C2V push above already happened.
    movAndMatmul(leftTile, rightTile, accTile, popTile, matTileD);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    GlobalOutCube globalOutCube(outCube);
    TSTORE(globalOutCube, accTile);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

    pipe_barrier(PIPE_ALL);
}

// Computation flow (DIR_BOTH pipe). Unlike tpushpop_dir_both, the two directions OVERLAP:
// the cube's C2V push is issued BEFORE its V2C pop and does not depend on it.
//   TILE_UP_DOWN:    each Vec core handles TOTAL_M/2 rows, full columns
//   TILE_LEFT_RIGHT: each Vec core handles full rows, K/2 or N/2 columns
//
//   Vec:  tileC = tileA + tileB                     (per vector core portion)
//   Vec→Cube (V2C):  TPUSH tileC                    → combined [TOTAL_M, K] in the FIFO
//   Cube: tileE = TMATMUL(A[M,K], D[K,N])           BOTH operands from GM, so this does
//                                                   not depend on anything the vector sent
//   Cube→Vec (C2V):  TPUSH tileE[TOTAL_M, N]        issued BEFORE the pop below
//   Cube: TPOP → popTile, TSTORE TMATMUL(popTile, D) to outCube
//                                                   golden (A + B) @ D  -- the cube's V2C
//                                                   read, checked so it is not silent
//   Vec:  TPOP → tileE_part, tileG = tileE_part - tileF, TSTORE tileG
//                                                   golden A @ D - F
template <typename T, int TOTAL_M, int K, int N, TileSplitAxis SplitAxis = TileSplitAxis::TILE_UP_DOWN>
__global__ AICORE void runTPushPopDirBoth(
    __gm__ uint64_t* ffts_addr, __gm__ T* out, __gm__ T* srcA, __gm__ T* srcB, __gm__ T* srcD, __gm__ T* srcF,
    __gm__ T* fifoMem, __gm__ T* outCube)
{
    set_ffts_base_addr((uint64_t)ffts_addr);

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint32_t SLOT_SIZE = TOTAL_M * N * sizeof(T);

    using BothPipe = TPipe<FLAG_ID, Direction::DIR_BOTH, SLOT_SIZE, FIFO_DEPTH>;

    constexpr uint32_t v2cL1Base = 0x0;
    constexpr uint32_t c2vUBBase = 0x0;

    BothPipe pipe((__gm__ void*)fifoMem, c2vUBBase, v2cL1Base);

    if constexpr (DAV_VEC) {
        uint32_t subBlockIdx = get_subblockid();
        runVecPushV2C<T, TOTAL_M, K, N, SplitAxis, BothPipe>(pipe, srcA, srcB, subBlockIdx);
        runVecPopC2V<T, TOTAL_M, K, N, SplitAxis, BothPipe>(pipe, out, srcF, subBlockIdx);
    }

    if constexpr (DAV_CUBE) {
        runCubeHalf<T, TOTAL_M, K, N, SplitAxis, BothPipe>(pipe, srcA, srcD, outCube);
    }
}

template <int32_t tilingKey>
void LaunchTPushPopDirBoth(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    uint8_t* outCube, void* stream)
{
    if constexpr (tilingKey == 1) {
        runTPushPopDirBoth<float, 128, 64, 128, TileSplitAxis::TILE_UP_DOWN><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(srcD), reinterpret_cast<float*>(srcF),
            reinterpret_cast<float*>(fifoMem), reinterpret_cast<float*>(outCube));
    } else if constexpr (tilingKey == 2) {
        runTPushPopDirBoth<float, 128, 64, 128, TileSplitAxis::TILE_LEFT_RIGHT><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(srcD), reinterpret_cast<float*>(srcF),
            reinterpret_cast<float*>(fifoMem), reinterpret_cast<float*>(outCube));
    }
}

template void LaunchTPushPopDirBoth<1>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    uint8_t* outCube, void* stream);
template void LaunchTPushPopDirBoth<2>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    uint8_t* outCube, void* stream);
