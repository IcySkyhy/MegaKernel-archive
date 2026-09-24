/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile_shape.cpp
 * \brief
 */

#include "tilefwk/tile_shape.h"
#include "interface/configs/config_manager_ng.h"

using namespace npu::tile_fwk;

bool VecTile::valid() const
{
    return std::all_of(tile.begin(), tile.end(), [](int x) { return x > 0; }) && tile.size() > 0;
}

bool CubeTile::valid() const
{
    return std::all_of(m.begin(), m.end(), [](int64_t x) { return x > 0; }) &&
           std::all_of(k.begin(), k.end(), [](int64_t x) { return x > 0; }) &&
           std::all_of(n.begin(), n.end(), [](int64_t x) { return x > 0; });
}

std::string CubeTile::ToString() const
{
    std::stringstream ss;
    ss << "CubeTile: " << '{' << "m: {" << m[0] << ", " << m[1] << '}' << ", "
       << "k: {" << k[0] << ", " << k[1] << ", " << k[0x2] << '}' << ", "
       << "n: {" << n[0] << ", " << n[1] << '}' << ", "
       << "enableSplitK: " << enableSplitK << "}; ";
    return ss.str();
}

bool ConvTile::valid() const
{
    if (tileL1Info.tileHin <= 0 || tileL1Info.tileHout <= 0 || tileL1Info.tileWin <= 0 || tileL1Info.tileWout <= 0 ||
        tileL1Info.tileCinFmap <= 0 || tileL1Info.tileCinWeight <= 0 || tileL1Info.tileN <= 0 ||
        tileL1Info.tileBatch <= 0) {
        return false;
    }

    if (setL0Tile) {
        if (tileL0Info.tileH <= 0 || tileL0Info.tileW <= 0 || tileL0Info.tileK <= 0 || tileL0Info.tileN <= 0) {
            return false;
        }
    }
    return true;
}

std::string ConvTile::ToString() const
{
    std::stringstream ss;
    ss << "ConvTile: " << '{' << "tileL1Info: {"
       << "tileHin: " << tileL1Info.tileHin << ", "
       << "tileHout: " << tileL1Info.tileHout << ", "
       << "tileWin: " << tileL1Info.tileWin << ", "
       << "tileWout: " << tileL1Info.tileWout << ", "
       << "tileCinFmap: " << tileL1Info.tileCinFmap << ", "
       << "tileCinWeight: " << tileL1Info.tileCinWeight << ", "
       << "tileN: " << tileL1Info.tileN << ", "
       << "tileBatch: " << tileL1Info.tileBatch << "}, "
       << "tileL0Info: {"
       << "tileH: " << tileL0Info.tileH << ", "
       << "tileW: " << tileL0Info.tileW << ", "
       << "tileK: " << tileL0Info.tileK << ", "
       << "tileN: " << tileL0Info.tileN << "}, "
       << "setL0Tile: " << (setL0Tile ? "true" : "false") << "};";
    return ss.str();
}

bool DistTile::valid() const
{
    return std::all_of(row.begin(), row.end(), [](int x) { return x > 0; }) &&
           std::all_of(col.begin(), col.end(), [](int x) { return x > 0; }) &&
           std::all_of(rank.begin(), rank.end(), [](int x) { return x > 0; }) && rankId >= 0;
}

std::string DistTile::ToString() const
{
    std::stringstream ss;
    ss << "DistTile: " << '{' << "row: {" << row[0] << ", " << row[1] << ", " << row[0x2] << "}, "
       << "col: {" << col[0] << ", " << col[1] << ", " << col[0x2] << "}, "
       << "rank: {" << rank[0] << ", " << rank[1] << ", " << rank[0x2] << "}, "
       << "rankId: " << rankId << "}; ";
    return ss.str();
}

TileShape::TileShape() : vecTile{}, cubeTile{}, convTile{}, distTile{}, matrixSize{} {}

TileShape::TileShape(const std::vector<int64_t>& vTile, const CubeTile& cTile, const ConvTile& cvTile,
                     const DistTile& dTile, const std::vector<int64_t>& mSize)
    : vecTile{vTile}, cubeTile(cTile), convTile(cvTile), distTile(dTile), matrixSize(mSize)
{}

TileShape& TileShape::Current()
{
    static TileShape instance;
    instance = ConfigManagerNg::CurrentScope()->GenerateTileShape();
    return instance;
}

void TileShape::SetConvTile(const Conv::TileL1Info& tileL1Info, const Conv::TileL0Info& tileL0Info, bool setL0Tile)
{
    convTile = {tileL1Info, tileL0Info, setL0Tile};
    ConfigManagerNg::CurrentScope()->UpdateValue("conv_tile_shapes", convTile);
}

void TileShape::SetVecTile(const std::vector<int64_t>& tile)
{
    vecTile = {tile};
    ConfigManagerNg::CurrentScope()->UpdateValue("vec_tile_shapes", tile);
}

void TileShape::SetVecTile(const VecTile& tile)
{
    vecTile = tile;
    ConfigManagerNg::CurrentScope()->UpdateValue("vec_tile_shapes", tile.tile);
}

void TileShape::SetCubeTile(const std::array<int64_t, MAX_M_DIM_SIZE>& m, const std::array<int64_t, MAX_K_DIM_SIZE>& k,
                            const std::array<int64_t, MAX_N_DIM_SIZE>& n, bool enableSplitK)
{
    auto nk = k;
    if (nk[2] == 0) {
        nk[2] = nk[1];
    }
    cubeTile = {m, nk, n, enableSplitK};
    ConfigManagerNg::CurrentScope()->UpdateValue("cube_tile_shapes", cubeTile);
}

void TileShape::SetMatrixSize(const std::vector<int64_t>& size)
{
    this->matrixSize = size;
    ConfigManagerNg::CurrentScope()->UpdateValue("matrix_size", size);
}

void TileShape::UpdateScopeDistTile() { ConfigManagerNg::CurrentScope()->UpdateValue("dist_tile_shapes", distTile); }

void TileShape::SetDistTile(const std::array<int, MAX_DIST_DIM_SIZE>& row,
                            const std::array<int, MAX_DIST_DIM_SIZE>& col,
                            const std::array<int, MAX_DIST_DIM_SIZE>& rank)
{
    distTile.row = row;
    distTile.col = col;
    distTile.rank = rank;
    UpdateScopeDistTile();
}

void TileShape::SetDistRankId(int64_t rankId)
{
    distTile.rankId = rankId;
    UpdateScopeDistTile();
}

void TileShape::SetDistTileCol(const std::array<int, MAX_DIST_DIM_SIZE>& col)
{
    distTile.col = col;
    UpdateScopeDistTile();
}

void TileShape::SetDistTileRow(const std::array<int, MAX_DIST_DIM_SIZE>& row)
{
    distTile.row = row;
    UpdateScopeDistTile();
}

void TileShape::SetDistTileRank(const std::array<int, MAX_DIST_DIM_SIZE>& rank)
{
    distTile.rank = rank;
    UpdateScopeDistTile();
}

std::string TileShape::ToString(TileType type) const
{
    std::stringstream ss;
    if (type == TileType::VEC || type == TileType::MAX) {
        ss << "VecTile: " << '{';
        for (size_t i = 0; i < vecTile.tile.size(); ++i) {
            if (i != 0)
                ss << ", ";
            ss << vecTile.tile[i];
        }
        ss << "}; ";
    }
    if (type == TileType::CUBE || type == TileType::MAX) {
        ss << cubeTile.ToString();
    }
    if (type == TileType::CONV || type == TileType::MAX) {
        ss << convTile.ToString();
    }
    if (type == TileType::DIST || type == TileType::MAX) {
        ss << distTile.ToString();
    }
    return ss.str();
}
