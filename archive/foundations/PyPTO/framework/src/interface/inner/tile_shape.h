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
 * \file tile_shape.h
 * \brief
 */

#pragma once
#include "tilefwk/tile_shape.h"
#include "interface/inner/hash_buffer.h"

namespace npu::tile_fwk {
inline HashBuffer& SerializeTo(const VecTile& vecTile, HashBuffer& hashBuffer)
{
    hashBuffer.assign(vecTile.tile.begin(), vecTile.tile.end());
    return hashBuffer;
}

inline HashBuffer& SerializeTo(const CubeTile& cubeTile, HashBuffer& hashBuffer)
{
    hashBuffer.Append(cubeTile.m);
    hashBuffer.Append(cubeTile.k);
    hashBuffer.Append(cubeTile.n);
    return hashBuffer;
}

inline HashBuffer& SerializeTo(const DistTile& distTile, HashBuffer& hashBuffer)
{
    hashBuffer.Append(distTile.row);
    hashBuffer.Append(distTile.col);
    hashBuffer.Append(distTile.rank);
    hashBuffer.Append(distTile.rankId);
    return hashBuffer;
}

inline void DeserializeFrom(HashBuffer& hashBuffer, VecTile& vecTile)
{
    vecTile.tile.assign(hashBuffer.begin(), hashBuffer.end());
}

inline void DeserializeFrom(HashBuffer& hashBuffer, CubeTile& cubeTile)
{
    cubeTile.m[0] = hashBuffer.Get<int64_t>(0);  // offset 0,  m[0]
    cubeTile.m[1] = hashBuffer.Get<int64_t>(2);  // offset 2,  m[1]
    cubeTile.k[0] = hashBuffer.Get<int64_t>(4);  // offset 4,  k[0]
    cubeTile.k[1] = hashBuffer.Get<int64_t>(6);  // offset 6,  k[1]
    cubeTile.k[2] = hashBuffer.Get<int64_t>(8);  // offset 8,  k[2]
    cubeTile.n[0] = hashBuffer.Get<int64_t>(10); // offset 10, n[0]
    cubeTile.n[1] = hashBuffer.Get<int64_t>(12); // offset 12, n[1]
}

inline void DeserializeFrom(HashBuffer& hashBuffer, DistTile& distTile)
{
    distTile.row[0] = hashBuffer[0];  // offset 0,  row[0]
    distTile.row[1] = hashBuffer[1];  // offset 1,  row[1]
    distTile.row[2] = hashBuffer[2];  // offset 2,  row[2]
    distTile.col[0] = hashBuffer[3];  // offset 3,  col[0]
    distTile.col[1] = hashBuffer[4];  // offset 4,  col[1]
    distTile.col[2] = hashBuffer[5];  // offset 5,  col[2]
    distTile.rank[0] = hashBuffer[6]; // offset 6,  rank[0]
    distTile.rank[1] = hashBuffer[7]; // offset 7,  rank[1]
    distTile.rank[2] = hashBuffer[8]; // offset 8,  rank[2]
    distTile.rankId = hashBuffer[9];  // offset 9,  rankId
}
}; // namespace npu::tile_fwk
