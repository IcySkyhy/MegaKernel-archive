/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_hash_buffer.cpp
 * \brief Unit tests for HashBuffer Get<> specializations and Append/Digest (interface/inner/hash_buffer.h,
 *        interface/cache/hash_buffer.cpp)
 */

#include <cstdint>
#include <climits>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "interface/inner/hash_buffer.h"

using namespace npu::tile_fwk;

class TestHashBuffer : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestHashBuffer, GetUint64SmallValue)
{
    HashBuffer buf(static_cast<uint64_t>(42));
    EXPECT_EQ(buf.Get<uint64_t>(0), 42ULL);
}

TEST_F(TestHashBuffer, GetUint64Zero)
{
    HashBuffer buf(static_cast<uint64_t>(0));
    EXPECT_EQ(buf.Get<uint64_t>(0), 0ULL);
}

TEST_F(TestHashBuffer, GetUint64MaxValue)
{
    HashBuffer buf(UINT64_MAX);
    EXPECT_EQ(buf.Get<uint64_t>(0), UINT64_MAX);
}

TEST_F(TestHashBuffer, GetUint64HighBitsOnly)
{
    uint64_t val = 1ULL << 63;
    HashBuffer buf(val);
    EXPECT_EQ(buf.Get<uint64_t>(0), val);
}

TEST_F(TestHashBuffer, GetInt64Positive)
{
    HashBuffer buf(static_cast<int64_t>(42));
    EXPECT_EQ(buf.Get<int64_t>(0), 42);
}

TEST_F(TestHashBuffer, GetInt64Negative)
{
    HashBuffer buf(static_cast<int64_t>(-1));
    EXPECT_EQ(buf.Get<int64_t>(0), -1);
}

TEST_F(TestHashBuffer, GetInt64MinValue)
{
    HashBuffer buf(INT64_MIN);
    EXPECT_EQ(buf.Get<int64_t>(0), INT64_MIN);
}

TEST_F(TestHashBuffer, MultipleAppendGetUint64)
{
    HashBuffer buf;
    buf.Update(static_cast<uint64_t>(10), static_cast<uint64_t>(20), static_cast<uint64_t>(30));
    EXPECT_EQ(buf.Get<uint64_t>(0), 10ULL);
    EXPECT_EQ(buf.Get<uint64_t>(2), 20ULL);
    EXPECT_EQ(buf.Get<uint64_t>(4), 30ULL);
}

TEST_F(TestHashBuffer, MultipleAppendGetInt64)
{
    HashBuffer buf;
    buf.Update(static_cast<int64_t>(-5), static_cast<int64_t>(100), static_cast<int64_t>(-999));
    EXPECT_EQ(buf.Get<int64_t>(0), -5);
    EXPECT_EQ(buf.Get<int64_t>(2), 100);
    EXPECT_EQ(buf.Get<int64_t>(4), -999);
}

TEST_F(TestHashBuffer, AppendInt32DirectAccess)
{
    HashBuffer buf(static_cast<int32_t>(7));
    EXPECT_EQ(buf.size(), 1UL);
    EXPECT_EQ(static_cast<int32_t>(buf[0]), 7);
}

TEST_F(TestHashBuffer, DigestConsistency)
{
    HashBuffer buf(static_cast<uint64_t>(12345));
    auto d1 = buf.Digest();
    auto d2 = buf.Digest();
    EXPECT_EQ(d1, d2);
}

TEST_F(TestHashBuffer, DigestDifferentContent)
{
    HashBuffer buf1(static_cast<uint64_t>(111));
    HashBuffer buf2(static_cast<uint64_t>(222));
    EXPECT_NE(buf1.Digest(), buf2.Digest());
}

TEST_F(TestHashBuffer, AppendString)
{
    HashBuffer buf;
    buf.Append(std::string("abc"));
    EXPECT_EQ(buf.size(), 3UL);
}

TEST_F(TestHashBuffer, AppendVectorInt32)
{
    std::vector<int32_t> v = {1, 2, 3};
    HashBuffer buf;
    buf.Append(v);
    EXPECT_EQ(buf.size(), 3UL);
    for (size_t i = 0; i < v.size(); i++) {
        EXPECT_EQ(static_cast<int32_t>(buf[i]), v[i]);
    }
}

TEST_F(TestHashBuffer, AppendVectorUint64)
{
    std::vector<uint64_t> v = {0x100000000ULL, 0ULL};
    HashBuffer buf;
    buf.Append(v);
    EXPECT_EQ(buf.size(), 4UL);
    EXPECT_EQ(buf.Get<uint64_t>(0), v[0]);
    EXPECT_EQ(buf.Get<uint64_t>(2), v[1]);
}

TEST_F(TestHashBuffer, EmptyBufferDigest)
{
    HashBuffer buf;
    auto d = buf.Digest();
    EXPECT_NE(d, 0UL);
}

TEST_F(TestHashBuffer, MixedAppendAndGet)
{
    HashBuffer buf;
    buf.Append(static_cast<uint64_t>(0xDEADBEEFCAFE1234ULL));
    buf.Append(static_cast<int32_t>(42));
    EXPECT_EQ(buf.Get<uint64_t>(0), 0xDEADBEEFCAFE1234ULL);
}
