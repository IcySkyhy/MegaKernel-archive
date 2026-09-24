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
 * \file test_queues.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <vector>
#include "machine/utils/queues.h"

using namespace npu::tile_fwk;

TEST(QueueGenericTest, CapacityAndSize)
{
    std::vector<uint32_t> buf(8, 0);
    QueueGeneric<uint32_t> q(8, buf.data());
    EXPECT_EQ(q.Capacity(), 8u);
    EXPECT_EQ(q.Size(), 0u);
}

TEST(QueueGenericTest, Str_ContainsHeadTailCapacity)
{
    std::vector<uint32_t> buf(4, 0);
    QueueGeneric<uint32_t> q(4, buf.data());
    auto s = q.Str();
    EXPECT_NE(s.find("head=0"), std::string::npos);
    EXPECT_NE(s.find("tail=0"), std::string::npos);
    EXPECT_NE(s.find("capacity=4"), std::string::npos);
}

TEST(QueueGenericTest, Dump_ReturnsSpaceSeparatedElements)
{
    std::vector<uint32_t> buf(4, 0);
    buf[0] = 10;
    buf[1] = 20;
    QueueGeneric<uint32_t> q(4, buf.data());
    q.head_ = 0;
    q.tail_ = 2;
    auto s = q.Dump();
    EXPECT_NE(s.find("10"), std::string::npos);
    EXPECT_NE(s.find("20"), std::string::npos);
}

TEST(QueueGenericTest, BeginEndIterators)
{
    std::vector<uint32_t> buf(4, 0);
    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    QueueGeneric<uint32_t> q(4, buf.data());
    q.head_ = 0;
    q.tail_ = 3;
    auto it = q.begin();
    EXPECT_EQ(*it, 1u);
    size_t count = 0;
    for (auto v = q.begin(); v != q.end(); ++v) {
        count++;
    }
    EXPECT_EQ(count, 3u);
}

TEST(QueueGenericTest, AssignmentOperator_CopiesElements)
{
    std::vector<uint32_t> buf1(4, 0);
    std::vector<uint32_t> buf2(4, 0);
    buf1[0] = 100;
    buf1[1] = 200;
    QueueGeneric<uint32_t> src(4, buf1.data());
    src.head_ = 0;
    src.tail_ = 2;
    QueueGeneric<uint32_t> dst(4, buf2.data());
    dst = src;
    EXPECT_EQ(dst.Size(), 2u);
    EXPECT_EQ(buf2[0], 100u);
    EXPECT_EQ(buf2[1], 200u);
}

TEST(QueueGenericTest, AssignmentOperator_ZeroCapacity_Noop)
{
    std::vector<uint32_t> buf1(4, 0);
    QueueGeneric<uint32_t> src(0, buf1.data());
    src.head_ = 0;
    src.tail_ = 0;
    std::vector<uint32_t> buf2(4, 0);
    QueueGeneric<uint32_t> dst(0, buf2.data());
    dst = src;
    EXPECT_EQ(dst.Size(), 0u);
}

TEST(LockableQueueGenericTest, LockUnlock)
{
    std::vector<uint32_t> buf(4, 0);
    LockableQueueGeneric<uint32_t> q(4, buf.data());
    q.lock();
    q.unlock();
    SUCCEED();
}

TEST(LockableQueueGenericTest, UnsafeEnqueueAndSize)
{
    std::vector<uint32_t> buf(4, 0);
    LockableQueueGeneric<uint32_t> q(4, buf.data());
    q.UnsafeEnqueue(10);
    q.UnsafeEnqueue(20);
    EXPECT_EQ(q.UnsafeSize(), 2u);
    EXPECT_EQ(buf[0], 10u);
    EXPECT_EQ(buf[1], 20u);
}

TEST(LockableQueueGenericTest, TryEnqueue_Success)
{
    std::vector<uint32_t> buf(4, 0);
    LockableQueueGeneric<uint32_t> q(4, buf.data());
    EXPECT_TRUE(q.TryEnqueue(42));
    EXPECT_EQ(buf[0], 42u);
}

TEST(LockableQueueGenericTest, TryEnqueue_Overflow_ReturnsFalse)
{
    std::vector<uint32_t> buf(2, 0);
    LockableQueueGeneric<uint32_t> q(2, buf.data());
    EXPECT_TRUE(q.TryEnqueue(1));
    EXPECT_TRUE(q.TryEnqueue(2));
    EXPECT_FALSE(q.TryEnqueue(3));
}

TEST(LockableQueueGenericTest, UnsafeEnqueueBatch)
{
    std::vector<uint32_t> buf(8, 0);
    LockableQueueGeneric<uint32_t> q(8, buf.data());
    uint32_t data[] = {100, 200, 300};
    q.UnsafeEnqueue(data, 3);
    EXPECT_EQ(q.UnsafeSize(), 3u);
    EXPECT_EQ(buf[0], 100u);
    EXPECT_EQ(buf[1], 200u);
    EXPECT_EQ(buf[2], 300u);
}

TEST(LockableQueueGenericTest, DequeueAll)
{
    std::vector<uint32_t> buf(8, 0);
    LockableQueueGeneric<uint32_t> q(8, buf.data());
    q.UnsafeEnqueue(1);
    q.UnsafeEnqueue(2);
    q.UnsafeEnqueue(3);
    auto range = q.DequeueAll();
    EXPECT_EQ(range.second - range.first, 3u);
    EXPECT_EQ(q.UnsafeAtomicSize(), 0u);
}

TEST(LockableQueueGenericTest, Dequeue_PartialCount)
{
    std::vector<uint32_t> buf(8, 0);
    LockableQueueGeneric<uint32_t> q(8, buf.data());
    q.UnsafeEnqueue(10);
    q.UnsafeEnqueue(20);
    q.UnsafeEnqueue(30);
    auto range = q.Dequeue(2);
    EXPECT_EQ(range.second - range.first, 2u);
    EXPECT_EQ(*range.first, 10u);
    EXPECT_EQ(*(range.first + 1), 20u);
}

TEST(LockableQueueGenericTest, Dequeue_EmptyQueue_ReturnsNull)
{
    std::vector<uint32_t> buf(8, 0);
    LockableQueueGeneric<uint32_t> q(8, buf.data());
    auto range = q.Dequeue(5);
    EXPECT_EQ(range.first, nullptr);
    EXPECT_EQ(range.second, nullptr);
}

TEST(LockableQueueGenericTest, DequeueTail)
{
    std::vector<uint32_t> buf(8, 0);
    LockableQueueGeneric<uint32_t> q(8, buf.data());
    q.UnsafeEnqueue(10);
    q.UnsafeEnqueue(20);
    q.UnsafeEnqueue(30);
    uint32_t out[4] = {0};
    auto range = q.DequeueTail(2, out);
    EXPECT_EQ(range.second - range.first, 2u);
    EXPECT_EQ(out[0], 20u);
    EXPECT_EQ(out[1], 30u);
}

TEST(LockableQueueGenericTest, UnsafeAtomicSize)
{
    std::vector<uint32_t> buf(4, 0);
    LockableQueueGeneric<uint32_t> q(4, buf.data());
    EXPECT_EQ(q.UnsafeAtomicSize(), 0u);
    q.UnsafeEnqueue(1);
    EXPECT_EQ(q.UnsafeAtomicSize(), 1u);
}
