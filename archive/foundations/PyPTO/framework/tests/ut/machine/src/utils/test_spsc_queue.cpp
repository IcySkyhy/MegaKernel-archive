/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <functional>
#include <thread>

#include "machine/utils/dynamic/spsc_queue.h"

class SPSCQueueTest : public testing::Test {};

TEST_F(SPSCQueueTest, FreeUntil_EmptyQueue_ReturnsFalse)
{
    SPSCQueue<int*, 8> queue;
    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = false;
        return true;
    });
    EXPECT_FALSE(result);
}

TEST_F(SPSCQueueTest, FreeUntil_SingleElement_CanFree)
{
    SPSCQueue<int*, 8> queue;
    int val = 42;
    queue.Enqueue(&val);

    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_SingleElement_CannotFree_NoContinue)
{
    SPSCQueue<int*, 8> queue;
    int val = 42;
    queue.Enqueue(&val);

    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = false;
        return false;
    });
    EXPECT_FALSE(result);
    EXPECT_FALSE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_SingleElement_CannotFree_Continue)
{
    SPSCQueue<int*, 8> queue;
    int val = 42;
    queue.Enqueue(&val);

    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = true;
        return false;
    });
    EXPECT_FALSE(result);
}

TEST_F(SPSCQueueTest, FreeUntil_MultipleElements_FreeFirst)
{
    SPSCQueue<int*, 8> queue;
    int v1 = 1, v2 = 2, v3 = 3;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    queue.Enqueue(&v3);

    int callCount = 0;
    bool result = queue.FreeUntil([&callCount]([[maybe_unused]] int* const& elem, bool& continueNext) -> bool {
        callCount++;
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
    EXPECT_EQ(callCount, 3);
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_MultipleElements_FreeAll)
{
    SPSCQueue<int*, 8> queue;
    int v1 = 1, v2 = 2, v3 = 3;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    queue.Enqueue(&v3);

    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_NullElement_SetsNull)
{
    SPSCQueue<int*, 8> queue;
    int v1 = 1, v2 = 2;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);

    int callIdx = 0;
    bool result = queue.FreeUntil([&callIdx]([[maybe_unused]] int* const& elem, bool& continueNext) -> bool {
        callIdx++;
        if (callIdx == 1) {
            continueNext = true;
            return false;
        }
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
}

TEST_F(SPSCQueueTest, FreeUntil_StopAtCannotFree)
{
    SPSCQueue<int*, 8> queue;
    int v1 = 1, v2 = 2, v3 = 3;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    queue.Enqueue(&v3);

    int callCount = 0;
    bool result = queue.FreeUntil([&callCount]([[maybe_unused]] int* const& elem, bool& continueNext) -> bool {
        callCount++;
        if (callCount == 2) {
            continueNext = false;
            return false;
        }
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
    EXPECT_EQ(callCount, 2);
}

TEST_F(SPSCQueueTest, EnqueueDequeue_Basic)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 10, v2 = 20;
    EXPECT_TRUE(queue.TryEnqueue(&v1));
    EXPECT_TRUE(queue.TryEnqueue(&v2));

    int* val = nullptr;
    EXPECT_TRUE(queue.TryDequeue(val));
    EXPECT_EQ(val, &v1);
    EXPECT_TRUE(queue.TryDequeue(val));
    EXPECT_EQ(val, &v2);
    EXPECT_FALSE(queue.TryDequeue(val));
}

TEST_F(SPSCQueueTest, TryEnqueue_Full_ReturnsFalse)
{
    SPSCQueue<int*, 2> queue;
    int v1 = 1, v2 = 2, v3 = 3;
    EXPECT_TRUE(queue.TryEnqueue(&v1));
    EXPECT_TRUE(queue.TryEnqueue(&v2));
    EXPECT_FALSE(queue.TryEnqueue(&v3));
}

TEST_F(SPSCQueueTest, PopFront_RemovesHead)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 1, v2 = 2;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    queue.PopFront();
    int* val = nullptr;
    EXPECT_TRUE(queue.TryDequeue(val));
    EXPECT_EQ(val, &v2);
}

TEST_F(SPSCQueueTest, PopFront_EmptyQueue_NoOp)
{
    SPSCQueue<int*, 4> queue;
    queue.PopFront();
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, TempDequeue_DoesNotRemove)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 1;
    queue.Enqueue(&v1);
    int* val = nullptr;
    EXPECT_TRUE(queue.TempDequeue(val));
    EXPECT_EQ(val, &v1);
    EXPECT_FALSE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, TempDequeue_Empty_ReturnsFalse)
{
    SPSCQueue<int*, 4> queue;
    int* val = nullptr;
    EXPECT_FALSE(queue.TempDequeue(val));
}

TEST_F(SPSCQueueTest, ResetEmpty_ClearsQueue)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 1;
    queue.Enqueue(&v1);
    EXPECT_FALSE(queue.IsEmpty());
    queue.ResetEmpty();
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, IsEmpty_InitiallyTrue)
{
    SPSCQueue<int*, 4> queue;
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_HeadCannotFree_SecondCanFree)
{
    SPSCQueue<int*, 8> queue;
    int v1 = 1, v2 = 2, v3 = 3;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    queue.Enqueue(&v3);

    int callCount = 0;
    bool result = queue.FreeUntil([&callCount]([[maybe_unused]] int* const& elem, bool& continueNext) -> bool {
        callCount++;
        if (callCount == 1) {
            continueNext = true;
            return false;
        }
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
    EXPECT_FALSE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_WrapAroundCircularBuffer)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 1, v2 = 2, v3 = 3, v4 = 4;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    int* tmp = nullptr;
    queue.TryDequeue(tmp);
    queue.TryDequeue(tmp);
    queue.Enqueue(&v3);
    queue.Enqueue(&v4);

    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_AlternatingFreeAndContinue)
{
    SPSCQueue<int*, 8> queue;
    int v1 = 1, v2 = 2, v3 = 3, v4 = 4;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    queue.Enqueue(&v3);
    queue.Enqueue(&v4);

    int callCount = 0;
    bool result = queue.FreeUntil([&callCount]([[maybe_unused]] int* const& elem, bool& continueNext) -> bool {
        callCount++;
        if (callCount % 2 == 0) {
            continueNext = true;
            return false;
        }
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
}

TEST_F(SPSCQueueTest, FreeUntil_AllCannotFree_ContinueTrue)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 1, v2 = 2;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);

    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = true;
        return false;
    });
    EXPECT_FALSE(result);
    EXPECT_FALSE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, FreeUntil_FullQueue_AllFreeable)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 1, v2 = 2, v3 = 3, v4 = 4;
    queue.Enqueue(&v1);
    queue.Enqueue(&v2);
    queue.Enqueue(&v3);
    queue.Enqueue(&v4);

    bool result = queue.FreeUntil([](int* const& elem, bool& continueNext) -> bool {
        (void)elem;
        continueNext = false;
        return true;
    });
    EXPECT_TRUE(result);
    EXPECT_TRUE(queue.IsEmpty());
}

TEST_F(SPSCQueueTest, Enqueue_BlockingAfterFull)
{
    SPSCQueue<int*, 2> queue;
    int v1 = 1, v2 = 2;
    EXPECT_TRUE(queue.TryEnqueue(&v1));
    EXPECT_TRUE(queue.TryEnqueue(&v2));
    EXPECT_FALSE(queue.TryEnqueue(&v1));
    int* tmp = nullptr;
    EXPECT_TRUE(queue.TryDequeue(tmp));
    EXPECT_TRUE(queue.TryEnqueue(&v1));
}

TEST_F(SPSCQueueTest, Dequeue_BlockingOnEmpty)
{
    SPSCQueue<int*, 4> queue;
    int v1 = 42;
    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        queue.Enqueue(&v1);
    });
    int* val = queue.Dequeue();
    EXPECT_EQ(val, &v1);
    producer.join();
}
