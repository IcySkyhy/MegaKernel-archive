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
 * \file SoftMemory.h
 * \brief
 */

#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <memory>

namespace CostModel {
class SoftMemory {
public:
    // 内存块结构
    struct MemoryBlock {
        uintptr_t startAddr;
        size_t size;
        bool isFree;
        std::string regionType; // "heap", "stack", "data"
        MemoryBlock(uintptr_t addr, size_t sz, bool free, const std::string& type)
            : startAddr(addr), size(sz), isFree(free), regionType(type)
        {}
    };
    static SoftMemory& Instance()
    {
        static SoftMemory softMemory;
        return softMemory;
    }

    // 构造函数，初始化内存布局
    SoftMemory()
    {
        // 模拟Linux内存布局
        // 0x00000000 - 0x08048000: 保留区域
        // 0x08048000 - 0x08049000: 代码段
        // 0x08049000 - 0x0804a000: 数据段
        // 0x0804a000 - 0x0804b000: BSS段
        // 堆从0x0804b000向上增长
        // 栈从0xc0000000向下增长

        // 初始化一些固定区域
        memoryMap[0x08048000] = std::vector<uint8_t>(0x1000, 0); // 代码段
        memoryMap[0x08049000] = std::vector<uint8_t>(0x1000, 0); // 数据段
        memoryMap[0x0804a000] = std::vector<uint8_t>(0x1000, 0); // BSS段

        // 初始化内存块列表
        memoryBlocks.emplace_back(0x08048000, 0x1000, false, "text");
        memoryBlocks.emplace_back(0x08049000, 0x1000, false, "data");
        memoryBlocks.emplace_back(0x0804a000, 0x1000, false, "bss");

        // 堆起始地址
        heapStart = 0x0804b000;
        heapEnd = heapStart;

        // 栈起始地址
        stackStart = 0xc0000000;
        stackEnd = stackStart;
    }

    // 分配堆内存
    uintptr_t AllocateHeap(size_t size)
    {
        if (!enable) {
            return 0;
        }
        // 堆向上增长
        uintptr_t addr = heapEnd;
        heapEnd += size;

        // 检查是否有足够的空间
        if (heapEnd >= stackEnd) {
            throw std::runtime_error("Out of memory: heap exhausted");
        }

        // 添加到内存映射
        memoryMap[addr] = std::vector<uint8_t>(size, 0);
        memoryBlocks.emplace_back(addr, size, false, "heap");

        return addr;
    }

    // 分配栈内存
    uintptr_t AllocateStack(size_t size)
    {
        if (!enable) {
            return 0;
        }
        // 栈向下增长
        stackEnd -= size;

        // 检查是否有足够的空间
        if (stackEnd <= heapEnd) {
            throw std::runtime_error("Out of memory: stack exhausted");
        }

        uintptr_t addr = stackEnd;

        // 添加到内存映射
        memoryMap[addr] = std::vector<uint8_t>(size, 0);
        memoryBlocks.emplace_back(addr, size, false, "stack");

        return addr;
    }

    // 分配数据块
    uintptr_t AllocateData(size_t size, const std::vector<uint8_t>& data)
    {
        if (!enable) {
            return 0;
        }
        // 数据块从堆和栈之间的区域分配
        // 使用首次适应算法

        // 按地址排序内存块
        std::sort(memoryBlocks.begin(), memoryBlocks.end(),
                  [](const MemoryBlock& a, const MemoryBlock& b) { return a.startAddr < b.startAddr; });

        // 寻找合适的空闲区域
        for (size_t i = 0; i < memoryBlocks.size() - 1; ++i) {
            uintptr_t gapStart = memoryBlocks[i].startAddr + memoryBlocks[i].size;
            uintptr_t gapEnd = memoryBlocks[i + 1].startAddr;
            size_t gapSize = gapEnd - gapStart;

            if (gapSize >= size) {
                // 找到合适的间隙
                uintptr_t addr = gapStart;
                memoryMap[addr] = data;
                memoryMap[addr] = std::vector<uint8_t>(size, 0);
                memoryBlocks.emplace_back(addr, size, false, "data");
                return addr;
            }
        }

        // 没有找到合适的间隙，尝试在堆之后分配
        uintptr_t addr = heapEnd;
        heapEnd += size;

        if (heapEnd >= stackEnd) {
            throw std::runtime_error("Out of memory: data allocation failed");
        }

        memoryMap[addr] = data;
        memoryBlocks.emplace_back(addr, size, false, "data");
        return addr;
    }

    // 释放内存
    void Deallocate(uintptr_t addr)
    {
        auto it = memoryMap.find(addr);
        if (it != memoryMap.end()) {
            // 标记为空闲
            for (auto& block : memoryBlocks) {
                if (block.startAddr == addr) {
                    block.isFree = true;
                    break;
                }
            }

            // 从内存映射中移除
            memoryMap.erase(it);
        }
    }

    // 打印内存布局
    void PrintMemoryLayout() const
    {
        std::cout << "Memory Layout:\n";
        std::cout << "--------------------------------------------------\n";

        // 按地址排序内存块
        std::vector<MemoryBlock> sorted_blocks = memoryBlocks;
        std::sort(sorted_blocks.begin(), sorted_blocks.end(),
                  [](const MemoryBlock& a, const MemoryBlock& b) { return a.startAddr < b.startAddr; });

        for (const auto& block : sorted_blocks) {
            std::cout << "0x" << std::hex << block.startAddr << " - 0x" << (block.startAddr + block.size) << " ("
                      << std::dec << block.size << " bytes) " << block.regionType
                      << (block.isFree ? " [FREE]" : " [USED]") << "\n";
        }

        std::cout << "--------------------------------------------------\n";
        std::cout << "Heap top: 0x" << std::hex << heapEnd << "\n";
        std::cout << "Stack bottom: 0x" << std::hex << stackEnd << "\n";
        std::cout << "--------------------------------------------------\n";
    }

    void Enable() { enable = true; }

private:
    std::map<uintptr_t, std::vector<uint8_t>> memoryMap;
    std::vector<MemoryBlock> memoryBlocks;
    uintptr_t heapStart;
    uintptr_t heapEnd;
    uintptr_t stackStart;
    uintptr_t stackEnd;
    bool enable = false;
};
} // namespace CostModel
