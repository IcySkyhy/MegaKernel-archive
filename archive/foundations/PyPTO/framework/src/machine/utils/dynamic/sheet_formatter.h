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
 * \file sheet_formatter.h
 * \brief
 */

#pragma once

#include "machine/utils/device_log.h"

#include <string>
#include <vector>
#include <ostream>
#include <sstream>
#include <iomanip>

#include <cstdint>
#include <cmath>

namespace npu::tile_fwk {

namespace sheet {

inline std::string Integer(int64_t value)
{
    std::ostringstream oss;
    oss << value;
    return std::move(oss).str();
}

inline std::string HexaInteger(int64_t value)
{
    std::ostringstream oss;
    oss << std::hex << value;
    return std::move(oss).str();
}

template <int decimalPrecision = 1>
std::string FixedFloat(float value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimalPrecision) << value;
    return std::move(oss).str();
}

} // namespace sheet

class SheetFormatter {
public:
    explicit SheetFormatter(const std::vector<std::string>& titles, char boundary = '=', char separator = '-')
        : columnTitles_(titles), boundary_(boundary), separator_(separator)
    {}

    template <typename... Args>
    void AddRow(Args&&... args)
    {
        DEV_ASSERT_MSG(DevDataErr::SHEET_COLUMN_MISMATCH, sizeof...(Args) == columnTitles_.size(),
                       "sizeof...(Args)=%zu != columnTitles_.size()=%zu", sizeof...(Args), columnTitles_.size());
        rows_.emplace_back();
        rows_.back().reserve(sizeof...(Args));
        auto toString = [](auto&& arg) {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, std::string>) {
                return arg;
            } else if constexpr (std::is_constructible_v<std::string, decltype(arg)>) {
                return std::string(arg);
            } else {
                return std::to_string(arg);
            }
        };
        (rows_.back().push_back(toString(std::forward<Args>(args))), ...);
    }

    void AddRowSeparator(size_t fromColumn = 0, char c = '-')
    {
        DEV_ASSERT_MSG(DevDataErr::SHEET_COLUMN_INDEX_OUT_OF_RANGE, fromColumn < columnTitles_.size(),
                       "fromColumn=%zu >= columnTitles_.size()=%zu", fromColumn, columnTitles_.size());
        if (!rows_.empty()) {
            rowSeparators_.push_back(RowSeparator{rows_.size(), fromColumn, c});
        }
    }

    std::vector<std::string> DumpLines() const
    {
        if (columnTitles_.empty()) {
            return {};
        }

        size_t sheetWidth = 0;
        std::vector<size_t> columnWidths(columnTitles_.size());
        for (size_t col = 0; col < columnTitles_.size(); col++) {
            columnWidths[col] = columnTitles_[col].length();
            for (auto&& row : rows_) {
                columnWidths[col] = std::max(columnWidths[col], row[col].length());
            }

            if (col != 0) {
                sheetWidth += 3; // length of " | " equals to 3
            } else {
                // Leave an extra space before first column to make it look better
                columnWidths[col]++;
            }
            sheetWidth += columnWidths[col];
        }
        // Leave an extra space after last column to make it look better
        sheetWidth++;

        std::vector<std::string> dumps;
        dumps.emplace_back(sheetWidth, boundary_);
        {
            std::ostringstream oss;
            for (size_t col = 0; col < columnTitles_.size(); col++) {
                if (col != 0) {
                    oss << " | ";
                }
                oss << std::setw(columnWidths[col]) << columnTitles_[col];
            }
            dumps.push_back(std::move(oss).str());
        }
        dumps.emplace_back(sheetWidth, separator_);
        auto rowSeparatorIter = rowSeparators_.begin();
        for (size_t row = 0; row < rows_.size(); row++) {
            while (rowSeparatorIter != rowSeparators_.end() && row == rowSeparatorIter->rowIdx) {
                std::ostringstream oss;
                size_t width = sheetWidth;
                for (size_t col = 0; col < columnTitles_.size(); col++) {
                    if (col == rowSeparatorIter->fromColumn) {
                        oss << std::string(width, rowSeparatorIter->c);
                        break;
                    }
                    oss << std::string(columnWidths[col], ' ') << " | ";
                    width -= columnWidths[col] + 3; // length of " | " equals to 3
                }
                dumps.push_back(std::move(oss).str());
                rowSeparatorIter++;
            }

            std::ostringstream oss;
            for (size_t col = 0; col < columnTitles_.size(); col++) {
                if (col != 0) {
                    oss << " | ";
                }
                oss << std::setw(columnWidths[col]) << rows_[row][col];
            }
            dumps.push_back(std::move(oss).str());
        }
        dumps.emplace_back(sheetWidth, boundary_);
        return dumps;
    }

private:
    struct RowSeparator {
        size_t rowIdx;
        size_t fromColumn;
        char c;
    };

    std::vector<std::string> columnTitles_;
    std::vector<std::vector<std::string>> rows_;
    std::vector<RowSeparator> rowSeparators_;

    char boundary_{'='};
    char separator_{'-'};
};

} // namespace npu::tile_fwk
