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
 * \file raw_tensor_data.cpp
 * \brief
 */

#include "raw_tensor_data.h"

namespace npu::tile_fwk {

constexpr size_t LOGICAL_TENSOR_DATA_HEAD_MAX_DIM = 5;
constexpr size_t LOGICAL_TENSOR_DATA_HEAD_PADDING_SIZE = 3;

struct LogicalTensorDataHead {
    uint32_t version{0};
    uint32_t dataType{0};
    uint32_t dimension{0};
    int64_t shape[LOGICAL_TENSOR_DATA_HEAD_MAX_DIM] = {0};
    uint32_t padding[LOGICAL_TENSOR_DATA_HEAD_PADDING_SIZE] = {0};
};

std::string LogicalTensorData::DumpRange(int idxBegin, int idxEnd,
                                         const std::vector<ElementDump>* elementDumpList) const
{
    std::vector<bool> elementDiffPrevList(idxEnd - idxBegin, false);

    std::vector<std::string> elementList;
    if (elementDumpList == nullptr) {
        for (int idx = 0; idx < idxEnd - idxBegin; idx++) {
            elementList.push_back(DumpElement(idxBegin + idx));
        }
        for (int idx = 1; idx < idxEnd - idxBegin; idx++) {
            elementDiffPrevList[idx] = elementList[idx] != elementList[idx - 1];
        }
    } else {
        for (int idx = 1; idx < idxEnd - idxBegin; idx++) {
            elementDiffPrevList[idx] = elementDumpList->at(idxBegin + idx) != elementDumpList->at(idxBegin + idx - 1);
        }
    }

    std::ostringstream oss;
    int base = 0;
    for (int idx = 0; idx <= idxEnd - idxBegin; idx++) {
        if (idx == 0 || idx == idxEnd - idxBegin || elementDiffPrevList[idx]) {
            if (idx != 0) {
                if (base != 0) {
                    oss << ",";
                }
                oss << "[";
                if (idx == base + 1) {
                    oss << base;
                } else {
                    oss << base << "..." << (idx - 1);
                }
                oss << "]=";
                if (elementDumpList == nullptr) {
                    oss << elementList[base];
                } else {
                    oss << elementDumpList->at(idxBegin + base).c_str();
                }
            }
            base = idx;
        }
    }
    return oss.str();
}

std::string LogicalTensorData::DumpCoord(int row) const
{
    std::ostringstream oss;

    std::vector<int> coord(GetShape().size() - 1);
    int coordDim = row;
    for (size_t k = 0; k < GetShape().size() - 1; k++) {
        int dim = GetShape().size() - 2 - k;
        int dimLength = GetShape()[dim];
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_SHAPE, dimLength != 0)
            << "Shape dimension " << dim << " must not be zero";
        coord[dim] = coordDim % dimLength;
        coordDim /= dimLength;
    }
    for (size_t k = 0; k < coord.size(); k++) {
        if (k != 0) {
            oss << ",";
        }
        oss << coord[k];
    }
    return oss.str();
}

std::string LogicalTensorData::DumpData(int indent, const std::vector<ElementDump>* elementDumpList) const
{
    std::string space(indent, ' ');

    std::ostringstream oss;
    oss << DumpType() << " {\n";
    int rowSize = 1;
    for (size_t k = 0; k < GetShape().size() - 1; k++) {
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_SHAPE, GetShape()[k] > 0 && rowSize <= INT_MAX / GetShape()[k])
            << "Tensor shape multiplication overflow at dimension " << k;
        rowSize *= GetShape()[k];
    }
    int colSize = GetShape().back();

    std::vector<std::string> rowList;
    for (int row = 0; row < rowSize; row++) {
        rowList.emplace_back(DumpRange(row * colSize, row * colSize + colSize, elementDumpList));
    }
    int base = 0;
    for (int row = 0; row <= rowSize; row++) {
        if (row == 0 || row == rowSize || rowList[row] != rowList[row - 1]) {
            if (row != 0) {
                oss << space << "[";
                if (row == base + 1) {
                    oss << DumpCoord(base);
                } else {
                    oss << DumpCoord(base) << "..." << DumpCoord(row - 1);
                }
                oss << "]=";
                oss << "{";
                oss << rowList[base];
                oss << "}\n";
            }
            base = row;
        }
    }
    oss << "}";
    return oss.str();
}

void LogicalTensorData::Save(const std::string& filepath) const
{
    FILE* fdata = fopen(filepath.c_str(), "wb");
    if (fdata != nullptr) {
        LogicalTensorDataHead head;
        head.dataType = GetDataType();
        head.dimension = GetShape().size();
        for (size_t k = 0; k < GetShape().size(); k++) {
            head.shape[k] = GetShape()[k];
        }
        fwrite(&head, sizeof(head), 1, fdata);

        int rowSize = GetShape().back();
        int totalSize = GetSize();

        switch (GetDataType()) {
            case DT_INT8:
                HandleSave<int8_t>(fdata, totalSize, rowSize);
                break;
            case DT_INT16:
                HandleSave<int16_t>(fdata, totalSize, rowSize);
                break;
            case DT_INT32:
                HandleSave<int32_t>(fdata, totalSize, rowSize);
                break;
            case DT_INT64:
                HandleSave<int64_t>(fdata, totalSize, rowSize);
                break;
            case DT_FP16:
                HandleSave<npu::tile_fwk::float16>(fdata, totalSize, rowSize);
                break;
            case DT_FP32:
                HandleSave<float>(fdata, totalSize, rowSize);
                break;
            case DT_BF16:
                HandleSave<npu::tile_fwk::bfloat16>(fdata, totalSize, rowSize);
                break;
            case DT_UINT8:
                HandleSave<uint8_t>(fdata, totalSize, rowSize);
                break;
            case DT_UINT16:
                HandleSave<uint16_t>(fdata, totalSize, rowSize);
                break;
            case DT_UINT32:
                HandleSave<uint32_t>(fdata, totalSize, rowSize);
                break;
            case DT_UINT64:
                HandleSave<uint64_t>(fdata, totalSize, rowSize);
                break;
            case DT_DOUBLE:
                HandleSave<double>(fdata, totalSize, rowSize);
                break;
            case DT_BOOL:
                HandleSave<bool>(fdata, totalSize, rowSize);
                break;
            default:
                ASSERT(false);
                break;
        }
        fclose(fdata);
    }
}

void LogicalTensorData::SaveFile(const char* filepath) const { return Save(filepath); }

std::string LogicalTensorData::ToString(const PrintOptions* options) const
{
    constexpr size_t MAX_AXES_DIM = 8;
    constexpr int EDGE_ITEMS_MULTIPLIER = 2;
    std::stringstream os;
    int64_t axes[MAX_AXES_DIM] = {0};

    if (options == nullptr) {
        options = &config::GetPrintOptions();
    }

    int edgeItems = options->edgeItems;
    int64_t totalItems = std::accumulate(shape_.begin(), shape_.end(), (int64_t)1, std::multiplies<int64_t>());
    if (options->threshold > totalItems) {
        edgeItems = options->threshold;
    }

    std::function<void(int dim)> printImpl;
    int ndim = shape_.size();
    auto& shape = validShape_.empty() ? shape_ : validShape_;

    auto repeat = [&](char c, int n) {
        for (int i = 0; i < n; i++) {
            os << c;
        }
    };

    auto print1d = [&](int dim, int s, int e) {
        int pos = os.tellp();
        for (int i = s; i < e; i++) {
            if (i != 0)
                os << " ";
            axes[ndim - 1] = i;
            auto elem = GetData()->GetElement(axes, ndim);
            if (elem.IsSigned())
                os << elem.GetSignedData();
            else if (elem.IsUnsigned())
                os << elem.GetUnsignedData();
            else
                os << std::setprecision(options->precision) << elem.GetFloatData();
            if (os.tellp() >= pos + options->linewidth) {
                os << "\n";
                repeat(' ', dim);
                pos = os.tellp() + 1L;
            }
        }
    };

    auto printnd = [&](int dim, int s, int e) {
        for (int i = s; i < e; i++) {
            if (i > 0) {
                repeat('\n', ndim - dim - 1);
                repeat(' ', dim + 1);
            }
            axes[dim] = i;
            printImpl(dim + 1);
        }
    };

    printImpl = [&](int dim) {
        os << "[";
        if (dim == ndim - 1) {
            if (shape[dim] > EDGE_ITEMS_MULTIPLIER * edgeItems) {
                print1d(dim, 0, edgeItems);
                os << " ...";
                print1d(dim, shape[dim] - edgeItems, shape[dim]);
            } else {
                print1d(dim, 0, shape[dim]);
            }
        } else {
            if (shape[dim] > EDGE_ITEMS_MULTIPLIER * edgeItems) {
                printnd(dim, 0, edgeItems);
                os << "\n";
                repeat(' ', dim + 1);
                os << "...";
                printnd(dim, shape[dim] - edgeItems, shape[dim]);
            } else {
                printnd(dim, 0, shape[dim]);
            }
        }
        os << "]";
    };
    os << DumpType() << '\n';
    printImpl(0);
    return os.str();
}

std::shared_ptr<LogicalTensorData> LogicalTensorData::Load(const std::string& filepath)
{
    std::shared_ptr<LogicalTensorData> dataView = nullptr;
    FILE* fdata = fopen(filepath.c_str(), "rb");
    if (fdata != nullptr) {
        LogicalTensorDataHead head;
        if (fread(&head, sizeof(head), 1, fdata) == 1) {
            std::vector<int64_t> shape(head.dimension, 0);
            for (int i = 0; i < static_cast<int>(head.dimension); i++) {
                shape[i] = head.shape[i];
            }
            auto data = std::make_shared<RawTensorData>(static_cast<DataType>(head.dataType), shape);
            if (fread(data->data(), 1, data->size(), fdata) == data->size()) {
                dataView = std::make_shared<LogicalTensorData>(data, shape, shape,
                                                               std::vector<int64_t>(shape.size(), 0));
            }
        }
        fclose(fdata);
    }
    return dataView;
}

ProgramData& ProgramData::GetInstance()
{
    static ProgramData data;
    return data;
}

} // namespace npu::tile_fwk
