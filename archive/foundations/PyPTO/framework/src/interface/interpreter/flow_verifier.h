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
 * \file flow_verifier.h
 * \brief
 */

#pragma once

#include "interface/interpreter/interpreter_log.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/operation/attribute.h"
#include "interface/function/function.h"
#include "interface/interpreter/function.h"
#include <float.h>
#include <cmath>
#include <utility>

namespace npu::tile_fwk {

constexpr int THREAD_THOUSAND = 1000;

class FlowVerifier {
public:
    static FlowVerifier& GetInstance();

    void VerifyTensorGraph(Function* entry, const std::vector<std::shared_ptr<LogicalTensorData>>& inputDataViewList,
                           const std::vector<std::shared_ptr<LogicalTensorData>>& outputDataViewList,
                           const std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList,
                           const std::shared_ptr<TensorSlotManager>& slotManager);
    void VerifyPass(Function* func, int passIndex, const std::string& passIdentifier);

    struct CompareResultDetail {
        size_t totalCnt;
        size_t zeroCnt;
        size_t toleranceCnt;
        size_t warnNum;
        size_t failNum;
        double mre;
        double mreTop8;
        double mreTop1Permil;
        double mae;
        double maeTop8;
        double maeTop1Permil;
        double aMax = FLT_MIN;
        double aMin = FLT_MAX;
        double aAvg;
        double aAavg;
        size_t aZero = 0;
        size_t aInfnan = 0;
        double bMax = FLT_MIN;
        double bMin = FLT_MAX;
        double bAvg;
        double bAavg;
        size_t bZero = 0;
        size_t bInfnan = 0;
        size_t infnanCnt = 0;
    };

    struct CompareElement {
        bool isError;
        size_t index;
        double goldenValue;
        double outputValue;
        double absDiff;
        double relDiff;
        double tolerance;

        CompareElement() = default;
        CompareElement(const CompareElement&) = default;
        CompareElement& operator=(const CompareElement&) = default;
        CompareElement(bool isError_, size_t index_, double goldenValue_, double outputValue_, double absDiff_,
                       double relDiff_, double tolerance_)
            : isError(isError_),
              index(index_),
              goldenValue(goldenValue_),
              outputValue(outputValue_),
              absDiff(absDiff_),
              relDiff(relDiff_),
              tolerance(tolerance_)
        {}

        std::string Dump() const
        {
            std::ostringstream oss;
            oss << "index:" << index << " golden:" << goldenValue << " output:" << outputValue << " absDiff:" << absDiff
                << " relDiff:" << relDiff;
            return oss.str();
        }
    };
    struct CompareResult : std::vector<CompareElement> {
        CompareResult() {}
        CompareResult(int size, float rtol, float atol, size_t errorCountThreshold = 0, size_t failNum = 0,
                      Shape shape = {})
            : size_(size),
              rtol_(rtol),
              atol_(atol),
              errorCountThreshold_(errorCountThreshold),
              failNum_(failNum),
              shape_(shape)
        {}

        template <typename... TyArgs>
        void AppendError(TyArgs&&... args)
        {
            errorCount_++;
            this->emplace_back(args...);
        }

        void AppendZero() { zeroCount_++; }

        void AppendFail() { failNum_++; }

        void UpdateErrorCountThreshold()
        {
            errorCountThreshold_ = static_cast<int>((size_ - zeroCount_) * std::min(rtol_, atol_));
            size_t cnt_adj = static_cast<int>(std::pow((size_ - zeroCount_), 0.5)) / 2;
            if (errorCountThreshold_ == 0) {
                size_t cnt_normal = 16;
                errorCountThreshold_ = std::min(cnt_normal, cnt_adj);
            }
        }

        bool Check() const { return errorCount_ <= errorCountThreshold_ && failNum_ == 0; }

        void sortByAbsAdesc()
        {
            std::sort(this->begin(), this->end(),
                      [](const CompareElement& lhs, const CompareElement& rhs) { return lhs.absDiff > rhs.absDiff; });
        }

        void sortByRelAdesc()
        {
            std::sort(this->begin(), this->end(),
                      [](const CompareElement& lhs, const CompareElement& rhs) { return lhs.relDiff > rhs.relDiff; });
        }

        double GetMeanTopN(size_t num, bool is_abs) const
        {
            double sum = 0;
            size_t count = std::min(this->size(), num);
            if (count == 0)
                return 0;
            if (is_abs) {
                sum = std::accumulate(this->begin(), this->begin() + count, 0.0,
                                      [](double acc, const CompareElement& item) { return acc + item.absDiff; });
            } else {
                sum = std::accumulate(this->begin(), this->begin() + count, 0.0,
                                      [](double acc, const CompareElement& item) { return acc + item.relDiff; });
            }
            return sum / count;
        }

        Shape GetOffsetRaw(int64_t offset) const
        {
            if (shape_.empty()) {
                return {};
            }
            int64_t total = 1;
            for (int64_t dim : shape_) {
                total *= dim;
            }
            ASSERT(VerifyResultScene::VERIFY_RESULT_INDEX_OUTOFBOUNDS, offset < total) << "Offset Out Of Bounds";

            std::vector<int64_t> indices(shape_.size());
            int64_t remaining = offset;

            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                indices[i] = remaining % shape_[i];
                remaining /= shape_[i];
            }
            return indices;
        }

        void DumpDataDetail(std::ostringstream& oss, size_t topk = 64)
        {
            constexpr auto max_precision{std::numeric_limits<float>::digits10 + 1};
            oss << std::setprecision(max_precision);
            oss << "GROUP,INDEX,OFFSET,OFFSET_RAW,A>data,B>data,AB>ae,AB>re,AB>tol\n";
            size_t count = std::min(topk, this->size());
            for (size_t k = 0; k < count; k++) {
                auto [isError, index, goldenValue, outputValue, absDiff, relDiff, tolerance] = (*this)[k];
                (void)isError;
                oss << "firstk," << k << "," << index << "," << FunctionInterpreter::ShapeToString(GetOffsetRaw(index))
                    << "," << goldenValue << "," << outputValue << "," << absDiff << "," << relDiff << "," << tolerance
                    << "\n";
            }
            sortByRelAdesc();
            for (size_t k = 0; k < count; k++) {
                auto [isError, index, goldenValue, outputValue, absDiff, relDiff, tolerance] = (*this)[k];
                (void)isError;
                oss << "topk_re," << k << "," << index << "," << FunctionInterpreter::ShapeToString(GetOffsetRaw(index))
                    << "," << goldenValue << "," << outputValue << "," << absDiff << "," << relDiff << "," << tolerance
                    << "\n";
            }
        }

        CompareResultDetail Dump(int indent = 2, size_t maxPrint = 5)
        {
            double maxAbsDiff = 0;
            double maxRelDiff = 0;
            double totalAbsDiff = 0;
            double totalRelDiff = 0;
            CompareResultDetail compareResultDetail;
            CompareElement maxAbsElement;
            CompareElement maxRelElement;
            std::ostringstream oss;
            std::string space(indent, ' ');
            std::string infoError = "\n  " + space + "Error rtol=" + std::to_string(rtol_) +
                                    " atol=" + std::to_string(atol_);
            std::string infoZero = "\n  " + space + "Zero";
            size_t count_ = 0;
            for (auto& element : *this) {
                auto [isError, index, goldenValue, outputValue, absDiff, relDiff, tolerance] = element;
                (void)index;
                (void)tolerance;
                if (absDiff > maxAbsDiff) {
                    maxAbsDiff = absDiff;
                    maxAbsElement = element;
                }
                if (relDiff > maxRelDiff) {
                    maxRelDiff = relDiff;
                    maxRelElement = element;
                }
                maxAbsDiff = std::max(maxAbsDiff, absDiff);
                maxRelDiff = std::max(maxRelDiff, relDiff);
                compareResultDetail.aMax = std::max(compareResultDetail.aMax, goldenValue);
                compareResultDetail.bMax = std::max(compareResultDetail.bMax, outputValue);
                compareResultDetail.aMin = std::min(compareResultDetail.aMin, goldenValue);
                compareResultDetail.bMin = std::min(compareResultDetail.bMin, outputValue);
                totalAbsDiff += absDiff;
                totalRelDiff += relDiff;

                std::string info = "";
                if (isError) {
                    info = infoError.c_str();
                } else {
                    continue;
                }
                if (count_ <= maxPrint) {
                    oss << space << info << " " << element.Dump() << "";
                }
                count_++;
            }
            oss << "\n"
                << space << "All size:" << size_ << " failNum:" << failNum_ << " maxAbsDiff:" << maxAbsDiff
                << " maxRelDiff:" << maxRelDiff << " averageAbsDiff:" << totalAbsDiff / size_
                << " averageRelDiff:" << totalRelDiff / size_ << " errorCount:" << errorCount_
                << " errorRatio:" << errorCount_ * 1.0 / size_ << " zeroCount:" << zeroCount_
                << " zeroRatio:" << zeroCount_ * 1.0 / size_ << "\n";
            if (errorCount_ + zeroCount_ > 0) {
                oss << space << "maxAbs-> " << maxAbsElement.Dump() << "\n"
                    << space << "maxRel-> " << maxRelElement.Dump() << "\n";
            }
            if (!Check()) {
                INTERPRETER_EVENT("%s", oss.str().c_str());
            }
            compareResultDetail.totalCnt = size_;
            compareResultDetail.zeroCnt = zeroCount_;
            compareResultDetail.toleranceCnt = errorCountThreshold_;
            compareResultDetail.warnNum = errorCount_;
            compareResultDetail.failNum = failNum_;
            compareResultDetail.mre = totalAbsDiff / size_;
            compareResultDetail.mae = totalRelDiff / size_;
            constexpr size_t topNCount = 8;
            constexpr size_t top1PermilCount = 1000;
            compareResultDetail.mreTop8 = GetMeanTopN(topNCount, false);
            compareResultDetail.mreTop1Permil = GetMeanTopN(top1PermilCount, false);
            sortByAbsAdesc();
            compareResultDetail.maeTop1Permil = GetMeanTopN(top1PermilCount, true);
            compareResultDetail.maeTop8 = GetMeanTopN(topNCount, true);
            compareResultDetail.aMax = goldenMax_;
            compareResultDetail.aMin = goldenMin_;
            compareResultDetail.aAvg = goldenSum_ / size_;
            compareResultDetail.aAavg = goldenAbsSum_ / size_;
            compareResultDetail.aZero = goldenZero_;
            compareResultDetail.bMax = outputMax_;
            compareResultDetail.bMin = outputMin_;
            compareResultDetail.bAvg = outputSum_ / size_;
            compareResultDetail.bAavg = outputAbsSum_ / size_;
            compareResultDetail.bZero = outputZero_;
            compareResultDetail.bInfnan = outputInfnan_;
            compareResultDetail.aInfnan = goldenInfnan_;
            compareResultDetail.infnanCnt = infnanCnt_;
            return compareResultDetail;
        }

        float GetRtol() const { return rtol_; }
        float GetAtol() const { return atol_; }
        double goldenMax_ = -DBL_MAX;
        double outputMax_ = -DBL_MAX;
        double goldenMin_ = DBL_MAX;
        double outputMin_ = DBL_MAX;
        double goldenSum_ = 0;
        double outputSum_ = 0;
        double goldenAbsSum_ = 0;
        double outputAbsSum_ = 0;
        size_t goldenZero_ = 0;
        size_t outputZero_ = 0;
        size_t goldenInfnan_ = 0;
        size_t outputInfnan_ = 0;
        size_t infnanCnt_ = 0;

    private:
        size_t size_{0};
        float rtol_{0};
        float atol_{0};
        size_t errorCountThreshold_{0};
        size_t failNum_{0};
        Shape shape_;
        size_t errorCount_ = 0;
        size_t zeroCount_ = 0;
    };

private:
    static void CompareScalarPair(CompareResult& compareResult, int64_t linearIndex, double goldenValue,
                                  double outputValue)
    {
        compareResult.goldenMax_ = std::max(compareResult.goldenMax_, goldenValue);
        compareResult.outputMax_ = std::max(compareResult.outputMax_, outputValue);
        compareResult.goldenMin_ = std::min(compareResult.goldenMin_, goldenValue);
        compareResult.outputMin_ = std::min(compareResult.outputMin_, outputValue);
        compareResult.goldenSum_ += goldenValue;
        compareResult.outputSum_ += outputValue;
        const double output_abs = std::abs(outputValue);
        const double golden_abs = std::abs(goldenValue);
        const double output_golden_sub_abs = std::abs(outputValue - goldenValue);
        compareResult.goldenAbsSum_ += golden_abs;
        compareResult.outputAbsSum_ += output_abs;
        if (output_abs <= 0) {
            compareResult.outputZero_++;
        }
        if (golden_abs <= 0) {
            compareResult.goldenZero_++;
        }
        if (!std::isfinite(outputValue)) {
            compareResult.outputInfnan_++;
        }
        if (!std::isfinite(goldenValue)) {
            compareResult.goldenInfnan_++;
        }
        if (!std::isfinite(output_golden_sub_abs)) {
            compareResult.infnanCnt_++;
        }
        const double output_golden_abs_add = output_abs + golden_abs;
        if (output_golden_abs_add <= 0) {
            compareResult.AppendZero();
            return;
        }

        const double relDiff = output_golden_sub_abs * 2 / output_golden_abs_add;
        const double tol_attn = output_golden_abs_add * compareResult.GetRtol() / 2 + compareResult.GetAtol();
        const double tol_fail = tol_attn * 128;
        if (output_golden_sub_abs > tol_attn) {
            compareResult.AppendError(true, static_cast<size_t>(linearIndex), goldenValue, outputValue,
                                      output_golden_sub_abs, relDiff, tol_attn);
        }
        if (output_golden_sub_abs > tol_fail) {
            compareResult.AppendFail();
        }
    }

    static int64_t GetValidStride(Shape validShape, int64_t axis)
    {
        int64_t stride = 1;
        int64_t dims = validShape.size();
        for (int64_t i = axis + 1; i < dims; i++) {
            stride = stride * validShape[i];
        }
        return stride;
    }

    template <typename Leaf>
    static void CompareDataRecursiveWithLeaf(CompareResult& compareResult, size_t axis, int64_t goldenOffset,
                                             int64_t outputOffset,
                                             const std::shared_ptr<LogicalTensorData>& goldenDataView,
                                             const std::shared_ptr<LogicalTensorData>& outputDataView, int64_t index,
                                             Leaf&& leaf)
    {
        auto& validShape = goldenDataView->GetValidShape();
        if (axis == validShape.size() - 1) {
            leaf(compareResult, validShape[axis], outputOffset, goldenOffset, index, goldenDataView, outputDataView);
        } else {
            for (int i = 0; i < validShape[axis]; i++) {
                int nGoldenOffset = goldenOffset + goldenDataView->GetData()->GetStride()[axis] * i;
                int nOutputOffset = outputOffset + outputDataView->GetData()->GetStride()[axis] * i;
                int64_t nindex = index + GetValidStride(validShape, axis) * i;
                CompareDataRecursiveWithLeaf(compareResult, axis + 1, nGoldenOffset, nOutputOffset, goldenDataView,
                                             outputDataView, nindex, std::forward<Leaf>(leaf));
            }
        }
    }

public:
    template <typename DataType, typename T>
    static void CompareData(CompareResult& compareResult, size_t count, int64_t offset, const DataType* goldenValueList,
                            const DataType* outputValueList)
    {
        for (size_t index = 0; index < count; index++) {
            const double goldenValue = static_cast<double>(static_cast<T>(goldenValueList[index]));
            const double outputValue = static_cast<double>(static_cast<T>(outputValueList[index]));
            CompareScalarPair(compareResult, offset + static_cast<int64_t>(index), goldenValue, outputValue);
        }
    }

    template <typename DataType, typename T>
    static void CompareDataRecursive(CompareResult& compareResult, size_t axis, int64_t goldenOffset,
                                     int64_t outputOffset, const std::shared_ptr<LogicalTensorData>& goldenDataView,
                                     const std::shared_ptr<LogicalTensorData>& outputDataView, int64_t index)
    {
        CompareDataRecursiveWithLeaf(
            compareResult, axis, goldenOffset, outputOffset, goldenDataView, outputDataView, index,
            [](CompareResult& cr, size_t lastAxisLen, int64_t outOff, int64_t gOff, int64_t idx,
               const std::shared_ptr<LogicalTensorData>& gv, const std::shared_ptr<LogicalTensorData>& ov) {
                CompareData<DataType, T>(cr, lastAxisLen, idx, &gv->Get<DataType>(gOff), &ov->Get<DataType>(outOff));
            });
    }

    template <typename DataType, typename T>
    static CompareResult CompareData(const std::shared_ptr<LogicalTensorData>& goldenDataView,
                                     const std::shared_ptr<LogicalTensorData>& outputDataView, float rtol, float atol,
                                     int errorCountThreshold = 0, int failNum = 0)
    {
        auto& validShape = goldenDataView->GetValidShape();
        auto size = std::accumulate(validShape.begin(), validShape.end(), 1, std::multiplies<>());
        CompareResult compareResult(size, rtol, atol, errorCountThreshold, failNum, validShape);
        CompareDataRecursive<DataType, T>(compareResult, 0, goldenDataView->GetStorageOffset(),
                                          outputDataView->GetStorageOffset(), goldenDataView, outputDataView, 0);
        compareResult.UpdateErrorCountThreshold();
        return compareResult;
    }

    static CompareResult VerifyResult(const std::shared_ptr<LogicalTensorData>& goldenDataView,
                                      const std::shared_ptr<LogicalTensorData>& outputDataView, float rtol, float atol);
    bool VerifyResult(const std::vector<std::shared_ptr<LogicalTensor>>& tensorDatalist,
                      const std::vector<std::shared_ptr<LogicalTensor>>& goldenDatalist, const std::string& key,
                      const std::string tensorNameList,
                      const std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList,
                      const std::vector<std::shared_ptr<LogicalTensorData>>& tensorDataViewList, float rtol,
                      float atol);

    std::string ParseErrorMsg(std::string errorMsg);

    void WriteUserGolden(const std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList);

    void WriteException();

private:
    static CompareResult CompareFp8TensorData(const std::shared_ptr<LogicalTensorData>& goldenDataView,
                                              const std::shared_ptr<LogicalTensorData>& outputDataView,
                                              DataType fp8Format, float rtol, float atol, int errorCountThreshold = 0,
                                              int failNum = 0);

    void UpdateInterpreterCache();
    void Initialize(Function* entry, const std::vector<std::shared_ptr<LogicalTensorData>>& inputDataViewList,
                    const std::vector<std::shared_ptr<LogicalTensorData>>& outputDataViewList,
                    const std::vector<std::shared_ptr<LogicalTensorData>>& goldenDataViewList,
                    const std::shared_ptr<TensorSlotManager>& slotManager);

private:
    Function* entry_;
    bool checkResult{true};
    std::vector<std::shared_ptr<LogicalTensorData>> inputDataViewList_;
    std::vector<std::shared_ptr<LogicalTensorData>> outputDataViewList_;
    std::vector<std::shared_ptr<LogicalTensorData>> goldenDataViewList_;
    std::shared_ptr<FunctionInterpreter> functionInterpreter_;

    std::shared_ptr<FunctionControlFlowExecution> controlFlowExecution_;
    std::unordered_map<Function*, std::vector<std::shared_ptr<FunctionCaptureExecution>>> lastCaptureExecution_;
};

} // namespace npu::tile_fwk
