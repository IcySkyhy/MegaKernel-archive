/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tilefwk_op.h
 * \brief
 */

#pragma once

#include <array>
#include <sstream>
#include <tuple>

#include "tilefwk/tensor.h"
#include "tilefwk/element.h"

namespace npu::tile_fwk {
class Function;
class SymbolicScalar;
class Element;
constexpr const int TILE_VEC_DIMS = 2;
constexpr const int TILE_CUBE_DIMS = 6;

enum class OpType {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
};
enum class OutType {
    BOOL,
    BIT,
};
enum class TopKAlgo {
    MERGE_SORT,
    RADIX_SELECT,
};

enum class AtomicRMWMode { ADD, MAX, MIN };

enum class DequantScaleRoundingMode : int64_t {
    ROUND_UP = 0,
    ROUND_DOWN = 1,
};

enum class SaturationMode : uint8_t {
    ON = 0,
    OFF = 1,
};

// 统一的精度类型枚举
enum class PrecisionType : uint8_t {
    INTRINSIC = 0,     // 指令模式，直接使用芯片指令
    HIGH_PRECISION = 1 // 高精度模式
};

namespace experimental {
struct PrintHelper {
    SymbolicScalar cond;
    std::vector<Tensor> tensors;
    std::vector<SymbolicScalar> scalars;
    std::stringstream ss;

    template <typename T>
    void Append(T t)
    {
        if constexpr (std::is_same_v<T, Tensor>) {
            tensors.push_back(t);
            ss << "{T}";
        } else if constexpr (std::is_same_v<T, SymbolicScalar>) {
            scalars.push_back(t);
            ss << "{S}";
        } else {
            ss << t;
        }
    }
};

void Print(SymbolicScalar cond, const std::string& format, const std::vector<Tensor>& tensors,
           const std::vector<SymbolicScalar>& scalars);

template <bool isB, bool isTrans>
Tensor GatherInL1(const Tensor& src, const Tensor& offsets, const Tensor& blockTable, int blockSize, int size);
Tensor GatherInUB(const Tensor& params, const Tensor& indices, const Tensor& blockTable, int blockSize, int axis);
} // namespace experimental

template <typename... Args>
void Print(Args... args)
{
    experimental::PrintHelper helper;
    (helper.Append(args), ...);
    experimental::Print(1, helper.ss.str(), helper.tensors, helper.scalars);
}

template <typename... Args>
void PrintIf(SymbolicScalar cond, Args... args)
{
    experimental::PrintHelper helper;
    (helper.Append(args), ...);
    experimental::Print(cond, helper.ss.str(), helper.tensors, helper.scalars);
}

/**
 * \brief Dump a tensor to file
 *
 * \param cond Dump the tensor only `cond` evaluate result is none zero
 * \param operand tensor to dump
 * \param fname filename, {S} can be used as scalar placeholder
 * \param scalars scalars to dump
 */
void ToFile(const Tensor& operand, const std::string& fname, const std::vector<SymbolicScalar>& scalars = {},
            SymbolicScalar cond = 1);

Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes, const std::vector<int64_t>& offsets);
Tensor View(const Tensor& operand, const DataType dstDataType);
Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes, const std::vector<SymbolicScalar>& newOffsets);
Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes,
            const std::initializer_list<SymbolicScalar>& newOffsets);
Tensor View(const Tensor& operand, const std::vector<int64_t>& shapes,
            const std::vector<SymbolicScalar>& newValidShapes, const std::vector<SymbolicScalar>& newOffsets);

Tensor Assemble(const std::vector<std::pair<Tensor, std::vector<int64_t>>>& tensors);
void Assemble(const Tensor& tensor, const std::vector<SymbolicScalar>& dynOffset, Tensor& dest, bool parallel = false);

struct AssembleItem {
    Tensor tensor;
    std::vector<SymbolicScalar> offsets;
};

void Assemble(const std::vector<AssembleItem>& items, Tensor& src, bool parallelInAssemble = false);

void AtomicRMW(const Tensor& tensor, const std::vector<SymbolicScalar>& dynOffset, Tensor& dest, AtomicRMWMode mode);

Tensor Reshape(const Tensor& operand, const std::vector<int64_t>& dstshape,
               const std::vector<SymbolicScalar>& validShape = {}, const bool inplace = false);
Tensor Reshape(const Tensor& operand, const std::initializer_list<int64_t>& dstshape,
               const std::initializer_list<SymbolicScalar>& validShape = {}, const bool inplace = false);
Tensor Reshape(const Tensor& operand, const std::vector<SymbolicScalar>& dstShape, const bool inplace);

void Reshape(const Tensor& operand, Tensor& dst);

Tensor Full(const Element& src, DataType dtype, const std::vector<int64_t>& dstShape,
            std::vector<SymbolicScalar> validShape = {});
Tensor Full(const SymbolicScalar& src, DataType dtype, const std::vector<int64_t>& dstShape,
            std::vector<SymbolicScalar> validShape = {});
Tensor Transpose(const Tensor& self, std::vector<int> perm);
Tensor TransData(const Tensor& self, TileOpFormat transDataType, const std::vector<int64_t>& outputShape,
                 const std::vector<SymbolicScalar>& validShape, int group = 1);
std::shared_ptr<LogicalTensor> TransData(Function& function, const std::shared_ptr<LogicalTensor>& self,
                                         const std::shared_ptr<LogicalTensor>& fakeDstTensor,
                                         TileOpFormat transDataType, int group = 1);
Tensor Cast(const Tensor& self, DataType dstDataType, CastMode mode = CAST_NONE,
            SaturationMode satmode = SaturationMode::OFF);
Tensor Permute(const Tensor& self, std::vector<int> perm);
Tensor Permute(Function& function, const Tensor& self, std::vector<int> perm);

Tensor Exp(const Tensor& self, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Exp2(const Tensor& self);
Tensor Expm1(const Tensor& self);
Tensor Erfc(const Tensor& self);
Tensor Atan(const Tensor& self);
Tensor Sin(const Tensor& self);
Tensor Cos(const Tensor& self);
Tensor Erf(const Tensor& self);
Tensor Neg(const Tensor& self);
Tensor Round(const Tensor& self, const int& decimals = 0);
Tensor Rsqrt(const Tensor& self, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Relu(const Tensor& self);
Tensor Pad(const Tensor& self, const std::vector<int64_t>& padding, std::string mode = "constant",
           const Element& value = Element(DT_FP32, 0.0));
Tensor FillPad(const Tensor& self, std::string mode = "constant", const Element& value = Element(DT_FP32, 0.0));
Tensor BitwiseNot(const Tensor& self);
Tensor Sqrt(const Tensor& self, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Ceil(const Tensor& self);
Tensor CeilDiv(const Tensor& self, const Tensor& other);
Tensor CeilDiv(const Tensor& self, const Element& other);
Tensor Floor(const Tensor& self);
Tensor FloorDiv(const Tensor& self, const Tensor& other);
Tensor FloorDiv(const Tensor& self, const Element& other);
Tensor Trunc(const Tensor& self);
Tensor Reciprocal(const Tensor& operand, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Abs(const Tensor& self);
Tensor Ln(const Tensor& operand, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Hub(const Tensor& operand);
Tensor Sign(const Tensor& operand);
Tensor Tan(const Tensor& operand);
Tensor Signbit(const Tensor& operand);
Tensor Sinh(const Tensor& self);
Tensor Cosh(const Tensor& self);
Tensor Tanh(const Tensor& operand);
Tensor Asin(const Tensor& self);
Tensor Acos(const Tensor& self);
Tensor ASinh(const Tensor& self);
Tensor ACosh(const Tensor& self);
Tensor Atanh(const Tensor& self);

Tensor Duplicate(const Tensor& operand);
Tensor Pack(const Tensor& self);
Tensor UnPack(const Tensor& self, DataType dstDataType);
Tensor Gather(const Tensor& params, const Tensor& indices, int axis);
Tensor GatherElements(const Tensor& params, const Tensor& indices, int axis);
Tensor GatherMask(const Tensor& self, const uint8_t patternMode);

enum class ScatterMode {
    NONE,
    ADD,
    MULTIPLY,
    UNKNOWN,
};

/**
 * \brief Write the scalar value of src into self Tensor, with the write position specified by the indices Tensor.
 *
 * \param self : Tensor to write into.
 * \param indices : the index Tensor of element to be dispersed.
 * \param src : scalar value or tensor to be dispersed.
 * \param axis : axis to be indexed.
 * \param reduce : scatter reduction mode to be applied. Support NONE, ADD, MULTIPLY. NONE is default.
 * \return Tensor
 */
Tensor Scatter(const Tensor& self, const Tensor& indices, const Element& src, int axis,
               ScatterMode reduce = ScatterMode::NONE);
Tensor Scatter(const Tensor& self, const Tensor& indices, const Tensor& src, int axis,
               ScatterMode reduce = ScatterMode::NONE);
void IndexPut_(Tensor& self, const std::vector<Tensor>& indices, const Tensor& values, bool accumulate = false);
Tensor IndexAddUB(const Tensor& self, const Tensor& src, const Tensor& indices, int axis,
                  const Element& alpha = Element(DT_FP32, 1.0f));
void IndexAdd_(Tensor& self, const Tensor& src, const Tensor& indices, int axis,
               const Element& alpha = Element(DT_FP32, 1.0f));
Tensor RowSumExpand(const Tensor& operand);
Tensor RowMaxExpand(const Tensor& operand);

Tensor Sum(const Tensor& self, int axis = -1, bool keepDim = false);
Tensor Amax(const Tensor& self, int axis = -1, bool keepDim = false);
Tensor ArgMax(const Tensor& self, int axis = -1, bool keepDim = false);
Tensor ArgMin(const Tensor& self, int axis = -1, bool keepDim = false);
Tensor Amin(const Tensor& self, int axis = -1, bool keepDim = false);
Tensor Prod(const Tensor& self, int axis = -1, bool keepDim = false);
std::tuple<Tensor, Tensor, Tensor> OnlineSoftmax(const Tensor& scores, float scale);
std::tuple<Tensor, Tensor, Tensor> OnlineSoftmaxUpdate(const Tensor& previousMax, const Tensor& previousSum,
                                                       const Tensor& previousOutput, const Tensor& currentMax,
                                                       const Tensor& currentSum, const Tensor& currentOutput);

Tensor Compact(const Tensor& operand);

Tensor Add(const Tensor& self, const Tensor& other);
Tensor Sub(const Tensor& self, const Tensor& other);
Tensor Div(const Tensor& self, const Tensor& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Mul(const Tensor& self, const Tensor& other);
Tensor Hypot(const Tensor& self, const Tensor& other);
Tensor Fmod(const Tensor& self, const Tensor& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Maximum(const Tensor& operand1, const Tensor& operand2);
Tensor Minimum(const Tensor& operand1, const Tensor& operand2);
Tensor BitwiseAnd(const Tensor& self, const Tensor& other);
Tensor BitwiseOr(const Tensor& self, const Tensor& other);
Tensor BitwiseXor(const Tensor& self, const Tensor& other);
Tensor ExpandExpDif(const Tensor& input, const Tensor& other);
Tensor Add(const Tensor& self, const Element& other);
Tensor Sub(const Tensor& self, const Element& other);
Tensor Div(const Tensor& self, const Element& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Mul(const Tensor& self, const Element& other);
Tensor Fmod(const Tensor& self, const Element& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor BitwiseAnd(const Tensor& self, const Element& other);
Tensor BitwiseOr(const Tensor& self, const Element& other);
Tensor BitwiseXor(const Tensor& self, const Element& other);
Tensor Minimum(const Tensor& operand1, const Element& operand2);
Tensor Maximum(const Tensor& operand1, const Element& operand2);
Tensor Compare(const Tensor& self, const Tensor& other, OpType op, OutType mode);
Tensor Compare(const Tensor& self, const Element& other, OpType op, OutType mode);
Tensor Compare(const Element& self, const Tensor& other, OpType op, OutType mode);
Tensor Pow(const Tensor& self, const Tensor& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Pow(const Tensor& self, const Element& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Remainder(const Tensor& self, const Tensor& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Remainder(const Tensor& self, const Element& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Remainder(const Element& self, const Tensor& other, PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor CopySign(const Tensor& self, const Tensor& other);
Tensor PReLU(const Tensor& self, const Tensor& weight);
Tensor Atan2(const Tensor& y, const Tensor& x);
Tensor Axpy(const Tensor& self, const Tensor& other, float alpha);

std::tuple<Tensor, Tensor> Interleave(const Tensor& self, const Tensor& other);
std::tuple<Tensor, Tensor> DeInterleave(const Tensor& self, const Tensor& other);
std::tuple<Tensor, Tensor> DeInterleave(const Tensor& self);

Tensor BitwiseRightShift(const Tensor& self, const Tensor& other);
Tensor BitwiseRightShift(const Tensor& self, const Element& other);
Tensor BitwiseRightShift(const Element& self, const Tensor& other);
Tensor BitwiseLeftShift(const Tensor& self, const Tensor& other);
Tensor BitwiseLeftShift(const Tensor& self, const Element& other);
Tensor BitwiseLeftShift(const Element& self, const Tensor& other);

Tensor Where(const Tensor& condition, const Tensor& input, const Tensor& other);
Tensor Where(const Tensor& condition, const Tensor& input, const Element& other);
Tensor Where(const Tensor& condition, const Element& input, const Tensor& other);
Tensor Where(const Tensor& condition, const Element& input, const Element& other);

Tensor LReLU(const Tensor& self, const Element& negative_slope);

Tensor Unsqueeze(const Tensor& old, int unsqueezeDimNum);
Tensor Squeeze(const Tensor& input, const std::vector<int>& dim = {});

Tensor TensorIndex(const Tensor& params, const Tensor& indices);
Tensor ScatterUpdate(const Tensor& dst, const Tensor& index, const Tensor& src, int axis = -2,
                     std::string cacheMode = "PA_BNSD", int chunkSize = 1);

Tensor Expand(const Tensor& self, const std::vector<int64_t>& dstShape, std::vector<SymbolicScalar> validShape = {});

Tensor Var(const Tensor& input, const std::vector<int>& dim = {}, float correction = 1.0f, bool keepDim = false);
Tensor Softmax(const Tensor& operand);
Tensor RmsNorm(const Tensor& operand);
Tensor RmsNorm(const Tensor& operand, const Tensor& gamma, float epsilon = 1e-05f);
Tensor Cat(const std::vector<Tensor>& tensors, int axis);
Tensor NewCompact(const Tensor& operand);
Tensor LogicalNot(const Tensor& self);
Tensor Range(const Element& start, const Element& end, const Element& step);
Tensor LogicalAnd(const Tensor& self, const Tensor& other);
Tensor IsFinite(const Tensor& self);
Tensor IsNan(const Tensor& self);
Tensor Assign(const Tensor& operand);

Tensor Uniform(const Element& key, const SymbolicScalar& counter0, const Element& counter1,
               const std::vector<int64_t>& shape, const Element& rounds, DataType dtype = DT_FP32);

// Implementation of `Tensor` type should be placed at first, so that it can be routed when only single input.
Tensor Clip(const Tensor& self, const Tensor& min = {}, const Tensor& max = {});
Tensor Clip(const Tensor& self, const Element& min = {}, const Element& max = {});

std::tuple<Tensor, Tensor> TopK(const Tensor& self, int k, int axis = -1, bool isLargest = true,
                                TopKAlgo algo = TopKAlgo::MERGE_SORT);
Tensor ArgSort(const Tensor& self, int axis = -1, bool descending = false);
Tensor Sort32(const Tensor& self, int idxStart = 0);
Tensor MrgSort(const Tensor& self, int mergeSize);

// support float32 -> int8/uint8
Tensor Quantize(const Tensor& input, const Tensor& scale, DataType dtype, int axis, const Tensor& zeroPoints);
// support int8/int16 -> float32
Tensor Dequantize(const Tensor& input, const Tensor& scale, DataType otype, int axis, const Tensor& zeroPoints);

/**
 * @brief Sort a tensor with shape (1, n) along the last dimension, n must be orders of 2.
 *        The vecTile (1, t), t must be orders of 2, maximum is 8K.
 * @param x The input tensor to be sorted, the indices are initialized to 0123...
 * @param descending If true, sorts in descending order; otherwise ascending order (default: true).
 * @return std::tuple<Tensor, Tensor> A tuple containing two tensors:
 *         - First tensor: The sorted data.
 *         - Second tensor: The corresponding indices.
 */
std::tuple<Tensor, Tensor> Sort(const Tensor& x, bool descending = true);

/**
 * @brief Sort a tensor & indices with shape (1, n) along the last dimension, n must be orders of 2.
 *        The vecTile (1, t), t must be orders of 2, maximum is 8K.
 * @param x The input tensor to be sorted.
 * @param idx The input indices corresponding to x.
 * @param descending If true, sorts in descending order; otherwise ascending order (default: true).
 * @return std::tuple<Tensor, Tensor> A tuple containing two tensors:
 *         - First tensor: The sorted data.
 *         - Second tensor: The corresponding indices.
 */
std::tuple<Tensor, Tensor> SortWithIndex(const Tensor& x, const Tensor& idx, bool descending = true);

Tensor SoftmaxNew(const Tensor& operand);
void SoftmaxDynamic(Tensor& input, Tensor& output);

Tensor RotateHalf(const Tensor& input);

// moe
Tensor Sigmoid(Tensor& input);

std::tuple<Tensor, Tensor> Quant(const Tensor& input, bool isSymmetry = true, bool hasSmoothFactor = false,
                                 const Tensor& smoothFactor = Tensor());
std::tuple<Tensor, Tensor> QuantMX(const Tensor& input, DataType quantDtype = DataType::DT_FP8E4M3,
                                   DequantScaleRoundingMode mode = DequantScaleRoundingMode::ROUND_DOWN,
                                   int64_t axis = -1, bool performanceMode = true);

Tensor ScalarDivS(const Tensor& operand, const Element& value, bool reverseOperand = false);
Tensor ScalarAddS(const Tensor& operand, const Element& value, bool reverseOperand = false);
Tensor ScalarMaxS(const Tensor& operand, const Element& value, bool reverseOperand = false);
Tensor ScalarSubS(const Tensor& operand, const Element& value, bool reverseOperand = false);
Tensor ScalarMulS(const Tensor& operand, const Element& value, bool reverseOperand = false);

Tensor ScalarSub(const Tensor& operand1, const Tensor& operand2);
Tensor ScalarDiv(const Tensor& operand1, const Tensor& operand2);
Tensor CumSum(const Tensor& input, const int& axis);
Tensor CumProd(const Tensor& input, const int& axis);
Tensor Gcd(const Tensor& input, const Tensor& other);
Tensor Gcd(const Tensor& input, const Element& other);
Tensor TriU(const Tensor& input, const SymbolicScalar& diagonal);
Tensor TriL(const Tensor& input, const SymbolicScalar& diagonal);
struct PaTileShapeConfig {
    int headNumQTile;
    std::array<int, TILE_VEC_DIMS> v0TileShape;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

enum class ReduceMode {
    ATOMIC_ADD,
};
// template <ReduceMode reduceMode>
Tensor Reduce(const std::vector<Tensor>& aggregation, const ReduceMode reduceMode);

Tensor Maxpool(const Tensor& operand, const std::vector<int>& pools, const std::vector<int>& strides,
               const std::vector<int>& paddings);

enum class LogBaseType {
    LOG_E,
    LOG_2,
    LOG_10,
};
Tensor Log(const Tensor& self, LogBaseType base = LogBaseType::LOG_E,
           PrecisionType precisionType = PrecisionType::INTRINSIC);
Tensor Log1p(const Tensor& self);

Tensor OneHot(const Tensor& self, int numClasses);

struct IfaTileShapeConfig {
    int blockSize;
    int headNumQTile;
    std::array<int, TILE_VEC_DIMS> v0TileShape;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

struct RoPETileShapeConfig {
    std::vector<int64_t> twoDimsTileShape;
    std::vector<int64_t> threeDimsTileShape;
    std::vector<int64_t> fourDimsTileShape;
    std::vector<int64_t> fiveDimsTileShape;
};

struct RoPETileShapeConfigNew {
    std::vector<int64_t> threeDimsTileShape;
    std::vector<int64_t> fourDimsTileShapeQ;
    std::vector<int64_t> fourDimsTileShapeK;
    std::vector<int64_t> fiveDimsTileShape;
};

namespace Matrix {

enum class ReLuType : int64_t { NoReLu = 0, ReLu = 1 };

enum class TransMode : int64_t { CAST_NONE = 0, CAST_RINT = 1, CAST_ROUND = 2 };

struct MatmulExtendParam {
    Tensor biasTensor{Tensor()};
    Tensor scaleTensor{Tensor()};
    float scaleValue{0.0f};
    ReLuType reluType{ReLuType::NoReLu};
    TransMode transMode{TransMode::CAST_NONE};

    MatmulExtendParam(Tensor bias, Tensor scale, float scaleVal, ReLuType relu, TransMode mode = TransMode::CAST_NONE)
        : biasTensor(std::move(bias)),
          scaleTensor(std::move(scale)),
          scaleValue(scaleVal),
          reluType(relu),
          transMode(mode)
    {}

    MatmulExtendParam() = default;
};

Tensor Matmul(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix, bool isATrans = false,
              bool isBTrans = false, bool isCMatrixNZ = false);

Tensor Matmul(DataType outType, const Tensor& aMatrix, const Tensor& bMatrix, const MatmulExtendParam& extendParam,
              bool isATrans = false, bool isBTrans = false, bool isCMatrixNZ = false);

Tensor MatmulMX(DataType outType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                const Tensor& bScale, bool isATrans = false, bool isAScaleTrans = false, bool isBTrans = false,
                bool isBScaleTrans = false, bool isCMatrixNZ = false);

Tensor MatmulMX(DataType outType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                const Tensor& bScale, const MatmulExtendParam& extendParam, bool isATrans = false,
                bool isAScaleTrans = false, bool isBTrans = false, bool isBScaleTrans = false,
                bool isCMatrixNZ = false);

Tensor BatchMatmul(DataType dataType, const Tensor& aMatrix, const Tensor& bMatrix, bool isATrans = false,
                   bool isBTrans = false, bool isCMatrixNZ = false);

Tensor BatchMatmul(DataType dataType, const Tensor& aMatrix, const Tensor& bMatrix,
                   const MatmulExtendParam& extendParam, bool isATrans = false, bool isBTrans = false,
                   bool isCMatrixNZ = false);

Tensor BatchMatmulMX(DataType dataType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                     const Tensor& bScale, bool isTransA = false, bool isAScaleTrans = false, bool isTransB = false,
                     bool isBScaleTrans = false, bool isCMatrixNZ = false);

Tensor BatchMatmulMX(DataType dataType, const Tensor& aMatrix, const Tensor& aScale, const Tensor& bMatrix,
                     const Tensor& bScale, const MatmulExtendParam& extendParam, bool isTransA = false,
                     bool isAScaleTrans = false, bool isTransB = false, bool isBScaleTrans = false,
                     bool isCMatrixNZ = false);

Tensor TransposedBatchMatmul(DataType dataType, const Tensor& aMatrix, const Tensor& bMatrix);

Tensor QuantMM(const Tensor& operand1, const Tensor& operand2, const Tensor& dequantScaleW);
} // namespace Matrix

namespace Conv {

struct TileL1Info {
    int64_t tileHin{0};
    int64_t tileHout{0};
    int64_t tileWin{0};
    int64_t tileWout{0};
    int64_t tileCinFmap{0};
    int64_t tileCinWeight{0};
    int64_t tileN{0};
    int64_t tileBatch{0};

    TileL1Info(int64_t hin, int64_t hout, int64_t win, int64_t wout, int64_t cinFmap, int64_t cinWeight, int64_t cout,
               int64_t n)
        : tileHin(hin),
          tileHout(hout),
          tileWin(win),
          tileWout(wout),
          tileCinFmap(cinFmap),
          tileCinWeight(cinWeight),
          tileN(cout),
          tileBatch(n)
    {}

    TileL1Info() = default;
};

struct TileL0Info {
    int64_t tileH{0};
    int64_t tileW{0};
    int64_t tileK{0};
    int64_t tileN{0};

    TileL0Info(int64_t h, int64_t w, int64_t k, int64_t n) : tileH(h), tileW(w), tileK(k), tileN(n) {}

    TileL0Info() = default;
};

enum class ReLuType : int64_t { NoReLu = 0, ReLu = 1 };

struct ConvExtendParam {
    Tensor biasTensor{Tensor()};
    Tensor scaleTensor{Tensor()};
    float scaleValue{0.0f};
    ReLuType reluType{ReLuType::NoReLu};

    ConvExtendParam(Tensor bias, Tensor scale, float scaleVal, ReLuType relu)
        : biasTensor(std::move(bias)), scaleTensor(std::move(scale)), scaleValue(scaleVal), reluType(relu)
    {}

    ConvExtendParam() = default;
};

Tensor Conv(DataType outType, const Tensor& inputTensor, const Tensor& weightTensor,
            const std::vector<int64_t>& strides, const std::vector<SymbolicScalar>& paddings,
            const std::vector<int64_t>& dilations, const ConvExtendParam& extendParam, const int64_t groups = 1);

} // namespace Conv

namespace Distributed {
enum class DistReduceType {
    DIST_REDUCE_ADD,
    DIST_REDUCE_MAX,
    DIST_REDUCE_MIN,
};

enum class AtomicType { SET, ADD };

struct ShmemTensor {
    std::string group;
    int64_t worldSize{0};
    Tensor data;
    Tensor signal;
    void* signalOp{nullptr};
};
ShmemTensor CreateShmemTensor(const char* group, int64_t worldSize, DataType dataType, const Shape& shape);
void CreateShmemTensor(const char* group, int64_t worldSize, DataType dataType, const Shape& shape, ShmemTensor& t);
ShmemTensor CreateShmemSignal(const char* group, int64_t worldSize);
void CreateShmemSignal(const char* group, int64_t worldSize, ShmemTensor& t);
ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::vector<int64_t>& offsets);
ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::vector<SymbolicScalar>& offsets);
ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::vector<SymbolicScalar>& newValidShapes, const std::vector<SymbolicScalar>& newOffsets);
ShmemTensor ShmemView(const ShmemTensor& operand, const std::vector<int64_t>& shapes,
                      const std::initializer_list<SymbolicScalar>& newOffsets);
Tensor ShmemPut(const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, AtomicType putOp,
                const Tensor& pred);
Tensor ShmemGet(const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred,
                DataType targetDataType = DataType::DT_BOTTOM);
Tensor ShmemSignal(const ShmemTensor& src, const SymbolicScalar& srcRank, const SymbolicScalar& targetRank,
                   int32_t signal, AtomicType sigOp, const Tensor& pred);
Tensor ShmemSignalAll(const ShmemTensor& src, const SymbolicScalar& srcRank, int32_t signal, AtomicType sigOp,
                      const Tensor& pred);
Tensor ShmemWaitUntil(const ShmemTensor& src, const SymbolicScalar& srcRank, OpType cmp, int32_t cmpValue,
                      bool clearSignal, const Tensor& pred);
Tensor ShmemClearData(const ShmemTensor& src, Tensor& pred);
Tensor ShmemClearSignal(const ShmemTensor& src, Tensor& pred);
Tensor ShmemBarrier(const ShmemTensor& src, const Tensor& pred);
Tensor ShmemLoad(const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred,
                 DataType targetDataType = DataType::DT_BOTTOM);
Tensor ShmemStore(const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, AtomicType putOp,
                  const Tensor& pred);

void AllGather(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out);
void ReduceScatter(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, DistReduceType reduceType,
                   Tensor& out);
void OneShotAllReduce(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out);
void TwoShotAllReduce(const Tensor& predToken, const Tensor& in, ShmemTensor& shmemTensor, Tensor& out);

} // namespace Distributed
std::tuple<Tensor, Tensor> TopKSort(const Tensor& x, int idxStart);
std::tuple<Tensor, Tensor> TopKSort(const Tensor& x, const SymbolicScalar& idxStart);
Tensor TopKExtract(const Tensor& x, int k, bool isIndex);
Tensor TopKMerge(const Tensor& x, int mergeSize);
Tensor Nop(const std::vector<Tensor>& inTensors);
} // namespace npu::tile_fwk
