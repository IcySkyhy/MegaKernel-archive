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
 * \file operation.cpp
 * \brief
 */

#include "pybind_common.h"

#include <vector>

using namespace npu::tile_fwk;

namespace pypto {
constexpr const int SCATTER_UPDATE_DIM = -2;
constexpr uint16_t DEFAULT_UNIFORM_ROUNDS = 10;
void BindOperation(py::module_& m)
{
    m.def(
        "Add", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::Add(self, other); }, "Tensor add.");
    m.def(
        "Axpy", [](const Tensor& y, const Tensor& x, float alpha) { return npu::tile_fwk::Axpy(y, x, alpha); },
        py::arg("y"), py::arg("x"), py::arg("alpha") = 1.0f, "Tensor axpy: y = alpha * x + y.");
    m.def(
        "Sub", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::Sub(self, other); }, "Tensor sub.");
    m.def(
        "Mul", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::Mul(self, other); }, "Tensor mul.");
    m.def(
        "Div",
        [](const Tensor& self, const Tensor& other, PrecisionType precisionType) {
            return npu::tile_fwk::Div(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION, "Tensor div.");
    m.def(
        "Hypot", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::Hypot(self, other); },
        "Tensor hypot.");
    m.def(
        "Fmod",
        [](const Tensor& self, const Tensor& other, PrecisionType precisionType) {
            return npu::tile_fwk::Fmod(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION, "Tensor fmod.");
    m.def(
        "Gcd", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::Gcd(self, other); }, "Tensor gcd.");
    m.def(
        "Remainder",
        [](const Tensor& self, const Tensor& other, PrecisionType precisionType) {
            return npu::tile_fwk::Remainder(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION,
        "Tensor remainder.");
    m.def(
        "Remainder",
        [](const Tensor& self, const Element& other, PrecisionType precisionType) {
            return npu::tile_fwk::Remainder(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION,
        "Tensor remainder scalar.");
    m.def(
        "Remainder",
        [](const Element& self, const Tensor& other, PrecisionType precisionType) {
            return npu::tile_fwk::Remainder(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION,
        "Scalar remainder tensor.");
    m.def(
        "BitwiseAnd", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::BitwiseAnd(self, other); },
        "Tensor bitwise and.");
    m.def(
        "BitwiseOr", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::BitwiseOr(self, other); },
        "Tensor bitwise or.");
    m.def(
        "BitwiseXor", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::BitwiseXor(self, other); },
        "Tensor bitwise xor.");
    m.def(
        "ExpandExpDif",
        [](const Tensor& input, const Tensor& other) { return npu::tile_fwk::ExpandExpDif(input, other); },
        "Tensor expand exp dif.");
    m.def(
        "View",
        [](const Tensor& operand, const std::vector<int64_t>& shapes, const py::sequence& offsets) {
            std::vector<SymbolicScalar> dynOffsets;
            for (const auto& item : offsets) {
                if (py::isinstance<SymbolicScalar>(item)) {
                    dynOffsets.push_back(item.cast<SymbolicScalar>());
                } else if (py::isinstance<py::int_>(item)) {
                    dynOffsets.push_back(item.cast<int64_t>());
                }
            }
            return npu::tile_fwk::View(operand, shapes, dynOffsets);
        },
        py::arg("operand"), py::arg("shapes"), py::arg("offsets"),
        "Create a view of a tensor. The 'offsets' can contain symbolic scalars.");
    m.def(
        "View",
        [](const Tensor& operand, const std::vector<int64_t>& shapes, const std::vector<SymbolicScalar>& newValidShapes,
           const std::vector<SymbolicScalar>& newOffsets) {
            return npu::tile_fwk::View(operand, shapes, newValidShapes, newOffsets);
        },
        py::arg("operand"), py::arg("shapes"), py::arg("new_valid_shapes"), py::arg("new_offsets"),
        "Tensor dview_pad.");
    m.def(
        "View",
        [](const Tensor& operand, const DataType dstDataType) { return npu::tile_fwk::View(operand, dstDataType); },
        py::arg("operand"), py::arg("dstDataType"), "Tensor view_type.");

    m.def(
        "Exp", [](const Tensor& self, PrecisionType precisionType) { return npu::tile_fwk::Exp(self, precisionType); },
        py::arg("self"), py::arg("precision_type") = PrecisionType::INTRINSIC, "Tensor exp.");
    m.def("Expm1", [](const Tensor& self) { return npu::tile_fwk::Expm1(self); }, "Tensor expm1.");
    m.def("Erf", [](const Tensor& self) { return npu::tile_fwk::Erf(self); }, "Tensor erf.");
    m.def("Sin", [](const Tensor& self) { return npu::tile_fwk::Sin(self); }, "Tensor sin.");
    m.def("Cos", [](const Tensor& self) { return npu::tile_fwk::Cos(self); }, "Tensor cos.");
    m.def("Exp2", [](const Tensor& self) { return npu::tile_fwk::Exp2(self); }, "Tensor exp2.");
    m.def(
        "Permute", [](const Tensor& self, const std::vector<int>& perm) { return npu::tile_fwk::Permute(self, perm); },
        "Tensor transpose.");

    m.def(
        "Transpose",
        [](const Tensor& self, const std::vector<int>& perm) { return npu::tile_fwk::Transpose(self, perm); },
        "Tensor transpose.");
    m.def("Abs", [](const Tensor& self) { return npu::tile_fwk::Abs(self); }, "Tensor abs.");
    m.def("Reciprocal", [](const Tensor& operand) { return npu::tile_fwk::Reciprocal(operand); }, "Tensor reciprocal.");
    m.def("Relu", [](const Tensor& operand) { return npu::tile_fwk::Relu(operand); }, "Tensor relu.");
    m.def("Pad", &npu::tile_fwk::Pad, "Pads tensor with constant value (supports right/bottom padding only).",
          py::arg("input"), py::arg("pad"), py::arg("mode") = "constant", py::arg("value") = Element(DT_FP32, 0.0));
    m.def("FillPad", &npu::tile_fwk::FillPad, "Fills padding region of tensor with constant value.", py::arg("input"),
          py::arg("mode") = "constant", py::arg("value") = Element(DT_FP32, 0.0));
    m.def(
        "Round", [](const Tensor& self, int decimals) { return npu::tile_fwk::Round(self, decimals); }, py::arg("self"),
        py::arg("decimals") = 0, "Tensor round.");
    m.def(
        "Rsqrt",
        [](const Tensor& self, PrecisionType precisionType) { return npu::tile_fwk::Rsqrt(self, precisionType); },
        py::arg("self"), py::arg("precision_type") = PrecisionType::INTRINSIC, "Tensor rsqrt.");
    m.def(
        "Sqrt",
        [](const Tensor& self, PrecisionType precisionType) { return npu::tile_fwk::Sqrt(self, precisionType); },
        py::arg("self"), py::arg("precision_type") = PrecisionType::INTRINSIC, "Tensor sqrt.");
    m.def("Sign", [](const Tensor& self) { return npu::tile_fwk::Sign(self); }, "Tensor sign.");
    m.def("Signbit", [](const Tensor& self) { return npu::tile_fwk::Signbit(self); }, "Tensor signbit.");
    m.def("Tanh", [](const Tensor& self) { return npu::tile_fwk::Tanh(self); }, "Tensor tanh.");
    m.def("Ceil", [](const Tensor& self) { return npu::tile_fwk::Ceil(self); }, "Tensor ceil.");
    m.def("Floor", [](const Tensor& self) { return npu::tile_fwk::Floor(self); }, "Tensor floor.");
    m.def("Sinh", [](const Tensor& self) { return npu::tile_fwk::Sinh(self); }, "Tensor sinh");
    m.def("Cosh", [](const Tensor& self) { return npu::tile_fwk::Cosh(self); }, "Tensor cosh");
    m.def("Erfc", [](const Tensor& self) { return npu::tile_fwk::Erfc(self); }, "Tensor erfc");
    m.def("ASinh", [](const Tensor& self) { return npu::tile_fwk::ASinh(self); }, "Tensor asinh");
    m.def("ACosh", [](const Tensor& self) { return npu::tile_fwk::ACosh(self); }, "Tensor acosh");
    m.def("Atan", [](const Tensor& self) { return npu::tile_fwk::Atan(self); }, "Tensor atan");
    m.def("Atan2", [](const Tensor& y, const Tensor& x) { return npu::tile_fwk::Atan2(y, x); }, "Tensor atan2");
    m.def("Asin", [](const Tensor& self) { return npu::tile_fwk::Asin(self); }, "Tensor asin.");
    m.def("Acos", [](const Tensor& self) { return npu::tile_fwk::Acos(self); }, "Tensor acos.");
    m.def("Atanh", [](const Tensor& self) { return npu::tile_fwk::Atanh(self); }, "Tensor Atanh");
    m.def("FloorDiv", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::FloorDiv(self, other); });
    m.def("FloorDiv", [](const Tensor& self, const Element& other) { return npu::tile_fwk::FloorDiv(self, other); });
    m.def("Trunc", [](const Tensor& self) { return npu::tile_fwk::Trunc(self); }, "Tensor trunc.");
    m.def("BitwiseNot", [](const Tensor& self) { return npu::tile_fwk::BitwiseNot(self); }, "Tensor bitwisenot.");
    m.def("Neg", [](const Tensor& self) { return npu::tile_fwk::Neg(self); }, "Tensor neg.");
    m.def(
        "Reciprocal",
        [](const Tensor& self, PrecisionType precisionType) { return npu::tile_fwk::Reciprocal(self, precisionType); },
        py::arg("self"), py::arg("precision_type") = PrecisionType::INTRINSIC, "Tensor reciprocal.");
    m.def(
        "Log",
        [](const Tensor& self, const LogBaseType base, PrecisionType precisionType) {
            return npu::tile_fwk::Log(self, base, precisionType);
        },
        py::arg("self"), py::arg("base"), py::arg("precision_type") = PrecisionType::INTRINSIC, "Tensor log.");
    m.def("Log1p", [](const Tensor& self) { return npu::tile_fwk::Log1p(self); }, "Tensor log1p.");
    m.def("Tan", [](const Tensor& self) { return npu::tile_fwk::Tan(self); }, "Tensor tan.");
    m.def("Pack", [](const Tensor& self) { return npu::tile_fwk::Pack(self); }, "Tensor pack.");
    m.def(
        "UnPack", [](const Tensor& self, DataType dstDataType) { return npu::tile_fwk::UnPack(self, dstDataType); },
        "Tensor unpack.");
    m.def(
        "Pow",
        [](const Tensor& self, const Tensor& other, PrecisionType precisionType) {
            return npu::tile_fwk::Pow(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION, "Tensor pow.");
    m.def(
        "Pow",
        [](const Tensor& self, const Element& other, PrecisionType precisionType) {
            return npu::tile_fwk::Pow(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION,
        "Tensor pow scalar.");
    m.def(
        "Cast",
        [](const Tensor& self, DataType dstDataType, CastMode mode, SaturationMode satmode) {
            return npu::tile_fwk::Cast(self, dstDataType, mode, satmode);
        },
        py::arg("operand"), py::arg("new_data_type"), py::arg("mode") = CAST_NONE,
        py::arg("satmode") = SaturationMode::OFF, "Tensor cast.");
    m.def(
        "Quantize",
        [](const Tensor& input, const Tensor& scale, DataType otype, int axis, const Tensor& zeroPoints) {
            return npu::tile_fwk::Quantize(input, scale, otype, axis, zeroPoints);
        },
        py::arg("input"), py::arg("scale"), py::arg("otype"), py::arg("axis"), py::arg("zero_points") = Tensor(),
        "Tensor Quantize.");
    m.def(
        "Dequantize",
        [](const Tensor& input, const Tensor& scale, DataType otype, int axis, const Tensor& zeroPoints) {
            return npu::tile_fwk::Dequantize(input, scale, otype, axis, zeroPoints);
        },
        py::arg("input"), py::arg("scale"), py::arg("otype"), py::arg("axis"), py::arg("zero_points") = Tensor(),
        "Tensor Dequantize.");
    m.def(
        "Add", [](const Tensor& self, const Element& other) { return npu::tile_fwk::Add(self, other); },
        "Tensor add scalar.");
    m.def(
        "Sub", [](const Tensor& left, const Element& right) { return npu::tile_fwk::Sub(left, right); },
        "Tensor sub scalar.");
    m.def(
        "Mul", [](const Tensor& self, const Element& other) { return npu::tile_fwk::Mul(self, other); },
        "Tensor mul scalar.");
    m.def(
        "Div",
        [](const Tensor& self, const Element& other, PrecisionType precisionType) {
            return npu::tile_fwk::Div(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION,
        "Tensor div scalar.");
    m.def(
        "Fmod",
        [](const Tensor& self, const Element& other, PrecisionType precisionType) {
            return npu::tile_fwk::Fmod(self, other, precisionType);
        },
        py::arg("self"), py::arg("other"), py::arg("precision_type") = PrecisionType::HIGH_PRECISION,
        "Tensor mod scalar.");
    m.def(
        "Gcd", [](const Tensor& self, const Element& other) { return npu::tile_fwk::Gcd(self, other); },
        "Tensor gcd scalar.");
    m.def(
        "LReLU", [](const Tensor& self, const Element& alpha) { return npu::tile_fwk::LReLU(self, alpha); },
        "Tensor LReLU scalar.");
    m.def(
        "BitwiseRightShift",
        [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::BitwiseRightShift(self, other); },
        "Tensor bitwise right shift.");
    m.def(
        "BitwiseLeftShift",
        [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::BitwiseLeftShift(self, other); },
        "Tensor bitwise left shift.");
    m.def(
        "BitwiseRightShift",
        [](const Tensor& self, const Element& other) { return npu::tile_fwk::BitwiseRightShift(self, other); },
        "Tensor bitwise right shift scalar.");
    m.def(
        "BitwiseLeftShift",
        [](const Tensor& self, const Element& other) { return npu::tile_fwk::BitwiseLeftShift(self, other); },
        "Tensor bitwise left shift scalar.");
    m.def(
        "BitwiseRightShift",
        [](const Element& self, const Tensor& other) { return npu::tile_fwk::BitwiseRightShift(self, other); },
        "Scalar bitwise right shift tensor.");
    m.def(
        "BitwiseLeftShift",
        [](const Element& self, const Tensor& other) { return npu::tile_fwk::BitwiseLeftShift(self, other); },
        "Scalar bitwise left shift tensor.");
    m.def(
        "BitwiseAnd", [](const Tensor& self, const Element& other) { return npu::tile_fwk::BitwiseAnd(self, other); },
        "Tensor bitwiseand scalar.");
    m.def(
        "BitwiseOr", [](const Tensor& self, const Element& other) { return npu::tile_fwk::BitwiseOr(self, other); },
        "Tensor bitwiseor scalar.");
    m.def(
        "BitwiseXor", [](const Tensor& self, const Element& other) { return npu::tile_fwk::BitwiseXor(self, other); },
        "Tensor bitwisexor scalar.");
    m.def(
        "Range",
        [](const Element& start, const Element& end, const Element& step) {
            return npu::tile_fwk::Range(start, end, step);
        },
        py::arg("start"), py::arg("end"), py::arg("step"), "Tensor range.");
    m.def(
        "Uniform",
        [](const Element& key, const SymbolicScalar& counter0, const Element& counter1,
           const std::vector<int64_t>& shape, const Element& rounds,
           DataType dtype) { return npu::tile_fwk::Uniform(key, counter0, counter1, shape, rounds, dtype); },
        py::arg("key"), py::arg("counter0"), py::arg("counter1"), py::arg("shape"),
        py::arg("rounds") = Element(DT_UINT16, static_cast<uint16_t>(DEFAULT_UNIFORM_ROUNDS)),
        py::arg("dtype") = DT_FP32, "Uniform random number generator.");
    m.def(
        "Amax",
        [](const Tensor& operand, int axis, bool keepDim) { return npu::tile_fwk::Amax(operand, axis, keepDim); },
        py::arg("operand"), py::arg("axis") = -1, py::arg("keepDim") = false, "Tensor row max single.");

    m.def(
        "Sum", [](const Tensor& operand, int axis, bool keepDim) { return npu::tile_fwk::Sum(operand, axis, keepDim); },
        py::arg("operand"), py::arg("axis") = -1, py::arg("keepDim") = false, "Tensor row sum single.");
    m.def(
        "Amin",
        [](const Tensor& operand, int axis, bool keepDim) { return npu::tile_fwk::Amin(operand, axis, keepDim); },
        py::arg("operand"), py::arg("axis") = -1, py::arg("keepDim") = false, "Tensor row min single.");
    m.def(
        "Prod",
        [](const Tensor& operand, int axis, bool keepDim) { return npu::tile_fwk::Prod(operand, axis, keepDim); },
        py::arg("operand"), py::arg("axis") = -1, py::arg("keepDim") = false, "Tensor row prod single.");
    m.def(
        "RowSumExpand", [](const Tensor& operand) { return npu::tile_fwk::RowSumExpand(operand); },
        "Tensor row sum expand.");
    m.def(
        "RowMaxExpand", [](const Tensor& operand) { return npu::tile_fwk::RowMaxExpand(operand); },
        "Tensor row max expand.");
    m.def("Compact", [](const Tensor& operand) { return npu::tile_fwk::Compact(operand); }, "Tensor compact.");
    m.def(
        "IndexPut_",
        [](Tensor& self, std::vector<Tensor> indices, const Tensor& values, bool accumulate) {
            npu::tile_fwk::IndexPut_(self, indices, values, accumulate);
        },
        "Tensor indexput_.");
    m.def(
        "Scatter",
        [](const Tensor& self, const Tensor& indices, const Element& src, int axis, ScatterMode reduce) {
            return npu::tile_fwk::Scatter(self, indices, src, axis, reduce);
        },
        py::arg("self"), py::arg("indices"), py::arg("src"), py::arg("axis"), py::arg("reduce") = ScatterMode::NONE,
        "Tensor scatter element noninplace.");
    m.def(
        "Scatter",
        [](const Tensor& self, const Tensor& indices, const Tensor& src, int axis, ScatterMode reduce) {
            return npu::tile_fwk::Scatter(self, indices, src, axis, reduce);
        },
        py::arg("self"), py::arg("indices"), py::arg("src"), py::arg("axis"), py::arg("reduce") = ScatterMode::NONE,
        "Tensor scatter noninplace.");
    m.def(
        "IndexAddUB",
        [](const Tensor& self, const Tensor& src, const Tensor& indices, int axis, const Element& alpha) {
            return npu::tile_fwk::IndexAddUB(self, src, indices, axis, alpha);
        },
        py::arg("self"), py::arg("src"), py::arg("indices"), py::arg("axis"),
        py::arg("alpha") = npu::tile_fwk::Element(DT_FP32, 1.0), "Tensor add with index.");
    m.def(
        "IndexAdd_",
        [](Tensor& self, const Tensor& src, const Tensor& indices, int axis, const Element& alpha) {
            npu::tile_fwk::IndexAdd_(self, src, indices, axis, alpha);
        },
        py::arg("self"), py::arg("src"), py::arg("indices"), py::arg("axis"),
        py::arg("alpha") = npu::tile_fwk::Element(DT_FP32, 1.0), "Tensor add with index inplacely.");
    m.def(
        "GatherElements",
        [](const Tensor& params, const Tensor& indices, int axis) {
            return npu::tile_fwk::GatherElements(params, indices, axis);
        },
        "Tensor gather element.");
    m.def(
        "Gather",
        [](const Tensor& params, const Tensor& indices, int axis) {
            return npu::tile_fwk::Gather(params, indices, axis);
        },
        "Tensor gather.");
    m.def(
        "GatherMask", [](const Tensor& self, int patternMode) { return npu::tile_fwk::GatherMask(self, patternMode); },
        "Tensor gather Mask.");
    m.def("Duplicate", [](const Tensor& operand) { return npu::tile_fwk::Duplicate(operand); }, "Tensor duplicate.");
    m.def(
        "Full",
        [](const Element& src, DataType dType, std::vector<int64_t> dstShape, std::vector<SymbolicScalar> validShape) {
            return npu::tile_fwk::Full(src, dType, dstShape, validShape);
        },
        py::arg("src"), py::arg("dType"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def(
        "Full",
        [](const SymbolicScalar& src, DataType dType, std::vector<int64_t> dstShape,
           std::vector<SymbolicScalar> validShape) { return npu::tile_fwk::Full(src, dType, dstShape, validShape); },
        py::arg("src"), py::arg("dType"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        "Tensor vector duplicate.");
    m.def(
        "Reshape",
        [](const Tensor& input, const std::vector<int64_t>& dstShape, const std::vector<SymbolicScalar> validShape,
           const bool inplace) { return npu::tile_fwk::Reshape(input, dstShape, validShape, inplace); },
        py::arg("input"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{},
        py::arg("inplace") = false, "Tensor reshape.");
    m.def(
        "Reshape",
        [](const Tensor& input, const std::vector<SymbolicScalar>& dstShape, const bool inplace) {
            return npu::tile_fwk::Reshape(input, dstShape, inplace);
        },
        py::arg("input"), py::arg("dstShape"), py::arg("inplace"), "Tensor reshapeInplace.");
    m.def(
        "ReshapeInplace", [](const Tensor& input, Tensor& dst) { npu::tile_fwk::Reshape(input, dst); },
        py::arg("input"), py::arg("dst"), "Tensor reshapeInplace.");
    m.def(
        "Assign", [](const Tensor& input) { return npu::tile_fwk::Assign(input); }, py::arg("input"), "Tensor clone.");
    m.def(
        "Reduce",
        [](const std::vector<Tensor>& aggregation, const ReduceMode& reduceMode) {
            return npu::tile_fwk::Reduce(aggregation, reduceMode);
        },
        py::arg("aggregation"), py::arg("reduce_mode"), "Tensor reduce.");

    m.def(
        "Maximum", [](const Tensor& left, const Tensor& right) { return npu::tile_fwk::Maximum(left, right); },
        py::arg("left"), py::arg("right"), "Tensor maximum.");
    m.def(
        "Minimum", [](const Tensor& left, const Tensor& right) { return npu::tile_fwk::Minimum(left, right); },
        py::arg("left"), py::arg("right"), "Tensor minimum.");
    m.def(
        "Unsqueeze",
        [](const Tensor& old, int unsqueezeDimNum) { return npu::tile_fwk::Unsqueeze(old, unsqueezeDimNum); },
        "Tensor unsqueeze.");
    m.def(
        "TensorIndex",
        [](const Tensor& params, const Tensor& indices) { return npu::tile_fwk::TensorIndex(params, indices); },
        "Tensor index.");
    m.def(
        "index_select",
        [](const Tensor& params, int dim, const Tensor& indices) {
            return npu::tile_fwk::Gather(params, indices, dim);
        },
        "Tensor index_select.");
    m.def(
        "ScatterUpdate",
        [](const Tensor& dst, const Tensor& index, const Tensor& src, int axis, std::string cacheMode, int chunkSize) {
            return npu::tile_fwk::ScatterUpdate(dst, index, src, axis, cacheMode, chunkSize);
        },
        py::arg("dst"), py::arg("index"), py::arg("src"), py::arg("axis") = SCATTER_UPDATE_DIM,
        py::arg("cacheMode") = "PA_BNSD", py::arg("chunkSize") = 1, "Tensor scatter update.");
    m.def(
        "Expand",
        [](const Tensor& self, const std::vector<int64_t>& dstShape, std::vector<SymbolicScalar> validShape) {
            return npu::tile_fwk::Expand(self, dstShape, validShape);
        },
        py::arg("self"), py::arg("dstShape"), py::arg("validShape") = std::vector<SymbolicScalar>{}, "Tensor expand.");
    m.def(
        "NewCompact", [](const Tensor& operand) { return npu::tile_fwk::NewCompact(operand); }, "Tensor new compact.");
    m.def("LogicalNot", [](const Tensor& self) { return npu::tile_fwk::LogicalNot(self); }, "Tensor logical not.");
    m.def(
        "LogicalAnd", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::LogicalAnd(self, other); },
        "Tensor logical and.");
    m.def(
        "Where", [](const Tensor& a, const Tensor& b, const Tensor& c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "Where", [](const Tensor& a, const Tensor& b, const Element& c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "Where", [](const Tensor& a, const Element& b, const Tensor& c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def(
        "Where", [](const Tensor& a, const Element& b, const Element& c) { return npu::tile_fwk::Where(a, b, c); },
        "Tensor where.");
    m.def("Assign", [](const Tensor& operand) { return npu::tile_fwk::Assign(operand); }, "Tensor assign.");
    m.def(
        "Cat", [](const std::vector<Tensor>& tensors, int axis) { return npu::tile_fwk::Cat(tensors, axis); },
        "Tensor concat.");
    m.def(
        "Interleave",
        [](const Tensor& self, const Tensor& other) {
            auto result = npu::tile_fwk::Interleave(self, other);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        },
        py::arg("self"), py::arg("other"), "Tensor interleave.");
    m.def(
        "DeInterleave",
        [](const Tensor& self, const Tensor& other) {
            auto result = npu::tile_fwk::DeInterleave(self, other);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        },
        py::arg("self"), py::arg("other"), "Tensor deinterleave.");
    m.def(
        "DeInterleave",
        [](const Tensor& self) {
            auto result = npu::tile_fwk::DeInterleave(self);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        },
        py::arg("self"), "Tensor deinterleave.");
    m.def("cumsum", [](const Tensor& input, int axis) { return npu::tile_fwk::CumSum(input, axis); }, "Tensor cumsum.");
    m.def(
        "cumprod", [](const Tensor& input, int axis) { return npu::tile_fwk::CumProd(input, axis); },
        "Tensor cumprod.");
    m.def(
        "TriU",
        [](const Tensor& input, const SymbolicScalar& diagonal) { return npu::tile_fwk::TriU(input, diagonal); },
        "Tensor triu.");
    m.def(
        "TriL",
        [](const Tensor& input, const SymbolicScalar& diagonal) { return npu::tile_fwk::TriL(input, diagonal); },
        "Tensor tril.");
    m.def(
        "TopK",
        [](const Tensor& self, int k, int axis, bool islargest, TopKAlgo algo) {
            return npu::tile_fwk::TopK(self, k, axis, islargest, algo);
        },
        py::arg("operand"), py::arg("k"), py::arg("axis"), py::arg("islargest") = true,
        py::arg("algo") = TopKAlgo::MERGE_SORT, "Tensor topk.");
    m.def(
        "QuantMX",
        [](const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode, int64_t axis,
           bool performanceMode) { return npu::tile_fwk::QuantMX(input, quantDtype, mode, axis, performanceMode); },
        py::arg("input"), py::arg("quantDtype") = DataType::DT_FP8E4M3,
        py::arg("mode") = DequantScaleRoundingMode::ROUND_DOWN, py::arg("axis") = -1, py::arg("performanceMode") = true,
        "Tensor MX quant.");
    m.def(
        "Sort32", [](const Tensor& self, int index) { return npu::tile_fwk::Sort32(self, index); }, py::arg("operand"),
        py::arg("index"), "Tensor sort32.");
    m.def(
        "MrgSort", [](const Tensor& self, int mergesize) { return npu::tile_fwk::MrgSort(self, mergesize); },
        py::arg("operand"), py::arg("mergesize"), "Tensor mrgsort.");
    m.def(
        "Var",
        [](const Tensor& input, const std::vector<int>& dim, float correction, bool keepDim) {
            return npu::tile_fwk::Var(input, dim, correction, keepDim);
        },
        py::arg("input"), py::arg("dim") = std::vector<int>{}, py::arg("correction") = 1.0f, py::arg("keepDim") = false,
        "Tensor Var.");
    m.def(
        "ArgSort",
        [](const Tensor& self, int axis, bool descending) { return npu::tile_fwk::ArgSort(self, axis, descending); },
        py::arg("operand"), py::arg("axis"), py::arg("descending") = false, "Tensor argsort.");
    m.def(
        "ArgMax",
        [](const Tensor& operand, int axis, bool keepDim) { return npu::tile_fwk::ArgMax(operand, axis, keepDim); },
        py::arg("operand"), py::arg("axis") = -1, py::arg("keepDim") = false, "Tensor argmax.");
    m.def(
        "ArgMin",
        [](const Tensor& operand, int axis, bool keepDim) { return npu::tile_fwk::ArgMin(operand, axis, keepDim); },
        py::arg("operand"), py::arg("axis") = -1, py::arg("keepDim") = false, "Tensor argmin.");
    m.def(
        "Matmul",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_b, bool a_trans, bool b_trans,
           bool c_matrix_nz) { return Matrix::Matmul(out_type, tensor_a, tensor_b, a_trans, b_trans, c_matrix_nz); },
        py::arg("out_type"), py::arg("tensor_a"), py::arg("tensor_b"), py::arg("a_trans") = false,
        py::arg("b_trans") = false, py::arg("c_matrix_nz") = false, "Matrix multiply.");
    m.def(
        "MatmulMX",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_a_scale, const Tensor& tensor_b,
           const Tensor& tensor_b_scale, bool a_trans, bool a_scale_trans, bool b_trans, bool b_scale_trans,
           bool c_matrix_nz) {
            return Matrix::MatmulMX(out_type, tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, a_trans,
                                    a_scale_trans, b_trans, b_scale_trans, c_matrix_nz);
        },
        py::arg("out_type"), py::arg("tensor_a"), py::arg("tensor_a_scale"), py::arg("tensor_b"),
        py::arg("tensor_b_scale"), py::arg("a_trans") = false, py::arg("a_scale_trans") = false,
        py::arg("b_trans") = false, py::arg("b_scale_trans") = false, py::arg("c_matrix_nz") = false,
        "MatrixMX multiply.");

    m.def(
        "BatchMatmulMX",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_a_scale, const Tensor& tensor_b,
           const Tensor& tensor_b_scale, bool a_trans, bool a_scale_trans, bool b_trans, bool b_scale_trans,
           bool c_matrix_nz) {
            return Matrix::BatchMatmulMX(out_type, tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, a_trans,
                                         a_scale_trans, b_trans, b_scale_trans, c_matrix_nz);
        },
        py::arg("out_type"), py::arg("tensor_a"), py::arg("tensor_a_scale"), py::arg("tensor_b"),
        py::arg("tensor_b_scale"), py::arg("a_trans") = false, py::arg("a_scale_trans") = false,
        py::arg("b_trans") = false, py::arg("b_scale_trans") = false, py::arg("c_matrix_nz") = false,
        "Batch MatrixMX multiply.");

    py::class_<Matrix::MatmulExtendParam>(m, "MatmulExtendParam")
        .def(py::init<>())
        .def(py::init<Tensor, Tensor, float, Matrix::ReLuType, Matrix::TransMode>(), py::arg("bias_tensor"),
             py::arg("scale_tensor"), py::arg("scale"), py::arg("relu_type"), py::arg("trans_mode"),
             "Matrix extend params.");

    m.def(
        "Matmul",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_b, bool a_trans, bool b_trans,
           bool c_matrix_nz, const Matrix::MatmulExtendParam& extendParam) {
            return Matrix::Matmul(out_type, tensor_a, tensor_b, extendParam, a_trans, b_trans, c_matrix_nz);
        },
        py::arg("out_type"), py::arg("tensor_a"), py::arg("tensor_b"), py::arg("a_trans") = false,
        py::arg("b_trans") = false, py::arg("c_matrix_nz") = false, py::arg("extend_params"),
        "Matrix multiply with extend param.");
    py::class_<Conv::ConvExtendParam>(m, "ConvExtendParam")
        .def(py::init<>())
        .def(py::init<Tensor, Tensor, float, Conv::ReLuType>(), py::arg("bias_tensor"), py::arg("scale_tensor"),
             py::arg("scale"), py::arg("relu_type"), "Conv extend params.");
    m.def(
        "MatmulMX",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_a_scale, const Tensor& tensor_b,
           const Tensor& tensor_b_scale, bool a_trans, bool a_scale_trans, bool b_trans, bool b_scale_trans,
           bool c_matrix_nz, const Matrix::MatmulExtendParam& extendParam) {
            return Matrix::MatmulMX(out_type, tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, extendParam, a_trans,
                                    a_scale_trans, b_trans, b_scale_trans, c_matrix_nz);
        },
        py::arg("out_type"), py::arg("tensor_a"), py::arg("tensor_a_scale"), py::arg("tensor_b"),
        py::arg("tensor_b_scale"), py::arg("a_trans") = false, py::arg("a_scale_trans") = false,
        py::arg("b_trans") = false, py::arg("b_scale_trans") = false, py::arg("c_matrix_nz") = false,
        py::arg("extend_params"), "MatrixMX multiply with extend param.");
    m.def(
        "BatchMatmulMX",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_a_scale, const Tensor& tensor_b,
           const Tensor& tensor_b_scale, bool a_trans, bool a_scale_trans, bool b_trans, bool b_scale_trans,
           bool c_matrix_nz, const Matrix::MatmulExtendParam& extendParam) {
            return Matrix::BatchMatmulMX(out_type, tensor_a, tensor_a_scale, tensor_b, tensor_b_scale, extendParam,
                                         a_trans, a_scale_trans, b_trans, b_scale_trans, c_matrix_nz);
        },
        py::arg("out_type"), py::arg("tensor_a"), py::arg("tensor_a_scale"), py::arg("tensor_b"),
        py::arg("tensor_b_scale"), py::arg("a_trans") = false, py::arg("a_scale_trans") = false,
        py::arg("b_trans") = false, py::arg("b_scale_trans") = false, py::arg("c_matrix_nz") = false,
        py::arg("extend_params"), "Batch MatrixMX multiply with extend param.");
    m.def(
        "Conv",
        [](DataType out_type, const Tensor& tensor_input, const Tensor& tensor_weight,
           const std::vector<int64_t>& strides, const std::vector<SymbolicScalar>& paddings,
           const std::vector<int64_t>& dilations, const Conv::ConvExtendParam& extendParam, const int64_t groups) {
            return Conv::Conv(out_type, tensor_input, tensor_weight, strides, paddings, dilations, extendParam, groups);
        },
        py::arg("out_type"), py::arg("tensor_input"), py::arg("tensor_weight"), py::arg("strides"), py::arg("paddings"),
        py::arg("dilations"), py::arg("extend_params"), py::arg("groups") = 1,
        "Convolution forward with extend param.");
    m.def(
        "BatchMatmul",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_b, bool a_trans, bool b_trans,
           bool c_matrix_nz) {
            return Matrix::BatchMatmul(out_type, tensor_a, tensor_b, a_trans, b_trans, c_matrix_nz);
        },
        py::arg("out_type"), py::arg("a"), py::arg("b"), py::arg("a_trans") = false, py::arg("b_trans") = false,
        py::arg("c_matrix_nz") = false, "Batch matrix multiply.");
    m.def(
        "BatchMatmul",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_b, bool a_trans, bool b_trans,
           bool c_matrix_nz, const Matrix::MatmulExtendParam& extend_param) {
            return Matrix::BatchMatmul(out_type, tensor_a, tensor_b, extend_param, a_trans, b_trans, c_matrix_nz);
        },
        py::arg("out_type"), py::arg("a"), py::arg("b"), py::arg("a_trans") = false, py::arg("b_trans") = false,
        py::arg("c_matrix_nz") = false, py::arg("extend_params"), "Batch matrix multiply with extend params.");
    m.def(
        "gather_in_l1",
        [](const Tensor& src, const Tensor& indices, const Tensor& blockTable, int blockSize, int size,
           bool is_b_matrix, bool is_trans) {
            if (!is_b_matrix && !is_trans) {
                std::cout << " gather in l1 m def" << std::endl;
                return experimental::GatherInL1<false, false>(src, indices, blockTable, blockSize, size);
            } else if (!is_b_matrix && is_trans) {
                return experimental::GatherInL1<false, true>(src, indices, blockTable, blockSize, size);
            } else if (is_b_matrix && !is_trans) {
                return experimental::GatherInL1<true, false>(src, indices, blockTable, blockSize, size);
            } else {
                return experimental::GatherInL1<true, true>(src, indices, blockTable, blockSize, size);
            }
        },
        py::arg("src"), py::arg("indices"), py::arg("blockTable"), py::arg("blockSize"), py::arg("size"),
        py::arg("is_b_matrix"), py::arg("is_trans"), "gather load L1.");
    m.def(
        "gather_in_ub",
        [](const Tensor& param, const Tensor& indices, const Tensor& blockTable, int blockSize, int axis) {
            return experimental::GatherInUB(param, indices, blockTable, blockSize, axis);
        },
        py::arg("param"), py::arg("indices"), py::arg("blockTable"), py::arg("blockSize"), py::arg("axis"),
        "Tensor gather_in_ub");
    m.def(
        "TransposedBatchMatmul",
        [](DataType out_type, const Tensor& tensor_a, const Tensor& tensor_b) {
            return Matrix::TransposedBatchMatmul(out_type, tensor_a, tensor_b);
        },
        py::arg("out_type"), py::arg("a"), py::arg("b"), "Transposed batch matrix multiply.");
    m.def(
        "ScalarDivS",
        [](const Tensor& operand, const Element& value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarDivS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar divs.");
    m.def(
        "ScalarAddS",
        [](const Tensor& operand, const Element& value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarAddS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar adds.");
    m.def(
        "ScalarMaxS",
        [](const Tensor& operand, const Element& value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarMaxS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar maxs.");
    m.def(
        "ScalarSubS",
        [](const Tensor& operand, const Element& value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarSubS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar subs.");
    m.def(
        "ScalarMulS",
        [](const Tensor& operand, const Element& value, bool reverse_operand = false) {
            return npu::tile_fwk::ScalarMulS(operand, value, reverse_operand);
        },
        py::arg("operand"), py::arg("value"), py::arg("reverse_operand") = false, "Tensor scalar muls.");
    m.def(
        "ScalarSub",
        [](const Tensor& operand1, const Tensor& operand2) { return npu::tile_fwk::ScalarSub(operand1, operand2); },
        py::arg("operand1"), py::arg("operand2"), "Tensor scalar sub.");
    m.def(
        "ScalarDiv",
        [](const Tensor& operand1, const Tensor& operand2) { return npu::tile_fwk::ScalarDiv(operand1, operand2); },
        py::arg("operand1"), py::arg("operand2"), "Tensor scalar div.");
    m.def(
        "Maxpool",
        [](const Tensor& operand, const std::vector<int>& pools, const std::vector<int>& stride,
           const std::vector<int>& paddings) { return npu::tile_fwk::Maxpool(operand, pools, stride, paddings); },
        py::arg("operand"), py::arg("pools"), py::arg("stride"), py::arg("paddings"), "Max pool.");
    m.def(
        "Compare",
        [](const Tensor& self, const Tensor& other, OpType op, OutType mode) {
            return npu::tile_fwk::Compare(self, other, op, mode);
        },
        py::arg("operand1"), py::arg("operand2"), py::arg("operation"), py::arg("mode"), "Tensor compare.");
    m.def(
        "Compare",
        [](const Tensor& self, const Element& other, OpType op, OutType mode) {
            return npu::tile_fwk::Compare(self, other, op, mode);
        },
        py::arg("operand"), py::arg("scalar"), py::arg("operation"), py::arg("mode"), "Tensor compare.");
    m.def(
        "Compare",
        [](const Element& self, const Tensor& other, OpType op, OutType mode) {
            return npu::tile_fwk::Compare(self, other, op, mode);
        },
        py::arg("scalar"), py::arg("operand"), py::arg("operation"), py::arg("mode"), "Tensor compare.");
    m.def(
        "Assemble",
        [](const std::vector<std::pair<Tensor, std::vector<SymbolicScalar>>>& inputs, Tensor& dest,
           bool parallel = false) {
            std::vector<npu::tile_fwk::AssembleItem> items;
            for (const auto& [tensor, offset] : inputs) {
                items.push_back({tensor, offset});
            }
            npu::tile_fwk::Assemble(items, dest, parallel);
        },
        "Tensor assemble");
    m.def(
        "Assemble",
        [](const Tensor& tensor, const std::vector<SymbolicScalar>& dynOffset, Tensor& dest, bool parallel = false) {
            npu::tile_fwk::Assemble(tensor, dynOffset, dest, parallel);
        },
        py::arg("tensor"), py::arg("dynOffset"), py::arg("dest"), py::arg("parallel") = false, "Tensor dassemble");
    m.def(
        "AtomicRMW",
        [](const Tensor& src, const std::vector<SymbolicScalar>& offsets, Tensor& dst, AtomicRMWMode mode) {
            npu::tile_fwk::AtomicRMW(src, offsets, dst, mode);
        },
        py::arg("src"), py::arg("offsets"), py::arg("dst"), py::arg("mode"), "Tensor atomic rmw");
    m.def("Maximum",
          [](const Tensor& operand1, const Element& operand2) { return npu::tile_fwk::Maximum(operand1, operand2); });
    m.def(
        "OnlineSoftmax",
        [](const Tensor& scores, float scale) {
            auto result = npu::tile_fwk::OnlineSoftmax(scores, scale);
            return py::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result));
        },
        py::arg("scores"), py::arg("scale"),
        "OnlineSoftmax(scores, scale) -> (exp_scores_bf16, column_max, column_sum).");
    m.def(
        "OnlineSoftmaxUpdate",
        [](const Tensor& previousMax, const Tensor& previousSum, const Tensor& previousOutput, const Tensor& currentMax,
           const Tensor& currentSum, const Tensor& currentOutput) {
            auto result = npu::tile_fwk::OnlineSoftmaxUpdate(previousMax, previousSum, previousOutput, currentMax,
                                                             currentSum, currentOutput);
            return py::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result));
        },
        py::arg("previous_max"), py::arg("previous_sum"), py::arg("previous_output"), py::arg("current_max"),
        py::arg("current_sum"), py::arg("current_output"),
        "OnlineSoftmaxUpdate(previous_max, previous_sum, previous_output, current_max, current_sum, current_output) "
        "-> (updated_max, updated_sum, updated_output).");
    m.def("Minimum",
          [](const Tensor& operand1, const Element& operand2) { return npu::tile_fwk::Minimum(operand1, operand2); });
    m.def("Clip",
          [](const Tensor& self, const Tensor& min, const Tensor& max) { return npu::tile_fwk::Clip(self, min, max); });
    m.def("Clip", [](const Tensor& self, const Element& min, const Element& max) {
        return npu::tile_fwk::Clip(self, min, max);
    });
    m.def("CeilDiv", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::CeilDiv(self, other); });
    m.def("CeilDiv", [](const Tensor& self, const Element& other) { return npu::tile_fwk::CeilDiv(self, other); });
    m.def(
        "OneHot", [](const Tensor& self, int numClasses) { return npu::tile_fwk::OneHot(self, numClasses); },
        "Tensor one hot.");
    m.def("PrintIf", [](SymbolicScalar cond, const std::string& format, const std::vector<Tensor>& tensors,
                        const std::vector<SymbolicScalar>& scalars) {
        npu::tile_fwk::experimental::Print(cond, format, tensors, scalars);
    });
    m.def("ToFile", [](const Tensor& operand, const std::string& fname, const std::vector<SymbolicScalar>& scalars,
                       SymbolicScalar cond) { npu::tile_fwk::ToFile(operand, fname, scalars, cond); });
    m.def(
        "topk_sort",
        [](const Tensor& x, int idx_start) {
            auto result = npu::tile_fwk::TopKSort(x, idx_start);
            // return as a Python tuple (y, temp)
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        },
        py::arg("x"), py::arg("idx_start"),
        "TopKSort(x, idx_start:int) -> (y, temp)\n"
        "Performs tiled Top-K sorting starting at a scalar index.\n"
        "Returns a tuple of (sorted_values, workspace_temp).");

    m.def(
        "topk_sort",
        [](const Tensor& x, const SymbolicScalar& idx_start) {
            auto result = npu::tile_fwk::TopKSort(x, idx_start);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        },
        py::arg("x"), py::arg("idx_start"),
        "TopKSort(x, idx_start:SymbolicScalar) -> (y, temp)\n"
        "Performs tiled Top-K sorting with a symbolic starting index.\n"
        "Returns a tuple of (sorted_values, workspace_temp).");

    m.def(
        "topk_merge", [](const Tensor& x, int merge_size) { return npu::tile_fwk::TopKMerge(x, merge_size); },
        py::arg("x"), py::arg("merge_size"),
        "TopKMerge(x, merge_size:int) -> y\n"
        "Merges partial Top-K results into a single tensor.");

    m.def(
        "topk_extract",
        [](const Tensor& x, int k, bool is_index) { return npu::tile_fwk::TopKExtract(x, k, is_index); }, py::arg("x"),
        py::arg("k"), py::arg("is_index") = false,
        "TopKExtract(x, k:int, is_index:bool=False) -> y\n"
        "Extracts the top-k values (or indices if is_index=True).");

    m.def(
        "CopySign", [](const Tensor& self, const Tensor& other) { return npu::tile_fwk::CopySign(self, other); },
        "Tensor copysign.");

    m.def(
        "isfinite", [](const Tensor& self) { return npu::tile_fwk::IsFinite(self); },
        "Judge whether the value is inf/nan/-inf. If it is, the value will be false.");

    m.def(
        "isnan", [](const Tensor& self) { return npu::tile_fwk::IsNan(self); },
        "Judge whether the value is nan. If it is, the value will be true, otherwise false.");

    m.def(
        "Nop", [](const std::vector<Tensor>& inTensors) { return npu::tile_fwk::Nop(inTensors); },
        py::arg("in_tensors"));
    m.def(
        "PReLU", [](const Tensor& self, const Tensor& weight) { return npu::tile_fwk::PReLU(self, weight); },
        "Tensor prelu.");
}
} // namespace pypto
