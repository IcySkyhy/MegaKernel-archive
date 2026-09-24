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
 * \file symbolic_scalar.cpp
 * \brief
 */

#include "pybind_common.h"

#include "ir/kind_traits.h"

using namespace npu::tile_fwk;

namespace pypto {

#define DEFINE_BINARY_OP(name, bop)                                \
    .def(name, [](const SymbolicScalar& self, py::object& other) { \
        if (py::isinstance<py::int_>(other)) {                     \
            auto immediate = other.cast<int64_t>();                \
            return self bop immediate;                             \
        } else if (py::isinstance<SymbolicScalar>(other)) {        \
            return self bop other.cast<SymbolicScalar>();          \
        }                                                          \
        throw py::type_error("Invalid type.");                     \
    })

#define DEFINE_BINARY_OP_CHECKED(name, bop)                        \
    .def(name, [](const SymbolicScalar& self, py::object& other) { \
        if (py::isinstance<py::int_>(other)) {                     \
            auto immediate = other.cast<int64_t>();                \
            if (immediate == 0) {                                  \
                throw py::value_error("Division or Mod by zero."); \
            }                                                      \
            return self bop immediate;                             \
        } else if (py::isinstance<SymbolicScalar>(other)) {        \
            return self bop other.cast<SymbolicScalar>();          \
        }                                                          \
        throw py::type_error("Invalid type.");                     \
    })

#define DEFINE_RBINARY_OP(name, bop)                               \
    .def(name, [](const SymbolicScalar& self, py::object& other) { \
        if (py::isinstance<py::int_>(other)) {                     \
            auto immediate = other.cast<int64_t>();                \
            return immediate bop self;                             \
        } else if (py::isinstance<SymbolicScalar>(other)) {        \
            return other.cast<SymbolicScalar>() bop self;          \
        }                                                          \
        throw py::type_error("Invalid type.");                     \
    })

#define DEFINE_UNARY_OP(name, uop) .def(name, [](const SymbolicScalar& self) { return uop self; })

void BindSymbolicScalar(py::module_& m)
{
    py::enum_<SatStatus>(m, "SatStatus")
        .value("SAT", SatStatus::kSat)
        .value("UNSAT", SatStatus::kUnsat)
        .value("UNKNOWN", SatStatus::kUnknown)
        .export_values();

    py::class_<SymbolicScalar>(m, "SymbolicScalar", py::dynamic_attr())
        .def(py::init([](int64_t value) { return SymbolicScalar(value); }), py::arg("value"),
             "Create SymbolicScalar from integer value")
        .def(py::init([](std::string name) { return SymbolicScalar(name); }), py::arg("name"),
             "Create SymbolicScalar from symbol name")
        .def(py::init([](std::string name, int64_t value) { return SymbolicScalar(name, value); }), py::arg("name"),
             py::arg("value"), "Create SymbolicScalar from symbol name and integer value")
        .def(py::init([](ir::ExprPtr expr) { return SymbolicScalar::FromExpr(expr); }), py::arg("expr"),
             "Create SymbolicScalar from expression")
        .def("__str__", &SymbolicScalar::Dump)
        .def("__repr__", &SymbolicScalar::Dump)
        // clang-format off
        DEFINE_BINARY_OP("__eq__", ==)
        DEFINE_BINARY_OP("__ne__", !=)
        DEFINE_BINARY_OP("__lt__", <)
        DEFINE_BINARY_OP("__le__", <=)
        DEFINE_BINARY_OP("__gt__", >)
        DEFINE_BINARY_OP("__ge__", >=)
        DEFINE_BINARY_OP("__and__", &&)
        DEFINE_RBINARY_OP("__rand__", &&)
        DEFINE_BINARY_OP("__or__", ||)
        DEFINE_RBINARY_OP("__ror__", ||)
        DEFINE_BINARY_OP("__add__", +)
        DEFINE_RBINARY_OP("__radd__", +)
        DEFINE_BINARY_OP("__sub__", -)
        DEFINE_RBINARY_OP("__rsub__", -)
        DEFINE_BINARY_OP("__mul__", *)
        DEFINE_RBINARY_OP("__rmul__", *)
        DEFINE_BINARY_OP_CHECKED("__truediv__", /)
        DEFINE_RBINARY_OP("__rtruediv__", /)
        DEFINE_BINARY_OP_CHECKED("__mod__", %)
        DEFINE_RBINARY_OP("__rmod__", %)
        DEFINE_BINARY_OP_CHECKED("__floordiv__", /)
        DEFINE_RBINARY_OP("__rfloordiv__", /)
        DEFINE_UNARY_OP("__pos__", +)
        DEFINE_UNARY_OP("__neg__", -)
        DEFINE_UNARY_OP("__invert__", !)
        // clang-format on
        .def("__bool__",
             [](const SymbolicScalar& self) {
                 if (self.ConcreteValid()) {
                     return self.Concrete() != 0;
                 }
                 throw py::value_error("Not concrete value.");
             })
        .def("__int__",
             [](const SymbolicScalar& self) {
                 if (self.ConcreteValid()) {
                     return self.Concrete();
                 }
                 throw py::value_error("Not concrete value.");
             })
        .def("is_concrete", &SymbolicScalar::ConcreteValid)
        .def("is_symbol", &SymbolicScalar::IsSymbol)
        .def("is_expression", &SymbolicScalar::IsExpression)
        .def("is_immediate", &SymbolicScalar::IsImmediate)
        .def("simplify", &SymbolicScalar::Simplify)
        .def("min",
             [](const SymbolicScalar& self, py::object& other) {
                 if (py::isinstance<py::int_>(other)) {
                     return self.Min(other.cast<int64_t>());
                 } else if (py::isinstance<SymbolicScalar>(other)) {
                     return self.Min(other.cast<SymbolicScalar>());
                 }
                 throw py::type_error("Invalid type.");
             })
        .def("max",
             [](const SymbolicScalar& self, py::object& other) {
                 if (py::isinstance<py::int_>(other)) {
                     return self.Max(other.cast<int64_t>());
                 } else if (py::isinstance<SymbolicScalar>(other)) {
                     return self.Max(other.cast<SymbolicScalar>());
                 }
                 throw py::type_error("Invalid type.");
             })
        .def("concrete", py::overload_cast<>(&SymbolicScalar::Concrete, py::const_))
        .def("as_variable", &SymbolicScalar::AsIntermediateVariable)
        .def_static("tenary", [](const SymbolicScalar& cond, const SymbolicScalar& true_val,
                                 const SymbolicScalar& false_val) { return std::ternary(cond, true_val, false_val); })
        .def("as_expr", &SymbolicScalar::AsExpr)
        .def("as_var", &SymbolicScalar::AsVar)
        .def_static("check", &SymbolicScalar::Check, py::arg("conds"),
                    "Satisfiability of the conjunction of conditions: SAT / UNSAT / UNKNOWN");

    py::implicitly_convertible<int64_t, SymbolicScalar>();
    py::implicitly_convertible<int, SymbolicScalar>();
}
} // namespace pypto
