/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file block_ops/struct_ops.cpp
 * \brief Struct array operations for clean C++ codegen.
 *
 *  - struct.create: Expression op that returns a named TupleType. Lowered by
 *    the CCE backend to ``Name var = {.f0=v0, ...};``.
 *  - struct.create_array: Expression op that returns TupleType(N × named tuple).
 *    Lowered to ``Name var[N] = {};`` plus per-slot field-init lines.
 */

#include <any>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/identifier.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

REGISTER_OP("struct.create")
    .set_op_category("StructOp")
    .set_description("Materialize a named tuple into a C++ struct. args are field values; kwargs `name` "
                     "is the C++ struct type name and `fields` is the list of field names. Result type is "
                     "TupleType(field_types, dbgName=fields). Must be let-bound to a Var by the producer.")
    .add_argument("...", "Field value expressions in declaration order")
    .set_attr<std::string>("name")
    .set_attr<std::vector<std::string>>("fields")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        std::string struct_name;
        std::vector<std::string> fields;
        bool has_name = false;
        bool has_fields = false;
        for (const auto& [key, value] : kwargs) {
            if (key == "name") {
                struct_name = std::any_cast<std::string>(value);
                has_name = true;
            } else if (key == "fields") {
                fields = std::any_cast<std::vector<std::string>>(value);
                has_fields = true;
            }
        }
        CHECK(has_name) << "struct.create requires kwarg 'name' (C++ struct type name)";
        CHECK(has_fields) << "struct.create requires kwarg 'fields' (list of field names)";
        CHECK(IsValidIdentifier(struct_name))
            << "struct.create 'name' must be a valid C identifier, got '" << struct_name << "'";
        CHECK(fields.size() == args.size())
            << "struct.create: fields count (" << fields.size() << ") must match args count (" << args.size() << ")";
        std::set<std::string> seen;
        for (const auto& f : fields) {
            CHECK(IsValidIdentifier(f)) << "struct.create field name '" << f << "' is not a valid identifier";
            CHECK(seen.insert(f).second) << "struct.create field name '" << f << "' is duplicated";
        }
        std::vector<TypePtr> field_types;
        field_types.reserve(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
            CHECK(args[i]) << "struct.create arg #" << i << " is null";
            field_types.push_back(args[i]->GetType());
        }
        // Positional TupleType: field names live in IRDebugInfo (registered by the
        // parser from the op's 'fields' kwarg), not on the type.
        (void)fields;
        return std::make_shared<TupleType>(std::move(field_types));
    });

REGISTER_OP("struct.set")
    .set_op_category("StructOp")
    .set_description("Statement side-effect call: write one struct field. args are (base, value) for a scalar "
                     "field or (base, index, value) for an array field element; kwarg `field` is the C++ field "
                     "name (same naming convention as struct.create's `fields`). Lowered by the CCE backend to "
                     "`base.field = value;` or `base.field[index] = value;`. Must be used inside an EvalStmt, "
                     "not as an RHS value.")
    .add_argument("base", "Struct instance (named TupleType)")
    .add_argument("value", "New field value expression (or index for array field writes)")
    .set_attr<std::string>("field")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        CHECK(args.size() == 2 || args.size() == 3)
            << "struct.set requires 2 args (base, value) or 3 args (base, index, value), got " << args.size();
        CHECK(args[0]) << "struct.set: base arg is null";
        if (args.size() == 3) {
            CHECK(args[1]) << "struct.set: index arg is null";
        }
        CHECK(args.back()) << "struct.set: value arg is null";
        bool has_field = false;
        for (const auto& [key, value] : kwargs) {
            if (key == "field") {
                CHECK(IsValidIdentifier(std::any_cast<std::string>(value)))
                    << "struct.set 'field' must be a valid identifier";
                has_field = true;
            }
        }
        CHECK(has_field) << "struct.set requires kwarg 'field' (the C++ field name to write)";
        CHECK(args[0]->GetType()->GetKind() == ObjectKind::TupleType) << "struct.set: base must have TupleType";
        // Statement side effect; the discarded result type mirrors the struct being written.
        return args[0]->GetType();
    });

} // namespace ir
} // namespace pypto
