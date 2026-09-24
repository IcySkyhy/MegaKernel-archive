/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "bindings.h"
#include "interface/tensor/irbuilder.h"
#include "interface/function/function.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/transforms/structural_comparison.h"

namespace npu::tile_fwk {

void BindIRBuilder(py::module_& m)
{
    // Get or create ir submodule
    py::module_ ir = m.attr("ir");
    ir.def(
        "set_assemble_new_logical_tensor", [](bool enabled) { IRContext::Get().SetAssembleNewLogicalTensor(enabled); },
        py::arg("enabled"));
    ir.def(
        "assemble_new_logical_tensor", []() { return IRContext::Get().AssembleNewLogicalTensor(); },
        "Get whether Assemble creates a new LogicalTensor version.\n\n"
        "Returns:\n"
        "    bool: True if Assemble creates a new LogicalTensor version");

    // IRBuilder class
    py::class_<IRBuilder>(ir, "IRBuilder",
                          "IR Builder for incremental IR construction with context management.\n\n"
                          "The IRBuilder provides a stateful API for building IR incrementally using\n"
                          "Begin/End patterns. It maintains a context stack to track nested scopes\n"
                          "and validates proper construction.")
        .def(py::init<>(), "Create a new IR builder")

        // Function building
        .def(
            "begin_function",
            [](IRBuilder& self, const std::string& name, const ir::Span& span, ir::FunctionType type) {
                self.BeginFunction(name, span, type);
            },
            py::arg("name"), py::arg("span"), py::arg("type") = ir::FunctionType::OPAQUE,
            "Begin building a function.\n\n"
            "Creates a new function context. Must be closed with end_function().\n\n"
            "Args:\n"
            "    name: Function name\n"
            "    span: Source location for function definition\n"
            "    type: Function type (default: Opaque)\n\n"
            "Raises:\n"
            "    RuntimeError: If already inside a function (nested functions not allowed)")

        .def("func_arg", &IRBuilder::FuncArg, py::arg("name"), py::arg("type"), py::arg("span"),
             "Add a function parameter.\n\n"
             "Must be called within a function context.\n\n"
             "Args:\n"
             "    name: Parameter name\n"
             "    type: Parameter type\n"
             "    span: Source location for parameter\n\n"
             "Returns:\n"
             "    Var: Variable representing the parameter\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a function context")

        .def("return_type", &IRBuilder::ReturnType, py::arg("type"),
             "Add a return type to the current function.\n\n"
             "Can be called multiple times for multiple return types.\n\n"
             "Args:\n"
             "    type: Return type\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a function context")

        .def("end_function", &IRBuilder::EndFunction, py::arg("end_span"),
             "End building a function.\n\n"
             "Finalizes the function and returns it.\n\n"
             "Args:\n"
             "    end_span: Source location for end of function\n\n"
             "Returns:\n"
             "    Function: The built function\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a function context")

        // For loop building
        .def(
            "begin_for_loop",
            [](IRBuilder& self, const ir::VarPtr& loop_var, const ir::ExprPtr& start, const ir::ExprPtr& stop,
               const ir::ExprPtr& step, const ir::Span& span) { self.BeginForLoop(loop_var, start, stop, step, span); },
            py::arg("loop_var"), py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("span"),
            "Begin building a for loop.\n\n"
            "Creates a new for loop context. Must be closed with end_for_loop().\n\n"
            "Args:\n"
            "    loop_var: Loop variable\n"
            "    start: Start value expression\n"
            "    stop: Stop value expression\n"
            "    step: Step value expression\n"
            "    span: Source location for loop definition\n\n"
            "Raises:\n"
            "    RuntimeError: If not inside a valid context")

        .def("add_iter_arg", &IRBuilder::AddIterArg, py::arg("iter_arg"),
             "Add an iteration argument to the current for loop.\n\n"
             "Iteration arguments are loop-carried values (SSA-style).\n\n"
             "Args:\n"
             "    iter_arg: Iteration argument with initial value\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a for loop context")

        .def("add_return_var", &IRBuilder::AddReturnVar, py::arg("var"),
             "Add a return variable to the current for loop.\n\n"
             "Return variables capture the final values of iteration arguments.\n"
             "Must match the number of iteration arguments.\n\n"
             "Args:\n"
             "    var: Return variable\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a for loop context")

        .def("end_for_loop", &IRBuilder::EndForLoop, py::arg("end_span"),
             "End building a for loop.\n\n"
             "Finalizes the loop and returns it.\n\n"
             "Args:\n"
             "    end_span: Source location for end of loop\n\n"
             "Returns:\n"
             "    ForStmt: The built for statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a for loop context\n"
             "    RuntimeError: If number of return variables doesn't match iteration arguments")

        // While loop building
        .def("begin_while_loop", &IRBuilder::BeginWhileLoop, py::arg("condition"), py::arg("span"),
             "Begin building a while loop.\n\n"
             "Creates a new while loop context. Must be closed with end_while_loop().\n\n"
             "Args:\n"
             "    condition: Condition expression\n"
             "    span: Source location for loop definition\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        .def("add_while_iter_arg", &IRBuilder::AddWhileIterArg, py::arg("iter_arg"),
             "Add an iteration argument to the current while loop.\n\n"
             "Iteration arguments are loop-carried values (SSA-style).\n\n"
             "Args:\n"
             "    iter_arg: Iteration argument with initial value\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a while loop context")

        .def("add_while_return_var", &IRBuilder::AddWhileReturnVar, py::arg("var"),
             "Add a return variable to the current while loop.\n\n"
             "Return variables capture the final values of iteration arguments.\n\n"
             "Args:\n"
             "    var: Return variable\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a while loop context")

        .def("set_while_loop_condition", &IRBuilder::SetWhileLoopCondition, py::arg("condition"),
             "Set the condition for the current while loop.\n\n"
             "Used to update the loop condition after setting up iter_args. This allows\n"
             "the condition to reference iter_arg variables that are defined in the loop.\n\n"
             "Args:\n"
             "    condition: New condition expression\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a while loop context")

        .def("end_while_loop", &IRBuilder::EndWhileLoop, py::arg("end_span"),
             "End building a while loop.\n\n"
             "Finalizes the loop and returns it.\n\n"
             "Args:\n"
             "    end_span: Source location for end of loop\n\n"
             "Returns:\n"
             "    WhileStmt: The built while statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a while loop context\n"
             "    RuntimeError: If number of return variables doesn't match iteration arguments")

        // If statement building
        .def("begin_if", &IRBuilder::BeginIf, py::arg("condition"), py::arg("span"),
             "Begin building an if statement.\n\n"
             "Creates a new if context. Must be closed with end_if().\n\n"
             "Args:\n"
             "    condition: Condition expression\n"
             "    span: Source location for if statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        .def("begin_else", &IRBuilder::BeginElse, py::arg("span"),
             "Begin the else branch of the current if statement.\n\n"
             "Must be called after building the then branch.\n\n"
             "Args:\n"
             "    span: Source location for else keyword\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside an if context\n"
             "    RuntimeError: If else branch already begun")

        .def("add_if_return_var", &IRBuilder::AddIfReturnVar, py::arg("var"),
             "Add a return variable to the current if statement.\n\n"
             "Return variables are used for SSA phi nodes.\n\n"
             "Args:\n"
             "    var: Return variable\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside an if context")

        .def("end_if", &IRBuilder::EndIf, py::arg("end_span"),
             "End building an if statement.\n\n"
             "Finalizes the if statement and returns it.\n\n"
             "Args:\n"
             "    end_span: Source location for end of if\n\n"
             "Returns:\n"
             "    IfStmt: The built if statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside an if context")

        // Section building
        .def("begin_section", &IRBuilder::BeginSection, py::arg("section_kind"), py::arg("span"),
             "Begin building a section statement.\n\n"
             "Creates a new section context. Must be closed with end_section().\n\n"
             "Args:\n"
             "    section_kind: The kind of section (Vector or Cube)\n"
             "    span: Source location for section statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a function or loop")
        .def("end_section", &IRBuilder::EndSection, py::arg("end_span"),
             "End building a section statement.\n\n"
             "Finalizes the section statement and returns it.\n\n"
             "Args:\n"
             "    end_span: Source location for end of section\n\n"
             "Returns:\n"
             "    SectionStmt: The built section statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a section context")

        // Statement recording
        .def("emit", &IRBuilder::Emit, py::arg("stmt"),
             "Emit a statement in the current context.\n\n"
             "Adds a statement to the current context's statement list.\n\n"
             "Args:\n"
             "    stmt: Statement to emit\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        .def("assign", &IRBuilder::Assign, py::arg("var"), py::arg("value"), py::arg("span"),
             "Create an assignment statement and emit it.\n\n"
             "Convenience method that creates and emits an assignment.\n\n"
             "Args:\n"
             "    var: Variable to assign to\n"
             "    value: Expression value\n"
             "    span: Source location for assignment\n\n"
             "Returns:\n"
             "    AssignStmt: The created assignment statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        .def("var", &IRBuilder::Var, py::arg("name"), py::arg("type"), py::arg("span"),
             "Create a variable (does not emit).\n\n"
             "Helper to create a variable. User must create assignment separately.\n\n"
             "Args:\n"
             "    name: Variable name\n"
             "    type: Variable type\n"
             "    span: Source location\n\n"
             "Returns:\n"
             "    Var: The created variable")

        .def("return_", py::overload_cast<const std::vector<ir::ExprPtr>&, const ir::Span&>(&IRBuilder::Return),
             py::arg("values"), py::arg("span"),
             "Create a return statement and emit it.\n\n"
             "Convenience method that creates and emits a return statement.\n\n"
             "Args:\n"
             "    values: List of expressions to return\n"
             "    span: Source location for return statement\n\n"
             "Returns:\n"
             "    ReturnStmt: The created return statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        .def("return_", py::overload_cast<const ir::Span&>(&IRBuilder::Return), py::arg("span"),
             "Create an empty return statement and emit it.\n\n"
             "Convenience method that creates and emits an empty return statement.\n\n"
             "Args:\n"
             "    span: Source location for return statement\n\n"
             "Returns:\n"
             "    ReturnStmt: The created return statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        .def("break_", &IRBuilder::Break, py::arg("span"),
             "Create a break statement and emit it.\n\n"
             "Convenience method that creates and emits a break statement.\n\n"
             "Args:\n"
             "    span: Source location for break statement\n\n"
             "Returns:\n"
             "    BreakStmt: The created break statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        .def("continue_", &IRBuilder::Continue, py::arg("span"),
             "Create a continue statement and emit it.\n\n"
             "Convenience method that creates and emits a continue statement.\n\n"
             "Args:\n"
             "    span: Source location for continue statement\n\n"
             "Returns:\n"
             "    ContinueStmt: The created continue statement\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a valid context")

        // Context state queries
        .def("in_function", &IRBuilder::InFunction,
             "Check if currently inside a function.\n\n"
             "Returns:\n"
             "    bool: True if inside a function context")

        .def("in_loop", &IRBuilder::InLoop,
             "Check if currently inside a for loop.\n\n"
             "Returns:\n"
             "    bool: True if inside a for loop context")

        .def("in_if", &IRBuilder::InIf,
             "Check if currently inside an if statement.\n\n"
             "Returns:\n"
             "    bool: True if inside an if statement context")

        .def("in_program", &IRBuilder::InProgram,
             "Check if currently inside a program.\n\n"
             "Returns:\n"
             "    bool: True if inside a program context")

        // Program building
        .def("begin_program", &IRBuilder::BeginProgram, py::arg("name"), py::arg("span"),
             "Begin building a program.\n\n"
             "Creates a new program context. Must be closed with end_program().\n\n"
             "Args:\n"
             "    name: Program name\n"
             "    span: Source location for program definition\n\n"
             "Raises:\n"
             "    RuntimeError: If already inside another program")

        .def("add_function", &IRBuilder::AddFunction, py::arg("func"),
             "Add a completed function to the current program.\n\n"
             "Args:\n"
             "    func: Completed function to add\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a program context")

        .def("end_program", &IRBuilder::EndProgram, py::arg("end_span"),
             "End building a program.\n\n"
             "Finalizes the program and returns it.\n\n"
             "Args:\n"
             "    end_span: Source location for end of program\n\n"
             "Returns:\n"
             "    Program: The built program\n\n"
             "Raises:\n"
             "    RuntimeError: If not inside a program context")

        .def("get_function_return_types", &IRBuilder::GetFunctionReturnTypes, py::arg("func_name"),
             "Get return types for a function by name.\n\n"
             "Returns the return types for a function if it has been added to the program.\n"
             "Returns empty list if not inside a program or function not yet added.\n\n"
             "Args:\n"
             "    func_name: Function name\n\n"
             "Returns:\n"
             "    List[Type]: Vector of return types")

        .def(
            "create_tensor_var",
            [](IRBuilder& self, DataType t, std::vector<SymbolicScalar> shape, TileOpFormat format, std::string name) {
                auto staticShape = SymbolicScalar::Concrete(shape, -1);
                return self.CreateTensorVar(t, std::move(staticShape), std::move(shape), format, std::move(name));
            },
            py::arg("dtype"), py::arg("shape"), py::arg("format") = TileOpFormat::TILEOP_ND, py::arg("name") = "",
            "Create a tensor variable.\n\n"
            "Args:\n"
            "    dtype: Data type of the tensor\n"
            "    shape: Shape of the tensor\n"
            "    format: Data format of the tensor\n"
            "    name: Name of the tensor\n\n"
            "Returns:\n"
            "    LogicalTensor: The created tensor variable")

        .def("create_const_int", &IRBuilder::CreateConstInt, py::arg("value"),
             "Create a constant integer scalar.\n\n"
             "Args:\n"
             "    value: Integer value\n\n"
             "Returns:\n"
             "    SymbolicScalar: The created constant integer scalar")

        .def("create_scalar_var", &IRBuilder::CreateScalarVar, py::arg("sym") = "",
             "Create a scalar variable.\n\n"
             "    sym: Symbol of the variable\n\n"
             "Returns:\n"
             "    SymbolicScalar: The created scalar variable")
        .def(
            "create_tensor_op_stmt",
            [](IRBuilder& self, std::vector<ir::VarPtr> result, ir::VarPtr result_token, std::string opcode,
               std::vector<ir::ExprPtr> args, std::vector<ir::VarPtr> tokens, py::dict attrs, ir::Span span) {
                auto attr_list = pypto::ir::ConvertAttrDict(attrs);
                return self.CreateTensorOpStmt(result, result_token, opcode, args, tokens, attr_list, span);
            },
            py::arg("result"), py::arg("result_token"), py::arg("opcode"), py::arg("args"), py::arg("tokens"),
            py::arg("attrs"), py::arg("span"),
            "Create a tensor operation statement.\n\n"
            "Args:\n"
            "    result: Result of the operation\n"
            "    result_token: Result token of the operation\n"
            "    opcode: Opcode of the operation\n"
            "    args: Arguments of the operation\n"
            "    tokens: Tokens of the operation\n"
            "    attrs: Attributes of the operation\n"
            "    span: Span of the operation\n\n"
            "Returns:\n"
            "    TensorOpStmt: The created tensor operation statement")

        .def("create_assign_stmt", &IRBuilder::CreateAssignStmt, py::arg("var"), py::arg("value"), py::arg("span"),
             "Create an assignment statement.\n\n"
             "Args:\n"
             "    var: Variable of the assignment\n"
             "    value: Right-hand side of the assignment\n"
             "    span: Span of the assignment\n\n"
             "Returns:\n"
             "    AssignStmt: The created assignment statement")

        .def("create_seq_stmts", &IRBuilder::CreateSeqStmts, py::arg("stmts"), py::arg("span"),
             "Create a sequence of statements.\n\n"
             "Args:\n"
             "    stmts: Statements to create\n"
             "    span: Span of the sequence\n\n"
             "Returns:\n"
             "    SeqStmts: The created sequence of statements")

        .def("create_if_stmt", &IRBuilder::CreateIfStmt, py::arg("cond"), py::arg("then_body"), py::arg("else_body"),
             py::arg("return_vars"), py::arg("span"),
             "Create an if statement.\n\n"
             "Args:\n"
             "    cond: Condition of the if statement\n"
             "    then_body: Then body of the if statement\n"
             "    else_body: Else body of the if statement\n"
             "    return_vars: Return variables of the if statement\n"
             "    span: Span of the if statement\n\n"
             "Returns:\n"
             "    IfStmt: The created if statement")

        .def("create_return_stmt", &IRBuilder::CreateReturnStmt, py::arg("return_vars"), py::arg("span"),
             "Create a return statement.\n\n"
             "Args:\n"
             "    return_vars: Return variables of the return statement\n"
             "    span: Span of the return statement\n\n"
             "Returns:\n"
             "    ReturnStmt: The created return statement")

        .def("create_yield_stmt", &IRBuilder::CreateYieldStmt, py::arg("return_vars"), py::arg("span"),
             "Create a yield statement.\n\n"
             "Args:\n"
             "    return_vars: Return variables of the yield statement\n"
             "    span: Span of the yield statement\n\n"
             "Returns:\n"
             "    YieldStmt: The created yield statement")

        .def(
            "create_for_stmt",
            [](IRBuilder& self, const ir::VarPtr& loop_var, const ir::ExprPtr& start, const ir::ExprPtr& stop,
               const ir::ExprPtr& step, const std::vector<ir::IterArgPtr>& iter_args, const ir::StmtPtr& body,
               const std::vector<ir::VarPtr>& return_vars, const ir::Span& span, const py::object& attrs) {
                py::dict attr_dict;
                if (!attrs.is_none()) {
                    attr_dict = attrs.cast<py::dict>();
                }
                auto attr_list = pypto::ir::ConvertAttrDict(attr_dict);
                return self.CreateForStmt(loop_var, start, stop, step, iter_args, body, return_vars, span, attr_list);
            },
            py::arg("loop_var"), py::arg("start"), py::arg("stop"), py::arg("step"), py::arg("iter_args"),
            py::arg("body"), py::arg("return_vars"), py::arg("span"), py::arg("attrs") = py::none(),
            "Create a for statement.\n\n"
            "Args:\n"
            "    loop_var: Loop variable of the for statement\n"
            "    start: Start value of the loop variable\n"
            "    stop: Stop value of the loop variable\n"
            "    step: Step value of the loop variable\n"
            "    body: Body of the for statement\n"
            "    return_vars: Return variables of the for statement\n"
            "    span: Span of the for statement\n\n"
            "    attrs: Attributes of the for statement\n"
            "Returns:\n"
            "    ForStmt: The created for statement")

        .def("create_while_stmt", &IRBuilder::CreateWhileStmt, py::arg("cond"), py::arg("iter_args"), py::arg("body"),
             py::arg("return_vars"), py::arg("span"),
             "Create a while statement.\n\n"
             "Args:\n"
             "    cond: Condition of the while statement\n"
             "    iterArgs: Iteration arguments of the loop\n"
             "    body: Body of the while statement\n"
             "    returnVars: Return variables of the while statement\n"
             "    span: Span of the while statement\n\n"
             "Returns:\n"
             "    WhileStmt: The created while statement")

        .def("create_break_stmt", &IRBuilder::CreateBreakStmt, py::arg("return_vars"), py::arg("span"),
             "Create a break statement.\n\n"
             "Args:\n"
             "    return_vars: Return variables of the break statement\n"
             "    span: Span of the break statement\n\n"
             "Returns:\n"
             "    BreakStmt: The created break statement")

        .def("create_continue_stmt", &IRBuilder::CreateContinueStmt, py::arg("return_vars"), py::arg("span"),
             "Create a continue statement.\n\n"
             "Args:\n"
             "    return_vars: Return variables of the continue statement\n"
             "    span: Span of the continue statement\n\n"
             "Returns:\n"
             "    ContinueStmt: The created continue statement")

        .def(
            "create_function",
            [](IRBuilder& self, std::string name, std::vector<ir::VarPtr> params, std::vector<ir::TypePtr> returnTypes,
               ir::StmtPtr body, ir::Span span) { return self.CreateFunction(name, params, returnTypes, body, span); },
            py::arg("name"), py::arg("params"), py::arg("returnTypes"), py::arg("body"), py::arg("span"),
            "Create a function.\n\n"
            "Args:\n"
            "    name: Name of the function\n"
            "    params: Parameters of the function\n"
            "    returnTypes: Return types of the function\n"
            "    body: Body of the function\n"
            "    span: Span of the function\n\n"
            "Returns:\n"
            "    Function: The created function")

        .def("create_program", &IRBuilder::CreateProgram, py::arg("functions"), py::arg("name"), py::arg("span"),
             "Create a program.\n\n"
             "Args:\n"
             "    functions: Functions of the program\n"
             "    name: Name of the program\n"
             "    span: Span of the program\n\n"
             "Returns:\n"
             "    Program: The created program")

        .def("create_var_like", &IRBuilder::CreateVarLike, py::arg("name"), py::arg("value"),
             "Create a variable like the given value.\n\n"
             "Args:\n"
             "    name: Name of the variable\n"
             "    value: Value to base the variable on\n"
             "Returns:\n"
             "    Var: The created variable")

        .def(
            "create_iter_arg",
            [](IRBuilder& self, ir::VarPtr var, ir::ExprPtr initValue) { return self.CreateIterArg(var, initValue); },
            py::arg("var"), py::arg("initValue"),
            "Create an iteration argument.\n\n"
            "Args:\n"
            "    var: Variable of the iteration argument\n"
            "    initValue: Initial value of the iteration argument\n"
            "Returns:\n"
            "    IterArg: The created iteration argument")

        .def(
            "create_iter_arg",
            [](IRBuilder& self, std::string name, ir::TypePtr type, ir::ExprPtr initValue, ir::Span span) {
                return self.CreateIterArg(name, type, initValue, span);
            },
            py::arg("name"), py::arg("type"), py::arg("initValue"), py::arg("span"),
            "Create an iteration argument.\n\n"
            "Args:\n"
            "    name: Name of the iteration argument\n"
            "    type: Type of the iteration argument\n"
            "    initValue: Initial value of the iteration argument\n"
            "    span: Span of the iteration argument\n\n"
            "Returns:\n"
            "    IterArg: The created iteration argument")

        .def("emit_tensor_stmts", &IRBuilder::EmitTensorStmts, "Emit tensor statements.\n\n")

        .def("none", &IRBuilder::None, "Get the None value.\n\n")

        .def("set_insert_point", &IRBuilder::SetInsertPoint, py::arg("point"),
             "Set insert point.\n\n"
             "Args:\n"
             "    point: Insert point\n"
             "Returns:\n"
             "    None")

        .def("clear_insert_point", &IRBuilder::ClearInsertPoint, "Clear insert point.\n\n")
        .def("checkpoint", &IRBuilder::Checkpoint, "Create a checkpoint.\n\n")
        .def("restore", &IRBuilder::Restore, "Restore the last checkpoint.\n\n");

    py::class_<ir::InsertPoint, std::shared_ptr<ir::InsertPoint>>(ir, "InsertPoint").def(py::init<ir::SeqStmtsPtr>());

    py::class_<Operation, std::shared_ptr<Operation>, ir::TensorOpStmt>(ir, "Operation")
        .def("__str__", [](Operation& self) { return self.Dump(); });
    py::class_<RawSymbolicExpression, std::shared_ptr<RawSymbolicExpression>, ir::Expr>(ir, "SymbolicExpression")
        .def("__str__", [](RawSymbolicExpression& self) { return self.Dump(); });
    py::class_<RawSymbolicSymbol, std::shared_ptr<RawSymbolicSymbol>, ir::Var>(ir, "SymbolicSymbol")
        .def("__str__", [](RawSymbolicSymbol& self) { return self.Dump(); });
    py::class_<RawSymbolicImmediate, std::shared_ptr<RawSymbolicImmediate>, ir::ConstInt>(ir, "SymbolicImmediate")
        .def("__str__", [](RawSymbolicImmediate& self) { return self.Dump(); });

    ir.def("type_equal", [](ir::ExprPtr a, ir::ExprPtr b) {
        bool ret = ir::structural_equal(a->GetType(), b->GetType());
        if (ret) {
            if (auto type = ir::As<ir::LogicalTensorType>(b->GetType())) {
                auto ta = std::dynamic_pointer_cast<LogicalTensor>(std::const_pointer_cast<ir::Expr>(a));
                auto tb = std::dynamic_pointer_cast<LogicalTensor>(std::const_pointer_cast<ir::Expr>(b));
                return TypeEqual(ta, tb);
            }
        }
        return ret;
    });
}

} // namespace npu::tile_fwk

namespace pypto::ir {
void BindIRBuilder(py::module_& m) { npu::tile_fwk::BindIRBuilder(m); }

} // namespace pypto::ir
