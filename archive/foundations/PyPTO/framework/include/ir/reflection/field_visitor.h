/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "core/logging.h"
#include "ir/reflection/field_traits.h"

namespace pypto {
namespace ir {

// Forward declarations
class IRNode;
using IRNodePtr = std::shared_ptr<const IRNode>;

namespace reflection {

/**
 * \brief Type trait to check if a type is a shared_ptr to an IRNode-derived type
 *
 * Used to dispatch field visiting logic based on field type.
 * This is the general trait that supports all IRNode types (Expr, Stmt, etc.).
 */
template <typename T, typename = void>
struct IsIRNodeField : std::false_type {};

// Generic specialization for any shared_ptr<const T> where T derives from IRNode
template <typename IRNodeType>
struct IsIRNodeField<std::shared_ptr<const IRNodeType>, std::enable_if_t<std::is_base_of_v<IRNode, IRNodeType>>>
    : std::true_type {};

// Non-const shared_ptr<T> where T derives from IRNode (T must not be const)
template <typename IRNodeType>
struct IsIRNodeField<std::shared_ptr<IRNodeType>,
                     std::enable_if_t<std::is_base_of_v<IRNode, IRNodeType> && !std::is_const_v<IRNodeType>>>
    : std::true_type {};

/**
 * \brief Type trait to check if a type is std::vector of IRNode pointers
 *
 * Used to handle collections of IR nodes specially.
 * Matches any vector<shared_ptr<const T>> where T derives from IRNode.
 */
template <typename T>
struct IsIRNodeVectorField : std::false_type {};

// Generic specialization for any vector<shared_ptr<const T>> where T derives from IRNode
template <typename IRNodeType>
struct IsIRNodeVectorField<std::vector<std::shared_ptr<const IRNodeType>>>
    : std::integral_constant<bool, std::is_base_of_v<IRNode, IRNodeType>> {};

/**
 * \brief Type trait to check if a type is std::optional of IRNode pointer
 *
 * Used to handle optional IR node fields specially.
 * Matches any optional<shared_ptr<const T>> where T derives from IRNode.
 */
template <typename T>
struct IsIRNodeOptionalField : std::false_type {};

// Specialization for std::optional<shared_ptr<const T>> where T derives from IRNode
template <typename IRNodeType>
struct IsIRNodeOptionalField<std::optional<std::shared_ptr<const IRNodeType>>>
    : std::integral_constant<bool, std::is_base_of_v<IRNode, IRNodeType>> {};

// // Non-const std::optional<shared_ptr<T>> where T derives from IRNode (T must not be const)
template <typename IRNodeType>
struct IsIRNodeOptionalField<std::optional<std::shared_ptr<IRNodeType>>>
    : std::integral_constant<bool, std::is_base_of_v<IRNode, IRNodeType> && !std::is_const_v<IRNodeType>> {};

/**
 * \brief Type trait to check if a type is std::map with IRNode pointer values
 *
 * Used to handle map fields specially (e.g., map of string to FunctionPtr).
 * Matches any map<shared_ptr<const K>, shared_ptr<const V>, Comp> where V derives from IRNode,
 * or map<string, shared_ptr<const V>> where V derives from IRNode.
 * The key type K does not need to derive from IRNode.
 */
template <typename T>
struct IsIRNodeMapField : std::false_type {};

// Specialization for std::map with shared_ptr key and IRNode-derived value type
template <typename KeyType, typename ValueType, typename Compare>
struct IsIRNodeMapField<std::map<std::shared_ptr<const KeyType>, std::shared_ptr<const ValueType>, Compare>>
    : std::integral_constant<bool, std::is_base_of_v<IRNode, ValueType>> {};

// Specialization for std::map with std::string key and IRNode-derived value type
template <typename ValueType>
struct IsIRNodeMapField<std::map<std::string, std::shared_ptr<const ValueType>>>
    : std::integral_constant<bool, std::is_base_of_v<IRNode, ValueType>> {};

/**
 * \brief Generic field iterator for compile-time field visitation
 *
 * Iterates over all fields in one or more IR nodes using field descriptors,
 * calling appropriate visitor methods for each field type.
 *
 * Supports single-node visitation (e.g., for hashing) and multi-node visitation
 * (e.g., for equality comparison). The visitor methods receive as many field
 * arguments as there are nodes being visited.
 *
 * Uses C++17 fold expressions for compile-time iteration.
 *
 * \tparam NodeType The IR node type being visited
 * \tparam Visitor The visitor type (must have ResultType and visit methods)
 * \tparam Descriptors Parameter pack of field descriptors
 */
template <typename NodeType, typename Visitor, typename... Descriptors>
class FieldIterator {
public:
    using ResultType = typename Visitor::ResultType;

    /**
     * \brief Visit all fields of a single node
     *
     * Visitor methods are called with single field arguments:
     *   - VisitIRNodeField(field)
     *   - VisitIRNodeVectorField(field)
     *   - VisitIRNodeMapField(field)
     *   - VisitLeafField(field)
     *
     * \param node The node instance to visit
     * \param visitor The visitor instance
     * \param descriptors Field descriptor instances
     * \return Accumulated result from visiting all fields
     */
    static ResultType Visit(const NodeType& node, Visitor& visitor, const Descriptors&... descriptors)
    {
        ResultType result = visitor.InitResult();
        (VisitField(visitor, descriptors, result, node), ...);
        return result;
    }

    /**
     * \brief Visit all fields of two nodes pairwise
     *
     * Visitor methods are called with two field arguments:
     *   - VisitIRNodeField(lhs_field, rhs_field)
     *   - VisitIRNodeVectorField(lhs_field, rhs_field)
     *   - VisitIRNodeMapField(lhs_field, rhs_field)
     *   - VisitLeafField(lhs_field, rhs_field)
     *
     * \param lhs Left-hand side node
     * \param rhs Right-hand side node
     * \param visitor The visitor instance
     * \param descriptors Field descriptor instances
     * \return Accumulated result from visiting all field pairs
     */
    static ResultType Visit(const NodeType& lhs, const NodeType& rhs, Visitor& visitor,
                            const Descriptors&... descriptors)
    {
        ResultType result = visitor.InitResult();
        (VisitField(visitor, descriptors, result, lhs, rhs), ...);
        return result;
    }

private:
    /**
     * \brief Visit a single field from N nodes using its descriptor
     *
     * Dispatches based on field kind (IGNORE/DEF/USUAL).
     *
     * \tparam Desc The field descriptor type
     * \tparam Nodes Parameter pack of node types (all must be NodeType)
     */
    template <typename Desc, typename... Nodes>
    static void VisitField(Visitor& visitor, const Desc& desc, ResultType& result, const Nodes&... nodes)
    {
        using KindTag = typename Desc::KindTag;

        if constexpr (std::is_same_v<KindTag, IgnoreFieldTag>) {
            visitor.VisitIgnoreField([&]() { VisitFieldImpl(visitor, desc, result, nodes...); });
        } else if constexpr (std::is_same_v<KindTag, DefFieldTag>) {
            visitor.VisitDefField([&]() { VisitFieldImpl(visitor, desc, result, nodes...); });
        } else if constexpr (std::is_same_v<KindTag, UsualFieldTag>) {
            visitor.VisitUsualField([&]() { VisitFieldImpl(visitor, desc, result, nodes...); });
        } else {
            INTERNAL_UNREACHABLE << "Invalid field kind tag: " << typeid(KindTag).name() << " for field " << desc.name;
        }
    }

    /**
     * \brief Implementation of field visitation
     *
     * Dispatches based on field type (IRNode/vector/map/scalar) and calls
     * the appropriate visitor method with fields from all nodes.
     */
    template <typename Desc, typename... Nodes>
    static void VisitFieldImpl(Visitor& visitor, const Desc& desc, ResultType& result, const Nodes&... nodes)
    {
        using FieldType = typename Desc::FieldType;

        if constexpr (IsIRNodeOptionalField<FieldType>::value) {
            // Optional IRNodePtr field - treat as IRNode field
            auto fieldResult = visitor.VisitIRNodeField(desc.Get(nodes)...);
            visitor.CombineResult(result, fieldResult, desc);
        } else if constexpr (IsIRNodeField<FieldType>::value) {
            // Single IRNodePtr field - expand to visitor.VisitIRNodeField(desc.Get(node1), desc.Get(node2), ...)
            auto fieldResult = visitor.VisitIRNodeField(desc.Get(nodes)...);
            visitor.CombineResult(result, fieldResult, desc);
        } else if constexpr (IsIRNodeVectorField<FieldType>::value) {
            // Vector of IRNodePtr
            auto fieldResult = visitor.VisitIRNodeVectorField(desc.Get(nodes)...);
            visitor.CombineResult(result, fieldResult, desc);
        } else if constexpr (IsIRNodeMapField<FieldType>::value) {
            // Map of IRNodePtr to IRNodePtr
            auto fieldResult = visitor.VisitIRNodeMapField(desc.Get(nodes)...);
            visitor.CombineResult(result, fieldResult, desc);
        } else {
            // Scalar field (int, string, OpPtr, etc.)
            auto fieldResult = visitor.VisitLeafField(desc.Get(nodes)...);
            visitor.CombineResult(result, fieldResult, desc);
        }
    }
};

} // namespace reflection
} // namespace ir
} // namespace pypto
