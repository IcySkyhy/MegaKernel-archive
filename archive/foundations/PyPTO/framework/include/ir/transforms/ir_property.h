/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_
#define PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_

#include <cstdint>
#include <string>
#include <vector>

namespace pypto {
namespace ir {

/**
 * \brief Verifiable IR properties
 */
enum class IRProperty : uint32_t {
    SSAForm = 0,
    TypeChecked = 1,
    NoNestedCalls = 2,
    NormalizedStmtStructure = 3,
    FlattenedSingleStmt = 4,
    SplitIncoreOrch = 5,
    HasMemRefs = 6,
    IncoreBlockOps = 7,
    AllocatedMemoryAddr = 8,
};

/**
 * \brief A set of IR properties (bitset-based)
 */
class IRPropertySet {
public:
    IRPropertySet() : bits_(0) {}
    IRPropertySet(std::initializer_list<IRProperty> props) : bits_(0)
    {
        for (auto p : props)
            Insert(p);
    }

    void Insert(IRProperty prop) { bits_ |= (1u << static_cast<uint32_t>(prop)); }
    void Remove(IRProperty prop) { bits_ &= ~(1u << static_cast<uint32_t>(prop)); }
    bool Contains(IRProperty prop) const { return (bits_ & (1u << static_cast<uint32_t>(prop))) != 0; }
    bool ContainsAll(const IRPropertySet& other) const { return (bits_ & other.bits_) == other.bits_; }
    IRPropertySet Union(const IRPropertySet& other) const
    {
        IRPropertySet result;
        result.bits_ = bits_ | other.bits_;
        return result;
    }
    IRPropertySet Intersection(const IRPropertySet& other) const
    {
        IRPropertySet result;
        result.bits_ = bits_ & other.bits_;
        return result;
    }
    IRPropertySet Difference(const IRPropertySet& other) const
    {
        IRPropertySet result;
        result.bits_ = bits_ & ~other.bits_;
        return result;
    }
    bool Empty() const { return bits_ == 0; }

    std::vector<IRProperty> ToVector() const
    {
        std::vector<IRProperty> result;
        for (uint32_t i = 0; i <= 0x8; ++i) {
            if (bits_ & (1u << i)) {
                result.push_back(static_cast<IRProperty>(i));
            }
        }
        return result;
    }

    std::string ToString() const
    {
        std::string result = "{";
        bool first = true;
        for (auto prop : ToVector()) {
            if (!first)
                result += ", ";
            result += IRPropertyToString(prop);
            first = false;
        }
        result += "}";
        return result;
    }

    bool operator==(const IRPropertySet& other) const { return bits_ == other.bits_; }
    bool operator!=(const IRPropertySet& other) const { return bits_ != other.bits_; }

    static std::string IRPropertyToString(IRProperty prop)
    {
        switch (prop) {
            case IRProperty::SSAForm:
                return "SSAForm";
            case IRProperty::TypeChecked:
                return "TypeChecked";
            case IRProperty::NoNestedCalls:
                return "NoNestedCalls";
            case IRProperty::NormalizedStmtStructure:
                return "NormalizedStmtStructure";
            case IRProperty::FlattenedSingleStmt:
                return "FlattenedSingleStmt";
            case IRProperty::SplitIncoreOrch:
                return "SplitIncoreOrch";
            case IRProperty::HasMemRefs:
                return "HasMemRefs";
            case IRProperty::IncoreBlockOps:
                return "IncoreBlockOps";
            case IRProperty::AllocatedMemoryAddr:
                return "AllocatedMemoryAddr";
            default:
                return "Unknown";
        }
    }

private:
    uint32_t bits_;
};

inline std::string IRPropertyToString(IRProperty prop) { return IRPropertySet::IRPropertyToString(prop); }

/**
 * \brief Controls automatic verification in PassPipeline
 */
enum class VerificationLevel {
    None,  ///< No automatic verification
    Basic, ///< Verify lightweight properties once per pipeline (default)
};

/**
 * \brief Declares required/produced/invalidated properties for passes
 */
struct PassProperties {
    IRPropertySet required;
    IRPropertySet produced;
    IRPropertySet invalidated;
};

inline const IRPropertySet& GetVerifiedProperties()
{
    static IRPropertySet props = []() {
        IRPropertySet s;
        s.Insert(IRProperty::SSAForm);
        s.Insert(IRProperty::TypeChecked);
        s.Insert(IRProperty::AllocatedMemoryAddr);
        return s;
    }();
    return props;
}

inline VerificationLevel GetDefaultVerificationLevel()
{
    const char* env = std::getenv("PYPTO_VERIFY_LEVEL");
    if (env && std::string(env) == "none") {
        return VerificationLevel::None;
    }
    return VerificationLevel::Basic;
}

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_IR_PROPERTY_H_
