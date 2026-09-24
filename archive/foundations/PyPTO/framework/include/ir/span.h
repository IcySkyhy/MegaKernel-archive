/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file span.h
 * \brief Source location tracking for IR nodes
 *
 * This header provides the Span class for tracking source code locations
 * throughout the IR. Span objects are immutable and can be shared across
 * multiple IR nodes.
 */

#pragma once
#include <string>

namespace pypto {
namespace ir {

class SpanImpl;

/**
 * \brief Represents a source code location span
 *
 * Immutable structure that captures the location of an IR node in the original
 * source code. Used for error reporting and debugging.
 *
 * The span includes:
 * - Source filename
 * - Beginning line and column (1-indexed)
 * - Ending line and column (1-indexed, -1 means unknown)
 */
class Span {
public:
    Span();

    Span(const Span& other);

    Span& operator=(const Span& other);

    Span(Span&& other);

    Span& operator=(Span&& other);

    /**
     * \brief Construct a source span
     *
     * \param file Source filename
     * \param beginLine Begin line (1-indexed)
     * \param beginColumn Begin column (1-indexed)
     * \param endLine End line (1-indexed), -1 means unknown
     * \param endColumn End column (1-indexed), -1 means unknown
     */
    Span(const std::string& file, int beginLine, int beginColumn, int endLine = -1, int endColumn = -1);

    ~Span();

    /**
     * \brief Get the source filename
     *
     * \return Source filename
     */
    const std::string& Filename() const;

    /**
     * \brief Get the beginning line number (1-indexed)
     *
     * \return Beginning line number
     */
    int BeginLine() const;

    /**
     * \brief Get the beginning column number (1-indexed)
     *
     * \return Beginning column number
     */
    int BeginColumn() const;

    /**
     * \brief Get the ending line number (1-indexed)
     *
     * \return Ending line number
     */
    int EndLine() const;

    /**
     * \brief Get the ending column number (1-indexed)
     *
     * \return Ending column number
     */
    int EndColumn() const;

    /**
     * \brief Convert span to string representation
     *
     * \return String in format "filename:begin_line:begin_column"
     */
    std::string ToString() const;

    /**
     * \brief Check if the span is unknown (has valid line/column numbers)
     *
     * \return true if the span impl is the same as the unknown singleton
     */
    bool IsUnknown() const { return impl_ == Unknown().impl_; }

    /**
     * \brief Check if the span is valid (has valid line/column numbers)
     *
     * \return true if the span has valid coordinates
     */
    bool IsValid() const;
    static bool IsUnknown(const Span& span);

    /**
     * \brief Create an unknown/invalid span
     *
     * \return Span with empty filename and invalid coordinates
     */
    static Span& Unknown();

    static void SetCurrent(const Span& span);

    static Span& Current();

    static void ClearCurrent() { SetCurrent(Unknown()); }

private:
    SpanImpl* impl_;
};

} // namespace ir
} // namespace pypto
