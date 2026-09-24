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
 * \file memory_tracer.h
 * \brief OoO Schedule memory visualization tracer (a ScheduleObserver).
 *        See docs/passes/memory_visualization_design.md.
 */

#ifndef MEMORY_TRACER_H
#define MEMORY_TRACER_H

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include "passes/statistics/schedule_observer.h"

namespace npu::tile_fwk {

class MemoryTracer : public ScheduleObserver {
public:
    // ordered_json preserves insertion order to match design-doc field order.
    using Json = nlohmann::ordered_json;

    MemoryTracer() = default;

    void SetOutputPrefix(const std::string& prefix) { prefix_ = prefix; }

    void OnOpLaunch(const OpLaunchEvent& e) override;
    void OnOpRetire(const OpRetireEvent& e) override;
    void OnAllocExec(const AllocExecEvent& e) override;
    void OnSpill(const SpillEvent& e) override;
    void OnBufferRearrange(const BufferRearrangeEvent& e) override;
    void OnAllocFail(const AllocFailEvent& e) override;
    void OnScheduleEnd(const ScheduleEndEvent& e) override;
    void OnInitDDRBuffer(const InitDDRBufferEvent& e) override;
    void OnMainLoopBegin() override { inMainLoop_ = true; }
    void OnMainLoopEnd() override { inMainLoop_ = false; }

    // Writes <prefix>_OoO_Memory_Trace.json; no-op if prefix is unset.
    Status Flush(const std::string& folderPath);

private:
    static std::string PipeTypeToStr(PipeType p);
    static std::string CoreClassToStr(CoreClass c);
    static std::string DDRKindToStr(DDRBufferKind k);
    static Json DDRRefToJson(const DDRRef& ref);

    std::string prefix_;
    Json timeline_ = Json::array();
    int totalCycles_{0};
    bool succeeded_{false};
    bool ended_{false};
    bool inMainLoop_{false}; // trace is MainLoop-only; SeqSchedule-side events dropped
    // Spill copyouts whose launch/retire are absorbed by the owning SPILL event.
    std::unordered_set<int> spillCopyoutOpMagics_;
};

} // namespace npu::tile_fwk
#endif // MEMORY_TRACER_H
