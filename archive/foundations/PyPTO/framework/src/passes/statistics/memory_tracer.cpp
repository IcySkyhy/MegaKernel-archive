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
 * \file memory_tracer.cpp
 */

#include "memory_tracer.h"
#include <fstream>
#include "tilefwk/platform.h"
#include "tilefwk/data_type.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "MemoryTracer"

namespace npu::tile_fwk {

namespace {
constexpr const char* TRACE_VERSION = "2.5";
constexpr const char* FALLBACK_ARCH = "DAV_2201";
} // namespace

std::string MemoryTracer::PipeTypeToStr(PipeType p)
{
    switch (p) {
        case PipeType::PIPE_S:
            return "PIPE_S";
        case PipeType::PIPE_V:
            return "PIPE_V";
        case PipeType::PIPE_V2:
            return "PIPE_V2";
        case PipeType::PIPE_M:
            return "PIPE_M";
        case PipeType::PIPE_MTE1:
            return "PIPE_MTE1";
        case PipeType::PIPE_MTE2:
            return "PIPE_MTE2";
        case PipeType::PIPE_MTE3:
            return "PIPE_MTE3";
        case PipeType::PIPE_FIX:
            return "PIPE_FIX";
        default:
            return "PIPE_UNKNOWN";
    }
}

std::string MemoryTracer::CoreClassToStr(CoreClass c)
{
    switch (c) {
        case CoreClass::AIC:
            return "AIC";
        case CoreClass::AIV:
            return "AIV";
        default:
            return "UNKNOWN";
    }
}

std::string MemoryTracer::DDRKindToStr(DDRBufferKind k)
{
    switch (k) {
        case DDRBufferKind::FUNCTION_TEMP:
            return "FUNCTION_TEMP";
        case DDRBufferKind::SPILL_TEMP:
            return "SPILL_TEMP";
        case DDRBufferKind::INCAST:
            return "INCAST";
        case DDRBufferKind::OUTCAST:
            return "OUTCAST";
        default:
            return "UNKNOWN";
    }
}

MemoryTracer::Json MemoryTracer::DDRRefToJson(const DDRRef& ref)
{
    Json j;
    j["memId"] = ref.memId;
    j["kind"] = DDRKindToStr(ref.kind);
    j["memType"] = MemoryTypeToString(ref.memType);
    j["addrStart"] = ref.addrStart;
    j["addrEnd"] = ref.addrEnd;
    j["size"] = ref.size;
    return j;
}

void MemoryTracer::OnOpLaunch(const OpLaunchEvent& e)
{
    if (!inMainLoop_)
        return;
    if (spillCopyoutOpMagics_.count(e.opMagic) != 0)
        return; // absorbed by SPILL event
    Json event;
    event["eventType"] = "OP_LAUNCH";
    event["clock"] = e.clock;
    event["opMagic"] = e.opMagic;
    event["cycleEnd"] = e.cycleEnd;
    event["pipeType"] = PipeTypeToStr(e.pipeType);
    event["coreType"] = CoreClassToStr(e.coreLocation.coreType);
    event["coreIdx"] = e.coreLocation.coreIdx;
    event["inputMemIds"] = e.inputMemIds;
    event["outputMemIds"] = e.outputMemIds;
    Json refs = Json::array();
    for (const auto& r : e.ddrRefs)
        refs.push_back(DDRRefToJson(r));
    event["ddrRefs"] = std::move(refs);
    timeline_.push_back(std::move(event));
}

void MemoryTracer::OnOpRetire(const OpRetireEvent& e)
{
    if (!inMainLoop_)
        return;
    if (spillCopyoutOpMagics_.count(e.opMagic) != 0)
        return; // see OnOpLaunch
    Json event;
    event["eventType"] = "OP_RETIRE";
    event["clock"] = e.clock;
    event["opMagic"] = e.opMagic;
    event["freedMemIds"] = e.freedMemIds;
    timeline_.push_back(std::move(event));
}

void MemoryTracer::OnAllocExec(const AllocExecEvent& e)
{
    if (!inMainLoop_)
        return;
    Json event;
    event["eventType"] = "ALLOC_EXEC";
    event["clock"] = e.clock;
    event["memId"] = e.memId;
    event["memType"] = MemoryTypeToString(e.memType);
    event["coreType"] = CoreClassToStr(e.coreLocation.coreType);
    event["coreIdx"] = e.coreLocation.coreIdx;
    event["addrStart"] = e.addrStart;
    event["addrEnd"] = e.addrEnd;
    event["size"] = e.addrEnd - e.addrStart;
    Json lts = Json::array();
    for (const auto& lt : e.logicalTensors) {
        Json l;
        l["magic"] = lt.magic;
        l["validSize"] = lt.validSize;
        lts.push_back(std::move(l));
    }
    event["logicalTensors"] = std::move(lts);
    timeline_.push_back(std::move(event));
}

void MemoryTracer::OnSpill(const SpillEvent& e)
{
    if (!inMainLoop_)
        return; // SeqSchedule-side spills not traced (design doc §1.3)
    // Track copyout magic so its later OP_LAUNCH/OP_RETIRE in MainLoop are suppressed.
    if (e.spillCopyoutOpMagic != -1) {
        spillCopyoutOpMagics_.insert(e.spillCopyoutOpMagic);
    }
    Json event;
    event["eventType"] = "SPILL";
    event["clock"] = e.clock;
    event["beginClock"] = e.beginClock;
    event["endClock"] = e.endClock;
    event["memId"] = e.spillMemId;
    event["memType"] = MemoryTypeToString(e.memType);
    event["coreType"] = CoreClassToStr(e.coreLocation.coreType);
    event["coreIdx"] = e.coreLocation.coreIdx;
    event["addrStart"] = e.addrStart;
    event["addrEnd"] = e.addrEnd;
    event["triggerOpMagic"] = e.triggerOpMagic;
    event["triggerTensorSize"] = e.triggerTensorSize;
    event["spillCopyoutOpMagic"] = e.spillCopyoutOpMagic;
    event["reloadAllocOpMagic"] = e.reloadAllocOpMagic;
    event["reloadCopyInOpMagic"] = e.reloadCopyInOpMagic;
    event["spillTensorMagic"] = e.spillTensorMagic;
    event["spillCopyoutSize"] = e.spillCopyoutSize;
    // DDR landing buffer (only present when spillCopyoutOpMagic != -1).
    if (e.spillDdrMemId != -1) {
        event["spillDdrMemId"] = e.spillDdrMemId;
        event["ddrKind"] = DDRKindToStr(e.ddrKind);
        event["ddrMemType"] = MemoryTypeToString(e.ddrMemType);
        event["ddrAddrStart"] = e.ddrAddrStart;
        event["ddrAddrEnd"] = e.ddrAddrEnd;
        event["ddrSize"] = e.ddrSize;
    }
    timeline_.push_back(std::move(event));
}

void MemoryTracer::OnBufferRearrange(const BufferRearrangeEvent& e)
{
    if (!inMainLoop_)
        return; // see OnSpill — gates SeqSchedule-side rearrange
    Json event;
    event["eventType"] = "BUFFER_REARRANGE";
    event["clock"] = e.clock;
    event["memType"] = MemoryTypeToString(e.memType);
    event["coreType"] = CoreClassToStr(e.coreLocation.coreType);
    event["coreIdx"] = e.coreLocation.coreIdx;
    event["triggerOpMagic"] = e.triggerOpMagic;
    Json changes = Json::array();
    for (const auto& c : e.changes) {
        Json item;
        item["memId"] = c.memId;
        item["oldStart"] = c.oldStart;
        item["oldEnd"] = c.oldEnd;
        item["newStart"] = c.newStart;
        item["newEnd"] = c.newEnd;
        changes.push_back(std::move(item));
    }
    event["changes"] = std::move(changes);
    timeline_.push_back(std::move(event));
}

void MemoryTracer::OnAllocFail(const AllocFailEvent& e)
{
    if (!inMainLoop_)
        return; // see OnSpill — gates SeqSchedule-side alloc fail
    Json event;
    event["eventType"] = "ALLOC_FAIL";
    event["clock"] = e.clock;
    event["triggerOpMagic"] = e.triggerOpMagic;
    event["memType"] = MemoryTypeToString(e.memType);
    event["coreType"] = CoreClassToStr(e.coreLocation.coreType);
    event["coreIdx"] = e.coreLocation.coreIdx;
    event["requestSize"] = e.requestSize;
    Json snap;
    snap["capacity"] = e.capacity;
    Json occupied = Json::array();
    for (const auto& s : e.occupiedSlices) {
        Json item;
        item["memId"] = s.memId;
        item["addrStart"] = s.addrStart;
        item["addrEnd"] = s.addrEnd;
        item["size"] = s.size;
        item["ownerOpMagic"] = s.ownerOpMagic;
        occupied.push_back(std::move(item));
    }
    snap["occupiedSlices"] = std::move(occupied);
    Json freeIv = Json::array();
    for (const auto& f : e.freeIntervals) {
        Json item;
        item["start"] = f.start;
        item["size"] = f.size;
        freeIv.push_back(std::move(item));
    }
    snap["freeIntervals"] = std::move(freeIv);
    event["bufferSnapshot"] = std::move(snap);
    timeline_.push_back(std::move(event));
}

void MemoryTracer::OnInitDDRBuffer(const InitDDRBufferEvent& e)
{
    Json event;
    event["eventType"] = "INIT_DDR_BUFFER";
    event["clock"] = e.clock;
    event["memId"] = e.memId;
    event["kind"] = DDRKindToStr(e.kind);
    event["magic"] = e.magic;
    event["dtype"] = DataType2String(e.dtype);
    event["shape"] = e.shape;
    timeline_.push_back(std::move(event));
}

void MemoryTracer::OnScheduleEnd(const ScheduleEndEvent& e)
{
    totalCycles_ = e.totalCycles;
    succeeded_ = e.success;
    ended_ = true;
}

Status MemoryTracer::Flush(const std::string& folderPath)
{
    if (prefix_.empty())
        return SUCCESS;
    if (!ended_ && timeline_.empty())
        return SUCCESS;

    Json metadata;
    metadata["totalCycles"] = totalCycles_;
    metadata["scheduleResult"] = succeeded_ ? "SUCCESS" : "FAILED";
    auto arch = Platform::Instance().GetSoc().GetNPUArch();
    auto archStr = (arch == NPUArch::DAV_UNKNOWN) ? std::string(FALLBACK_ARCH) : NPUArchToString(arch);
    metadata["chipArch"] = std::move(archStr);

    Json doc;
    doc["version"] = TRACE_VERSION;
    doc["metadata"] = std::move(metadata);
    doc["timeline"] = std::move(timeline_);

    std::string traceFile = folderPath + "/" + prefix_ + "_OoO_Memory_Trace.json";
    std::ofstream out(traceFile);
    if (!out) {
        APASS_LOG_WARN_F(Elements::Function, "MemoryTracer: cannot open %s for write.", traceFile.c_str());
        return FAILED;
    }
    out << doc.dump(1) << std::endl;
    out.close();
    return SUCCESS;
}

} // namespace npu::tile_fwk
