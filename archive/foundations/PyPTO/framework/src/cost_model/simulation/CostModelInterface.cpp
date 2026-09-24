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
 * \file CostModelInterface.cpp
 * \brief
 */

#include "CostModelInterface.h"

#include <iostream>

#include "utils/file_utils.h"
#include "cost_model/simulation/tools/ParseArgs.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace CostModel {
using namespace std;

int CostModelInterface::BuildCostModel(std::vector<std::string>& inputConfigs)
{
    // Get Input Config Parameter
    CostModel::ParseArgs argParser;
    string filename;
    string outdir = "ASCPPModelOut";
    bool drawGraph = false;
    int mode = 0;
    int logLevel = 3;
    int accLevel = 1;
    int executeCycleLimit = -1;
    vector<string> configs;
    vector<string> configFilePath;
    argParser.RegisterParam("-m", mode, "Mode. 0: Common Simulation; 1: Function Mode; 2: LeafFunction Simulation");
    argParser.RegisterParam("-f", filename, "Input File to load");
    argParser.RegisterParam("-t", logLevel, "MLOG_LEVEL. 1: DEBUG; 2: INFO; 3: WARN; 4: ERROR, 5: FATAL");
    argParser.RegisterParam("-o", outdir, "Output Directory. Default: ./ASCPPModelOut");
    argParser.RegisterParam("-d", drawGraph, "Draw Graph. 0: NONE; 1: DRAW");
    argParser.RegisterParam("-a", accLevel, "Accuracy Level. 1: LOW, Fast Model; 2: HIGH, CA Model");
    argParser.RegisterParam("-l", executeCycleLimit, "Cost Model Execute Cycle Threshold");
    argParser.RegisterParam("-s", configs, "Override default configs");
    argParser.RegisterParam("--conf", configFilePath, "Configuration combination files for different hardware");

    argParser.Parse(inputConfigs);

    for (auto& path : configFilePath) {
        std::vector<std::string> conf;
        if (path.find(".json") != std::string::npos) {
            parser.ParseJsonConfig(path, conf);
        } else {
            parser.ParseConfig(path, conf);
        }
        configs.insert(configs.end(), conf.begin(), conf.end());
    }

    if (!configs.empty()) {
        SIMULATION_LOGW("Override configurations:");
        for (auto& cfg : configs) {
            SIMULATION_LOGW("%s", cfg.c_str());
        }
    }

    (void)CreateDir(outdir);
    std::string graphsOutDir = outdir + "/graphs";
    (void)CreateDir(graphsOutDir);

    // Build Simulation System
    sim = std::make_shared<CostModel::SimSys>();
    sim->cfgs = configs;
    sim->config.OverrideDefaultConfig(&configs);
    sim->jsonPath = filename;
    sim->outdir = outdir;
    sim->graphsOutdir = graphsOutDir;
    sim->drawGraph = drawGraph;
    sim->totalTraceLogger->sim = sim;
    sim->accLevel = accLevel;
    sim->logLevel = logLevel;
    sim->mode = static_cast<SimMode>(mode);
    sim->executeCycleThreshold = (executeCycleLimit < 0) ? uint64_t(-1) : uint64_t(executeCycleLimit);

    if (sim->mode == SimMode::NORMAL) {
        sim->BuildSystem();
    } else if (sim->mode == SimMode::LEAF_FUNCTION) {
        sim->dynamicWorkflow = true;
        sim->config.aicpuMachineNumber = 1;
        sim->config.cubeMachineNumberPerAICPU = 1;
        sim->config.vecMachineNumberPerAICPU = 1;
        sim->config.coreMachineNumberPerAICPU = sim->config.cubeMachineNumberPerAICPU +
                                                sim->config.vecMachineNumberPerAICPU;
        sim->BuildSystem();
    }
    return 0;
}

void CostModelInterface::GetInput(std::vector<npu::tile_fwk::Function*>& inputFuncs, bool topoFromRootFunc,
                                  std::string& startFuncName)
{
    if (IsNeedInput(sim->mode)) {
        if (!startFuncName.empty()) {
            sim->config.startFunctionLabel = startFuncName;
        }
        if (!sim->jsonPath.empty()) {
            parser.ParseJson(sim, sim->jsonPath);
        } else if (sim->config.simulationFixedLatencyTask) {
            parser.ParseFixedLatencyTask(sim, sim->config.fixedLatencyTaskInfoPath);
        } else {
            parser.ParseFunction(sim, inputFuncs, topoFromRootFunc);
        }

        // load json with calendar information
        if (sim->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) {
            parser.ParseCalendarJson(sim, sim->config.calendarFile);
            sim->InitCalendarMode();
        }
    }
}

void CostModelInterface::Submit(std::vector<npu::tile_fwk::Function*>& inputFuncs, bool topoFromRootFunc,
                                std::string startFuncName)
{
    GetInput(inputFuncs, topoFromRootFunc, startFuncName);
}

void CostModelInterface::SubmitSingleFunction(npu::tile_fwk::Function* func)
{
    parser.ParseSingleFunction(sim, func);
    sim->InitCoreTask();
}

void CostModelInterface::Run()
{
    if (sim->mode == SimMode::EMULATOR) {
        RunFunctional();
    } else {
        RunPerformance();
    }
}

void CostModelInterface::RunPerformance()
{
    // Simulation System Work
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& device : sim->machineGroup[static_cast<int>(MachineType::DEVICE)]) {
        device->LoggerRecordTaskStart("Start Device Machine ");
    }

    bool terminate = false;
    while (!terminate) {
        sim->Step();
        terminate = sim->IsTerminate() || sim->IsDeadlock();
    }
    sim->globalCycles = sim->lastSimulationCycles;
    for (auto& device : sim->machineGroup[static_cast<int>(MachineType::DEVICE)]) {
        device->LoggerRecordTaskEnd();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    if (sim->IsDeadlock()) {
        SIMULATION_LOGE(CostModel::ForwardSimErrorScene::DEAD_LOCK, "Simulation is deadlock at cycle %lu !!!!!!!!!",
                        sim->globalCycles);
    }
    SIMULATION_LOGW("CostModel Simulation Runtime: %ld(s)", duration.count());
}

void CostModelInterface::RunFunctional()
{
    auto start = std::chrono::high_resolution_clock::now();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    SIMULATION_LOGW("CostModel Functional Simulation Runtime: %ld(s)", duration.count());
}

void CostModelInterface::Report()
{
    if (sim->mode == SimMode::NORMAL) {
        sim->OutputTrace();
        sim->OutputPerfettoTrace();
        sim->OutputLogForSwimLane();
        sim->OutputCalendarScheduleCpp();
        sim->OutputPerfMetrics();
    } else if (sim->mode == SimMode::LEAF_FUNCTION) {
        sim->OutputTrace();
        sim->OutputPerfettoTrace();
        sim->DumpFunctionExecuteTime();
    }
    if (sim->mode != SimMode::EMULATOR) {
        sim->OutputConfig();
        sim->PrintStat();
        sim->OutputPerfMetrics();
    }
    if (sim->IsDeadlock() && !sim->config.testDeadLock) {
        throw std::invalid_argument("Simulation Deadlock Error");
    }
}
} // namespace CostModel
