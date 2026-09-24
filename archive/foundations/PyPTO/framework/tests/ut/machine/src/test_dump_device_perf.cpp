/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <fstream>
#include <filesystem>

#include "machine/runtime/runner/dump_device_perf.h"
#include "tilefwk/aicpu_common.h"
#include "interface/configs/config_manager.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace npu::tile_fwk::dynamic {
extern json BuildSyncEventsJson(const TaskStat& taskStat, const uint8_t* perfDataPtr);
} // namespace npu::tile_fwk::dynamic

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DumpDevicePerfTest : public testing::Test {};

TEST_F(DumpDevicePerfTest, DumpDevTaskPerfData_EmptyPerfData)
{
    DeviceArgs args{};
    std::vector<void*> perfData;
    DumpDevTaskPerfData(args, perfData, false);
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, DumpDevTaskPerfData_EnvNotSet)
{
    DeviceArgs args{};
    std::vector<void*> perfData(1, nullptr);
    DumpDevTaskPerfData(args, perfData, false);
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_NoEvents)
{
    TaskStat stat{};
    stat.setEventNum = 0;
    stat.waitEventNum = 0;
    stat.setEventAddr = 0;
    stat.waitEventAddr = 0;
    std::vector<uint8_t> buf(256, 0);
    json result = BuildSyncEventsJson(stat, buf.data());
    EXPECT_TRUE(result.empty());
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_SetEventsOnly)
{
    std::vector<uint8_t> buf(1024, 0);
    uint64_t* setEventArea = reinterpret_cast<uint64_t*>(buf.data() + 64);
    setEventArea[0] = 100;
    setEventArea[1] = 0;
    setEventArea[2] = 300;
    TaskStat stat{};
    stat.setEventNum = 3;
    stat.waitEventNum = 0;
    stat.setEventAddr = 64;
    stat.waitEventAddr = 512;
    json result = BuildSyncEventsJson(stat, buf.data());
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0]["type"], "CV_SYNC_SET");
    EXPECT_EQ(result[0]["idx"], 0);
    EXPECT_EQ(result[0]["time"], 100u);
    EXPECT_EQ(result[1]["idx"], 2);
    EXPECT_EQ(result[1]["time"], 300u);
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_WaitEventsOnly)
{
    std::vector<uint8_t> buf(1024, 0);
    uint64_t* waitEventArea = reinterpret_cast<uint64_t*>(buf.data() + 256);
    waitEventArea[0] = 200;
    waitEventArea[1] = 400;
    TaskStat stat{};
    stat.setEventNum = 0;
    stat.waitEventNum = 2;
    stat.setEventAddr = 0;
    stat.waitEventAddr = 256;
    json result = BuildSyncEventsJson(stat, buf.data());
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0]["type"], "CV_SYNC_WAIT");
    EXPECT_EQ(result[0]["time"], 200u);
    EXPECT_EQ(result[1]["time"], 400u);
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_MixedAndSorted)
{
    std::vector<uint8_t> buf(2048, 0);
    uint64_t* setArea = reinterpret_cast<uint64_t*>(buf.data() + 128);
    uint64_t* waitArea = reinterpret_cast<uint64_t*>(buf.data() + 512);
    setArea[0] = 500;
    waitArea[0] = 100;
    waitArea[1] = 300;
    TaskStat stat{};
    stat.setEventNum = 1;
    stat.waitEventNum = 2;
    stat.setEventAddr = 128;
    stat.waitEventAddr = 512;
    json result = BuildSyncEventsJson(stat, buf.data());
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0]["time"], 100u);
    EXPECT_EQ(result[0]["type"], "CV_SYNC_WAIT");
    EXPECT_EQ(result[1]["time"], 300u);
    EXPECT_EQ(result[1]["type"], "CV_SYNC_WAIT");
    EXPECT_EQ(result[2]["time"], 500u);
    EXPECT_EQ(result[2]["type"], "CV_SYNC_SET");
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_AllZeroTimestamps)
{
    std::vector<uint8_t> buf(1024, 0);
    TaskStat stat{};
    stat.setEventNum = 3;
    stat.waitEventNum = 2;
    stat.setEventAddr = 0;
    stat.waitEventAddr = 256;
    json result = BuildSyncEventsJson(stat, buf.data());
    EXPECT_TRUE(result.empty());
}

TEST_F(DumpDevicePerfTest, DumpAicoreTaskExectInfo_Basic)
{
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 1;
    args.nrValidAic = 1;
    args.nrAicpu = 2;

    std::vector<uint8_t> buffer1(PERF_DATA_TOTAL_SIZE, 0);
    std::vector<uint8_t> buffer2(PERF_DATA_TOTAL_SIZE, 0);
    std::vector<uint8_t> buffer3(PERF_DATA_TOTAL_SIZE, 0);
    std::vector<void*> perfData = {buffer1.data(), buffer2.data(), buffer3.data()};

    DumpAicoreTaskExectInfo(args, perfData);
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, DumpAicoreTaskExectInfo_MultipleBlocks)
{
    DeviceArgs args{};
    args.nrAic = 2;
    args.nrAiv = 2;
    args.nrValidAic = 2;
    args.nrAicpu = 3;

    std::vector<std::vector<uint8_t>> buffers(7, std::vector<uint8_t>(PERF_DATA_TOTAL_SIZE, 0));
    std::vector<void*> perfData;
    for (auto& buf : buffers) {
        perfData.push_back(buf.data());
    }

    DumpAicoreTaskExectInfo(args, perfData);
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, DumpAicoreTaskExectInfo_WithSwimlaneFiles)
{
    DeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 1;
    args.nrValidAic = 1;
    args.nrAicpu = 2;
    args.archInfo = ArchInfo::DAV_2201;

    std::string logFolder = npu::tile_fwk::config::LogTopFolder();
    std::filesystem::create_directories(logFolder);

    std::string programJsonPath = logFolder + "/program.json";
    std::string dynTopoPath = logFolder + "/dyn_topo.txt";

    std::ofstream programJson(programJsonPath);
    programJson << "{}";
    programJson.close();

    std::ofstream dynTopo(dynTopoPath);
    dynTopo << "dummy topo";
    dynTopo.close();

    std::vector<uint8_t> buffer1(PERF_DATA_TOTAL_SIZE, 0);
    std::vector<uint8_t> buffer2(PERF_DATA_TOTAL_SIZE, 0);
    std::vector<uint8_t> buffer3(PERF_DATA_TOTAL_SIZE, 0);
    std::vector<void*> perfData = {buffer1.data(), buffer2.data(), buffer3.data()};

    DumpAicoreTaskExectInfo(args, perfData);

    std::filesystem::remove(programJsonPath);
    std::filesystem::remove(dynTopoPath);
    SUCCEED();
}
