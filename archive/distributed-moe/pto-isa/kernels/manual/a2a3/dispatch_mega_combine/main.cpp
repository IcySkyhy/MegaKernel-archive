/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_types.h"

#include "comm_mpi.h"
#include "data_utils.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/utils/const_args.hpp"
#include "runtime_context.hpp"
#include "tiling_builder.hpp"

extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int kDefaultWarmupIters = 3;
constexpr int kDefaultMeasureIters = 5;
constexpr double kMicrosecondsPerSecond = 1000.0 * 1000.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
static double g_sys_cnt_multiple = 20.0; // Default A2/A3, in ns per SYS_CNT tick.

struct DeviceMemoryReleaser {
    void operator()(void* allocation) const noexcept { (void)aclrtFree(allocation); }
};

struct HostMemoryReleaser {
    void operator()(void* allocation) const noexcept { (void)aclrtFreeHost(allocation); }
};

template <typename Releaser>
class AclOwnedBuffer {
public:
    AclOwnedBuffer() = default;
    AclOwnedBuffer(void* allocation, size_t allocationBytes) : storage_(allocation), byteCount_(allocationBytes) {}

    void* data() const noexcept { return storage_.get(); }

    size_t size() const noexcept { return byteCount_; }

private:
    std::unique_ptr<void, Releaser> storage_;
    size_t byteCount_ = 0;
};

using DeviceBuffer = AclOwnedBuffer<DeviceMemoryReleaser>;
using HostBuffer = AclOwnedBuffer<HostMemoryReleaser>;

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct RunOptions {
    int warmup_iters = kDefaultWarmupIters;
    int measure_iters = kDefaultMeasureIters;
    bool start_sync_debug = false;
};

struct RankHostInputs {
    std::vector<uint8_t> x;
    std::vector<uint8_t> weight1;
    std::vector<uint8_t> weight2;
    std::vector<uint8_t> expert_idx;
    std::vector<uint8_t> scale1;
    std::vector<uint8_t> scale2;
    std::vector<uint8_t> probs;
    std::vector<uint16_t> expected_out;
};

struct RankDeviceBuffers {
    DeviceBuffer x;
    DeviceBuffer weight1;
    DeviceBuffer weight2;
    DeviceBuffer expert_idx;
    DeviceBuffer scale1;
    DeviceBuffer scale2;
    DeviceBuffer probs;
    DeviceBuffer out;
    DeviceBuffer expert_token_nums;
    DeviceBuffer workspace;
    DeviceBuffer tiling;
    DeviceBuffer profile;
    HostBuffer profile_host;
};

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void* host_src = nullptr)
{
    if (bytes == 0) {
        return {};
    }
    void* allocation = nullptr;
    if (aclrtMalloc(&allocation, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed");
    }
    DeviceBuffer buffer(allocation, bytes);
    if (host_src != nullptr &&
        aclrtMemcpy(buffer.data(), bytes, host_src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMemcpy host->device failed");
    }
    return buffer;
}

HostBuffer MakeHostBuffer(size_t bytes)
{
    if (bytes == 0) {
        return {};
    }
    void* allocation = nullptr;
    if (aclrtMallocHost(&allocation, bytes) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMallocHost failed");
    }
    return HostBuffer(allocation, bytes);
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("fp16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    for (size_t idx = 0; idx < out.size(); ++idx) {
        const size_t byteOffset = idx * sizeof(uint16_t);
        out[idx] = static_cast<uint16_t>(bytes[byteOffset]) |
                   static_cast<uint16_t>(static_cast<uint16_t>(bytes[byteOffset + 1U]) << 8U);
    }
    return out;
}

int ParseEnvInt(const char* name, int default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer in env: ") + name);
    }
}

uint64_t AlignUpU64(uint64_t value, uint64_t align)
{
    if (align == 0U) {
        throw std::runtime_error("AlignUpU64 requires nonzero align");
    }
    return (value + align - 1U) / align * align;
}

uint64_t SwigluFullRowUbBytes(uint32_t n)
{
    auto align_ub = [](uint64_t value) { return AlignUpU64(value, 32U); };
    uint64_t ub_offset = 0;
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // activation
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // gate
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // temp / sigmoid
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // quant work
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(int8_t)); // int8 output
    ub_offset += 2U * 32U;                                            // max/scale scratch
    return ub_offset;
}

bool ZeroWindowMemory(const StandaloneRankRuntime& runtime)
{
    const uint64_t window_bytes = runtime.hccl.WindowBytes();
    void* window_ptr = runtime.hccl.WindowIn(static_cast<uint32_t>(runtime.hccl.rank_id));
    if (aclrtMemset(window_ptr, window_bytes, 0, window_bytes) != ACL_SUCCESS) {
        return false;
    }
    return true;
}

void ZeroDeviceBuffer(const DeviceBuffer& buffer, const char* name)
{
    if (buffer.size() == 0) {
        return;
    }
    if (aclrtMemset(buffer.data(), buffer.size(), 0, buffer.size()) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to zero ") + name);
    }
}

void PrepareIterationState(
    const StandaloneRankRuntime& runtime, const DeviceBuffer& out_dev, const DeviceBuffer& expert_token_nums_dev,
    const DeviceBuffer& workspace_dev, const DeviceBuffer& profile_dev)
{
    if (!ZeroWindowMemory(runtime)) {
        throw std::runtime_error("failed to zero HCCL windows");
    }
    ZeroDeviceBuffer(out_dev, "out buffer");
    ZeroDeviceBuffer(expert_token_nums_dev, "expert_token_nums");
    ZeroDeviceBuffer(workspace_dev, "workspace");
    ZeroDeviceBuffer(profile_dev, "profile buffer");
}

PerfStats CalcStats(const std::vector<double>& samples)
{
    PerfStats stats;
    if (samples.empty()) {
        return stats;
    }
    stats.min = *std::min_element(samples.begin(), samples.end());
    stats.max = *std::max_element(samples.begin(), samples.end());
    stats.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double sample : samples) {
        const double delta = sample - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

double ToTokensPerSecond(double tokens, double us) { return us > 0.0 ? tokens * kMicrosecondsPerSecond / us : 0.0; }

double ToTflops(double flops, double us) { return us > 0.0 ? flops * kMicrosecondsPerSecond / us / 1e12 : 0.0; }

double ToGbs(double bytes, double us) { return us > 0.0 ? bytes * kMicrosecondsPerSecond / us / kBytesPerGiB : 0.0; }

double SysCntTicksToUs(uint64_t ticks) { return static_cast<double>(ticks) * g_sys_cnt_multiple / 1000.0; }

std::vector<double> GatherMaxSamplesToRoot(const std::vector<double>& local_samples, int rank_id, int world_size)
{
    if (local_samples.empty()) {
        return {};
    }
    const size_t sample_count = local_samples.size();
    const int bytes_per_rank = static_cast<int>(sample_count * sizeof(double));
    std::vector<double> gathered;
    if (rank_id == 0) {
        gathered.resize(sample_count * static_cast<size_t>(world_size));
    }
    CommMpiGather(
        local_samples.data(), bytes_per_rank, COMM_MPI_CHAR,
        rank_id == 0 ? static_cast<void*>(gathered.data()) : nullptr, bytes_per_rank, COMM_MPI_CHAR, 0);
    if (rank_id != 0) {
        return {};
    }

    std::vector<double> max_samples(sample_count, 0.0);
    for (size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        double max_value = gathered[sample_idx];
        for (int rank = 1; rank < world_size; ++rank) {
            max_value = std::max(max_value, gathered[static_cast<size_t>(rank) * sample_count + sample_idx]);
        }
        max_samples[sample_idx] = max_value;
    }
    return max_samples;
}

double ReadKernelProfileUs(const DeviceBuffer& profile_dev, HostBuffer& profile_host, uint32_t block_dim)
{
    if (profile_dev.size() == 0 || profile_host.size() == 0 || block_dim == 0) {
        return 0.0;
    }
    if (aclrtMemcpy(
            profile_host.data(), profile_host.size(), profile_dev.data(), profile_dev.size(),
            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host profile copy failed");
    }

    uint64_t start_min = std::numeric_limits<uint64_t>::max();
    uint64_t end_max = 0;
    const auto* profile = static_cast<const uint8_t*>(profile_host.data());
    for (uint32_t block = 0; block < block_dim; ++block) {
        for (size_t profile_idx = 0; profile_idx < kMegaMoeProfileEntriesPerBlock; ++profile_idx) {
            const uint64_t* entry = reinterpret_cast<const uint64_t*>(
                profile + static_cast<size_t>(block) * kMegaMoeProfileBytesPerBlock +
                profile_idx * kMegaMoeProfileEntryBytes);
            const uint64_t start = entry[kMegaMoeProfileKernelStart];
            const uint64_t end = entry[kMegaMoeProfileKernelEnd];
            if (start == 0 && end == 0) {
                continue;
            }
            start_min = std::min(start_min, start);
            end_max = std::max(end_max, end);
        }
    }
    if (start_min == std::numeric_limits<uint64_t>::max() || end_max < start_min) {
        return 0.0;
    }
    const uint64_t duration_ticks = end_max - start_min;
    return SysCntTicksToUs(duration_ticks);
}

std::string BuildAccuracyReportText(int rank_id, const AccuracyReport& report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rank_id << " max_diff=" << report.max_abs_err
       << " max_ratio=" << report.max_rel_err << " err=" << report.mismatch_count << "/" << report.err_threshold
       << " -> " << (report.pass ? "PASS" : "FAIL");
    return os.str();
}

void PrintOrderedByRank(int rank_id, int world_size, const std::string& text)
{
    for (int turn = 0; turn < world_size; ++turn) {
        CommMpiBarrier();
        if (turn == rank_id) {
            std::cout << text << std::endl;
        }
    }
    CommMpiBarrier();
}

void PrintPerfSummary(
    const CaseConfig& cfg, uint32_t launch_block_dim, int warmup_iters, int measure_iters,
    const std::vector<double>& kernel_samples_us)
{
    if (kernel_samples_us.empty()) {
        return;
    }
    const PerfStats kernel_stats = CalcStats(kernel_samples_us);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===============================================================\n";
    std::cout << "[PROFILE] dispatch_mega_combine\n";
    std::cout << "  shape: m=" << cfg.m << " k=" << cfg.k << " n=" << cfg.n << " topk=" << cfg.topk
              << " expert_per_rank=" << cfg.expert_per_rank << " world_size=" << cfg.world_size << '\n';
    std::cout << "  schedule: aic_num=" << cfg.aic_num << " aiv_num=" << cfg.aiv_num
              << " launch_block_dim=" << launch_block_dim << " front_reorder_aiv_num=" << cfg.aiv_num << '\n';
    std::cout << "  iters: warmup=" << warmup_iters << " measure=" << measure_iters << '\n';
    std::cout << "  logical work(all ranks): input_tokens=" << cfg.input_tokens_all_ranks
              << " routed_tokens=" << cfg.routed_tokens_all_ranks
              << " remote_routed_tokens=" << cfg.remote_routed_tokens_all_ranks
              << " compute_flops=" << cfg.compute_flops_all_ranks << " comm_bytes=" << cfg.comm_bytes_all_ranks << '\n';
    std::cout << "  kernel(syscnt max rank per iter): avg=" << kernel_stats.avg << " us";
    std::cout << " min=" << kernel_stats.min << " us";
    std::cout << " max=" << kernel_stats.max << " us";
    std::cout << " std=" << kernel_stats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << ToTokensPerSecond(cfg.input_tokens_all_ranks, kernel_stats.avg)
              << " routed_tokens/s=" << ToTokensPerSecond(cfg.routed_tokens_all_ranks, kernel_stats.avg)
              << " eq_compute=" << ToTflops(cfg.compute_flops_all_ranks, kernel_stats.avg) << " TFLOPS"
              << " eq_comm=" << ToGbs(cfg.comm_bytes_all_ranks, kernel_stats.avg) << " GB/s\n";
    std::cout << "  note: equivalent compute/comm are logical workload estimates, not hardware counters.\n";
    std::cout << "===============================================================\n" << std::endl;
}

void ValidateFullPathConstraints(const CaseConfig& cfg)
{
    if (cfg.expert_per_rank != 8U && cfg.expert_per_rank != 16U && cfg.expert_per_rank != 32U) {
        throw std::runtime_error("expert_per_rank must be one of 8, 16 or 32");
    }
    if (cfg.k % 128U != 0U) {
        throw std::runtime_error("GMM1 requires K % 128 == 0");
    }
    if (cfg.n % 32U != 0U) {
        throw std::runtime_error("GMM1 requires N % 32 == 0");
    }
    if (cfg.n % 64U != 0U) {
        throw std::runtime_error("SwiGLU requires N % 64 == 0");
    }
    if (SwigluFullRowUbBytes(cfg.n) > AtlasA2::UB_SIZE) {
        throw std::runtime_error(
            "SwiGLU full-row UB capacity exceeded: ub_bytes=" + std::to_string(SwigluFullRowUbBytes(cfg.n)) +
            " max=" + std::to_string(AtlasA2::UB_SIZE));
    }
    if ((cfg.k * sizeof(uint16_t)) % 32U != 0U) {
        throw std::runtime_error("combine/unpermute requires K * sizeof(float16) to be 32-byte aligned");
    }
    if (cfg.max_output_size < cfg.m * cfg.topk) {
        throw std::runtime_error("unpermute requires max_output_size >= M * topK for large dropless path");
    }
}

RunOptions LoadRunOptions()
{
    RunOptions options;
    options.warmup_iters = ParseEnvInt("DISPATCH_MEGA_COMBINE_WARMUP_ITERS", kDefaultWarmupIters);
    options.measure_iters = ParseEnvInt("DISPATCH_MEGA_COMBINE_MEASURE_ITERS", kDefaultMeasureIters);
    options.start_sync_debug = ParseEnvInt("DISPATCH_MEGA_COMBINE_START_SYNC_DEBUG", 0) != 0;
    if (options.warmup_iters < 0 || options.measure_iters < 0) {
        throw std::runtime_error("warmup/measure iters must be non-negative");
    }
    return options;
}

MegaMoeBuildResult BuildAndValidateTiling(const CaseConfig& cfg, const StandaloneRankRuntime& runtime)
{
    MegaMoeBuildResult build = BuildMegaMoeTiling(cfg, runtime);
    const auto& front = build.tiling.frontReorderTiling;
    if (!FrontCaseIsSupported(front.frontCase)) {
        throw std::runtime_error("front unsupported case has no legacy fallback");
    }
    ValidateFullPathConstraints(cfg);
    return build;
}

RankHostInputs LoadRankHostInputs(const RankFileSet& files)
{
    RankHostInputs inputs;
    inputs.x = ReadBinaryFile(files.x);
    inputs.weight1 = ReadBinaryFile(files.weight1);
    inputs.weight2 = ReadBinaryFile(files.weight2);
    inputs.expert_idx = ReadBinaryFile(files.expert_idx);
    inputs.scale1 = ReadBinaryFile(files.scale1);
    inputs.scale2 = ReadBinaryFile(files.scale2);
    inputs.probs = ReadBinaryFile(files.probs);
    inputs.expected_out = BytesToU16(ReadBinaryFile(files.expected_out));
    return inputs;
}

RankDeviceBuffers AllocateRankDeviceBuffers(
    const CaseConfig& cfg, const MegaMoeBuildResult& build, const RankHostInputs& inputs)
{
    RankDeviceBuffers buffers;
    buffers.x = MakeDeviceBuffer(inputs.x.size(), inputs.x.data());
    buffers.weight1 = MakeDeviceBuffer(inputs.weight1.size(), inputs.weight1.data());
    buffers.weight2 = MakeDeviceBuffer(inputs.weight2.size(), inputs.weight2.data());
    buffers.expert_idx = MakeDeviceBuffer(inputs.expert_idx.size(), inputs.expert_idx.data());
    buffers.scale1 = MakeDeviceBuffer(inputs.scale1.size(), inputs.scale1.data());
    buffers.scale2 = MakeDeviceBuffer(inputs.scale2.size(), inputs.scale2.data());
    buffers.probs = MakeDeviceBuffer(inputs.probs.size(), inputs.probs.data());
    buffers.out = MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t));
    buffers.expert_token_nums = MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t));
    buffers.workspace = MakeDeviceBuffer(build.workspace_bytes);
    buffers.tiling = MakeDeviceBuffer(sizeof(build.tiling), &build.tiling);
    const size_t profile_bytes = static_cast<size_t>(build.block_dim) * kMegaMoeProfileBytesPerBlock;
    buffers.profile = MakeDeviceBuffer(profile_bytes);
    buffers.profile_host = MakeHostBuffer(profile_bytes);
    return buffers;
}

MegaMoeLaunchArgs BuildLaunchArgs(
    const MegaMoeBuildResult& build, const RankDeviceBuffers& buffers, bool start_sync_debug)
{
    uint64_t ffts_addr = 0;
    uint32_t ffts_len = 0;
    if (rtGetC2cCtrlAddr(&ffts_addr, &ffts_len) != 0) {
        throw std::runtime_error("rtGetC2cCtrlAddr failed");
    }
    MegaMoeLaunchArgs args;
    args.ffts = reinterpret_cast<void*>(ffts_addr);
    args.block_dim = build.block_dim;
    args.tiling = buffers.tiling.data();
    args.workspace = buffers.workspace.data();
    args.x = buffers.x.data();
    args.weight1 = buffers.weight1.data();
    args.weight2 = buffers.weight2.data();
    args.expert_idx = buffers.expert_idx.data();
    args.scale1 = buffers.scale1.data();
    args.scale2 = buffers.scale2.data();
    args.probs = buffers.probs.data();
    args.out = buffers.out.data();
    args.expert_token_nums = buffers.expert_token_nums.data();
    args.profile_data = buffers.profile.data();
    args.start_sync_debug = start_sync_debug ? 1U : 0U;
    return args;
}

void LaunchAndSync(const MegaMoeLaunchArgs& args, aclrtStream stream)
{
    launchMegaMoe(args, stream);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("stream sync failed");
    }
}

void RunWarmupIterations(
    const StandaloneRankRuntime& runtime, const RankDeviceBuffers& buffers, const MegaMoeLaunchArgs& args,
    int warmup_iters)
{
    CommMpiBarrier();
    for (int iter = 0; iter < warmup_iters; ++iter) {
        PrepareIterationState(runtime, buffers.out, buffers.expert_token_nums, buffers.workspace, buffers.profile);
        CommMpiBarrier();
        LaunchAndSync(args, runtime.compute_stream);
        CommMpiBarrier();
    }
}

std::vector<double> RunMeasureIterations(
    const StandaloneRankRuntime& runtime, RankDeviceBuffers& buffers, const MegaMoeLaunchArgs& args,
    const MegaMoeBuildResult& build, int measure_iters)
{
    std::vector<double> kernel_times_us;
    kernel_times_us.reserve(static_cast<size_t>(measure_iters));
    for (int iter = 0; iter < measure_iters; ++iter) {
        PrepareIterationState(runtime, buffers.out, buffers.expert_token_nums, buffers.workspace, buffers.profile);
        CommMpiBarrier();
        LaunchAndSync(args, runtime.compute_stream);
        CommMpiBarrier();
        kernel_times_us.push_back(ReadKernelProfileUs(buffers.profile, buffers.profile_host, build.block_dim));
    }
    return kernel_times_us;
}

std::vector<uint16_t> CopyActualOutputToHost(const CaseConfig& cfg, const DeviceBuffer& out_dev)
{
    std::vector<uint16_t> actual_out(static_cast<size_t>(cfg.m) * cfg.k);
    if (aclrtMemcpy(
            actual_out.data(), actual_out.size() * sizeof(uint16_t), out_dev.data(),
            actual_out.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host output copy failed");
    }
    return actual_out;
}

bool ReportRankAccuracy(
    int rank_id, int world_size, const CaseConfig& cfg, const RankHostInputs& inputs,
    const std::vector<uint16_t>& actual_out)
{
    const AccuracyReport report = CompareFp16File(inputs.expected_out, actual_out, cfg.compare_atol, cfg.compare_rtol);
    PrintOrderedByRank(
        rank_id, world_size,
        BuildAccuracyReportText(rank_id, report) + "\n" + (report.pass ? "PASS" : "FAIL") + std::string(" rank=") +
            std::to_string(rank_id));
    return report.pass;
}

bool RunOneRank(int rank_id, int world_size, const std::string& case_dir, const HcclRootInfo& root_info)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rank_id, world_size, root_info)) {
        return false;
    }

    bool ok = false;
    try {
        const RunOptions options = LoadRunOptions();
        const CaseConfig cfg = LoadCaseConfig(case_dir + "/case.json");
        const RankFileSet files = BuildRankFileSet(case_dir, rank_id);
        const MegaMoeBuildResult build = BuildAndValidateTiling(cfg, runtime);
        const RankHostInputs inputs = LoadRankHostInputs(files);
        RankDeviceBuffers buffers = AllocateRankDeviceBuffers(cfg, build, inputs);

        const MegaMoeLaunchArgs args = BuildLaunchArgs(build, buffers, options.start_sync_debug);
        RunWarmupIterations(runtime, buffers, args, options.warmup_iters);
        const std::vector<double> kernel_times_us =
            RunMeasureIterations(runtime, buffers, args, build, options.measure_iters);

        const std::vector<double> kernel_max_samples = GatherMaxSamplesToRoot(kernel_times_us, rank_id, world_size);
        if (rank_id == 0) {
            PrintPerfSummary(cfg, build.block_dim, options.warmup_iters, options.measure_iters, kernel_max_samples);
        }

        PrepareIterationState(runtime, buffers.out, buffers.expert_token_nums, buffers.workspace, buffers.profile);
        CommMpiBarrier();
        LaunchAndSync(args, runtime.compute_stream);
        CommMpiBarrier();

        const std::vector<uint16_t> actual_out = CopyActualOutputToHost(cfg, buffers.out);
        WriteBinaryFile(
            case_dir + "/output_rank" + std::to_string(rank_id) + ".bin", actual_out.data(),
            actual_out.size() * sizeof(uint16_t));
        ok = ReportRankAccuracy(rank_id, world_size, cfg, inputs, actual_out);
    } catch (const std::exception& ex) {
        std::cerr << "rank=" << rank_id << " error: " << ex.what() << std::endl;
        ok = false;
    }

    DestroyStandaloneRankRuntime(runtime);
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rank_id = CommMpiRank();
    const int world_size = CommMpiSize();
    const char* case_dir_env = std::getenv("DISPATCH_MEGA_COMBINE_CASE_DIR");
    const std::string case_dir = case_dir_env ? case_dir_env : "../out";

    if (aclInit(nullptr) != ACL_SUCCESS) {
        CommMpiFinalize();
        return 1;
    }
    if (rtSetDevice(rank_id) != 0) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    if (aclrtSetDevice(rank_id) != ACL_SUCCESS) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo root_info{};
    if (rank_id == 0 && HcclGetRootInfo(&root_info) != HCCL_SUCCESS) {
        aclrtResetDevice(rank_id);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    CommMpiBcast(&root_info, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const bool ok = RunOneRank(rank_id, world_size, case_dir, root_info);

    CommMpiBarrier();
    aclFinalize();
    CommMpiFinalize();
    return ok ? 0 : 1;
}
