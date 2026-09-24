/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_LIFECYCLE_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_LIFECYCLE_HPP

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP
#error "Include hns_1825_workspace_manager.hpp instead of this internal implementation header."
#endif

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {

inline WorkspaceManager::~WorkspaceManager() { (void)Finalize(); }

inline bool WorkspaceManager::Init(
    uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
    const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
    const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize)
{
    if (HasOwnedResources() && !Finalize()) {
        std::cerr << "[RoCE] previous session cleanup reported errors; continuing with a fresh HCOMM session"
                  << std::endl;
    }
    if (!ValidateInitArguments(
            rankId, rankCount, phyId, localIp, basePort, peerIps, peerPhyIds, peerSymAddrs, symmetricAddr,
            symmetricSize)) {
        return false;
    }
    AssignInitArguments(
        rankId, rankCount, phyId, localIp, basePort, peerIps, peerPhyIds, peerSymAddrs, symmetricAddr, symmetricSize);
    const auto initStart = std::chrono::steady_clock::now();
    Trace(
        "INIT begin manager=", static_cast<const void*>(this), " endpoint=", endpoint_, " memHandle=", memHandle_,
        " channels=", channelHandles_.size(), " rdmaInfo=", rdmaInfoDevice_, " symAddr=", symmetricAddr_,
        " symSize=", symmetricSize_, " basePort=", basePort_);
    LogInitConfiguration();
    if (!RunInitStage(&WorkspaceManager::CreateEndpoint, "EndpointCreate") ||
        !RunInitStage(&WorkspaceManager::RegisterMemory, "MemReg") ||
        !RunInitStage(&WorkspaceManager::BuildChannels, "ChannelCreateReady") ||
        !RunInitStage(&WorkspaceManager::FillRdmaInfo, "FillRdmaInfo")) {
        (void)Finalize();
        return false;
    }
    if (Verbose()) {
        std::cerr << "[RoCE][rank " << rankId_ << "] connect DONE: workspace(RdmaInfo)=0x" << std::hex
                  << reinterpret_cast<uint64_t>(rdmaInfoDevice_) << std::dec << std::endl;
    }
    initialized_ = true;
    Trace(
        "INIT success total_us=", ElapsedUs(initStart), " endpoint=", endpoint_, " memHandle=", memHandle_,
        " channels=", channelHandles_.size(), " rdmaInfo=", rdmaInfoDevice_);
    return true;
}

inline bool WorkspaceManager::Verbose()
{
    const char* value = std::getenv("PTO_ROCE_VERBOSE");
    return value != nullptr && value[0] == '1';
}

inline bool WorkspaceManager::Finalize()
{
    const bool hadResources = HasOwnedResources();
    const auto finalizeStart = std::chrono::steady_clock::now();
    if (hadResources) {
        Trace(
            "FINALIZE begin initialized=", initialized_, " endpoint=", endpoint_, " memHandle=", memHandle_,
            " channels=", channelHandles_.size(), " rdmaInfo=", rdmaInfoDevice_, " symAddr=", symmetricAddr_);
    }
    bool ok = true;
    auto phaseStart = std::chrono::steady_clock::now();
    const bool channelsOk = DestroyChannels();
    if (!channelsOk) {
        Trace("FINALIZE failed stage=ChannelDestroy elapsed_us=", ElapsedUs(phaseStart));
        ok = false;
    } else if (hadResources) {
        Trace("FINALIZE stage=ChannelDestroy ok elapsed_us=", ElapsedUs(phaseStart));
    }
    phaseStart = std::chrono::steady_clock::now();
    const bool memoryOk = UnregisterMemory();
    if (!memoryOk) {
        Trace("FINALIZE failed stage=MemUnreg elapsed_us=", ElapsedUs(phaseStart));
        ok = false;
    } else if (hadResources) {
        Trace("FINALIZE stage=MemUnreg ok elapsed_us=", ElapsedUs(phaseStart));
    }
    phaseStart = std::chrono::steady_clock::now();
    const bool endpointOk = DestroyEndpoint();
    if (!endpointOk) {
        Trace("FINALIZE failed stage=EndpointDestroy elapsed_us=", ElapsedUs(phaseStart));
        ok = false;
    } else if (hadResources) {
        Trace("FINALIZE stage=EndpointDestroy ok elapsed_us=", ElapsedUs(phaseStart));
    }
    phaseStart = std::chrono::steady_clock::now();
    const bool deviceInfoOk = FreeDeviceInfo();
    if (!deviceInfoOk) {
        Trace("FINALIZE failed stage=FreeRdmaInfo elapsed_us=", ElapsedUs(phaseStart));
        ok = false;
    } else if (hadResources) {
        Trace("FINALIZE stage=FreeRdmaInfo ok elapsed_us=", ElapsedUs(phaseStart));
    }
    ResetOwnedState();
    if (hadResources) {
        Trace("FINALIZE end ok=", ok, " total_us=", ElapsedUs(finalizeStart));
    }
    return ok;
}

template <typename... Args>
inline void WorkspaceManager::Trace(Args&&... args) const
{
    if (!Verbose()) {
        return;
    }
    std::ostringstream stream;
    stream << "[RoCE][case " << traceId_ << "][rank " << rankId_ << "] ";
    (stream << ... << std::forward<Args>(args));
    stream << '\n';
    const std::string line = stream.str();
    (void)std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
}

inline uint64_t WorkspaceManager::ElapsedUs(const std::chrono::steady_clock::time_point& start)
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

inline std::string WorkspaceManager::Hex(uint64_t value)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << value;
    return stream.str();
}

inline void WorkspaceManager::AssignInitArguments(
    uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
    const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
    const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize)
{
    rankId_ = rankId;
    rankCount_ = rankCount;
    phyId_ = phyId;
    localIp_ = localIp;
    basePort_ = basePort;
    peerIps_ = peerIps;
    peerPhyIds_ = peerPhyIds;
    peerSymAddrs_ = peerSymAddrs;
    symmetricAddr_ = symmetricAddr;
    symmetricSize_ = symmetricSize;
}

inline void WorkspaceManager::LogInitConfiguration() const
{
    if (!Verbose()) {
        return;
    }
    std::cerr << "[RoCE][rank " << rankId_ << "/" << rankCount_ << "] connect start: phyId=" << phyId_
              << " localIp=" << localIp_ << " basePort=" << basePort_ << " symAddr=0x" << std::hex
              << reinterpret_cast<uint64_t>(symmetricAddr_) << std::dec << " symSize=" << symmetricSize_ << std::endl;
    for (uint32_t rank = 0; rank < rankCount_; ++rank) {
        std::cerr << "[RoCE][rank " << rankId_ << "]   peer[" << rank << "] ip=" << peerIps_[rank]
                  << " phyId=" << peerPhyIds_[rank] << " symAddr=0x" << std::hex << peerSymAddrs_[rank] << std::dec
                  << std::endl;
    }
}

inline bool WorkspaceManager::RunInitStage(bool (WorkspaceManager::*stage)(), const char* stageName)
{
    const auto phaseStart = std::chrono::steady_clock::now();
    if (!(this->*stage)()) {
        return false;
    }
    Trace("INIT stage=", stageName, " ok elapsed_us=", ElapsedUs(phaseStart));
    return true;
}

inline bool WorkspaceManager::DestroyChannels()
{
    if (channelHandles_.empty()) {
        return true;
    }
    std::vector<ChannelHandle> valid;
    valid.reserve(channelHandles_.size());
    for (ChannelHandle handle : channelHandles_) {
        if (handle != 0) {
            valid.push_back(handle);
        }
    }
    if (valid.empty()) {
        channelHandles_.clear();
        channelPeer_.clear();
        return true;
    }
    if (Verbose()) {
        std::cerr << "[RoCE][rank " << rankId_ << "] HcommChannelDestroy n=" << valid.size() << std::endl;
    }
    for (size_t index = 0; index < channelHandles_.size(); ++index) {
        const ChannelHandle handle = channelHandles_[index];
        if (handle == 0) {
            continue;
        }
        const uint32_t peer = index < channelPeer_.size() ? channelPeer_[index] : UINT32_MAX;
        Trace("DESTROY channel index=", index, " peer=", peer, " handle=", Hex(static_cast<uint64_t>(handle)));
    }
    const HcommResult result = HcommChannelDestroy(valid.data(), static_cast<uint32_t>(valid.size()));
    channelHandles_.clear();
    channelPeer_.clear();
    if (result != 0) {
        std::cerr << "[RoCE] HcommChannelDestroy failed: " << result << std::endl;
        return false;
    }
    if (Verbose()) {
        std::cerr << "[RoCE][rank " << rankId_ << "] HcommChannelDestroy ok" << std::endl;
    }
    Trace("DESTROY channels API returned success");
    return true;
}

inline bool WorkspaceManager::UnregisterMemory()
{
    if (memHandle_ == nullptr) {
        return true;
    }
    const HcommMemHandle memHandle = memHandle_;
    memHandle_ = nullptr;
    if (endpoint_ == nullptr) {
        std::cerr << "[RoCE] cannot unregister memory without its endpoint" << std::endl;
        return false;
    }
    const HcommResult result = HcommMemUnreg(endpoint_, memHandle);
    if (result != 0) {
        std::cerr << "[RoCE] HcommMemUnreg failed: " << result << std::endl;
        return false;
    }
    Trace("DESTROY mem-unreg API returned success handle=", memHandle);
    return true;
}

inline bool WorkspaceManager::DestroyEndpoint()
{
    if (endpoint_ == nullptr) {
        return true;
    }
    const EndpointHandle endpoint = endpoint_;
    endpoint_ = nullptr;
    const HcommResult result = HcommEndpointDestroy(endpoint);
    if (result != 0) {
        std::cerr << "[RoCE] HcommEndpointDestroy failed: " << result << std::endl;
        return false;
    }
    Trace("DESTROY endpoint API returned success handle=", endpoint);
    return true;
}

inline bool WorkspaceManager::FreeDeviceInfo()
{
    if (rdmaInfoDevice_ == nullptr) {
        return true;
    }
    void* rdmaInfo = rdmaInfoDevice_;
    rdmaInfoDevice_ = nullptr;
    const aclError result = aclrtFree(rdmaInfo);
    if (result != ACL_SUCCESS) {
        std::cerr << "[RoCE] aclrtFree(rdmaInfo) failed: " << static_cast<int>(result) << std::endl;
        return false;
    }
    Trace("DESTROY rdma-info free returned success addr=", rdmaInfo);
    return true;
}

inline bool WorkspaceManager::HasOwnedResources() const
{
    return initialized_ || endpoint_ != nullptr || memHandle_ != nullptr || rdmaInfoDevice_ != nullptr ||
           !channelHandles_.empty();
}

inline bool WorkspaceManager::ValidateBasicInitArguments(
    uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
    const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
    const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize)
{
    if (rankCount == 0 || rankId >= rankCount || peerIps.size() != rankCount || peerPhyIds.size() != rankCount ||
        peerSymAddrs.size() != rankCount || localIp.empty() || symmetricAddr == nullptr || symmetricSize == 0) {
        std::cerr << "[RoCE] invalid rank, peer, IP, or symmetric-memory arguments" << std::endl;
        return false;
    }
    if (peerIps[rankId] != localIp || peerPhyIds[rankId] != phyId ||
        peerSymAddrs[rankId] != reinterpret_cast<uint64_t>(symmetricAddr)) {
        std::cerr << "[RoCE] local IP/phyId/address does not match this rank's gathered peer entry" << std::endl;
        return false;
    }
    if (basePort > UINT16_MAX - (kMaxRanksPerNic * kMaxRanksPerNic - 1)) {
        std::cerr << "[RoCE] basePort leaves no room for the per-rank channel port range" << std::endl;
        return false;
    }
    return true;
}

inline bool WorkspaceManager::ValidatePeerPortMapping(
    uint32_t rankCount, const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
    const std::vector<uint64_t>& peerSymAddrs)
{
    for (uint32_t rank = 0; rank < rankCount; ++rank) {
        if (peerIps[rank].empty() || peerPhyIds[rank] == UINT32_MAX || peerSymAddrs[rank] == 0) {
            std::cerr << "[RoCE] peer " << rank << " has an empty IP, invalid phyId, or null symmetric address"
                      << std::endl;
            return false;
        }
        uint32_t ranksOnNic = 0;
        bool occupiedSlots[kMaxRanksPerNic] = {};
        for (uint32_t peer = 0; peer < rankCount; ++peer) {
            if (peerIps[peer] != peerIps[rank]) {
                continue;
            }
            ++ranksOnNic;
            const uint32_t slot = peer % kMaxRanksPerNic;
            if (occupiedSlots[slot]) {
                std::cerr << "[RoCE] ranks sharing NIC IP " << peerIps[rank]
                          << " collide after modulo port mapping at slot " << slot << std::endl;
                return false;
            }
            occupiedSlots[slot] = true;
        }
        if (ranksOnNic > kMaxRanksPerNic) {
            std::cerr << "[RoCE] more than " << kMaxRanksPerNic << " ranks share NIC IP " << peerIps[rank]
                      << "; the channel port mapping would collide" << std::endl;
            return false;
        }
    }
    return true;
}

inline bool WorkspaceManager::ValidateInitArguments(
    uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
    const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
    const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize)
{
    return ValidateBasicInitArguments(
               rankId, rankCount, phyId, localIp, basePort, peerIps, peerPhyIds, peerSymAddrs, symmetricAddr,
               symmetricSize) &&
           ValidatePeerPortMapping(rankCount, peerIps, peerPhyIds, peerSymAddrs);
}

inline void WorkspaceManager::ResetOwnedState()
{
    channelHandles_.clear();
    channelPeer_.clear();
    peerIps_.clear();
    peerPhyIds_.clear();
    peerSymAddrs_.clear();
    memHandle_ = nullptr;
    endpoint_ = nullptr;
    rdmaInfoDevice_ = nullptr;
    symmetricAddr_ = nullptr;
    symmetricSize_ = 0;
    localIp_.clear();
    rankId_ = 0;
    rankCount_ = 1;
    phyId_ = 0;
    basePort_ = 60032;
    initialized_ = false;
}

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_LIFECYCLE_HPP
