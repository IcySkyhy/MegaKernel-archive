/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host-only bootstrap helpers for the RoCE (HNS_1825) control plane.
//
//   1. ResolvePhyId — global physical device id via aclrt/runtime (HCCP/topo use phyId, not ACL id).
//   2. ResolveLocalRdmaIp — CLOS IPv4 for that phyId from /etc/hccl_rootinfo.json.
//   3. ResolveLocalRdmaIpFromVirtualTopology — RoCE IPv4 selected from topologyd's virtualTopology.xml.
//
// Optional symbols are resolved with dlsym(RTLD_DEFAULT, ...); missing symbols degrade gracefully
// so callers can fall back to environment overrides.

#ifndef PTO_TESTS_NPU_A5_COMM_ST_RDMA_BACKENDS_HNS_1825_BOOTSTRAP_HPP
#define PTO_TESTS_NPU_A5_COMM_ST_RDMA_BACKENDS_HNS_1825_BOOTSTRAP_HPP

#if defined(__CCE_KT_TEST__)
#error "hns_1825_bootstrap.hpp is a host-only header and cannot be included in device code."
#endif

#include <dlfcn.h>

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "securec.h"

#include "../../../comm_mpi.h"

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {
namespace bootstrap {

constexpr const char* kDefaultRootInfoPath = "/etc/hccl_rootinfo.json";
constexpr const char* kDefaultVirtualTopologyPath = "/var/run/ascend-topologyd/virtualTopology.xml";

// Resolve the global physical device id used by HCOMM and the topology data.
// Returns false (and leaves phyId untouched) if the runtime symbols are unavailable or fail.
inline bool ResolvePhyId(uint32_t& phyId)
{
    using GetDevFn = int (*)(int32_t*);
    using MapFn = int (*)(int32_t, int32_t*);

    auto getDev = reinterpret_cast<GetDevFn>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
    auto byLogic = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
    auto byUser = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByUserDevId"));
    auto byIndex = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "rtGetDevicePhyIdByIndex"));
    if (getDev == nullptr) {
        return false;
    }

    int32_t userId = -1;
    if (getDev(&userId) != 0 || userId < 0) {
        return false;
    }

    int32_t phy = -1;
    // Prefer ByUserDevId; deprecated ByLogicDevId still expects userId (not logic id).
    if (byUser != nullptr && byUser(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    if (byLogic != nullptr && byLogic(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    if (byIndex != nullptr && byIndex(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    return false;
}

namespace detail {

// Read an unsigned integer that follows `"key"` (value may be a bare number or a quoted number).
// Returns false if the key is not present at/after `from`.
inline bool ExtractUintAfterKey(
    const std::string& s, const std::string& key, size_t from, uint32_t& value, size_t& keyPos)
{
    size_t p = s.find(key, from);
    if (p == std::string::npos) {
        return false;
    }
    keyPos = p;
    p += key.size();
    // skip ':' whitespace and an optional opening quote
    while (p < s.size() && (s[p] == ':' || std::isspace(static_cast<unsigned char>(s[p])) || s[p] == '"')) {
        ++p;
    }
    size_t start = p;
    while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
        ++p;
    }
    if (p == start) {
        return false;
    }
    value = static_cast<uint32_t>(std::stoul(s.substr(start, p - start)));
    return true;
}

// Extract the quoted string value that follows `"key"` (e.g. "addr":"10.0.0.1").
inline bool ExtractQuotedAfterKey(
    const std::string& s, const std::string& key, size_t from, std::string& out, size_t limit)
{
    size_t p = s.find(key, from);
    if (p == std::string::npos || p >= limit) {
        return false;
    }
    p = s.find('"', p + key.size());
    if (p == std::string::npos || p >= limit) {
        return false;
    }
    size_t start = p + 1;
    size_t end = s.find('"', start);
    if (end == std::string::npos || end > limit) {
        return false;
    }
    out = s.substr(start, end - start);
    return true;
}

inline bool LooksLikeIpv4(const std::string& v)
{
    int dots = 0;
    for (char c : v) {
        if (c == '.') {
            ++dots;
        } else if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return dots == 3 && !v.empty();
}

inline bool ReadTextFile(const char* path, std::string& text)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    std::stringstream stream;
    stream << file.rdbuf();
    text = stream.str();
    return !text.empty();
}

inline bool FindDeviceBlock(const std::string& text, uint32_t phyId, size_t& blockBegin, size_t& blockEnd)
{
    const std::string deviceKey = "\"device_id\"";
    size_t scan = 0;
    while (scan < text.size()) {
        uint32_t device = 0;
        size_t deviceKeyPos = 0;
        if (!ExtractUintAfterKey(text, deviceKey, scan, device, deviceKeyPos)) {
            return false;
        }
        const size_t nextDevice = text.find(deviceKey, deviceKeyPos + deviceKey.size());
        blockEnd = nextDevice == std::string::npos ? text.size() : nextDevice;
        if (device == phyId) {
            blockBegin = deviceKeyPos;
            return true;
        }
        scan = blockEnd;
    }
    return false;
}

inline bool FindFirstIpv4After(const std::string& text, size_t begin, size_t end, std::string& ip)
{
    size_t cursor = begin;
    std::string candidate;
    while (ExtractQuotedAfterKey(text, "\"addr\"", cursor, candidate, end)) {
        if (LooksLikeIpv4(candidate)) {
            ip = candidate;
            return true;
        }
        const size_t addressKey = text.find("\"addr\"", cursor);
        if (addressKey == std::string::npos || addressKey >= end) {
            return false;
        }
        cursor = addressKey + 6;
    }
    return false;
}

} // namespace detail

// Parse the local RDMA NIC IPv4 for `phyId` from a rootinfo JSON file (default /etc/hccl_rootinfo.json).
// Locate the rank whose "device_id" equals phyId, then take the CLOS-level
// ("net_type":"CLOS") IPv4 address from that rank's block. This dependency-free
// targeted scan bounds each rank block by the next "device_id" occurrence;
// level_list entries do not contain "device_id".
inline bool ResolveLocalRdmaIp(uint32_t phyId, std::string& ip)
{
    std::string text;
    if (!detail::ReadTextFile(kDefaultRootInfoPath, text)) {
        return false;
    }
    size_t blockBegin = 0;
    size_t blockEnd = 0;
    if (!detail::FindDeviceBlock(text, phyId, blockBegin, blockEnd)) {
        return false;
    }
    const size_t closPos = text.find("\"CLOS\"", blockBegin);
    if (closPos == std::string::npos || closPos >= blockEnd) {
        return false;
    }
    return detail::FindFirstIpv4After(text, closPos, blockEnd, ip);
}

// Resolve the RoCE IPv4 selected for `phyId` by HCOMM's virtual-topology parser.
// The parser consumes /var/run/ascend-topologyd/virtualTopology.xml directly; it
// does not generate or modify rootinfo. The symbol belongs to HCOMM's topology
// component, so older installations may not provide it and callers must retain
// another bootstrap fallback.
inline bool ResolveLocalRdmaIpFromVirtualTopology(uint32_t phyId, std::string& ip)
{
    if (phyId > static_cast<uint32_t>(INT_MAX)) {
        return false;
    }

    using ResolveFn = int (*)(int, char*, size_t);
    void* topoHandle = nullptr;
    auto resolve = reinterpret_cast<ResolveFn>(dlsym(RTLD_DEFAULT, "GetRoceIpFromXml"));
    if (resolve == nullptr) {
        topoHandle = dlopen("libtopoaddrinfo.so", RTLD_NOW | RTLD_LOCAL);
        if (topoHandle == nullptr) {
            return false;
        }
        resolve = reinterpret_cast<ResolveFn>(dlsym(topoHandle, "GetRoceIpFromXml"));
    }

    char ipBuffer[64] = {0};
    const bool resolved = resolve != nullptr && resolve(static_cast<int>(phyId), ipBuffer, sizeof(ipBuffer)) == 0 &&
                          detail::LooksLikeIpv4(ipBuffer);
    if (resolved) {
        ip = ipBuffer;
    }
    if (topoHandle != nullptr) {
        dlclose(topoHandle);
    }
    return resolved;
}

using RankAgreementFn = bool (*)(bool, int, const char*, bool*);

// Out-of-band test bootstrap for HNS_1825.
//
// Per rank: phyId, RDMA NIC IPv4, and the registered-buffer base VA.
//   phyId    : ResolvePhyId(), else PTO_ROCE_PHYIDS[rank], else ACL device id
//   local IP : fixed rootinfo CLOS entry, then fixed virtualTopology.xml, then
//              PTO_ROCE_LOCAL_IP / PTO_ROCE_IPS test overrides
//   sym addr : MPI_Allgather of each rank's local registered-buffer base VA
// Missing local IP on any rank produces a collective skip rather than a false pass.
struct BootstrapConfig {
    bool skipped{false};
    std::vector<std::string> peerIps;
    std::vector<uint32_t> peerPhyIds;
    std::vector<uint64_t> peerSymAddrs;
    uint32_t phyId{0};
    uint16_t basePort{60032};

    static std::vector<std::string> SplitCsv(const char* env)
    {
        std::vector<std::string> out;
        if (env == nullptr) {
            return out;
        }
        std::stringstream ss(env);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t b = item.find_first_not_of(" \t");
            size_t e = item.find_last_not_of(" \t");
            if (b != std::string::npos) {
                out.push_back(item.substr(b, e - b + 1));
            }
        }
        return out;
    }

    void ResolveBootstrapPhyId(int rankId, int nRanks, int deviceId)
    {
        const std::vector<std::string> phyIds = SplitCsv(std::getenv("PTO_ROCE_PHYIDS"));
        if (static_cast<int>(phyIds.size()) == nRanks) {
            phyId = static_cast<uint32_t>(std::strtoul(phyIds[rankId].c_str(), nullptr, 10));
            return;
        }
        if (!ResolvePhyId(phyId)) {
            phyId = static_cast<uint32_t>(deviceId);
            std::cerr << "[RDMA][HNS_1825] phyId resolution unavailable, falling back to device id " << deviceId
                      << std::endl;
        }
    }

    static std::string ResolveBootstrapLocalIp(uint32_t phyId, int rankId, int nRanks)
    {
        std::string localIp;
        if (ResolveLocalRdmaIp(phyId, localIp) || ResolveLocalRdmaIpFromVirtualTopology(phyId, localIp)) {
            return localIp;
        }
        const char* localIpEnv = std::getenv("PTO_ROCE_LOCAL_IP");
        if (localIpEnv != nullptr && localIpEnv[0] != '\0') {
            return localIpEnv;
        }
        const std::vector<std::string> peerIpEnv = SplitCsv(std::getenv("PTO_ROCE_IPS"));
        return static_cast<int>(peerIpEnv.size()) == nRanks ? peerIpEnv[rankId] : std::string();
    }

    bool AgreeOnLocalIp(const std::string& localIp, int nRanks, RankAgreementFn agree)
    {
        const bool localIpReady = !localIp.empty();
        if (!localIpReady) {
            std::cerr << "[SKIP] RDMA test using HNS_1825 could not resolve local IP for phyId " << phyId
                      << " (no usable " << kDefaultRootInfoPath << " CLOS entry, no usable "
                      << kDefaultVirtualTopologyPath << ", and no PTO_ROCE_LOCAL_IP / PTO_ROCE_IPS)" << std::endl;
        }
        bool anyIpMissing = false;
        if (!agree(localIpReady, nRanks, "local RDMA IP resolution", &anyIpMissing)) {
            skipped = anyIpMissing;
            return false;
        }
        return true;
    }

    bool ResolveAndAgreeBasePort(int nRanks, RankAgreementFn agree)
    {
        const char* portEnv = std::getenv("PTO_ROCE_BASE_PORT");
        bool portReady = true;
        if (portEnv != nullptr) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(portEnv, &end, 10);
            portReady = end != portEnv && *end == '\0' && parsed <= UINT16_MAX;
            if (portReady) {
                basePort = static_cast<uint16_t>(parsed);
            } else {
                std::cerr << "[ERROR] invalid PTO_ROCE_BASE_PORT='" << portEnv << "'" << std::endl;
            }
        }
        if (!agree(portReady, nRanks, "base-port parsing", nullptr)) {
            return false;
        }
        std::vector<uint16_t> allPorts(static_cast<size_t>(nRanks), 0);
        if (CommMpiAllgather(
                &basePort, static_cast<int>(sizeof(basePort)), allPorts.data(), static_cast<int>(sizeof(basePort))) !=
            0) {
            std::cerr << "[ERROR] MPI_Allgather(basePort) failed" << std::endl;
            return false;
        }
        for (int rank = 0; rank < nRanks; ++rank) {
            if (allPorts[rank] != basePort) {
                std::cerr << "[ERROR] inconsistent PTO_ROCE_BASE_PORT: rank " << rank << " uses " << allPorts[rank]
                          << ", local rank uses " << basePort << std::endl;
                return false;
            }
        }
        return true;
    }

    bool GatherPeerIps(const std::string& localIp, int nRanks)
    {
        constexpr int kIpBufLen = 64;
        std::vector<char> localBuffer(kIpBufLen, 0);
        if (memcpy_s(localBuffer.data(), localBuffer.size(), localIp.c_str(), localIp.size() + 1) != EOK) {
            std::cerr << "[ERROR] local RDMA IP exceeds the bootstrap exchange buffer" << std::endl;
            return false;
        }
        std::vector<char> allBuffers(static_cast<size_t>(nRanks) * kIpBufLen, 0);
        if (CommMpiAllgather(localBuffer.data(), kIpBufLen, allBuffers.data(), kIpBufLen) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(localIp) failed" << std::endl;
            return false;
        }
        peerIps.assign(static_cast<size_t>(nRanks), std::string());
        for (int rank = 0; rank < nRanks; ++rank) {
            peerIps[rank] = std::string(allBuffers.data() + static_cast<size_t>(rank) * kIpBufLen);
        }
        return true;
    }

    bool GatherPeerResources(int nRanks, void* symAddr)
    {
        peerPhyIds.assign(static_cast<size_t>(nRanks), 0);
        if (CommMpiAllgather(
                &phyId, static_cast<int>(sizeof(phyId)), peerPhyIds.data(), static_cast<int>(sizeof(phyId))) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(phyId) failed" << std::endl;
            return false;
        }
        peerSymAddrs.assign(static_cast<size_t>(nRanks), 0);
        const uint64_t localAddr = reinterpret_cast<uint64_t>(symAddr);
        if (CommMpiAllgather(
                &localAddr, static_cast<int>(sizeof(localAddr)), peerSymAddrs.data(),
                static_cast<int>(sizeof(localAddr))) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(symAddr) failed" << std::endl;
            return false;
        }
        return true;
    }

    // rankId/nRanks — global rank index space (0-based when first_rank_id == 0).
    // deviceId      — this rank's ACL device id (phyId fallback).
    // symAddr       — this rank's registered communication-buffer base VA.
    bool Init(int rankId, int nRanks, int deviceId, void* symAddr, RankAgreementFn agree)
    {
        skipped = false;
        if (agree == nullptr) {
            std::cerr << "[ERROR] HNS_1825 bootstrap requires collective rank agreement" << std::endl;
            return false;
        }
        ResolveBootstrapPhyId(rankId, nRanks, deviceId);
        const std::string localIp = ResolveBootstrapLocalIp(phyId, rankId, nRanks);
        if (!AgreeOnLocalIp(localIp, nRanks, agree) || !ResolveAndAgreeBasePort(nRanks, agree)) {
            return false;
        }
        if (!GatherPeerIps(localIp, nRanks)) {
            return false;
        }
        return GatherPeerResources(nRanks, symAddr);
    }
};

} // namespace bootstrap
} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_TESTS_NPU_A5_COMM_ST_RDMA_BACKENDS_HNS_1825_BOOTSTRAP_HPP
