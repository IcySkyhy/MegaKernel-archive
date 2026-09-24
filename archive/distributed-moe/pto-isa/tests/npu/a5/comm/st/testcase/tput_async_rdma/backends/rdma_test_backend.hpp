/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TESTS_NPU_A5_COMM_ST_RDMA_TEST_BACKEND_HPP
#define PTO_TESTS_NPU_A5_COMM_ST_RDMA_TEST_BACKEND_HPP

#include <cstdint>
#include <cstdlib>

#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
#include "hns_1825/hns_1825_bootstrap.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_types.hpp"
#else
#error "The RDMA ST has no adapter for the configured backend."
#endif

namespace pto {
namespace comm {
namespace rdma {
namespace test {

#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
using BackendBootstrap = hns_1825::bootstrap::BootstrapConfig;

inline bool BackendVerboseEnabled()
{
    const char* verbose = std::getenv("PTO_ROCE_VERBOSE");
    return verbose != nullptr && verbose[0] == '1';
}

inline const char* DescribeBackendCompletionStatus(uint32_t status)
{
    switch (status) {
        case hns_1825::kHns1825PollCqTimeoutError:
            return "poll_cq_timeout";
        case hns_1825::kHns1825InvalidArgumentError:
            return "invalid_transfer";
        case hns_1825::kHns1825InvalidContextError:
            return "invalid_context";
        case hns_1825::kHns1825CqeError:
            return "cqe_error_without_syndrome";
        default:
            return nullptr;
    }
}
#endif

} // namespace test
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_TESTS_NPU_A5_COMM_ST_RDMA_TEST_BACKEND_HPP
