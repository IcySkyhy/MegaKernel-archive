/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_RDMA_BACKEND_HPP
#define PTO_COMM_RDMA_BACKEND_HPP

#include <cstdint>

namespace pto {
namespace comm {

// A process may activate at most one RDMA backend. NONE means that no RDMA
// control plane has been initialized for the session.
enum class RdmaBackend : uint32_t {
    NONE = 0,
    HNS_1825 = 1,
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_RDMA_BACKEND_HPP
