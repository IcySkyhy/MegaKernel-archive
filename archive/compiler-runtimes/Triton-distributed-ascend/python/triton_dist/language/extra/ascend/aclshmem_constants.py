# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
from enum import IntEnum

# aclshmem_cmp_op_type_t
class ACLSHMEMCmpOp(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5

# aclshmem_signal_op_type_t
class ACLSHMEMSignalOp(IntEnum):
    SET = 0
    ADD = 1


# aclshmem_team_t
class ACLSHMEMTeam:
    INVALID = -1
    WORLD = 0
