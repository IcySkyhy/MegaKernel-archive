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
 * \file pybind_common.h
 * \brief
 */

#pragma once

#include <pybind11/pybind11.h>

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Python.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/stl.h"

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/calc.h"
#include "interface/configs/config_manager_ng.h"
#include "tilefwk/symbolic_distributed.h"

namespace py = pybind11;
