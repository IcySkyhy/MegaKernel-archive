#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import os
import pathlib

import toml


def delete_config(path_to_config):
    if not path_to_config.exists():
        return
    with open(file=path_to_config, mode='r') as f:
        line = f.readline()
        if line == '// generated from config.toml\n':
            print('delete', path_to_config)
            if path_to_config.exists():
                path_to_config.unlink()

            path_cpp = path_to_config.with_suffix('.cpp')
            print('delete', path_cpp)
            if path_cpp.exists():
                path_cpp.unlink()


def process_toml(path_to_toml):
    d = toml.load(path_to_toml)

    print("processing", path_to_toml)
    name = list(d.keys())[0]
    cfgs = d[name]
    dirname = os.path.dirname(path_to_toml)

    write_h(name, cfgs, dirname)
    write_cpp(name, cfgs, dirname)


def write_cpp(name, cfgs, dirname):
    def gen_disp_line(k, v):
        if isinstance(v, str):
            s = 'String'
        elif isinstance(v, bool):
            s = 'Boolean'
        elif isinstance(v, int):
            s = 'Integer'
        elif isinstance(v, list):
            if all(isinstance(x, str) for x in v):
                s = 'StrVec'
            else:
                s = 'IntVec'

        if not isinstance(v, list):
            return f"        {{\"{k}\", [&](string v){{ {k} = Parse{s}(v); }}}},\n"
        else:
            return f"        {{\"{k}\", [&](string v){{ Parse{s}(v, {k}); }}}},\n"

    def gen_record_line(k, v):
        if isinstance(v, str):
            s = 'String'
        elif isinstance(v, bool):
            s = 'Boolean'
        elif isinstance(v, int):
            s = 'Integer'
        elif isinstance(v, list):
            s = 'IntVec'

        if s == 'String':
            return f"        {{\"{k}\", [&](){{ return \"{k} = \" + {k}; }}}},\n"
        else:
            return f"        {{\"{k}\", [&](){{ return \"{k} = \" + ParameterToStr({k}); }}}},\n"

    header = f"""/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \\file {name}Config.cpp
 * \\brief
 */

#include "cost_model/simulation/config/{name}Config.h"

using namespace std;

namespace CostModel {{
{name}Config::{name}Config()
{{
    Config::prefix = "{name}";
    Config::dispatcher = {{
"""
    middle = "    };\n"
    record = """
    Config::recorder = {
"""
    footer = "    };\n}\n}"

    with open(os.path.join(dirname, name + "Config.cpp"), 'w+') as f:
        f.write(header)
        for k in cfgs:
            f.write(gen_disp_line(k, cfgs[k]))
        f.write(middle)
        f.write(record)
        for k in cfgs:
            f.write(gen_record_line(k, cfgs[k]))
        f.write(footer)


def write_h(name, cfgs, dirname):
    def gen_cfg_line(k, v):
        if isinstance(v, str):
            return f"    std::string {k} = \"{v}\";\n"
        elif isinstance(v, bool):
            return f"    bool {k} = {'true' if v else 'false'};\n"
        elif isinstance(v, int):
            return f"    uint64_t {k} = {v};\n"
        elif isinstance(v, list):
            if all(isinstance(x, str) for x in v):
                items = '", "'.join(v)
                return f'    std::vector<std::string> {k} = {{"{items}"}};\n'
            return f"    std::vector<uint64_t> {k} = {str(v).replace('[', '{').replace(']', '}')};\n"
        return "error"

    header = f"""/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \\file {name}Config.h
 * \\brief
 */

// generated from config.toml
#pragma once

#include <cstdint>
#include <string>
#include "cost_model/simulation/base/Config.h"

namespace CostModel {{
struct {name}Config : public Config {{
    {name}Config();
"""
    footer = "};\n}"

    with open(os.path.join(dirname, name + 'Config.h'), 'w+') as f:
        f.write(header)
        for k in cfgs:
            f.write(gen_cfg_line(k, cfgs[k]))
        f.write(footer)


if __name__ == '__main__':
    search_path = os.path.join(os.path.dirname(__file__), '../..')
    toml_paths = pathlib.Path(search_path).rglob('*.toml')

    for t in toml_paths:
        process_toml(t)
