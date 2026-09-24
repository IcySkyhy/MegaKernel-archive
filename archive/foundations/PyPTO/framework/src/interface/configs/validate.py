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

import json
import logging
import sys

if __name__ == "__main__":
    try:
        import jsonschema

        has_jsonschema = True
    except ImportError:
        logging.warning("jsonschema module not found, ignore the schema validation")
        has_jsonschema = False

    if has_jsonschema:
        with open(sys.argv[1]) as f:
            instance = json.load(f)
        with open(sys.argv[2]) as f:
            schema = json.load(f)
        jsonschema.validate(instance=instance, schema=schema)
