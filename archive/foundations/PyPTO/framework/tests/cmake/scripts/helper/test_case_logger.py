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

import sys


class TestCaseLogger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.logger = open(log_file, "w", encoding="utf-8")

    def __del__(self):
        self.terminal = None
        if self.logger is not None and not self.logger.closed:
            self.logger.close()
        self.logger = None

    def write(self, msg):
        self.terminal.write(msg)
        self.logger.write(msg)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.logger.flush()

    def isatty(self):
        return False
