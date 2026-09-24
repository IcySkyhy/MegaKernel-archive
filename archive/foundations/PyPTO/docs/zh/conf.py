# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import sys

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------
# 避免构建失败的虚拟导入
autodoc_mock_imports = ["torch"]

# 项目基本信息
project = "PyPTO"
html_title = "PyPTO文档"
copyright = "2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved."

# 指定主文档
master_doc = 'index'

# 启用的扩展列表
extensions = [
    "sphinx.ext.mathjax",  # 数学公式
    "myst_parser",  # Markdown 解析
    "sphinx.ext.viewcode",  # 显示代码源文件链接
    "sphinx.ext.intersphinx",  # 跨文档链接
    "sphinx_reredirects",  # 页面重定向
    "sphinx_tabs.tabs",  # 标签页
    "sphinx_toolbox.collapse",  # 折叠块
    "sphinx.ext.napoleon",  # Docstring 解析
    "sphinxcontrib.httpdomain",  # HTTP 接口文档
]

# source_suffix 配置
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Myst 扩展配置
myst_enable_extensions = [
    "colon_fence",  # 支持 ``` 代码块
    "deflist",  # 支持定义列表
    "dollarmath",  # 支持美元数学公式
]

# 语言改为中文
language = "zh_CN"

# 排除目录
exclude_patterns = ["_build", "README.md"]

# 代码高亮风格
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------
# Furo 主题配置
html_theme = "furo"

html_theme_options = {"source_edit_link": "https://gitcode.com/cann/pypto/blob/master/docs/zh/{filename}"}

# 模板路径
templates_path = []
