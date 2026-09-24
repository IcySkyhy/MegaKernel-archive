#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import argparse
from typing import Callable, Dict, Optional


class Argument:
    """Store definition of a single argument"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Command:
    # 全局命令注册表
    cmd_entry_dict: Dict[Callable, 'Command'] = {}
    cmd_dict: Dict[str, 'Command'] = {}

    def __init__(self, name: str, help_info: str):
        self.name = name
        self.help_info = help_info
        self.argument_list = []
        self.entry = None

    @classmethod
    def get_entry(cls, entry: Callable):
        if entry not in cls.cmd_entry_dict:
            name = entry.__name__
            cmd = Command(name, f'执行 {name} 命令')
            cmd.set_entry(entry)
            cls.cmd_entry_dict[entry] = cmd
        else:
            cmd = cls.cmd_entry_dict[entry]
        return cmd

    def add_argument(self, arg: Argument):
        self.argument_list.append(arg)

    def set_entry(self, entry: Callable):
        self.entry = entry

    def set_name_help(self, name: str, help_info: str):
        self.name = name
        self.help_info = help_info


def command(name: Optional[str] = None, help_info: Optional[str] = None):
    """Decorator: mark a function as a subcommand"""

    def decorator(func: Callable):
        cmd = Command.get_entry(func)
        cmd_name = name or func.__name__
        cmd_help = help_info or func.__doc__ or f"Execute {cmd_name} command"
        cmd.set_name_help(cmd_name, cmd_help)
        Command.cmd_dict[cmd_name] = cmd
        return func

    return decorator


def argument(*args, **kwargs):
    """Decorator: add arguments to the current command (must be used after @command)"""

    def decorator(func: Callable):
        cmd = Command.get_entry(func)
        cmd.add_argument(Argument(*args, **kwargs))
        return func

    return decorator


def build_parser(prog: str = None, description: str = None) -> argparse.ArgumentParser:
    """Build argparse parser from command registry"""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    subparsers = parser.add_subparsers(dest='command', required=True, help='可用的子命令')

    for _, cmd in Command.cmd_entry_dict.items():
        subparser = subparsers.add_parser(cmd.name, help=cmd.help_info)
        for arg in cmd.argument_list:
            subparser.add_argument(*arg.args, **arg.kwargs)
    return parser


def dispatch(args: argparse.Namespace):
    """Execute the corresponding command function with parsed arguments"""
    cmd_name = args.command
    if cmd_name not in Command.cmd_dict:
        raise ValueError(f"Unknown command: {cmd_name}")
    cmd = Command.cmd_dict[cmd_name]

    # 过滤掉 'command' 键，保留其他参数
    kwargs = {k: v for k, v in vars(args).items() if k != 'command'}
    # 调用函数
    return cmd.entry(**kwargs)


def run(main_prog: str = None, description: str = None, commands_dir: str = "commands"):
    """Main entry: load commands, build parser, execute"""
    parser = build_parser(prog=main_prog, description=description)
    parsed_args = parser.parse_args()
    return dispatch(parsed_args)
