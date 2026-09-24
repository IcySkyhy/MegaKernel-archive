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

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn


class ThreadTaskRunner:
    def __init__(self, title: str, max_workers: int):
        self.title = title
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=False,
            refresh_per_second=5,
        )
        self.main_task_id = None
        self.success_task_id = None
        self.sub_tasks = {}
        self.queue = Queue()
        self.max_workers = max_workers

    def run_batch(self, tasks, run_single_task, is_huge, task_info):
        with self.progress:
            self.main_task_id = self.progress.add_task(f"Execution progress ({escape(self.title)})", total=len(tasks))
            self.success_task_id = self.progress.add_task("[green]Successful tasks[/green]", total=len(tasks))

            def handle_messages():
                while True:
                    msg = self.queue.get()
                    if msg is None:
                        break
                    action = msg["action"]
                    if action == "add_huge":
                        info = msg["info"]
                        sub_id = self.progress.add_task(f"  [yellow]Huge task:[/yellow] {info}", total=1)
                        self.sub_tasks[msg["task_id"]] = sub_id
                    elif action == "remove_huge":
                        sub_id = self.sub_tasks.pop(msg["task_id"], None)
                        if sub_id is not None:
                            self.progress.update(sub_id, advance=1)
                            self.progress.remove_task(sub_id)
                    elif action == "success":
                        self.progress.update(self.success_task_id, advance=1)
                    elif action == "fail":
                        info = msg["info"]
                        error = msg["error"]
                        self.progress.console.print(f"[bold red]Task failed: {info}[/bold red]\n  {error}")
                    elif action == "advance_main":
                        self.progress.update(self.main_task_id, advance=1)

            handler_thread = Thread(target=handle_messages)
            handler_thread.start()

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for task in tasks:
                    pool.submit(self._wrap_task, task, run_single_task, is_huge, task_info)

            self.queue.put(None)
            handler_thread.join()

    def _wrap_task(self, task, run_single_task, is_huge, task_info):
        task_id = id(task)
        info = escape(task_info(task))

        if is_huge(task):
            self.queue.put({"action": "add_huge", "task_id": task_id, "info": info})

        try:
            run_single_task(task)
            self.queue.put({"action": "success"})
        except Exception as e:
            self.queue.put({"action": "fail", "info": info, "error": f"{type(e).__name__}: {e}"})

        if is_huge(task):
            self.queue.put({"action": "remove_huge", "task_id": task_id})

        self.queue.put({"action": "advance_main"})
