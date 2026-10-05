# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

"""CPU-only tests for the Qwen3.8 comparison harness."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from benchmark_workers import run_llama_baseline, run_worker


class LlamaBaselineTests(unittest.TestCase):
    def test_rejects_worker_without_exact_bf16_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "legacy-worker"
            worker.write_text(
                """#!/usr/bin/env python3
print("QWEN_CHAT READY", flush=True)
"""
            )
            worker.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, "exact BF16 autoregressive"):
                run_worker(
                    "legacy", worker, Path(directory), "1234", 0, 1, 2, 5.0
                )

    def test_uses_raw_sample_median_and_records_command(self) -> None:
        payload = [
            {
                "build_commit": "pinned",
                "n_prompt": 0,
                "n_gen": 128,
                "samples_ts": [4.1, 4.5, 4.3, 4.7, 4.2],
            }
        ]
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr="cuda init"
        )
        with tempfile.NamedTemporaryFile() as gguf, mock.patch(
            "benchmark_workers.subprocess.run", return_value=completed
        ) as run:
            result = run_llama_baseline(
                Path("/bin/true"), Path(gguf.name), 1, 5, 128, 10.0
            )
        self.assertEqual(result["decodeTokensPerSecondP50"], 4.3)
        self.assertEqual(result["result"], payload[0])
        self.assertEqual(result["command"][-2:], ["-o", "json"])
        self.assertEqual(run.call_args.kwargs["timeout"], 10.0)

    def test_rejects_wrong_sample_count(self) -> None:
        payload = [{"n_prompt": 0, "n_gen": 8, "samples_ts": [4.0]}]
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )
        with tempfile.NamedTemporaryFile() as gguf, mock.patch(
            "benchmark_workers.subprocess.run", return_value=completed
        ):
            with self.assertRaisesRegex(RuntimeError, "expected 3"):
                run_llama_baseline(Path("/bin/true"), Path(gguf.name), 0, 3, 8, 10.0)


if __name__ == "__main__":
    unittest.main()
