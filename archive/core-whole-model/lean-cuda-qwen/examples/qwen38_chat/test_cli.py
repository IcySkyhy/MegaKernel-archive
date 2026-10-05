# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

import io
from contextlib import redirect_stdout
import json
import unittest
from unittest import mock

import cli


class TracePrinterTests(unittest.TestCase):
    def test_prints_reasoning_and_answer_as_separate_sections(self):
        output = io.StringIO()
        printer = cli.TracePrinter(raw=False)
        with redirect_stdout(output):
            printer.consume({"type": "reasoning", "text": "check"})
            printer.consume({"type": "reasoning", "text": "check carefully"})
            printer.consume({"type": "reasoning_done", "text": "check carefully"})
            printer.consume({"type": "token", "text": "42"})
            printer.finish()
        self.assertEqual(
            output.getvalue(),
            "Reasoning:\ncheck carefully\nAnswer:\n42\n",
        )

    def test_raw_mode_preserves_event_shape(self):
        output = io.StringIO()
        printer = cli.TracePrinter(raw=True)
        with redirect_stdout(output):
            printer.consume({"type": "reasoning", "token": 7, "text": "why"})
        self.assertEqual(
            output.getvalue(),
            '{"type":"reasoning","token":7,"text":"why"}\n',
        )


class RequestTests(unittest.TestCase):
    def test_omits_artificial_generation_limit(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.__iter__.return_value = iter([b'{"type":"done","text":"ok"}\n'])
        with mock.patch.object(cli, "urlopen", return_value=response) as urlopen:
            events = list(
                cli.request_events(
                    "http://127.0.0.1:8080",
                    [{"role": "user", "content": "hello"}],
                    system_prompt="",
                    enable_thinking=True,
                    reasoning_effort="low",
                )
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertNotIn("max_new_tokens", payload)
        self.assertEqual(events, [{"type": "done", "text": "ok"}])


if __name__ == "__main__":
    unittest.main()
