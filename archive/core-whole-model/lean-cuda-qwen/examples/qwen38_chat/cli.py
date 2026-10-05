#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

"""Streaming command-line chat client with separate Qwen3.8 reasoning and answer output."""

from __future__ import annotations

import argparse
from http.client import HTTPResponse
import json
import os
import sys
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = os.environ.get("QWEN_CHAT_URL", "http://127.0.0.1:8080")
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."


def event_stream(response: HTTPResponse) -> Iterator[dict[str, Any]]:
    for raw_line in response:
        if not raw_line.strip():
            continue
        event = json.loads(raw_line)
        if not isinstance(event, dict):
            raise RuntimeError("chat server returned a non-object event")
        yield event


def request_events(
    url: str,
    messages: list[dict[str, str]],
    *,
    system_prompt: str,
    enable_thinking: bool,
    reasoning_effort: str,
) -> Iterator[dict[str, Any]]:
    endpoint = url if url.rstrip("/").endswith("/api/chat") else url.rstrip("/") + "/api/chat"
    body = json.dumps(
        {
            "messages": messages,
            "system_prompt": system_prompt,
            "enable_thinking": enable_thinking,
            "reasoning_effort": reasoning_effort,
        }
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            yield from event_stream(response)
    except HTTPError as error:
        payload = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(payload).get("error", payload)
        except json.JSONDecodeError:
            message = payload
        raise RuntimeError(f"chat server returned HTTP {error.code}: {message}") from error
    except URLError as error:
        raise RuntimeError(f"cannot reach Qwen chat server at {endpoint}: {error.reason}") from error


class TracePrinter:
    def __init__(self, raw: bool):
        self.raw = raw
        self.previous = {"reasoning": "", "answer": ""}
        self.started: set[str] = set()
        self.last_channel: str | None = None

    def _update(self, channel: str, text: str) -> None:
        if channel not in self.started:
            if self.last_channel is not None:
                print()
            print("Reasoning:" if channel == "reasoning" else "Answer:")
            self.started.add(channel)
        previous = self.previous[channel]
        delta = text[len(previous) :] if text.startswith(previous) else text
        print(delta, end="", flush=True)
        self.previous[channel] = text
        self.last_channel = channel

    def consume(self, event: dict[str, Any]) -> None:
        if self.raw:
            print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            return
        event_type = event.get("type")
        if event_type in {"reasoning", "reasoning_done"}:
            self._update("reasoning", str(event.get("text", "")))
        elif event_type == "token":
            self._update("answer", str(event.get("text", "")))
        elif event_type == "error":
            raise RuntimeError(str(event.get("message", "unknown streaming error")))

    def finish(self) -> None:
        if not self.raw and self.last_channel is not None:
            print()


def chat_once(
    args: argparse.Namespace, messages: list[dict[str, str]]
) -> tuple[str, bool]:
    printer = TracePrinter(args.raw)
    done: dict[str, Any] | None = None
    for event in request_events(
        args.url,
        messages,
        system_prompt=args.system,
        enable_thinking=not args.no_thinking,
        reasoning_effort=args.reasoning_effort,
    ):
        printer.consume(event)
        if event.get("type") == "done":
            done = event
    printer.finish()
    if done is None:
        raise RuntimeError("chat stream ended without a done event")
    complete = bool(done.get("reasoning_complete", True))
    if not complete and not args.raw:
        print(
            "[reasoning trace did not close before generation stopped]",
            file=sys.stderr,
        )
    return str(done.get("text", "")), complete


def interactive(args: argparse.Namespace) -> int:
    history: list[dict[str, str]] = []
    print("Qwen3.8 Lean CUDA CLI. Enter /quit to exit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            return 0
        if prompt in {"/quit", "/exit"}:
            return 0
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        try:
            answer, _ = chat_once(args, history)
        except RuntimeError as error:
            history.pop()
            print(f"error: {error}", file=sys.stderr)
            continue
        if answer:
            history.append({"role": "assistant", "content": answer})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*", help="one-shot prompt; omit for an interactive session")
    parser.add_argument("--url", default=DEFAULT_URL, help="running Qwen chat server URL")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="system prompt")
    parser.add_argument(
        "--reasoning-effort",
        choices=("xhigh", "medium", "low"),
        default="xhigh",
        help="checkpoint reasoning effort (default: xhigh)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable the model reasoning trace and generate only the answer",
    )
    parser.add_argument("--raw", action="store_true", help="print raw NDJSON events")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prompt:
        prompt = " ".join(args.prompt).strip()
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        return interactive(args)
    if not prompt:
        print("error: prompt is empty", file=sys.stderr)
        return 2
    try:
        chat_once(args, [{"role": "user", "content": prompt}])
        return 0
    except (RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
