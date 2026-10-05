# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

"""Local streaming chat bridge for the persistent Lean CUDA Qwen3.8 worker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Iterator


PROTOCOL_PREFIX = "QWEN_CHAT "
CONTEXT_SIZE = 262_144
MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."
REASONING_EFFORTS = ("xhigh", "medium", "low")
REASONING_INSTRUCTIONS = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer."
    ),
    "medium": "",
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
        "directly to the conclusion without unnecessary elaboration."
    ),
}


def render_chat_history(
    messages: list[dict[str, str]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    enable_thinking: bool = False,
    reasoning_effort: str = "xhigh",
) -> str:
    """Render the stable system/user/assistant history before the generation suffix."""
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be text")
    if not isinstance(enable_thinking, bool):
        raise ValueError("enable_thinking must be a boolean")
    if enable_thinking and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of " + ", ".join(REASONING_EFFORTS)
        )

    pieces: list[str] = []
    system_parts: list[str] = []
    if enable_thinking and REASONING_INSTRUCTIONS[reasoning_effort]:
        system_parts.append(REASONING_INSTRUCTIONS[reasoning_effort])
    if system_prompt.strip():
        system_parts.append(system_prompt.strip())
    if system_parts:
        pieces.append(
            "<|im_start|>system\n" + "\n\n".join(system_parts) + "<|im_end|>\n"
        )
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(pieces)


def render_assistant_prefix(enable_thinking: bool) -> str:
    prefix = "<|im_start|>assistant\n"
    if enable_thinking:
        return prefix + "<think>\n"
    return prefix + "<think>\n\n</think>\n\n"


def render_chat(
    messages: list[dict[str, str]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    enable_thinking: bool = False,
    reasoning_effort: str = "xhigh",
) -> str:
    """Render the published text-only template, optionally opening a reasoning trace."""
    return render_chat_history(
        messages,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    ) + render_assistant_prefix(enable_thinking)


def _encode_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.encode(prompt)
    return list(encoded.ids if hasattr(encoded, "ids") else encoded)


def _validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    checked: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(f"message {index} has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} must contain text")
        checked.append({"role": role, "content": content})
    if checked[-1]["role"] != "user":
        raise ValueError("the latest message must be from the user")
    return checked


@dataclass(frozen=True)
class FittedPrompt:
    text: str
    token_ids: list[int]
    messages: list[dict[str, str]]
    cache_prefix_tokens: int


def fit_prompt(
    tokenizer: Any,
    messages: Any,
    *,
    system_prompt: str,
    max_new_tokens: int | None = None,
    enable_thinking: bool = False,
    reasoning_effort: str = "xhigh",
    context_size: int = CONTEXT_SIZE,
) -> FittedPrompt:
    """Drop complete oldest turns until prompt plus generation fits the megakernel."""
    kept = _validate_messages(messages)
    if max_new_tokens is None:
        prompt_budget = context_size
    else:
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or not 1 <= max_new_tokens <= context_size
        ):
            raise ValueError(f"max_new_tokens must be between 1 and {context_size}")
        prompt_budget = context_size - max_new_tokens + 1
    while kept:
        history = render_chat_history(
            kept,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        prompt = history + render_assistant_prefix(enable_thinking)
        token_ids = _encode_ids(tokenizer, prompt)
        if len(token_ids) <= prompt_budget:
            cache_prefix_ids = _encode_ids(tokenizer, history)
            if token_ids[: len(cache_prefix_ids)] != cache_prefix_ids:
                raise ValueError("tokenizer merged across the assistant cache boundary")
            return FittedPrompt(
                text=prompt,
                token_ids=token_ids,
                messages=kept,
                cache_prefix_tokens=len(cache_prefix_ids),
            )
        if len(kept) == 1:
            break
        if (
            len(kept) >= 2
            and kept[0]["role"] == "user"
            and kept[1]["role"] == "assistant"
        ):
            kept = kept[2:]
        else:
            kept = kept[1:]
    if max_new_tokens is None:
        raise ValueError(
            f"latest user turn does not fit the {context_size}-token context"
        )
    raise ValueError(
        f"latest user turn does not fit the {context_size}-token context with "
        f"{max_new_tokens} generated tokens"
    )


def remaining_generation_capacity(
    prompt_tokens: int, context_size: int = CONTEXT_SIZE
) -> int:
    """Return the tokens available through the final model-context position."""
    if not 1 <= prompt_tokens <= context_size:
        raise ValueError("prompt token count must fit the model context")
    return context_size - prompt_tokens + 1


def parse_worker_line(line: str) -> tuple[str, Any] | None:
    if not line.startswith(PROTOCOL_PREFIX):
        return None
    payload = line[len(PROTOCOL_PREFIX) :].strip()
    command, _, value = payload.partition(" ")
    command = command.lower()
    if command in {"ready", "bye"}:
        return command, None
    if command in {"token", "done"}:
        return command, int(value)
    if command == "summary":
        summary = json.loads(value)
        if not isinstance(summary, dict):
            raise RuntimeError("worker summary must be a JSON object")
        return command, summary
    if command == "error":
        return command, value
    raise RuntimeError(f"unknown worker message: {payload}")


class LeanWorker:
    """Serialize requests through one checkpoint-resident Lean process."""

    def __init__(self, executable: Path, model_directory: Path):
        environment = os.environ.copy()
        environment["QWEN_MODEL_DIR"] = str(model_directory)
        self.process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.lock = threading.Lock()
        kind, value = self._read_message()
        if kind == "error":
            raise RuntimeError(f"Lean worker failed to start: {value}")
        if kind != "ready":
            raise RuntimeError(f"Lean worker sent {kind!r} before READY")

    def _read_message(self) -> tuple[str, Any]:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                code = self.process.poll()
                raise RuntimeError(f"Lean worker exited unexpectedly with status {code}")
            parsed = parse_worker_line(line)
            if parsed is not None:
                return parsed
            print(f"[lean] {line.rstrip()}", file=sys.stderr, flush=True)

    def _write(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Lean worker stdin is unavailable")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _drain_request(self) -> None:
        while True:
            kind, _ = self._read_message()
            if kind in {"done", "error"}:
                return

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        stop_token: int,
        cache_prefix_tokens: int,
    ) -> Iterator[int]:
        command = (
            f"GENERATE {max_new_tokens} {stop_token} {cache_prefix_tokens} "
            + ",".join(str(token) for token in prompt_ids)
        )
        with self.lock:
            self._write(command)
            complete = False
            try:
                while True:
                    kind, value = self._read_message()
                    if kind == "token":
                        yield value
                    elif kind == "done":
                        complete = True
                        return
                    elif kind == "error":
                        complete = True
                        raise RuntimeError(value)
                    else:
                        raise RuntimeError(f"unexpected worker message {kind!r}")
            finally:
                if not complete and self.process.poll() is None:
                    self._drain_request()

    def benchmark(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        warmup: int,
        repeats: int,
    ) -> list[dict[str, Any]]:
        command = (
            f"BENCHMARK {warmup} {repeats} {max_new_tokens} "
            + ",".join(str(token) for token in prompt_ids)
        )
        summaries: list[dict[str, Any]] = []
        with self.lock:
            self._write(command)
            while True:
                kind, value = self._read_message()
                if kind == "summary":
                    summaries.append(value)
                elif kind == "done":
                    if len(summaries) != 2:
                        raise RuntimeError(
                            f"worker returned {len(summaries)} benchmark summaries"
                        )
                    return summaries
                elif kind == "error":
                    raise RuntimeError(value)
                else:
                    raise RuntimeError(f"unexpected worker message {kind!r}")

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            with self.lock:
                self._write("QUIT")
                self.process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class GenerationDecoder:
    """Split streamed token IDs at the model's closing reasoning delimiter."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        enable_thinking: bool,
        think_end_token: int,
        eos_token: int,
    ):
        self.tokenizer = tokenizer
        self.in_reasoning = enable_thinking
        self.think_end_token = think_end_token
        self.eos_token = eos_token
        self.reasoning_ids: list[int] = []
        self.answer_ids: list[int] = []
        self.generated_tokens = 0

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def reasoning(self) -> str:
        return self._decode(self.reasoning_ids)

    @property
    def answer(self) -> str:
        return self._decode(self.answer_ids)

    def push(self, token: int) -> dict[str, Any] | None:
        self.generated_tokens += 1
        if token == self.eos_token:
            return None
        if self.in_reasoning and token == self.think_end_token:
            self.in_reasoning = False
            return {"type": "reasoning_done", "text": self.reasoning}
        if self.in_reasoning:
            self.reasoning_ids.append(token)
            return {"type": "reasoning", "token": token, "text": self.reasoning}
        self.answer_ids.append(token)
        return {"type": "token", "token": token, "text": self.answer}


class ChatApplication:
    def __init__(
        self,
        tokenizer: Any,
        worker: LeanWorker,
        index: bytes,
        eos_token: int,
        think_end_token: int,
    ):
        self.tokenizer = tokenizer
        self.worker = worker
        self.index = index
        self.eos_token = eos_token
        self.think_end_token = think_end_token


def make_handler(application: ChatApplication) -> type[BaseHTTPRequestHandler]:
    class ChatHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, pattern: str, *args: Any) -> None:
            print(f"[http] {pattern % args}", file=sys.stderr)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(application.index)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(application.index)
            elif self.path == "/health":
                self._json(
                    200,
                    {
                        "ready": True,
                        "context_size": CONTEXT_SIZE,
                        "reasoning": {
                            "supported": True,
                            "efforts": list(REASONING_EFFORTS),
                        },
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in {"/api/chat", "/api/benchmark"}:
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("request body is empty or too large")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                requested_max_new_tokens = request.get("max_new_tokens")
                if self.path == "/api/benchmark" and requested_max_new_tokens is None:
                    requested_max_new_tokens = 64
                system_prompt = request.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
                enable_thinking = request.get("enable_thinking", False)
                reasoning_effort = request.get("reasoning_effort", "xhigh")
                fitted = fit_prompt(
                    application.tokenizer,
                    request.get("messages"),
                    system_prompt=system_prompt,
                    max_new_tokens=requested_max_new_tokens,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                )
                prompt_ids = fitted.token_ids
                kept = fitted.messages
                max_new_tokens = (
                    requested_max_new_tokens
                    if requested_max_new_tokens is not None
                    else remaining_generation_capacity(len(prompt_ids))
                )
                warmup = request.get("warmup", 1)
                repeats = request.get("repeats", 5)
                if self.path == "/api/benchmark" and (
                    not isinstance(warmup, int)
                    or isinstance(warmup, bool)
                    or warmup < 0
                    or not isinstance(repeats, int)
                    or isinstance(repeats, bool)
                    or repeats < 1
                ):
                    raise ValueError("warmup must be nonnegative and repeats must be positive")
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
                return

            if self.path == "/api/benchmark":
                try:
                    summaries = application.worker.benchmark(
                        prompt_ids, max_new_tokens, warmup, repeats
                    )
                    self._json(
                        200,
                        {
                            "summaries": summaries,
                            "prompt_tokens": len(prompt_ids),
                            "kept_messages": len(kept),
                        },
                    )
                except Exception as error:
                    self._json(500, {"error": str(error)})
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            decoder = GenerationDecoder(
                application.tokenizer,
                enable_thinking=enable_thinking,
                think_end_token=application.think_end_token,
                eos_token=application.eos_token,
            )
            stream = application.worker.generate(
                prompt_ids,
                max_new_tokens,
                application.eos_token,
                fitted.cache_prefix_tokens,
            )
            try:
                for token in stream:
                    event = decoder.push(token)
                    if event is not None:
                        self._write_event(event)
                self._write_event(
                    {
                        "type": "done",
                        "text": decoder.answer,
                        "reasoning": decoder.reasoning,
                        "reasoning_complete": not decoder.in_reasoning,
                        "generated_tokens": decoder.generated_tokens,
                        "reasoning_tokens": len(decoder.reasoning_ids),
                        "answer_tokens": len(decoder.answer_ids),
                        "generation_capacity": max_new_tokens,
                        "prompt_tokens": len(prompt_ids),
                        "kept_messages": len(kept),
                    }
                )
            except (BrokenPipeError, ConnectionResetError):
                stream.close()
            except Exception as error:  # The response is already streaming.
                try:
                    self._write_event({"type": "error", "message": str(error)})
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _write_event(self, event: dict[str, Any]) -> None:
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

    return ChatHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from tokenizers import Tokenizer

    tokenizer_path = args.model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise SystemExit(f"missing tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token = tokenizer.token_to_id("<|im_end|>")
    if eos_token is None:
        raise SystemExit("tokenizer has no <|im_end|> token")
    think_end_token = tokenizer.token_to_id("</think>")
    if think_end_token is None:
        raise SystemExit("tokenizer has no </think> token")

    worker = LeanWorker(args.worker, args.model_dir)
    index = (
        Path(__file__)
        .with_name("index.html")
        .read_text(encoding="utf-8")
        .replace("__CONTEXT_SIZE__", str(CONTEXT_SIZE))
        .replace("__CONTEXT_SIZE_LABEL__", f"{CONTEXT_SIZE:,}")
        .encode("utf-8")
    )
    application = ChatApplication(
        tokenizer, worker, index, eos_token, think_end_token
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))
    server.daemon_threads = True
    print(f"Qwen3.8 chat ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
