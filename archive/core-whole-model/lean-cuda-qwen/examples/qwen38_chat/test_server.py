# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

import unittest

import server


class CharacterTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))


class TokenMapTokenizer:
    pieces = {1: "check ", 2: "carefully", 3: "final"}

    def decode(self, token_ids, *, skip_special_tokens):
        self.assert_skip_special_tokens = skip_special_tokens
        return "".join(self.pieces[token] for token in token_ids)


class ChatPromptTests(unittest.TestCase):
    def test_exposes_published_model_context(self):
        self.assertEqual(server.CONTEXT_SIZE, 262_144)

    def test_renders_qwen_text_chat_template(self):
        prompt = server.render_chat(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Ready?"},
            ],
            system_prompt="Be concise.",
        )
        self.assertEqual(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi<|im_end|>\n"
            "<|im_start|>user\nReady?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_renders_published_reasoning_prompt(self):
        prompt = server.render_chat(
            [{"role": "user", "content": "Solve it."}],
            system_prompt="Be concise.",
            enable_thinking=True,
            reasoning_effort="low",
        )
        self.assertEqual(
            prompt,
            "<|im_start|>system\n"
            + server.REASONING_INSTRUCTIONS["low"]
            + "\n\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nSolve it.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n",
        )

    def test_rejects_unknown_reasoning_effort(self):
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            server.render_chat(
                [{"role": "user", "content": "Solve it."}],
                enable_thinking=True,
                reasoning_effort="maximum",
            )

    def test_drops_oldest_turns_to_fit_context(self):
        tokenizer = CharacterTokenizer()
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new"},
        ]
        newest_only = server.render_chat(messages[-1:], system_prompt="")
        fitted = server.fit_prompt(
            tokenizer,
            messages,
            system_prompt="",
            max_new_tokens=2,
            context_size=len(newest_only.encode("utf-8")) + 1,
        )
        self.assertEqual(fitted.messages, messages[-1:])
        self.assertNotIn("old question", fitted.text)
        self.assertEqual(fitted.token_ids, list(fitted.text.encode("utf-8")))

    def test_rejects_latest_turn_that_cannot_fit(self):
        with self.assertRaisesRegex(ValueError, "latest user turn"):
            server.fit_prompt(
                CharacterTokenizer(),
                [{"role": "user", "content": "too long"}],
                system_prompt="",
                max_new_tokens=2,
                context_size=8,
            )

    def test_accepts_generation_beyond_previous_256_token_cap(self):
        tokenizer = CharacterTokenizer()
        messages = [{"role": "user", "content": "continue"}]
        prompt = server.render_chat(messages, system_prompt="")
        fitted = server.fit_prompt(
            tokenizer,
            messages,
            system_prompt="",
            max_new_tokens=257,
            context_size=len(prompt.encode("utf-8")) + 256,
        )
        self.assertEqual(fitted.messages, messages)
        self.assertEqual(fitted.token_ids, list(prompt.encode("utf-8")))

    def test_omitted_generation_limit_uses_remaining_context(self):
        tokenizer = CharacterTokenizer()
        messages = [{"role": "user", "content": "continue"}]
        prompt = server.render_chat(messages, system_prompt="")
        context_size = len(prompt.encode("utf-8")) + 17
        fitted = server.fit_prompt(
            tokenizer,
            messages,
            system_prompt="",
            context_size=context_size,
        )
        self.assertEqual(fitted.messages, messages)
        self.assertEqual(
            server.remaining_generation_capacity(len(fitted.token_ids), context_size),
            18,
        )

    def test_marks_stable_history_as_exact_cache_prefix(self):
        tokenizer = CharacterTokenizer()
        fitted = server.fit_prompt(
            tokenizer,
            [{"role": "user", "content": "hello"}],
            system_prompt="system",
            enable_thinking=True,
        )
        history = server.render_chat_history(
            fitted.messages,
            system_prompt="system",
            enable_thinking=True,
        )
        self.assertEqual(fitted.cache_prefix_tokens, len(history.encode("utf-8")))
        self.assertEqual(
            fitted.token_ids[: fitted.cache_prefix_tokens],
            list(history.encode("utf-8")),
        )


class WorkerProtocolTests(unittest.TestCase):
    def test_parses_tagged_worker_messages(self):
        self.assertEqual(server.parse_worker_line("QWEN_CHAT READY"), ("ready", None))
        self.assertEqual(
            server.parse_worker_line("QWEN_CHAT READY BF16_EXACT_V1"), ("ready", None)
        )
        self.assertEqual(server.parse_worker_line("QWEN_CHAT TOKEN 42"), ("token", 42))
        self.assertEqual(server.parse_worker_line("QWEN_CHAT DONE 2"), ("done", 2))
        summary = {"event": "summary", "caseId": "qwen38_decode"}
        self.assertEqual(
            server.parse_worker_line(
                'QWEN_CHAT SUMMARY {"event":"summary","caseId":"qwen38_decode"}'
            ),
            ("summary", summary),
        )
        self.assertEqual(
            server.parse_worker_line("QWEN_CHAT ERROR invalid prompt"),
            ("error", "invalid prompt"),
        )
        self.assertIsNone(server.parse_worker_line("loading shard 1/18"))


class GenerationDecoderTests(unittest.TestCase):
    def test_splits_reasoning_from_answer_at_model_delimiter(self):
        decoder = server.GenerationDecoder(
            TokenMapTokenizer(),
            enable_thinking=True,
            think_end_token=99,
            eos_token=100,
        )
        self.assertEqual(
            decoder.push(1), {"type": "reasoning", "token": 1, "text": "check "}
        )
        self.assertEqual(
            decoder.push(2),
            {"type": "reasoning", "token": 2, "text": "check carefully"},
        )
        self.assertEqual(
            decoder.push(99),
            {"type": "reasoning_done", "text": "check carefully"},
        )
        self.assertEqual(
            decoder.push(3), {"type": "token", "token": 3, "text": "final"}
        )
        self.assertIsNone(decoder.push(100))
        self.assertEqual(decoder.reasoning, "check carefully")
        self.assertEqual(decoder.answer, "final")
        self.assertFalse(decoder.in_reasoning)
        self.assertEqual(decoder.generated_tokens, 5)

    def test_disabled_thinking_routes_tokens_directly_to_answer(self):
        decoder = server.GenerationDecoder(
            TokenMapTokenizer(),
            enable_thinking=False,
            think_end_token=99,
            eos_token=100,
        )
        self.assertEqual(
            decoder.push(3), {"type": "token", "token": 3, "text": "final"}
        )
        self.assertEqual(decoder.reasoning, "")


class ChatIndexTests(unittest.TestCase):
    def test_requests_and_renders_expandable_reasoning(self):
        index = server.Path(server.__file__).with_name("index.html").read_text()
        self.assertIn("enable_thinking: true", index)
        self.assertIn("className = 'reasoning-trace'", index)
        self.assertIn("event.type === 'reasoning'", index)
        self.assertNotIn("max_new_tokens", index)
        self.assertNotIn('id="max-tokens"', index)


if __name__ == "__main__":
    unittest.main()
