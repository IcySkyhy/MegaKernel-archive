#!/usr/bin/env python3

import math
import struct
import tempfile
import unittest
from pathlib import Path

import prepare_grpo_dataset as prep


POLICY_SHA256 = "12" * 32


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        return _Encoding([ord(character) for character in text])


class GRPODatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _CharacterTokenizer()

    @staticmethod
    def response(text: str, reward: float) -> dict[str, object]:
        count = len(text + prep.preference.TURN_END)
        return {"response": text, "reward": reward, "behavior_logprobs": [-0.1] * count}

    def test_group_has_shared_prompt_rewards_and_action_masks(self) -> None:
        record = {
            "prompt": "Calculate.",
            "behavior_policy_sha256": POLICY_SHA256,
            "rollouts": [
                self.response("four", 1.0),
                self.response("five", -0.25),
            ],
        }
        prefix = prep.preference.render_history(record)
        prefix_length = len(self.tokenizer.encode(prefix).ids)
        group = prep.encode_record(
            self.tokenizer, record, sequence_length=160, group_size=2, pad_token=0
        )

        self.assertEqual(len(group.rollouts), 2)
        self.assertEqual(
            group.rollouts[0].tokens[:prefix_length],
            group.rollouts[1].tokens[:prefix_length],
        )
        self.assertEqual(
            group.rollouts[0].mask[: prefix_length - 1],
            (0,) * (prefix_length - 1),
        )
        self.assertEqual(group.rollouts[0].mask[prefix_length - 1], 1)
        self.assertEqual([rollout.reward for rollout in group.rollouts], [1.0, -0.25])

    def test_segments_preserve_noncontiguous_assistant_mask(self) -> None:
        record = {
            "prompt": "Use a tool.",
            "behavior_policy_sha256": POLICY_SHA256,
            "rollouts": [
                {
                    "segments": [
                        {"text": "CALL", "train": True},
                        {"text": "TOOL_OUTPUT", "train": False},
                        {"text": "FINAL", "train": True},
                    ],
                    "reward": 1,
                    "behavior_logprobs": [-0.2] * len("CALLFINAL"),
                },
                self.response("FAILED", 0),
            ],
        }
        prefix = prep.preference.render_history(record)
        prefix_length = len(self.tokenizer.encode(prefix).ids)
        group = prep.encode_record(
            self.tokenizer, record, sequence_length=192, group_size=2, pad_token=0
        )
        mask = group.rollouts[0].mask
        call_start = prefix_length - 1
        tool_start = call_start + len("CALL")
        final_start = tool_start + len("TOOL_OUTPUT")
        self.assertTrue(all(mask[index] == 1 for index in range(call_start, tool_start)))
        self.assertTrue(all(mask[index] == 0 for index in range(tool_start, final_start)))
        self.assertTrue(
            all(mask[index] == 1 for index in range(final_start, final_start + len("FINAL")))
        )
        old = group.rollouts[0].behavior_logprobs
        self.assertTrue(all(old[index] == 0 for index in range(tool_start, final_start)))
        self.assertTrue(all(old[index] == -0.2 for index in range(call_start, tool_start)))

    def test_binary_header_payload_and_float_rewards(self) -> None:
        record = {
            "prompt": "Choose.",
            "behavior_policy_sha256": POLICY_SHA256,
            "rollouts": [
                self.response("yes", 0.75),
                self.response("no", -1.5),
            ],
        }
        group = prep.encode_record(
            self.tokenizer, record, sequence_length=128, group_size=2, pad_token=0
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.bin"
            prep.write_dataset(path, 128, 2, [group])
            payload = path.read_bytes()
        self.assertEqual(payload[:8], prep.MAGIC)
        self.assertEqual(struct.unpack_from("<IIII", payload, 8), (prep.VERSION, 128, 2, 1))
        self.assertEqual(payload[24:56], bytes.fromhex(POLICY_SHA256))
        self.assertEqual(len(payload), 24 + 32 + 2 * (3 * 128 + 2) * 4)
        self.assertAlmostEqual(struct.unpack_from("<f", payload, len(payload) - 4)[0], -1.5)

    def test_nonfinite_reward_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            prep._require_reward(math.inf, "reward")

    def test_missing_behavior_logprobs_are_rejected(self) -> None:
        record = {
            "prompt": "Choose.",
            "behavior_policy_sha256": POLICY_SHA256,
            "rollouts": [{"response": "yes", "reward": 1}, self.response("no", 0)],
        }
        with self.assertRaisesRegex(ValueError, "behavior_logprobs"):
            prep.encode_record(self.tokenizer, record, 128, 2, 0)

    def test_positive_behavior_logprob_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most zero"):
            prep._require_behavior_logprobs([0.01], 1, "rollout")

    def test_invalid_policy_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            prep._require_policy_sha256("not-a-digest")

if __name__ == "__main__":
    unittest.main()
