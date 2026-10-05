#!/usr/bin/env python3

import struct
import tempfile
import unittest
from pathlib import Path

import prepare_preference_dataset as prep


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        return _Encoding([ord(character) for character in text])


class PreferenceDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _CharacterTokenizer()

    def test_pair_has_shared_prompt_and_completion_only_masks(self) -> None:
        record = {
            "system": "Be precise.",
            "prompt": "What is 2 + 2?",
            "chosen": "4",
            "rejected": "5",
        }
        prefix = prep.render_history(record)
        prefix_length = len(self.tokenizer.encode(prefix).ids)
        pair = prep.encode_record(self.tokenizer, record, 192, pad_token=0)

        self.assertEqual(pair.chosen.tokens[:prefix_length], pair.rejected.tokens[:prefix_length])
        self.assertEqual(pair.chosen.mask[: prefix_length - 1], (0,) * (prefix_length - 1))
        self.assertEqual(pair.chosen.mask[prefix_length - 1], 1)
        self.assertEqual(len(pair.chosen.tokens), 193)
        self.assertEqual(len(pair.chosen.mask), 192)
        self.assertGreater(sum(pair.chosen.mask), 0)
        self.assertGreater(sum(pair.rejected.mask), 0)

    def test_binary_header_and_payload_size(self) -> None:
        pair = prep.encode_record(
            self.tokenizer,
            {"prompt": "Choose.", "chosen": "yes", "rejected": "no"},
            128,
            pad_token=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.bin"
            prep.write_dataset(path, 128, [pair])
            payload = path.read_bytes()

        self.assertEqual(payload[:8], prep.MAGIC)
        self.assertEqual(struct.unpack_from("<III", payload, 8), (prep.VERSION, 128, 1))
        self.assertEqual(len(payload), 20 + 2 * (2 * 128 + 1) * 4)

    def test_overlong_pair_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeding sequence length"):
            prep.encode_record(
                self.tokenizer,
                {"prompt": "long prompt", "chosen": "yes", "rejected": "no"},
                2,
                pad_token=0,
            )


if __name__ == "__main__":
    unittest.main()
